import os
import json
import logging
import asyncio
import mimetypes
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import requests
from io import BytesIO
import base64
from urllib.parse import urlparse

# Azure Storage
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.core.exceptions import AzureError

logger = logging.getLogger(__name__)

class UploadHandler:
    def __init__(self, cosmos_container, openai_client):
        """Initialize upload handler with necessary clients."""
        self.cosmos_container = cosmos_container
        self.openai_client = openai_client
        
        # Initialize Azure Blob Storage
        self.blob_service_client = BlobServiceClient(
            account_url=f"https://{os.getenv('AZURE_STORAGE_ACCOUNT_NAME')}.blob.core.windows.net",
            credential=os.getenv('AZURE_STORAGE_KEY')
        )
        self.container_name = os.getenv('AZURE_STORAGE_CONTAINER_NAME', 'weezfiles')
        
        # Platform configurations
        self.platform_configs = {
            'googledrive': {
                'name': 'Google Drive',
                'upload_url': 'https://www.googleapis.com/upload/drive/v3/files',
                'scopes': ['https://www.googleapis.com/auth/drive.file']
            },
            'dropbox': {
                'name': 'Dropbox',
                'upload_url': 'https://content.dropboxapi.com/2/files/upload',
                'content_upload_url': 'https://content.dropboxapi.com/2/files/upload'
            },
            'onedrive': {
                'name': 'OneDrive',
                'upload_url': 'https://graph.microsoft.com/v1.0/me/drive/root/children',
                'scopes': ['https://graph.microsoft.com/Files.ReadWrite']
            },
            'notion': {
                'name': 'Notion',
                'upload_url': 'https://api.notion.com/v1/blocks',
                'version': '2022-06-28'
            }
        }

    async def execute(self, parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Main execute method for upload operations.
        
        Handles:
        1. Direct file upload from user (like ChatGPT file upload)
        2. Platform-to-platform transfer
        3. Multi-platform upload
        """
        try:
            upload_type = parameters.get('upload_type', 'direct')
            
            if upload_type == 'direct':
                return await self._handle_direct_upload(parameters, user_id)
            elif upload_type == 'transfer':
                return await self._handle_platform_transfer(parameters, user_id)
            elif upload_type == 'multi':
                return await self._handle_multi_platform_upload(parameters, user_id)
            else:
                return self._error_response(f"Unknown upload type: {upload_type}")
                
        except Exception as e:
            logger.error(f"Upload execution failed: {str(e)}")
            return self._error_response(f"Upload failed: {str(e)}")

    async def _handle_direct_upload(self, parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Handle direct file upload from user's device.
        Similar to uploading files to ChatGPT interface.
        """
        try:
            # Extract upload parameters
            file_data = parameters.get('file_data')  # Base64 encoded file content
            file_name = parameters.get('file_name')
            file_type = parameters.get('file_type', 'application/octet-stream')
            target_platforms = parameters.get('target_platforms', ['blob'])  # Default to blob storage
            upload_metadata = parameters.get('metadata', {})
            
            if not file_data or not file_name:
                return self._error_response("Missing file data or file name")
            
            # Decode file content
            try:
                if isinstance(file_data, str):
                    # Handle base64 encoded data
                    file_content = base64.b64decode(file_data)
                else:
                    file_content = file_data
            except Exception as e:
                return self._error_response(f"Invalid file data format: {str(e)}")
            
            # Generate unique file ID
            file_id = self._generate_file_id(file_name, user_id)
            
            # Upload results tracking
            upload_results = {}
            success_count = 0
            
            # Upload to each target platform
            for platform in target_platforms:
                try:
                    if platform == 'blob':
                        result = await self._upload_to_blob(file_content, file_name, file_id, user_id, upload_metadata)
                    elif platform in self.platform_configs:
                        result = await self._upload_to_platform(
                            file_content, file_name, file_type, platform, user_id, upload_metadata
                        )
                    else:
                        result = {'success': False, 'error': f'Unsupported platform: {platform}'}
                    
                    upload_results[platform] = result
                    if result.get('success'):
                        success_count += 1
                        
                except Exception as e:
                    upload_results[platform] = {'success': False, 'error': str(e)}
            
            # Store file metadata in Cosmos DB
            if success_count > 0:
                await self._store_file_metadata(file_id, file_name, file_type, user_id, upload_results, upload_metadata)
            
            return {
                'success': success_count > 0,
                'message': f'File uploaded to {success_count}/{len(target_platforms)} platforms successfully',
                'file_id': file_id,
                'file_name': file_name,
                'upload_results': upload_results,
                'platforms_successful': success_count,
                'platforms_total': len(target_platforms)
            }
            
        except Exception as e:
            logger.error(f"Direct upload failed: {str(e)}")
            return self._error_response(f"Direct upload failed: {str(e)}")

    async def _handle_platform_transfer(self, parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Handle platform-to-platform file transfer.
        E.g., move file from Google Drive to Dropbox.
        """
        try:
            source_file_id = parameters.get('source_file_id')
            source_platform = parameters.get('source_platform', 'blob')
            target_platforms = parameters.get('target_platforms', [])
            transfer_metadata = parameters.get('metadata', {})
            
            if not source_file_id or not target_platforms:
                return self._error_response("Missing source file ID or target platforms")
            
            # Get source file from our database
            source_file = await self._get_file_metadata(source_file_id, user_id)
            if not source_file:
                return self._error_response(f"Source file {source_file_id} not found")
            
            # Download file content from source platform
            file_content, file_info = await self._download_from_platform(
                source_file_id, source_platform, user_id
            )
            
            if not file_content:
                return self._error_response("Failed to download source file")
            
            # Upload to target platforms
            upload_results = {}
            success_count = 0
            
            for platform in target_platforms:
                try:
                    if platform == source_platform:
                        upload_results[platform] = {'success': True, 'message': 'Already exists on this platform', 'skipped': True}
                        continue
                    
                    result = await self._upload_to_platform(
                        file_content, 
                        file_info.get('name', source_file.get('fileName')),
                        file_info.get('type', source_file.get('fileType', 'application/octet-stream')),
                        platform, 
                        user_id, 
                        transfer_metadata
                    )
                    
                    upload_results[platform] = result
                    if result.get('success'):
                        success_count += 1
                        
                except Exception as e:
                    upload_results[platform] = {'success': False, 'error': str(e)}
            
            # Update file metadata with new platform locations
            if success_count > 0:
                await self._update_file_platforms(source_file_id, user_id, upload_results)
            
            return {
                'success': success_count > 0,
                'message': f'File transferred to {success_count}/{len(target_platforms)} platforms successfully',
                'source_file_id': source_file_id,
                'source_platform': source_platform,
                'transfer_results': upload_results,
                'platforms_successful': success_count,
                'platforms_total': len(target_platforms)
            }
            
        except Exception as e:
            logger.error(f"Platform transfer failed: {str(e)}")
            return self._error_response(f"Platform transfer failed: {str(e)}")

    async def _handle_multi_platform_upload(self, parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Handle uploading multiple files to multiple platforms simultaneously.
        """
        try:
            files_data = parameters.get('files', [])
            target_platforms = parameters.get('target_platforms', ['blob'])
            upload_metadata = parameters.get('metadata', {})
            
            if not files_data:
                return self._error_response("No files provided for upload")
            
            all_results = []
            total_success = 0
            
            # Process each file
            for file_data in files_data:
                file_params = {
                    'upload_type': 'direct',
                    'file_data': file_data.get('content'),
                    'file_name': file_data.get('name'),
                    'file_type': file_data.get('type', 'application/octet-stream'),
                    'target_platforms': target_platforms,
                    'metadata': {**upload_metadata, **file_data.get('metadata', {})}
                }
                
                result = await self._handle_direct_upload(file_params, user_id)
                all_results.append({
                    'file_name': file_data.get('name'),
                    'result': result
                })
                
                if result.get('success'):
                    total_success += 1
            
            return {
                'success': total_success > 0,
                'message': f'Successfully uploaded {total_success}/{len(files_data)} files',
                'files_processed': len(files_data),
                'files_successful': total_success,
                'detailed_results': all_results
            }
            
        except Exception as e:
            logger.error(f"Multi-platform upload failed: {str(e)}")
            return self._error_response(f"Multi-platform upload failed: {str(e)}")

    async def _upload_to_blob(self, file_content: bytes, file_name: str, file_id: str, user_id: str, metadata: Dict) -> Dict[str, Any]:
        """Upload file to Azure Blob Storage."""
        try:
            blob_name = f"{user_id}/{file_id}/{file_name}"
            
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            # Upload with metadata
            blob_metadata = {
                'user_id': user_id,
                'file_id': file_id,
                'original_name': file_name,
                'upload_timestamp': datetime.now().isoformat(),
                **metadata
            }
            
            blob_client.upload_blob(
                file_content,
                overwrite=True,
                metadata=blob_metadata
            )
            
            return {
                'success': True,
                'platform': 'blob',
                'blob_name': blob_name,
                'url': blob_client.url,
                'message': 'Successfully uploaded to Azure Blob Storage'
            }
            
        except AzureError as e:
            return {'success': False, 'error': f'Azure Blob error: {str(e)}'}
        except Exception as e:
            return {'success': False, 'error': f'Blob upload error: {str(e)}'}

    async def _upload_to_platform(self, file_content: bytes, file_name: str, file_type: str, platform: str, user_id: str, metadata: Dict) -> Dict[str, Any]:
        """Upload file to external platform (Google Drive, Dropbox, etc.)."""
        try:
            platform_config = self.platform_configs.get(platform)
            if not platform_config:
                return {'success': False, 'error': f'Unsupported platform: {platform}'}
            
            # Get user's access token for the platform
            access_token = await self._get_platform_access_token(user_id, platform)
            if not access_token:
                return {'success': False, 'error': f'No access token found for {platform}'}
            
            if platform == 'googledrive':
                return await self._upload_to_google_drive(file_content, file_name, file_type, access_token, metadata)
            elif platform == 'dropbox':
                return await self._upload_to_dropbox(file_content, file_name, access_token, metadata)
            elif platform == 'onedrive':
                return await self._upload_to_onedrive(file_content, file_name, file_type, access_token, metadata)
            elif platform == 'notion':
                return await self._upload_to_notion(file_content, file_name, file_type, access_token, metadata)
            else:
                return {'success': False, 'error': f'Platform {platform} not implemented yet'}
                
        except Exception as e:
            return {'success': False, 'error': f'{platform} upload error: {str(e)}'}

    async def _upload_to_google_drive(self, file_content: bytes, file_name: str, file_type: str, access_token: str, metadata: Dict) -> Dict[str, Any]:
        """Upload file to Google Drive."""
        try:
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            # Create file metadata
            file_metadata = {
                'name': file_name,
                'description': metadata.get('description', f'Uploaded via Weez AI on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            }
            
            # Upload file using multipart
            files = {
                'metadata': (None, json.dumps(file_metadata), 'application/json'),
                'media': (file_name, file_content, file_type)
            }
            
            response = requests.post(
                'https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart',
                headers={'Authorization': f'Bearer {access_token}'},
                files=files
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'platform': 'googledrive',
                    'file_id': result.get('id'),
                    'name': result.get('name'),
                    'web_view_link': result.get('webViewLink'),
                    'message': 'Successfully uploaded to Google Drive'
                }
            else:
                return {'success': False, 'error': f'Google Drive API error: {response.text}'}
                
        except Exception as e:
            return {'success': False, 'error': f'Google Drive upload error: {str(e)}'}

    async def _upload_to_dropbox(self, file_content: bytes, file_name: str, access_token: str, metadata: Dict) -> Dict[str, Any]:
        """Upload file to Dropbox."""
        try:
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/octet-stream',
                'Dropbox-API-Arg': json.dumps({
                    'path': f'/{file_name}',
                    'mode': 'overwrite',
                    'autorename': True
                })
            }
            
            response = requests.post(
                'https://content.dropboxapi.com/2/files/upload',
                headers=headers,
                data=file_content
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'platform': 'dropbox',
                    'file_id': result.get('id'),
                    'name': result.get('name'),
                    'path': result.get('path_display'),
                    'message': 'Successfully uploaded to Dropbox'
                }
            else:
                return {'success': False, 'error': f'Dropbox API error: {response.text}'}
                
        except Exception as e:
            return {'success': False, 'error': f'Dropbox upload error: {str(e)}'}

    async def _upload_to_onedrive(self, file_content: bytes, file_name: str, file_type: str, access_token: str, metadata: Dict) -> Dict[str, Any]:
        """Upload file to OneDrive."""
        try:
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': file_type
            }
            
            # For small files (< 4MB), use simple upload
            if len(file_content) < 4 * 1024 * 1024:
                response = requests.put(
                    f'https://graph.microsoft.com/v1.0/me/drive/root:/{file_name}:/content',
                    headers=headers,
                    data=file_content
                )
                
                if response.status_code in [200, 201]:
                    result = response.json()
                    return {
                        'success': True,
                        'platform': 'onedrive',
                        'file_id': result.get('id'),
                        'name': result.get('name'),
                        'web_url': result.get('webUrl'),
                        'message': 'Successfully uploaded to OneDrive'
                    }
                else:
                    return {'success': False, 'error': f'OneDrive API error: {response.text}'}
            else:
                # For large files, use upload session (not implemented in this example)
                return {'success': False, 'error': 'Large file upload to OneDrive not implemented yet'}
                
        except Exception as e:
            return {'success': False, 'error': f'OneDrive upload error: {str(e)}'}

    async def _upload_to_notion(self, file_content: bytes, file_name: str, file_type: str, access_token: str, metadata: Dict) -> Dict[str, Any]:
        """Upload file to Notion (as attachment to a page)."""
        try:
            # Note: Notion doesn't have direct file upload API
            # This would typically involve creating a page and attaching the file
            # For now, we'll return a placeholder implementation
            return {
                'success': False,
                'error': 'Notion file upload requires page context and is not yet fully implemented'
            }
                
        except Exception as e:
            return {'success': False, 'error': f'Notion upload error: {str(e)}'}

    async def _download_from_platform(self, file_id: str, platform: str, user_id: str) -> tuple:
        """Download file content from specified platform."""
        try:
            if platform == 'blob':
                return await self._download_from_blob(file_id, user_id)
            elif platform in self.platform_configs:
                access_token = await self._get_platform_access_token(user_id, platform)
                if not access_token:
                    return None, None
                
                if platform == 'googledrive':
                    return await self._download_from_google_drive(file_id, access_token)
                elif platform == 'dropbox':
                    return await self._download_from_dropbox(file_id, access_token)
                elif platform == 'onedrive':
                    return await self._download_from_onedrive(file_id, access_token)
                else:
                    return None, None
            else:
                return None, None
                
        except Exception as e:
            logger.error(f"Download from {platform} failed: {str(e)}")
            return None, None

    async def _download_from_blob(self, file_id: str, user_id: str) -> tuple:
        """Download file from Azure Blob Storage."""
        try:
            # Find blob by file_id
            file_metadata = await self._get_file_metadata(file_id, user_id)
            if not file_metadata:
                return None, None
            
            blob_name = file_metadata.get('blob_name')
            if not blob_name:
                return None, None
            
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            file_content = blob_client.download_blob().readall()
            file_info = {
                'name': file_metadata.get('fileName'),
                'type': file_metadata.get('fileType'),
                'size': len(file_content)
            }
            
            return file_content, file_info
            
        except Exception as e:
            logger.error(f"Blob download failed: {str(e)}")
            return None, None

    async def _get_platform_access_token(self, user_id: str, platform: str) -> Optional[str]:
        """Get user's access token for the specified platform."""
        try:
            # Query Cosmos DB for user's platform tokens
            query = f"SELECT * FROM c WHERE c.user_id = '{user_id}' AND c.platform = '{platform}' AND c.type = 'access_token'"
            
            items = list(self.cosmos_container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            
            if items:
                token_data = items[0]
                # Check if token is expired and refresh if needed
                return token_data.get('access_token')
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get access token for {platform}: {str(e)}")
            return None

    async def _store_file_metadata(self, file_id: str, file_name: str, file_type: str, user_id: str, upload_results: Dict, metadata: Dict):
        """Store file metadata in Cosmos DB."""
        try:
            file_document = {
                'id': file_id,
                'user_id': user_id,
                'fileName': file_name,
                'fileType': file_type,
                'uploadTimestamp': datetime.now().isoformat(),
                'platforms': {},
                'metadata': metadata,
                'type': 'file_upload'
            }
            
            # Add platform-specific information
            for platform, result in upload_results.items():
                if result.get('success'):
                    file_document['platforms'][platform] = {
                        'uploaded': True,
                        'upload_time': datetime.now().isoformat(),
                        'platform_file_id': result.get('file_id'),
                        'url': result.get('url') or result.get('web_view_link') or result.get('web_url'),
                        'path': result.get('path') or result.get('blob_name')
                    }
            
            self.cosmos_container.create_item(file_document)
            
        except Exception as e:
            logger.error(f"Failed to store file metadata: {str(e)}")

    async def _get_file_metadata(self, file_id: str, user_id: str) -> Optional[Dict]:
        """Get file metadata from Cosmos DB."""
        try:
            query = f"SELECT * FROM c WHERE c.id = '{file_id}' AND c.user_id = '{user_id}'"
            
            items = list(self.cosmos_container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            
            return items[0] if items else None
            
        except Exception as e:
            logger.error(f"Failed to get file metadata: {str(e)}")
            return None

    async def _update_file_platforms(self, file_id: str, user_id: str, upload_results: Dict):
        """Update file metadata with new platform information."""
        try:
            file_doc = await self._get_file_metadata(file_id, user_id)
            if not file_doc:
                return
            
            # Update platforms information
            if 'platforms' not in file_doc:
                file_doc['platforms'] = {}
            
            for platform, result in upload_results.items():
                if result.get('success') and not result.get('skipped'):
                    file_doc['platforms'][platform] = {
                        'uploaded': True,
                        'upload_time': datetime.now().isoformat(),
                        'platform_file_id': result.get('file_id'),
                        'url': result.get('url') or result.get('web_view_link') or result.get('web_url'),
                        'path': result.get('path') or result.get('blob_name')
                    }
            
            # Update document
            self.cosmos_container.replace_item(file_doc['id'], file_doc)
            
        except Exception as e:
            logger.error(f"Failed to update file platforms: {str(e)}")

    def _generate_file_id(self, file_name: str, user_id: str) -> str:
        """Generate unique file ID."""
        import hashlib
        timestamp = datetime.now().isoformat()
        content = f"{user_id}_{file_name}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()

    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate standardized error response."""
        return {
            'success': False,
            'error': error_message,
            'message': f"Upload failed: {error_message}",
            'timestamp': datetime.now().isoformat()
        }


def create_upload_handler(cosmos_container, openai_client) -> UploadHandler:
    """Create and return an upload handler instance."""
    return UploadHandler(cosmos_container, openai_client)


# Example usage and testing
async def test_upload_handler():
    """Test the upload handler with sample data."""
    # This would be called with actual cosmos container and openai client
    pass

if __name__ == "__main__":
    # Run tests
    asyncio.run(test_upload_handler())