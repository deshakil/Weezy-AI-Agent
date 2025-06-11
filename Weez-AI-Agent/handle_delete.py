import os
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import CosmosResourceNotFoundError

# Platform-specific imports (you'll need to install these)
try:
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False

try:
    import requests
    ONEDRIVE_AVAILABLE = True
except ImportError:
    ONEDRIVE_AVAILABLE = False

try:
    import dropbox
    DROPBOX_AVAILABLE = True
except ImportError:
    DROPBOX_AVAILABLE = False

try:
    from notion_client import Client as NotionClient
    NOTION_AVAILABLE = True
except ImportError:
    NOTION_AVAILABLE = False

logger = logging.getLogger(__name__)

class DeleteHandler:
    def __init__(self, cosmos_container, openai_client):
        """Initialize the delete handler with necessary clients."""
        self.cosmos_container = cosmos_container
        self.openai_client = openai_client
        
        # Initialize blob storage client
        self.blob_service_client = BlobServiceClient(
            account_url=f"https://{os.getenv('AZURE_STORAGE_ACCOUNT_NAME')}.blob.core.windows.net",
            credential=os.getenv('AZURE_STORAGE_KEY')
        )
        self.blob_container_name = os.getenv('BLOB_CONTAINER_NAME', 'files')
        
        # Platform clients (initialize based on available credentials)
        self.platform_clients = self._initialize_platform_clients()
    
    def _initialize_platform_clients(self) -> Dict[str, Any]:
        """Initialize clients for different platforms based on available credentials."""
        clients = {}
        
        # Google Drive
        if GOOGLE_DRIVE_AVAILABLE and os.getenv('GOOGLE_DRIVE_CREDENTIALS'):
            try:
                # Initialize Google Drive client
                # You'll need to implement OAuth flow for production
                clients['drive'] = self._init_google_drive_client()
            except Exception as e:
                logger.warning(f"Failed to initialize Google Drive client: {e}")
        
        # OneDrive
        if ONEDRIVE_AVAILABLE and os.getenv('ONEDRIVE_ACCESS_TOKEN'):
            clients['onedrive'] = {
                'access_token': os.getenv('ONEDRIVE_ACCESS_TOKEN'),
                'base_url': 'https://graph.microsoft.com/v1.0/me/drive'
            }
        
        # Dropbox
        if DROPBOX_AVAILABLE and os.getenv('DROPBOX_ACCESS_TOKEN'):
            try:
                clients['dropbox'] = dropbox.Dropbox(os.getenv('DROPBOX_ACCESS_TOKEN'))
            except Exception as e:
                logger.warning(f"Failed to initialize Dropbox client: {e}")
        
        # Notion
        if NOTION_AVAILABLE and os.getenv('NOTION_TOKEN'):
            try:
                clients['notion'] = NotionClient(auth=os.getenv('NOTION_TOKEN'))
            except Exception as e:
                logger.warning(f"Failed to initialize Notion client: {e}")
        
        return clients
    
    def _init_google_drive_client(self):
        """Initialize Google Drive client (placeholder - you'll need to implement OAuth)."""
        # This is a placeholder - you'll need to implement proper OAuth flow
        # For now, assuming you have stored credentials
        creds_path = os.getenv('GOOGLE_DRIVE_CREDENTIALS')
        if creds_path and os.path.exists(creds_path):
            # Load credentials and build service
            # Implementation depends on your OAuth setup
            pass
        return None
    
    def execute(self, parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Main execution method for delete operations.
        
        Expected parameters:
        - file_id: The ID of the file to delete (from Cosmos DB)
        - platform: The platform from which to delete (drive, onedrive, dropbox, notion, local)
        - confirm: Boolean confirmation from user
        """
        try:
            file_id = parameters.get('file_id')
            platform = parameters.get('platform', '').lower()
            confirm = parameters.get('confirm', False)
            
            if not file_id:
                return {
                    "success": False,
                    "error": "Missing file_id parameter",
                    "message": "Please specify which file you want to delete."
                }
            
            if not confirm:
                # First, get file info for confirmation
                file_info = self._get_file_info(file_id, user_id)
                if not file_info:
                    return {
                        "success": False,
                        "error": "File not found",
                        "message": f"Could not find file with ID: {file_id}"
                    }
                
                return {
                    "success": False,
                    "error": "Confirmation required",
                    "message": f"Are you sure you want to delete '{file_info.get('fileName', 'Unknown file')}' from {platform or 'all platforms'}? This action cannot be undone.",
                    "requires_confirmation": True,
                    "file_info": file_info
                }
            
            # Execute the deletion
            return self._delete_file(file_id, platform, user_id)
        
        except Exception as e:
            logger.error(f"Delete execution error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to delete file: {str(e)}"
            }
    
    def _get_file_info(self, file_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get file information from Cosmos DB."""
        try:
            query = f"SELECT * FROM c WHERE c.id = '{file_id}' AND c.user_id = '{user_id}'"
            items = list(self.cosmos_container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            
            return items[0] if items else None
        
        except Exception as e:
            logger.error(f"Error getting file info: {str(e)}")
            return None
    
    def _delete_file(self, file_id: str, platform: str, user_id: str) -> Dict[str, Any]:
        """Delete file from specified platform and blob storage."""
        try:
            # Get file information
            file_info = self._get_file_info(file_id, user_id)
            if not file_info:
                return {
                    "success": False,
                    "error": "File not found",
                    "message": f"Could not find file with ID: {file_id}"
                }
            
            deletion_results = []
            overall_success = True
            
            # If platform is specified, delete from that platform only
            if platform:
                platform_result = self._delete_from_platform(file_info, platform)
                deletion_results.append(platform_result)
                if not platform_result['success']:
                    overall_success = False
            else:
                # Delete from all platforms where the file exists
                file_platform = file_info.get('platform', '').lower()
                if file_platform:
                    platform_result = self._delete_from_platform(file_info, file_platform)
                    deletion_results.append(platform_result)
                    if not platform_result['success']:
                        overall_success = False
            
            # Delete from blob storage
            blob_result = self._delete_from_blob_storage(file_info)
            deletion_results.append(blob_result)
            if not blob_result['success']:
                overall_success = False
            
            # Delete from Cosmos DB
            cosmos_result = self._delete_from_cosmos_db(file_id, user_id)
            deletion_results.append(cosmos_result)
            if not cosmos_result['success']:
                overall_success = False
            
            return {
                "success": overall_success,
                "message": f"File deletion {'completed' if overall_success else 'completed with errors'}",
                "file_name": file_info.get('fileName', 'Unknown'),
                "deletion_results": deletion_results,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to delete file: {str(e)}"
            }
    
    def _delete_from_platform(self, file_info: Dict[str, Any], platform: str) -> Dict[str, Any]:
        """Delete file from specific platform."""
        platform = platform.lower()
        
        try:
            if platform == 'drive' or platform == 'google_drive':
                return self._delete_from_google_drive(file_info)
            elif platform == 'onedrive':
                return self._delete_from_onedrive(file_info)
            elif platform == 'dropbox':
                return self._delete_from_dropbox(file_info)
            elif platform == 'notion':
                return self._delete_from_notion(file_info)
            elif platform == 'local':
                return self._delete_from_local(file_info)
            else:
                return {
                    "success": False,
                    "platform": platform,
                    "error": f"Unsupported platform: {platform}",
                    "message": f"Cannot delete from {platform} - platform not supported"
                }
        
        except Exception as e:
            logger.error(f"Error deleting from {platform}: {str(e)}")
            return {
                "success": False,
                "platform": platform,
                "error": str(e),
                "message": f"Failed to delete from {platform}: {str(e)}"
            }
    
    def _delete_from_google_drive(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Delete file from Google Drive."""
        if 'drive' not in self.platform_clients:
            return {
                "success": False,
                "platform": "google_drive",
                "error": "Google Drive client not available",
                "message": "Google Drive integration not configured"
            }
        
        try:
            drive_service = self.platform_clients['drive']
            drive_file_id = file_info.get('platform_file_id') or file_info.get('drive_file_id')
            
            if not drive_file_id:
                return {
                    "success": False,
                    "platform": "google_drive",
                    "error": "No Drive file ID found",
                    "message": "Cannot delete from Google Drive - no file ID available"
                }
            
            # Delete file from Google Drive
            drive_service.files().delete(fileId=drive_file_id).execute()
            
            return {
                "success": True,
                "platform": "google_drive",
                "message": "Successfully deleted from Google Drive"
            }
        
        except Exception as e:
            return {
                "success": False,
                "platform": "google_drive",
                "error": str(e),
                "message": f"Failed to delete from Google Drive: {str(e)}"
            }
    
    def _delete_from_onedrive(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Delete file from OneDrive."""
        if 'onedrive' not in self.platform_clients:
            return {
                "success": False,
                "platform": "onedrive",
                "error": "OneDrive client not available",
                "message": "OneDrive integration not configured"
            }
        
        try:
            client_info = self.platform_clients['onedrive']
            onedrive_file_id = file_info.get('platform_file_id') or file_info.get('onedrive_file_id')
            
            if not onedrive_file_id:
                return {
                    "success": False,
                    "platform": "onedrive",
                    "error": "No OneDrive file ID found",
                    "message": "Cannot delete from OneDrive - no file ID available"
                }
            
            # Delete file from OneDrive using Microsoft Graph API
            headers = {
                'Authorization': f"Bearer {client_info['access_token']}",
                'Content-Type': 'application/json'
            }
            
            delete_url = f"{client_info['base_url']}/items/{onedrive_file_id}"
            response = requests.delete(delete_url, headers=headers)
            
            if response.status_code == 204:
                return {
                    "success": True,
                    "platform": "onedrive",
                    "message": "Successfully deleted from OneDrive"
                }
            else:
                return {
                    "success": False,
                    "platform": "onedrive",
                    "error": f"HTTP {response.status_code}",
                    "message": f"Failed to delete from OneDrive: {response.text}"
                }
        
        except Exception as e:
            return {
                "success": False,
                "platform": "onedrive",
                "error": str(e),
                "message": f"Failed to delete from OneDrive: {str(e)}"
            }
    
    def _delete_from_dropbox(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Delete file from Dropbox."""
        if 'dropbox' not in self.platform_clients:
            return {
                "success": False,
                "platform": "dropbox",
                "error": "Dropbox client not available",
                "message": "Dropbox integration not configured"
            }
        
        try:
            dbx = self.platform_clients['dropbox']
            dropbox_path = file_info.get('platform_file_path') or file_info.get('dropbox_path')
            
            if not dropbox_path:
                return {
                    "success": False,
                    "platform": "dropbox",
                    "error": "No Dropbox file path found",
                    "message": "Cannot delete from Dropbox - no file path available"
                }
            
            # Delete file from Dropbox
            dbx.files_delete_v2(dropbox_path)
            
            return {
                "success": True,
                "platform": "dropbox",
                "message": "Successfully deleted from Dropbox"
            }
        
        except Exception as e:
            return {
                "success": False,
                "platform": "dropbox",
                "error": str(e),
                "message": f"Failed to delete from Dropbox: {str(e)}"
            }
    
    def _delete_from_notion(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Delete file from Notion."""
        if 'notion' not in self.platform_clients:
            return {
                "success": False,
                "platform": "notion",
                "error": "Notion client not available",
                "message": "Notion integration not configured"
            }
        
        try:
            notion = self.platform_clients['notion']
            notion_page_id = file_info.get('platform_file_id') or file_info.get('notion_page_id')
            
            if not notion_page_id:
                return {
                    "success": False,
                    "platform": "notion",
                    "error": "No Notion page ID found",
                    "message": "Cannot delete from Notion - no page ID available"
                }
            
            # Archive the page in Notion (Notion doesn't have a true delete)
            notion.pages.update(
                page_id=notion_page_id,
                archived=True
            )
            
            return {
                "success": True,
                "platform": "notion",
                "message": "Successfully archived in Notion"
            }
        
        except Exception as e:
            return {
                "success": False,
                "platform": "notion",
                "error": str(e),
                "message": f"Failed to delete from Notion: {str(e)}"
            }
    
    def _delete_from_local(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Delete file from local storage."""
        try:
            local_path = file_info.get('local_path') or file_info.get('file_path')
            
            if not local_path:
                return {
                    "success": False,
                    "platform": "local",
                    "error": "No local file path found",
                    "message": "Cannot delete from local storage - no file path available"
                }
            
            # Delete local file
            if os.path.exists(local_path):
                os.remove(local_path)
                return {
                    "success": True,
                    "platform": "local",
                    "message": "Successfully deleted from local storage"
                }
            else:
                return {
                    "success": False,
                    "platform": "local",
                    "error": "File not found",
                    "message": "File not found in local storage"
                }
        
        except Exception as e:
            return {
                "success": False,
                "platform": "local",
                "error": str(e),
                "message": f"Failed to delete from local storage: {str(e)}"
            }
    
    def _delete_from_blob_storage(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Delete file from Azure Blob Storage."""
        try:
            blob_name = file_info.get('blob_name') or file_info.get('fileName')
            
            if not blob_name:
                return {
                    "success": False,
                    "platform": "blob_storage",
                    "error": "No blob name found",
                    "message": "Cannot delete from blob storage - no blob name available"
                }
            
            # Delete blob
            blob_client = self.blob_service_client.get_blob_client(
                container=self.blob_container_name,
                blob=blob_name
            )
            
            blob_client.delete_blob()
            
            return {
                "success": True,
                "platform": "blob_storage",
                "message": "Successfully deleted from blob storage"
            }
        
        except Exception as e:
            return {
                "success": False,
                "platform": "blob_storage",
                "error": str(e),
                "message": f"Failed to delete from blob storage: {str(e)}"
            }
    
    def _delete_from_cosmos_db(self, file_id: str, user_id: str) -> Dict[str, Any]:
        """Delete file record from Cosmos DB."""
        try:
            # Delete the document
            self.cosmos_container.delete_item(
                item=file_id,
                partition_key=user_id
            )
            
            return {
                "success": True,
                "platform": "cosmos_db",
                "message": "Successfully deleted from database"
            }
        
        except CosmosResourceNotFoundError:
            return {
                "success": False,
                "platform": "cosmos_db",
                "error": "File not found in database",
                "message": "File record not found in database"
            }
        except Exception as e:
            return {
                "success": False,
                "platform": "cosmos_db",
                "error": str(e),
                "message": f"Failed to delete from database: {str(e)}"
            }


def create_delete_handler(cosmos_container, openai_client) -> DeleteHandler:
    """Create and return a delete handler instance."""
    return DeleteHandler(cosmos_container, openai_client)


def handle_delete_intent(delete_intent: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """
    Handle delete intent from the main brain.
    
    Args:
        delete_intent: Dictionary containing delete parameters
        user_id: User identifier
    
    Returns:
        Delete operation result
    """
    try:
        # This would be called from your main_brain.py
        # You'll need to pass the cosmos_container and openai_client
        from main_brain import create_weez_brain
        
        brain = create_weez_brain()
        delete_handler = create_delete_handler(brain.container, brain.openai_client)
        
        return delete_handler.execute(delete_intent, user_id)
    
    except Exception as e:
        logger.error(f"Error in handle_delete_intent: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to process delete request: {str(e)}"
        }