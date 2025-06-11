import os
import json
import uuid
from datetime import datetime, timezone
from flask import Flask, request, jsonify, redirect, session, url_for
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient
import requests
from requests_oauthlib import OAuth2Session
import io
import mimetypes
from urllib.parse import urlencode
import base64
from datetime import timedelta

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'baigan77')

# Configuration
AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING_1')
AZURE_STORAGE_CONTAINER = os.environ.get('AZURE_STORAGE_CONTAINER', 'weezyaifiles')
AZURE_USER_STORAGE_CONTAINER='weez-users-info'

COSMOS_ENDPOINT = os.getenv('COSMOS_ENDPOINT')
COSMOS_KEY = os.getenv('COSMOS_KEY')
COSMOS_DATABASE = os.getenv('COSMOS_DATABASE', 'weezyai')
COSMOS_CONTAINER = os.getenv('COSMOS_CONTAINER', 'files')

# OAuth Configuration
GOOGLE_CLIENT_ID =os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET =os.getenv('GOOGLE_CLIENT_SECRET')
MICROSOFT_CLIENT_ID = os.getenv('MICROSOFT_CLIENT_ID')
MICROSOFT_CLIENT_SECRET = os.getenv('MICROSOFT_CLIENT_SECRET')
DROPBOX_CLIENT_ID = os.getenv('DROPBOX_CLIENT_ID')
DROPBOX_CLIENT_SECRET = os.getenv('DROPBOX_CLIENT_SECRET')
SLACK_CLIENT_ID = os.getenv('SLACK_CLIENT_ID')
SLACK_CLIENT_SECRET = os.getenv('SLACK_CLIENT_SECRET')
NOTION_CLIENT_ID = os.environ.get('NOTION_CLIENT_ID')
NOTION_CLIENT_SECRET = os.environ.get('NOTION_CLIENT_SECRET')

app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)
app.config['SESSION_COOKIE_SECURE'] = True if os.environ.get('FLASK_ENV') == 'production' else False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

def make_session_permanent():
    """Make current session permanent"""
    session.permanent = True

# Add this after your environment variable declarations
def debug_oauth_config():
    config_status = {
        'GOOGLE_CLIENT_ID': bool(GOOGLE_CLIENT_ID),
        'GOOGLE_CLIENT_SECRET': bool(GOOGLE_CLIENT_SECRET),
        'MICROSOFT_CLIENT_ID': bool(MICROSOFT_CLIENT_ID),
        'MICROSOFT_CLIENT_SECRET': bool(MICROSOFT_CLIENT_SECRET),
        'DROPBOX_CLIENT_ID': bool(DROPBOX_CLIENT_ID),
        'DROPBOX_CLIENT_SECRET': bool(DROPBOX_CLIENT_SECRET),
        'NOTION_CLIENT_ID': bool(NOTION_CLIENT_ID),
        'NOTION_CLIENT_SECRET': bool(NOTION_CLIENT_SECRET),
    }
    print("OAuth Configuration Status:", config_status)
    return config_status

# Call this in debug mode
if app.debug:
    debug_oauth_config()

# Supported file extensions
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.xlsx', '.doc', '.ppt', '.xls'}

# Initialize Azure clients
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
database = cosmos_client.get_database_client(COSMOS_DATABASE)
container = database.get_container_client(COSMOS_CONTAINER)

class PlatformIntegration:
    def __init__(self):
        pass
    
    def get_file_extension(self, filename):
        """Extract file extension from filename"""
        return os.path.splitext(filename.lower())[1]
    
    def is_supported_file(self, filename):
        """Check if file extension is supported"""
        return self.get_file_extension(filename) in SUPPORTED_EXTENSIONS
    
    def upload_to_blob_storage(self, file_content, username, filename):
        """Upload file to Azure Blob Storage"""
        try:
            blob_name = f"{username}/{filename}"
            blob_client = blob_service_client.get_blob_client(
                container=AZURE_STORAGE_CONTAINER, 
                blob=blob_name
            )
            blob_client.upload_blob(file_content, overwrite=True)
            return blob_name
        except Exception as e:
            print(f"Error uploading to blob storage: {e}")
            return None
    
    def save_file_metadata(self, user_id, filename, blob_path, platform, original_date, file_size=None, mime_type=None):
     """Save file metadata to Cosmos DB"""
     try:
         metadata = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "document_title": os.path.splitext(filename)[0],
            "fileName": filename,
            "filePath": blob_path,
            "platform": platform,  # CORRECTED: This correctly marks the platform
            "source": "platform_sync",  # CORRECTED: Added source field to distinguish from local uploads
            "uploaded_at": original_date.isoformat() if isinstance(original_date, datetime) else original_date,
            "file_size": file_size,
            "mime_type": mime_type,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
         container.create_item(body=metadata)
         return metadata
     except Exception as e:
         print(f"Error saving metadata: {e}")
         return None

# Google Drive Integration
class GoogleDriveIntegration(PlatformIntegration):
    def __init__(self):
        super().__init__()
        self.authorization_base_url = 'https://accounts.google.com/o/oauth2/auth'
        self.token_url = 'https://oauth2.googleapis.com/token'
        self.scope = ['https://www.googleapis.com/auth/drive.readonly']
        
    
    def get_auth_url(self):
     """Get Google OAuth authorization URL"""
     if not GOOGLE_CLIENT_ID:
        raise ValueError("GOOGLE_CLIENT_ID not configured")
        
     redirect_uri = url_for('auth_google_callback', _external=True)
     google = OAuth2Session(
        GOOGLE_CLIENT_ID,
        scope=self.scope,
        redirect_uri=redirect_uri
    )
     authorization_url, state = google.authorization_url(
        self.authorization_base_url,
        access_type="offline",
        prompt="select_account"
    )
     session['google_oauth_state'] = state
     return authorization_url

    def get_access_token(self, authorization_response):
     """Exchange authorization code for access token"""
     if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise ValueError("Google OAuth credentials not configured")
        
     redirect_uri = url_for('auth_google_callback', _external=True)
     google = OAuth2Session(
        GOOGLE_CLIENT_ID,
        state=session['google_oauth_state'],  # CORRECTED: Use google_oauth_state
        redirect_uri=redirect_uri
     )
     token = google.fetch_token(
        self.token_url,
        authorization_response=authorization_response,
        client_secret=GOOGLE_CLIENT_SECRET
     )
     return token
    
    def sync_files(self, access_token, user_email):
        """Sync files from Google Drive"""
        headers = {'Authorization': f'Bearer {access_token}'}
        
        # Get files list
        url = 'https://www.googleapis.com/drive/v3/files'
        params = {
            'fields': 'files(id,name,mimeType,createdTime,size)',
            'q': 'trashed=false'
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            return {'error': 'Failed to fetch files from Google Drive'}
        
        files = response.json().get('files', [])
        synced_files = []
        
        for file_info in files:
            filename = file_info.get('name', '')
            if not self.is_supported_file(filename):
                continue
            
            # Download file content
            file_id = file_info['id']
            download_url = f'https://www.googleapis.com/drive/v3/files/{file_id}?alt=media'
            file_response = requests.get(download_url, headers=headers)
            
            if file_response.status_code == 200:
                # Upload to blob storage
                blob_path = self.upload_to_blob_storage(
                    file_response.content, 
                    user_email, 
                    filename
                )
                
                if blob_path:
                    # Save metadata
                    original_date = file_info.get('createdTime')
                    if original_date:
                        original_date = datetime.fromisoformat(original_date.replace('Z', '+00:00'))
                    
                    metadata = self.save_file_metadata(
                        user_email,
                        filename,
                        blob_path,
                        'google_drive',
                        original_date,
                        file_info.get('size'),
                        file_info.get('mimeType')
                    )
                    
                    if metadata:
                        synced_files.append(metadata)
        
        return {'synced_files': synced_files, 'count': len(synced_files)}

# Microsoft OneDrive Integration
class OneDriveIntegration(PlatformIntegration):
    def __init__(self):
        super().__init__()
        self.authorization_base_url = 'https://login.microsoftonline.com/common/oauth2/v2.0/authorize'
        self.token_url = 'https://login.microsoftonline.com/common/oauth2/v2.0/token'
        self.scope = ['Files.Read', 'User.Read']
    
    def get_auth_url(self):
     """Get Microsoft OAuth authorization URL"""
     if not MICROSOFT_CLIENT_ID:
        raise ValueError("MICROSOFT_CLIENT_ID not configured")
        
     redirect_uri = url_for('auth_microsoft_callback', _external=True)
     microsoft = OAuth2Session(
        MICROSOFT_CLIENT_ID,
        scope=self.scope,
        redirect_uri=redirect_uri
     )
     authorization_url, state = microsoft.authorization_url(self.authorization_base_url)
     session['microsoft_oauth_state'] = state
     return authorization_url

    def get_access_token(self, authorization_response):
     """Exchange authorization code for access token"""
     if not MICROSOFT_CLIENT_ID or not MICROSOFT_CLIENT_SECRET:
        raise ValueError("Microsoft OAuth credentials not configured")
        
     redirect_uri = url_for('auth_microsoft_callback', _external=True)
     microsoft = OAuth2Session(
        MICROSOFT_CLIENT_ID,
        state=session['microsoft_oauth_state'],  # CORRECTED: Use microsoft_oauth_state
        redirect_uri=redirect_uri
     )
     token = microsoft.fetch_token(
        self.token_url,
        authorization_response=authorization_response,
        client_secret=MICROSOFT_CLIENT_SECRET
     )
     return token
    
    def sync_files(self, access_token, user_email):
        """Sync files from OneDrive"""
        headers = {'Authorization': f'Bearer {access_token}'}
        
        # Get files list
        url = 'https://graph.microsoft.com/v1.0/me/drive/root/children'
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return {'error': 'Failed to fetch files from OneDrive'}
        
        files = response.json().get('value', [])
        synced_files = []
        
        for file_info in files:
            if file_info.get('folder'):  # Skip folders
                continue
                
            filename = file_info.get('name', '')
            if not self.is_supported_file(filename):
                continue
            
            # Download file content
            download_url = file_info.get('@microsoft.graph.downloadUrl')
            if download_url:
                file_response = requests.get(download_url)
                
                if file_response.status_code == 200:
                    # Upload to blob storage
                    blob_path = self.upload_to_blob_storage(
                        file_response.content, 
                        user_email, 
                        filename
                    )
                    
                    if blob_path:
                        # Save metadata
                        original_date = file_info.get('createdDateTime')
                        if original_date:
                            original_date = datetime.fromisoformat(original_date.replace('Z', '+00:00'))
                        
                        metadata = self.save_file_metadata(
                            user_email,
                            filename,
                            blob_path,
                            'onedrive',
                            original_date,
                            file_info.get('size'),
                            file_info.get('file', {}).get('mimeType')
                        )
                        
                        if metadata:
                            synced_files.append(metadata)
        
        return {'synced_files': synced_files, 'count': len(synced_files)}

# Dropbox Integration
class DropboxIntegration(PlatformIntegration):
    def __init__(self):
        super().__init__()
        self.authorization_base_url = 'https://www.dropbox.com/oauth2/authorize'  # ADD THIS LINE
        self.token_url = 'https://api.dropbox.com/oauth2/token'  # ADD THIS LINE
    
    def get_auth_url(self, user_email=None):
        """Get Dropbox OAuth authorization URL"""
        if not DROPBOX_CLIENT_ID:
            raise ValueError("DROPBOX_CLIENT_ID not configured")
            
        redirect_uri = url_for('auth_dropbox_callback', _external=True)
        state = str(uuid.uuid4())
        
        # Store user_email in session with state
        session[f'dropbox_oauth_state_{state}'] = {
            'state': state,
            'user_email': user_email
        }
        
        params = {
            'client_id': DROPBOX_CLIENT_ID,
            'redirect_uri': redirect_uri,
            'response_type': 'code',
            'state': state
        }
        return f"{self.authorization_base_url}?{urlencode(params)}"
    
    def get_access_token(self, authorization_code, state):
        """Exchange authorization code for access token"""
        if not DROPBOX_CLIENT_ID or not DROPBOX_CLIENT_SECRET:
            raise ValueError("Dropbox OAuth credentials not configured")
            
        # CORRECTED: Verify state and get user_email
        state_data = session.get(f'dropbox_oauth_state_{state}')
        if not state_data or state_data['state'] != state:
            raise ValueError("State mismatch - possible CSRF attack")
            
        redirect_uri = url_for('auth_dropbox_callback', _external=True)
        data = {
            'code': authorization_code,
            'grant_type': 'authorization_code',
            'client_id': DROPBOX_CLIENT_ID,
            'client_secret': DROPBOX_CLIENT_SECRET,
            'redirect_uri': redirect_uri
        }
        
        response = requests.post(self.token_url, data=data)
        if response.status_code != 200:
            raise ValueError(f"Failed to get access token: {response.text}")
        
        # Clean up session
        del session[f'dropbox_oauth_state_{state}']
        
        return response.json(), state_data['user_email']
    
    def sync_files(self, access_token, user_email):
        """Sync files from Dropbox"""
        headers = {'Authorization': f'Bearer {access_token}'}
        
        # List files
        url = 'https://api.dropboxapi.com/2/files/list_folder'
        data = {'path': '', 'recursive': True}
        
        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            return {'error': 'Failed to fetch files from Dropbox'}
        
        files = response.json().get('entries', [])
        synced_files = []
        
        for file_info in files:
            if file_info.get('.tag') != 'file':  # Skip folders
                continue
                
            filename = file_info.get('name', '')
            if not self.is_supported_file(filename):
                continue
            
            # Download file content
            download_url = 'https://content.dropboxapi.com/2/files/download'
            download_headers = {
                'Authorization': f'Bearer {access_token}',
                'Dropbox-API-Arg': json.dumps({'path': file_info['path_lower']})
            }
            
            file_response = requests.post(download_url, headers=download_headers)
            
            if file_response.status_code == 200:
                # Upload to blob storage
                blob_path = self.upload_to_blob_storage(
                    file_response.content, 
                    user_email, 
                    filename
                )
                
                if blob_path:
                    # Save metadata
                    original_date = file_info.get('client_modified')
                    if original_date:
                        original_date = datetime.fromisoformat(original_date.replace('Z', '+00:00'))
                    
                    metadata = self.save_file_metadata(
                        user_email,
                        filename,
                        blob_path,
                        'dropbox',
                        original_date,
                        file_info.get('size')
                    )
                    
                    if metadata:
                        synced_files.append(metadata)
        
        return {'synced_files': synced_files, 'count': len(synced_files)}

# Notion Integration
class NotionIntegration(PlatformIntegration):
    def __init__(self):
        super().__init__()
        self.authorization_base_url = 'https://api.notion.com/v1/oauth/authorize'  # Use the API URL
        self.token_url = 'https://api.notion.com/v1/oauth/token'
    
    def get_auth_url(self, user_email=None):
        """Get Notion OAuth authorization URL with user_email in state"""
        if not NOTION_CLIENT_ID:
            raise ValueError("NOTION_CLIENT_ID not configured")
            
        redirect_uri = url_for('auth_notion_callback', _external=True)
        
        # SOLUTION 2: Include user_email in state parameter
        state_data = {
            'uuid': str(uuid.uuid4()),
            'user_email': user_email
        }
        state = base64.urlsafe_b64encode(json.dumps(state_data).encode()).decode()
        
        session['notion_oauth_state'] = state
        
        params = {
            'response_type': 'code',
            'client_id': NOTION_CLIENT_ID,
            'redirect_uri': redirect_uri,
            'owner': 'user',
            'state': state
        }
        return f"{self.authorization_base_url}?{urlencode(params)}"
    
    def get_access_token(self, authorization_code, state=None):
        """Exchange authorization code for access token"""
        # Decode state to get user_email
        user_email = None
        if state and state.strip():
            try:
                state_data = json.loads(base64.urlsafe_b64decode(state.encode()).decode())
                user_email = state_data.get('user_email')
                print(f"User email from state: {user_email}")
            except:
                print("Could not decode state parameter")
        
        # Verify state if provided
        if state and state.strip() and state != session.get('notion_oauth_state'):
            raise ValueError("State mismatch - possible CSRF attack")
            
        redirect_uri = url_for('auth_notion_callback', _external=True)
        auth_string = base64.b64encode(f"{NOTION_CLIENT_ID}:{NOTION_CLIENT_SECRET}".encode()).decode()
        
        headers = {
            'Authorization': f'Basic {auth_string}',
            'Content-Type': 'application/json',
            'Notion-Version': '2022-06-28'
        }
        
        data = {
            'grant_type': 'authorization_code',
            'code': authorization_code,
            'redirect_uri': redirect_uri
        }
        
        print(f"Sending token request with data: {data}")
        
        response = requests.post(self.token_url, headers=headers, json=data)
        
        print(f"Token response status: {response.status_code}")
        print(f"Token response: {response.text}")
        
        if response.status_code != 200:
            raise ValueError(f"Failed to get access token: {response.text}")
        
        token_data = response.json()
        
        # Add user_email to token data for easier access
        if user_email:
            token_data['user_email'] = user_email
        
        # Clean up session state
        if 'notion_oauth_state' in session:
            del session['notion_oauth_state']
            
        return token_data
    
    
    def sync_files(self, access_token, user_email):
        """Sync files from Notion"""
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Notion-Version': '2022-06-28',
            'Content-Type': 'application/json'
        }
        
        # Search for pages and databases
        search_url = 'https://api.notion.com/v1/search'
        search_data = {
            'filter': {
                'property': 'object',
                'value': 'page'
            }
        }
        
        response = requests.post(search_url, headers=headers, json=search_data)
        if response.status_code != 200:
            return {'error': 'Failed to fetch pages from Notion'}
        
        pages = response.json().get('results', [])
        synced_files = []
        
        for page in pages:
            page_id = page['id']
            page_title = 'Untitled'
            
            # Get page title
            if page.get('properties'):
                title_prop = None
                for prop_name, prop_data in page['properties'].items():
                    if prop_data.get('type') == 'title':
                        title_prop = prop_data
                        break
                
                if title_prop and title_prop.get('title'):
                    page_title = ''.join([t['plain_text'] for t in title_prop['title']])
            
            # Export page as PDF (Notion doesn't provide direct file download)
            # We'll create a text representation and save it as .txt for now
            # In production, you might want to use a service to convert to PDF
            
            # Get page content
            blocks_url = f'https://api.notion.com/v1/blocks/{page_id}/children'
            blocks_response = requests.get(blocks_url, headers=headers)
            
            if blocks_response.status_code == 200:
                blocks = blocks_response.json().get('results', [])
                content = self.extract_text_from_blocks(blocks)
                
                # Create filename
                safe_title = "".join(c for c in page_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                filename = f"{safe_title}.txt"
                
                # Upload to blob storage
                blob_path = self.upload_to_blob_storage(
                    content.encode('utf-8'), 
                    user_email, 
                    filename
                )
                
                if blob_path:
                    # Save metadata
                    original_date = page.get('created_time')
                    if original_date:
                        original_date = datetime.fromisoformat(original_date.replace('Z', '+00:00'))
                    
                    metadata = self.save_file_metadata(
                        user_email,
                        filename,
                        blob_path,
                        'notion',
                        original_date,
                        len(content.encode('utf-8')),
                        'text/plain'
                    )
                    
                    if metadata:
                        synced_files.append(metadata)
        
        return {'synced_files': synced_files, 'count': len(synced_files)}
    
    def extract_text_from_blocks(self, blocks):
        """Extract text content from Notion blocks"""
        content = []
        
        for block in blocks:
            block_type = block.get('type', '')
            
            if block_type == 'paragraph':
                text = self.extract_rich_text(block.get('paragraph', {}).get('rich_text', []))
                if text:
                    content.append(text)
            
            elif block_type == 'heading_1':
                text = self.extract_rich_text(block.get('heading_1', {}).get('rich_text', []))
                if text:
                    content.append(f"# {text}")
            
            elif block_type == 'heading_2':
                text = self.extract_rich_text(block.get('heading_2', {}).get('rich_text', []))
                if text:
                    content.append(f"## {text}")
            
            elif block_type == 'heading_3':
                text = self.extract_rich_text(block.get('heading_3', {}).get('rich_text', []))
                if text:
                    content.append(f"### {text}")
            
            elif block_type == 'bulleted_list_item':
                text = self.extract_rich_text(block.get('bulleted_list_item', {}).get('rich_text', []))
                if text:
                    content.append(f"• {text}")
            
            elif block_type == 'numbered_list_item':
                text = self.extract_rich_text(block.get('numbered_list_item', {}).get('rich_text', []))
                if text:
                    content.append(f"1. {text}")
            
            elif block_type == 'code':
                text = self.extract_rich_text(block.get('code', {}).get('rich_text', []))
                language = block.get('code', {}).get('language', '')
                if text:
                    content.append(f"```{language}\n{text}\n```")
        
        return '\n\n'.join(content)
    
    def extract_rich_text(self, rich_text_array):
        """Extract plain text from Notion rich text array"""
        return ''.join([rt.get('plain_text', '') for rt in rich_text_array])

# Initialize platform integrations
google_drive = GoogleDriveIntegration()
onedrive = OneDriveIntegration()
dropbox_integration = DropboxIntegration()
notion_integration = NotionIntegration()


def store_user_token(user_email, platform, token_data):
    """Store user's platform token in their blob storage userInfo.json"""
    try:
        print(f"Attempting to store token for {user_email} on platform {platform}")  # Debug log
        
        blob_name = f"{user_email}/userInfo.json"
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_USER_STORAGE_CONTAINER, 
            blob=blob_name
        )
        
        # Try to get existing user info
        try:
            existing_data = blob_client.download_blob().readall()
            user_info = json.loads(existing_data.decode('utf-8'))
            print(f"Loaded existing user info for {user_email}")  # Debug log
        except Exception as e:
            print(f"Creating new user info file for {user_email}, error loading existing: {e}")  # Debug log
            # If file doesn't exist, create new structure
            user_info = {
                "email": user_email,
                "platform_tokens": {}
            }
        
        # Add/update platform token
        if "platform_tokens" not in user_info:
            user_info["platform_tokens"] = {}
            
        user_info["platform_tokens"][platform] = {
            "access_token": token_data.get("access_token"),
            "refresh_token": token_data.get("refresh_token"),
            "token_type": token_data.get("token_type"),
            "expires_at": token_data.get("expires_at"),
            "expires_in": token_data.get("expires_in"),
            "scope": token_data.get("scope"),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Upload updated user info
        updated_json = json.dumps(user_info, indent=2)
        blob_client.upload_blob(
            updated_json.encode('utf-8'), 
            overwrite=True
        )
        
        print(f"Successfully stored {platform} token for {user_email}")  # Debug log
        print(f"Token preview: {token_data.get('access_token', 'N/A')[:10]}...")  # Debug log
        
    except Exception as e:
        print(f"Error storing user token for {user_email}: {e}")
        import traceback
        traceback.print_exc()


def load_user_token(user_email, platform):
    """Load user's platform token from their blob storage"""
    try:
        blob_name = f"{user_email}/userInfo.json"
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_USER_STORAGE_CONTAINER, 
            blob=blob_name
        )
        
        data = blob_client.download_blob().readall()
        user_info = json.loads(data.decode('utf-8'))
        
        return user_info.get("platform_tokens", {}).get(platform)
        
    except Exception as e:
        print(f"Error loading user token: {e}")
        return None

# Routes
@app.route('/')
def index():
    return jsonify({
        'message': 'Multi-Platform File Integration API',
        'available_platforms': ['google_drive', 'onedrive', 'dropbox', 'notion', 'slack'],
        'endpoints': {
            'auth': '/auth/{platform}',
            'sync': '/sync/{platform}',
            'status': '/status'
        }
    })

# Google Drive Routes
@app.route('/auth/google')
def auth_google():
    """Initiate Google Drive OAuth"""
    auth_url = google_drive.get_auth_url()
    return redirect(auth_url)

@app.route('/auth/google/callback')
def auth_google_callback():
    """Handle Google Drive OAuth callback"""
    try:
        if 'error' in request.args:
            return jsonify({'error': f"OAuth error: {request.args.get('error')}"}), 400
        
        # CORRECTED: Use google_oauth_state instead of oauth_state
        if request.args.get('state') != session.get('google_oauth_state'):
            return jsonify({'error': 'State mismatch - possible CSRF attack'}), 400
            
        token = google_drive.get_access_token(request.url)
        make_session_permanent()
        session['google_token'] = token
        
        # CORRECTED: Store token in user's blob storage
        user_email = session.get('user_email')  # Assuming you set this during login
        if user_email:
            store_user_token(user_email, 'google_drive', token)
        
        return jsonify({'message': 'Google Drive authentication successful', 'token_type': token.get('token_type')})
    except Exception as e:
        print(f"Google OAuth callback error: {str(e)}")
        return jsonify({'error': f"Authentication failed: {str(e)}"}), 400


# Apply similar pattern to other callback routes

@app.route('/sync/google_drive')
def sync_google_drive():
    """Sync files from Google Drive"""
    user_email = request.args.get('user_email')
    if not user_email:
        return jsonify({'error': 'user_email parameter required'}), 400
    
    try:
        # CORRECTED: Load token from blob storage instead of session
        token_data = load_user_token(user_email, 'google_drive')
        if not token_data:
            return jsonify({'error': 'Not authenticated with Google Drive'}), 401
            
        access_token = token_data['access_token']
        result = google_drive.sync_files(access_token, user_email)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Microsoft OneDrive Routes
@app.route('/auth/microsoft')
def auth_microsoft():
    """Initiate Microsoft OneDrive OAuth"""
    auth_url = onedrive.get_auth_url()
    return redirect(auth_url)

@app.route('/test/tokens/<user_email>')
def test_user_tokens(user_email):
    """Test endpoint to check stored tokens for a user"""
    try:
        blob_name = f"{user_email}/userInfo.json"
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_USER_STORAGE_CONTAINER, 
            blob=blob_name
        )
        
        data = blob_client.download_blob().readall()
        user_info = json.loads(data.decode('utf-8'))
        
        # Don't expose full tokens, just status
        token_status = {}
        for platform, token_data in user_info.get("platform_tokens", {}).items():
            token_status[platform] = {
                'has_access_token': bool(token_data.get('access_token')),
                'updated_at': token_data.get('updated_at'),
                'token_type': token_data.get('token_type')
            }
        
        return jsonify({
            'user_email': user_email,
            'token_status': token_status
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/auth/microsoft/callback')
def auth_microsoft_callback():
    """Handle Microsoft OneDrive OAuth callback"""
    try:
        if 'error' in request.args:
            return jsonify({'error': f"OAuth error: {request.args.get('error')}"}), 400
        
        # CORRECTED: Use microsoft_oauth_state instead of oauth_state    
        if request.args.get('state') != session.get('microsoft_oauth_state'):
            return jsonify({'error': 'State mismatch - possible CSRF attack'}), 400
            
        token = onedrive.get_access_token(request.url)
        make_session_permanent()
        session['microsoft_token'] = token
        
        # CORRECTED: Store token in user's blob storage
        user_email = session.get('user_email')
        if user_email:
            store_user_token(user_email, 'onedrive', token)
            
        return jsonify({'message': 'Microsoft OneDrive authentication successful', 'token_type': token.get('token_type')})
    except Exception as e:
        print(f"Microsoft OAuth callback error: {str(e)}")
        return jsonify({'error': f"Authentication failed: {str(e)}"}), 400


@app.route('/sync/onedrive')
def sync_onedrive():
    """Sync files from OneDrive"""
    if 'microsoft_token' not in session:
        return jsonify({'error': 'Not authenticated with Microsoft OneDrive'}), 401
    
    user_email = request.args.get('user_email')
    if not user_email:
        return jsonify({'error': 'user_email parameter required'}), 400
    
    try:
        access_token = session['microsoft_token']['access_token']
        result = onedrive.sync_files(access_token, user_email)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Dropbox Routes

@app.route('/auth/dropbox')
def auth_dropbox():
    """Initiate Dropbox OAuth"""
    user_email = request.args.get('user_email')
    if not user_email:
        return jsonify({'error': 'user_email parameter required'}), 400
        
    auth_url = dropbox_integration.get_auth_url(user_email)
    return redirect(auth_url)

# CORRECTED: Updated callback to properly handle user_email
@app.route('/auth/dropbox/callback')
def auth_dropbox_callback():
    """Handle Dropbox OAuth callback"""
    try:
        if 'error' in request.args:
            return jsonify({'error': f"OAuth error: {request.args.get('error')}"}), 400
            
        auth_code = request.args.get('code')
        state = request.args.get('state')
        
        if not auth_code or not state:
            return jsonify({'error': 'Missing authorization code or state'}), 400
            
        # CORRECTED: Get token and user_email together
        token, user_email = dropbox_integration.get_access_token(auth_code, state)
        make_session_permanent()
        session['dropbox_token'] = token
        
        # CORRECTED: Now we have user_email to store the token
        if user_email:
            store_user_token(user_email, 'dropbox', token)
            print(f"Token stored for user: {user_email}")  # Debug log
        else:
            print("Warning: No user_email found, token not stored")  # Debug log
            
        return jsonify({
            'message': 'Dropbox authentication successful', 
            'token_type': token.get('token_type'),
            'user_email': user_email
        })
    except Exception as e:
        print(f"Dropbox OAuth callback error: {str(e)}")
        return jsonify({'error': f"Authentication failed: {str(e)}"}), 400


@app.route('/sync/dropbox')
def sync_dropbox():
    """Sync files from Dropbox"""
    user_email = request.args.get('user_email')
    if not user_email:
        return jsonify({'error': 'user_email parameter required'}), 400
    
    try:
        # CORRECTED: Load token from blob storage instead of session
        token_data = load_user_token(user_email, 'dropbox')
        if not token_data:
            return jsonify({'error': 'Not authenticated with Dropbox'}), 401
            
        access_token = token_data['access_token']
        result = dropbox_integration.sync_files(access_token, user_email)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Notion Routes
@app.route('/sync/notion')
def sync_notion():
    """Sync files from Notion"""
    user_email = request.args.get('user_email')
    if not user_email:
        return jsonify({'error': 'user_email parameter required'}), 400
    
    try:
        # Load token from blob storage
        token_data = load_user_token(user_email, 'notion')
        if not token_data:
            return jsonify({
                'error': 'Not authenticated with Notion',
                'suggestion': f'Please authenticate first at /auth/notion/{user_email}'
            }), 401
            
        access_token = token_data['access_token']
        print(f"Using access token for sync: {access_token[:10]}...")  # Debug log
        
        result = notion_integration.sync_files(access_token, user_email)
        return jsonify(result)
    except Exception as e:
        print(f"Notion sync error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/auth/notion/callback')
def auth_notion_callback():
    """Handle Notion OAuth callback - Updated version"""
    try:
        if 'error' in request.args:
            error_description = request.args.get('error_description', 'Unknown error')
            return jsonify({
                'error': f"OAuth error: {request.args.get('error')}",
                'description': error_description
            }), 400

        auth_code = request.args.get('code')
        state = request.args.get('state')
        
        if not auth_code:
            return jsonify({'error': 'No authorization code received'}), 400
            
        print(f"Received auth code: {auth_code[:10]}...")
        print(f"Received state: '{state}'")
        print(f"Session contents: {dict(session)}")  # Debug log
        
        # Get access token
        token = notion_integration.get_access_token(auth_code, state if state else None)
        make_session_permanent()
        session['notion_token'] = token
        
        print(f"Received token: {token}")
        
        # Try multiple ways to get user_email
        user_email = None
        
        # Method 1: From token data (if we added it in get_access_token)
        user_email = token.get('user_email')
        
        # Method 2: From session
        if not user_email:
            user_email = session.get('user_email')
            
        # Method 3: From token owner info
        if not user_email and 'owner' in token:
            owner_info = token.get('owner', {})
            if owner_info.get('type') == 'user':
                user_info = owner_info.get('user', {})
                if user_info.get('type') == 'person':
                    person_info = user_info.get('person', {})
                    user_email = person_info.get('email')
        
        print(f"Final user_email: {user_email}")
        
        # Store token
        if user_email:
            store_user_token(user_email, 'notion', token)
            print(f"✅ Token stored successfully for user: {user_email}")
        else:
            print("❌ No user_email found - token not stored")
            
        return jsonify({
            'message': 'Notion authentication successful!',
            'token_type': token.get('token_type'),
            'workspace_name': token.get('workspace_name', 'Unknown'),
            'workspace_id': token.get('workspace_id'),
            'bot_id': token.get('bot_id'),
            'user_email': user_email,
            'token_stored': bool(user_email)
        })
        
    except Exception as e:
        print(f"Notion OAuth callback error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Authentication failed: {str(e)}"}), 400

# Don't forget to add session configuration at the top of your file
# app.config['SESSION_PERMANENT'] = True
# app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour

@app.route('/auth/notion/<user_email>')
def auth_notion_with_email(user_email):
    """Initiate Notion OAuth with user email"""
    try:
        # Store user_email in session for callback
        make_session_permanent()
        session['user_email'] = user_email
        session.permanent = True  # Make session permanent
        
        print(f"Storing user_email in session: {user_email}")  # Debug log
        print(f"Session after storing: {dict(session)}")  # Debug log
        
        # SOLUTION 2: Use the updated get_auth_url method that includes user_email in state
        auth_url = notion_integration.get_auth_url(user_email)
        return redirect(auth_url)
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to initiate Notion authentication',
            'details': str(e)
        }), 500


# Status endpoint
@app.route('/status')
def status():
    """Check authentication status for all platforms"""
    return jsonify({
        'google_drive': 'google_token' in session,
        'onedrive': 'microsoft_token' in session,
        'dropbox': 'dropbox_token' in session,
        'slack': 'slack_token' in session,
        'notion': 'notion_token' in session
    })

# File management endpoints
@app.route('/files/<user_email>')
def get_user_files(user_email):
    """Get all files for a user from Cosmos DB"""
    try:
        query = f"SELECT * FROM c WHERE c.user_id = '{user_email}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        return jsonify({'files': items, 'count': len(items)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/files/<user_email>/<platform>')
def get_user_files_by_platform(user_email, platform):
    """Get files for a user from a specific platform"""
    try:
        query = f"SELECT * FROM c WHERE c.user_id = '{user_email}' AND c.platform = '{platform}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        return jsonify({'files': items, 'count': len(items)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/session/info')
def session_info():
    """Check session information"""
    return jsonify({
        'session_permanent': session.permanent,
        'session_keys': list(session.keys()),
        'has_tokens': {
            'google_drive': 'google_token' in session,
            'microsoft': 'microsoft_token' in session,
            'dropbox': 'dropbox_token' in session,
            'notion': 'notion_token' in session
        }
    })


if __name__ == '__main__':
    app.run(debug=True)