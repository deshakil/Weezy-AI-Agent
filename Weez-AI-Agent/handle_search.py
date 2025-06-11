"""
Weezy AI Agent - Search Handler (FIXED VERSION)
Handles semantic file search across multiple platforms using vector embeddings.
Returns only the essential response body without ChatGPT responses.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import os
from azure.cosmos import CosmosClient, PartitionKey
from openai import AzureOpenAI
import json
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchHandler:
    def __init__(self):
        """Initialize the search handler with Azure services."""
        self.cosmos_client = None
        self.database = None
        self.container = None
        self.openai_client = None
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize Azure Cosmos DB and OpenAI clients."""
        try:
            # Initialize Cosmos DB
            cosmos_endpoint = os.getenv('COSMOS_ENDPOINT')
            cosmos_key = os.getenv('COSMOS_KEY')
            database_name = 'weezyai'
            container_name = 'files'
            
            self.cosmos_client = CosmosClient(cosmos_endpoint, cosmos_key)
            self.database = self.cosmos_client.get_database_client(database_name)
            self.container = self.database.get_container_client(container_name)
            
            # Initialize Azure OpenAI
            self.openai_client = AzureOpenAI(
                api_key=os.getenv('OPENAI_API_KEY'),
                api_version="2024-12-01-preview",
                azure_endpoint="https://weez-openai-resource.openai.azure.com/"
            )
            
            logger.info("Search handler services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {str(e)}")
            raise
    
    def handle_search(self, intent: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Main search handler that processes search intents.
        
        Args:
            intent: Parsed intent from GPT-4o with keys: action, platform, file_name, date_filter, keywords, limit
            user_id: User identifier for scoping results
            
        Returns:
            Dict containing search results and metadata (no ChatGPT responses)
        """
        try:
            logger.info(f"Processing search intent for user {user_id}: {intent}")
            
            # Extract search parameters
            platform = intent.get('platform')
            file_name = intent.get('file_name')
            date_filter = intent.get('date_filter')
            keywords = intent.get('keywords', [])
            limit = intent.get('limit', 10)
            
            # Modified validation: Allow searches with platform and/or date_filter
            if not keywords and not file_name and not platform and not date_filter:
                # Return all files when no criteria specified
                return self._search_all_files(user_id, limit)
            
            # Perform search based on available parameters
            if file_name:
                results = self._search_by_filename(user_id, file_name, platform, date_filter, limit)
            else:
                results = self._search_by_semantic_similarity(user_id, keywords, platform, date_filter, limit)
            
            # Format and return results
            return self._format_search_results(results, intent)
            
        except Exception as e:
            logger.error(f"Error in search handler: {str(e)}")
            return self._error_response(f"Search failed: {str(e)}")
    
    def _search_by_filename(self, user_id: str, file_name: str, platform: Optional[str], date_filter: Optional[str], limit: Optional[int] = None) -> List[Dict]:
        """Search for files by filename with optional platform and date filtering."""
        try:
            # Build SQL query with proper user isolation
            query = "SELECT * FROM c WHERE c.user_id = @user_id"
            parameters = [{"name": "@user_id", "value": user_id}]
            
            # Add filename filter (case-insensitive partial match)
            query += " AND CONTAINS(LOWER(c.fileName), LOWER(@file_name))"
            parameters.append({"name": "@file_name", "value": file_name})
            
            # Add platform filter if specified
            if platform:
                query += " AND c.platform = @platform"
                parameters.append({"name": "@platform", "value": platform})
            
            # Add date filter with flexible range
            if date_filter:
                date_obj = datetime.fromisoformat(date_filter.replace('Z', '+00:00'))
                date_start = (date_obj - timedelta(days=1)).isoformat()
                date_end = (date_obj + timedelta(days=2)).isoformat()
                query += " AND c.uploaded_at >= @date_start AND c.uploaded_at < @date_end"
                parameters.extend([
                    {"name": "@date_start", "value": date_start},
                    {"name": "@date_end", "value": date_end}
                ])
            
            # Order by date
            query += " ORDER BY c.uploaded_at DESC"
            
            # Execute query
            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            # Apply limit if specified
            if limit and limit > 0:
                items = items[:limit]
                logger.info(f"User {user_id}: Limited results to {limit} files")
            
            logger.info(f"User {user_id}: Found {len(items)} files matching filename '{file_name}'")
            return items
            
        except Exception as e:
            logger.error(f"Error in filename search for user {user_id}: {str(e)}")
            raise
    
    def _search_by_semantic_similarity(self, user_id: str, keywords: List[str], platform: Optional[str], date_filter: Optional[str], limit: Optional[int] = None) -> List[Dict]:
        """Search for files using semantic similarity with vector embeddings or fallback text search."""
        try:
            # Build filter conditions with proper user isolation
            filter_conditions = [f"c.user_id = '{user_id}'"]
            
            if platform:
                filter_conditions.append(f"c.platform = '{platform}'")
            
            if date_filter:
                date_obj = datetime.fromisoformat(date_filter.replace('Z', '+00:00'))
                date_start = (date_obj - timedelta(days=1)).isoformat()
                date_end = (date_obj + timedelta(days=2)).isoformat()
                filter_conditions.append(f"c.uploaded_at >= '{date_start}' AND c.uploaded_at < '{date_end}'")
            
            where_clause = " AND ".join(filter_conditions)
            
            # Try vector search if keywords are provided
            if keywords:
                search_text = " ".join(keywords)
                search_embedding = self._generate_embedding(search_text)
                
                try:
                    # Execute vector search query - FIXED: Using proper VectorDistance syntax
                    query = f"""
                        SELECT c.id, c.user_id, c.fileName, c.platform, c.uploaded_at, 
                               c.document_title, c.textSummary, c.filePath,
                               VectorDistance(c.embedding, @embedding) AS distance
                        FROM c 
                        WHERE {where_clause} 
                        ORDER BY VectorDistance(c.embedding, @embedding)
                    """
                    
                    items = list(self.container.query_items(
                        query=query,
                        parameters=[{"name": "@embedding", "value": search_embedding}],
                        enable_cross_partition_query=True
                    ))
                    
                    logger.info(f"Vector search succeeded with {len(items)} results")
                    
                    # Log all results with their similarity scores for debugging
                    logger.info("=== VECTOR SEARCH RESULTS ===")
                    for i, item in enumerate(items, 1):
                        distance = item.get('distance', float('inf'))
                        logger.info(f"Result {i}: {item.get('fileName')} - Distance: {distance:.4f}")
                    
                    # FIXED: Convert distance to similarity and apply more lenient threshold
                    logger.info("=== SIMILARITY CALCULATIONS ===")
                    for item in items:
                        distance = item.get('distance', float('inf'))
                        # Convert distance to similarity (0-1 scale, where 1 is perfect match)
                        # Using exponential decay: similarity = e^(-distance)
                        similarity = math.exp(-distance) if distance != float('inf') else 0
                        item['similarity'] = similarity
                        logger.info(f"📄 File: {item.get('fileName')}")
                        logger.info(f"   📊 Distance: {distance:.4f}, Similarity: {similarity:.4f} ({similarity*100:.1f}%)")
                        logger.info(f"   📝 Title: {item.get('document_title', 'N/A')}")
                        logger.info(f"   📋 Summary: {(item.get('textSummary', 'N/A')[:100])}")
                        logger.info("   " + "─"*50)
                    
                    # FIXED: Balanced similarity threshold to avoid bad results
                    similarity_threshold = 0.6
                    filtered_items = [item for item in items if item.get('similarity', 0) >= similarity_threshold]
                    
                    logger.info(f"=== FILTERING RESULTS (Threshold: {similarity_threshold}) ===")
                    logger.info(f"✅ Results passing threshold: {len(filtered_items)}")
                    logger.info(f"❌ Results filtered out: {len(items) - len(filtered_items)}")
                    
                    if filtered_items:
                        logger.info("📋 FILES MEETING THRESHOLD:")
                        for item in filtered_items:
                            logger.info(f"   ✓ {item.get('fileName')} - {item.get('similarity')*100:.1f}% match")
                    
                    if len(items) - len(filtered_items) > 0:
                        logger.info("📋 FILES BELOW THRESHOLD:")
                        below_threshold = [item for item in items if item.get('similarity', 0) < similarity_threshold]
                        for item in below_threshold[:3]:  # Show top 3 below threshold
                            logger.info(f"   ✗ {item.get('fileName')} - {item.get('similarity')*100:.1f}% match")
                    
                    # If no items pass the threshold, return top 5 results with warning
                    if not filtered_items and items:
                        logger.info("⚠️  NO ITEMS PASSED SIMILARITY THRESHOLD - USING FALLBACK")
                        logger.info("🔄 Returning top 5 results with lower confidence scores")
                        filtered_items = items[:5]
                        for item in filtered_items:
                            original_similarity = item.get('similarity', 0)
                            item['similarity'] = 0.3  # Assign lower similarity to indicate uncertainty
                            logger.info(f"   🔄 {item.get('fileName')} - Original: {original_similarity*100:.1f}%, Adjusted: 30%")
                    
                    logger.info(f"📊 FINAL RESULTS: {len(filtered_items)} files returned")
                    items = filtered_items
                    
                except Exception as vector_error:
                    logger.warning(f"Vector search failed: {str(vector_error)}, falling back to text search")
                    items = self._fallback_text_search(user_id, keywords, platform, date_filter)
            else:
                # No keywords: Use basic query with platform and/or date filter
                query = f"SELECT * FROM c WHERE {where_clause} ORDER BY c.uploaded_at DESC"
                items = list(self.container.query_items(
                    query=query,
                    parameters=[],
                    enable_cross_partition_query=True
                ))
            
            # Apply limit if specified
            default_limit = 10
            effective_limit = limit if limit and limit > 0 else default_limit
            final_results = items[:effective_limit]
            
            logger.info(f"User {user_id}: Found {len(final_results)} files for keywords: {keywords}")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in semantic search for user {user_id}: {str(e)}")
            raise
    
    def _fallback_text_search(self, user_id: str, keywords: List[str], platform: Optional[str], date_filter: Optional[str]) -> List[Dict]:
        """Fallback text-based search when vector search is not available."""
        try:
            query = "SELECT * FROM c WHERE c.user_id = @user_id"
            parameters = [{"name": "@user_id", "value": user_id}]
            
            # Add keyword search across multiple fields
            keyword_conditions = []
            for i, keyword in enumerate(keywords):
                param_name = f"@keyword{i}"
                keyword_conditions.append(f"""(
                    CONTAINS(LOWER(c.document_title), LOWER({param_name})) OR 
                    CONTAINS(LOWER(c.textSummary), LOWER({param_name})) OR 
                    CONTAINS(LOWER(c.fileName), LOWER({param_name}))
                )""")
                parameters.append({"name": param_name, "value": keyword})
            
            if keyword_conditions:
                query += " AND (" + " OR ".join(keyword_conditions) + ")"
            
            # Add platform filter if specified
            if platform:
                query += " AND c.platform = @platform"
                parameters.append({"name": "@platform", "value": platform})
            
            # Add date filter if specified
            if date_filter:
                date_obj = datetime.fromisoformat(date_filter.replace('Z', '+00:00'))
                date_start = (date_obj - timedelta(days=1)).isoformat()
                date_end = (date_obj + timedelta(days=2)).isoformat()
                query += " AND c.uploaded_at >= @date_start AND c.uploaded_at < @date_end"
                parameters.extend([
                    {"name": "@date_start", "value": date_start},
                    {"name": "@date_end", "value": date_end}
                ])
            
            query += " ORDER BY c.uploaded_at DESC"
            
            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            logger.info(f"Fallback text search returned {len(items)} results")
            return items
            
        except Exception as e:
            logger.error(f"Error in fallback text search: {str(e)}")
            return []
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for the given text using Azure OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def _format_search_results(self, results: List[Dict], intent: Dict[str, Any]) -> Dict[str, Any]:
        """Format search results with essential data only (no ChatGPT responses)."""
        try:
            if not results:
                return {
                    "success": True,
                    "action": "search",
                    "file_data": {
                        "files": [],
                        "count": 0,
                        "has_more": False
                    },
                    "metadata": {
                        "search_terms": intent.get('keywords', []) or [intent.get('file_name', '')],
                        "platform": intent.get('platform'),
                        "date_filter": intent.get('date_filter'),
                        "limit": intent.get('limit', 10),
                        "total_found": 0,
                        "user_scoped": True,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            # Format file data for frontend rendering
            file_data = []
            for item in results:
                file_item = {
                    "id": item.get('id'),
                    "file_name": item.get('fileName'),
                    "document_title": item.get('document_title'),
                    "platform": item.get('platform'),
                    "date": item.get('uploaded_at'),
                    "formatted_date": self._format_date(item.get('uploaded_at')),
                    "file_path": item.get('filePath'),
                    "text_summary": item.get('textSummary', ''),
                    "preview_summary": (item.get('textSummary', '')[:200] + "...") if len(item.get('textSummary', '')) > 200 else item.get('textSummary', ''),
                    "similarity_score": round(item.get('similarity', 0), 3) if item.get('similarity') else None,
                    "similarity_percentage": int(item.get('similarity', 0) * 100) if item.get('similarity') else None,
                    "file_size": item.get('file_size'),
                    "file_type": item.get('file_type'),
                    "download_url": item.get('download_url'),
                    "preview_url": item.get('preview_url')
                }
                file_data.append(file_item)
            
            return {
                "success": True,
                "action": "search",
                "file_data": {
                    "files": file_data,
                    "count": len(file_data),
                    "has_more": False,  # Simplified: Assuming no pagination for now
                    "display_limit": min(10, len(file_data))
                },
                "metadata": {
                    "search_terms": intent.get('keywords', []) or [intent.get('file_name', '')] or ['all files'],
                    "platform": intent.get('platform'),
                    "date_filter": intent.get('date_filter'),
                    "limit": intent.get('limit', 10),
                    "total_found": len(file_data),
                    "user_scoped": True,
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        except Exception as e:
            logger.error(f"Error formatting search results: {str(e)}")
            raise

    def _format_date(self, date_str: str) -> str:
        """Format date string for display."""
        if not date_str:
            return "Unknown date"
        try:
            date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return date_obj.strftime('%b %d, %Y')
        except:
            return date_str

    def _error_response(self, message: str) -> Dict[str, Any]:
        """Generate standardized error response."""
        return {
            "success": False,
            "action": "search",
            "file_data": {
                "files": [],
                "count": 0,
                "has_more": False
            },
            "metadata": {
                "error": message,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _search_all_files(self, user_id: str, limit: int) -> Dict[str, Any]:
        """Retrieve all files for a user"""
        try:
            query = "SELECT * FROM c WHERE c.user_id = @user_id ORDER BY c.uploaded_at DESC"
            parameters = [{"name": "@user_id", "value": user_id}]
            
            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            if limit and limit > 0:
                items = items[:limit]
            
            # Format the results properly
            mock_intent = {
                "keywords": ["all files"],
                "platform": None,
                "date_filter": None,
                "limit": limit
            }
            
            return self._format_search_results(items, mock_intent)
            
        except Exception as e:
            logger.error(f"Error retrieving all files: {str(e)}")
            return self._error_response(f"Failed to retrieve files: {str(e)}")

# Factory function for creating search handler instance
def create_search_handler() -> SearchHandler:
    """Factory function to create and return a SearchHandler instance."""
    return SearchHandler()

# Main handler function called by the intent dispatcher
def handle_search_intent(intent: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """
    Main entry point for search intent handling.
    
    Args:
        intent: Parsed intent dictionary
        user_id: User identifier
        
    Returns:
        Formatted search results with file data and metadata (no ChatGPT responses)
    """
    handler = create_search_handler()
    return handler.handle_search(intent, user_id)

# Test function with modified intent for the original query
if __name__ == "__main__":
    # Example usage for "Find my files from google drive last week"
    example_intent = {
        "action": "search",
        "platform": None,
        "file_name": None,
        "date_filter": None,
        "keywords": ['History of AI'],
        "limit": 10,
        "user_id": "sayyadshakil@gmail.com"
    }
    user_id = "sayyadshakil@gmail.com"
    result = handle_search_intent(example_intent, user_id)
    print(json.dumps(result, indent=2))