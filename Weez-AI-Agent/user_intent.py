"""
Weezy AI Agent - Intent Handler with Result Limits
Enhanced intent extraction to handle numeric constraints and result limits.
"""

import logging
from typing import List, Dict, Any
import json
import os
from openai import AzureOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentHandler:
    def __init__(self):
        """Initialize the intent handler with Azure OpenAI."""
        self.openai_client = None
        self._initialize_openai()
    
    def _initialize_openai(self):
        """Initialize Azure OpenAI client."""
        try:
            self.openai_client = AzureOpenAI(
                api_key=os.getenv('OPENAI_API_KEY'),
                api_version="2024-12-01-preview",
                azure_endpoint="https://weez-openai-resource.openai.azure.com/"
            )
            logger.info("Intent handler OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

    def extract_intents(self, user_query: str) -> List[Dict[str, Any]]:
        """
        Extract structured intents from user query using GPT-4o.
        
        Args:
            user_query: Natural language query from user
            
        Returns:
            List of intent dictionaries with action, platform, file_name, date_filter, keywords, and limit
        """
        try:
            # Enhanced system prompt to handle result limits
            system_prompt = """You are an intelligent file management assistant that extracts structured intents from natural language queries.

Your task is to analyze user queries and return a JSON array of actions. Each action should have these fields:
- "action": one of ["search", "delete", "upload", "summarize", "ask"]
- "platform": one of ["Drive", "OneDrive", "Slack", "Local"] or null if not specified
- "file_name": exact file name mentioned or null
- "date_filter": ISO date (YYYY-MM-DD) or null. Convert relative dates like "yesterday", "last week", "2 days ago" to actual dates based on today being 2025-05-29
- "keywords": array of important search terms, tags, or content descriptors
- "limit": number of results user wants to see (e.g., "first 3", "top 5", "show me 2") or null for default

IMPORTANT RULES:
1. If user asks for multiple actions (e.g., "show me files and summarize them"), create separate action objects
2. For relative dates:
   - "yesterday" = 2025-05-28
   - "last week"/"week ago" = 2025-05-22 (7 days ago)
   - "2 days ago" = 2025-05-27
   - "last month" = 2025-04-29
3. Extract numeric limits from phrases like:
   - "first 3 files" → limit: 3
   - "top 5 results" → limit: 5  
   - "show me 2" → limit: 2
   - "latest 10" → limit: 10
   - No number mentioned → limit: null (use default)
4. If no specific keywords are mentioned but action is clear, use context clues
5. For summarize actions, inherit search parameters from previous search intent

Examples:
Query: "show me the first 3 files about AI from last week"
Response: [{"action": "search", "platform": null, "file_name": null, "date_filter": "2025-05-22", "keywords": ["AI"], "limit": 3}]

Query: "find the top 5 presentations and summarize the first one"  
Response: [{"action": "search", "platform": null, "file_name": null, "date_filter": null, "keywords": ["presentations"], "limit": 5}, {"action": "summarize", "platform": null, "file_name": null, "date_filter": null, "keywords": ["presentations"], "limit": 1}]

Return only valid JSON array, no other text."""

            # Make API call to extract intents
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.1,  # Low temperature for consistent structured output
                max_tokens=1000
            )
            
            # Parse the JSON response
            intent_text = response.choices[0].message.content.strip()
            logger.info(f"Raw intent extraction response: {intent_text}")
            
            # Clean and parse JSON
            if intent_text.startswith('```json'):
                intent_text = intent_text.replace('```json', '').replace('```', '').strip()
            
            intents = json.loads(intent_text)
            
            # Validate and clean intents
            validated_intents = self._validate_intents(intents)
            
            logger.info(f"Extracted {len(validated_intents)} intents from query: '{user_query}'")
            return validated_intents
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse intent JSON: {str(e)}")
            return self._fallback_intent(user_query)
        except Exception as e:
            logger.error(f"Error in intent extraction: {str(e)}")
            return self._fallback_intent(user_query)
    
    def _validate_intents(self, intents: List[Dict]) -> List[Dict[str, Any]]:
        """Validate and clean extracted intents."""
        validated = []
        
        valid_actions = ["search", "delete", "upload", "summarize", "ask"]
        valid_platforms = ["Drive", "OneDrive", "Slack", "Local"]
        
        for intent in intents:
            try:
                # Validate required fields
                action = intent.get('action', '').lower()
                if action not in valid_actions:
                    logger.warning(f"Invalid action '{action}', defaulting to 'search'")
                    action = 'search'
                
                # Validate platform
                platform = intent.get('platform')
                if platform and platform not in valid_platforms:
                    logger.warning(f"Invalid platform '{platform}', setting to null")
                    platform = None
                
                # Validate limit (must be positive integer)
                limit = intent.get('limit')
                if limit is not None:
                    try:
                        limit = int(limit)
                        if limit <= 0:
                            limit = None
                        elif limit > 50:  # Cap at reasonable maximum
                            limit = 50
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid limit value '{limit}', setting to null")
                        limit = None
                
                # Ensure keywords is a list
                keywords = intent.get('keywords', [])
                if isinstance(keywords, str):
                    keywords = [keywords]
                elif not isinstance(keywords, list):
                    keywords = []
                
                validated_intent = {
                    'action': action,
                    'platform': platform,
                    'file_name': intent.get('file_name'),
                    'date_filter': intent.get('date_filter'),
                    'keywords': keywords,
                    'limit': limit
                }
                
                validated.append(validated_intent)
                
            except Exception as e:
                logger.error(f"Error validating intent {intent}: {str(e)}")
                continue
        
        return validated if validated else self._fallback_intent("")
    
    def _fallback_intent(self, query: str) -> List[Dict[str, Any]]:
        """Generate fallback intent when extraction fails."""
        # Simple keyword extraction as fallback
        keywords = [word.lower() for word in query.split() 
                   if len(word) > 2 and word.lower() not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'way', 'too']]
        
        # Extract simple numeric limits from fallback
        limit = None
        limit_keywords = ['first', 'top', 'latest', 'last']
        words = query.lower().split()
        
        for i, word in enumerate(words):
            if word in limit_keywords and i + 1 < len(words):
                try:
                    limit = int(words[i + 1])
                    break
                except ValueError:
                    continue
        
        return [{
            'action': 'search',
            'platform': None,
            'file_name': None,
            'date_filter': None,
            'keywords': keywords[:5],  # Limit to 5 keywords
            'limit': limit
        }]

# Main function called by the dispatcher
def handle_intent_extraction(user_query: str) -> List[Dict[str, Any]]:
    """
    Main entry point for intent extraction.
    
    Args:
        user_query: Natural language query from user
        
    Returns:
        List of structured intent dictionaries
    """
    handler = IntentHandler()
    return handler.extract_intents(user_query)

if __name__ == "__main__":
    # Example usage
    query = "i think you gave me the wrong file please search for file on AI and i think it was my local file"
    intents = handle_intent_extraction(query)
    print(json.dumps(intents, indent=2))