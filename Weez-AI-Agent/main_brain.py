import json
import re
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from openai import AzureOpenAI
from handle_summarize import SummarizeComponent


logger = logging.getLogger(__name__)

class WeezReActBrain:
    def __init__(self):
        """Initialize the ReAct brain with OpenAI for autonomous thinking."""
        # Securely load credentials from environment variables
        self.openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        # Initialize Cosmos DB client
        self.cosmos_client = self._get_cosmos_client()
        self.database = self.cosmos_client.get_database_client('weezyai')
        self.container = self.database.get_container_client('files')
        self.summarize_component = SummarizeComponent(self.container, self.openai_client)
        self.max_iterations = 10  # Prevent infinite loops
        self.available_actions = [
            "search", "delete", "upload", "summarize", "rag", "complete"
        ]

    def process_user_request(self, user_message: str, user_id: str, conversation_context: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Main entry point - processes user request using autonomous ReAct framework.
        
        Args:
            user_message: What the user wants to do
            user_id: User identifier
            conversation_context: Previous conversation history for follow-ups
            
        Returns:
            Final response with all actions taken
        """
        logger.info(f"🧠 Processing request for user {user_id}: {user_message}")
        
        # Initialize or continue conversation history
        conversation_history = conversation_context or []
        iteration = 0
        final_user_response = ""
        
        # Keep thinking and acting until task is complete
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"🔄 Iteration {iteration}")
            
            # THINK: What should I do next?
            thought_response = self._think(user_message, conversation_history, user_id)
            current_thought = thought_response.get("thought", "")
            next_action = thought_response.get("action", "")
            action_params = thought_response.get("parameters", {})
            
            logger.info(f"💭 Thought: {current_thought}")
            logger.info(f"🎯 Action: {next_action}")
            
            # Add thought to history
            conversation_history.append({
                "type": "thought",
                "content": current_thought,
                "iteration": iteration
            })
            
            # Check if we're done
            if next_action == "complete":
                final_user_response = thought_response.get("final_response", "Task completed successfully!")
                logger.info("✅ Task completed!")
                break
            
            # ACT: Execute the chosen action
            action_result = self._act(next_action, action_params, user_id)
            
            logger.info(f"📊 Action result: {action_result.get('success', False)}")
            
            # OBSERVE: Record what happened
            conversation_history.append({
                "type": "action",
                "action": next_action,
                "parameters": action_params,
                "result": action_result,
                "iteration": iteration
            })
            
            # If action failed, let the brain know for next iteration
            if not action_result.get("success", False):
                conversation_history.append({
                    "type": "observation",
                    "content": f"Action {next_action} failed: {action_result.get('error', 'Unknown error')}",
                    "iteration": iteration
                })
            else:
                # Add successful observation
                conversation_history.append({
                    "type": "observation", 
                    "content": f"Action {next_action} succeeded: {action_result.get('message', 'Success')}",
                    "iteration": iteration
                })
        
        return {
            "success": True,
            "user_message": user_message,
            "final_response": final_user_response,
            "conversation_history": conversation_history,
            "iterations": iteration,
            "timestamp": datetime.now().isoformat()
        }

    def _think(self, user_message: str, history: List[Dict], user_id: str) -> Dict[str, Any]:
        """
        The THINKING step - autonomous decision making about what to do next.
        """
        # Create context from conversation history
        context = self._build_context(history)
        
        system_prompt = f"""You are Weez AI's autonomous thinking brain. You help users manage their files by reasoning step-by-step and making intelligent decisions about what actions to take.

Available Actions:
- search: Find files based on keywords, filename, platform, or date ranges
- delete: Remove files (always ask user confirmation first!)
- upload: Add new files to the system
- summarize: Read files and create intelligent summaries
- rag: Answer questions using file contents and context
- complete: Finish the task with a final response to the user

Your Reasoning Process:
1. ANALYZE the user's request and current context
2. DECIDE what the most logical next step should be
3. CHOOSE the appropriate action and parameters
4. If the task is complete, use "complete" with a helpful final response

Context Analysis Rules:
- If user says "show me more", "give me other files", "not satisfied" → they want additional/different results
- If user asks for "other three", "next batch", "remaining files" → use pagination with offset
- If previous search found N files but only showed X → get remaining with offset=X
- Always consider what the user is actually trying to accomplish
- Look for patterns in failed attempts to try different approaches

Decision Making Rules:
- Start with search if you need to find files first
- Always ask permission before deleting anything
- Break complex requests into logical steps
- Don't repeat the same failed action - try different parameters or approaches
- When task is truly complete, use "complete" with a natural, helpful response

Think step-by-step and be autonomous in your reasoning. Don't just follow templates - actually think about what the user needs.

Respond in JSON format:
{{
  "thought": "Your step-by-step reasoning about the situation and what to do",
  "action": "the_action_to_take",
  "parameters": {{"param1": "value1", "param2": "value2"}},
  "final_response": "only include this if action is 'complete' - your natural response to the user"
}}

User ID: {user_id}
Original Request: {user_message}

Current Context:
{context}

Think carefully about what the user needs and what the best next step is."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt}
                ],
                temperature=0.4,  # Slightly higher for more creative thinking
                max_tokens=800
            )
            
            # Parse JSON response
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response (handle cases where AI adds extra text)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_json = json.loads(json_match.group())
            else:
                response_json = json.loads(response_text)
            
            # Validate the response structure
            if "thought" not in response_json or "action" not in response_json:
                raise ValueError("Invalid response structure from thinking step")
                
            return response_json
            
        except Exception as e:
            logger.error(f"Error in thinking step: {str(e)}")
            # Fallback autonomous thinking
            return self._fallback_autonomous_thinking(user_message, history)

    def _fallback_autonomous_thinking(self, user_message: str, history: List[Dict]) -> Dict[str, Any]:
        """Fallback autonomous thinking when main thinking fails."""
        
        # Analyze recent actions to avoid repetition
        recent_actions = [entry.get("action") for entry in history[-3:] if entry.get("type") == "action"]
        
        # Simple autonomous decision logic
        if not history:
            # First interaction - likely need to search
            return {
                "thought": "This is the user's first request. I should search for what they're asking about to understand what files are available.",
                "action": "search",
                "parameters": {"keywords": user_message.split()[:3]}  # Use first 3 words as keywords
            }
        elif "search" not in recent_actions:
            # Haven't searched recently, probably need to search
            return {
                "thought": "I need to search for files to help with this request.",
                "action": "search", 
                "parameters": {"keywords": user_message.split()[:3]}
            }
        elif recent_actions.count("search") >= 2:
            # Searched multiple times, maybe complete the task
            return {
                "thought": "I've searched multiple times. Let me complete this request with what I've found.",
                "action": "complete",
                "final_response": "I've searched through your files based on your request. Let me know if you need anything else!"
            }
        else:
            # Default to search with different parameters
            return {
                "thought": "Let me try a different search approach to better help with this request.",
                "action": "search",
                "parameters": {"keywords": [user_message]}
            }

    def _act(self, action: str, parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        The ACTING step - executes the chosen action.
        """
        logger.info(f"🎬 Executing action: {action} with params: {parameters}")
        
        try:
            if action == "search":
                return self._execute_search(parameters, user_id)
            elif action == "delete":
                return self._execute_delete(parameters, user_id)
            elif action == "upload":
                return self._execute_upload(parameters, user_id)
            elif action == "summarize":
                return self._execute_summarize(parameters, user_id)
            elif action == "rag":
                return self._execute_rag(parameters, user_id)
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}",
                    "message": f"I don't know how to {action} yet."
                }
                
        except Exception as e:
            logger.error(f"Error executing action {action}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to execute {action}: {str(e)}"
            }

    def _execute_search(self, parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Execute search action using semantic search and Cosmos DB integration."""
        try:
            # Try to import your search handler
            try:
                from handle_search import handle_search_intent
    
                # Convert parameters to search intent format
                search_intent = {
                    "action": "search",
                    "platform": parameters.get("platform"),
                    "file_name": parameters.get("file_name"),
                    "date_filter": parameters.get("date_filter"),
                    "keywords": parameters.get("keywords", []),
                    "limit": parameters.get("limit", 10),
                    "offset": parameters.get("offset", 0),
                    "user_id": user_id
                }
                
                # Call your actual search handler
                result = handle_search_intent(search_intent, user_id)
                
                # Verify expected metadata fields exist
                if result.get('success') and 'file_data' in result:
                    for file in result['file_data'].get('files', []):
                        if 'document_title' not in file:
                            file['document_title'] = file.get('fileName', 'Untitled')
                        if 'textSummary' not in file:
                            file['textSummary'] = file.get('preview_summary', 'No summary available')
                            
                return result
                
            except ImportError:
                # Use fallback implementation
                return self._fallback_search_implementation(parameters, user_id)
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Search failed: {str(e)}",
                "file_data": {
                    "files": [],
                    "count": 0,
                    "has_more": False
                }
            }

   

    def _get_cosmos_client(self):
        """Get Cosmos DB client with configuration."""
        import os
        from azure.cosmos import CosmosClient
    
        cosmos_endpoint = os.getenv('COSMOS_ENDPOINT')
        cosmos_key = os.getenv('COSMOS_KEY')
    
        if not cosmos_endpoint or not cosmos_key:
            raise ValueError("Cosmos DB configuration not found. Set COSMOS_ENDPOINT and COSMOS_KEY environment variables.")
    
        return CosmosClient(cosmos_endpoint, cosmos_key)

    
    def _format_date(self, date_str: str) -> str:
        """Format date string for display."""
        if not date_str:
            return "Unknown date"
        try:
            date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return date_obj.strftime('%b %d, %Y')
        except:
            return date_str

    def _execute_delete(self, parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Execute delete action - requires user confirmation."""
        return {
            "success": False,
            "error": "Not implemented yet",
            "message": "Delete functionality coming soon! For safety, this will always require explicit user confirmation."
        }

    def _execute_upload(self, parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Execute upload action."""
        return {
            "success": False,
            "error": "Not implemented yet",
            "message": "Upload functionality coming soon!"
        }

    def _execute_summarize(self, parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Execute summarize action using the SummarizeComponent."""
        try:
            # Call the summarization component
            return self.summarize_component.execute(parameters, user_id)
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to generate summaries: {str(e)}"
            }

    def _execute_rag(self, parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
      """Execute RAG (Retrieval Augmented Generation) action using the RAG component."""
      try:
        # Import the RAG component
        from handle_ragg import create_rag_component
        
        # Create RAG component instance
        rag_component = create_rag_component(self.container, self.openai_client)
        
        # Execute RAG operation
        return rag_component.execute(parameters, user_id)
        
      except ImportError:
        logger.error("RAG component not available - handle_rag.py not found")
        return {
            "success": False,
            "error": "RAG component not available",
            "message": "RAG functionality is not yet configured. Please ensure handle_rag.py is available.",
            "rag_response": "I apologize, but I cannot answer questions using your files right now because the RAG component is not available."
        }
      except Exception as e:
        logger.error(f"RAG execution failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to execute RAG operation: {str(e)}",
            "rag_response": f"I encountered an error while trying to answer your question using your files: {str(e)}"
        }

    def _build_context(self, history: List[Dict]) -> str:
        """Build intelligent context string from conversation history."""
        if not history:
            return "This is the first interaction - no previous context."
    
        context_parts = []
        recent_entries = history[-6:]  # Last 6 entries for better context
        
        for entry in recent_entries:
            if entry["type"] == "thought":
                context_parts.append(f"💭 Thought {entry['iteration']}: {entry['content']}")
            elif entry["type"] == "action":
                success = entry['result'].get('success', False)
                action_name = entry['action']
                
                if action_name == "search" and success:
                    file_count = entry['result'].get('file_data', {}).get('count', 0)
                    context_parts.append(f"🔍 Search {entry['iteration']}: Found {file_count} files")
                else:
                    context_parts.append(f"⚡ Action {entry['iteration']}: {action_name} -> {'✅ Success' if success else '❌ Failed'}")
                    
            elif entry["type"] == "observation":
                context_parts.append(f"👁️ Observed {entry['iteration']}: {entry['content']}")
    
        return "\n".join(context_parts) if context_parts else "No significant context from previous actions."
    


def create_weez_brain() -> WeezReActBrain:
    """Create and return a Weez ReAct brain instance."""
    return WeezReActBrain()

# Main entry point
def process_user_request(user_message: str, user_id: str, conversation_context: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """
    Main entry point for processing user requests with ReAct framework.
    
    Args:
        user_message: What the user wants to do
        user_id: User identifier
        conversation_context: Previous conversation history for follow-ups
        
    Returns:
        Complete response with all actions taken
    """
    brain = create_weez_brain()
    return brain.process_user_request(user_message, user_id, conversation_context)

if __name__ == "__main__":
    # Test with a simple request
    test_request = "can you summarize the one file related to history of ai"
    test_user = "sayyadshakil@gmail.com"
    
    # Set environment variables for testing
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://weez-openai-resource.openai.azure.com/"
    os.environ["COSMOS_ENDPOINT"] = "https://weez-cosmos-db.documents.azure.com:443/"
    os.environ["COSMOS_KEY"] = os.getenv("COSMOS_KEY")
    
    result = process_user_request(test_request, test_user)
    print(json.dumps(result, indent=2))