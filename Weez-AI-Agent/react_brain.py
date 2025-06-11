"""
Weez AI Agent - ReAct Brain (Basic Version)
This is the thinking system that coordinates all actions for Weez AI.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import re
from openai import AzureOpenAI
import os

# Import your existing search handler
# from search_handler import handle_search_intent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeezReActBrain:
    def __init__(self):
        """Initialize the ReAct brain with OpenAI for thinking."""
        self.openai_client = AzureOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            api_version="2024-12-01-preview",
            azure_endpoint="https://weez-openai-resource.openai.azure.com/"
        )
        self.max_iterations = 10  # Prevent infinite loops
        self.available_actions = [
            "search", "delete", "upload", "summarize", "rag", "complete"
        ]
    
    def process_user_request(self, user_message: str, user_id: str, conversation_context: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Main entry point - processes user request using ReAct framework.
        
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
        current_thought = ""
        iteration = 0
        
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
            
            # If action failed, let the brain know
            if not action_result.get("success", False):
                conversation_history.append({
                    "type": "observation",
                    "content": f"Action {next_action} failed: {action_result.get('error', 'Unknown error')}",
                    "iteration": iteration
                })
        
        # Generate final response
        final_response = self._generate_final_response(user_message, conversation_history, user_id)
        
        return {
            "success": True,
            "user_message": user_message,
            "final_response": final_response,
            "conversation_history": conversation_history,
            "iterations": iteration,
            "timestamp": datetime.now().isoformat()
        }
    
    def _think(self, user_message: str, history: List[Dict], user_id: str) -> Dict[str, Any]:
        """
        The THINKING step - decides what to do next.
        """
        # Create context from conversation history
        context = self._build_context(history)
        
        system_prompt = f"""You are Weez AI's thinking brain. You help users manage their files by thinking step-by-step and choosing actions.

Available Actions:
- search: Find files based on keywords, filename, platform, or date
- delete: Remove files (ask user first!)
- upload: Add new files
- summarize: Read files and create summaries
- rag: Answer questions using file contents
- complete: Finish the task and respond to user

Your job is to:
1. Think about what the user wants
2. Choose the next best action
3. Provide parameters for that action

IMPORTANT - Handling Follow-ups:
- If user says "show me more", "give me other files", "not satisfied", they want additional results
- If user asks for "other three", "next batch", "remaining files", use offset/pagination
- If previous search found N files but only showed X, you can get the rest with offset=X
- Always check conversation context to understand what user is referring to

Rules:
- Always think step-by-step
- Pay attention to follow-up requests and context
- If you need to search first, do that
- If you need to delete files, ask user permission first
- Break complex requests into simple steps
- Use "complete" when the task is done

Respond in JSON format:
{{
  "thought": "What you're thinking about",
  "action": "next_action_to_take",
  "parameters": {{"param1": "value1", "param2": "value2"}}
}}

User ID: {user_id}
Current Request: {user_message}

Context from previous actions:
{context}

What should I do next?"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse JSON response
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response (in case there's extra text)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_json = json.loads(json_match.group())
            else:
                response_json = json.loads(response_text)
            
            return response_json
            
        except Exception as e:
            logger.error(f"Error in thinking step: {str(e)}")
            return {
                "thought": "I encountered an error while thinking. Let me try to search for what you asked.",
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
        """Execute search action using your existing search handler."""
        try:
            # Convert parameters to search intent format
            search_intent = {
                "action": "search",
                "platform": parameters.get("platform"),
                "file_name": parameters.get("file_name"),
                "date_filter": parameters.get("date_filter"),
                "keywords": parameters.get("keywords", []),
                "limit": parameters.get("limit", 10),
                "offset": parameters.get("offset", 0),  # For pagination
                "user_id": user_id
            }
            
            # TODO: Call your actual search handler here
            # result = handle_search_intent(search_intent, user_id)
            
            # For now, simulate a search result with pagination support
            offset = parameters.get("offset", 0)
            limit = parameters.get("limit", 3)
            
            # Simulate a database with 6 total files
            all_files = [
                {"id": "file-1", "file_name": "AI Research Paper 1.pdf", "platform": "google_drive", "date": "2025-05-28"},
                {"id": "file-2", "file_name": "Machine Learning Notes.docx", "platform": "dropbox", "date": "2025-05-27"},
                {"id": "file-3", "file_name": "Neural Networks Study.pdf", "platform": "google_drive", "date": "2025-05-26"},
                {"id": "file-4", "file_name": "AI Ethics Discussion.pdf", "platform": "google_drive", "date": "2025-05-25"},
                {"id": "file-5", "file_name": "Deep Learning Tutorial.docx", "platform": "dropbox", "date": "2025-05-24"},
                {"id": "file-6", "file_name": "AI History Timeline.pdf", "platform": "google_drive", "date": "2025-05-23"}
            ]
            
            # Apply pagination
            paginated_files = all_files[offset:offset + limit]
            
            mock_result = {
                "success": True,
                "action": "search",
                "user_response": f"Found {len(paginated_files)} files (showing {offset + 1}-{offset + len(paginated_files)} of {len(all_files)} total).",
                "file_data": {
                    "files": paginated_files,
                    "count": len(paginated_files),
                    "total_count": len(all_files),
                    "offset": offset,
                    "has_more": (offset + len(paginated_files)) < len(all_files)
                }
            }
            
            return mock_result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Search failed: {str(e)}"
            }
    
    def _execute_delete(self, parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Execute delete action (placeholder for now)."""
        return {
            "success": False,
            "error": "Not implemented yet",
            "message": "Delete functionality coming soon!"
        }
    
    def _execute_upload(self, parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Execute upload action (placeholder for now)."""
        return {
            "success": False,
            "error": "Not implemented yet",
            "message": "Upload functionality coming soon!"
        }
    
    def _execute_summarize(self, parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Execute summarize action (placeholder for now)."""
        return {
            "success": False,
            "error": "Not implemented yet",
            "message": "Summarize functionality coming soon!"
        }
    
    def _execute_rag(self, parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Execute RAG action (placeholder for now)."""
        return {
            "success": False,
            "error": "Not implemented yet",
            "message": "RAG functionality coming soon!"
        }
    
    def _build_context(self, history: List[Dict]) -> str:
        """Build context string from conversation history."""
        if not history:
            return "This is the first step."
        
        context_parts = []
        for entry in history[-5:]:  # Last 5 entries
            if entry["type"] == "thought":
                context_parts.append(f"Thought: {entry['content']}")
            elif entry["type"] == "action":
                context_parts.append(f"Action: {entry['action']} -> Success: {entry['result'].get('success', False)}")
            elif entry["type"] == "observation":
                context_parts.append(f"Observation: {entry['content']}")
        
        return "\n".join(context_parts)
    
    def _generate_final_response(self, user_message: str, history: List[Dict], user_id: str) -> str:
        """Generate final response to user based on all actions taken."""
        
        # Extract key information from history
        actions_taken = [entry for entry in history if entry["type"] == "action"]
        successful_actions = [action for action in actions_taken if action["result"].get("success", False)]
        
        if not successful_actions:
            return "I wasn't able to complete your request. Let me know if you'd like me to try a different approach!"
        
        # Create summary of what was accomplished
        summary_parts = []
        for action in successful_actions:
            action_name = action["action"]
            result = action["result"]
            
            if action_name == "search":
                file_count = result.get("file_data", {}).get("count", 0)
                summary_parts.append(f"Found {file_count} files")
            elif action_name == "delete":
                summary_parts.append("Deleted files")
            elif action_name == "summarize":
                summary_parts.append("Created summaries")
            # Add more action types as needed
        
        if summary_parts:
            return f"I've completed your request! Here's what I did: {', '.join(summary_parts)}. Is there anything else you'd like me to help you with?"
        else:
            return "I've processed your request. Let me know if you need anything else!"

# Factory function
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

# Test the brain
if __name__ == "__main__":
    # Test with a simple request
    test_request = "Find my files google drive"
    test_user = "sayyadshakil@gmail.com"
    
    result = process_user_request(test_request, test_user)
    print(json.dumps(result, indent=2))