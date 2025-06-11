"""
Weezy AI Agent - RAG (Retrieval Augmented Generation) Component
Handles intelligent question answering using file contents with context retention.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from openai import AzureOpenAI
from handle_search import handle_search_intent

logger = logging.getLogger(__name__)

class RAGComponent:
    def __init__(self, cosmos_container, openai_client: AzureOpenAI):
        """
        Initialize RAG component with database and OpenAI client.
        
        Args:
            cosmos_container: Azure Cosmos DB container client
            openai_client: Azure OpenAI client instance
        """
        self.container = cosmos_container
        self.openai_client = openai_client
        self.max_context_length = 12000  # Maximum characters for context
        self.max_files_per_rag = 5  # Maximum number of files to use per RAG query
        
    def execute(self, parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Execute RAG operation - answer questions using file contents.
        
        Args:
            parameters: RAG parameters including:
                - question: The user's question
                - file_ids: Optional specific file IDs to search in
                - keywords: Optional keywords to find relevant files
                - conversation_context: Previous chat context for follow-ups
                - max_files: Maximum files to use (default: 5)
                
        Returns:
            Dict with RAG response and metadata
        """
        try:
            question = parameters.get('question', '')
            file_ids = parameters.get('file_ids', [])
            keywords = parameters.get('keywords', [])
            conversation_context = parameters.get('conversation_context', [])
            max_files = parameters.get('max_files', self.max_files_per_rag)
            
            if not question:
                return self._error_response("No question provided for RAG operation")
            
            logger.info(f"🤖 RAG Query: {question}")
            logger.info(f"📁 Specific files: {file_ids}")
            logger.info(f"🔍 Keywords: {keywords}")
            
            # Step 1: Retrieve relevant files
            relevant_files = self._retrieve_relevant_files(
                question, user_id, file_ids, keywords, max_files
            )
            
            if not relevant_files:
                return self._no_files_response(question)
            
            # Step 2: Extract and prepare file contents
            file_contexts = self._prepare_file_contexts(relevant_files)
            
            # Step 3: Build conversation context
            chat_context = self._build_chat_context(conversation_context)
            
            # Step 4: Generate RAG response
            rag_response = self._generate_rag_response(
                question, file_contexts, chat_context
            )
            
            # Step 5: Format and return response
            return self._format_rag_response(
                question, rag_response, relevant_files, file_contexts
            )
            
        except Exception as e:
            logger.error(f"RAG execution failed: {str(e)}")
            return self._error_response(f"RAG operation failed: {str(e)}")
    
    def _retrieve_relevant_files(self, question: str, user_id: str, 
                               file_ids: List[str], keywords: List[str], 
                               max_files: int) -> List[Dict[str, Any]]:
        """
        Retrieve files relevant to the question using various strategies.
        """
        try:
            relevant_files = []
            
            # Strategy 1: Use specific file IDs if provided
            if file_ids:
                logger.info("📋 Using specific file IDs for RAG")
                specific_files = self._get_files_by_ids(user_id, file_ids)
                relevant_files.extend(specific_files)
            
            # Strategy 2: Use provided keywords for search
            elif keywords:
                logger.info("🔍 Using provided keywords for file search")
                search_intent = {
                    "action": "search",
                    "keywords": keywords,
                    "limit": max_files,
                    "user_id": user_id
                }
                search_result = handle_search_intent(search_intent, user_id)
                if search_result.get('success') and search_result.get('file_data'):
                    relevant_files = search_result['file_data'].get('files', [])
            
            # Strategy 3: Extract keywords from question and search
            else:
                logger.info("🧠 Extracting keywords from question for search")
                extracted_keywords = self._extract_question_keywords(question)
                if extracted_keywords:
                    search_intent = {
                        "action": "search", 
                        "keywords": extracted_keywords,
                        "limit": max_files,
                        "user_id": user_id
                    }
                    search_result = handle_search_intent(search_intent, user_id)
                    if search_result.get('success') and search_result.get('file_data'):
                        relevant_files = search_result['file_data'].get('files', [])
            
            # Limit results
            relevant_files = relevant_files[:max_files]
            
            logger.info(f"📊 Retrieved {len(relevant_files)} relevant files for RAG")
            for file in relevant_files:
                logger.info(f"   📄 {file.get('file_name')} - {file.get('document_title', 'No title')}")
            
            return relevant_files
            
        except Exception as e:
            logger.error(f"Error retrieving relevant files: {str(e)}")
            return []
    
    def _get_files_by_ids(self, user_id: str, file_ids: List[str]) -> List[Dict[str, Any]]:
        """Get specific files by their IDs."""
        try:
            files = []
            for file_id in file_ids:
                try:
                    # Query for specific file with user validation
                    query = "SELECT * FROM c WHERE c.id = @file_id AND c.user_id = @user_id"
                    parameters = [
                        {"name": "@file_id", "value": file_id},
                        {"name": "@user_id", "value": user_id}
                    ]
                    
                    items = list(self.container.query_items(
                        query=query,
                        parameters=parameters,
                        enable_cross_partition_query=True
                    ))
                    
                    if items:
                        # Convert to expected format
                        file_item = items[0]
                        formatted_file = {
                            "id": file_item.get('id'),
                            "file_name": file_item.get('fileName'),
                            "document_title": file_item.get('document_title'),
                            "platform": file_item.get('platform'),
                            "date": file_item.get('uploaded_at'),
                            "text_summary": file_item.get('textSummary', ''),
                            "file_path": file_item.get('filePath'),
                            "raw_content": file_item.get('raw_content', ''),
                            "processed_content": file_item.get('processed_content', '')
                        }
                        files.append(formatted_file)
                        
                except Exception as e:
                    logger.warning(f"Could not retrieve file {file_id}: {str(e)}")
                    continue
            
            return files
            
        except Exception as e:
            logger.error(f"Error getting files by IDs: {str(e)}")
            return []
    
    def _extract_question_keywords(self, question: str) -> List[str]:
        """Extract relevant keywords from the user's question using GPT-4."""
        try:
            system_prompt = """Extract the most important keywords from the user's question that would help find relevant files.
            
Rules:
- Extract 3-5 key terms that are most likely to appear in relevant documents
- Focus on nouns, technical terms, and specific concepts
- Avoid common words like 'what', 'how', 'the', 'and', etc.
- Return only the keywords as a JSON array of strings

Examples:
Question: "What is machine learning?" → ["machine learning", "ML", "artificial intelligence"]
Question: "How does blockchain work?" → ["blockchain", "cryptocurrency", "distributed ledger"]
Question: "Tell me about financial reports" → ["financial", "reports", "finance", "accounting"]"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {question}"}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                keywords = json.loads(response_text)
                if isinstance(keywords, list):
                    logger.info(f"🎯 Extracted keywords: {keywords}")
                    return keywords
            except json.JSONDecodeError:
                # Fallback: extract words manually
                pass
            
            # Fallback: Simple keyword extraction
            import re
            words = re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())
            stop_words = {'what', 'how', 'when', 'where', 'why', 'who', 'the', 'and', 'or', 'but', 'for', 'with', 'about', 'tell', 'me'}
            keywords = [word for word in words if word not in stop_words][:5]
            
            logger.info(f"🎯 Fallback keywords: {keywords}")
            return keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            # Final fallback: use first few words
            words = question.split()[:3]
            return [word.lower().strip('?.,!') for word in words if len(word) > 2]
    
    def _prepare_file_contexts(self, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare file contents for RAG context."""
        file_contexts = []
        total_chars = 0
        
        for file in files:
            # Get the best available content
            content = self._get_best_file_content(file)
            
            if not content:
                continue
            
            # Truncate if needed to fit within context limits
            remaining_space = self.max_context_length - total_chars
            if remaining_space <= 0:
                break
                
            if len(content) > remaining_space:
                content = content[:remaining_space - 100] + "... [Content truncated]"
            
            file_context = {
                "file_name": file.get('file_name', 'Unknown'),
                "document_title": file.get('document_title', 'Untitled'),
                "platform": file.get('platform', 'Unknown'),
                "content": content,
                "content_length": len(content),
                "file_id": file.get('id')
            }
            
            file_contexts.append(file_context)
            total_chars += len(content)
            
            logger.info(f"📄 Added {file.get('file_name')} ({len(content)} chars)")
        
        logger.info(f"📊 Total context: {total_chars} characters across {len(file_contexts)} files")
        return file_contexts
    
    def _get_best_file_content(self, file: Dict[str, Any]) -> str:
        """Get the best available content from file (raw, processed, or summary)."""
        # Priority: processed_content > raw_content > text_summary
        content = file.get('processed_content', '')
        if content and len(content.strip()) > 50:
            return content
        
        content = file.get('raw_content', '')
        if content and len(content.strip()) > 50:
            return content
        
        content = file.get('text_summary', '')
        if content and len(content.strip()) > 20:
            return content
        
        return f"File: {file.get('file_name', 'Unknown')} - No content available"
    
    def _build_chat_context(self, conversation_context: List[Dict]) -> str:
        """Build chat context from previous conversation history."""
        if not conversation_context:
            return ""
        
        context_parts = []
        # Get last few exchanges for context
        recent_context = conversation_context[-6:]
        
        for entry in recent_context:
            if entry.get("type") == "thought":
                context_parts.append(f"AI Thought: {entry.get('content', '')}")
            elif entry.get("type") == "action" and entry.get("action") == "rag":
                # Include previous RAG questions and responses
                params = entry.get("parameters", {})
                result = entry.get("result", {})
                if params.get("question") and result.get("success"):
                    context_parts.append(f"Previous Q: {params['question']}")
                    if result.get("rag_response"):
                        response_preview = result["rag_response"][:200] + "..." if len(result["rag_response"]) > 200 else result["rag_response"]
                        context_parts.append(f"Previous A: {response_preview}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _generate_rag_response(self, question: str, file_contexts: List[Dict[str, Any]], 
                             chat_context: str) -> str:
        """Generate RAG response using GPT-4 with file contents."""
        try:
            # Build file content section
            files_section = ""
            for i, file_ctx in enumerate(file_contexts, 1):
                files_section += f"""
=== FILE {i}: {file_ctx['file_name']} ===
Title: {file_ctx['document_title']}
Platform: {file_ctx['platform']}
Content:
{file_ctx['content']}

"""
            
            # Build chat context section
            context_section = ""
            if chat_context:
                context_section = f"""
=== PREVIOUS CONVERSATION CONTEXT ===
{chat_context}

"""
            
            system_prompt = f"""You are Weez AI, an intelligent assistant that answers questions using the user's file contents.

Your task is to:
1. Answer the user's question using information from their files
2. Be accurate and cite specific files when referencing information
3. If the files don't contain enough information to answer fully, say so clearly
4. Provide helpful, detailed responses that demonstrate understanding of the content
5. Consider previous conversation context for follow-up questions

Guidelines:
- Always mention which files you're referencing: "According to [filename]..." or "In your [document title]..."
- If multiple files have relevant info, synthesize information across them
- Be conversational but informative
- If the question can't be answered from the files, explain what information is missing
- For follow-up questions, consider the previous conversation context

{context_section}=== USER'S FILES ===
{files_section}

Remember: Only use information from the provided files. If you need to make inferences, be clear about what's directly stated vs. what you're inferring."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.5,
                max_tokens=1500
            )
            
            rag_response = response.choices[0].message.content.strip()
            logger.info(f"✅ Generated RAG response ({len(rag_response)} chars)")
            
            return rag_response
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            return f"I apologize, but I encountered an error while analyzing your files to answer your question. Error: {str(e)}"
    
    def _format_rag_response(self, question: str, rag_response: str, 
                           relevant_files: List[Dict[str, Any]], 
                           file_contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format the final RAG response."""
        return {
            "success": True,
            "action": "rag",
            "question": question,
            "rag_response": rag_response,
            "source_files": [
                {
                    "id": file.get('id'),
                    "file_name": file.get('file_name'),
                    "document_title": file.get('document_title'),
                    "platform": file.get('platform'),
                    "content_used": len(next((ctx['content'] for ctx in file_contexts 
                                           if ctx['file_id'] == file.get('id')), ''))
                }
                for file in relevant_files
            ],
            "metadata": {
                "files_analyzed": len(relevant_files),
                "total_content_chars": sum(ctx['content_length'] for ctx in file_contexts),
                "response_length": len(rag_response),
                "timestamp": datetime.now().isoformat()
            },
            "message": f"Successfully answered your question using {len(relevant_files)} file(s)"
        }
    
    def _no_files_response(self, question: str) -> Dict[str, Any]:
        """Response when no relevant files are found."""
        return {
            "success": False,
            "action": "rag",
            "question": question,
            "rag_response": "I couldn't find any relevant files to answer your question. You might need to upload some documents first, or try rephrasing your question with different keywords.",
            "source_files": [],
            "metadata": {
                "files_analyzed": 0,
                "total_content_chars": 0,
                "response_length": 0,
                "timestamp": datetime.now().isoformat()
            },
            "message": "No relevant files found for RAG operation"
        }
    
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate standardized error response."""
        return {
            "success": False,
            "action": "rag",
            "question": "",
            "rag_response": error_message,
            "source_files": [],
            "metadata": {
                "error": error_message,
                "timestamp": datetime.now().isoformat()
            },
            "message": f"RAG operation failed: {error_message}"
        }


# Factory function for creating RAG component instance
def create_rag_component(cosmos_container, openai_client: AzureOpenAI) -> RAGComponent:
    """Factory function to create and return a RAGComponent instance."""
    return RAGComponent(cosmos_container, openai_client)