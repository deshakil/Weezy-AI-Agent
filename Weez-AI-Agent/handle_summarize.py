import json
import re
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from openai import AzureOpenAI
from azure.cosmos import CosmosClient

logger = logging.getLogger(__name__)

class SummarizeComponent:
    """
    A comprehensive document summarization component that handles context-aware 
    summarization with embedding-based content retrieval and multi-file analysis.
    """
    
    def __init__(self, container, openai_client: AzureOpenAI):
        """
        Initialize the SummarizeComponent.
        
        Args:
            container: Azure Cosmos DB container for file operations
            openai_client: Configured Azure OpenAI client
        """
        self.container = container
        self.openai_client = openai_client
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
    
    def execute(self, parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Main execution method for document summarization.
        
        Args:
            parameters: Dictionary containing summarization parameters
            user_id: User identifier
            
        Returns:
            Dict containing summarization results and metadata
        """
        try:
            # Extract parameters with defaults
            file_ids = parameters.get("file_ids", [])
            keywords = parameters.get("keywords", [])
            summary_type = parameters.get("summary_type", "general")
            max_files = parameters.get("max_files", 5)
            include_metadata = parameters.get("include_metadata", True)
            context_reference = parameters.get("context_reference", False)
            conversation_history = parameters.get("conversation_history", [])
            
            logger.info(f"📝 Summarizing for user {user_id} with params: {parameters}")
            
            # Determine files to summarize based on input criteria
            files_to_summarize = self._determine_files_to_summarize(
                file_ids, keywords, context_reference, conversation_history, 
                user_id, max_files
            )
            
            if not files_to_summarize:
                return self._create_error_response(
                    "No files found to summarize",
                    "I couldn't find any files matching your criteria to summarize. Try being more specific or check if you have uploaded files."
                )
            
            # Generate summaries for each file
            summaries, successful_count = self._generate_file_summaries(
                files_to_summarize[:max_files], summary_type, include_metadata
            )
            
            # Create multi-file analysis if applicable
            overall_summary = None
            cross_file_insights = None
            if len(summaries) > 1:
                overall_summary = self._create_multi_file_summary(summaries, summary_type)
                cross_file_insights = self._extract_cross_file_insights(summaries)
            
            # Create context for future reference
            summarization_context = self._create_summarization_context(
                summaries, summary_type
            )
            
            return {
                "success": True,
                "message": f"Successfully summarized {successful_count} out of {len(files_to_summarize)} files",
                "summary_type": summary_type,
                "file_count": len(summaries),
                "summaries": summaries,
                "overall_summary": overall_summary,
                "cross_file_insights": cross_file_insights,
                "context": summarization_context,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in summarize execution: {str(e)}")
            return self._create_error_response(str(e), f"Failed to generate summaries: {str(e)}")
    
    def _determine_files_to_summarize(self, file_ids: List[str], keywords: List[str], 
                                    context_reference: bool, conversation_history: List[Dict], 
                                    user_id: str, max_files: int) -> List[Dict[str, Any]]:
        """Determine which files to summarize based on input criteria."""
        # Handle contextual references
        if context_reference or self._is_contextual_request(keywords):
            return self._get_files_from_context(conversation_history, user_id)
        
        # Handle specific file IDs
        elif file_ids:
            return self._get_files_by_ids_with_embeddings(file_ids, user_id)
        
        # Handle keyword search
        elif keywords:
            search_result = self._search_files_for_summary(keywords, user_id, max_files)
            if search_result.get("success") and search_result.get("file_data"):
                return search_result["file_data"].get("files", [])
        
        # Default to recent files
        else:
            return self._get_recent_files_with_embeddings(user_id, limit=max_files)
        
        return []
    
    def _generate_file_summaries(self, files_to_summarize: List[Dict], 
                               summary_type: str, include_metadata: bool) -> tuple:
        """Generate summaries for all files and return summaries with success count."""
        summaries = []
        successful_summaries = 0
        
        for file_data in files_to_summarize:
            try:
                file_summary = self._generate_embedding_based_summary(
                    file_data, summary_type, include_metadata
                )
                if file_summary:
                    summaries.append(file_summary)
                    if file_summary.get("success"):
                        successful_summaries += 1
            except Exception as e:
                logger.error(f"Failed to summarize file {file_data.get('id', 'unknown')}: {str(e)}")
                summaries.append(self._create_file_error_summary(file_data, str(e)))
        
        return summaries, successful_summaries
    
    def _create_error_response(self, error: str, message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "success": False,
            "error": error,
            "message": message,
            "summaries": []
        }
    
    def _create_file_error_summary(self, file_data: Dict, error: str) -> Dict[str, Any]:
        """Create error summary for individual file."""
        return {
            "file_id": file_data.get("id"),
            "file_name": file_data.get("fileName", "Unknown"),
            "error": f"Failed to summarize: {error}",
            "success": False
        }
    
    def _create_summarization_context(self, summaries: List[Dict], summary_type: str) -> Dict[str, Any]:
        """Create context information for future reference."""
        return {
            "action": "summarize",
            "files_summarized": [
                {"id": s.get("file_id"), "name": s.get("file_name")} 
                for s in summaries if s.get("success")
            ],
            "summary_type": summary_type,
            "timestamp": datetime.now().isoformat()
        }
    
    def _is_contextual_request(self, keywords: List[str]) -> bool:
        """Check if the request contains contextual references."""
        if not keywords:
            return False
        
        contextual_terms = [
            "this", "that", "above", "previous", "last", "recent", 
            "earlier", "before", "mentioned", "discussed", "shown"
        ]
        
        keyword_text = " ".join(keywords).lower()
        return any(term in keyword_text for term in contextual_terms)
    
    def _get_files_from_context(self, conversation_history: List[Dict], user_id: str) -> List[Dict[str, Any]]:
        """Extract files from recent conversation context."""
        try:
            context_files = []
            
            # Look through recent conversation history for file references
            for entry in reversed(conversation_history[-10:]):
                if entry.get("type") == "action" and entry.get("action") == "search":
                    result = entry.get("result", {})
                    if result.get("success") and result.get("file_data"):
                        files = result["file_data"].get("files", [])
                        for file_ref in files[:3]:
                            full_file = self._get_file_with_embeddings(file_ref.get("id"), user_id)
                            if full_file:
                                context_files.append(full_file)
                        break
            
            return context_files[:5]
            
        except Exception as e:
            logger.error(f"Error getting files from context: {str(e)}")
            return []
    
    def _search_files_for_summary(self, keywords: List[str], user_id: str, max_files: int) -> Dict[str, Any]:
        """Search files for summarization using keywords."""
        try:
            # Try to use handle_search_intent if available
            try:
                from handle_search import handle_search_intent
                
                search_intent = {
                    "action": "search",
                    "keywords": keywords,
                    "limit": max_files,
                    "offset": 0,
                    "user_id": user_id
                }
                
                search_result = handle_search_intent(search_intent, user_id)
                
                # Enhance search results with full embedding data
                if search_result.get("success") and search_result.get("file_data"):
                    files = search_result["file_data"].get("files", [])
                    enhanced_files = []
                    
                    for file_ref in files:
                        full_file = self._get_file_with_embeddings(file_ref.get("id"), user_id)
                        if full_file:
                            enhanced_files.append(full_file)
                    
                    search_result["file_data"]["files"] = enhanced_files
                
                return search_result
                
            except ImportError:
                logger.warning("handle_search_intent not available, using fallback search")
                return self._fallback_search_for_summary(keywords, user_id, max_files)
                
        except Exception as e:
            logger.error(f"Error in search for summary: {str(e)}")
            return {"success": False, "file_data": {"files": []}}
    
    def _fallback_search_for_summary(self, keywords: List[str], user_id: str, max_files: int) -> Dict[str, Any]:
        """Fallback search when handle_search_intent is not available."""
        try:
            query_parts = ["SELECT * FROM c WHERE c.user_id = @user_id"]
            query_parameters = [{"name": "@user_id", "value": user_id}]
            
            if keywords:
                keyword_conditions = []
                for i, keyword in enumerate(keywords):
                    param_name = f"@keyword{i}"
                    keyword_conditions.extend([
                        f"CONTAINS(LOWER(c.fileName), {param_name})",
                        f"CONTAINS(LOWER(c.textSummary), {param_name})",
                        f"CONTAINS(LOWER(c.document_title), {param_name})"
                    ])
                    query_parameters.append({"name": param_name, "value": keyword.lower()})
                
                if keyword_conditions:
                    query_parts.append("AND (" + " OR ".join(keyword_conditions) + ")")
            
            query = " ".join(query_parts) + " ORDER BY c.uploaded_at DESC"
            
            items = list(self.container.query_items(
                query=query,
                parameters=query_parameters,
                enable_cross_partition_query=True
            ))
            
            return {
                "success": True,
                "file_data": {"files": items[:max_files]}
            }
            
        except Exception as e:
            logger.error(f"Fallback search failed: {str(e)}")
            return {"success": False, "file_data": {"files": []}}
    
    def _get_file_with_embeddings(self, file_id: str, user_id: str) -> Dict[str, Any]:
        """Get a single file with its embedding content."""
        try:
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
            
            return items[0] if items else None
            
        except Exception as e:
            logger.error(f"Error getting file with embeddings: {str(e)}")
            return None
    
    def _get_files_by_ids_with_embeddings(self, file_ids: List[str], user_id: str) -> List[Dict[str, Any]]:
        """Get specific files by their IDs with embedding content."""
        files = []
        for file_id in file_ids:
            file_data = self._get_file_with_embeddings(file_id, user_id)
            if file_data:
                files.append(file_data)
        return files
    
    def _get_recent_files_with_embeddings(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent files with embedding content."""
        try:
            query = "SELECT * FROM c WHERE c.user_id = @user_id ORDER BY c.uploaded_at DESC"
            parameters = [{"name": "@user_id", "value": user_id}]
            
            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            return items[:limit]
            
        except Exception as e:
            logger.error(f"Error getting recent files: {str(e)}")
            return []
    
    def _generate_embedding_based_summary(self, file_data: Dict[str, Any], 
                                        summary_type: str, include_metadata: bool) -> Dict[str, Any]:
        """Generate summary using embedding-stored content and AI analysis."""
        try:
            file_name = file_data.get("fileName", "Unknown File")
            file_type = file_data.get("fileType", "unknown")
            platform = file_data.get("platform", "unknown")
            upload_date = file_data.get("uploaded_at", "")
            
            # Get content from embeddings or stored text
            text_content = file_data.get("textContent", "")
            embedding_content = file_data.get("embeddingContent", "")
            content_to_summarize = embedding_content or text_content
            
            if not content_to_summarize:
                return self._create_no_content_summary(file_data, summary_type, include_metadata)
            
            # Generate AI summary
            ai_summary = self._generate_ai_summary(
                file_data, content_to_summarize, summary_type
            )
            
            # Extract additional insights
            key_topics = self._extract_key_topics_from_embeddings(content_to_summarize, ai_summary)
            sentiment = self._analyze_content_sentiment(ai_summary)
            key_entities = self._extract_key_entities(content_to_summarize)
            
            return {
                "file_id": file_data.get("id"),
                "file_name": file_name,
                "file_type": file_type,
                "platform": platform,
                "summary": ai_summary,
                "key_topics": key_topics,
                "key_entities": key_entities,
                "sentiment": sentiment,
                "summary_type": summary_type,
                "content_length": len(content_to_summarize),
                "word_count": len(content_to_summarize.split()) if content_to_summarize else 0,
                "success": True,
                "metadata": self._extract_file_metadata(file_data) if include_metadata else None,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating embedding-based summary: {str(e)}")
            return self._create_file_error_summary(file_data, str(e))
    
    def _create_no_content_summary(self, file_data: Dict, summary_type: str, include_metadata: bool) -> Dict[str, Any]:
        """Create summary response when no content is available."""
        return {
            "file_id": file_data.get("id"),
            "file_name": file_data.get("fileName", "Unknown File"),
            "file_type": file_data.get("fileType", "unknown"),
            "platform": file_data.get("platform", "unknown"),
            "summary": "No content available for summarization",
            "summary_type": summary_type,
            "success": True,
            "metadata": self._extract_file_metadata(file_data) if include_metadata else None
        }
    
    def _generate_ai_summary(self, file_data: Dict, content: str, summary_type: str) -> str:
        """Generate AI-powered summary of the content."""
        file_name = file_data.get("fileName", "Unknown File")
        file_type = file_data.get("fileType", "unknown")
        platform = file_data.get("platform", "unknown")
        upload_date = file_data.get("uploaded_at", "")
        
        summary_prompts = {
            "brief": "Provide a concise 1-2 sentence summary highlighting the most important points.",
            "general": "Provide a comprehensive summary covering main topics, key insights, and important details in 3-5 sentences.",
            "detailed": "Provide an in-depth summary including main topics, key arguments, important details, conclusions, actionable items, and any data points. Use structured format with bullet points if helpful."
        }
        
        prompt = f"""Analyze and summarize this document with deep understanding:

File Information:
- Name: {file_name}
- Type: {file_type}
- Platform: {platform}
- Upload Date: {self._format_date(upload_date)}

Document Content:
{content[:6000]}

Summary Requirements: {summary_prompts.get(summary_type, summary_prompts['general'])}

Please provide:
1. Main themes and topics
2. Key insights and findings
3. Important data points or statistics
4. Conclusions and recommendations
5. Any actionable items or next steps
6. Context and significance

Focus on extracting maximum value and understanding from the content."""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert document analyst specializing in creating insightful, comprehensive summaries. Extract key insights, identify patterns, and provide actionable intelligence from documents."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800 if summary_type == "detailed" else 400
        )
        
        return response.choices[0].message.content.strip()
    
    def _create_multi_file_summary(self, summaries: List[Dict], summary_type: str) -> Dict[str, Any]:
        """Create intelligent multi-file summary with cross-document analysis."""
        try:
            successful_summaries = [s for s in summaries if s.get("success", False)]
            
            if not successful_summaries:
                return None
            
            # Prepare data for analysis
            all_summaries = []
            all_topics = []
            platforms = set()
            file_types = set()
            
            for summary in successful_summaries:
                all_summaries.append(f"File: {summary['file_name']}\nSummary: {summary['summary']}")
                all_topics.extend(summary.get("key_topics", []))
                platforms.add(summary.get("platform", "unknown"))
                file_types.add(summary.get("file_type", "unknown"))
            
            combined_content = "\n\n".join(all_summaries)
            
            # Generate cross-file analysis
            analysis_prompt = f"""Analyze this collection of {len(successful_summaries)} document summaries to provide meta-insights:

{combined_content[:4000]}

Provide a comprehensive analysis that:
1. Identifies overarching themes and patterns
2. Highlights connections between documents
3. Extracts key insights that emerge from the collection
4. Notes any contradictions or complementary information
5. Provides strategic recommendations based on the entire set
6. Summarizes the collective value and purpose

Focus on synthesis rather than repetition."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert analyst specializing in synthesizing insights across multiple documents to identify patterns, connections, and strategic implications."
                    },
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Count topic frequency for common themes
            topic_counts = {}
            for topic in all_topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            common_topics = [topic for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True) if count > 1]
            
            return {
                "multi_file_analysis": response.choices[0].message.content.strip(),
                "file_count": len(successful_summaries),
                "common_themes": common_topics[:10],
                "platforms_analyzed": list(platforms),
                "file_types_analyzed": list(file_types),
                "total_content_analyzed": sum(s.get("word_count", 0) for s in successful_summaries),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating multi-file summary: {str(e)}")
            return None
    
    def _extract_cross_file_insights(self, summaries: List[Dict]) -> Dict[str, Any]:
        """Extract insights that span across multiple files."""
        try:
            successful_summaries = [s for s in summaries if s.get("success", False)]
            
            if len(successful_summaries) < 2:
                return None
            
            # Find common topics
            all_topics = []
            for summary in successful_summaries:
                all_topics.extend(summary.get("key_topics", []))
            
            topic_counts = {}
            for topic in all_topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            cross_cutting_topics = [topic for topic, count in topic_counts.items() if count >= 2]
            
            # Timeline analysis if dates available
            timeline_insights = self._analyze_document_timeline(successful_summaries)
            
            return {
                "cross_cutting_topics": cross_cutting_topics[:8],
                "document_relationships": self._identify_document_relationships(successful_summaries),
                "timeline_insights": timeline_insights,
                "content_diversity": {
                    "platforms": len(set(s.get("platform") for s in successful_summaries)),
                    "file_types": len(set(s.get("file_type") for s in successful_summaries)),
                    "total_words": sum(s.get("word_count", 0) for s in successful_summaries)
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting cross-file insights: {str(e)}")
            return None
    
    def _extract_key_topics_from_embeddings(self, content: str, summary: str) -> List[str]:
        """Enhanced topic extraction using embeddings context."""
        try:
            combined_text = f"{content} {summary}".lower()
            
            # Extract capitalized terms and proper nouns
            proper_nouns = re.findall(r'\b[A-Z][a-z]{2,}\b', content)
            quoted_terms = re.findall(r'"([^"]{3,30})"', content)
            key_phrases = re.findall(r'\b(?:key|main|important|critical|primary)\s+([^.,:;]{3,30})', combined_text, re.IGNORECASE)
            
            topics = set()
            
            # Add proper nouns as topics
            for noun in proper_nouns[:10]:
                if len(noun) > 3:
                    topics.add(noun)
            
            # Add quoted terms
            for term in quoted_terms[:5]:
                clean_term = re.sub(r'[^\w\s]', '', term).strip()
                if len(clean_term) > 3:
                    topics.add(clean_term.title())
            
            # Add key phrases
            for phrase in key_phrases[:5]:
                clean_phrase = re.sub(r'[^\w\s]', '', phrase[0] if isinstance(phrase, tuple) else phrase).strip()
                if len(clean_phrase) > 3:
                    topics.add(clean_phrase.title())
            
            return list(topics)[:10]
            
        except Exception as e:
            logger.error(f"Error extracting topics from embeddings: {str(e)}")
            return []
    
    def _analyze_content_sentiment(self, summary: str) -> Dict[str, Any]:
        """Simple sentiment analysis of the summary."""
        try:
            positive_words = ['good', 'excellent', 'positive', 'success', 'achievement', 'growth', 'improvement', 'opportunity']
            negative_words = ['bad', 'poor', 'negative', 'failure', 'problem', 'issue', 'decline', 'concern']
            neutral_words = ['analysis', 'review', 'report', 'study', 'data', 'information', 'overview']
            
            summary_lower = summary.lower()
            
            positive_count = sum(1 for word in positive_words if word in summary_lower)
            negative_count = sum(1 for word in negative_words if word in summary_lower)
            neutral_count = sum(1 for word in neutral_words if word in summary_lower)
            
            total_sentiment_words = positive_count + negative_count + neutral_count
            
            if total_sentiment_words == 0:
                return {"sentiment": "neutral", "confidence": 0.5}
            
            if positive_count > negative_count and positive_count > neutral_count:
                sentiment = "positive"
                confidence = positive_count / total_sentiment_words
            elif negative_count > positive_count and negative_count > neutral_count:
                sentiment = "negative"
                confidence = negative_count / total_sentiment_words
            else:
                sentiment = "neutral"
                confidence = max(neutral_count, positive_count, negative_count) / total_sentiment_words
            
            return {
                "sentiment": sentiment,
                "confidence": round(confidence, 2),
                "positive_indicators": positive_count,
                "negative_indicators": negative_count
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {"sentiment": "neutral", "confidence": 0.5}
    
    def _extract_key_entities(self, content: str) -> List[str]:
        """Extract key entities (names, organizations, locations) from content."""
        try:
            entities = set()
            
            # Extract potential names (capitalized words)
            names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', content)
            entities.update(names[:5])
            
            # Extract potential organizations
            orgs = re.findall(r'\b[A-Z][a-zA-Z\s]*(?:Corp|Inc|LLC|Ltd|Company|Organization|Agency)\b', content)
            entities.update(orgs[:3])
            
            # Extract potential locations
            locations = re.findall(r'(?:in|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', content)
            entities.update([loc for loc in locations[:3]])
            
            return list(entities)[:8]
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []
    
    def _identify_document_relationships(self, summaries: List[Dict]) -> List[Dict[str, Any]]:
        """Identify relationships between documents."""
        try:
            relationships = []
            
            for i, doc1 in enumerate(summaries):
                for j, doc2 in enumerate(summaries[i+1:], i+1):
                    # Find common topics
                    topics1 = set(doc1.get("key_topics", []))
                    topics2 = set(doc2.get("key_topics", []))
                    common_topics = topics1.intersection(topics2)
                    
                    if common_topics:
                        relationships.append({
                            "doc1": {"id": doc1["file_id"], "name": doc1["file_name"]},
                            "doc2": {"id": doc2["file_id"], "name": doc2["file_name"]},
                            "common_topics": list(common_topics),
                            "relationship_strength": len(common_topics) / max(len(topics1), len(topics2), 1)
                        })
            
            # Sort by relationship strength
            relationships.sort(key=lambda x: x["relationship_strength"], reverse=True)
            return relationships[:5]
            
        except Exception as e:
            logger.error(f"Error identifying relationships: {str(e)}")
            return []
    
    def _analyze_document_timeline(self, summaries: List[Dict]) -> Dict[str, Any]:
        """Analyze timeline patterns in documents."""
        try:
            # Extract dates from metadata
            dated_docs = []
            for summary in summaries:
                metadata = summary.get("metadata", {})
                upload_date = metadata.get("upload_date")
                if upload_date and upload_date != "Unknown date":
                    try:
                        date_obj = datetime.strptime(upload_date, '%b %d, %Y')
                        dated_docs.append({
                            "date": date_obj,
                            "file": summary["file_name"],
                            "topics": summary.get("key_topics", [])
                        })
                    except:
                        continue
            
            if len(dated_docs) < 2:
                return None
            
            # Sort by date
            dated_docs.sort(key=lambda x: x["date"])
            
            return {
                "earliest_doc": {"date": dated_docs[0]["date"].strftime('%b %d, %Y'), "file": dated_docs[0]["file"]},
                "latest_doc": {"date": dated_docs[-1]["date"].strftime('%b %d, %Y'), "file": dated_docs[-1]["file"]},
                "span_days": (dated_docs[-1]["date"] - dated_docs[0]["date"]).days,
                "chronological_pattern": "identified" if len(dated_docs) > 2 else "limited_data"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing timeline: {str(e)}")
            return None

    def _extract_file_metadata(self, file_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format comprehensive file metadata."""
        return {
        "upload_date": self._format_date(file_data.get("uploaded_at")),
        "file_size": file_data.get("fileSize"),
        "platform": file_data.get("platform"),
        "file_type": file_data.get("fileType"),
        "last_modified": self._format_date(file_data.get("lastModified")),
        "tags": file_data.get("tags", []),
        "word_count": len(file_data.get("textContent", "").split()) if file_data.get("textContent") else 0,
        "has_embeddings": bool(file_data.get("embeddingContent")),
        "content_length": len(file_data.get("embeddingContent", "") or file_data.get("textContent", "")),
        "similarity_score": file_data.get("similarity"),
        "document_title": file_data.get("document_title", file_data.get("fileName", ""))
    }

   # Make sure this block is at the TOP LEVEL of the file, NOT inside the class
if __name__ == "__main__":
    import os
    from azure.cosmos import CosmosClient
    from openai import AzureOpenAI
    
    # Initialize Azure Cosmos DB client
    cosmos_client = CosmosClient(
        url=os.getenv("COSMOS_ENDPOINT"),
        credential=os.getenv("COSMOS_KEY")
    )
    database = cosmos_client.get_database_client(os.getenv("COSMOS_DB_NAME", "weezyai"))
    container = database.get_container_client(os.getenv("COSMOS_CONTAINER_NAME", "files"))
    
    # Initialize Azure OpenAI client
    openai_client = AzureOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        azure_endpoint="https://weez-openai-resource.openai.azure.com/"
    )
    
    # Create SummarizeComponent instance
    summarizer = SummarizeComponent(container, openai_client)
    
    # Test user ID
    test_user_id = "sayyadshakil@gmail.com"
    
    # Example 1: Summarize specific files by ID
    print("=== Example 1: Summarize specific files ===")
    parameters_1 = {
        "file_ids": ["history_ai_notes.pdf"],
        "summary_type": "detailed",
        "include_metadata": True
    }
    result_1 = summarizer.execute(parameters_1, test_user_id)
    print(f"Result 1: {result_1['success']}")
    if result_1['success']:
        print(f"Summarized {result_1['file_count']} files")
        for summary in result_1['summaries']:
            if summary.get('success'):
                print(f"- {summary['file_name']}: {summary['summary'][:100]}...")
    
    # Example 2: Keyword-based summarization
    print("\n=== Example 2: Keyword-based summarization ===")
    parameters_2 = {
        "keywords": ["report", "analysis", "data"],
        "summary_type": "general",
        "max_files": 3,
        "include_metadata": False
    }
    result_2 = summarizer.execute(parameters_2, test_user_id)
    print(f"Result 2: {result_2['success']}")
    if result_2['success'] and result_2.get('overall_summary'):
        print(f"Multi-file analysis: {result_2['overall_summary']['multi_file_analysis'][:150]}...")
    
    # Example 3: Recent files summarization with context
    print("\n=== Example 3: Recent files with conversation context ===")
    mock_conversation = [
        {
            "type": "action",
            "action": "search",
            "result": {
                "success": True,
                "file_data": {
                    "files": [
                        {"id": "recent_file_1", "fileName": "quarterly_report.pdf"},
                        {"id": "recent_file_2", "fileName": "meeting_notes.docx"}
                    ]
                }
            }
        }
    ]
    parameters_3 = {
        "context_reference": True,
        "conversation_history": mock_conversation,
        "summary_type": "brief",
        "max_files": 2
    }
    result_3 = summarizer.execute(parameters_3, test_user_id)
    print(f"Result 3: {result_3['success']}")
    if result_3['success']:
        print(f"Context-based summaries: {len(result_3['summaries'])}")
    
    # Example 4: Default recent files summarization
    print("\n=== Example 4: Default recent files ===")
    parameters_4 = {
        "summary_type": "general",
        "include_metadata": True,
        "max_files": 5
    }
    result_4 = summarizer.execute(parameters_4, test_user_id)
    print(f"Result 4: {result_4['success']}")
    if result_4['success']:
        print(f"Recent files summary count: {result_4['file_count']}")
        if result_4.get('cross_file_insights'):
            insights = result_4['cross_file_insights']
            print(f"Cross-cutting topics: {insights.get('cross_cutting_topics', [])}")
    
    # Example 5: Error handling demonstration
    print("\n=== Example 5: Error handling ===")
    parameters_5 = {
        "file_ids": ["non_existent_file_id"],
        "summary_type": "detailed"
    }
    result_5 = summarizer.execute(parameters_5, test_user_id)
    print(f"Result 5: {result_5['success']}")
    if not result_5['success']:
        print(f"Error message: {result_5['message']}")
    
    print("\n=== Testing Complete ===")
    print("Make sure to set the following environment variables:")
    print("- COSMOS_DB_ENDPOINT")
    print("- COSMOS_DB_KEY") 
    print("- COSMOS_DB_NAME")
    print("- COSMOS_CONTAINER_NAME")
    print("- AZURE_OPENAI_API_KEY")
    print("- AZURE_OPENAI_API_VERSION")
    print("- AZURE_OPENAI_ENDPOINT")