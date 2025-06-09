import json
import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

import numpy as np
import faiss
from openai import OpenAI

from database import CosmosDB
from agent_tools import InventoryAgent
from config import (
    OPENAI_API_KEY, 
    OPENAI_MODEL, 
    OPENAI_EMBEDDING_MODEL,
    SEARCH_MODEL
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RAGAssistant")

class RAGAssistant:
    """
    AI-Powered Intent Detection RAG Assistant
    
    Uses your fine-tuned OpenAI model for intent understanding.
    Configured to use your specific models and settings.
    """

    def __init__(self, user_id: str, threshold: float = 0.70):
        self.user_id = user_id
        self.cosmos_db = CosmosDB()
        self.openai = OpenAI(api_key=OPENAI_API_KEY)
        self.agent = InventoryAgent(user_id)
        self.threshold = threshold

        # Use your configured models
        self.intent_model = OPENAI_MODEL  # Your fine-tuned model
        self.search_model = SEARCH_MODEL  # For web search capabilities
        self.embedding_model = OPENAI_EMBEDDING_MODEL

        # Inventory data
        self.inventory_cache: List[Dict[str, Any]] = []
        self.item_names: List[str] = []
        self.vectors: np.ndarray = None
        self.index = None

    def _clean_json_response(self, response_text: str) -> str:
        """Clean JSON response by removing markdown formatting"""
        if not response_text:
            return response_text
            
        # Remove leading/trailing whitespace
        cleaned = response_text.strip()
        
        # Remove markdown code block markers
        cleaned = re.sub(r'^```json\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^```\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        
        # Remove any other potential prefixes/suffixes
        cleaned = cleaned.strip()
        
        return cleaned

    def _safe_json_parse(self, response_text: str) -> Dict[str, Any]:
        """Safely parse JSON with fallback handling"""
        try:
            # First try direct parsing
            return json.loads(response_text)
        except json.JSONDecodeError:
            try:
                # Try cleaning markdown formatting
                cleaned = self._clean_json_response(response_text)
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON after cleaning: {e}")
                logger.error(f"Cleaned response: {cleaned}")
                logger.error(f"Original response: {response_text}")
                
                # Return a fallback response structure
                return {
                    "intent": "general",
                    "confidence": 0.3,
                    "entities": {},
                    "requires_search": False,
                    "reasoning": "JSON parsing failed, using fallback"
                }

    async def initialize(self) -> None:
        """Initialize with semantic embeddings using your embedding model"""
        try:
            docs = await self.cosmos_db.get_user_documents(self.user_id)
            if not docs:
                raise ValueError(f"No inventory found for user {self.user_id}")

            self.inventory_cache = docs[0].get('items', [])
            logger.info(f"Loaded {len(self.inventory_cache)} inventory items")

            # Create item embeddings using your embedding model
            self.item_names = [item.get('Inventory Item Name', '') for item in self.inventory_cache]
            
            if self.item_names:
                resp = self.openai.embeddings.create(
                    model=self.embedding_model,  # text-embedding-3-small
                    input=self.item_names
                )
                
                self.vectors = np.vstack([r.embedding for r in resp.data]).astype('float32')
                faiss.normalize_L2(self.vectors)

                # Build FAISS index
                dim = self.vectors.shape[1]
                self.index = faiss.IndexFlatIP(dim)
                self.index.add(self.vectors)
                
                logger.info("Built semantic search index with your embedding model")

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    async def detect_intent_and_extract(self, user_question: str) -> Dict[str, Any]:
        """Use your fine-tuned model to detect intent and extract parameters"""
        
        # Create inventory context for your fine-tuned model
        item_list = ", ".join(self.item_names[:20])  # First 20 items as context
        
        system_prompt = f"""You are a restaurant inventory management assistant using a fine-tuned model.

Available inventory items (sample): {item_list}

Analyze the user's query and respond with ONLY valid JSON containing:
1. intent: One of [greeting, farewell, help, show_all, show_category, search_item, count_summary, update_price, update_quantity, market_info, stock_status, general]
2. confidence: Float 0-1 indicating confidence in intent detection
3. entities: Dict containing extracted information like item_name, new_price, new_quantity, category
4. requires_search: Boolean if query needs web search
5. reasoning: Brief explanation of your analysis

IMPORTANT: Return ONLY the JSON object without any markdown formatting, code blocks, or additional text.

Examples:

{{"intent": "greeting", "confidence": 0.95, "entities": {{}}, "requires_search": false, "reasoning": "Simple greeting"}}

{{"intent": "update_price", "confidence": 0.90, "entities": {{"item_name": "mayonnaise", "new_price": 0.50, "price_type": "Cost of a Unit"}}, "requires_search": false, "reasoning": "Clear price update request for mayonnaise item"}}

{{"intent": "update_quantity", "confidence": 0.85, "entities": {{"item_name": "sanitizer", "new_quantity": 500}}, "requires_search": false, "reasoning": "Quantity update for sanitizer item"}}

{{"intent": "market_info", "confidence": 0.80, "entities": {{"item_name": "chicken", "query_type": "market_price"}}, "requires_search": true, "reasoning": "Asking for external market pricing information"}}"""

        try:
            # Use your fine-tuned model for intent detection
            response = self.openai.chat.completions.create(
                model=self.intent_model,  # ft:gpt-4o-2024-08-06:culvana::B4wUeDCH
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            logger.info(f"AI Intent Response: {result_text}")
            
            # Parse JSON response using safe parser
            intent_data = self._safe_json_parse(result_text)
            logger.info(f"Intent: {intent_data.get('intent')}, Confidence: {intent_data.get('confidence', 0)}, Entities: {intent_data.get('entities', {})}")
            return intent_data
                
        except Exception as e:
            logger.error(f"Fine-tuned model intent detection failed: {e}")
            return self._fallback_intent_detection(user_question)

    def _fallback_intent_detection(self, query: str) -> Dict[str, Any]:
        """Enhanced fallback intent detection"""
        query_lower = query.lower().strip()
        
        # Simple keyword-based fallback with better patterns
        if query_lower in ["hi", "hello", "hey", "heelo", "helo"]:
            return {"intent": "greeting", "confidence": 0.8, "entities": {}, "requires_search": False}
        
        if any(word in query_lower for word in ["bye", "goodbye", "thanks", "thank you"]):
            return {"intent": "farewell", "confidence": 0.8, "entities": {}, "requires_search": False}
        
        if "help" in query_lower:
            return {"intent": "help", "confidence": 0.9, "entities": {}, "requires_search": False}
        
        if any(phrase in query_lower for phrase in ["show all", "list all", "all items", "full inventory"]):
            return {"intent": "show_all", "confidence": 0.9, "entities": {}, "requires_search": False}
        
        if any(phrase in query_lower for phrase in ["how many", "total", "summary", "overview"]):
            return {"intent": "count_summary", "confidence": 0.8, "entities": {}, "requires_search": False}
        
        # Action detection with better patterns
        if any(word in query_lower for word in ["price", "cost"]) and any(word in query_lower for word in ["change", "update", "set", "to"]):
            return {"intent": "update_price", "confidence": 0.6, "entities": {}, "requires_search": False}
        
        if any(word in query_lower for word in ["units", "quantity"]) and any(word in query_lower for word in ["change", "update", "set", "to"]):
            return {"intent": "update_quantity", "confidence": 0.6, "entities": {}, "requires_search": False}
        
        if "market price" in query_lower:
            return {"intent": "market_info", "confidence": 0.7, "entities": {}, "requires_search": True}
        
        # Default to search_item for anything else
        return {"intent": "general", "confidence": 0.7, "entities": {}, "requires_search": True}

    def _find_best_matching_item(self, item_name: str) -> Optional[Dict[str, Any]]:
        """Find best matching item using your embedding model"""
        if not item_name or not self.index:
            return None
            
        try:
            # Use your embedding model for semantic search
            query_embedding = self.openai.embeddings.create(
                model=self.embedding_model,  # text-embedding-3-small
                input=[item_name]
            ).data[0].embedding
            
            query_vector = np.array(query_embedding, dtype='float32').reshape(1, -1)
            faiss.normalize_L2(query_vector)
            
            D, I = self.index.search(query_vector, 1)
            
            if D[0][0] >= 0.6:  # Minimum similarity threshold
                best_match = self.inventory_cache[I[0][0]]
                logger.info(f"Found match for '{item_name}': {best_match.get('Inventory Item Name')} (similarity: {D[0][0]:.3f})")
                return best_match
            else:
                logger.info(f"No good match found for '{item_name}' (best similarity: {D[0][0]:.3f})")
                
        except Exception as e:
            logger.error(f"Semantic matching failed: {e}")
        
        return None

    def _semantic_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Semantic search using your embedding model"""
        if not self.index:
            return []
            
        try:
            query_embedding = self.openai.embeddings.create(
                model=self.embedding_model,  # text-embedding-3-small
                input=[query]
            ).data[0].embedding
            
            query_vector = np.array(query_embedding, dtype='float32').reshape(1, -1)
            faiss.normalize_L2(query_vector)
            
            D, I = self.index.search(query_vector, min(top_k, len(self.inventory_cache)))
            
            hits = []
            for score, idx in zip(D[0], I[0]):
                if score >= self.threshold and idx < len(self.inventory_cache):
                    hits.append({
                        "item": self.inventory_cache[idx],
                        "score": float(score)
                    })
            
            return hits
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    async def _extract_price_update_params(self, query: str) -> Dict[str, Any]:
        """Extract price update parameters using your fine-tuned model"""
        system_prompt = f"""Extract price update parameters from the user query.
        
Available items: {', '.join(self.item_names[:20])}

Respond with ONLY valid JSON containing:
- item_name: The item to update (match to available items)
- new_price: The new price as a number

Do not include markdown formatting or code blocks.

Query: {query}"""

        try:
            response = self.openai.chat.completions.create(
                model=self.intent_model,  # Your fine-tuned model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Extract the parameters"}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            result = self._safe_json_parse(result_text)
            return result
            
        except Exception as e:
            logger.error(f"Parameter extraction failed: {e}")
            return {}

    async def _extract_quantity_update_params(self, query: str) -> Dict[str, Any]:
        """Extract quantity update parameters using your fine-tuned model"""
        system_prompt = f"""Extract quantity update parameters from the user query.
        
Available items: {', '.join(self.item_names[:20])}

Respond with ONLY valid JSON containing:
- item_name: The item to update (match to available items)  
- new_quantity: The new quantity as a number

Do not include markdown formatting or code blocks.

Query: {query}"""

        try:
            response = self.openai.chat.completions.create(
                model=self.intent_model,  # Your fine-tuned model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Extract the parameters"}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            result = self._safe_json_parse(result_text)
            return result
            
        except Exception as e:
            logger.error(f"Parameter extraction failed: {e}")
            return {}

    async def query(self, user_question: str) -> Dict[str, Any]:
        """Main query handler using AI intent detection"""
        try:
            logger.info(f"Processing query: {user_question}")
            
            # Step 1: Detect intent using AI
            intent_data = await self.detect_intent_and_extract(user_question)
            intent = intent_data.get("intent", "general")
            entities = intent_data.get("entities", {})
            confidence = intent_data.get("confidence", 0.5)
            requires_search = intent_data.get("requires_search", False)
            
            logger.info(f"Intent: {intent}, Confidence: {confidence}, Entities: {entities}")
            
            # Step 2: Route based on detected intent
            if intent == "greeting":
                return self._handle_greeting()
            
            elif intent == "farewell":
                return self._handle_farewell()
            
            elif intent == "help":
                return self._handle_help()
            
            elif intent == "show_all":
                return self._handle_show_all()
            
            elif intent == "count_summary":
                return self._handle_count_summary()
            
            elif intent == "update_price":
                return await self._handle_update_price(entities, user_question)
            
            elif intent == "update_quantity":
                return await self._handle_update_quantity(entities, user_question)
            
            elif intent == "search_item":
                return self._handle_search_item(entities, user_question)
            
            elif intent == "market_info":
                return await self._handle_market_info(entities, user_question)
            
            elif intent == "show_category":
                return self._handle_show_category(entities, user_question)
            
            else:  # general or unknown intent
                return self._handle_general(user_question)
                
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return self._error_response("I encountered an error processing your request. Please try again.")

    async def _handle_update_price(self, entities: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """Handle price update requests"""
        item_name = entities.get("item_name")
        new_price = entities.get("new_price")
        
        if not item_name or new_price is None:
            # Extract using AI if not found
            extract_result = await self._extract_price_update_params(original_query)
            item_name = extract_result.get("item_name")
            new_price = extract_result.get("new_price")
        
        if not item_name:
            return self._error_response("I couldn't identify which item you want to update. Please specify the item name.")
        
        if new_price is None:
            return self._error_response("I couldn't identify the new price. Please specify the price amount.")
        
        # Find matching item
        matched_item = self._find_best_matching_item(item_name)
        if not matched_item:
            return self._error_response(f"I couldn't find an item matching '{item_name}'. Please check the item name.")
        
        try:
            # Execute price update
            result = await self.agent.update_item_price(
                matched_item.get('Inventory Item Name'),
                float(new_price)
            )
            
            return {
                "type": "success",
                "message": f"""
                <div style="background-color: #f0fdf4; padding: 20px; border-radius: 12px; border-left: 4px solid #22c55e;">
                    <h3 style="color: #15803d; margin-top: 0; margin-bottom: 15px;">✅ Price Updated Successfully</h3>
                    <p style="color: #374151; margin: 0;">
                        Updated <strong>{matched_item.get('Inventory Item Name')}</strong> price to <strong>${new_price:.3f}</strong> per unit.
                    </p>
                </div>
                """
            }
            
        except Exception as e:
            logger.error(f"Price update failed: {e}")
            return self._error_response(f"Failed to update price for {matched_item.get('Inventory Item Name')}. Please try again.")

    async def _handle_update_quantity(self, entities: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """Handle quantity update requests"""
        item_name = entities.get("item_name")
        new_quantity = entities.get("new_quantity")
        
        if not item_name or new_quantity is None:
            # Extract using AI if not found
            extract_result = await self._extract_quantity_update_params(original_query)
            item_name = extract_result.get("item_name")
            new_quantity = extract_result.get("new_quantity")
        
        if not item_name:
            return self._error_response("I couldn't identify which item you want to update. Please specify the item name.")
        
        if new_quantity is None:
            return self._error_response("I couldn't identify the new quantity. Please specify the number of units.")
        
        # Find matching item
        matched_item = self._find_best_matching_item(item_name)
        if not matched_item:
            return self._error_response(f"I couldn't find an item matching '{item_name}'. Please check the item name.")
        
        try:
            # Execute quantity update
            result = await self.agent.update_item_quantity(
                matched_item.get('Inventory Item Name'),
                float(new_quantity)
            )
            
            return {
                "type": "success",
                "message": f"""
                <div style="background-color: #f0fdf4; padding: 20px; border-radius: 12px; border-left: 4px solid #22c55e;">
                    <h3 style="color: #15803d; margin-top: 0; margin-bottom: 15px;">✅ Quantity Updated Successfully</h3>
                    <p style="color: #374151; margin: 0;">
                        Updated <strong>{matched_item.get('Inventory Item Name')}</strong> quantity to <strong>{new_quantity:.0f}</strong> units.
                    </p>
                </div>
                """
            }
            
        except Exception as e:
            logger.error(f"Quantity update failed: {e}")
            return self._error_response(f"Failed to update quantity for {matched_item.get('Inventory Item Name')}. Please try again.")

    def _handle_greeting(self) -> Dict[str, Any]:
        """Handle greeting intents"""
        return {
            "type": "response",
            "message": """
            <div style="background-color: #f0f9ff; padding: 16px; border-radius: 8px; border-left: 4px solid #3b82f6;">
                <p style="margin: 0; color: #1e40af; font-size: 18px;">👋 Hello!</p>
                <p style="margin: 8px 0 0 0; color: #475569;">I'm your restaurant inventory assistant. I can help you find items, update prices and quantities, and search for market information. What would you like to know?</p>
            </div>
            """
        }

    def _handle_farewell(self) -> Dict[str, Any]:
        """Handle farewell intents"""
        return {
            "type": "response",
            "message": """
            <div style="background-color: #f0fdf4; padding: 16px; border-radius: 8px; border-left: 4px solid #22c55e;">
                <p style="margin: 0; color: #15803d;">👋 You're welcome! Feel free to ask anytime you need help with your inventory.</p>
            </div>
            """
        }

    def _handle_help(self) -> Dict[str, Any]:
        """Handle help requests"""
        return {
            "type": "response", 
            "message": """
            <div style="background-color: #f0f9ff; padding: 20px; border-radius: 12px; border-left: 4px solid #3b82f6;">
                <h3 style="color: #1e40af; margin-top: 0; margin-bottom: 15px;">🤖 How I Can Help</h3>
                <ul style="color: #475569; margin-left: 20px;">
                    <li><strong>Find items:</strong> "tell me about chicken" or "show mayo details"</li>
                    <li><strong>Update prices:</strong> "change mayo price to $0.50" or "set chicken cost to $0.25"</li>
                    <li><strong>Update quantities:</strong> "sanitizer to 500 units" or "change mayo to 400 units"</li>
                    <li><strong>View inventory:</strong> "show all items" or "list everything"</li>
                    <li><strong>Market info:</strong> "what's the market price of chicken?"</li>
                    <li><strong>Get summaries:</strong> "how many items do we have?"</li>
                </ul>
            </div>
            """
        }

    def _handle_search_item(self, entities: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Handle item search requests"""
        item_name = entities.get("item_name", "")
        
        if not item_name:
            # Fall back to semantic search on the full query
            hits = self._semantic_search(query, top_k=5)
        else:
            # Search specifically for the named item
            hits = self._semantic_search(item_name, top_k=5)
        
        if not hits:
            return self._error_response(f"I couldn't find any items matching your search. Try 'show all items' to see what's available.")
        
        if len(hits) == 1:
            return self._format_single_item(hits[0]["item"])
        else:
            return self._format_multiple_items(hits, f"Found {len(hits)} items:")

    def _handle_general(self, query: str) -> Dict[str, Any]:
        # First try semantic search for inventory-related queries
        hits = self._semantic_search(query, top_k=3)

        # If we found inventory matches, show them first, then also search web
        inventory_response = ""
        if hits:
            if len(hits) == 1:
                # Single inventory item found - show it and also search web
                item = hits[0]["item"]
                name = item.get('Inventory Item Name', 'Unknown')
                category = item.get('Category', 'Unknown')
                cost = item.get('Cost of a Unit', 0)
                units = item.get('Total Units', 0)
                total_value = cost * units

                inventory_response = f"""
                <div style="background-color: #f0f9ff; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #3b82f6;">
                    <h4 style="color: #1e40af; margin-top: 0; margin-bottom: 10px;">📦 From Your Inventory:</h4>
                    <p style="margin: 0;"><strong>{name}</strong> - ${cost:.3f}/unit ({units} units) - Category: {category}</p>
                </div>
                """
            else:
                # Multiple inventory items found
                inventory_response = f"""
                <div style="background-color: #f0f9ff; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #3b82f6;">
                    <h4 style="color: #1e40af; margin-top: 0; margin-bottom: 10px;">📦 From Your Inventory ({len(hits)} items):</h4>
                """
                for hit in hits[:3]:  # Show top 3
                    item = hit["item"]
                    name = item.get('Inventory Item Name', 'Unknown')
                    cost = item.get('Cost of a Unit', 0)
                    units = item.get('Total Units', 0)
                    inventory_response += f"<p style='margin: 2px 0;'><strong>{name}</strong> - ${cost:.3f}/unit ({units} units)</p>"
                inventory_response += "</div>"

        # Now ALWAYS search the web for additional information
        try:
            logger.info(f"Using search model for query: {query}")

            # Use search model to get comprehensive answer
            response = self.openai.chat.completions.create(
                model=self.search_model,  # gpt-4o-search-preview
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate, comprehensive answers to any question. Use web search capabilities to find current, relevant information."},
                    {"role": "user", "content": query}
                ]
                # No temperature, max_tokens, etc. - search model doesn't support them
            )

            search_answer = response.choices[0].message.content

            # Combine inventory results (if any) with search results
            if inventory_response:
                combined_message = f"""
                <div style="background-color: #f8fafc; padding: 20px; border-radius: 12px;">
                    {inventory_response}

                    <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; border-left: 4px solid #22c55e;">
                        <h4 style="color: #15803d; margin-top: 0; margin-bottom: 10px;">🌐 Additional Information:</h4>
                        <div style="color: #374151; line-height: 1.6;">
                            {search_answer}
                        </div>
                    </div>

                    <div style="background-color: #ecfdf5; padding: 12px; border-radius: 6px; margin-top: 15px;">
                        <p style="margin: 0; color: #065f46; font-size: 12px;">
                            💡 Information provided by AI search with web access
                        </p>
                    </div>
                </div>
                """
            else:
                # No inventory matches, just show search results
                combined_message = f"""
                <div style="background-color: #f0fdf4; padding: 20px; border-radius: 12px; border-left: 4px solid #22c55e;">
                    <h3 style="color: #15803d; margin-top: 0; margin-bottom: 15px;">🌐 Search Results</h3>
                    <div style="color: #374151; line-height: 1.6; background-color: #ffffff; padding: 15px; border-radius: 8px;">
                        {search_answer}
                    </div>

                    <div style="background-color: #ecfdf5; padding: 12px; border-radius: 6px; margin-top: 15px;">
                        <p style="margin: 0; color: #065f46; font-size: 12px;">
                            💡 Information provided by AI search with web access
                        </p>
                    </div>
                </div>
                """

            return {
                "type": "response",
                "message": combined_message
            }

        except Exception as e:
            logger.error(f"Search model failed for general query: {e}")

            # Fallback to regular model if search model fails
            try:
                logger.info("Falling back to regular model for general query")

                response = self.openai.chat.completions.create(
                    model=self.intent_model,  # Your fine-tuned model
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Provide the best answer you can based on your knowledge."},
                        {"role": "user", "content": query}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )

                fallback_answer = response.choices[0].message.content

                # Combine inventory (if any) with fallback answer
                if inventory_response:
                    combined_message = f"""
                    <div style="background-color: #f8fafc; padding: 20px; border-radius: 12px;">
                        {inventory_response}

                        <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; border-left: 4px solid #3b82f6;">
                            <h4 style="color: #1e40af; margin-top: 0; margin-bottom: 10px;">🤖 General Information:</h4>
                            <div style="color: #374151; line-height: 1.6;">
                                {fallback_answer}
                            </div>
                        </div>

                        <div style="background-color: #eff6ff; padding: 12px; border-radius: 6px; margin-top: 15px;">
                            <p style="margin: 0; color: #1e40af; font-size: 12px;">
                                💡 General knowledge provided by AI assistant
                            </p>
                        </div>
                    </div>
                    """
                else:
                    combined_message = f"""
                    <div style="background-color: #f0f9ff; padding: 20px; border-radius: 12px; border-left: 4px solid #3b82f6;">
                        <h3 style="color: #1e40af; margin-top: 0; margin-bottom: 15px;">🤖 General Information</h3>
                        <div style="color: #374151; line-height: 1.6; background-color: #ffffff; padding: 15px; border-radius: 8px;">
                            {fallback_answer}
                        </div>

                        <div style="background-color: #eff6ff; padding: 12px; border-radius: 6px; margin-top: 15px;">
                            <p style="margin: 0; color: #1e40af; font-size: 12px;">
                                💡 General knowledge provided by AI assistant
                            </p>
                        </div>
                    </div>
                    """

                return {
                    "type": "response",
                    "message": combined_message
                }

            except Exception as e2:
                logger.error(f"Both search and fallback models failed: {e2}")

                # Final fallback - show inventory only if available
                if inventory_response:
                    return {
                        "type": "response",
                        "message": f"""
                        <div style="background-color: #f8fafc; padding: 20px; border-radius: 12px;">
                            {inventory_response}

                            <div style="background-color: #fef3c7; padding: 15px; border-radius: 8px; border-left: 4px solid #f59e0b; margin-top: 15px;">
                                <p style="color: #92400e; margin: 0;">
                                    ⚠️ Unable to search for additional information right now, but I found some relevant items in your inventory above.
                                </p>
                            </div>
                        </div>
                        """
                    }
                else:
                    return self._error_response("I'm having trouble accessing search capabilities right now. Please try again or ask about your inventory items.")
    async def _handle_market_info(self, entities: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Handle market info requests with fallback strategy"""
        
        # Try search model first, then fallback to regular model
        try:
            logger.info("Attempting market info with search model...")
            return await self._handle_market_info_with_search(entities, query)
        except Exception as e:
            logger.warning(f"Search model failed ({e}), using fallback method...")
            return await self._handle_market_info_fallback(entities, query)

    async def _handle_market_info_with_search(self, entities: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Market info using search model (primary method)"""
        item_name = entities.get("item_name", "")
        
        # First show current inventory prices
        inventory_response = ""
        if item_name:
            hits = self._semantic_search(item_name, top_k=3)
            if hits:
                inventory_response = "<h4>Current Inventory Prices:</h4>"
                for hit in hits:
                    item = hit["item"]
                    name = item.get('Inventory Item Name', 'Unknown')
                    cost = item.get('Cost of a Unit', 0)
                    units = item.get('Total Units', 0)
                    inventory_response += f"<p><strong>{name}:</strong> ${cost:.3f}/unit ({units} units available)</p>"
        
        # Try to get market information using search model with corrected parameters
        try:
            market_prompt = f"What is the current market price for {item_name} in the restaurant industry? Provide current pricing trends and factors affecting the price."
            
            # Use search model with ONLY supported parameters
            response = self.openai.chat.completions.create(
                model=self.search_model,  # gpt-4o-search-preview
                messages=[
                    {"role": "system", "content": "You are a restaurant industry expert providing current market pricing information. Be specific about prices when available and mention market factors."},
                    {"role": "user", "content": market_prompt}
                ]
                # DO NOT include: temperature, max_tokens, frequency_penalty, presence_penalty, top_p, n
                # These parameters are not supported by gpt-4o-search-preview
            )
            
            market_info = response.choices[0].message.content
            
            return {
                "type": "response",
                "message": f"""
                <div style="background-color: #f0fdf4; padding: 20px; border-radius: 12px; border-left: 4px solid #22c55e;">
                    <h3 style="color: #15803d; margin-top: 0; margin-bottom: 15px;">📈 Market Information for {item_name.title()}</h3>
                    
                    {inventory_response}
                    
                    <h4 style="color: #15803d; margin-top: 20px; margin-bottom: 10px;">Current Market Analysis:</h4>
                    <div style="color: #374151; line-height: 1.6; background-color: #ffffff; padding: 15px; border-radius: 8px;">
                        {market_info}
                    </div>
                    
                    <div style="background-color: #ecfdf5; padding: 12px; border-radius: 6px; margin-top: 15px;">
                        <p style="margin: 0; color: #065f46; font-size: 12px;">
                            💡 Market data provided by AI search model with web access. For real-time pricing, consult industry databases or suppliers.
                        </p>
                    </div>
                </div>
                """
            }
            
        except Exception as e:
            logger.error(f"Search model API call failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            
            # Re-raise the exception so the main handler can catch it and use fallback
            raise e

    async def _handle_market_info_fallback(self, entities: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Fallback market info using regular GPT model when search model fails"""
        item_name = entities.get("item_name", "")
        
        # Show inventory prices first
        inventory_response = ""
        if item_name:
            hits = self._semantic_search(item_name, top_k=3)
            if hits:
                inventory_response = "<h4>Current Inventory Prices:</h4>"
                for hit in hits:
                    item = hit["item"]
                    name = item.get('Inventory Item Name', 'Unknown')
                    cost = item.get('Cost of a Unit', 0)
                    units = item.get('Total Units', 0)
                    inventory_response += f"<p><strong>{name}:</strong> ${cost:.3f}/unit ({units} units available)</p>"
        
        try:
            # Use regular GPT model for market analysis (supports all parameters)
            market_prompt = f"""Provide general market price information for {item_name} in the restaurant/foodservice industry. 
            Include typical price ranges, factors affecting pricing, and general market trends. 
            Note that this is general information and actual prices may vary by region and supplier."""
            
            response = self.openai.chat.completions.create(
                model=self.intent_model,  # Your fine-tuned model or regular GPT
                messages=[
                    {"role": "system", "content": "You are a restaurant industry consultant providing general market pricing guidance."},
                    {"role": "user", "content": market_prompt}
                ],
                temperature=0.3,
                max_tokens=400
            )
            
            market_info = response.choices[0].message.content
            
            return {
                "type": "response", 
                "message": f"""
                <div style="background-color: #f0f9ff; padding: 20px; border-radius: 12px; border-left: 4px solid #3b82f6;">
                    <h3 style="color: #1e40af; margin-top: 0; margin-bottom: 15px;">📊 Market Information for {item_name.title()}</h3>
                    
                    {inventory_response}
                    
                    <h4 style="color: #1e40af; margin-top: 20px; margin-bottom: 10px;">General Market Guidance:</h4>
                    <div style="color: #374151; line-height: 1.6; background-color: #ffffff; padding: 15px; border-radius: 8px;">
                        {market_info}
                    </div>
                    
                    <div style="background-color: #eff6ff; padding: 12px; border-radius: 6px; margin-top: 15px;">
                        <p style="margin: 0; color: #1e40af; font-size: 12px;">
                            💡 General market guidance provided. Search model unavailable - for current pricing, contact your suppliers or check industry price sheets.
                        </p>
                    </div>
                </div>
                """
            }
            
        except Exception as e:
            logger.error(f"Fallback market analysis also failed: {e}")
            
            # Final fallback with inventory info only
            if inventory_response:
                return {
                    "type": "response",
                    "message": f"""
                    <div style="background-color: #fef3c7; padding: 20px; border-radius: 12px; border-left: 4px solid #f59e0b;">
                        <h3 style="color: #92400e; margin-top: 0; margin-bottom: 15px;">📊 Current Inventory Prices</h3>
                        {inventory_response}
                        <div style="background-color: #fffbeb; padding: 12px; border-radius: 6px; margin-top: 15px;">
                            <p style="margin: 0; color: #92400e; font-size: 12px;">
                                ⚠️ Market data services unavailable. Here are your current inventory prices. For market rates, contact your suppliers directly.
                            </p>
                        </div>
                    </div>
                    """
                }
            else:
                return {
                    "type": "error",
                    "message": f"""
                    <div style="background-color: #fef2f2; padding: 20px; border-radius: 12px; border-left: 4px solid #ef4444;">
                        <h3 style="color: #dc2626; margin-top: 0; margin-bottom: 15px;">❌ Market Data Unavailable</h3>
                        <p style="color: #7f1d1d; margin-bottom: 10px;">
                            I couldn't retrieve market information for "{item_name}" right now.
                        </p>
                        <p style="color: #7f1d1d; margin: 0;">
                            Try searching for specific items in your inventory instead, or contact your suppliers for current market prices.
                        </p>
                    </div>
                    """
                }

    def _format_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Format single item response"""
        name = item.get('Inventory Item Name', 'Unknown')
        category = item.get('Category', 'Unknown')
        cost = item.get('Cost of a Unit', 0)
        units = item.get('Total Units', 0)
        total_value = cost * units
        
        return {
            "type": "response",
            "message": f"""
            <div style="background-color: #f0f9ff; padding: 20px; border-radius: 12px; border-left: 4px solid #3b82f6;">
                <h3 style="color: #1e40af; margin-top: 0; margin-bottom: 15px;">{name}</h3>
                <div style="background-color: #ffffff; padding: 15px; border-radius: 8px;">
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px;">
                        <div style="text-align: center;">
                            <div style="font-size: 18px; font-weight: bold; color: #059669;">${cost:.3f}</div>
                            <div style="color: #6b7280; font-size: 12px;">Per Unit</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 18px; font-weight: bold; color: #7c3aed;">{units}</div>
                            <div style="color: #6b7280; font-size: 12px;">Available</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 18px; font-weight: bold; color: #dc2626;">${total_value:.2f}</div>
                            <div style="color: #6b7280; font-size: 12px;">Total Value</div>
                        </div>
                    </div>
                    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e5e7eb;">
                        <span style="color: #374151;"><strong>Category:</strong> {category}</span>
                    </div>
                </div>
            </div>
            """
        }

    def _error_response(self, message: str) -> Dict[str, Any]:
        """Format error response"""
        return {
            "type": "error",
            "message": f"""
            <div style="background-color: #fef2f2; padding: 16px; border-radius: 8px; border-left: 4px solid #ef4444;">
                <p style="margin: 0; color: #dc2626;">{message}</p>
            </div>
            """
        }

    def _handle_show_all(self) -> Dict[str, Any]:
        """Handle show all inventory requests"""
        if not self.inventory_cache:
            return self._error_response("Your inventory appears to be empty.")
        
        # Group by category
        categories = {}
        for item in self.inventory_cache:
            category = item.get('Category', 'OTHER')
            if category not in categories:
                categories[category] = []
            categories[category].append(item)
        
        html_content = f"""
        <div style="background-color: #f8fafc; padding: 20px; border-radius: 12px; border-left: 4px solid #3b82f6;">
            <h3 style="color: #1e40af; margin-top: 0; margin-bottom: 20px;">📋 Complete Inventory ({len(self.inventory_cache)} items)</h3>
        """
        
        total_value = 0
        for category, items in sorted(categories.items()):
            sorted_items = sorted(items, key=lambda x: x.get('Inventory Item Name', ''))
            category_value = sum(item.get('Cost of a Unit', 0) * item.get('Total Units', 0) for item in sorted_items)
            total_value += category_value
            
            html_content += f"""
            <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <h4 style="color: #374151; margin: 0; text-transform: uppercase; font-size: 14px; font-weight: 600;">{category}</h4>
                    <span style="background-color: #e5e7eb; color: #374151; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: 500;">
                        {len(sorted_items)} items • ${category_value:.2f}
                    </span>
                </div>
            """
            
            for item in sorted_items:
                name = item.get('Inventory Item Name', 'Unknown')
                cost = item.get('Cost of a Unit', 0)
                units = item.get('Total Units', 0)
                
                html_content += f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #f3f4f6;">
                    <span style="color: #374151; font-weight: 500;">{name}</span>
                    <div style="text-align: right;">
                        <div style="color: #059669; font-weight: 600; font-size: 14px;">${cost:.3f}/unit</div>
                        <div style="color: #6b7280; font-size: 12px;">{units} units</div>
                    </div>
                </div>
                """
            
            html_content += "</div>"
        
        html_content += f"""
            <div style="background-color: #eff6ff; padding: 15px; border-radius: 8px; margin-top: 15px; text-align: center;">
                <div style="color: #1e40af; font-weight: 600; font-size: 18px;">Total Inventory Value: ${total_value:,.2f}</div>
            </div>
        </div>
        """
        
        return {"type": "response", "message": html_content}

    def _handle_count_summary(self) -> Dict[str, Any]:
        """Handle count and summary requests"""
        total_items = len(self.inventory_cache)
        total_units = sum(item.get('Total Units', 0) for item in self.inventory_cache)
        total_value = sum(item.get('Cost of a Unit', 0) * item.get('Total Units', 0) for item in self.inventory_cache)
        
        # Category breakdown
        categories = {}
        for item in self.inventory_cache:
            cat = item.get('Category', 'OTHER')
            categories[cat] = categories.get(cat, 0) + 1
        
        category_html = ""
        for cat, count in sorted(categories.items()):
            category_html += f"<span style='background-color: #e5e7eb; padding: 4px 8px; margin: 2px; border-radius: 4px; font-size: 12px;'>{cat}: {count}</span> "
        
        return {
            "type": "response",
            "message": f"""
            <div style="background-color: #f8fafc; padding: 20px; border-radius: 12px; border-left: 4px solid #6366f1;">
                <h3 style="color: #4338ca; margin-top: 0; margin-bottom: 20px;">📊 Inventory Summary</h3>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 20px;">
                    <div style="background-color: #fff; padding: 15px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 24px; font-weight: bold; color: #059669;">{total_items}</div>
                        <div style="color: #6b7280; font-size: 12px;">Total Items</div>
                    </div>
                    <div style="background-color: #fff; padding: 15px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 24px; font-weight: bold; color: #7c3aed;">{total_units:,}</div>
                        <div style="color: #6b7280; font-size: 12px;">Total Units</div>
                    </div>
                    <div style="background-color: #fff; padding: 15px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 24px; font-weight: bold; color: #dc2626;">${total_value:,.2f}</div>
                        <div style="color: #6b7280; font-size: 12px;">Total Value</div>
                    </div>
                </div>
                
                <div>
                    <h4 style="color: #374151; margin-bottom: 8px;">Categories:</h4>
                    <div>{category_html}</div>
                </div>
            </div>
            """
        }

    def _handle_show_category(self, entities: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Handle category-specific requests"""
        # Try to identify category from entities or query
        category_name = entities.get("category", "")
        
        if not category_name:
            # Look for category keywords in the query
            query_lower = query.lower()
            for item in self.inventory_cache:
                cat = item.get('Category', '')
                if cat.lower() in query_lower:
                    category_name = cat
                    break
        
        if category_name:
            items = [item for item in self.inventory_cache if item.get('Category', '').upper() == category_name.upper()]
            if items:
                return self._format_category_response(category_name, items)
        
        # Show available categories
        categories = set(item.get('Category', 'OTHER') for item in self.inventory_cache)
        category_list = ", ".join(sorted(categories))
        
        return {
            "type": "response",
            "message": f"""
            <div style="background-color: #fef3c7; padding: 20px; border-radius: 12px; border-left: 4px solid #f59e0b;">
                <h3 style="color: #92400e; margin-top: 0; margin-bottom: 15px;">📦 Categories Available</h3>
                <p style="color: #78350f; margin-bottom: 10px;">I couldn't find that specific category. Here are the available categories:</p>
                <p style="color: #374151; font-weight: 500;">{category_list}</p>
                <p style="color: #6b7280; font-size: 14px; margin-top: 10px;">Try asking: "show me dairy products" or "list meat items"</p>
            </div>
            """
        }

    def _format_multiple_items(self, hits: List[Dict[str, Any]], title: str) -> Dict[str, Any]:
        """Format response for multiple items"""
        html_content = f"""
        <div style="background-color: #f8fafc; padding: 20px; border-radius: 12px; border-left: 4px solid #3b82f6;">
            <h3 style="color: #1e40af; margin-top: 0; margin-bottom: 15px;">{title}</h3>
        """
        
        for hit in hits:
            item = hit["item"]
            score = hit.get("score", 0)
            name = item.get('Inventory Item Name', 'Unknown')
            category = item.get('Category', 'Unknown')
            cost = item.get('Cost of a Unit', 0)
            units = item.get('Total Units', 0)
            
            # Relevance indicator
            relevance_width = min(100, int(score * 100))
            
            html_content += f"""
            <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; margin-bottom: 12px; border: 1px solid #e5e7eb;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                    <div style="flex: 1;">
                        <h4 style="color: #374151; margin: 0 0 4px 0; font-size: 16px; font-weight: 600;">{name}</h4>
                        <span style="background-color: #f3f4f6; color: #6b7280; padding: 2px 6px; border-radius: 4px; font-size: 11px;">{category}</span>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: #059669; font-weight: 600; font-size: 14px;">${cost:.3f}/unit</div>
                        <div style="color: #6b7280; font-size: 12px;">{units} units</div>
                    </div>
                </div>
                <div style="background-color: #f1f5f9; height: 3px; border-radius: 2px; overflow: hidden;">
                    <div style="background-color: #3b82f6; height: 100%; width: {relevance_width}%; transition: width 0.3s;"></div>
                </div>
                <div style="color: #6b7280; font-size: 10px; margin-top: 2px;">Relevance: {score:.1%}</div>
            </div>
            """
        
        html_content += "</div>"
        return {"type": "response", "message": html_content}

    def _format_category_response(self, category: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format response for category-specific queries"""
        total_items = len(items)
        total_value = sum(item.get('Cost of a Unit', 0) * item.get('Total Units', 0) for item in items)
        total_units = sum(item.get('Total Units', 0) for item in items)
        
        html_content = f"""
        <div style="background-color: #f0fdf4; padding: 20px; border-radius: 12px; border-left: 4px solid #22c55e;">
            <h3 style="color: #15803d; margin-top: 0; margin-bottom: 15px;">📦 {category} Category</h3>
            
            <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px; text-align: center;">
                    <div>
                        <div style="font-size: 18px; font-weight: bold; color: #059669;">{total_items}</div>
                        <div style="color: #6b7280; font-size: 11px;">Items</div>
                    </div>
                    <div>
                        <div style="font-size: 18px; font-weight: bold; color: #7c3aed;">{total_units:,}</div>
                        <div style="color: #6b7280; font-size: 11px;">Total Units</div>
                    </div>
                    <div>
                        <div style="font-size: 18px; font-weight: bold; color: #dc2626;">${total_value:.2f}</div>
                        <div style="color: #6b7280; font-size: 11px;">Total Value</div>
                    </div>
                </div>
            </div>
        """
        
        # Sort items by value (highest first)
        sorted_items = sorted(items, key=lambda x: x.get('Cost of a Unit', 0) * x.get('Total Units', 0), reverse=True)
        
        for item in sorted_items:
            name = item.get('Inventory Item Name', 'Unknown')
            cost = item.get('Cost of a Unit', 0)
            units = item.get('Total Units', 0)
            item_value = cost * units
            
            html_content += f"""
            <div style="background-color: #ffffff; padding: 12px; border-radius: 6px; margin-bottom: 8px; border: 1px solid #e5e7eb;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #374151; font-weight: 500; flex: 1;">{name}</span>
                    <div style="display: flex; gap: 15px; align-items: center; text-align: right;">
                        <div>
                            <div style="color: #059669; font-weight: 600; font-size: 13px;">${cost:.3f}/unit</div>
                            <div style="color: #6b7280; font-size: 11px;">{units} units</div>
                        </div>
                        <div style="color: #374151; font-weight: 500; font-size: 13px; min-width: 60px;">
                            ${item_value:.2f}
                        </div>
                    </div>
                </div>
            </div>
            """
        
        html_content += "</div>"
        return {"type": "response", "message": html_content}

    async def index_user_documents(self) -> bool:
        """Re-index user documents"""
        try:
            await self.initialize()
            return True
        except Exception as e:
            logger.error(f"Re-indexing failed: {e}")
            return False