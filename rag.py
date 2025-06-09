# rag.py
from database import CosmosDB
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL, SEARCH_MODEL
import uuid
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAGAssistant")

class RAGAssistant:
    def __init__(self, user_id):
        logger.info(f"Initializing RAGAssistant for user {user_id}")
        self.user_id = user_id
        self.cosmos_db = CosmosDB()
        self.inventory_cache = []
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        

    def _create_item_content(self, item):
        """Create rich, searchable content for an inventory item with improved structure."""
        try:
            logger.info(f"Creating content for item: {item.get('Inventory Item Name', 'Unknown')}")
            
            # Ensure all fields have default values to prevent KeyErrors
            item_name = item.get('Inventory Item Name', 'Unknown Item')
            category = item.get('Category', 'unknown').lower()
            brand = item.get('Brand', '')
            full_name = item.get('Item Name', 'Unknown')
            case_price = item.get('Case Price', 0)
            unit_cost = item.get('Cost of a Unit', 0)
            priced_by = item.get('Priced By', 'unit').replace('per ', '')
            qty_in_case = item.get('Quantity In a Case', 0)
            measured_in = item.get('Measured In', 'units')
            total_units = item.get('Total Units', 0)
            item_number = item.get('Item Number', 'unknown')
            splitable = item.get('Splitable', 'NO')
            
            # Building a more structured content with clear sections
            sections = {
                "Product Overview": f"This is {item_name}, a {category} product{f' from {brand}' if brand else ''}. The full product name is {full_name}.",
                
                "Pricing Details": f"It costs ${case_price} per {priced_by}. Each unit costs ${unit_cost}.",
                
                "Quantity Information": f"Each case contains {qty_in_case} {measured_in}. Total available units are {total_units}.",
                
                "Specifications": f"The item number is {item_number}. {'This item cannot be split.' if splitable == 'NO' else 'This item can be split.'}"
            }
            
            # Add category-specific details
            if category.upper() == "DAIRY":
                sections["Storage Requirements"] = "This is a dairy product that should be stored refrigerated."
            elif category.upper() == "FROZEN":
                sections["Storage Requirements"] = "This is a frozen product that must be kept frozen."
            elif category.upper() == "PRODUCE":
                sections["Storage Requirements"] = "This is a fresh produce item with limited shelf life."
                
            # Assemble the final content with clear section formatting
            content = "\n\n".join([f"{key}:\n{value}" for key, value in sections.items()])
            
            logger.info(f"Created content successfully for {item_name}")
            return content
            
        except Exception as e:
            logger.error(f"Error creating content: {str(e)}")
            # Return a minimal content to avoid complete failure
            return f"Item: {item.get('Inventory Item Name', 'Unknown Item')}"


    async def initialize(self):
        """Initialize the RAG system with better error handling and logging."""
        try:
            # Get user inventory
            logger.info(f"Fetching inventory data for user {self.user_id}")
            inventory = await self.cosmos_db.get_user_documents(self.user_id)
            
            if not inventory:
                logger.error(f"No inventory found for user {self.user_id}")
                raise ValueError(f"No inventory found for user {self.user_id}")
            
            logger.info(f"Retrieved {len(inventory)} inventory documents")
            
            # Cache inventory data for later use
            self.inventory_cache = inventory[0].get('items', [])
            logger.info("Inventory cached for user")
            
            logger.info("Initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise

    async def query(self, user_question, top_k=5):
    
     try:
        has_action_intent, action_type, action_params = await self.check_for_action_intent(user_question)
        
        if has_action_intent:
            logger.info(f"Detected action intent: {action_type} with params: {action_params}")
            # Return a message indicating we detected an action intent
            # The actual action will be performed in the API layer
            return f"I detected that you want to {action_type}. I'll process this action for you."

        # Continue with the existing query logic for non-action questions...
        logger.info(f"Processing query: '{user_question}'")
        
        # Determine if this question might benefit from web search
        needs_web_search = self._should_use_web_search(user_question)
        logger.info(f"Query needs web search: {needs_web_search}")
        
        # Select appropriate model and options
        selected_model = SEARCH_MODEL if needs_web_search else OPENAI_MODEL
        logger.info(f"Selected model: {selected_model}")
        
        # Prepare inventory data for the prompt
        if not self.inventory_cache:
            docs = await self.cosmos_db.get_user_documents(self.user_id)
            if docs:
                self.inventory_cache = docs[0].get('items', [])

        formatted_results = self._format_search_results(self.inventory_cache[:top_k])
        
        # Construct a better prompt with clear sections
        prompt = self._construct_prompt(user_question, formatted_results, needs_web_search)
        
        # Base message structure
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful restaurant inventory assistant that provides accurate information about inventory items, prices, and quantities. Answer questions based only on the inventory data provided. If the data doesn't contain the information needed, acknowledge that limitation. Format your response in a clear, professional manner."
            },
            {"role": "user", "content": prompt}
        ]
        
        # Generate response with the selected model, using model-specific parameters
        logger.info(f"Generating response with model: {selected_model}")
        
        try:
            # First attempt with model-specific parameters
            if needs_web_search:
                # For search-enabled models, don't include temperature
                response = self.openai_client.chat.completions.create(
                    model=selected_model,
                    web_search_options={"search_context_size": "medium"},
                    messages=messages,
                    max_tokens=15000
                )
            else:
                # For fine-tuned models, include temperature
                response = self.openai_client.chat.completions.create(
                    model=selected_model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=15000
                )
                
            logger.info("Response generated successfully")
            return response.choices[0].message.content
            
        except Exception as api_error:
            logger.error(f"First attempt error: {str(api_error)}")
            
            # If server error with fine-tuned model, try fallback to base model
            if not needs_web_search and "500" in str(api_error):
                logger.info("Falling back to base GPT-4o model due to fine-tuned model error")
                try:
                    # Use base GPT-4o model as fallback
                    fallback_model = "gpt-4o"
                    response = self.openai_client.chat.completions.create(
                        model=fallback_model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=15000
                    )
                    
                    logger.info("Response generated successfully with fallback model")
                    return response.choices[0].message.content
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback model error: {str(fallback_error)}")
                    raise
            else:
                # If other type of error, or if web search model failed, re-raise
                raise
            
     except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        # Provide a graceful error message to the user
        return f"I encountered an issue while processing your question. Please try again or contact support if the problem persists. Error details: {str(e)}"
    def _should_use_web_search(self, question):
        """Determine if the question would benefit from web search."""
        # Define keywords that suggest external information might be needed
        web_search_keywords = [
            'market price', 'market trend', 'industry standard', 'compared to market',
            'current price', 'price trend', 'availability', 'shortage', 'surplus',
            'industry news', 'restaurant news', 'food trend', 'seasonal availability',
            'supply chain', 'latest', 'recent', 'today', 'this week', 'this month',
            'forecast', 'prediction', 'upcoming', 'expected', 'projected',
            'compare with', 'versus', 'vs'
        ]
        
        # Check if any keywords are present in the question
        question_lower = question.lower()
        for keyword in web_search_keywords:
            if keyword in question_lower:
                return True
        
        # Check for questions about external information that our inventory system wouldn't know
        external_info_patterns = [
            'what is the average', 'what are typical', 'how does this compare',
            'what should', 'what would', 'is this a good price', 'is this price fair',
            'what are other restaurants', 'what do other', 'is there a shortage',
            'when will', 'why is', 'how long will'
        ]
        
        for pattern in external_info_patterns:
            if pattern in question_lower:
                return True
        
        # Default to using the fine-tuned model
        return False

    def _construct_prompt(self, question, formatted_results, using_web_search=False):
        """Construct a clear prompt with explicit instructions, adapted for web search when needed."""
        base_prompt = f"""
I need information from my restaurant inventory to answer this question:

QUESTION:
{question}

RELEVANT INVENTORY DATA:
{formatted_results}
"""

        if using_web_search:
            base_prompt += """
Based on the inventory data above AND relevant web information, please provide a detailed answer.
If the web search provides helpful context about market prices, availability trends, or comparative data,
include that information clearly marked as external market data.
"""
        else:
            base_prompt += """
Based ONLY on the inventory data above, please provide a detailed answer to my question.
"""

        base_prompt += """
Format your response using HTML tags for better rendering in a chat interface.

For each inventory item, use this exact format:
<div style="margin-bottom: 20px; padding: 15px; background-color: #f8fafc; border-radius: 8px; border-left: 4px solid #3b82f6;">
  <h3 style="color: #334155; margin-top: 0; margin-bottom: 10px; font-size: 18px;">X. [Item Name]</h3>
  <ul style="list-style-type: none; padding-left: 10px; margin: 0;">
    <li style="margin-bottom: 8px;"><strong style="color: #475569;">Category:</strong> [Category]</li>
    <li style="margin-bottom: 8px;"><strong style="color: #475569;">Unit Cost:</strong> $[Cost]</li>
    <li style="margin-bottom: 8px;"><strong style="color: #475569;">Total Units Available:</strong> [Quantity]</li>
    <li style="margin-bottom: 8px;"><strong style="color: #475569;">Item Number:</strong> [Item Number]</li>
    <li style="margin-bottom: 0;"><strong style="color: #475569;">Notes:</strong> [Any additional notes]</li>
  </ul>
</div>

Then include a section called "Practical Insights:" with this format:
<div style="margin-top: 30px; background-color: #eff6ff; padding: 20px; border-radius: 8px;">
  <h3 style="color: #1e40af; margin-top: 0; margin-bottom: 15px; font-size: 20px;">Practical Insights:</h3>
  <div style="margin-bottom: 15px;">
    <p style="margin-top: 0; margin-bottom: 5px;"><strong style="color: #334155;">Key Insight 1:</strong> [Insight details]</p>
  </div>
  <div style="margin-bottom: 15px;">
    <p style="margin-top: 0; margin-bottom: 5px;"><strong style="color: #334155;">Key Insight 2:</strong> [Insight details]</p>
  </div>
  <!-- Additional insights as needed -->
</div>
"""

        if using_web_search:
            base_prompt += """
If external market data was used, include this additional section:
<div style="margin-top: 30px; background-color: #fff7ed; padding: 20px; border-radius: 8px; border: 1px solid #fdba74;">
  <h3 style="color: #c2410c; margin-top: 0; margin-bottom: 15px; font-size: 20px;">Market Context:</h3>
  <p style="margin-top: 0; margin-bottom: 10px;">
    [Include relevant market information obtained from web search here, with proper citations]
  </p>
</div>
"""

        base_prompt += """
Do not use Markdown formatting like ** or -. Use only HTML as specified above.
"""

        return base_prompt

    def _format_search_results(self, search_results):
        """Format search results in a clear, structured way for the prompt."""
        formatted_items = []
        
        for i, item in enumerate(search_results):
            # Extract key information, handling different field names
            inventory_item_name = item.get('inventory_item_name') or item.get('Inventory Item Name', 'Unknown')
            category = item.get('category') or item.get('Category', 'Unknown')
            cost = item.get('cost_of_unit') or item.get('Cost of a Unit', 0)
            total_units = item.get('total_units') or item.get('Total Units', 0)
            case_price = item.get('case_price') or item.get('Case Price', 0)
            
            # Format as structured data
            formatted_item = (
                f"Item {i+1}: {inventory_item_name}\n"
                f"  Category: {category}\n"
                f"  Unit Cost: ${cost}\n"
                f"  Total Units Available: {total_units}\n"
                f"  Case Price: ${case_price}\n"
                f"  Details: {item.get('content', '')}"
            )
            
            formatted_items.append(formatted_item)
        
        return "\n\n".join(formatted_items)

    async def index_user_documents(self):
        """Re-index user documents (for refreshing the index)."""
        try:
            logger.info(f"Re-indexing documents for user {self.user_id}")
            
            # Get updated inventory
            inventory = await self.cosmos_db.get_user_documents(self.user_id)
            
            if not inventory:
                logger.error(f"No inventory found for user {self.user_id}")
                raise ValueError(f"No inventory found for user {self.user_id}")
            
            # Refresh cached inventory data
            self.inventory_cache = inventory[0].get('items', [])
            logger.info("Inventory cache refreshed")
            
            logger.info("Re-indexing completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during re-indexing: {str(e)}")
            raise

    # Add this method to the RAGAssistant class in rag.py

    async def check_for_action_intent(self, user_question):

    # Use the LLM to detect if this is an action request
     prompt = f"""
You are analyzing a user request to determine if it's asking to perform an action on inventory data.
Please categorize this request and extract relevant parameters if it's an action request.

User request: "{user_question}"

Available actions:
1. update_price - Update an item's price (requires: item identifier and new price)
2. update_quantity - Update an item's quantity (requires: item identifier and new quantity)
3. add_item - Add a new inventory item (requires: item details)
4. delete_item - Delete an inventory item (requires: item identifier)
5. get_item_details - Get detailed information about an item (requires: item identifier)
6. search_by_category - Search for items by category (requires: category name)

If this appears to be an action request, identify the item's name or number, and extract specific values like prices or quantities.

Then respond with:
{{
  "is_action": true,
  "action_type": "action_name_from_list_above",
  "parameters": {{
    "item_identifier": "extracted item name or number",
    "price_type": "Cost of a Unit or Case Price",
    "new_price": "extracted price value, numbers only",
    "new_quantity": "extracted quantity value, numbers only",
    "category": "extracted category name"
  }}
}}

Only include parameters relevant to the detected action. For example, only include "new_price" for update_price actions.

If this is NOT an action request but rather a question about inventory, respond with:
{{
  "is_action": false
}}

Respond ONLY with the JSON structure described above, no other text.
"""
    
     messages = [
        {"role": "system", "content": "You are a helpful assistant that categorizes user requests and extracts parameters."},
        {"role": "user", "content": prompt}
    ]
    
     try:
        # If SEARCH_MODEL is available, use it for better action detection capability
        model_to_use = self.openai_model if hasattr(self, 'openai_model') else "gpt-4o"
        
        response = self.openai_client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            temperature=0,
            max_tokens=500
        )
        
        result_text = response.choices[0].message.content
        logger.debug(f"Action detection response: {result_text}")
        
        # Parse JSON response - handle potential formatting issues
        try:
            import json
            import re
            
            # Extract JSON if it's wrapped in backticks or other formatting
            json_match = re.search(r'({.*})', result_text.replace('\n', ' '))
            if json_match:
                result_json = json.loads(json_match.group(1))
            else:
                result_json = json.loads(result_text)
            
            if result_json.get("is_action", False):
                action_type = result_json.get("action_type")
                parameters = result_json.get("parameters", {})
                
                # Log the extracted parameters
                logger.info(f"Detected action intent: {action_type}")
                logger.info(f"Extracted parameters: {parameters}")
                
                # Standardize parameter names for compatibility
                if "item_identifier" in parameters and "item_number" not in parameters:
                    parameters["item_number"] = parameters["item_identifier"]
                
                if "new_price" in parameters:
                    try:
                        # Convert to float if possible
                        parameters["new_price"] = float(parameters["new_price"])
                    except (ValueError, TypeError):
                        # Keep as string if conversion fails
                        pass
                
                if "new_quantity" in parameters:
                    try:
                        # Convert to float if possible
                        parameters["new_quantity"] = float(parameters["new_quantity"])
                    except (ValueError, TypeError):
                        # Keep as string if conversion fails
                        pass
                
                return (
                    True,
                    action_type,
                    parameters
                )
            else:
                logger.info("No action intent detected")
                return (False, None, {})
                
        except Exception as e:
            logger.error(f"Error parsing action detection response: {str(e)}")
            logger.error(f"Raw response: {result_text}")
            import traceback
            logger.error(traceback.format_exc())
            return (False, None, {})
            
     except Exception as e:
        logger.error(f"Error in action detection: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return (False, None, {})