# main.py - Context-Aware Version (CORRECTED)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import RAGAssistant
from database import CosmosDB
from voice_assistant import VoiceAssistant  # Fixed typo: was voice_assitant
from unified_context_manager import UnifiedContextManager
import logging
import uvicorn
import time
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ContextAwareInventoryAPI")

app = FastAPI(
    title="Context-Aware Restaurant Inventory Assistant API",
    description="AI-powered restaurant inventory management with conversation memory",
    version="2.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
context_manager = UnifiedContextManager()
voice_assistant = VoiceAssistant()
cosmos_db = CosmosDB()

# Request models
class QueryRequest(BaseModel):
    user_id: str
    question: str
    input_method: Optional[str] = "text"

class UpdateInventoryRequest(BaseModel):
    user_id: str
    item_identifier: str
    field: str
    value: float
    price_type: Optional[str] = "Cost of a Unit"

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Context-Aware Restaurant Inventory Assistant API",
        "version": "2.1.0",
        "features": ["context_memory", "conversation_tracking", "voice_assistant", "text_chat"]
    }

@app.post("/query")
async def query_inventory(request: QueryRequest):
    """Main query endpoint with conversation context"""
    start_time = time.time()
    
    try:
        logger.info(f"Processing {request.input_method} query for user {request.user_id}: {request.question}")
        
        # Get user's inventory data for context
        inventory_data = await get_user_inventory(request.user_id)
        
        # STEP 1: Check if this is a context-dependent query first
        context_result = context_manager.process_context_query(
            request.user_id, 
            request.question, 
            inventory_data
        )
        
        if context_result:
            # Handle context-based actions
            if context_result.get("type") == "action":
                action_result = await execute_action_with_context(
                    request.user_id,
                    context_result["action_type"],
                    context_result["parameters"],
                    request.input_method
                )
                
                processing_time = time.time() - start_time
                logger.info(f"Context action completed in {processing_time:.3f}s")
                
                if request.input_method == "voice":
                    return {
                        "display_response": action_result,
                        "speech_response": context_result.get("speech_response", "Task completed."),
                        "input_method": "voice",
                        "processing_time": processing_time,
                        "context_used": True,
                        "context_explanation": context_result.get("context_explanation", "")
                    }
                else:
                    return {
                        "response": action_result,
                        "input_method": "text", 
                        "processing_time": processing_time,
                        "context_used": True,
                        "context_explanation": context_result.get("context_explanation", "")
                    }
            
            # Handle context-based responses
            elif context_result.get("type") == "response":
                processing_time = time.time() - start_time
                
                if request.input_method == "voice":
                    return {
                        "display_response": context_result.get("display_response", ""),
                        "speech_response": context_result.get("speech_response", ""),
                        "input_method": "voice",
                        "processing_time": processing_time,
                        "context_used": True
                    }
                else:
                    return {
                        "response": context_result.get("display_response", context_result.get("speech_response", "")),
                        "input_method": "text",
                        "processing_time": processing_time,
                        "context_used": True
                    }
            
            # Handle context-based errors
            elif context_result.get("type") == "error":
                processing_time = time.time() - start_time
                
                if request.input_method == "voice":
                    return {
                        "display_response": create_error_display(context_result.get("message", "")),
                        "speech_response": context_result.get("speech_response", ""),
                        "input_method": "voice",
                        "processing_time": processing_time,
                        "context_used": True
                    }
                else:
                    return {
                        "response": create_error_display(context_result.get("message", "")),
                        "input_method": "text",
                        "processing_time": processing_time,
                        "context_used": True
                    }
        
        # STEP 2: No context match, proceed with normal RAG processing
        if request.input_method == "voice":
            # Process through RAG first
            rag_assistant = RAGAssistant(request.user_id)
            await rag_assistant.initialize()
            rag_result = await rag_assistant.query(request.question)
            
            # Handle RAG results
            if isinstance(rag_result, dict) and rag_result.get("type") == "action":
                action_result = await execute_action_with_context(
                    request.user_id,
                    rag_result["action_type"],
                    rag_result["parameters"],
                    request.input_method
                )
                
                # Optimize for voice
                voice_optimized = voice_assistant.process_voice_query(request.question, {"response": action_result})
                
                processing_time = time.time() - start_time
                return {
                    "display_response": action_result,
                    "speech_response": voice_optimized.get("speech_response", "Task completed."),
                    "input_method": "voice",
                    "processing_time": processing_time,
                    "context_used": False
                }
            else:
                # Regular response - optimize for voice
                response_text = rag_result.get("message") if isinstance(rag_result, dict) else str(rag_result)
                voice_optimized = voice_assistant.process_voice_query(request.question, {"response": response_text})
                
                processing_time = time.time() - start_time
                return {
                    "display_response": response_text,
                    "speech_response": voice_optimized.get("speech_response", response_text),
                    "input_method": "voice",
                    "processing_time": processing_time,
                    "context_used": False
                }
        
        else:
            # Text processing
            rag_assistant = RAGAssistant(request.user_id)
            await rag_assistant.initialize()
            rag_result = await rag_assistant.query(request.question)
            
            if isinstance(rag_result, dict) and rag_result.get("type") == "action":
                action_result = await execute_action_with_context(
                    request.user_id,
                    rag_result["action_type"],
                    rag_result["parameters"],
                    request.input_method
                )
                
                processing_time = time.time() - start_time
                return {
                    "response": action_result,
                    "input_method": "text",
                    "processing_time": processing_time,
                    "context_used": False
                }
            else:
                response_text = rag_result.get("message") if isinstance(rag_result, dict) else str(rag_result)
                processing_time = time.time() - start_time
                return {
                    "response": response_text,
                    "input_method": "text",
                    "processing_time": processing_time,
                    "context_used": False
                }
                
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error in query endpoint ({processing_time:.3f}s): {str(e)}")
        
        error_message = "I encountered an error processing your request. Please try again."
        
        if request.input_method == "voice":
            return {
                "display_response": create_error_display(error_message),
                "speech_response": error_message,
                "input_method": "voice",
                "processing_time": processing_time,
                "error": True
            }
        else:
            return {
                "response": create_error_display(error_message),
                "input_method": "text",
                "processing_time": processing_time,
                "error": True
            }

async def execute_action_with_context(user_id: str, action_type: str, parameters: dict, input_method: str):
    """Execute action and record it in context"""
    try:
        # Extract information for context tracking
        item_identifier = parameters.get("item_identifier") or parameters.get("item_number")
        item_name = parameters.get("item_name", item_identifier)
        
        # Get current item data to record old values
        inventory_data = await get_user_inventory(user_id)
        current_item = find_item_in_inventory(item_identifier, inventory_data)
        
        if not current_item:
            # Record failed action
            context_manager.add_action(
                user_id=user_id,
                action_type=action_type,
                item_name=item_name or "unknown",
                item_identifier=item_identifier or "unknown",
                old_value=None,
                new_value=None,
                input_method=input_method,
                success=False
            )
            return create_error_display(f"Item '{item_identifier}' not found")
        
        # Execute different action types
        if action_type == "update_price":
            old_price = current_item.get('Cost of a Unit', 0)
            new_price = parameters.get("new_price")
            
            if new_price is None:
                return create_error_display("New price value is required")
            
            # Execute the update
            result = await update_item_price(cosmos_db, user_id, parameters)
            
            # Record in context - FIXED: Use correct variables and action type
            context_manager.add_action(
                user_id=user_id,
                action_type="update_price",  # Fixed: was "update_quantity"
                item_name=current_item.get('Inventory Item Name'),
                item_identifier=item_identifier,
                old_value=old_price,  # Fixed: was old_quantity
                new_value=new_price,  # Fixed: was new_quantity
                input_method=input_method,
                success=True,
                item_data=current_item
            )
            
            return result
            
        elif action_type == "update_quantity":
            old_quantity = current_item.get('Total Units', 0)
            new_quantity = parameters.get("new_quantity")
            
            if new_quantity is None:
                return create_error_display("New quantity value is required")
            
            # Execute the update
            result = await update_item_quantity(cosmos_db, user_id, parameters)
            
            # Record in context
            context_manager.add_action(
                user_id=user_id,
                action_type="update_quantity",
                item_name=current_item.get('Inventory Item Name'),
                item_identifier=item_identifier,
                old_value=old_quantity,
                new_value=new_quantity,
                input_method=input_method,
                success=True,
                item_data=current_item
            )
            
            return result
            
        elif action_type == "get_item_details":
            # Execute the action first
            result = await get_item_details(cosmos_db, user_id, parameters)
            
            # Record the lookup in context
            context_manager.add_action(
                user_id=user_id,
                action_type="check_item",  # Fixed: better action name
                item_name=current_item.get('Inventory Item Name'),
                item_identifier=item_identifier,
                old_value=None,
                new_value=None,
                input_method=input_method,
                success=True,
                item_data=current_item
            )
            
            return result
        
        else:
            # Record unknown action attempt
            context_manager.add_action(
                user_id=user_id,
                action_type=action_type,
                item_name=current_item.get('Inventory Item Name', 'unknown'),
                item_identifier=item_identifier,
                old_value=None,
                new_value=None,
                input_method=input_method,
                success=False
            )
            return create_error_display(f"Action '{action_type}' is not yet implemented")
            
    except Exception as e:
        logger.error(f"Error executing action with context: {e}")
        
        # Record failed action
        context_manager.add_action(
            user_id=user_id,
            action_type=action_type,
            item_name=item_name or "unknown",
            item_identifier=item_identifier or "unknown",
            old_value=None,
            new_value=None,
            input_method=input_method,
            success=False
        )
        
        return create_error_display(f"Failed to execute {action_type.replace('_', ' ')}: {str(e)}")

async def get_user_inventory(user_id: str) -> dict:
    """Get user inventory data"""
    try:
        inventory = await cosmos_db.get_user_documents(user_id)
        return inventory[0] if inventory else {}
    except Exception as e:
        logger.error(f"Error getting inventory for user {user_id}: {e}")
        return {}

def find_item_in_inventory(item_identifier: str, inventory_data: dict) -> dict:
    """Find item in inventory data"""
    if not inventory_data or 'items' not in inventory_data:
        return None
    
    if not item_identifier:
        return None
    
    item_name_lower = str(item_identifier).lower()
    
    # Try exact match first
    for item in inventory_data['items']:
        if item.get('Inventory Item Name', '').lower() == item_name_lower:
            return item
    
    # Try partial match
    for item in inventory_data['items']:
        item_full_name = item.get('Inventory Item Name', '').lower()
        if item_name_lower in item_full_name or item_full_name.startswith(item_name_lower):
            return item
    
    # Try item number match
    for item in inventory_data['items']:
        item_number = str(item.get('Item Number', ''))
        if item_number == str(item_identifier):
            return item
    
    return None

async def update_item_price(cosmos_db: CosmosDB, user_id: str, parameters: dict):
    """Update an item's price"""
    try:
        item_identifier = parameters.get("item_identifier") or parameters.get("item_number")
        new_price = parameters.get("new_price")
        price_type = parameters.get("price_type", "Cost of a Unit")
        
        if not item_identifier or new_price is None:
            return create_missing_params_message("price update", ["item name/number", "new price"])
        
        # Validate price
        try:
            new_price = float(new_price)
            if new_price < 0:
                return create_error_display("Price cannot be negative")
        except (ValueError, TypeError):
            return create_error_display("Invalid price format")
        
        # Get current inventory
        inventory = await cosmos_db.get_user_documents(user_id)
        if not inventory:
            return create_no_inventory_message()
        
        inventory_doc = inventory[0]
        items = inventory_doc.get('items', [])
        updated_item = None
        old_price = 0
        
        # Find and update the item
        for item in items:
            if is_item_match(item, item_identifier):
                old_price = item.get(price_type, 0)
                item[price_type] = new_price
                updated_item = item
                break
        
        if not updated_item:
            return create_item_not_found_message(item_identifier)
        
        # Update the document in Cosmos DB
        try:
            inventory_doc['items'] = items
            # You need to implement update_document method in CosmosDB class
            await cosmos_db.update_document(inventory_doc)
        except Exception as update_error:
            logger.error(f"Failed to update database: {update_error}")
            return create_error_display("Failed to save changes to database")
        
        return create_price_update_success_message(
            updated_item.get('Inventory Item Name'),
            price_type,
            old_price,
            new_price
        )
        
    except Exception as e:
        logger.error(f"Error updating price: {str(e)}")
        return create_action_error_message("price update", str(e))

async def update_item_quantity(cosmos_db: CosmosDB, user_id: str, parameters: dict):
    """Update an item's quantity"""
    try:
        item_identifier = parameters.get("item_identifier") or parameters.get("item_number")
        new_quantity = parameters.get("new_quantity")
        
        if not item_identifier or new_quantity is None:
            return create_missing_params_message("quantity update", ["item name/number", "new quantity"])
        
        # Validate quantity
        try:
            new_quantity = float(new_quantity)
            if new_quantity < 0:
                return create_error_display("Quantity cannot be negative")
        except (ValueError, TypeError):
            return create_error_display("Invalid quantity format")
        
        inventory = await cosmos_db.get_user_documents(user_id)
        if not inventory:
            return create_no_inventory_message()
        
        inventory_doc = inventory[0]
        items = inventory_doc.get('items', [])
        updated_item = None
        old_quantity = 0
        
        for item in items:
            if is_item_match(item, item_identifier):
                old_quantity = item.get('Total Units', 0)
                item['Total Units'] = new_quantity
                updated_item = item
                break
        
        if not updated_item:
            return create_item_not_found_message(item_identifier)
        
        # Update database
        try:
            inventory_doc['items'] = items
            await cosmos_db.update_document(inventory_doc)
        except Exception as update_error:
            logger.error(f"Failed to update database: {update_error}")
            return create_error_display("Failed to save changes to database")
        
        return create_quantity_update_success_message(
            updated_item.get('Inventory Item Name'),
            old_quantity,
            new_quantity
        )
        
    except Exception as e:
        logger.error(f"Error updating quantity: {str(e)}")
        return create_action_error_message("quantity update", str(e))

async def get_item_details(cosmos_db: CosmosDB, user_id: str, parameters: dict):
    """Get detailed information about a specific item"""
    try:
        item_identifier = parameters.get("item_identifier") or parameters.get("item_number")
        
        if not item_identifier:
            return create_missing_params_message("item details", ["item name/number"])
        
        inventory = await cosmos_db.get_user_documents(user_id)
        if not inventory:
            return create_no_inventory_message()
        
        items = inventory[0].get('items', [])
        
        for item in items:
            if is_item_match(item, item_identifier):
                return create_item_details_response(item)
        
        return create_item_not_found_message(item_identifier)
        
    except Exception as e:
        logger.error(f"Error getting item details: {str(e)}")
        return create_action_error_message("get item details", str(e))

def is_item_match(item, identifier):
    """Check if an item matches the given identifier"""
    if not item or not identifier:
        return False
        
    item_name = item.get('Inventory Item Name', '').lower()
    item_num = str(item.get('Item Number', ''))
    identifier_str = str(identifier).lower()
    
    return (identifier_str == item_name or
            identifier_str in item_name or
            identifier_str == item_num or
            item_name.startswith(identifier_str))

# Helper functions for responses
def create_error_display(message: str) -> str:
    """Create error display"""
    safe_message = str(message).replace('<', '&lt;').replace('>', '&gt;')
    return f"""
    <div style="background-color: #fef2f2; padding: 15px; border-radius: 8px; border-left: 4px solid #ef4444;">
        <p style="margin: 0; color: #dc2626;">❌ {safe_message}</p>
    </div>
    """

def create_missing_params_message(action, required_params):
    """Create message for missing parameters"""
    params_text = " and ".join(required_params)
    return f"""
    <div style="background-color: #fef3c7; padding: 20px; border-radius: 8px; border-left: 4px solid #f59e0b;">
        <h3 style="color: #92400e; margin-top: 0;">📝 Missing Information</h3>
        <p style="color: #78350f; margin-bottom: 15px;">
            To perform the {action}, I need both the {params_text}.
        </p>
        <p style="color: #78350f; margin-bottom: 0;">
            <strong>Example:</strong> "Update mayonnaise price to $0.15" or "Set gloves quantity to 500"
        </p>
    </div>
    """

def create_no_inventory_message():
    """Create message when no inventory is found"""
    return """
    <div style="background-color: #fef2f2; padding: 20px; border-radius: 8px; border-left: 4px solid #ef4444;">
        <h3 style="color: #dc2626; margin-top: 0;">📦 No Inventory Found</h3>
        <p style="color: #7f1d1d; margin-bottom: 0;">
            I couldn't find any inventory data for your account. Please make sure your inventory has been uploaded 
            or contact support if you believe this is an error.
        </p>
    </div>
    """

def create_item_not_found_message(item_identifier):
    """Create message when item is not found"""
    safe_identifier = str(item_identifier).replace('<', '&lt;').replace('>', '&gt;')
    return f"""
    <div style="background-color: #fef3c7; padding: 20px; border-radius: 8px; border-left: 4px solid #f59e0b;">
        <h3 style="color: #92400e; margin-top: 0;">🔍 Item Not Found</h3>
        <p style="color: #78350f; margin-bottom: 15px;">
            I couldn't find an item matching "<strong>{safe_identifier}</strong>" in your inventory.
        </p>
        <p style="color: #78350f; margin-bottom: 0;">
            <strong>Tip:</strong> Try using the exact item name or ask me "What items do we have?" to see your full inventory.
        </p>
    </div>
    """

def create_price_update_success_message(item_name, price_type, old_price, new_price):
    """Create success message for price updates"""
    safe_name = str(item_name).replace('<', '&lt;').replace('>', '&gt;')
    return f"""
    <div style="background-color: #f0fdf4; padding: 20px; border-radius: 8px; border-left: 4px solid #22c55e;">
        <h3 style="color: #15803d; margin-top: 0;">✅ Price Updated Successfully!</h3>
        <div style="background-color: #ffffff; padding: 15px; border-radius: 6px; margin-bottom: 15px;">
            <h4 style="color: #166534; margin-top: 0; margin-bottom: 10px;">{safe_name}</h4>
            <p style="margin: 5px 0; color: #374151;"><strong>Price Type:</strong> {price_type}</p>
            <p style="margin: 5px 0; color: #374151;"><strong>Old Price:</strong> ${old_price:.2f}</p>
            <p style="margin: 5px 0; color: #22c55e; font-weight: bold;"><strong>New Price:</strong> ${new_price:.2f}</p>
        </div>
        <p style="color: #15803d; margin-bottom: 0;">
            Your inventory has been updated and the changes are now active. 📊
        </p>
    </div>
    """

def create_quantity_update_success_message(item_name, old_quantity, new_quantity):
    """Create success message for quantity updates"""
    safe_name = str(item_name).replace('<', '&lt;').replace('>', '&gt;')
    return f"""
    <div style="background-color: #f0fdf4; padding: 20px; border-radius: 8px; border-left: 4px solid #22c55e;">
        <h3 style="color: #15803d; margin-top: 0;">✅ Quantity Updated Successfully!</h3>
        <div style="background-color: #ffffff; padding: 15px; border-radius: 6px; margin-bottom: 15px;">
            <h4 style="color: #166534; margin-top: 0; margin-bottom: 10px;">{safe_name}</h4>
            <p style="margin: 5px 0; color: #374151;"><strong>Previous Quantity:</strong> {old_quantity} units</p>
            <p style="margin: 5px 0; color: #22c55e; font-weight: bold;"><strong>New Quantity:</strong> {new_quantity} units</p>
        </div>
        <p style="color: #15803d; margin-bottom: 0;">
            Your inventory levels have been updated successfully. 📦
        </p>
    </div>
    """

def create_item_details_response(item):
    """Create detailed item information response"""
    item_name = str(item.get('Inventory Item Name', 'Unknown')).replace('<', '&lt;').replace('>', '&gt;')
    category = str(item.get('Category', 'Unknown')).replace('<', '&lt;').replace('>', '&gt;')
    unit_cost = item.get('Cost of a Unit', 0)
    total_units = item.get('Total Units', 0)
    item_number = item.get('Item Number', 'N/A')
    
    return f"""
    <div style="background-color: #f8fafc; padding: 20px; border-radius: 8px; border-left: 4px solid #3b82f6;">
        <h3 style="color: #1e40af; margin-top: 0; margin-bottom: 20px;">📋 {item_name}</h3>
        <div style="background-color: #ffffff; padding: 15px; border-radius: 8px;">
            <p style="margin: 8px 0; color: #475569;"><strong>Item Number:</strong> {item_number}</p>
            <p style="margin: 8px 0; color: #475569;"><strong>Category:</strong> {category}</p>
            <p style="margin: 8px 0; color: #475569;"><strong>Unit Cost:</strong> ${unit_cost:.3f}</p>
            <p style="margin: 8px 0; color: #475569;"><strong>Available Units:</strong> {total_units}</p>
        </div>
    </div>
    """

def create_action_error_message(action_type, error_details):
    """Create error message for failed actions"""
    safe_action = str(action_type).replace('_', ' ').replace('<', '&lt;').replace('>', '&gt;')
    return f"""
    <div style="background-color: #fef2f2; padding: 20px; border-radius: 8px; border-left: 4px solid #ef4444;">
        <h3 style="color: #dc2626; margin-top: 0;">❌ Action Failed</h3>
        <p style="color: #7f1d1d; margin-bottom: 15px;">
            I wasn't able to complete the {safe_action} due to a technical issue.
        </p>
        <p style="color: #7f1d1d; margin-bottom: 0;">
            Please try again in a moment, or ask me a question about your inventory instead.
        </p>
    </div>
    """

@app.get("/health")
async def health_check():
    """Health check with context info"""
    try:
        total_users = len(context_manager.user_contexts)
        return {
            "status": "healthy",
            "version": "2.1.0-context",
            "features": ["context_memory", "conversation_tracking"],
            "active_users": total_users,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": time.time()
        }

@app.post("/clear-context/{user_id}")
async def clear_user_context(user_id: str):
    """Clear conversation context for a specific user"""
    try:
        context_manager.clear_user_context(user_id)
        return {"message": f"Context cleared for user {user_id}", "success": True}
    except Exception as e:
        logger.error(f"Error clearing context for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear context: {str(e)}")

@app.get("/context/{user_id}")
async def get_user_context_summary(user_id: str):
    """Get context summary for debugging"""
    try:
        summary = context_manager.get_context_summary(user_id)
        recent_actions = context_manager.get_recent_actions(user_id, limit=5)
        
        return {
            "user_id": user_id,
            "summary": summary,
            "total_actions": len(context_manager.user_contexts.get(user_id, [])),
            "recent_actions": [
                {
                    "timestamp": action.timestamp.isoformat(),
                    "action_type": action.action_type,
                    "item_name": action.item_name,
                    "item_identifier": action.item_identifier,
                    "old_value": action.old_value,
                    "new_value": action.new_value,
                    "input_method": action.input_method,
                    "success": action.success
                }
                for action in recent_actions
            ]
        }
    except Exception as e:
        logger.error(f"Error getting context summary for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get context summary: {str(e)}")

@app.post("/manual-update")
async def manual_update_inventory(request: UpdateInventoryRequest):
    """Manual inventory update endpoint (for direct API calls)"""
    try:
        logger.info(f"Manual update request for user {request.user_id}: {request.item_identifier}")
        
        parameters = {
            "item_identifier": request.item_identifier,
            "item_name": request.item_identifier
        }
        
        if request.field.lower() in ["price", "cost", "unit_cost", "cost_of_a_unit"]:
            parameters["new_price"] = request.value
            parameters["price_type"] = request.price_type
            action_type = "update_price"
        elif request.field.lower() in ["quantity", "units", "total_units", "stock"]:
            parameters["new_quantity"] = request.value
            action_type = "update_quantity"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported field: {request.field}")
        
        result = await execute_action_with_context(
            request.user_id,
            action_type,
            parameters,
            "api"
        )
        
        return {
            "success": True,
            "result": result,
            "user_id": request.user_id,
            "action_type": action_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in manual update: {e}")
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")

@app.get("/inventory/{user_id}")
async def get_inventory_summary(user_id: str):
    """Get inventory summary for a user"""
    try:
        inventory_data = await get_user_inventory(user_id)
        
        if not inventory_data or 'items' not in inventory_data:
            return {
                "user_id": user_id,
                "total_items": 0,
                "items": [],
                "message": "No inventory found"
            }
        
        items = inventory_data['items']
        summary_items = []
        
        for item in items[:50]:  # Limit to first 50 items for performance
            summary_items.append({
                "name": item.get('Inventory Item Name', 'Unknown'),
                "item_number": item.get('Item Number', 'N/A'),
                "category": item.get('Category', 'Unknown'),
                "unit_cost": item.get('Cost of a Unit', 0),
                "total_units": item.get('Total Units', 0)
            })
        
        return {
            "user_id": user_id,
            "total_items": len(items),
            "showing_items": len(summary_items),
            "items": summary_items
        }
        
    except Exception as e:
        logger.error(f"Error getting inventory summary for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get inventory: {str(e)}")

# Additional utility endpoints
@app.get("/stats")
async def get_api_stats():
    """Get API usage statistics"""
    try:
        total_users_with_context = len(context_manager.user_contexts)
        total_actions = sum(len(actions) for actions in context_manager.user_contexts.values())
        
        return {
            "total_users_with_context": total_users_with_context,
            "total_actions_recorded": total_actions,
            "api_version": "2.1.0",
            "features": [
                "context_memory",
                "conversation_tracking", 
                "voice_assistant",
                "text_chat",
                "action_logging",
                "undo_support"
            ]
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    logger.info("Starting Context-Aware Restaurant Inventory Assistant API...")
    logger.info("Features: Context Memory, Conversation Tracking, Voice Assistant")
    
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise