# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import RAGAssistant
from database import CosmosDB
import logging
import uvicorn
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RestaurantInventoryAPI")

app = FastAPI(
    title="Restaurant Inventory Assistant API",
    description="AI-powered restaurant inventory management system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class QueryRequest(BaseModel):
    user_id: str
    question: str

class UpdateInventoryRequest(BaseModel):
    user_id: str
    item_identifier: str
    field: str  # "price" or "quantity"
    value: float
    price_type: Optional[str] = "Cost of a Unit"

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Restaurant Inventory Assistant API is running",
        "version": "1.0.0"
    }

@app.post("/query")
async def query_inventory(request: QueryRequest):
    """Main query endpoint for conversational inventory assistance"""
    try:
        logger.info(f"Processing query for user {request.user_id}: {request.question}")
        
        # Initialize RAG assistant
        rag_assistant = RAGAssistant(request.user_id)
        await rag_assistant.initialize()
        
        # Process the query - now with natural conversation
        result = await rag_assistant.query(request.question)
        
        # Handle different response types
        if isinstance(result, dict):
            if result.get("type") == "action":
                # Execute the detected action
                action_result = await execute_action(
                    request.user_id,
                    result["action_type"], 
                    result["parameters"]
                )
                return {"response": action_result}
            
            elif result.get("type") == "error":
                return {"response": result["message"]}
            
            else:  # regular response, greeting, weather, etc.
                return {"response": result["message"]}
        
        else:
            # Backward compatibility
            return {"response": result}
            
    except Exception as e:
        logger.error(f"Error in query endpoint: {str(e)}")
        error_message = create_friendly_error_message(str(e))
        return {"response": error_message}

async def execute_action(user_id: str, action_type: str, parameters: dict):
    """Execute inventory management actions"""
    try:
        cosmos_db = CosmosDB()
        
        if action_type == "update_price":
            return await update_item_price(cosmos_db, user_id, parameters)
        
        elif action_type == "update_quantity":
            return await update_item_quantity(cosmos_db, user_id, parameters)
        
        elif action_type == "add_item":
            return await add_new_item(cosmos_db, user_id, parameters)
        
        elif action_type == "delete_item":
            return await delete_item(cosmos_db, user_id, parameters)
        
        elif action_type == "get_item_details":
            return await get_item_details(cosmos_db, user_id, parameters)
        
        else:
            return create_not_implemented_message(action_type)
            
    except Exception as e:
        logger.error(f"Error executing action {action_type}: {str(e)}")
        return create_action_error_message(action_type, str(e))

async def update_item_price(cosmos_db: CosmosDB, user_id: str, parameters: dict):
    """Update an item's price"""
    try:
        item_identifier = parameters.get("item_identifier") or parameters.get("item_number")
        new_price = parameters.get("new_price")
        price_type = parameters.get("price_type", "Cost of a Unit")
        
        if not item_identifier or new_price is None:
            return create_missing_params_message("price update", ["item name/number", "new price"])
        
        # Get current inventory
        inventory = await cosmos_db.get_user_documents(user_id)
        if not inventory:
            return create_no_inventory_message()
        
        items = inventory[0].get('items', [])
        updated_item = None
        
        # Find and update the item
        for item in items:
            if is_item_match(item, item_identifier):
                old_price = item.get(price_type, 0)
                item[price_type] = float(new_price)
                updated_item = item
                break
        
        if not updated_item:
            return create_item_not_found_message(item_identifier)
        
        # Update the document in Cosmos DB
        await cosmos_db.update_user_inventory(user_id, items)
        
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
        
        inventory = await cosmos_db.get_user_documents(user_id)
        if not inventory:
            return create_no_inventory_message()
        
        items = inventory[0].get('items', [])
        updated_item = None
        
        for item in items:
            if is_item_match(item, item_identifier):
                old_quantity = item.get('Total Units', 0)
                item['Total Units'] = float(new_quantity)
                updated_item = item
                break
        
        if not updated_item:
            return create_item_not_found_message(item_identifier)
        
        await cosmos_db.update_user_inventory(user_id, items)
        
        return create_quantity_update_success_message(
            updated_item.get('Inventory Item Name'),
            old_quantity,
            new_quantity
        )
        
    except Exception as e:
        logger.error(f"Error updating quantity: {str(e)}")
        return create_action_error_message("quantity update", str(e))

async def add_new_item(cosmos_db: CosmosDB, user_id: str, parameters: dict):
    """Add a new item to inventory"""
    return """
    <div style="background-color: #fef3c7; padding: 20px; border-radius: 8px; border-left: 4px solid #f59e0b;">
        <h3 style="color: #92400e; margin-top: 0;">🚧 Feature Coming Soon!</h3>
        <p style="color: #78350f; margin-bottom: 0;">Adding new items is currently being developed. For now, you can update existing items or contact support to add new inventory items.</p>
    </div>
    """

async def delete_item(cosmos_db: CosmosDB, user_id: str, parameters: dict):
    """Delete an item from inventory"""
    return """
    <div style="background-color: #fef3c7; padding: 20px; border-radius: 8px; border-left: 4px solid #f59e0b;">
        <h3 style="color: #92400e; margin-top: 0;">🚧 Feature Coming Soon!</h3>
        <p style="color: #78350f; margin-bottom: 0;">Item deletion is currently being developed for safety reasons. Please contact support if you need to remove items from your inventory.</p>
    </div>
    """

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
    item_name = item.get('Inventory Item Name', '').lower()
    item_num = str(item.get('Item Number', ''))
    identifier_str = str(identifier).lower()
    
    return (identifier_str in item_name or 
            identifier_str == item_num or
            item_name.startswith(identifier_str))

# Response message creators
def create_friendly_error_message(error_details):
    """Create a user-friendly error message"""
    return """
    <div style="background-color: #fef2f2; padding: 20px; border-radius: 8px; border-left: 4px solid #ef4444;">
        <h3 style="color: #dc2626; margin-top: 0;">😅 Oops! Something went wrong</h3>
        <p style="color: #7f1d1d; margin-bottom: 15px;">I encountered a technical issue while processing your request. This usually happens when:</p>
        <ul style="color: #7f1d1d; margin-bottom: 15px; padding-left: 20px;">
            <li>The inventory database is temporarily unavailable</li>
            <li>There's a network connectivity issue</li>
            <li>The AI service is experiencing high demand</li>
        </ul>
        <p style="color: #7f1d1d; margin-bottom: 0;">
            <strong>What you can do:</strong> Try asking your question again in a few moments, or rephrase it differently. 
            If the problem persists, please contact support.
        </p>
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
    return f"""
    <div style="background-color: #fef3c7; padding: 20px; border-radius: 8px; border-left: 4px solid #f59e0b;">
        <h3 style="color: #92400e; margin-top: 0;">🔍 Item Not Found</h3>
        <p style="color: #78350f; margin-bottom: 15px;">
            I couldn't find an item matching "<strong>{item_identifier}</strong>" in your inventory.
        </p>
        <p style="color: #78350f; margin-bottom: 0;">
            <strong>Tip:</strong> Try using the exact item name or ask me "What items do we have?" to see your full inventory.
        </p>
    </div>
    """

def create_price_update_success_message(item_name, price_type, old_price, new_price):
    """Create success message for price updates"""
    return f"""
    <div style="background-color: #f0fdf4; padding: 20px; border-radius: 8px; border-left: 4px solid #22c55e;">
        <h3 style="color: #15803d; margin-top: 0;">✅ Price Updated Successfully!</h3>
        <div style="background-color: #ffffff; padding: 15px; border-radius: 6px; margin-bottom: 15px;">
            <h4 style="color: #166534; margin-top: 0; margin-bottom: 10px;">{item_name}</h4>
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
    return f"""
    <div style="background-color: #f0fdf4; padding: 20px; border-radius: 8px; border-left: 4px solid #22c55e;">
        <h3 style="color: #15803d; margin-top: 0;">✅ Quantity Updated Successfully!</h3>
        <div style="background-color: #ffffff; padding: 15px; border-radius: 6px; margin-bottom: 15px;">
            <h4 style="color: #166534; margin-top: 0; margin-bottom: 10px;">{item_name}</h4>
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
    item_name = item.get('Inventory Item Name', 'Unknown')
    category = item.get('Category', 'Unknown')
    unit_cost = item.get('Cost of a Unit', 0)
    case_price = item.get('Case Price', 0)
    total_units = item.get('Total Units', 0)
    qty_in_case = item.get('Quantity In a Case', 0)
    measured_in = item.get('Measured In', 'units')
    item_number = item.get('Item Number', 'N/A')
    brand = item.get('Brand', 'N/A')
    
    total_value = unit_cost * total_units
    
    return f"""
    <div style="background-color: #f8fafc; padding: 20px; border-radius: 8px; border-left: 4px solid #3b82f6;">
        <h3 style="color: #1e40af; margin-top: 0; margin-bottom: 20px;">📋 Detailed Item Information</h3>
        
        <div style="background-color: #ffffff; padding: 18px; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h4 style="color: #334155; margin-top: 0; margin-bottom: 15px; font-size: 20px;">{item_name}</h4>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                <div>
                    <p style="margin: 8px 0; color: #475569;"><strong>Category:</strong> {category}</p>
                    <p style="margin: 8px 0; color: #475569;"><strong>Item Number:</strong> {item_number}</p>
                    <p style="margin: 8px 0; color: #475569;"><strong>Brand:</strong> {brand}</p>
                </div>
                
                <div>
                    <p style="margin: 8px 0; color: #475569;"><strong>Unit Cost:</strong> ${unit_cost:.2f}</p>
                    <p style="margin: 8px 0; color: #475569;"><strong>Case Price:</strong> ${case_price:.2f}</p>
                    <p style="margin: 8px 0; color: #475569;"><strong>Units per Case:</strong> {qty_in_case} {measured_in}</p>
                </div>
                
                <div>
                    <p style="margin: 8px 0; color: #475569;"><strong>Available Units:</strong> {total_units}</p>
                    <p style="margin: 8px 0; color: #475569;"><strong>Total Value:</strong> ${total_value:.2f}</p>
                    <p style="margin: 8px 0; color: #475569;"><strong>Measured In:</strong> {measured_in}</p>
                </div>
            </div>
        </div>
        
        <div style="background-color: #eff6ff; padding: 15px; border-radius: 6px;">
            <h5 style="color: #1e40af; margin-top: 0; margin-bottom: 10px;">💡 Quick Actions:</h5>
            <p style="color: #334155; margin-bottom: 0; font-size: 14px;">
                • Update price: "Update {item_name.split(',')[0]} price to $X.XX"<br>
                • Update quantity: "Set {item_name.split(',')[0]} quantity to X units"
            </p>
        </div>
    </div>
    """

def create_not_implemented_message(action_type):
    """Create message for not yet implemented actions"""
    return f"""
    <div style="background-color: #fef3c7; padding: 20px; border-radius: 8px; border-left: 4px solid #f59e0b;">
        <h3 style="color: #92400e; margin-top: 0;">🚧 Feature Coming Soon!</h3>
        <p style="color: #78350f; margin-bottom: 15px;">
            The "{action_type.replace('_', ' ')}" feature is currently being developed and will be available in a future update.
        </p>
        <p style="color: #78350f; margin-bottom: 0;">
            For now, you can ask questions about your inventory or update existing item prices and quantities.
        </p>
    </div>
    """

def create_action_error_message(action_type, error_details):
    """Create error message for failed actions"""
    return f"""
    <div style="background-color: #fef2f2; padding: 20px; border-radius: 8px; border-left: 4px solid #ef4444;">
        <h3 style="color: #dc2626; margin-top: 0;">❌ Action Failed</h3>
        <p style="color: #7f1d1d; margin-bottom: 15px;">
            I wasn't able to complete the {action_type.replace('_', ' ')} due to a technical issue.
        </p>
        <p style="color: #7f1d1d; margin-bottom: 0;">
            Please try again in a moment, or ask me a question about your inventory instead.
        </p>
    </div>
    """

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    try:
        # Test database connection
        cosmos_db = CosmosDB()
        # You could add a simple DB connectivity test here
        
        return {
            "status": "healthy",
            "services": {
                "api": "online",
                "database": "connected",
                "ai_service": "available"
            },
            "timestamp": "2024-01-01T00:00:00Z"  # You'd use actual timestamp
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/inventory/{user_id}")
async def get_inventory_summary(user_id: str):
    """Get a summary of user's inventory"""
    try:
        cosmos_db = CosmosDB()
        inventory = await cosmos_db.get_user_documents(user_id)
        
        if not inventory:
            raise HTTPException(status_code=404, detail="Inventory not found")
        
        items = inventory[0].get('items', [])
        
        # Calculate summary statistics
        total_items = len(items)
        total_value = sum(item.get('Cost of a Unit', 0) * item.get('Total Units', 0) for item in items)
        categories = set(item.get('Category', 'Unknown') for item in items)
        
        return {
            "user_id": user_id,
            "summary": {
                "total_items": total_items,
                "total_value": round(total_value, 2),
                "categories": list(categories),
                "last_updated": inventory[0].get('last_updated', 'Unknown')
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting inventory summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve inventory summary")

if __name__ == "__main__":
    logger.info("Starting Restaurant Inventory Assistant API...")
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )