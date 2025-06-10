# unified_context_manager.py - IMPROVED VERSION
import logging
import re
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger("UnifiedContextManager")

@dataclass
class ActionRecord:
    """Record of a user action"""
    timestamp: datetime
    action_type: str  # "update_price", "update_quantity", "check_item", etc.
    item_name: str
    item_identifier: str  # The actual ID/name used to find the item
    old_value: Any
    new_value: Any
    input_method: str  # "text" or "voice"
    success: bool
    item_data: Dict = field(default_factory=dict)

class UnifiedContextManager:
    """
    Manages conversation context for both text and voice interactions
    Tracks recent actions, remembers last updated items, handles pronouns
    """
    
    def __init__(self, max_history_per_user: int = 20, context_expiry_hours: int = 24):
        self.max_history_per_user = max_history_per_user
        self.context_expiry_hours = context_expiry_hours
        # Store context per user: user_id -> list of ActionRecords
        self.user_contexts: Dict[str, List[ActionRecord]] = {}
        
        # Patterns for context-aware queries
        self.context_patterns = {
            # Revert/undo patterns - improved regex
            'revert_to_normal': re.compile(
                r'(?:update|change|set|revert|go|put)\s+(?:back|it|that|this|price)?\s*'
                r'(?:back\s+)?(?:to\s+)?(?:normal|original|default|previous)(?:\s+price)?',
                re.IGNORECASE
            ),
            'undo_last': re.compile(
                r'(?:undo|revert|cancel)\s+(?:that|last|previous|it|the\s+last)',
                re.IGNORECASE
            ),
            
            # Reference to recent items with pronouns - improved  
            'update_it_price': re.compile(
                r'(?:update|change|set)\s+(?:it|that|this|its)\s+(?:price\s+)?(?:to\s+)?\$?(\d+\.?\d*)',
                re.IGNORECASE
            ),
            'update_it_quantity': re.compile(
                r'(?:update|change|set)\s+(?:it|that|this|its)\s+(?:quantity|units|stock)\s+(?:to\s+)?(\d+)',
                re.IGNORECASE
            ),
            'price_of_it': re.compile(
                r'(?:what|price|cost)\s+(?:is\s+)?(?:it|that|this)(?:\s+cost)?(?:\s+now)?',
                re.IGNORECASE
            ),
            'quantity_of_it': re.compile(
                r'(?:how\s+many|quantity|stock)\s+(?:of\s+)?(?:it|that|this)',
                re.IGNORECASE
            ),
            
            # What did I just do?
            'what_changed': re.compile(
                r'what\s+(?:did\s+)?(?:i\s+|we\s+)?(?:just\s+)?(?:change|update|modify)',
                re.IGNORECASE
            ),
            'last_action': re.compile(
                r'(?:what\s+was\s+)?(?:my\s+)?(?:last|previous|recent)\s+(?:action|change|update)',
                re.IGNORECASE
            ),
            
            # Show recent changes
            'recent_changes': re.compile(
                r'(?:show|list|what)\s+(?:recent|last|all)\s+(?:changes|updates|actions)',
                re.IGNORECASE
            ),
            
            # Check status of recently mentioned item
            'check_it': re.compile(
                r'(?:check|show|tell\s+me\s+about)\s+(?:it|that|this)',
                re.IGNORECASE
            ),
        }
    
    def add_action(self, user_id: str, action_type: str, item_name: str, item_identifier: str, 
                   old_value: Any, new_value: Any, input_method: str, success: bool = True, 
                   item_data: Dict = None):
        """Record a user action for context tracking"""
        
        try:
            if user_id not in self.user_contexts:
                self.user_contexts[user_id] = []
            
            action = ActionRecord(
                timestamp=datetime.now(),
                action_type=action_type,
                item_name=item_name or "Unknown",
                item_identifier=item_identifier or "Unknown",
                old_value=old_value,
                new_value=new_value,
                input_method=input_method,
                success=success,
                item_data=item_data or {}
            )
            
            self.user_contexts[user_id].append(action)
            
            # Keep only recent actions
            if len(self.user_contexts[user_id]) > self.max_history_per_user:
                self.user_contexts[user_id].pop(0)
            
            # Clean up old contexts periodically
            self._cleanup_old_contexts()
            
            logger.info(f"Recorded action for {user_id}: {action_type} on {item_name}")
            
        except Exception as e:
            logger.error(f"Error recording action for {user_id}: {e}")
    
    def get_last_action(self, user_id: str, action_type: str = None, minutes_ago: int = 30) -> Optional[ActionRecord]:
        """Get the most recent action, optionally filtered by type"""
        try:
            if user_id not in self.user_contexts:
                return None
            
            cutoff_time = datetime.now() - timedelta(minutes=minutes_ago)
            recent_actions = [a for a in self.user_contexts[user_id] 
                             if a.timestamp > cutoff_time and a.success]
            
            if not recent_actions:
                return None
            
            # Filter by action type if specified
            if action_type:
                filtered_actions = [a for a in recent_actions if a.action_type == action_type]
                return filtered_actions[-1] if filtered_actions else None
            
            return recent_actions[-1]
            
        except Exception as e:
            logger.error(f"Error getting last action for {user_id}: {e}")
            return None
    
    def get_recent_actions(self, user_id: str, limit: int = 5, minutes_ago: int = 30) -> List[ActionRecord]:
        """Get recent actions for this user"""
        try:
            if user_id not in self.user_contexts:
                return []
            
            cutoff_time = datetime.now() - timedelta(minutes=minutes_ago)
            recent_actions = [a for a in self.user_contexts[user_id] 
                             if a.timestamp > cutoff_time and a.success]
            
            return recent_actions[-limit:] if recent_actions else []
            
        except Exception as e:
            logger.error(f"Error getting recent actions for {user_id}: {e}")
            return []
    
    def process_context_query(self, user_id: str, query: str, inventory_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process queries that require conversation context"""
        
        try:
            # Validate inputs
            if not query or not query.strip():
                return None
                
            query = query.strip()
            
            # Revert to normal/original price
            if self.context_patterns['revert_to_normal'].search(query):
                last_price_action = self.get_last_action(user_id, "update_price")
                if last_price_action and last_price_action.old_value is not None:
                    return {
                        "type": "action",
                        "action_type": "update_price", 
                        "parameters": {
                            "item_identifier": last_price_action.item_identifier,
                            "new_price": last_price_action.old_value,
                            "item_name": last_price_action.item_name
                        },
                        "speech_response": f"Reverting {last_price_action.item_name} back to ${last_price_action.old_value:.2f} per unit.",
                        "context_explanation": f"Referring to the {last_price_action.item_name} you just updated"
                    }
                else:
                    return {
                        "type": "error",
                        "message": "I don't see any recent price updates to revert. What item would you like to update?",
                        "speech_response": "I don't see any recent price updates to revert. What item would you like to update?"
                    }
            
            # Undo last action
            if self.context_patterns['undo_last'].search(query):
                last_action = self.get_last_action(user_id)
                if last_action and last_action.old_value is not None:
                    if last_action.action_type == "update_price":
                        return {
                            "type": "action",
                            "action_type": "update_price",
                            "parameters": {
                                "item_identifier": last_action.item_identifier,
                                "new_price": last_action.old_value,
                                "item_name": last_action.item_name
                            },
                            "speech_response": f"Undoing price change for {last_action.item_name}. Setting it back to ${last_action.old_value:.2f}.",
                            "context_explanation": f"Undoing your last action on {last_action.item_name}"
                        }
                    elif last_action.action_type == "update_quantity":
                        return {
                            "type": "action", 
                            "action_type": "update_quantity",
                            "parameters": {
                                "item_identifier": last_action.item_identifier,
                                "new_quantity": last_action.old_value,
                                "item_name": last_action.item_name
                            },
                            "speech_response": f"Undoing quantity change for {last_action.item_name}. Setting it back to {last_action.old_value} units.",
                            "context_explanation": f"Undoing your last action on {last_action.item_name}"
                        }
                else:
                    return {
                        "type": "error",
                        "message": "I don't see any recent actions to undo.",
                        "speech_response": "I don't see any recent actions to undo."
                    }
            
            # Update "it" price - referring to last mentioned item
            price_match = self.context_patterns['update_it_price'].search(query)
            if price_match:
                try:
                    new_price = float(price_match.group(1))
                    if new_price < 0:
                        return {
                            "type": "error",
                            "message": "Price cannot be negative. Please provide a valid price.",
                            "speech_response": "Price cannot be negative. Please provide a valid price."
                        }
                    
                    last_action = self.get_last_action(user_id)  # Any recent action
                    if last_action:
                        return {
                            "type": "action",
                            "action_type": "update_price",
                            "parameters": {
                                "item_identifier": last_action.item_identifier,
                                "new_price": new_price,
                                "item_name": last_action.item_name
                            },
                            "speech_response": f"Updating {last_action.item_name} price to ${new_price:.2f} per unit.",
                            "context_explanation": f"Referring to the {last_action.item_name} you just worked with"
                        }
                    else:
                        return {
                            "type": "error",
                            "message": "I'm not sure which item you're referring to. Please specify the item name.",
                            "speech_response": "I'm not sure which item you're referring to. Please specify the item name."
                        }
                except ValueError:
                    return {
                        "type": "error",
                        "message": "Invalid price format. Please provide a valid number.",
                        "speech_response": "Invalid price format. Please provide a valid number."
                    }
            
            # Update "it" quantity  
            qty_match = self.context_patterns['update_it_quantity'].search(query)
            if qty_match:
                try:
                    new_quantity = int(qty_match.group(1))
                    if new_quantity < 0:
                        return {
                            "type": "error",
                            "message": "Quantity cannot be negative. Please provide a valid quantity.",
                            "speech_response": "Quantity cannot be negative. Please provide a valid quantity."
                        }
                    
                    last_action = self.get_last_action(user_id)
                    if last_action:
                        return {
                            "type": "action",
                            "action_type": "update_quantity",
                            "parameters": {
                                "item_identifier": last_action.item_identifier,
                                "new_quantity": new_quantity,
                                "item_name": last_action.item_name
                            },
                            "speech_response": f"Updating {last_action.item_name} quantity to {new_quantity} units.",
                            "context_explanation": f"Referring to the {last_action.item_name} you just worked with"
                        }
                    else:
                        return {
                            "type": "error",
                            "message": "I'm not sure which item you're referring to. Please specify the item name.",
                            "speech_response": "I'm not sure which item you're referring to. Please specify the item name."
                        }
                except ValueError:
                    return {
                        "type": "error",
                        "message": "Invalid quantity format. Please provide a valid number.",
                        "speech_response": "Invalid quantity format. Please provide a valid number."
                    }
            
            # Price of "it"
            if self.context_patterns['price_of_it'].search(query):
                last_action = self.get_last_action(user_id)
                if last_action:
                    # Find current item data
                    item = self._find_item_in_inventory(last_action.item_identifier, inventory_data)
                    if item:
                        price = item.get('Cost of a Unit', 0)
                        units = item.get('Total Units', 0)
                        return {
                            "type": "response",
                            "speech_response": f"{last_action.item_name} currently costs ${price:.2f} per unit with {units} units in stock.",
                            "display_response": self._create_item_info_display(item),
                            "context_explanation": f"Referring to the {last_action.item_name} you just worked with"
                        }
                    else:
                        return {
                            "type": "error",
                            "message": f"I can't find current data for {last_action.item_name} in your inventory.",
                            "speech_response": f"I can't find current data for {last_action.item_name} in your inventory."
                        }
                else:
                    return {
                        "type": "error",
                        "message": "I'm not sure which item you're asking about. Please specify the item name.",
                        "speech_response": "I'm not sure which item you're asking about. Please specify the item name."
                    }
            
            # Quantity of "it"
            if self.context_patterns['quantity_of_it'].search(query):
                last_action = self.get_last_action(user_id)
                if last_action:
                    item = self._find_item_in_inventory(last_action.item_identifier, inventory_data)
                    if item:
                        units = item.get('Total Units', 0)
                        return {
                            "type": "response",
                            "speech_response": f"{last_action.item_name} has {units} units in stock.",
                            "display_response": self._create_item_info_display(item),
                            "context_explanation": f"Referring to the {last_action.item_name} you just worked with"
                        }
                    else:
                        return {
                            "type": "error",
                            "message": f"I can't find current data for {last_action.item_name} in your inventory.",
                            "speech_response": f"I can't find current data for {last_action.item_name} in your inventory."
                        }
                else:
                    return {
                        "type": "error",
                        "message": "I'm not sure which item you're asking about. Please specify the item name.",
                        "speech_response": "I'm not sure which item you're asking about. Please specify the item name."
                    }
            
            # Check details of "it"
            if self.context_patterns['check_it'].search(query):
                last_action = self.get_last_action(user_id)
                if last_action:
                    return {
                        "type": "action",
                        "action_type": "get_item_details",
                        "parameters": {
                            "item_identifier": last_action.item_identifier,
                            "item_name": last_action.item_name
                        },
                        "speech_response": f"Getting details for {last_action.item_name}.",
                        "context_explanation": f"Referring to the {last_action.item_name} you just worked with"
                    }
                else:
                    return {
                        "type": "error",
                        "message": "I'm not sure which item you want me to check. Please specify the item name.",
                        "speech_response": "I'm not sure which item you want me to check. Please specify the item name."
                    }
            
            # What did I change?
            if self.context_patterns['what_changed'].search(query) or self.context_patterns['last_action'].search(query):
                last_action = self.get_last_action(user_id)
                if last_action:
                    if last_action.action_type == "update_price":
                        response = f"You just updated {last_action.item_name} price from ${last_action.old_value:.2f} to ${last_action.new_value:.2f} per unit."
                    elif last_action.action_type == "update_quantity":
                        response = f"You just updated {last_action.item_name} quantity from {last_action.old_value} to {last_action.new_value} units."
                    elif last_action.action_type == "check_item":
                        response = f"You just checked details for {last_action.item_name}."
                    else:
                        response = f"Your last action was {last_action.action_type.replace('_', ' ')} on {last_action.item_name}."
                    
                    return {
                        "type": "response",
                        "speech_response": response,
                        "display_response": self._create_simple_display(response)
                    }
                else:
                    return {
                        "type": "response",
                        "speech_response": "I don't see any recent changes in this conversation.",
                        "display_response": self._create_simple_display("No recent changes found.")
                    }
            
            # Show recent changes
            if self.context_patterns['recent_changes'].search(query):
                recent_actions = self.get_recent_actions(user_id, limit=5)
                if recent_actions:
                    changes_text = "Recent changes: "
                    for i, action in enumerate(recent_actions[-5:], 1):
                        if action.action_type == "update_price":
                            changes_text += f"{action.item_name} price ${action.old_value:.2f} to ${action.new_value:.2f}. "
                        elif action.action_type == "update_quantity":
                            changes_text += f"{action.item_name} quantity {action.old_value} to {action.new_value} units. "
                    
                    return {
                        "type": "response",
                        "speech_response": f"You've made {len(recent_actions)} recent changes. " + changes_text,
                        "display_response": self._create_changes_display(recent_actions)
                    }
                else:
                    return {
                        "type": "response",
                        "speech_response": "No recent changes found.",
                        "display_response": self._create_simple_display("No recent changes found.")
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing context query for {user_id}: {e}")
            return {
                "type": "error",
                "message": "I encountered an error processing your request. Please try again.",
                "speech_response": "I encountered an error processing your request. Please try again."
            }
    
    def _find_item_in_inventory(self, item_identifier: str, inventory_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find item in inventory data"""
        try:
            if not inventory_data or 'items' not in inventory_data:
                return None
            
            if not item_identifier:
                return None
            
            item_name_lower = item_identifier.lower()
            
            # Try exact match first
            for item in inventory_data['items']:
                item_full_name = item.get('Inventory Item Name', '').lower()
                if item_name_lower == item_full_name:
                    return item
            
            # Try partial match
            for item in inventory_data['items']:
                item_full_name = item.get('Inventory Item Name', '').lower()
                if (item_name_lower in item_full_name or 
                    item_full_name.startswith(item_name_lower)):
                    return item
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding item {item_identifier}: {e}")
            return None
    
    def _create_item_info_display(self, item: Dict[str, Any]) -> str:
        """Create display for item info"""
        try:
            name = item.get('Inventory Item Name', 'Unknown')
            price = item.get('Cost of a Unit', 0)
            units = item.get('Total Units', 0)
            category = item.get('Category', 'Unknown')
            
            return f"""
            <div style="background-color: #f0f9ff; padding: 15px; border-radius: 8px; border-left: 4px solid #3b82f6;">
                <h3 style="color: #1e40af; margin: 0 0 10px 0;">📦 {name}</h3>
                <p style="margin: 0; color: #374151;">
                    <strong>Price:</strong> ${price:.3f}/unit | 
                    <strong>Stock:</strong> {units} units | 
                    <strong>Category:</strong> {category}
                </p>
            </div>
            """
        except Exception as e:
            logger.error(f"Error creating item display: {e}")
            return f"<div>Error displaying item information</div>"
    
    def _create_simple_display(self, text: str) -> str:
        """Create simple display"""
        try:
            safe_text = str(text).replace('<', '&lt;').replace('>', '&gt;')
            return f"""
            <div style="background-color: #f8fafc; padding: 15px; border-radius: 8px; border-left: 4px solid #64748b;">
                <p style="margin: 0; color: #374151;">{safe_text}</p>
            </div>
            """
        except Exception as e:
            logger.error(f"Error creating simple display: {e}")
            return f"<div>Error displaying message</div>"
    
    def _create_changes_display(self, actions: List[ActionRecord]) -> str:
        """Create display for recent changes"""
        try:
            html = """
            <div style="background-color: #f0f9ff; padding: 15px; border-radius: 8px; border-left: 4px solid #3b82f6;">
                <h3 style="color: #1e40af; margin: 0 0 15px 0;">📋 Recent Changes</h3>
            """
            
            for i, action in enumerate(actions[-5:], 1):
                time_ago = datetime.now() - action.timestamp
                minutes_ago = max(1, int(time_ago.total_seconds() / 60))
                
                if action.action_type == "update_price":
                    html += f"""
                    <div style="margin: 8px 0; padding: 8px; background-color: #ffffff; border-radius: 4px;">
                        <strong>{action.item_name}</strong><br>
                        <small>Price: ${action.old_value:.2f} → ${action.new_value:.2f} ({minutes_ago}m ago)</small>
                    </div>
                    """
                elif action.action_type == "update_quantity":
                    html += f"""
                    <div style="margin: 8px 0; padding: 8px; background-color: #ffffff; border-radius: 4px;">
                        <strong>{action.item_name}</strong><br>
                        <small>Quantity: {action.old_value} → {action.new_value} units ({minutes_ago}m ago)</small>
                    </div>
                    """
                elif action.action_type == "check_item":
                    html += f"""
                    <div style="margin: 8px 0; padding: 8px; background-color: #ffffff; border-radius: 4px;">
                        <strong>{action.item_name}</strong><br>
                        <small>Checked details ({minutes_ago}m ago)</small>
                    </div>
                    """
            
            html += "</div>"
            return html
            
        except Exception as e:
            logger.error(f"Error creating changes display: {e}")
            return "<div>Error displaying recent changes</div>"
    
    def _cleanup_old_contexts(self):
        """Clean up contexts older than expiry time"""
        try:
            if len(self.user_contexts) > 100:  # Only cleanup if we have many users
                cutoff_time = datetime.now() - timedelta(hours=self.context_expiry_hours)
                users_to_clean = []
                
                for user_id, actions in self.user_contexts.items():
                    if actions and actions[-1].timestamp < cutoff_time:
                        users_to_clean.append(user_id)
                
                for user_id in users_to_clean:
                    del self.user_contexts[user_id]
                
                if users_to_clean:
                    logger.info(f"Cleaned up context for {len(users_to_clean)} inactive users")
                    
        except Exception as e:
            logger.error(f"Error during context cleanup: {e}")
    
    def clear_user_context(self, user_id: str):
        """Clear context for a specific user"""
        try:
            if user_id in self.user_contexts:
                del self.user_contexts[user_id]
            logger.info(f"Cleared context for user {user_id}")
        except Exception as e:
            logger.error(f"Error clearing context for {user_id}: {e}")
    
    def get_context_summary(self, user_id: str) -> str:
        """Get a summary of user's recent context"""
        try:
            recent_actions = self.get_recent_actions(user_id, limit=3)
            if not recent_actions:
                return "No recent actions in this conversation."
            
            summary = f"Recent context ({len(recent_actions)} actions): "
            for action in recent_actions:
                summary += f"{action.action_type.replace('_', ' ')} on {action.item_name}; "
            
            return summary.rstrip("; ")
            
        except Exception as e:
            logger.error(f"Error getting context summary for {user_id}: {e}")
            return "Error retrieving context summary."