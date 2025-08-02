# # # Updated Executor Agent
# # # Compatible with our working AndroidEnv

# # import logging
# # from typing import Dict, Any, Optional
# # from working_android_env import AndroidEnv

# # class UpdatedExecutorAgent:
# #     """Updated Executor Agent that works perfectly with our AndroidEnv"""
    
# #     def __init__(self, env: AndroidEnv):
# #         self.name = "UpdatedExecutor"
# #         self.env = env
# #         self.logger = logging.getLogger("UpdatedExecutor")
# #         self.execution_history = []
        
# #         print("⚡ Updated Executor Agent initialized")
    
# #     def execute_subgoal(self, subgoal) -> Dict[str, Any]:
# #         """Execute subgoal with perfect AndroidEnv compatibility"""
        
# #         self.logger.info(f"Executing: {subgoal.description}")
        
# #         try:
# #             subgoal.status = "in_progress"
            
# #             # Get current observation using the correct method
# #             current_obs = self.env._get_observation()
            
# #             # Plan the action
# #             action = self._plan_action(subgoal, current_obs["ui_tree"])
            
# #             if not action:
# #                 result = {
# #                     "success": False,
# #                     "observation": current_obs,
# #                     "action_taken": None,
# #                     "error": f"Could not plan action for: {subgoal.description}"
# #                 }
# #                 subgoal.status = "failed"
# #                 return result
            
# #             # Execute the action using the AndroidEnv.step method
# #             observation, reward, done, info = self.env.step(action)
            
# #             # Check for execution success
# #             action_success = info.get("action_success", True)
            
# #             if not action_success:
# #                 result = {
# #                     "success": False,
# #                     "observation": observation,
# #                     "action_taken": action,
# #                     "error": "Action execution failed"
# #                 }
# #                 subgoal.status = "failed"
# #                 return result
            
# #             result = {
# #                 "success": True,
# #                 "observation": observation,
# #                 "action_taken": action,
# #                 "reward": reward,
# #                 "done": done,
# #                 "info": info,
# #                 "error": None
# #             }
            
# #             subgoal.status = "completed"
# #             self.logger.info(f"✅ Success: {subgoal.description}")
            
# #             # Store in history
# #             self.execution_history.append({
# #                 "subgoal": subgoal.__dict__,
# #                 "result": result
# #             })
            
# #             return result
            
# #         except Exception as e:
# #             error_msg = f"Execution failed: {str(e)}"
# #             self.logger.error(error_msg)
            
# #             subgoal.status = "failed"
# #             return {
# #                 "success": False,
# #                 "observation": self.env._get_observation(),
# #                 "action_taken": None,
# #                 "error": error_msg
# #             }
    
# #     def _plan_action(self, subgoal, ui_tree: Dict[str, Any]) -> Optional[Dict[str, Any]]:
# #         """Plan action based on subgoal and UI tree"""
        
# #         if subgoal.action_type == "touch":
# #             # Find the target element
# #             target_element = self._find_element(ui_tree, subgoal.target_element)
            
# #             if target_element:
# #                 action = {
# #                     "action_type": "touch",
# #                     "element_id": target_element.get("resource-id", subgoal.target_element)
# #                 }
                
# #                 # Add coordinates if available
# #                 bounds = target_element.get("bounds")
# #                 if bounds and len(bounds) >= 4:
# #                     x = (bounds[0] + bounds[2]) // 2
# #                     y = (bounds[1] + bounds[3]) // 2
# #                     action["coordinates"] = [x, y]
                
# #                 return action
# #             else:
# #                 self.logger.warning(f"Element '{subgoal.target_element}' not found")
# #                 self._debug_ui_tree(ui_tree)
# #                 return None
                
# #         elif subgoal.action_type == "type":
# #             return {
# #                 "action_type": "type",
# #                 "text": subgoal.target_element
# #             }
            
# #         elif subgoal.action_type == "scroll":
# #             return {
# #                 "action_type": "scroll",
# #                 "direction": subgoal.target_element or "down"
# #             }
        
# #         return None
    
# #     def _find_element(self, ui_tree: Dict[str, Any], element_id: str) -> Optional[Dict[str, Any]]:
# #         """Find UI element in tree"""
        
# #         def search_recursive(node):
# #             if isinstance(node, dict):
# #                 # Check resource-id first
# #                 if node.get("resource-id") == element_id:
# #                     return node
                
# #                 # Check old-style id
# #                 if node.get("id") == element_id:
# #                     return node
                
# #                 # Search children
# #                 for child in node.get("children", []):
# #                     result = search_recursive(child)
# #                     if result:
# #                         return result
# #             return None
        
# #         return search_recursive(ui_tree)
    
# #     def _debug_ui_tree(self, ui_tree: Dict[str, Any]):
# #         """Debug helper to show available UI elements"""
# #         available = []
        
# #         def collect(node):
# #             if isinstance(node, dict):
# #                 if node.get("resource-id"):
# #                     available.append(node["resource-id"])
# #                 elif node.get("id"):
# #                     available.append(node["id"])
# #                 for child in node.get("children", []):
# #                     collect(child)
        
# #         collect(ui_tree)
# #         self.logger.info(f"Available elements: {available}")

# # updated_executor.py
# # Executor Agent for Multi-Agent QA System

# import time
# import json
# from typing import Dict, Any, List, Optional, Tuple
# from dataclasses import dataclass

# @dataclass
# class ExecutionResult:
#     """Result of executing a subgoal"""
#     success: bool
#     observation: Dict[str, Any]
#     action_taken: Dict[str, Any]
#     error_message: Optional[str] = None
#     execution_time: float = 0.0
#     ui_state_before: Optional[Dict[str, Any]] = None
#     ui_state_after: Optional[Dict[str, Any]] = None

# class UpdatedExecutorAgent:
#     """
#     Enhanced Executor Agent that executes subgoals in Android UI environment
#     with grounded mobile gestures and improved error handling
#     """
    
#     def __init__(self, android_env):
#         self.env = android_env
#         self.execution_history = []
#         self.action_mappings = {
#             "tap": "touch",
#             "click": "touch", 
#             "touch": "touch",
#             "type": "type",
#             "input": "type",
#             "scroll": "scroll",
#             "swipe": "scroll"
#         }
        
#         print("⚡ UpdatedExecutorAgent initialized")
    
#     def execute_subgoal(self, subgoal) -> Dict[str, Any]:
#         """
#         Execute a subgoal and return detailed results
        
#         Args:
#             subgoal: SubGoal object with description and expected outcome
            
#         Returns:
#             Dict containing execution results
#         """
        
#         start_time = time.time()
        
#         print(f"🎯 Executing: {subgoal.description}")
        
#         # Get current UI state before execution
#         ui_state_before = self.env._get_observation()
        
#         # Parse the subgoal to determine action
#         action = self._parse_subgoal_to_action(subgoal.description, ui_state_before)
        
#         if not action:
#             return {
#                 "success": False,
#                 "observation": ui_state_before,
#                 "action_taken": {},
#                 "error_message": "Could not parse subgoal into executable action",
#                 "execution_time": time.time() - start_time
#             }
        
#         try:
#             # Execute the action
#             print(f"   🤖 Action: {action}")
            
#             observation, reward, done, info = self.env.step(action)
#             execution_time = time.time() - start_time
            
#             # Determine if execution was successful
#             success = self._evaluate_execution_success(action, observation, info, subgoal)
            
#             result = {
#                 "success": success,
#                 "observation": observation,
#                 "action_taken": action,
#                 "reward": reward,
#                 "done": done,
#                 "info": info,
#                 "execution_time": execution_time,
#                 "ui_state_before": ui_state_before,
#                 "ui_state_after": observation,
#                 "error_message": info.get("error") if not success else None
#             }
            
#             # Update execution history
#             self.execution_history.append({
#                 "subgoal": subgoal.description,
#                 "action": action,
#                 "success": success,
#                 "timestamp": time.time(),
#                 "screen_before": ui_state_before.get("screen"),
#                 "screen_after": observation.get("screen")
#             })
            
#             print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
#             if not success and result.get("error_message"):
#                 print(f"   Error: {result['error_message']}")
            
#             return result
            
#         except Exception as e:
#             execution_time = time.time() - start_time
#             error_msg = f"Execution error: {str(e)}"
            
#             print(f"   ❌ Exception: {error_msg}")
            
#             return {
#                 "success": False,
#                 "observation": ui_state_before,
#                 "action_taken": action,
#                 "error_message": error_msg,
#                 "execution_time": execution_time
#             }
    
#     def _parse_subgoal_to_action(self, description: str, ui_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
#         """
#         Parse subgoal description into executable action
        
#         Args:
#             description: Natural language description of what to do
#             ui_state: Current UI state with available elements
            
#         Returns:
#             Action dictionary or None if parsing fails
#         """
        
#         description_lower = description.lower()
#         available_elements = ui_state.get("ui_tree", {}).get("elements", {})
        
#         # Extract action type
#         action_type = None
#         for keyword, action in self.action_mappings.items():
#             if keyword in description_lower:
#                 action_type = action
#                 break
        
#         if not action_type:
#             # Default to touch for most interactions
#             if any(word in description_lower for word in ["open", "go", "navigate", "select", "enable", "disable", "toggle"]):
#                 action_type = "touch"
#             else:
#                 return None
        
#         # Find target element
#         target_element = self._find_target_element(description_lower, available_elements)
        
#         if not target_element and action_type == "touch":
#             print(f"   ⚠️ No suitable element found for: {description}")
#             print(f"   Available elements: {list(available_elements.keys())}")
#             return None
        
#         action = {"action_type": action_type}
        
#         if target_element:
#             action["element_id"] = target_element
        
#         # Add text for typing actions
#         if action_type == "type":
#             # Extract text to type (simplified)
#             if "type" in description_lower:
#                 words = description.split()
#                 type_index = next((i for i, word in enumerate(words) if "type" in word.lower()), -1)
#                 if type_index >= 0 and type_index + 1 < len(words):
#                     action["text"] = " ".join(words[type_index + 1:])
        
#         # Add direction for scroll actions
#         if action_type == "scroll":
#             if "up" in description_lower:
#                 action["direction"] = "up"
#             elif "down" in description_lower:
#                 action["direction"] = "down"
#             else:
#                 action["direction"] = "down"  # default
        
#         return action
    
#     def _find_target_element(self, description: str, available_elements: Dict[str, Any]) -> Optional[str]:
#         """
#         Find the UI element that best matches the description
        
#         Args:
#             description: Lowercased description
#             available_elements: Dictionary of available UI elements
            
#         Returns:
#             Element ID or None if no match found
#         """
        
#         # Direct keyword matching
#         element_keywords = {
#             "settings": ["settings_icon", "settings_option"],
#             "wifi": ["wifi_option", "wifi_toggle", "wifi_switch"],
#             "bluetooth": ["bluetooth_option", "bluetooth_toggle"],
#             "clock": ["clock_icon", "clock_app"],
#             "back": ["back_button"],
#             "toggle": ["wifi_toggle", "bluetooth_toggle"],
#             "switch": ["wifi_toggle", "bluetooth_toggle"]
#         }
        
#         # Find best matching element
#         best_match = None
#         best_score = 0
        
#         for element_id, element_info in available_elements.items():
#             score = 0
#             element_text = element_info.get("text", "").lower()
            
#             # Check direct element ID matches
#             for keyword, element_ids in element_keywords.items():
#                 if keyword in description and element_id in element_ids:
#                     score += 10
            
#             # Check text content matches
#             description_words = description.split()
#             for word in description_words:
#                 if word in element_id.lower():
#                     score += 5
#                 if word in element_text:
#                     score += 3
            
#             # Prefer certain element types based on action
#             if "toggle" in description or "enable" in description or "disable" in description:
#                 if element_info.get("type") == "switch":
#                     score += 8
            
#             if "open" in description or "go to" in description:
#                 if element_info.get("type") in ["app_icon", "preference"]:
#                     score += 8
            
#             if score > best_score:
#                 best_score = score
#                 best_match = element_id
        
#         return best_match if best_score > 0 else None
    
#     def _evaluate_execution_success(self, action: Dict[str, Any], observation: Dict[str, Any], 
#                                   info: Dict[str, Any], subgoal) -> bool:
#         """
#         Evaluate if the execution was successful
        
#         Args:
#             action: Action that was executed
#             observation: Resulting observation
#             info: Additional info from environment
#             subgoal: Original subgoal
            
#         Returns:
#             True if execution was successful
#         """
        
#         # Check for explicit errors
#         if info.get("error"):
#             return False
        
#         # Check if we're on expected screen
#         current_screen = observation.get("screen", "")
        
#         # Screen-based success criteria
#         if "settings" in subgoal.description.lower():
#             if "settings" in current_screen:
#                 return True
        
#         if "wifi" in subgoal.description.lower():
#             if "wifi" in current_screen or "settings" in current_screen:
#                 return True
        
#         if "bluetooth" in subgoal.description.lower():
#             if "bluetooth" in current_screen:
#                 return True
        
#         if "clock" in subgoal.description.lower():
#             if "clock" in current_screen:
#                 return True
        
#         # Toggle-specific success (check if toggle happened)
#         if "toggle" in subgoal.description.lower():
#             # For toggle actions, success is typically immediate
#             return action.get("action_type") == "touch" and "toggle" in action.get("element_id", "")
        
#         # General success criteria
#         action_type = action.get("action_type", "")
        
#         if action_type == "touch":
#             # Touch is successful if no error occurred
#             return True
#         elif action_type == "type":
#             return True
#         elif action_type == "scroll":
#             return True
        
#         return False
    
#     def get_execution_history(self) -> List[Dict[str, Any]]:
#         """Get history of all executions"""
#         return self.execution_history.copy()
    
#     def get_execution_stats(self) -> Dict[str, Any]:
#         """Get execution statistics"""
#         if not self.execution_history:
#             return {"total_executions": 0}
        
#         total = len(self.execution_history)
#         successful = sum(1 for exec in self.execution_history if exec["success"])
        
#         return {
#             "total_executions": total,
#             "successful_executions": successful,
#             "success_rate": successful / total if total > 0 else 0,
#             "most_recent_screen": self.execution_history[-1].get("screen_after", "unknown")
#         }
    
#     def reset_history(self):
#         """Reset execution history"""
#         self.execution_history = []
#         print("🔄 Executor history reset")

# # Test function
# def test_executor_agent():
#     """Test the executor agent"""
#     print("🧪 Testing UpdatedExecutorAgent")
    
#     # Mock environment for testing
#     class MockEnv:
#         def __init__(self):
#             self.screen = "home"
        
#         def _get_observation(self):
#             return {
#                 "screen": self.screen,
#                 "ui_tree": {
#                     "elements": {
#                         "settings_icon": {"type": "app_icon", "text": "Settings"},
#                         "wifi_toggle": {"type": "switch", "text": "Wi-Fi ON"}
#                     }
#                 }
#             }
        
#         def step(self, action):
#             self.screen = "settings_main" if action.get("element_id") == "settings_icon" else self.screen
#             return self._get_observation(), 1.0, False, {}
    
#     # Mock subgoal
#     class MockSubgoal:
#         def __init__(self, desc):
#             self.description = desc
    
#     env = MockEnv()
#     executor = UpdatedExecutorAgent(env)
    
#     subgoal = MockSubgoal("Open settings")
#     result = executor.execute_subgoal(subgoal)
    
#     print(f"Execution result: {result['success']}")
#     print(f"Action taken: {result['action_taken']}")
    
#     stats = executor.get_execution_stats()
#     print(f"Execution stats: {stats}")

# if __name__ == "__main__":
#     test_executor_agent()

# updated_executor.py
# Executor Agent for Multi-Agent QA System

import time
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ExecutionResult:
    """Result of executing a subgoal"""
    success: bool
    observation: Dict[str, Any]
    action_taken: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time: float = 0.0
    ui_state_before: Optional[Dict[str, Any]] = None
    ui_state_after: Optional[Dict[str, Any]] = None

class UpdatedExecutorAgent:
    """
    Enhanced Executor Agent that executes subgoals in Android UI environment
    with grounded mobile gestures and improved error handling
    """
    
    def __init__(self, android_env):
        self.env = android_env
        self.execution_history = []
        self.action_mappings = {
            "tap": "touch",
            "click": "touch", 
            "touch": "touch",
            "type": "type",
            "input": "type",
            "scroll": "scroll",
            "swipe": "scroll"
        }
        
        print("⚡ UpdatedExecutorAgent initialized")
    
    def execute_subgoal(self, subgoal) -> Dict[str, Any]:
        """
        Execute a subgoal and return detailed results
        
        Args:
            subgoal: SubGoal object with description and expected outcome
            
        Returns:
            Dict containing execution results
        """
        
        start_time = time.time()
        
        print(f"🎯 Executing: {subgoal.description}")
        
        # Get current UI state before execution
        ui_state_before = self.env._get_observation()
        
        # Parse the subgoal to determine action
        action = self._parse_subgoal_to_action(subgoal.description, ui_state_before)
        
        if not action:
            return {
                "success": False,
                "observation": ui_state_before,
                "action_taken": {},
                "error_message": "Could not parse subgoal into executable action",
                "execution_time": time.time() - start_time
            }
        
        try:
            # Execute the action
            print(f"   🤖 Action: {action}")
            
            observation, reward, done, info = self.env.step(action)
            execution_time = time.time() - start_time
            
            # Determine if execution was successful
            success = self._evaluate_execution_success(action, observation, info, subgoal)
            
            result = {
                "success": success,
                "observation": observation,
                "action_taken": action,
                "reward": reward,
                "done": done,
                "info": info,
                "execution_time": execution_time,
                "ui_state_before": ui_state_before,
                "ui_state_after": observation,
                "error_message": info.get("error") if not success else None
            }
            
            # Update execution history
            self.execution_history.append({
                "subgoal": subgoal.description,
                "action": action,
                "success": success,
                "timestamp": time.time(),
                "screen_before": ui_state_before.get("screen"),
                "screen_after": observation.get("screen")
            })
            
            print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
            if not success and result.get("error_message"):
                print(f"   Error: {result['error_message']}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Execution error: {str(e)}"
            
            print(f"   ❌ Exception: {error_msg}")
            
            return {
                "success": False,
                "observation": ui_state_before,
                "action_taken": action,
                "error_message": error_msg,
                "execution_time": execution_time
            }
    
    def _parse_subgoal_to_action(self, description: str, ui_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse subgoal description into executable action
        
        Args:
            description: Natural language description of what to do
            ui_state: Current UI state with available elements
            
        Returns:
            Action dictionary or None if parsing fails
        """
        
        description_lower = description.lower()
        available_elements = ui_state.get("ui_tree", {}).get("elements", {})
        
        # Special handling for verification tasks
        if any(word in description_lower for word in ["verify", "check", "confirm"]):
            # Verification tasks are treated as successful no-ops
            return {"action_type": "verify", "description": description}
        
        # Special handling for generic execution tasks
        if "execute main action" in description_lower:
            # Try to find a reasonable default action based on context
            if available_elements:
                # Pick the first reasonable interactive element
                for element_id, element_info in available_elements.items():
                    if element_info.get("type") in ["preference", "switch", "app_icon"]:
                        return {"action_type": "touch", "element_id": element_id}
            return {"action_type": "touch", "element_id": "settings_icon"}  # Fallback
        
        # Extract action type
        action_type = None
        for keyword, action in self.action_mappings.items():
            if keyword in description_lower:
                action_type = action
                break
        
        # Default action type inference
        if not action_type:
            if any(word in description_lower for word in ["open", "go", "navigate", "select"]):
                action_type = "touch"
            elif any(word in description_lower for word in ["enable", "disable", "toggle", "turn"]):
                action_type = "touch"  # Toggle actions are touch actions
            elif any(word in description_lower for word in ["verify", "check", "confirm"]):
                action_type = "verify"
            else:
                action_type = "touch"  # Default fallback
        
        # Handle verification actions specially
        if action_type == "verify":
            return {"action_type": "verify", "description": description}
        
        # Find target element
        target_element = self._find_target_element(description_lower, available_elements)
        
        # Special handling for WiFi/Bluetooth toggle actions
        if any(word in description_lower for word in ["turn", "toggle"]) and any(word in description_lower for word in ["wifi", "wi-fi", "bluetooth"]):
            if "wifi" in description_lower or "wi-fi" in description_lower:
                target_element = "wifi_toggle"
            elif "bluetooth" in description_lower:
                target_element = "bluetooth_toggle"
        
        # If no element found but action seems valid, try fallbacks
        if not target_element and action_type == "touch":
            if "wifi" in description_lower or "wi-fi" in description_lower:
                target_element = "wifi_option"  # Try navigation first
            elif "bluetooth" in description_lower:
                target_element = "bluetooth_option"
            elif "settings" in description_lower:
                target_element = "settings_icon"
            elif "clock" in description_lower:
                target_element = "clock_icon"
        
        action = {"action_type": action_type}
        
        if target_element:
            action["element_id"] = target_element
        elif action_type != "verify":
            print(f"   ⚠️ No suitable element found for: {description}")
            print(f"   Available elements: {list(available_elements.keys())}")
            return None
        
        # Add text for typing actions
        if action_type == "type":
            # Extract text to type (simplified)
            if "type" in description_lower:
                words = description.split()
                type_index = next((i for i, word in enumerate(words) if "type" in word.lower()), -1)
                if type_index >= 0 and type_index + 1 < len(words):
                    action["text"] = " ".join(words[type_index + 1:])
        
        # Add direction for scroll actions
        if action_type == "scroll":
            if "up" in description_lower:
                action["direction"] = "up"
            elif "down" in description_lower:
                action["direction"] = "down"
            else:
                action["direction"] = "down"  # default
        
        return action
    
    def _find_target_element(self, description: str, available_elements: Dict[str, Any]) -> Optional[str]:
        """
        Find the UI element that best matches the description
        
        Args:
            description: Lowercased description
            available_elements: Dictionary of available UI elements
            
        Returns:
            Element ID or None if no match found
        """
        
        # Direct keyword matching
        element_keywords = {
            "settings": ["settings_icon", "settings_option"],
            "wifi": ["wifi_option", "wifi_toggle", "wifi_switch"],
            "wi-fi": ["wifi_option", "wifi_toggle", "wifi_switch"],
            "bluetooth": ["bluetooth_option", "bluetooth_toggle"],
            "clock": ["clock_icon", "clock_app"],
            "back": ["back_button"],
            "toggle": ["wifi_toggle", "bluetooth_toggle"],
            "switch": ["wifi_toggle", "bluetooth_toggle"],
            "turn": ["wifi_toggle", "bluetooth_toggle"],  # Added for "turn wifi off"
            "off": ["wifi_toggle", "bluetooth_toggle"],   # Added for "turn wifi off"
            "on": ["wifi_toggle", "bluetooth_toggle"],    # Added for "turn wifi on"
            "verify": [],  # Verification doesn't need UI elements
            "display": [],  # Display verification doesn't need UI elements
            "time": ["current_time"],  # For time display verification
            "execute": ["settings_icon", "wifi_option", "bluetooth_option"],  # Generic execution
            "main": ["settings_icon", "wifi_option"]  # Main actions
        }
        
        # Special cases for verification and non-action subgoals
        if any(word in description for word in ["verify", "check", "confirm"]):
            # Verification tasks don't need specific UI elements
            return None
        
        # Find best matching element
        best_match = None
        best_score = 0
        
        for element_id, element_info in available_elements.items():
            score = 0
            element_text = element_info.get("text", "").lower()
            
            # Check direct element ID matches
            for keyword, element_ids in element_keywords.items():
                if keyword in description and element_id in element_ids:
                    score += 10
            
            # Special scoring for wifi actions
            if "wifi" in description or "wi-fi" in description:
                if "wifi" in element_id:
                    score += 15
                if "turn" in description and "toggle" in element_id:
                    score += 12
                if ("off" in description or "on" in description) and "toggle" in element_id:
                    score += 12
            
            # Special scoring for bluetooth actions  
            if "bluetooth" in description:
                if "bluetooth" in element_id:
                    score += 15
                if "toggle" in element_id:
                    score += 10
            
            # Check text content matches
            description_words = description.split()
            for word in description_words:
                if word in element_id.lower():
                    score += 5
                if word in element_text:
                    score += 3
            
            # Prefer certain element types based on action
            if "toggle" in description or "enable" in description or "disable" in description or "turn" in description:
                if element_info.get("type") == "switch":
                    score += 8
            
            if "open" in description or "go to" in description or "navigate" in description:
                if element_info.get("type") in ["app_icon", "preference"]:
                    score += 8
            
            if score > best_score:
                best_score = score
                best_match = element_id
        
        return best_match if best_score > 0 else None
    
    def _evaluate_execution_success(self, action: Dict[str, Any], observation: Dict[str, Any], 
                                  info: Dict[str, Any], subgoal) -> bool:
        """
        Evaluate if the execution was successful
        
        Args:
            action: Action that was executed
            observation: Resulting observation
            info: Additional info from environment
            subgoal: Original subgoal
            
        Returns:
            True if execution was successful
        """
        
        # Check for explicit errors
        if info.get("error"):
            return False
        
        action_type = action.get("action_type", "")
        
        # Handle verification actions specially
        if action_type == "verify":
            # Verification actions are always considered successful
            # The actual verification happens in the VerifierAgent
            return True
        
        # Check if we're on expected screen
        current_screen = observation.get("screen", "")
        
        # Screen-based success criteria
        if "settings" in subgoal.description.lower():
            if "settings" in current_screen:
                return True
        
        if "wifi" in subgoal.description.lower():
            if "wifi" in current_screen or "settings" in current_screen:
                return True
        
        if "bluetooth" in subgoal.description.lower():
            if "bluetooth" in current_screen:
                return True
        
        if "clock" in subgoal.description.lower():
            if "clock" in current_screen:
                return True
        
        # Toggle-specific success (check if toggle happened)
        if "toggle" in subgoal.description.lower() or "turn" in subgoal.description.lower():
            # For toggle actions, success is typically immediate
            return action.get("action_type") == "touch" and "toggle" in action.get("element_id", "")
        
        # General success criteria
        if action_type == "touch":
            # Touch is successful if no error occurred
            return True
        elif action_type == "type":
            return True
        elif action_type == "scroll":
            return True
        elif action_type == "verify":
            return True
        
        return False
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get history of all executions"""
        return self.execution_history.copy()
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {"total_executions": 0}
        
        total = len(self.execution_history)
        successful = sum(1 for exec in self.execution_history if exec["success"])
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "success_rate": successful / total if total > 0 else 0,
            "most_recent_screen": self.execution_history[-1].get("screen_after", "unknown")
        }
    
    def reset_history(self):
        """Reset execution history"""
        self.execution_history = []
        print("🔄 Executor history reset")

# Test function
def test_executor_agent():
    """Test the executor agent"""
    print("🧪 Testing UpdatedExecutorAgent")
    
    # Mock environment for testing
    class MockEnv:
        def __init__(self):
            self.screen = "home"
        
        def _get_observation(self):
            return {
                "screen": self.screen,
                "ui_tree": {
                    "elements": {
                        "settings_icon": {"type": "app_icon", "text": "Settings"},
                        "wifi_toggle": {"type": "switch", "text": "Wi-Fi ON"}
                    }
                }
            }
        
        def step(self, action):
            self.screen = "settings_main" if action.get("element_id") == "settings_icon" else self.screen
            return self._get_observation(), 1.0, False, {}
    
    # Mock subgoal
    class MockSubgoal:
        def __init__(self, desc):
            self.description = desc
    
    env = MockEnv()
    executor = UpdatedExecutorAgent(env)
    
    subgoal = MockSubgoal("Open settings")
    result = executor.execute_subgoal(subgoal)
    
    print(f"Execution result: {result['success']}")
    print(f"Action taken: {result['action_taken']}")
    
    stats = executor.get_execution_stats()
    print(f"Execution stats: {stats}")

if __name__ == "__main__":
    test_executor_agent()