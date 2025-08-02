# Phase 1: Fixed Version - Setup + Planner + Executor
# Bug fix: Better pattern matching for WiFi detection

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class SubGoal:
    id: str
    description: str
    action_type: str  # "touch", "type", "scroll"
    target_element: Optional[str] = None
    expected_outcome: Optional[str] = None
    status: str = "pending"  # "pending", "in_progress", "completed", "failed"

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"

class MockAndroidEnv:
    """Mock Android Environment - works exactly like android_world"""
    
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.current_state = "home_screen"
        self.step_count = 0
        self.wifi_enabled = False
        self.wifi_toggle_count = 0  # Track how many times WiFi was toggled
        
        print(f" Mock AndroidEnv initialized for task: {task_name}")
    
    def reset(self):
        """Reset environment"""
        self.current_state = "home_screen"
        self.step_count = 0
        self.wifi_enabled = False
        self.wifi_toggle_count = 0
        
        observation = {
            "ui_tree": self._get_current_ui_tree(),
            "screenshot": None
        }
        
        print(f" Environment reset. Current state: {self.current_state}")
        return observation
    
    def step(self, action: Dict[str, Any]):
        """Execute action in environment"""
        self.step_count += 1
        
        print(f" Step {self.step_count}: Executing {action}")
        
        # Simulate state transitions based on actions
        if action["action_type"] == "touch":
            self._handle_touch(action.get("element_id"))
        elif action["action_type"] == "type":
            self._handle_type(action.get("text", ""))
        elif action["action_type"] == "scroll":
            self._handle_scroll(action.get("direction", "down"))
        
        # Get new observation
        observation = {
            "ui_tree": self._get_current_ui_tree(),
            "screenshot": None
        }
        
        # Calculate reward
        reward = 1.0 if self._made_progress() else 0.0
        
        # Check if episode is done
        done = self._is_task_complete() or self.step_count >= 20
        
        info = {
            "current_state": self.current_state, 
            "step_count": self.step_count,
            "wifi_toggle_count": self.wifi_toggle_count
        }
        
        return observation, reward, done, info
    
    def _get_current_ui_tree(self):
        """Generate UI tree based on current state"""
        
        if self.current_state == "home_screen":
            return {
                "package": "com.android.launcher3",
                "class": "LinearLayout",
                "children": [
                    {"id": "settings_app", "class": "AppIcon", "text": "Settings", "bounds": [100, 200, 200, 300]},
                    {"id": "clock_app", "class": "AppIcon", "text": "Clock", "bounds": [250, 200, 350, 300]}
                ]
            }
        
        elif self.current_state == "settings_main":
            return {
                "package": "com.android.settings", 
                "class": "LinearLayout",
                "children": [
                    {"id": "wifi_setting", "class": "MenuItem", "text": "Wi-Fi", "bounds": [50, 150, 400, 200]},
                    {"id": "bluetooth_setting", "class": "MenuItem", "text": "Bluetooth", "bounds": [50, 220, 400, 270]},
                    {"id": "back_button", "class": "Button", "text": "Back", "bounds": [20, 50, 80, 100]}
                ]
            }
        
        elif self.current_state == "wifi_settings":
            return {
                "package": "com.android.settings",
                "class": "LinearLayout", 
                "children": [
                    {
                        "id": "wifi_toggle", 
                        "class": "Switch", 
                        "checked": self.wifi_enabled, 
                        "bounds": [300, 100, 350, 150],
                        "text": f"WiFi {'ON' if self.wifi_enabled else 'OFF'}"
                    },
                    {"id": "back_button", "class": "Button", "text": "Back", "bounds": [20, 50, 80, 100]}
                ]
            }
        
        else:
            return {"package": "unknown", "class": "LinearLayout", "children": []}
    
    def _handle_touch(self, element_id: str):
        """Handle touch actions"""
        if element_id == "settings_app" and self.current_state == "home_screen":
            self.current_state = "settings_main"
            print(" Opened Settings app")
            
        elif element_id == "wifi_setting" and self.current_state == "settings_main":
            self.current_state = "wifi_settings"
            print(" Navigated to WiFi settings")
            
        elif element_id == "wifi_toggle" and self.current_state == "wifi_settings":
            self.wifi_enabled = not self.wifi_enabled
            self.wifi_toggle_count += 1
            print(f" WiFi toggled {'ON' if self.wifi_enabled else 'OFF'} (toggle #{self.wifi_toggle_count})")
            
        elif element_id == "back_button":
            if self.current_state == "wifi_settings":
                self.current_state = "settings_main"
                print(" Back to Settings main")
            elif self.current_state == "settings_main":
                self.current_state = "home_screen"
                print(" Back to Home screen")
    
    def _handle_type(self, text: str):
        """Handle typing actions"""
        print(f" Typed: {text}")
    
    def _handle_scroll(self, direction: str):
        """Handle scroll actions"""
        print(f" Scrolled {direction}")
    
    def _made_progress(self):
        """Check if the last action made progress"""
        return True  # Any state change is progress for now
    
    def _is_task_complete(self):
        """Check if the task is completed"""
        if self.task_name == "settings_wifi":
            # Success: WiFi was toggled at least twice (on and off)
            return self.wifi_toggle_count >= 2
        return False

class PlannerAgent:
    """Planner Agent - Creates plans from high-level goals"""
    
    def __init__(self):
        self.name = "Planner"
        self.logger = logging.getLogger("PlannerAgent")
        self.current_plan: List[SubGoal] = []
        self.planning_history: List[Dict] = []
        
        print(" Planner Agent initialized")
    
    def create_plan(self, qa_goal: str) -> List[SubGoal]:
        """Create a plan from high-level QA goal"""
        
        self.logger.info(f"Creating plan for goal: {qa_goal}")
        
        # Clear previous plan
        self.current_plan = []
        
        # Improved pattern matching
        goal_lower = qa_goal.lower().replace("-", "").replace(" ", "")
        
        if "wifi" in goal_lower or "wi-fi" in qa_goal.lower():
            self.current_plan = self._create_wifi_plan()
            print(f" Detected WiFi task, creating WiFi-specific plan")
        elif "alarm" in goal_lower:
            self.current_plan = self._create_alarm_plan()
            print(f" Detected alarm task, creating alarm-specific plan")
        elif "bluetooth" in goal_lower:
            self.current_plan = self._create_bluetooth_plan()
            print(f" Detected bluetooth task, creating bluetooth-specific plan")
        else:
            self.current_plan = self._create_generic_plan(qa_goal)
            print(f" Unknown task type, creating generic plan")
        
        # Log the plan
        self.logger.info(f"Created plan with {len(self.current_plan)} steps:")
        for i, subgoal in enumerate(self.current_plan, 1):
            self.logger.info(f"  Step {i}: {subgoal.description}")
        
        return self.current_plan
    
    def _create_wifi_plan(self) -> List[SubGoal]:
        """Create plan for WiFi testing"""
        return [
            SubGoal(
                id="wifi_step_1",
                description="Navigate to Settings app",
                action_type="touch",
                target_element="settings_app",
                expected_outcome="settings_main_screen"
            ),
            SubGoal(
                id="wifi_step_2", 
                description="Open WiFi settings",
                action_type="touch",
                target_element="wifi_setting",
                expected_outcome="wifi_settings_screen"
            ),
            SubGoal(
                id="wifi_step_3",
                description="Toggle WiFi off",
                action_type="touch",
                target_element="wifi_toggle", 
                expected_outcome="wifi_disabled"
            ),
            SubGoal(
                id="wifi_step_4",
                description="Toggle WiFi back on",
                action_type="touch",
                target_element="wifi_toggle",
                expected_outcome="wifi_enabled"
            )
        ]
    
    def _create_alarm_plan(self) -> List[SubGoal]:
        """Create plan for alarm testing"""
        return [
            SubGoal(
                id="alarm_step_1",
                description="Open Clock app",
                action_type="touch", 
                target_element="clock_app",
                expected_outcome="clock_app_open"
            )
        ]
    
    def _create_bluetooth_plan(self) -> List[SubGoal]:
        """Create plan for Bluetooth testing"""
        return [
            SubGoal(
                id="bt_step_1",
                description="Navigate to Settings",
                action_type="touch",
                target_element="settings_app", 
                expected_outcome="settings_main_screen"
            ),
            SubGoal(
                id="bt_step_2",
                description="Open Bluetooth settings",
                action_type="touch",
                target_element="bluetooth_setting",
                expected_outcome="bluetooth_settings_screen"
            )
        ]
    
    def _create_generic_plan(self, goal: str) -> List[SubGoal]:
        """Create generic plan for unknown goals"""
        return [
            SubGoal(
                id="generic_step_1",
                description=f"Execute generic action for: {goal}",
                action_type="touch",
                target_element="unknown_element",
                expected_outcome="unknown_outcome"
            )
        ]

class ExecutorAgent:
    """Executor Agent - Executes subgoals in the Android environment"""
    
    def __init__(self, env: MockAndroidEnv):
        self.name = "Executor"
        self.env = env
        self.logger = logging.getLogger("ExecutorAgent")
        self.execution_history: List[Dict] = []
        
        print(" Executor Agent initialized")
    
    def execute_subgoal(self, subgoal: SubGoal) -> Dict[str, Any]:
        """Execute a single subgoal"""
        
        self.logger.info(f"Executing subgoal: {subgoal.description}")
        
        try:
            # Update subgoal status
            subgoal.status = "in_progress"
            
            # Get current UI state
            current_obs = self.env._get_current_ui_tree()
            
            # Plan the specific action
            action = self._plan_action(subgoal, current_obs)
            
            if not action:
                result = {
                    "success": False,
                    "observation": current_obs,
                    "action_taken": None,
                    "error": f"Could not find target element '{subgoal.target_element}' in current UI state"
                }
                subgoal.status = "failed"
                return result
            
            # Execute the action
            observation, reward, done, info = self.env.step(action)
            
            # Record successful execution
            result = {
                "success": True,
                "observation": observation,
                "action_taken": action,
                "reward": reward,
                "done": done,
                "info": info,
                "error": None
            }
            
            subgoal.status = "completed"
            
            # Store in history
            self.execution_history.append({
                "timestamp": time.time(),
                "subgoal": subgoal.__dict__,
                "result": result
            })
            
            self.logger.info(f"✅ Successfully executed: {subgoal.description}")
            return result
            
        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            self.logger.error(error_msg)
            
            subgoal.status = "failed"
            
            result = {
                "success": False,
                "observation": self.env._get_current_ui_tree(),
                "action_taken": None,
                "error": error_msg
            }
            
            return result
    
    def _plan_action(self, subgoal: SubGoal, ui_tree: Dict) -> Optional[Dict[str, Any]]:
        """Plan the specific action for a subgoal"""
        
        if subgoal.action_type == "touch":
            # Find the target element in the UI tree
            target_element = self._find_element(ui_tree, subgoal.target_element)
            
            if target_element:
                return {
                    "action_type": "touch",
                    "element_id": target_element.get("id", subgoal.target_element)
                }
            else:
                self.logger.warning(f"Target element '{subgoal.target_element}' not found in UI")
                self._debug_ui_tree(ui_tree)
                return None
                
        elif subgoal.action_type == "type":
            return {
                "action_type": "type", 
                "text": subgoal.target_element
            }
            
        elif subgoal.action_type == "scroll":
            return {
                "action_type": "scroll",
                "direction": subgoal.target_element or "down"
            }
        
        return None
    
    def _find_element(self, ui_tree: Dict, element_id: str) -> Optional[Dict]:
        """Find UI element by ID in the UI tree"""
        
        def search_recursive(node):
            if isinstance(node, dict):
                # Check if this node matches
                if node.get("id") == element_id:
                    return node
                
                # Search children
                for child in node.get("children", []):
                    result = search_recursive(child)
                    if result:
                        return result
            
            return None
        
        return search_recursive(ui_tree)
    
    def _debug_ui_tree(self, ui_tree: Dict):
        """Debug helper to show available UI elements"""
        available_elements = []
        
        def collect_elements(node):
            if isinstance(node, dict):
                if node.get("id"):
                    available_elements.append(node["id"])
                for child in node.get("children", []):
                    collect_elements(child)
        
        collect_elements(ui_tree)
        self.logger.info(f"Available UI elements: {available_elements}")

def test_phase_1_fixed():
    """Test Phase 1 with the WiFi detection fix"""
    
    print("\n" + "="*60)
    print(" TESTING PHASE 1: Setup + Planner + Executor (FIXED)")
    print("="*60)
    
    # Step 1: Initialize the environment
    print("\n1. Initializing Environment...")
    env = MockAndroidEnv("settings_wifi")
    initial_obs = env.reset()
    print(f"   Initial state: {env.current_state}")
    
    # Step 2: Initialize agents
    print("\n2. Initializing Agents...")
    planner = PlannerAgent()
    executor = ExecutorAgent(env)
    
    # Step 3: Create a plan
    print("\n3. Creating Test Plan...")
    test_goal = "Test turning Wi-Fi on and off"
    plan = planner.create_plan(test_goal)
    
    print(f"\n    Plan created with {len(plan)} steps:")
    for i, subgoal in enumerate(plan, 1):
        print(f"      {i}. {subgoal.description}")
    
    # Step 4: Execute the plan
    print("\n4. Executing Plan...")
    results = []
    
    for i, subgoal in enumerate(plan, 1):
        print(f"\n   Step {i}/{len(plan)}: {subgoal.description}")
        print(f"   Current state: {env.current_state}")
        
        result = executor.execute_subgoal(subgoal)
        results.append(result)
        
        if result["success"]:
            print(f"    Success! New state: {env.current_state}")
            
            # Show additional info for WiFi toggles
            if "wifi_toggle_count" in result.get("info", {}):
                toggle_count = result["info"]["wifi_toggle_count"]
                print(f"    WiFi toggle count: {toggle_count}")
        else:
            print(f"    Failed: {result['error']}")
    
    # Step 5: Summary
    print("\n5. Execution Summary...")
    successful_steps = sum(1 for r in results if r["success"])
    print(f"   Total steps: {len(results)}")
    print(f"   Successful: {successful_steps}")
    print(f"   Failed: {len(results) - successful_steps}")
    print(f"   Final state: {env.current_state}")
    print(f"   WiFi toggle count: {env.wifi_toggle_count}")
    print(f"   Task completed: {env._is_task_complete()}")
    
    return {
        "plan": plan,
        "results": results,
        "final_state": env.current_state,
        "success_rate": successful_steps / len(results) if results else 0,
        "task_completed": env._is_task_complete()
    }

if __name__ == "__main__":
    # Run the fixed Phase 1 test
    test_results = test_phase_1_fixed()
    
    print("\n" + "="*60)
    print(" PHASE 1 COMPLETE!")
    print("="*60)
    print(f"Success rate: {test_results['success_rate']:.1%}")
    print(f"Task completed: {test_results['task_completed']}")
    
    if test_results['task_completed']:
        print(" EXCELLENT! The WiFi toggle test was successful!")
    else:
        print(" Task not fully completed. May need more steps or error handling.")
        
    print("\nNext steps:")
    print("- Phase 2: Add Verifier Agent")
    print("- Add error handling and replanning") 
    print("- Phase 3: Add Supervisor Agent")