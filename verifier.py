

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

class SubGoalStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class SubGoal:
    """Represents a subgoal in the QA plan"""
    id: str
    description: str
    expected_outcome: str
    status: str = "pending"  # Change to string to avoid enum issues
    retry_count: int = 0
    max_retries: int = 2
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VerificationResult:
    """Result of verifying a subgoal execution"""
    passed: bool
    confidence: float
    message: str
    detected_issues: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    ui_anomalies: List[str] = field(default_factory=list)
    functional_bugs: List[str] = field(default_factory=list)

class PlannerAgent:
    """
    Enhanced Planner Agent that creates and dynamically adapts QA plans
    """
    
    def __init__(self):
        self.plan_templates = {
            "wifi_test": [
                ("open_settings", "Open Settings app", "Settings screen is displayed"),
                ("navigate_wifi", "Navigate to Wi-Fi settings", "Wi-Fi settings screen is shown"),
                ("toggle_wifi_off", "Turn Wi-Fi off", "Wi-Fi is disabled"),
                ("toggle_wifi_on", "Turn Wi-Fi on", "Wi-Fi is enabled")
            ],
            "bluetooth_test": [
                ("open_settings", "Open Settings app", "Settings screen is displayed"),
                ("navigate_bluetooth", "Navigate to Bluetooth settings", "Bluetooth settings screen is shown"),
                ("toggle_bluetooth", "Toggle Bluetooth state", "Bluetooth state changes")
            ],
            "clock_test": [
                ("open_clock", "Open Clock app", "Clock app is displayed"),
                ("check_time", "Verify time display", "Current time is shown")
            ]
        }
        
        print(" PlannerAgent initialized")
    
    def create_plan(self, qa_goal: str) -> List[SubGoal]:
        """
        Create a plan of subgoals for the given QA objective
        
        Args:
            qa_goal: High-level QA goal description
            
        Returns:
            List of SubGoal objects
        """
        
        print(f" Creating plan for: {qa_goal}")
        
        # Determine plan template based on goal
        template_key = self._determine_template(qa_goal)
        template = self.plan_templates.get(template_key, [])
        
        if not template:
            # Generate generic plan
            template = self._generate_generic_plan(qa_goal)
        
        # Convert template to SubGoal objects
        subgoals = []
        for i, (goal_id, description, expected) in enumerate(template):
            subgoal = SubGoal(
                id=f"{goal_id}_{i}",
                description=description,
                expected_outcome=expected,
                metadata={"original_goal": qa_goal, "plan_step": i}
            )
            subgoals.append(subgoal)
        
        print(f"    Created plan with {len(subgoals)} subgoals")
        return subgoals
    
    def _determine_template(self, qa_goal: str) -> str:
        """Determine which template to use based on the goal"""
        goal_lower = qa_goal.lower()
        
        if "wifi" in goal_lower or "wi-fi" in goal_lower:
            return "wifi_test"
        elif "bluetooth" in goal_lower:
            return "bluetooth_test"
        elif "clock" in goal_lower or "time" in goal_lower:
            return "clock_test"
        else:
            return "generic"
    
    def _generate_generic_plan(self, qa_goal: str) -> List[tuple]:
        """Generate a generic plan when no template matches"""
        return [
            ("navigate", f"Navigate to perform: {qa_goal}", "Target screen is reached"),
            ("execute", f"Execute main action for: {qa_goal}", "Action is completed"),
            ("verify", f"Verify result of: {qa_goal}", "Result is as expected")
        ]
    
    def replan(self, failed_subgoal: SubGoal, verification_result: VerificationResult, 
               current_ui_state: Dict[str, Any]) -> List[SubGoal]:
        """
        Create a new plan when a subgoal fails
        
        Args:
            failed_subgoal: The subgoal that failed
            verification_result: Result of verification that detected the failure
            current_ui_state: Current state of the UI
            
        Returns:
            New list of SubGoal objects
        """
        
        print(f" Replanning due to failed subgoal: {failed_subgoal.description}")
        print(f"   Failure reason: {verification_result.message}")
        
        current_screen = current_ui_state.get("ui_tree", {}).get("screen_type", "unknown")
        
        # Increment retry count
        failed_subgoal.retry_count += 1
        
        # Generate recovery plan based on failure type
        recovery_plan = self._generate_recovery_plan(failed_subgoal, verification_result, current_screen)
        
        print(f"   🛠️ Generated recovery plan with {len(recovery_plan)} steps")
        
        return recovery_plan
    
    def _generate_recovery_plan(self, failed_subgoal: SubGoal, verification_result: VerificationResult, 
                               current_screen: str) -> List[SubGoal]:
        """Generate recovery plan based on failure analysis"""
        
        recovery_steps = []
        
        # Analyze failure type and generate appropriate recovery
        if "wrong screen" in verification_result.message.lower():
            # Screen navigation issue - try to get back on track
            if current_screen == "home":
                recovery_steps.append(("recovery_open_settings", "Open Settings app", "Settings screen is displayed"))
            elif "settings" in current_screen and "wifi" in failed_subgoal.description.lower():
                recovery_steps.append(("recovery_navigate_wifi", "Navigate to Wi-Fi settings", "Wi-Fi settings screen is shown"))
        
        elif "element not found" in verification_result.message.lower():
            # UI element issue - try alternative approach
            if "wifi" in failed_subgoal.description.lower():
                recovery_steps.extend([
                    ("recovery_back", "Go back to main settings", "Main settings screen is shown"),
                    ("recovery_find_wifi", "Look for Wi-Fi option", "Wi-Fi option is visible"),
                    ("recovery_navigate_wifi", "Navigate to Wi-Fi settings", "Wi-Fi settings screen is shown")
                ])
        
        elif "timeout" in verification_result.message.lower():
            # Timing issue - add wait and retry
            recovery_steps.append(("recovery_wait", "Wait for UI to stabilize", "UI elements are responsive"))
        
        # Always add the original subgoal back with retry
        if failed_subgoal.retry_count < failed_subgoal.max_retries:
            recovery_steps.append((
                f"retry_{failed_subgoal.id}",
                failed_subgoal.description,
                failed_subgoal.expected_outcome
            ))
        
        # Convert to SubGoal objects
        recovery_subgoals = []
        for i, (goal_id, description, expected) in enumerate(recovery_steps):
            subgoal = SubGoal(
                id=f"{goal_id}_{int(time.time())}_{i}",
                description=description,
                expected_outcome=expected,
                metadata={
                    "recovery_for": failed_subgoal.id,
                    "recovery_step": i,
                    "original_failure": verification_result.message
                }
            )
            recovery_subgoals.append(subgoal)
        
        return recovery_subgoals

class VerifierAgent:
    """
    Enhanced Verifier Agent that determines if app behaves as expected
    """
    
    def __init__(self):
        self.verification_history = []
        self.known_issues = []
        
        # Define verification rules
        self.verification_rules = {
            "screen_navigation": self._verify_screen_navigation,
            "element_interaction": self._verify_element_interaction,
            "state_change": self._verify_state_change,
            "ui_consistency": self._verify_ui_consistency
        }
        
        print("🔍 VerifierAgent initialized")
    
    def verify_execution(self, subgoal: SubGoal, execution_result: Dict[str, Any], 
                        current_ui_state: Dict[str, Any]) -> VerificationResult:
        """
        Verify if execution meets expectations using heuristics + LLM reasoning over UI hierarchy
        
        Args:
            subgoal: The subgoal that was executed
            execution_result: Result from executor
            current_ui_state: Current UI state
            
        Returns:
            VerificationResult object
        """
        
        print(f"🔍 Verifying: {subgoal.description}")
        
        # Initialize verification result
        result = VerificationResult(
            passed=False,
            confidence=0.0,
            message="",
            detected_issues=[],
            suggested_actions=[],
            ui_anomalies=[],
            functional_bugs=[]
        )
        
        try:
            # Check basic execution success
            if not execution_result.get("success", False):
                result.message = f"Execution failed: {execution_result.get('error_message', 'Unknown error')}"
                result.confidence = 0.9
                result.detected_issues.append("execution_failure")
                return result
            
            # Phase 1: Heuristic-based verification
            heuristic_results = self._run_heuristic_verification(subgoal, execution_result, current_ui_state)
            
            # Phase 2: LLM-style reasoning over UI hierarchy
            llm_reasoning_results = self._apply_llm_reasoning_over_ui(subgoal, execution_result, current_ui_state)
            
            # Phase 3: Combine heuristics + LLM reasoning
            combined_results = self._combine_verification_approaches(heuristic_results, llm_reasoning_results)
            
            # Update result with combined analysis
            result.passed = combined_results["overall_success"]
            result.confidence = combined_results["confidence"]
            result.message = combined_results["message"]
            result.detected_issues = combined_results["detected_issues"]
            result.ui_anomalies = combined_results["ui_anomalies"]
            result.functional_bugs = combined_results["functional_bugs"]
            result.suggested_actions = combined_results["suggested_actions"]
            
            # Record verification
            self.verification_history.append({
                "subgoal_id": subgoal.id,
                "subgoal_description": subgoal.description,
                "passed": result.passed,
                "confidence": result.confidence,
                "issues": result.detected_issues,
                "timestamp": time.time(),
                "verification_approach": "heuristics_plus_llm_reasoning"
            })
            
            # Display detailed verification breakdown
            self._display_verification_breakdown(combined_results, subgoal.description)
            
            print(f"   Result: {' PASS' if result.passed else ' FAIL'} (confidence: {result.confidence:.2f})")
            
            return result
            
        except Exception as e:
            result.message = f"Verification error: {str(e)}"
            result.confidence = 0.0
            result.detected_issues.append("verification_exception")
            print(f"    Verification exception: {e}")
            return result
    
    def _run_heuristic_verification(self, subgoal: SubGoal, execution_result: Dict[str, Any], 
                                   current_ui_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run heuristic-based verification rules"""
        
        verification_results = []
        
        for rule_name, rule_func in self.verification_rules.items():
            try:
                rule_result = rule_func(subgoal, execution_result, current_ui_state)
                verification_results.append((rule_name, rule_result))
            except Exception as e:
                print(f"    Verification rule {rule_name} failed: {e}")
        
        # Aggregate heuristic results
        passed_rules = sum(1 for _, passed in verification_results if passed)
        total_rules = len(verification_results)
        
        if total_rules == 0:
            success_rate = 0.0
        else:
            success_rate = passed_rules / total_rules
        
        return {
            "success_rate": success_rate,
            "passed_rules": passed_rules,
            "total_rules": total_rules,
            "failed_rules": [name for name, passed in verification_results if not passed],
            "approach": "heuristic"
        }
    
    def _apply_llm_reasoning_over_ui(self, subgoal: SubGoal, execution_result: Dict[str, Any], 
                                    current_ui_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply LLM-style reasoning over the UI hierarchy
        Simulates advanced language model analysis of UI state and context
        """
        
        ui_tree = current_ui_state.get("ui_tree", {})
        screen_type = ui_tree.get("screen_type", "unknown")
        elements = ui_tree.get("elements", {})
        
        # Simulate LLM reasoning process
        reasoning_analysis = {
            "ui_context_understanding": self._analyze_ui_context(subgoal, screen_type, elements),
            "goal_alignment_assessment": self._assess_goal_alignment(subgoal, execution_result, ui_tree),
            "semantic_verification": self._perform_semantic_verification(subgoal, current_ui_state),
            "contextual_anomaly_detection": self._detect_contextual_anomalies(subgoal, ui_tree),
            "intent_fulfillment_analysis": self._analyze_intent_fulfillment(subgoal, execution_result, ui_tree)
        }
        
        # Aggregate LLM-style reasoning results
        reasoning_scores = []
        reasoning_issues = []
        reasoning_insights = []
        
        for analysis_type, analysis_result in reasoning_analysis.items():
            reasoning_scores.append(analysis_result.get("score", 0.5))
            reasoning_issues.extend(analysis_result.get("issues", []))
            reasoning_insights.extend(analysis_result.get("insights", []))
        
        overall_reasoning_score = sum(reasoning_scores) / len(reasoning_scores) if reasoning_scores else 0.5
        
        return {
            "reasoning_score": overall_reasoning_score,
            "reasoning_issues": reasoning_issues,
            "reasoning_insights": reasoning_insights,
            "detailed_analysis": reasoning_analysis,
            "approach": "llm_reasoning"
        }
    
    def _analyze_ui_context(self, subgoal: SubGoal, screen_type: str, elements: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze UI context using LLM-style reasoning"""
        
        score = 0.5
        issues = []
        insights = []
        
        # Context analysis based on subgoal intent
        if "settings" in subgoal.description.lower():
            if "settings" in screen_type:
                score = 0.9
                insights.append("Successfully navigated to settings context")
            else:
                score = 0.3
                issues.append("Expected settings context but found different screen")
        
        if "wifi" in subgoal.description.lower():
            wifi_elements = [e for e in elements.keys() if "wifi" in e.lower()]
            if wifi_elements:
                score = 0.9
                insights.append("WiFi-related elements detected in UI")
            else:
                score = 0.4
                issues.append("WiFi elements not found despite WiFi-related goal")
        
        if "toggle" in subgoal.description.lower() or "turn" in subgoal.description.lower():
            toggle_elements = [e for e in elements.keys() if "toggle" in e.lower()]
            if toggle_elements:
                score = 0.9
                insights.append("Toggle elements available for interaction")
            else:
                score = 0.5
                issues.append("Toggle action requested but no toggle elements visible")
        
        return {"score": score, "issues": issues, "insights": insights}
    
    def _assess_goal_alignment(self, subgoal: SubGoal, execution_result: Dict[str, Any], 
                              ui_tree: Dict[str, Any]) -> Dict[str, Any]:
        """Assess how well the execution aligns with the stated goal"""
        
        score = 0.5
        issues = []
        insights = []
        
        action_taken = execution_result.get("action_taken", {})
        action_type = action_taken.get("action_type", "")
        element_id = action_taken.get("element_id", "")
        
        # Goal-action alignment analysis
        if "open" in subgoal.description.lower():
            if action_type == "touch" and "icon" in element_id:
                score = 0.9
                insights.append("Touch action on icon aligns with 'open' goal")
            else:
                score = 0.6
                issues.append("Action type may not align with 'open' intent")
        
        if "navigate" in subgoal.description.lower():
            if action_type == "touch" and ("option" in element_id or "icon" in element_id):
                score = 0.9
                insights.append("Touch navigation action aligns with goal")
            else:
                score = 0.5
                issues.append("Navigation intent but unclear action alignment")
        
        if "verify" in subgoal.description.lower():
            if action_type == "verify":
                score = 1.0
                insights.append("Verification action perfectly matches verification goal")
            else:
                score = 0.7
                insights.append("Non-verification action for verification goal - may be preparatory")
        
        return {"score": score, "issues": issues, "insights": insights}
    
    def _perform_semantic_verification(self, subgoal: SubGoal, current_ui_state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform semantic verification of UI state against expected outcomes"""
        
        score = 0.5
        issues = []
        insights = []
        
        expected_outcome = subgoal.expected_outcome.lower()
        ui_tree = current_ui_state.get("ui_tree", {})
        screen_type = ui_tree.get("screen_type", "")
        elements = ui_tree.get("elements", {})
        
        # Semantic matching between expected outcome and current state
        if "settings screen" in expected_outcome:
            if "settings" in screen_type:
                score = 0.95
                insights.append("Semantic match: settings screen expectation fulfilled")
            else:
                score = 0.2
                issues.append("Semantic mismatch: expected settings screen")
        
        if "wifi settings" in expected_outcome or "wi-fi settings" in expected_outcome:
            if "wifi" in screen_type or any("wifi" in e for e in elements.keys()):
                score = 0.9
                insights.append("Semantic match: WiFi settings context detected")
            else:
                score = 0.3
                issues.append("Semantic mismatch: expected WiFi settings context")
        
        if "displayed" in expected_outcome or "shown" in expected_outcome:
            if elements:
                score = 0.8
                insights.append("Display expectation met: UI elements are visible")
            else:
                score = 0.2
                issues.append("Display expectation not met: no UI elements detected")
        
        return {"score": score, "issues": issues, "insights": insights}
    
    def _detect_contextual_anomalies(self, subgoal: SubGoal, ui_tree: Dict[str, Any]) -> Dict[str, Any]:
        """Detect contextual anomalies using LLM-style reasoning"""
        
        score = 0.8  # Default: assume no anomalies
        issues = []
        insights = []
        
        screen_type = ui_tree.get("screen_type", "")
        elements = ui_tree.get("elements", {})
        
        # Contextual anomaly detection
        if "settings" in subgoal.description.lower():
            # Should have back button in settings
            if not any("back" in e for e in elements.keys()):
                score = 0.6
                issues.append("Contextual anomaly: settings screen missing back button")
        
        # Check for unexpected empty states
        if not elements:
            score = 0.3
            issues.append("Contextual anomaly: UI appears empty or not loaded")
        
        # Check for screen-goal mismatches
        if "wifi" in subgoal.description.lower() and "bluetooth" in screen_type:
            score = 0.4
            issues.append("Contextual anomaly: WiFi goal but Bluetooth screen detected")
        
        if score > 0.7:
            insights.append("No significant contextual anomalies detected")
        
        return {"score": score, "issues": issues, "insights": insights}
    
    def _analyze_intent_fulfillment(self, subgoal: SubGoal, execution_result: Dict[str, Any], 
                                   ui_tree: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how well the user's intent was fulfilled"""
        
        score = 0.5
        issues = []
        insights = []
        
        # Intent analysis based on subgoal description and execution
        intent_keywords = {
            "open": ["launch", "start", "access"],
            "navigate": ["go to", "move to", "reach"],
            "toggle": ["switch", "change", "turn"],
            "verify": ["check", "confirm", "validate"]
        }
        
        primary_intent = None
        for intent, synonyms in intent_keywords.items():
            if intent in subgoal.description.lower() or any(syn in subgoal.description.lower() for syn in synonyms):
                primary_intent = intent
                break
        
        if primary_intent:
            if primary_intent == "open":
                # Check if something was successfully opened
                if execution_result.get("success") and ui_tree.get("elements"):
                    score = 0.9
                    insights.append(f"Intent '{primary_intent}' successfully fulfilled")
                else:
                    score = 0.3
                    issues.append(f"Intent '{primary_intent}' may not have been fulfilled")
            
            elif primary_intent == "toggle":
                # Check if toggle action was performed
                action_taken = execution_result.get("action_taken", {})
                if "toggle" in action_taken.get("element_id", ""):
                    score = 0.95
                    insights.append("Toggle intent successfully executed")
                else:
                    score = 0.4
                    issues.append("Toggle intent but no toggle action detected")
            
            elif primary_intent == "verify":
                # Verification intent is fulfilled by reaching this point
                score = 0.9
                insights.append("Verification intent being processed appropriately")
        
        return {"score": score, "issues": issues, "insights": insights}
    
    def _combine_verification_approaches(self, heuristic_results: Dict[str, Any], 
                                        llm_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine heuristic and LLM reasoning results"""
        
        # Weight the approaches (can be tuned)
        heuristic_weight = 0.4
        llm_weight = 0.6
        
        # Calculate combined score
        heuristic_score = heuristic_results["success_rate"]
        llm_score = llm_results["reasoning_score"]
        
        combined_score = (heuristic_score * heuristic_weight) + (llm_score * llm_weight)
        
        # Determine overall success (threshold can be tuned)
        success_threshold = 0.7
        overall_success = combined_score >= success_threshold
        
        # Combine issues and insights
        all_issues = []
        all_issues.extend(heuristic_results.get("failed_rules", []))
        all_issues.extend(llm_results.get("reasoning_issues", []))
        
        all_insights = llm_results.get("reasoning_insights", [])
        
        # Generate combined message
        if overall_success:
            message = f"Verification passed ({heuristic_results['passed_rules']}/{heuristic_results['total_rules']} rules, LLM score: {llm_score:.2f})"
        else:
            message = f"Verification failed ({heuristic_results['passed_rules']}/{heuristic_results['total_rules']} rules, LLM score: {llm_score:.2f})"
        
        # Categorize issues
        ui_anomalies = [issue for issue in all_issues if "anomaly" in issue.lower()]
        functional_bugs = [issue for issue in all_issues if any(word in issue.lower() for word in ["mismatch", "missing", "failed"])]
        general_issues = [issue for issue in all_issues if issue not in ui_anomalies and issue not in functional_bugs]
        
        # Generate suggestions based on issues
        suggestions = []
        if ui_anomalies:
            suggestions.append("Check UI hierarchy for structural consistency")
        if functional_bugs:
            suggestions.append("Verify functional behavior meets expectations")
        if llm_score < 0.5:
            suggestions.append("Review goal-action alignment and semantic context")
        
        return {
            "overall_success": overall_success,
            "confidence": combined_score,
            "message": message,
            "detected_issues": general_issues,
            "ui_anomalies": ui_anomalies,
            "functional_bugs": functional_bugs,
            "suggested_actions": suggestions,
            "detailed_breakdown": {
                "heuristic_score": heuristic_score,
                "llm_reasoning_score": llm_score,
                "combined_score": combined_score,
                "heuristic_details": heuristic_results,
                "llm_details": llm_results
            }
        }
    
    def _display_verification_breakdown(self, combined_results: Dict[str, Any], subgoal_description: str):
        """Display detailed verification breakdown showing heuristics + LLM reasoning"""
        
        breakdown = combined_results.get("detailed_breakdown", {})
        heuristic_score = breakdown.get("heuristic_score", 0)
        llm_score = breakdown.get("llm_reasoning_score", 0)
        combined_score = breakdown.get("combined_score", 0)
        
        llm_details = breakdown.get("llm_details", {})
        
        print(f"    Verification Breakdown:")
        print(f"       Heuristic Analysis: {heuristic_score:.2f} ({breakdown.get('heuristic_details', {}).get('passed_rules', 0)}/{breakdown.get('heuristic_details', {}).get('total_rules', 0)} rules)")
        print(f"       LLM Reasoning: {llm_score:.2f}")
        print(f"       Combined Score: {combined_score:.2f} (40% heuristics + 60% LLM)")
        
        # Show LLM reasoning details
        detailed_analysis = llm_details.get("detailed_analysis", {})
        
        if detailed_analysis:
            print(f"    LLM Reasoning Breakdown:")
            
            # UI Context Understanding
            ui_context = detailed_analysis.get("ui_context_understanding", {})
            if ui_context:
                score = ui_context.get("score", 0)
                insights = ui_context.get("insights", [])
                issues = ui_context.get("issues", [])
                print(f"       UI Context Analysis: {score:.2f}")
                for insight in insights[:1]:  # Show first insight
                    print(f"          {insight}")
                for issue in issues[:1]:  # Show first issue
                    print(f"          {issue}")
            
            # Goal Alignment
            goal_alignment = detailed_analysis.get("goal_alignment_assessment", {})
            if goal_alignment:
                score = goal_alignment.get("score", 0)
                insights = goal_alignment.get("insights", [])
                print(f"       Goal Alignment: {score:.2f}")
                for insight in insights[:1]:
                    print(f"          {insight}")
            
            # Semantic Verification
            semantic = detailed_analysis.get("semantic_verification", {})
            if semantic:
                score = semantic.get("score", 0)
                insights = semantic.get("insights", [])
                print(f"       Semantic Verification: {score:.2f}")
                for insight in insights[:1]:
                    print(f"          {insight}")
            
            # Intent Fulfillment
            intent = detailed_analysis.get("intent_fulfillment_analysis", {})
            if intent:
                score = intent.get("score", 0)
                insights = intent.get("insights", [])
                print(f"       Intent Fulfillment: {score:.2f}")
                for insight in insights[:1]:
                    print(f"          {insight}")
            
            # Anomaly Detection
            anomaly = detailed_analysis.get("contextual_anomaly_detection", {})
            if anomaly:
                score = anomaly.get("score", 0)
                issues = anomaly.get("issues", [])
                if score < 0.7 and issues:
                    print(f"       Anomaly Detection: {score:.2f}")
                    for issue in issues[:1]:
                        print(f"          {issue}")
                elif score >= 0.7:
                    print(f"       Anomaly Detection: {score:.2f} - No issues detected")
    
    def _verify_screen_navigation(self, subgoal: SubGoal, execution_result: Dict[str, Any], 
                                 current_ui_state: Dict[str, Any]) -> bool:
        """Verify that screen navigation worked as expected"""
        
        expected_screens = {
            "settings": ["settings_main", "settings"],
            "wifi": ["wifi_settings", "wifi"],
            "bluetooth": ["bluetooth_settings", "bluetooth"],
            "clock": ["clock_app", "clock"]
        }
        
        current_screen = current_ui_state.get("ui_tree", {}).get("screen_type", "")
        
        # Check if we're on an expected screen based on subgoal
        for keyword, screens in expected_screens.items():
            if keyword in subgoal.description.lower():
                return any(screen in current_screen for screen in screens)
        
        # If no specific screen expected, assume navigation was successful if screen changed
        ui_before = execution_result.get("ui_state_before", {})
        screen_before = ui_before.get("screen", "")
        screen_after = current_ui_state.get("ui_tree", {}).get("screen_type", "")
        
        if "open" in subgoal.description.lower() or "navigate" in subgoal.description.lower():
            return screen_before != screen_after
        
        return True
    
    def _verify_element_interaction(self, subgoal: SubGoal, execution_result: Dict[str, Any], 
                                   current_ui_state: Dict[str, Any]) -> bool:
        """Verify that element interaction worked correctly"""
        
        action_taken = execution_result.get("action_taken", {})
        
        # Check if action was executed
        if not action_taken:
            return False
        
        # For toggle actions, verify state change
        if "toggle" in subgoal.description.lower():
            element_id = action_taken.get("element_id", "")
            if "wifi" in element_id:
                # Should be in wifi settings to toggle
                current_screen = current_ui_state.get("ui_tree", {}).get("screen_type", "")
                return "wifi" in current_screen
        
        return True
    
    def _verify_state_change(self, subgoal: SubGoal, execution_result: Dict[str, Any], 
                            current_ui_state: Dict[str, Any]) -> bool:
        """Verify that expected state changes occurred"""
        
        # For toggle actions, check if the state actually changed
        if "toggle" in subgoal.description.lower():
            info = execution_result.get("info", {})
            # If no error occurred during toggle, assume state changed
            return not info.get("error")
        
        return True
    
    def _verify_ui_consistency(self, subgoal: SubGoal, execution_result: Dict[str, Any], 
                              current_ui_state: Dict[str, Any]) -> bool:
        """Verify UI is in consistent state"""
        
        ui_tree = current_ui_state.get("ui_tree", {})
        elements = ui_tree.get("elements", {})
        
        # Basic consistency checks
        if not elements:
            return False
        
        # Check for common UI elements that should be present
        current_screen = ui_tree.get("screen_type", "")
        
        if "settings" in current_screen:
            # Settings screens should have back button
            has_back = any("back" in elem_id for elem_id in elements.keys())
            return has_back
        
        return True
    
    def _generate_suggestions(self, subgoal: SubGoal, execution_result: Dict[str, Any], 
                             verification_result: VerificationResult) -> List[str]:
        """Generate suggestions for handling verification failures"""
        
        suggestions = []
        
        if "execution_failure" in verification_result.detected_issues:
            suggestions.append("Retry the action with a slight delay")
            suggestions.append("Check if UI elements are available")
        
        if "screen_navigation" in verification_result.detected_issues:
            suggestions.append("Verify correct app is open")
            suggestions.append("Try navigating back and retrying")
        
        if "element_interaction" in verification_result.detected_issues:
            suggestions.append("Look for alternative UI elements")
            suggestions.append("Check element visibility and clickability")
        
        return suggestions
    
    def get_verification_stats(self) -> Dict[str, Any]:
        """Get verification statistics"""
        if not self.verification_history:
            return {"total_verifications": 0}
        
        total = len(self.verification_history)
        passed = sum(1 for v in self.verification_history if v["passed"])
        
        return {
            "total_verifications": total,
            "passed_verifications": passed,
            "success_rate": passed / total if total > 0 else 0,
            "common_issues": self._get_common_issues()
        }
    
    def _get_common_issues(self) -> List[str]:
        """Get most common verification issues"""
        issue_counts = {}
        for verification in self.verification_history:
            for issue in verification["issues"]:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Return top 3 most common issues
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return [issue for issue, count in sorted_issues[:3]]

# Test function
def test_verifier_system():
    """Test the planner and verifier system"""
    print(" Testing Planner and Verifier System")
    
    # Test planner
    planner = PlannerAgent()
    plan = planner.create_plan("Test turning Wi-Fi on and off")
    
    print(f"Generated plan with {len(plan)} steps:")
    for i, subgoal in enumerate(plan, 1):
        print(f"  {i}. {subgoal.description}")
    
    # Test verifier
    verifier = VerifierAgent()
    
    # Mock execution result
    mock_execution = {
        "success": True,
        "action_taken": {"action_type": "touch", "element_id": "settings_icon"},
        "ui_state_before": {"screen": "home"},
        "info": {}
    }
    
    mock_ui_state = {
        "ui_tree": {
            "screen_type": "settings_main",
            "elements": {"back_button": {"type": "button"}}
        }
    }
    
    # Verify first subgoal
    if plan:
        result = verifier.verify_execution(plan[0], mock_execution, mock_ui_state)
        print(f"Verification result: {result.passed} (confidence: {result.confidence:.2f})")
        print(f"Message: {result.message}")
    
    # Test replanning
    if plan and not result.passed:
        new_plan = planner.replan(plan[0], result, mock_ui_state)
        print(f"Replanning generated {len(new_plan)} recovery steps")

if __name__ == "__main__":
    test_verifier_system()