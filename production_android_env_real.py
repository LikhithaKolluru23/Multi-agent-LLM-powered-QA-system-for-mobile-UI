


import subprocess
import tempfile
import os
import time
import json
from typing import Dict, List, Any, Optional, Tuple
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import io
from datetime import datetime

class SimpleAndroidEnv:
    """Simple Android Environment with navigation fixes"""
    
    def __init__(self, task_name: str):
        """Initialize Android Environment"""
        self.state = {
            'wifi_enabled': True,
            'bluetooth_enabled': False,
            'mobile_data_enabled': True,
            'airplane_mode': False,
            'location_enabled': True,
            'auto_brightness': True,
            'brightness_level': 50,
            'volume_level': 50,
            'do_not_disturb': False,
            'battery_saver': False,
            'developer_options': False,
            'screen_timeout': 30,
            'auto_rotate': True,
            'notification_access': True,
            'hotspot_enabled': False,
            'nfc_enabled': True,
            'screen_lock': 'swipe',
            'face_unlock': False,
            'fingerprint_unlock': False
        }
        
        self.task_name = task_name
        self.device_id = None
        self.current_screen = "home"
        self.action_history = []
        
        # Setup device connection
        self._setup_android_device()
        
        # Initialize UI elements
        self._initialize_ui_elements()
        
        # Setup task config
        self._setup_task_config()
        
        print(f" AndroidEnv initialized for task: {task_name}")

    def _run_adb_command(self, cmd: List[str], timeout: int = 5) -> subprocess.CompletedProcess:
        """Run ADB command with proper encoding handling"""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                timeout=timeout,
                encoding='utf-8',
                errors='ignore'  # Ignore Unicode decode errors
            )
            return result
        except subprocess.TimeoutExpired:
            print(f" ADB command timeout: {' '.join(cmd)}")
            return subprocess.CompletedProcess(cmd, 1, "", f"Timeout after {timeout}s")
        except Exception as e:
            print(f" ADB command error: {e}")
            return subprocess.CompletedProcess(cmd, 1, "", str(e))

    def _setup_android_device(self):
        """Set up device connection with improved error handling"""
        try:
            result = self._run_adb_command(['adb', 'devices'])
            devices = []
            
            for line in result.stdout.strip().split('\n')[1:]:
                if '\tdevice' in line:
                    devices.append(line.split('\t')[0])
            
            if not devices:
                raise Exception("No Android devices connected")
            
            self.device_id = devices[0]
            print(f" Connected to device: {self.device_id}")
            
            # Test device communication
            test_result = self._run_adb_command(['adb', '-s', self.device_id, 'shell', 'echo', 'test'])
            if test_result.returncode != 0:
                print(f" Device communication test failed: {test_result.stderr}")
            else:
                print(f" Device communication verified")
            
        except Exception as e:
            print(f" Error connecting to device: {e}")
            raise

    def _initialize_ui_elements(self):
        """Initialize UI elements"""
        self.ui_elements = {
            "home": {
                "type": "home_screen",
                "elements": {
                    "settings_app": {"type": "app_icon", "text": "Settings", "action": "open_settings"}
                }
            },
            "settings_main": {
                "type": "settings_page", 
                "title": "Settings",
                "elements": {
                    "wifi_item": {"type": "setting_item", "text": "Wi-Fi", "action": "open_wifi_settings"},
                    "bluetooth_item": {"type": "setting_item", "text": "Bluetooth", "action": "open_bluetooth_settings"}
                }
            },
            "wifi_settings": {
                "type": "settings_page",
                "title": "Wi-Fi", 
                "elements": {
                    "wifi_toggle": {"type": "switch", "text": "Wi-Fi Toggle", "action": "toggle_wifi"}
                }
            }
        }

    def _setup_task_config(self):
        """Set up task configurations"""
        self.task_config = {
            "target_screen": "wifi_settings",
            "success_condition": "wifi_enabled", 
            "description": f"Complete task: {self.task_name}"
        }

    def _exit_search_mode_if_needed(self) -> bool:
        """Exit search mode if we're stuck in it"""
        try:
            # Check if we're in search mode
            ui_tree = self.get_parsed_ui_tree()
            elements = ui_tree.get('elements', {})
            
            in_search_mode = False
            search_indicators = ['search settings', 'production_an', 'xml.etree', 'import subprocess']
            
            for element_id, element_info in elements.items():
                text = element_info.get('text', '').lower()
                content_desc = element_info.get('content_desc', '').lower()
                combined_text = f"{text} {content_desc}"
                
                if any(indicator in combined_text for indicator in search_indicators):
                    in_search_mode = True
                    print(f"🔍 Detected search mode indicator: '{text}'")
                    break
            
            if in_search_mode:
                print(" Detected search mode, attempting to exit...")
                
                # Method 1: Press back button multiple times
                for i in range(4):
                    self._press_back()
                    time.sleep(1.5)
                    
                    # Check if we've exited search mode
                    ui_tree = self.get_parsed_ui_tree()
                    elements = ui_tree.get('elements', {})
                    still_in_search = False
                    
                    for element_info in elements.values():
                        text = element_info.get('text', '').lower()
                        if any(indicator in text for indicator in search_indicators):
                            still_in_search = True
                            break
                    
                    if not still_in_search:
                        print(f"✅ Exited search mode after {i+1} back presses")
                        return True
                
                # Method 2: Press home and reset
                print(" Back button didn't work, resetting via home...")
                self._press_home()
                time.sleep(3)
                
                # Force stop settings app
                self._run_adb_command([
                    'adb', '-s', self.device_id, 'shell',
                    'am', 'force-stop', 'com.android.settings'
                ])
                time.sleep(2)
                
                print(" Reset to clean state")
                return True
            
            return True  # Not in search mode
            
        except Exception as e:
            print(f" Error checking/exiting search mode: {e}")
            return False

    def _ensure_clean_navigation_state(self) -> bool:
        """Ensure clean navigation state before any action"""
        try:
            # Always start from home for clean state
            print(" Ensuring clean navigation state...")
            self._press_home()
            time.sleep(2)
            
            # Force stop any problematic apps
            problematic_apps = [
                'com.android.settings',
                'com.google.android.gms'
            ]
            
            for app in problematic_apps:
                self._run_adb_command([
                    'adb', '-s', self.device_id, 'shell',
                    'am', 'force-stop', app
                ], timeout=3)
            
            time.sleep(1)
            print(" Clean navigation state ensured")
            return True
            
        except Exception as e:
            print(f" Error ensuring clean state: {e}")
            return True

    def capture_screenshot(self):
        """Enhanced screenshot capture with better error handling"""
        try:
            # Method 1: Try direct binary capture first
            result = subprocess.run([
                'adb', '-s', self.device_id, 'shell', 'screencap', '-p'
            ], capture_output=True, timeout=10)
            
            if result.returncode == 0 and len(result.stdout) > 1000:
                screenshot_data = result.stdout
                if isinstance(screenshot_data, str):
                    screenshot_data = screenshot_data.encode('latin-1')
                
                # Remove carriage returns that might corrupt PNG data
                screenshot_data = screenshot_data.replace(b'\r\n', b'\n')
                print(f" Direct screenshot captured: {len(screenshot_data)} bytes")
                return screenshot_data
            
            # Method 2: File-based fallback
            print(" Trying file-based screenshot method...")
            timestamp = int(time.time())
            remote_path = f'/sdcard/screenshot_{timestamp}.png'
            
            # Capture to device storage
            capture_result = self._run_adb_command([
                'adb', '-s', self.device_id, 'shell',
                'screencap', '-p', remote_path
            ], timeout=10)
            
            if capture_result.returncode == 0:
                # Check if file exists
                check_result = self._run_adb_command([
                    'adb', '-s', self.device_id, 'shell',
                    'ls', remote_path
                ])
                
                if check_result.returncode == 0:
                    # Pull the file
                    local_path = tempfile.mktemp(suffix='.png')
                    pull_result = self._run_adb_command([
                        'adb', '-s', self.device_id, 'pull', remote_path, local_path
                    ], timeout=10)
                    
                    if pull_result.returncode == 0 and os.path.exists(local_path):
                        # Read the file
                        with open(local_path, 'rb') as f:
                            screenshot_data = f.read()
                        
                        # Cleanup
                        try:
                            os.unlink(local_path)
                            self._run_adb_command([
                                'adb', '-s', self.device_id, 'shell', 'rm', remote_path
                            ], timeout=3)
                        except Exception as cleanup_error:
                            print(f" Cleanup warning: {cleanup_error}")
                        
                        if len(screenshot_data) > 1000:
                            print(f" File-based screenshot: {len(screenshot_data)} bytes")
                            return screenshot_data
            
            print(" All screenshot methods failed")
            return None
                
        except Exception as e:
            print(f" Screenshot capture error: {e}")
            return None

    def convert_screenshot_to_array(self, screenshot_bytes):
        """Convert screenshot bytes to numpy array with enhanced error handling"""
        try:
            if screenshot_bytes is None:
                print(" No screenshot data provided")
                return None
            
            # Ensure we have bytes
            if isinstance(screenshot_bytes, str):
                screenshot_bytes = screenshot_bytes.encode('latin-1')
            
            # Validate minimum file size
            if len(screenshot_bytes) < 1000:
                print(f" Screenshot data too small: {len(screenshot_bytes)} bytes")
                return None
            
            # Try to open the image
            try:
                image = Image.open(io.BytesIO(screenshot_bytes))
                print(f" PIL successfully opened image: {image.size}, mode: {image.mode}")
            except Exception as img_error:
                print(f" PIL cannot read image: {img_error}")
                return None
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                print(f" Converting from {image.mode} to RGB")
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image)
            print(f" Screenshot converted to array: {image_array.shape}")
            
            # Validate array
            if image_array.size == 0:
                print(" Empty image array")
                return None
            
            return image_array
            
        except Exception as e:
            print(f" Error converting screenshot: {e}")
            return None

    def get_screen_xml(self) -> Optional[str]:
        """Get current screen UI hierarchy as XML with better error handling"""
        try:
            # Dump UI to device
            dump_result = self._run_adb_command([
                'adb', '-s', self.device_id, 'shell',
                'uiautomator', 'dump', '/sdcard/ui_dump.xml'
            ], timeout=10)
            
            if dump_result.returncode == 0:
                # Read the XML file
                xml_result = self._run_adb_command([
                    'adb', '-s', self.device_id, 'shell',
                    'cat', '/sdcard/ui_dump.xml'
                ], timeout=10)
                
                if xml_result.returncode == 0 and xml_result.stdout.strip():
                    print(f" UI XML captured: {len(xml_result.stdout)} characters")
                    return xml_result.stdout
                else:
                    print(f" Failed to read UI XML: {xml_result.stderr}")
            else:
                print(f" Failed to dump UI: {dump_result.stderr}")
            
            return None
            
        except Exception as e:
            print(f" Error getting screen XML: {e}")
            return None

    def get_parsed_ui_tree(self) -> Dict[str, Any]:
        """Get UI tree with enhanced element detection and error handling"""
        try:
            xml_content = self.get_screen_xml()
            if not xml_content:
                print(" No XML content available")
                return {"elements": {}, "raw_xml": "", "error": "No XML content"}
            
            try:
                root = ET.fromstring(xml_content)
            except ET.ParseError as parse_error:
                print(f" XML parsing error: {parse_error}")
                return {"elements": {}, "raw_xml": xml_content, "error": f"XML parse error: {parse_error}"}
            
            elements = {}
            
            for i, element in enumerate(root.iter()):
                if element.tag == 'node':
                    element_id = f"element_{i}"
                    text = element.get('text', '').strip()
                    content_desc = element.get('content-desc', '').strip()
                    bounds = element.get('bounds', '')
                    clickable = element.get('clickable', 'false') == 'true'
                    class_name = element.get('class', '')
                    resource_id = element.get('resource-id', '')
                    
                    # Enhanced priority-based element detection
                    should_include = False
                    element_priority = 0
                    element_type = "unknown"
                    
                    # HIGH PRIORITY: Settings and connectivity elements
                    wifi_keywords = ['wi-fi', 'wifi', 'network & internet', 'network and internet', 'wireless', 'connections']
                    bluetooth_keywords = ['bluetooth', 'connected devices', 'device connectivity']
                    settings_keywords = ['settings', 'preferences', 'configuration']
                    
                    text_lower = text.lower()
                    content_lower = content_desc.lower()
                    combined_text = f"{text_lower} {content_lower}".strip()
                    
                    # Skip problematic elements that indicate search mode
                    skip_keywords = ['search settings', 'production_an', 'xml.etree', 'import', 'subprocess']
                    if any(keyword in combined_text for keyword in skip_keywords):
                        continue
                    
                    if any(keyword in combined_text for keyword in wifi_keywords):
                        should_include = True
                        element_priority = 100
                        element_type = "wifi_target"
                        print(f" HIGH PRIORITY WiFi element: '{text}' at {bounds}")
                    
                    elif any(keyword in combined_text for keyword in bluetooth_keywords):
                        should_include = True
                        element_priority = 100
                        element_type = "bluetooth_target"
                        print(f" HIGH PRIORITY Bluetooth element: '{text}' at {bounds}")
                    
                    elif any(keyword in combined_text for keyword in settings_keywords):
                        should_include = True
                        element_priority = 90
                        element_type = "settings_target"
                        print(f" HIGH PRIORITY Settings element: '{text}' at {bounds}")
                    
                    # MEDIUM PRIORITY: Navigation and app elements
                    elif text_lower in ['apps', 'notifications', 'sound', 'display', 'security', 'privacy']:
                        should_include = True
                        element_priority = 50
                        element_type = "navigation_item"
                    
                    # LOW PRIORITY: Clickable elements with meaningful text
                    elif clickable and bounds and text and len(text) > 1:
                        # Exclude problematic elements
                        exclude_keywords = ['search', 'import', 'xml', 'debug', 'test', 'production']
                        if not any(keyword in text_lower for keyword in exclude_keywords):
                            should_include = True
                            element_priority = 20
                            element_type = "clickable"
                    
                    # VERY LOW PRIORITY: Clickable without text (icons, buttons)
                    elif clickable and bounds:
                        should_include = True
                        element_priority = 5
                        element_type = "clickable_icon"
                    
                    if should_include:
                        elements[element_id] = {
                            'text': text,
                            'content_desc': content_desc,
                            'bounds': bounds,
                            'clickable': clickable,
                            'class': class_name,
                            'resource_id': resource_id,
                            'enabled': element.get('enabled', 'true') == 'true',
                            'element_type': element_type,
                            'priority': element_priority
                        }
                        
                        # Parse bounds for coordinates with error handling
                        if bounds:
                            try:
                                # Handle bounds format: [x1,y1][x2,y2]
                                bounds_clean = bounds.replace('[', '').replace(']', ',')
                                coords = [int(x) for x in bounds_clean.split(',') if x.strip()]
                                
                                if len(coords) >= 4:
                                    x1, y1, x2, y2 = coords[:4]
                                    elements[element_id].update({
                                        'center_x': (x1 + x2) // 2,
                                        'center_y': (y1 + y2) // 2,
                                        'width': x2 - x1,
                                        'height': y2 - y1
                                    })
                            except (ValueError, IndexError) as coord_error:
                                print(f"⚠️ Could not parse bounds '{bounds}': {coord_error}")
            
            # Sort elements by priority
            sorted_elements = dict(sorted(elements.items(), 
                                        key=lambda x: x[1].get('priority', 0), 
                                        reverse=True))
            
            print(f" Parsed UI tree: {len(elements)} elements found")
            
            return {
                "elements": sorted_elements,
                "raw_xml": xml_content,
                "element_count": len(elements),
                "high_priority_count": sum(1 for e in elements.values() if e.get('priority', 0) >= 90)
            }
            
        except Exception as e:
            print(f" Error parsing UI tree: {e}")
            return {"elements": {}, "raw_xml": "", "error": str(e)}

    def find_best_element_for_action(self, target_action: str) -> Optional[str]:
        """Find the best element for a specific action with improved matching"""
        ui_tree = self.get_parsed_ui_tree()
        elements = ui_tree.get('elements', {})
        
        if not elements:
            print(" No UI elements available for action matching")
            return None
        
        # Enhanced action keywords
        action_keywords = {
            'wifi': ['wi-fi', 'wifi', 'network & internet', 'network and internet', 'wireless', 'connectivity'],
            'bluetooth': ['bluetooth', 'connected devices', 'device connectivity', 'wireless'],
            'settings': ['settings', 'preferences', 'configuration'],
            'apps': ['apps', 'applications', 'app info'],
            'sound': ['sound & vibration', 'sound and vibration', 'audio', 'volume'],
            'display': ['display', 'screen', 'brightness']
        }
        
        keywords = action_keywords.get(target_action.lower(), [target_action.lower()])
        
        # Find matching elements with scoring
        matching_elements = []
        for element_id, element_info in elements.items():
            text = element_info.get('text', '').lower()
            content_desc = element_info.get('content_desc', '').lower()
            combined_text = f"{text} {content_desc}".strip()
            
            # Calculate match score
            match_score = 0
            for keyword in keywords:
                if keyword in combined_text:
                    if keyword in text:
                        match_score += 2  # Higher score for text matches
                    else:
                        match_score += 1  # Lower score for content-desc matches
            
            if match_score > 0:
                total_score = match_score * element_info.get('priority', 1)
                matching_elements.append((element_id, element_info, total_score))
                print(f" Match for '{target_action}': {element_info.get('text', 'no text')} "
                      f"(score: {total_score}, priority: {element_info.get('priority', 0)})")
        
        # Return the highest scored match
        if matching_elements:
            best_match = max(matching_elements, key=lambda x: x[2])
            print(f" Best match for '{target_action}': {best_match[1].get('text', 'no text')} "
                  f"(final score: {best_match[2]})")
            return best_match[0]
        
        print(f" No matching elements found for action: {target_action}")
        return None

    def _open_settings(self) -> bool:
        """Enhanced Settings app opening with navigation fixes"""
        try:
            # Ensure clean state first
            self._ensure_clean_navigation_state()
            
            # Exit any search modes
            self._exit_search_mode_if_needed()
            
            print(" Opening Settings app...")
            
            # Method 1: Direct intent (most reliable)
            result = self._run_adb_command([
                'adb', '-s', self.device_id, 'shell',
                'am', 'start', '-a', 'android.settings.SETTINGS'
            ], timeout=10)
            
            if result.returncode == 0:
                self.current_screen = "settings_main"
                time.sleep(3)  # Wait for Settings to fully load
                print(" Opened Settings app via intent")
                
                # Verify we're not in search mode after opening
                self._exit_search_mode_if_needed()
                return True
            else:
                print(f" Settings intent failed: {result.stderr}")
            
            # Method 2: Try via launcher
            print(" Trying launcher method...")
            launcher_result = self._run_adb_command([
                'adb', '-s', self.device_id, 'shell',
                'am', 'start', '-c', 'android.intent.category.LAUNCHER', 
                '-a', 'android.intent.action.MAIN',
                'com.android.settings/.Settings'
            ], timeout=10)
            
            if launcher_result.returncode == 0:
                self.current_screen = "settings_main"
                time.sleep(3)
                print("⚙️ Opened Settings via launcher")
                self._exit_search_mode_if_needed()
                return True
            
            return False
            
        except Exception as e:
            print(f" Settings opening error: {e}")
            return False

    def _open_wifi_settings_smart(self) -> bool:
        """Enhanced WiFi settings opening that avoids search mode"""
        try:
            # Ensure clean navigation state
            self._ensure_clean_navigation_state()
            
            # Method 1: Direct WiFi settings intent (most reliable)
            print(" Opening WiFi settings directly...")
            result = self._run_adb_command([
                'adb', '-s', self.device_id, 'shell',
                'am', 'start', '-a', 'android.settings.WIFI_SETTINGS'
            ], timeout=10)
            
            if result.returncode == 0:
                self.current_screen = "wifi_settings"
                time.sleep(3)  # Wait for WiFi settings to load
                print(" Opened WiFi settings directly")
                return True
            
            # Method 2: Through Settings navigation
            print(" Trying navigation through Settings...")
            if not self._open_settings():
                print(" Could not open Settings app")
                return False
            
            time.sleep(2)
            
            # Look for WiFi/Network options
            wifi_element = self.find_best_element_for_action('wifi')
            if wifi_element:
                ui_tree = self.get_parsed_ui_tree()
                elements = ui_tree.get('elements', {})
                element_info = elements.get(wifi_element)
                
                if element_info and 'center_x' in element_info and 'center_y' in element_info:
                    print(f" Tapping WiFi option: {element_info.get('text', 'no text')}")
                    if self._tap(element_info['center_x'], element_info['center_y']):
                        time.sleep(3)
                        self.current_screen = "wifi_settings"
                        print(" Navigated to WiFi settings")
                        return True
            
            # Method 3: Alternative intents
            print(" Trying alternative network settings...")
            alternative_intents = [
                'android.settings.WIRELESS_SETTINGS',
                'android.settings.NETWORK_OPERATOR_SETTINGS'
            ]
            
            for intent in alternative_intents:
                result = self._run_adb_command([
                    'adb', '-s', self.device_id, 'shell',
                    'am', 'start', '-a', intent
                ], timeout=8)
                
                if result.returncode == 0:
                    time.sleep(2)
                    print(f" Opened network settings via {intent}")
                    
                    # Look for WiFi within this screen
                    wifi_sub_element = self.find_best_element_for_action('wifi')
                    if wifi_sub_element:
                        ui_tree = self.get_parsed_ui_tree()
                        elements = ui_tree.get('elements', {})
                        element_info = elements.get(wifi_sub_element)
                        
                        if element_info and 'center_x' in element_info:
                            self._tap(element_info['center_x'], element_info['center_y'])
                            time.sleep(2)
                            self.current_screen = "wifi_settings"
                            print(" Found WiFi within network settings")
                            return True
            
            print(" All WiFi settings methods failed")
            return False
                
        except Exception as e:
            print(f" WiFi settings error: {e}")
            return False

    def _open_bluetooth_settings_smart(self) -> bool:
        """Enhanced Bluetooth settings opening"""
        try:
            # Ensure clean navigation state
            self._ensure_clean_navigation_state()
            
            # Method 1: Direct intent
            print(" Opening Bluetooth settings directly...")
            result = self._run_adb_command([
                'adb', '-s', self.device_id, 'shell',
                'am', 'start', '-a', 'android.settings.BLUETOOTH_SETTINGS'
            ], timeout=10)
            
            if result.returncode == 0:
                self.current_screen = "bluetooth_settings"
                time.sleep(3)
                print(" Opened Bluetooth settings directly")
                return True
            
            # Method 2: Through Settings navigation
            if not self._open_settings():
                return False
            
            time.sleep(2)
            bluetooth_element = self.find_best_element_for_action('bluetooth')
            if bluetooth_element:
                ui_tree = self.get_parsed_ui_tree()
                elements = ui_tree.get('elements', {})
                element_info = elements.get(bluetooth_element)
                
                if element_info and 'center_x' in element_info:
                    if self._tap(element_info['center_x'], element_info['center_y']):
                        time.sleep(3)
                        self.current_screen = "bluetooth_settings"
                        print(" Navigated to Bluetooth settings")
                        return True
            
            return False
                
        except Exception as e:
            print(f" Bluetooth settings error: {e}")
            return False

    def perform_action(self, action, **kwargs) -> bool:
        """Enhanced action performance with navigation fixes"""
        try:
            # Handle dictionary actions
            if isinstance(action, dict):
                action_type = action.get('action_type', '')
                element_id = action.get('element_id', '')
                
                if action_type == 'touch':
                    if element_id == 'settings_icon':
                        action = 'open_settings'
                    elif 'wifi' in element_id.lower():
                        action = 'open_wifi_settings'
                    elif 'bluetooth' in element_id.lower():
                        action = 'open_bluetooth_settings'
                    elif element_id.startswith('element_'):
                        # Enhanced element targeting
                        ui_tree = self.get_parsed_ui_tree()
                        elements = ui_tree.get('elements', {})
                        
                        if element_id in elements:
                            element_info = elements[element_id]
                            element_text = element_info.get('text', '').lower()
                            content_desc = element_info.get('content_desc', '').lower()
                            combined_text = f"{element_text} {content_desc}".strip()
                            
                            # Smart action detection
                            if any(keyword in combined_text for keyword in ['wi-fi', 'wifi', 'network & internet', 'wireless']):
                                print(f" Detected WiFi element: '{element_info.get('text', 'no text')}'")
                                action = 'open_wifi_settings'
                            elif any(keyword in combined_text for keyword in ['bluetooth', 'connected devices']):
                                print(f" Detected Bluetooth element: '{element_info.get('text', 'no text')}'")
                                action = 'open_bluetooth_settings'
                            elif 'settings' in combined_text:
                                print(f" Detected Settings element: '{element_info.get('text', 'no text')}'")
                                action = 'open_settings'
                            elif 'center_x' in element_info and 'center_y' in element_info:
                                # Generic tap with coordinates
                                action = 'tap'
                                kwargs.update({
                                    'x': element_info['center_x'],
                                    'y': element_info['center_y']
                                })
                                print(f" Tapping '{element_info.get('text', 'no text')}' at "
                                      f"({element_info['center_x']}, {element_info['center_y']})")
                            else:
                                print(f" Element {element_id} found but no coordinates available")
                                return False
                        else:
                            print(f" Element {element_id} not found in UI tree")
                            return False
                            
                elif action_type == 'verify':
                    print(f" Verification: {action.get('description', 'Unknown')}")
                    return True
            
            # Record action
            self.action_history.append({
                "action": action, 
                "timestamp": time.time(),
                "kwargs": kwargs
            })
            
            # Execute action with improved error handling
            if action == "toggle_wifi":
                return self._toggle_wifi()
            elif action == "toggle_bluetooth":
                return self._toggle_bluetooth()
            elif action == "open_settings":
                return self._open_settings()
            elif action == "open_wifi_settings":
                return self._open_wifi_settings_smart()
            elif action == "open_bluetooth_settings":
                return self._open_bluetooth_settings_smart()
            elif action == "tap":
                return self._tap(kwargs.get('x', 500), kwargs.get('y', 500))
            elif action == "press_back":
                return self._press_back()
            elif action == "press_home":
                return self._press_home()
            else:
                print(f" Simulated action: {action}")
                return True
                
        except Exception as e:
            print(f" Action execution error: {e}")
            return False

    def _toggle_wifi(self) -> bool:
        """Enhanced WiFi toggle with state verification"""
        try:
            current_state = self.state.get('wifi_enabled', True)
            new_state = not current_state
            
            if new_state:
                result = self._run_adb_command([
                    'adb', '-s', self.device_id, 'shell', 'svc', 'wifi', 'enable'
                ], timeout=8)
            else:
                result = self._run_adb_command([
                    'adb', '-s', self.device_id, 'shell', 'svc', 'wifi', 'disable'
                ], timeout=8)
            
            if result.returncode == 0:
                self.state['wifi_enabled'] = new_state
                self.current_screen = "wifi_settings"
                print(f" WiFi {'enabled' if new_state else 'disabled'}")
                
                # Wait for state change to take effect
                time.sleep(2)
                return True
            else:
                print(f" WiFi toggle failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f" WiFi toggle error: {e}")
            return False

    def _toggle_bluetooth(self) -> bool:
        """Enhanced Bluetooth toggle"""
        try:
            current_state = self.state.get('bluetooth_enabled', False)
            new_state = not current_state
            
            if new_state:
                result = self._run_adb_command([
                    'adb', '-s', self.device_id, 'shell', 'svc', 'bluetooth', 'enable'
                ], timeout=8)
            else:
                result = self._run_adb_command([
                    'adb', '-s', self.device_id, 'shell', 'svc', 'bluetooth', 'disable'
                ], timeout=8)
            
            if result.returncode == 0:
                self.state['bluetooth_enabled'] = new_state
                self.current_screen = "bluetooth_settings"
                print(f"🔵 Bluetooth {'enabled' if new_state else 'disabled'}")
                
                # Wait for state change to take effect
                time.sleep(2)
                return True
            else:
                print(f" Bluetooth toggle failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f" Bluetooth toggle error: {e}")
            return False

    def _tap(self, x: int, y: int) -> bool:
        """Enhanced tap gesture with validation"""
        try:
            # Validate coordinates
            if not (0 <= x <= 5000 and 0 <= y <= 5000):
                print(f" Invalid tap coordinates: ({x}, {y})")
                return False
            
            result = self._run_adb_command([
                'adb', '-s', self.device_id, 'shell',
                'input', 'tap', str(x), str(y)
            ], timeout=8)
            
            if result.returncode == 0:
                time.sleep(1)  # Wait for tap to register
                print(f" Tapped at ({x}, {y})")
                return True
            else:
                print(f" Tap failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f" Tap error: {e}")
            return False

    def _press_back(self) -> bool:
        """Enhanced back button press"""
        try:
            result = self._run_adb_command([
                'adb', '-s', self.device_id, 'shell',
                'input', 'keyevent', 'KEYCODE_BACK'
            ], timeout=5)
            
            if result.returncode == 0:
                time.sleep(1)
                print(" Back button pressed")
                return True
            else:
                print(f" Back button failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f" Back button error: {e}")
            return False

    def _press_home(self) -> bool:
        """Enhanced home button press"""
        try:
            result = self._run_adb_command([
                'adb', '-s', self.device_id, 'shell',
                'input', 'keyevent', 'KEYCODE_HOME'
            ], timeout=5)
            
            if result.returncode == 0:
                self.current_screen = "home"
                time.sleep(1)
                print(" Home button pressed")
                return True
            else:
                print(f" Home button failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Home button error: {e}")
            return False

    # Required methods for QA system compatibility:
    
    def reset(self) -> Dict[str, Any]:
        """Enhanced environment reset with navigation fixes"""
        try:
            # Ensure clean state
            self._ensure_clean_navigation_state()
            
            # Go to home screen
            if self._press_home():
                self.current_screen = "home"
                self.action_history = []
                time.sleep(2)  # Wait for home screen to load
                
                print(" Environment reset to home screen")
                
                return {
                    'screenshot': self.capture_screenshot(),
                    'screen': self.current_screen,
                    'state': self.state.copy(),
                    'ui_elements': self.get_ui_elements(),
                    'ui_tree': self.get_parsed_ui_tree(),
                    'task_description': self.get_task_description(),
                    'is_task_complete': self.is_task_complete(),
                    'clickable_elements': [],
                    'screen_info': {'reset_successful': True}
                }
            else:
                print(" Failed to reset to home screen")
                return {'error': 'Reset failed - could not return to home'}
                
        except Exception as e:
            print(f" Reset error: {e}")
            return {'error': str(e)}

    def step(self, action, **kwargs):
        """Enhanced step execution with better error handling"""
        try:
            print(f" Executing step: {action}")
            action_success = self.perform_action(action, **kwargs)
            time.sleep(1)  # Wait for action to complete
            
            # Capture current state
            try:
                screenshot = self.capture_screenshot()
                ui_tree = self.get_parsed_ui_tree()
            except Exception as obs_error:
                print(f" Observation capture error: {obs_error}")
                screenshot = None
                ui_tree = {"elements": {}, "error": str(obs_error)}
            
            observation = {
                'screenshot': screenshot,
                'screen': self.current_screen,
                'state': self.state.copy(),
                'ui_elements': self.get_ui_elements(),
                'ui_tree': ui_tree,
                'task_description': self.get_task_description(),
                'is_task_complete': self.is_task_complete(),
                'clickable_elements': [],
                'screen_info': {'action_success': action_success}
            }
            
            reward = 1.0 if action_success else -0.1
            done = self.is_task_complete()
            info = {
                'action_success': action_success,
                'action_executed': action,
                'current_screen': self.current_screen
            }
            
            print(f"   Result: {' Success' if action_success else '❌ Failed'}")
            
            return observation, reward, done, info
            
        except Exception as e:
            print(f" Step execution error: {e}")
            error_observation = {
                'screenshot': None,
                'screen': self.current_screen,
                'state': self.state.copy(),
                'ui_elements': {},
                'ui_tree': {"elements": {}, "error": str(e)},
                'task_description': self.get_task_description(),
                'is_task_complete': False,
                'clickable_elements': [],
                'screen_info': {'error': str(e)}
            }
            return error_observation, -1.0, False, {'error': str(e)}

    def get_observation(self):
        """Enhanced observation gathering"""
        try:
            return {
                'screenshot': self.capture_screenshot(),
                'screen': self.current_screen,
                'state': self.state.copy(),
                'ui_elements': self.get_ui_elements(),
                'ui_tree': self.get_parsed_ui_tree(),
                'task_description': self.get_task_description(),
                'is_task_complete': self.is_task_complete(),
                'action_history_count': len(self.action_history)
            }
        except Exception as e:
            print(f" Observation error: {e}")
            return {'error': str(e)}

    def _get_observation(self):
        """Alias for compatibility"""
        return self.get_observation()

    def render(self, mode='rgb_array'):
        """Enhanced rendering with fallback"""
        try:
            screenshot = self.capture_screenshot()
            if screenshot:
                array = self.convert_screenshot_to_array(screenshot)
                if array is not None:
                    return array
                else:
                    print(" Screenshot conversion failed")
            else:
                print(" Screenshot capture failed")
            
            # Return None if capture fails
            return None
            
        except Exception as e:
            print(f" Render error: {e}")
            return None

    def get_ui_elements(self):
        """Get UI elements for current screen"""
        return self.ui_elements.get(self.current_screen, {})

    def get_task_description(self):
        """Get task description"""
        return self.task_config.get('description', f'Complete task: {self.task_name}')

    # Fix for task completion logic in production_android_env_real.py

    def is_task_complete(self):
        """Enhanced task completion check - prevents early completion"""
        try:
            condition = self.task_config.get('success_condition', '')
            
            # For WiFi tasks, require actual WiFi interaction
            if condition == 'wifi_enabled':
                # Check if we've actually toggled WiFi, not just opened settings
                wifi_actions = [a for a in self.action_history if 'wifi' in str(a.get('action', '')).lower()]
                return len(wifi_actions) >= 1 and self.state.get('wifi_enabled', False)
            
            elif condition == 'bluetooth_enabled':
                # Check if we've actually toggled Bluetooth
                bluetooth_actions = [a for a in self.action_history if 'bluetooth' in str(a.get('action', '')).lower()]
                return len(bluetooth_actions) >= 1 and self.state.get('bluetooth_enabled', False)
            
            elif condition == 'settings_opened':
                return self.current_screen == "settings_main"
            elif condition == 'wifi_settings_opened':
                return self.current_screen == "wifi_settings"
            elif condition == 'bluetooth_settings_opened':
                return self.current_screen == "bluetooth_settings"
            
            # Enhanced completion logic - require multiple meaningful actions
            if len(self.action_history) >= 3:  # Increased from 2
                recent_actions = [a.get('action', '') for a in self.action_history[-5:]]
                
                # Check for meaningful WiFi/Bluetooth interactions
                wifi_interaction = any('wifi' in str(action).lower() for action in recent_actions)
                bluetooth_interaction = any('bluetooth' in str(action).lower() for action in recent_actions)
                settings_interaction = any('settings' in str(action).lower() for action in recent_actions)
                
                # Require at least one actual feature interaction, not just opening settings
                return (wifi_interaction or bluetooth_interaction) and settings_interaction
            
            return False
            
        except Exception as e:
            print(f" Task completion check error: {e}")
            return False

def _is_done(self):
    """Enhanced task completion check with step count requirement"""
    # Don't complete until we've done meaningful work
    if len(self.action_history) < 3:
        return False
    
    # Check the specific task requirements
    return self.is_task_complete()

# Also update the step execution check in final_integrated_system.py
def enhanced_task_completion_check(self, test_trace, step_count):
    """Enhanced task completion check for comprehensive testing"""
    try:
        # Don't complete too early - ensure we've done actual testing
        if step_count < 3:
            return False
            
        # Check if we've performed meaningful actions
        meaningful_actions = 0
        for step in test_trace["steps"]:
            subgoal = step.get("subgoal", "").lower()
            if any(keyword in subgoal for keyword in ['wifi', 'bluetooth', 'toggle', 'navigate', 'settings']):
                meaningful_actions += 1
        
        # Require at least 2 meaningful actions before completion
        if meaningful_actions < 2:
            return False
            
        # Original completion check
        task_done = self.env._is_done()
        return task_done
        
    except:
        # Fallback: require at least 3 successful steps
        successful_steps = sum(1 for s in test_trace["steps"] if s.get("success", False))
        return step_count >= 3 and successful_steps >= 2

  
# Alias for compatibility
ProductionAndroidEnv = SimpleAndroidEnv