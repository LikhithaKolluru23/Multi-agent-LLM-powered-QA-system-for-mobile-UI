# Multi-Agent LLM QA System for Mobile UI Task Execution

## Overview
This project implements a **multi-agent Large Language Model (LLM)** system designed to plan, execute, and verify user-requested tasks in **mobile environments**.  
It leverages modular AI agents — **PlannerAgent**, **ExecutorAgent**, **VerifierAgent**, and **SupervisorAgent** — to collaboratively interact with mobile UIs, execute subgoals, and validate results.

The system can operate in:
- **AndroidEnv** for real device/task interaction.
- **MockAndroidEnv** for lightweight local testing.
- **Evaluation mode** using **Android in the Wild** dataset episodes.

---

## Architecture

### Agent Roles
- **PlannerAgent**  
  Decomposes natural language tasks into ordered subgoals with reasoning traces.
  
- **ExecutorAgent**  
  Executes subgoals step-by-step in AndroidEnv/MockAndroidEnv, collecting UI state information.
  
- **VerifierAgent**  
  Validates whether each subgoal’s outcome matches the expected result.
  
- **SupervisorAgent**  
  Performs end-to-end review, generates execution summaries, detects issues, and suggests improvements.  
  Provides **efficiency** and **robustness** scores for evaluation.

---

## Key Features
- **Natural Language to Actionable Plan** — LLM-powered planning with explicit subgoal dependencies.
- **Environment Abstraction** — Swap between MockAndroidEnv (fast simulation) and AndroidEnv (realistic UI interaction).
- **Verification Layer** — Detects failed actions or mismatches early.
- **Evaluation on Real UI Traces** — Supports Android in the Wild dataset for reproducible benchmarks.
- **Metrics & Analysis** —  
  - Execution success rate  
  - Step efficiency score  
  - Robustness under noisy/ambiguous prompts  
  - LLM-based Gemini-style qualitative analysis

---

# 1. Clone the Repository
git clone https://github.com/your-username/multiagent-llm-qa.git
cd multiagent-llm-qa

### Android SDK & Emulator Setup (via Android Studio)

If you plan to run the project in **AndroidEnv** (realistic mobile environment), you’ll need the **Android SDK** and an emulator.  
We recommend installing these through **Android Studio** for a graphical setup.

#### 1. Install Android Studio
- Download Android Studio from: https://developer.android.com/studio
- Follow the installation steps for your OS (Windows, macOS, or Linux).
- During setup, make sure to select:
  - **Android SDK**
  - **Android SDK Platform**
  - **Android Virtual Device**

#### 2. Open Virtual Device Manager
- Launch **Android Studio**.
- Go to **Tools → Device Manager** (or press `Shift` twice and type "Device Manager").
- Click **Create Device**.
- Choose **Pixel 6** from the device list.
- Select a **System Image** (e.g., `Android 13.0 Google APIs`).
- Click **Download** if the image is not already installed.
- Complete the wizard and create the virtual device.

#### 3. Start the Pixel 6 Emulator
- In the **Device Manager**, find your Pixel 6 entry.
- Click the **▶** (Play) button to launch the emulator.
- Wait for the emulator to fully boot before running the app.

#### 4. Run in AndroidEnv Mode
Once the emulator is running:
```bash
python final_integated_system.py





