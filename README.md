---
title: AI Email Agent
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---
# 🤖 AI Email Agent — Interactive Training & Simulation

> A smart AI-powered platform that simulates how intelligent agents read, classify, and act on emails — with step-by-step execution and transparent scoring.

---

## 🚀 Overview

Most email systems stop at classification.

This project goes further by simulating a **complete decision-making workflow**, where an AI agent:

- Reads emails
- Classifies them
- Decides actions (reply / archive / escalate)
- Processes emails sequentially
- Produces a final performance score

Unlike traditional systems, every step is **visible and measurable**, making it easier to understand how decisions are made.

---

## 🎯 What This Project Demonstrates

- Sequential decision-making in AI systems
- How agents move beyond prediction to action
- The importance of workflow order and efficiency
- The role of fallback systems in real-world reliability

---

## ✨ Key Features

### 🎯 Training Mode (Interactive Setup)

- Select inbox size:
  - Starter → 5 emails
  - Medium → 10 emails
  - Advanced → 15 emails

- Preview all emails before execution

- Understand the dataset before running the agent

---

### ▶️ Step-by-Step Execution (Core Highlight)

Run the simulation and observe:

- Emails processed one-by-one
- For each email:
  - Subject
  - Classification
  - Action taken
  - Reward earned

This ensures **complete transparency of agent behavior**.

---

### 🔍 Analyze Email (Custom Input)

- Input any custom email
- System returns:
  - Predicted email type
  - Suggested action

System behavior:

- Uses Gemini API (primary)
- Falls back to internal agent if unavailable

---

### 🤖 Hybrid AI System

#### 🟢 Fallback Agent (Default)

- Fast and reliable
- No external API required
- Handles the majority of emails

---

#### 🔵 Gemini API (Advanced Layer)

- Used selectively for:
  - Complex inputs
  - High-uncertainty cases
  - Response generation

---

#### 🔁 Automatic Fallback (Reliability Layer)

If API fails due to:

- Rate limits
- Invalid key
- Timeout

👉 System automatically switches to fallback
👉 Execution continues without interruption

---

### ⚡ Performance-Focused Design

- Fast execution
- Minimal latency
- Efficient API usage

---

### 🎨 UI Experience

- Dark-themed interface
- Clean and readable layout
- Designed for clarity during step-by-step execution

---

## 🧠 How It Works

### Step 1

User selects inbox size

### Step 2

System displays all emails

### Step 3

User runs the simulation

### Step 4

Agent processes emails sequentially:

```id="flow_block"
Email → Classification → Action → Reward
```

### Step 5

Final score is calculated and displayed

---

## 📊 Scoring System

Each run is evaluated across the following dimensions:

| Metric         | Description              |
| -------------- | ------------------------ |
| Classification | Accuracy of labels       |
| Action         | Correct decision-making  |
| Workflow       | Logical order of actions |
| Efficiency     | Optimal step usage       |

---

## ⚙️ Run Locally

```bash id="setup_1"
git clone https://github.com/your-username/ai-email-agent.git
cd ai-email-agent
```

```bash id="setup_2"
pip install -r requirements.txt
```

```bash id="setup_3"
# Optional (for advanced AI features)
set GEMINI_API_KEY=your_key_here
```

```bash id="setup_4"
streamlit run app.py
```

---

## 🧪 CLI Mode

Run without UI:

```bash id="cli_mode"
python inference.py
```

---

## 🐳 Docker

```bash id="docker_block"
docker build -t ai-email-agent .
docker run -p 8501:8501 ai-email-agent
```

---

## 🌐 Deployment (Hugging Face)

- Create a **Streamlit Space**
- Push this repository
- Ensure `requirements.txt` is included

Optional:

- Add `GEMINI_API_KEY` in repository secrets

---

## 📂 Project Structure

```id="project_structure"
EmailEnv/
├── app.py
├── inference.py
├── agent.py
├── models/
├── tests/
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🔧 Core Components Explained

- **app.py**
  Handles the Streamlit UI, user interaction, and visualization of the simulation.

- **agent.py**
  Implements the core logic for classification and action selection, including fallback and API integration.

- **inference.py**
  Runs the simulation in CLI mode without UI.

- **models/**
  Contains data structures and representations used across the system.

- **tests/**
  Includes test cases to validate system functionality.

---

## 🎯 What Makes This Project Stand Out

- Moves beyond classification → full workflow simulation
- Step-by-step explainable AI system
- Hybrid architecture (API + fallback)
- Interactive and beginner-friendly
- Designed for reliability and real-world behavior

---

## 🧠 Learnings

- Fallback systems are essential for robust AI applications
- Not all problems require heavy AI models
- Transparency improves trust in AI systems
- Workflow design is as important as prediction accuracy

---

## 👨‍💻 Author

**Sumanth Mamidi**

---

## 📜 License

MIT License
