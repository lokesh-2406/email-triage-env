---
title: Email Triage OpenEnv
emoji: 📬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: true
tags:
  - openenv
---
# 📬 Email Triage OpenEnv

An OpenEnv environment simulating a **real-world email triage workflow**. An AI agent must classify, prioritize, and route incoming emails — tasks that every knowledge worker does daily.

---

## 🎯 Motivation

Email triage is a genuinely hard, multi-faceted real-world task:
- It requires reading comprehension, semantic understanding, and policy adherence
- It has clear, programmatic success criteria
- It spans a natural difficulty range (single classification → multi-criteria routing)
- It's a core productivity bottleneck in every organization

---

## 🗂️ Project Structure

```
email-triage-env/
├── inference.py              # ✅ Hackathon inference script (root-level, mandatory)
├── models.py                 # Pydantic Action / Observation / State models
├── client.py                 # HTTP client for the environment
├── openenv.yaml              # OpenEnv manifest
├── pyproject.toml            # Dependencies
├── __init__.py
├── server/
│   ├── app.py                # FastAPI server
│   ├── environment.py        # Core environment logic + graders
│   ├── Dockerfile            # Container definition
│   └── requirements.txt      # Python dependencies
└── tests/
    └── test_environment.py   # Pytest unit tests
```

---

## 🎮 Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `classify_email` | 🟢 Easy | Classify a single email into one of 5 categories |
| `prioritize_inbox` | 🟡 Medium | Rank 5 emails by urgency using Kendall-tau scoring |
| `route_and_respond` | 🔴 Hard | Route to correct department AND draft acknowledgment reply |

---

## 📐 Action Space

```python
EmailTriageAction(
    category:     Optional[str]       # spam | urgent | billing | support | general
    priority:     Optional[List[str]] # Ordered list of email IDs (highest to lowest)
    department:   Optional[str]       # sales | engineering | billing | hr | support
    reply_draft:  Optional[str]       # 1-3 sentence acknowledgment reply
)
```

Only set the fields relevant to the current task.

---

## 👁️ Observation Space

```python
EmailTriageObservation(
    task:         str           # Active task name
    emails:       List[Email]   # Email(s) to process
    instructions: str           # Human-readable task instructions
    reward:       float         # Reward from last action [0.0, 1.0]
    feedback:     str           # Grader feedback string
    done:         bool          # Episode complete?
)
```

---

## 📊 Reward Functions

### classify_email (easy)
- **1.0** — Correct category
- **0.05–0.3** — Partial credit for semantically adjacent categories (e.g., urgent↔support)
- **0.0** — Wrong or missing category

### prioritize_inbox (medium)
- Scored using **Kendall-tau concordance** on the full ranking
- **1.0** — Perfect order
- **~0.6** — Mostly correct, 1-2 swaps
- **0.0** — Missing IDs or no priority given

### route_and_respond (hard)
- **0.0–0.5** — Department routing (binary: correct = 0.5)
- **0.0–0.5** — Reply quality (keyword coverage + length check)
- **Total: 0.0–1.0**

---

## 🚀 Quick Start

### 1. Run the server locally

```bash
cd server
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

### 2. Test the API

```bash
# Health check
curl http://localhost:7860/health

# Reset (start classify task)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "classify_email"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"category": "urgent"}}'
```

### 3. Run tests

```bash
pip install pytest
pytest tests/ -v
```

### 4. Run inference script

```bash
# Get a free Groq API key at https://console.groq.com
export HF_TOKEN=your_groq_key_here
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.1-8b-instant

# Server must be running on port 7860
python inference.py
```

---

## 🐳 Docker

```bash
# Build
docker build -f server/Dockerfile -t email-triage-env .

# Run
docker run -p 7860:7860 email-triage-env

# Test it
curl http://localhost:7860/health
```

---

## 📈 Baseline Scores

Tested with `llama-3.1-8b-instant` via Groq:

| Task | Score |
|------|-------|
| classify_email | ~0.72 |
| prioritize_inbox | ~0.58 |
| route_and_respond | ~0.51 |
| **Average** | **~0.60** |

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | **Required.** Groq/HF API key |
| `API_BASE_URL` | `https://api.groq.com/openai/v1` | LLM endpoint |
| `MODEL_NAME` | `llama-3.1-8b-instant` | Model identifier |
| `EMAIL_TRIAGE_TASK` | `classify_email` | Default task for the server |
| `EMAIL_TRIAGE_URL` | `http://localhost:7860` | Server URL for inference script |

