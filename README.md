# Email Triage Environment

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-00a393.svg)](https://fastapi.tiangolo.com)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Context-aware email triage benchmark for agent evaluation**  
> Built for Meta x Scalar Hackathon 2026

## 🎯 Overview

The **Email Triage Environment** is an OpenEnv-compatible benchmark that tests AI agents' ability to intelligently prioritize emails based on user context. Unlike simple spam classification, this environment requires agents to understand:

- **User persona** (Student, Professional, Business Owner)
- **Life context** (recent searches, upcoming events, active relationships, purchases)
- **Email relevance** and **priority** classification
- **Reasoning** for decisions (hard task)

### Three Difficulty Levels

| Task | Difficulty | Requirements | Reward Weight |
|------|-----------|--------------|---------------|
| **Relevance Check** | Easy | Binary classification (relevant/not_relevant) | 1.0 |
| **Priority Classification** | Medium | Relevance + priority (urgent/normal/low) | 0.6 + 0.4 |
| **Full Triage** | Hard | Relevance + priority + context-aware reasoning | 0.25 + 0.35 + 0.40 |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)
- OpenAI API key or Gemini API key

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/email-triage-env.git
cd email-triage-env

# Install with development dependencies
pip install -e ".[dev]"

# Or install from requirements.txt
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:
```env
# Option 1: OpenAI
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
HF_TOKEN=your_openai_api_key_here

# Option 2: Gemini (fallback)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL_NAME=gemini-2.5-flash
```

## 🎮 Usage

### 1. Start the API Server
```bash
# Local development
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Or using Docker
docker build -t email-triage-env .
docker run -p 8000:8000 email-triage-env
```

### 2. Test Health Endpoint
```bash
curl http://localhost:8000/health
# Response: {"status": "ok", "active_sessions": 0}
```

### 3. Run Agent Evaluation
```bash
python inference.py
```

This runs all three difficulty levels and outputs OpenEnv-compliant results:
```text
[START] task=relevance_check env=email-triage model=gpt-4o-mini
[STEP] step=1 action=relevant/null reward=1.00 done=false error=null
[STEP] step=2 action=not_relevant/null reward=1.00 done=false error=null
...
[END] success=true steps=8 score=0.875 rewards=1.00,1.00,0.00,1.00,1.00,1.00,0.00,1.00
```

### 4. API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/env/create` | POST | Create new environment session |
| `/env/{id}/reset` | POST | Reset environment, get first observation |
| `/env/{id}/step` | POST | Submit action, get observation + reward |
| `/env/{id}/state` | GET | Inspect current state |
| `/env/{id}/close` | POST | Close session |

#### Example API Flow
```bash
# 1. Create environment
SESSION_ID=$(curl -s -X POST http://localhost:8000/env/create \
  -H "Content-Type: application/json" \
  -d '{"task_level": "hard", "max_emails": 5}' | jq -r '.session_id')

# 2. Reset to get first observation
curl -X POST "http://localhost:8000/env/$SESSION_ID/reset"

# 3. Take a step
curl -X POST "http://localhost:8000/env/$SESSION_ID/step" \
  -H "Content-Type: application/json" \
  -d '{
    "relevance": "relevant",
    "priority": "urgent",
    "reason": "User searched google interview questions and has interview tomorrow"
  }'

# 4. Check state
curl "http://localhost:8000/env/$SESSION_ID/state"

# 5. Close session
curl -X POST "http://localhost:8000/env/$SESSION_ID/close"
```

## 🧪 Testing

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test class
pytest test_email_triage.py::TestEasyGrader -v

# Run linting
black . --check
ruff check .
mypy .
```

## 📊 Data Structure

### Personas (3 Types)
- **Student**: Job searching, interviews, exams, placements
- **Professional**: Work deadlines, EMI payments, performance reviews
- **Business Owner**: GST filing, supplier management, client meetings

### Context Signals
Each persona has:
- `recent_searches`: What the user recently searched for
- `recent_purchases`: Recent transactions
- `upcoming_events`: Calendar items (interviews, deadlines, meetings)
- `active_relationships`: Important contacts (recruiters, managers, suppliers)

### Email Labels
Every email in `data/personas.json` includes:
```json
{
  "subject": "Google Interview Confirmed for Tomorrow 10 AM",
  "body": "Your interview with Google is confirmed...",
  "label": "relevant",
  "priority": "urgent",
  "reason": "User searched google interview questions and has interview tomorrow"
}
```

## 🏆 Grading System

### EasyGrader (Relevance Only)
- Correct: 1.0
- Wrong: 0.0
- Invalid input: 0.0

### MediumGrader (Relevance + Priority)

| Scenario | Score |
|---|---|
| Both correct | 1.0 |
| Relevance correct, priority 1 step off | 0.6 |
| Relevance correct, priority 2 steps off | 0.4 |
| Relevance wrong, priority correct | 0.15 |
| Both wrong | 0.0 |

### HardGrader (Full Triage)

| Component | Weight | Criteria |
|---|---|---|
| Relevance | 0.25 | Exact match |
| Priority | 0.35 | Exact (0.35), 1-step (0.20), 2-step (0.10) |
| Reason Quality | 0.40 | Context keyword matching, specificity, justification |

Reason scoring checks if the agent cites:
- Recent searches that relate to the email (up to 0.40)
- Upcoming events or relationships (up to 0.30)
- Specific, non-generic language (up to 0.20)
- Alignment with relevance decision (up to 0.10)

## 🏗️ Architecture

```text
email-triage-env/
├── app.py                 # FastAPI server
├── email_env.py           # Environment logic (reset, step, state)
├── rewards.py             # Grading system (Easy/Medium/Hard)
├── model.py               # Pydantic models (Observation, Action, Reward)
├── inference.py           # LLM agent baseline
├── data/
│   └── personas.json      # 60 emails across 3 personas
├── tests/
│   ├── conftest.py        # Pytest configuration
│   └── test_email_triage.py  # Comprehensive test suite
├── openenv.yaml           # OpenEnv specification
├── docker.yaml            # Docker Compose config
├── Dockerfile             # Production container
├── pyproject.toml         # Modern Python packaging
└── requirements.txt       # Dependencies
```

## 🐳 Docker Deployment

```bash
# Build image
docker build -t email-triage-env .

# Run container
docker run -d \
  --name email-triage \
  -p 8000:8000 \
  --env-file .env \
  email-triage-env

# Using Docker Compose
docker-compose -f docker.yaml up -d
```

## 📈 Example Output

```text
==================================================
EMAIL TRIAGE BASELINE EVALUATION
==================================================

[START] task=relevance_check env=email-triage model=gpt-4o-mini
[STEP] step=1 action=relevant/null reward=1.00 done=false error=null
...
[END] success=true steps=8 score=0.625 rewards=1.00,1.00,0.00,1.00,1.00,1.00,0.00,1.00

[START] task=priority_classification env=email-triage model=gpt-4o-mini
...
[END] success=true steps=8 score=0.812 rewards=1.00,0.60,1.00,0.60,1.00,1.00,1.00,0.60

[START] task=context_aware_triage env=email-triage model=gpt-4o-mini
...
[END] success=true steps=8 score=0.743 rewards=0.84,0.70,0.62,0.88,0.75,0.78,0.65,0.72

==================================================
FINAL SCORES
==================================================
easy       → 0.625
medium     → 0.812
hard       → 0.743
AVERAGE    → 0.727
```

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License
MIT License - see LICENSE file for details.

## 🙏 Acknowledgments
Built for Meta x Scalar Hackathon 2026
OpenEnv benchmark framework by Scalar
FastAPI for the web framework
Pydantic for data validation
Made with ❤️ for the agent evaluation community
