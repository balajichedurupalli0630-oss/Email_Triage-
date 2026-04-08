"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""
import asyncio
import os
import json
import textwrap
from typing import List, Optional
import re
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
load_dotenv()
from email_env import EmailTriageEnv, Action

# Environment variables exactly as required by OpenEnv
API_BASE_URL = os.environ.get("API_BASE_URL", os.getenv("API_BASE_URL"))
MODEL_NAME = os.environ.get("MODEL_NAME", os.getenv("MODEL_NAME"))
HF_TOKEN = os.environ.get("HF_TOKEN", os.getenv("HF_TOKEN"))

# Fallbacks for Gemini if OpenEnv not fully set or user left placeholder token
if (not API_BASE_URL or "your_openai_or_hf_token_here" in str(HF_TOKEN)) and os.getenv(
    "GEMINI_API_KEY"
):
    API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    HF_TOKEN = os.getenv("GEMINI_API_KEY")
    if not MODEL_NAME or "gpt-4o-mini" in str(MODEL_NAME):
        MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

TASK_NAME = "email-triage"
BENCHMARK = "email-triage"
MAX_STEPS = 8
SUCCESS_SCORE_THRESHOLD = 0.6

# ── Semaphore — max 3 concurrent API calls ────────────────
semaphore = asyncio.Semaphore(3)

# ── Logging ───────────────────────────────────────────────
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True
    )

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True
    )

# ── System Prompts ────────────────────────────────────────
SYSTEM_PROMPTS = {
    "easy": textwrap.dedent("""
        You are an email triage agent.
        You will receive an email and user context.
        Decide if the email is relevant or not relevant to the user.
        
        Reply ONLY with this JSON:
        {"relevance": "relevant"}
        OR
        {"relevance": "not_relevant"}
        
        No extra text. JSON only.
    """).strip(),

    "medium": textwrap.dedent("""
        You are an email triage agent.
        You will receive an email and user context.
        Decide relevance AND priority.
        
        Reply ONLY with this JSON:
        {
          "relevance": "relevant" or "not_relevant",
          "priority": "urgent" or "normal" or "low"
        }
        
        No extra text. JSON only.
    """).strip(),

    "hard": textwrap.dedent("""
        You are a context-aware email triage agent.
        You will receive an email and user context.
        Use the context to decide relevance, priority and explain why.
        
        Reply ONLY with this JSON:
        {
          "relevance": "relevant" or "not_relevant",
          "priority": "urgent" or "normal" or "low",
          "reason": "explain using context keywords"
        }
        
        No extra text. JSON only.
    """).strip()
}

# ── Build User Prompt ─────────────────────────────────────
def build_prompt(obs) -> str:
    return textwrap.dedent(f"""
        PERSONA: {obs.persona}
        
        USER CONTEXT:
        {json.dumps(obs.context, indent=2)}
        
        EMAIL:
        Subject: {obs.subject}
        Body: {obs.body}
        
        Make your triage decision based on the user context.
    """).strip()

def extract_json(text: str) -> str:
    """Robust JSON extraction from LLM response."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"(\{.*?\})", text, re.DOTALL)
    if match:
        return match.group(1)
    return text

@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    reraise=True,
)
async def _call_api(client: AsyncOpenAI, prompt: str, task_level: str) -> str:
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS[task_level]},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=150,
    )
    return response.choices[0].message.content or "{}"

async def ask_llm_safe(client: AsyncOpenAI, obs, task_level: str) -> Action:
    async with semaphore:
        try:
            prompt = build_prompt(obs)
            response_text = await _call_api(client, prompt, task_level)
            json_str = extract_json(response_text)
            data = json.loads(json_str)
            return Action(
                relevance=data.get("relevance", "not_relevant"),
                priority=data.get("priority", None),
                reason=data.get("reason", None)
            )
        except Exception as e:
            print(f"[ERROR] LLM task failed: {e}", flush=True)
            return Action(relevance="not_relevant")

# ── Run One Task ──────────────────────────────────────────
async def run_task(client, task_level: str) -> float:
    task_name = {
        "easy":   "relevance_check",
        "medium": "priority_classification",
        "hard":   "context_aware_triage"
    }[task_level]

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    env = await EmailTriageEnv.from_env(
        task_level=task_level,
        max_emails=MAX_STEPS
    )

    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        obs = await env.reset()

        for step in range(1, MAX_STEPS + 1):

            # ask LLM with semaphore + retry
            action = await ask_llm_safe(client, obs, task_level)

            obs, reward, done, info = await env.step(action)

            rewards.append(reward.value)
            steps_taken = step

            log_step(
                step=step,
                action=f"{action.relevance}/{action.priority}",
                reward=reward.value,
                done=done,
                error=None
            )

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        await env.close()
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards
        )

    return score

# ── Main — Run All 3 Tasks in Parallel ───────────────────
async def main():
    client = AsyncOpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    print("\n" + "="*50)
    print("EMAIL TRIAGE BASELINE EVALUATION")
    print("="*50 + "\n")

    # Run sequentially to avoid rate limits
    results = []
    for level in ["easy", "medium", "hard"]:
        results.append(await run_task(client, level))

    scores = dict(zip(["easy", "medium", "hard"], results))

    print("\n" + "="*50)
    print("FINAL SCORES")
    print("="*50)
    for task, score in scores.items():
        print(f"{task:10} → {score:.3f}")
    print(f"{'AVERAGE':10} → {sum(scores.values())/3:.3f}")

if __name__ == "__main__":
    asyncio.run(main())