import sys
import asyncio
import json
import os
import random
from typing import List, Optional
from pydantic import BaseModel, Field
from rewards import get_grader
from model import Observation, Action, Reward

# ── Environment ──────────────────────────────────────────
class EmailTriageEnv:

    def __init__(self, task_level: str = "easy", max_emails: int = 10):
        
        self.task_level = task_level
        self.max_emails = max_emails
        self.current_step = 0
        self.episode_rewards = []
        self.current_email = None
        self.current_persona = None
        self.email_data = {}
        self.persona_list = ["Student", "Professional", "Business Owner"]
        self.email_queue = []
        self.grader = get_grader(self.task_level)
    async def initialize(self):
        print("[DEBUG] Initializing EmailTriageEnv...", file=sys.stderr)
        try:
            # Support running from project root OR from the directory containing email_env.py
            _here = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(_here, "data", "personas.json")
            if not os.path.exists(data_path):
                data_path = os.path.join(os.getcwd(), "data", "personas.json")
            with open(data_path, "r", encoding="utf-8") as f:
                self.email_data = json.load(f)
            print(f"[INFO] Loaded personas: {list(self.email_data.keys())}", file=sys.stderr)
        except FileNotFoundError:
            print("[WARN] personas.json not found", file=sys.stderr)
            self.email_data = {}

    async def reset(self) -> Observation:
        self.current_step = 0
        self.episode_rewards = []
        self.current_persona = random.choice(self.persona_list)

        emails = self.email_data.get(self.current_persona, {}).get("emails", [])
        shuffled = emails.copy()
        episode_seed = random.randint(0, 10**6)
        random.seed(episode_seed)
        random.shuffle(shuffled)
        self.email_queue = (shuffled * ((self.max_emails // len(shuffled)) + 2))[:self.max_emails]
        self.current_email = self.email_queue[0] if self.email_queue else None

        return self._make_observation()

    async def step(self, action: Action) -> tuple:
        self.current_step += 1
        reward_value = self._grade(action)
        self.episode_rewards.append(reward_value)

        
        if self.current_step < len(self.email_queue):
            self.current_email = self.email_queue[self.current_step]
        else:
            self.current_email = None

        done = self.current_step >= self.max_emails

        obs = self._make_observation()
        info = {
            "step": self.current_step,
            "cumulative_reward": sum(self.episode_rewards),
            "task_level": self.task_level
        }

        return obs, Reward(value=reward_value), done, info
    
    def _grade(self, action: Action) -> float:

        if not self.current_email:
            return 0.0
        context = self.email_data.get(self.current_persona, {}).get("context", {})
        agent_response = {
            "relevance": action.relevance,
            "priority":  action.priority,
            "reason":    action.reason,
        }
        return self.grader.score(agent_response, self.current_email, context)
    

    def _make_observation(self) -> Observation:
        email = self.current_email or {}
        context = self.email_data.get(
            self.current_persona, {}).get("context", {})
        return Observation(
            email_id=f"email_{self.current_step}",
            subject=email.get("subject", ""),
            body=email.get("body", ""),
            persona=self.current_persona or "",
            context=context,
            task_level=self.task_level,
            step=self.current_step
        )

    async def state(self) -> dict:
        return {
            "step": self.current_step,
            "max_steps": self.max_emails,
            "persona": self.current_persona,
            "task_level": self.task_level,
            "episode_rewards": self.episode_rewards,
            "cumulative_reward": sum(self.episode_rewards)
        }

    async def close(self):
        pass

    @classmethod
    async def from_env(cls, **kwargs):
        env = cls(**kwargs)
        await env.initialize()
        return env


if __name__ == "__main__":
    async def test():
        env = await EmailTriageEnv.from_env(task_level="hard")
        obs = await env.reset()
        print(f"Reset: {obs.persona}, {obs.subject[:50]}")

        action = Action(
            relevance="relevant",
            priority="urgent",
            reason="user searched google interview questions"
        )
        obs, reward, done, info = await env.step(action)
        print(f"Step 1: reward={reward.value}, done={done}")

        await env.close()

    asyncio.run(test())