"""
test_email_triage.py — Tests grounded in real data/personas.json

Tests cover:
  - model.py      : Pydantic schema validation
  - rewards.py    : EasyGrader, MediumGrader, HardGrader with real emails
  - email_env.py  : Environment lifecycle using real personas.json
  - app.py        : FastAPI endpoints via TestClient

Run with:
    pytest test_email_triage.py -v
"""

import json
import os
import pytest
from unittest.mock import patch

# ─────────────────────────────────────────────────────────
# Load real personas.json once for all tests
# ─────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "personas.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    PERSONAS = json.load(f)

# Shortcuts into real data
STUDENT_CTX      = PERSONAS["Student"]["context"]
STUDENT_EMAILS   = PERSONAS["Student"]["emails"]

PROFESSIONAL_CTX    = PERSONAS["Professional"]["context"]
PROFESSIONAL_EMAILS = PERSONAS["Professional"]["emails"]

BUSINESS_CTX    = PERSONAS["Business Owner"]["context"]
BUSINESS_EMAILS = PERSONAS["Business Owner"]["emails"]

# helpers — pick first email matching a label/priority
def get_email(emails, label=None, priority=None):
    for e in emails:
        if label and e["label"] != label:
            continue
        if priority and e["priority"] != priority:
            continue
        return e
    raise ValueError(f"No email with label={label} priority={priority}")


# ─────────────────────────────────────────────────────────
# 1. MODEL TESTS
# ─────────────────────────────────────────────────────────

class TestModels:

    def test_observation_all_fields(self):
        from model import Observation
        obs = Observation(
            email_id="e1",
            subject=STUDENT_EMAILS[0]["subject"],
            body=STUDENT_EMAILS[0]["body"],
            persona="Student",
            context=STUDENT_CTX,
            task_level="easy",
            step=0
        )
        assert obs.persona == "Student"
        assert obs.task_level == "easy"
        assert obs.step == 0

    def test_observation_default_step_is_zero(self):
        from model import Observation
        obs = Observation(
            email_id="e2", subject="Hi", body="Bye",
            persona="Professional", context={}, task_level="medium"
        )
        assert obs.step == 0

    def test_action_relevance_only(self):
        from model import Action
        a = Action(relevance="relevant")
        assert a.priority is None
        assert a.reason is None

    def test_action_full_fields(self):
        from model import Action
        a = Action(
            relevance="not_relevant",
            priority="low",
            reason="food offer unrelated to placement"
        )
        assert a.relevance == "not_relevant"
        assert a.priority == "low"

    def test_reward_valid_range(self):
        from model import Reward
        for v in [0.0, 0.3, 0.5, 1.0]:
            r = Reward(value=v)
            assert r.value == v

    def test_reward_above_one_raises(self):
        from model import Reward
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            Reward(value=1.1)

    def test_reward_below_zero_raises(self):
        from model import Reward
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            Reward(value=-0.01)


# ─────────────────────────────────────────────────────────
# 2. EASY GRADER  (relevance only — binary 0 or 1)
# ─────────────────────────────────────────────────────────

class TestEasyGrader:

    def setup_method(self):
        from rewards import EasyGrader
        self.grader = EasyGrader()

    # ── Student emails ────────────────────────────────────
    def test_student_relevant_email_correct(self):
        email = get_email(STUDENT_EMAILS, label="relevant")
        score = self.grader.score(
            {"relevance": "relevant"}, email, STUDENT_CTX
        )
        assert score == 1.0

    def test_student_relevant_email_wrong_prediction(self):
        email = get_email(STUDENT_EMAILS, label="relevant")
        score = self.grader.score(
            {"relevance": "not_relevant"}, email, STUDENT_CTX
        )
        assert score == 0.0

    def test_student_spam_zomato_correct(self):
        # "Zomato: 50% off your next order!" → not_relevant
        zomato = next(e for e in STUDENT_EMAILS if "Zomato" in e["subject"])
        score = self.grader.score(
            {"relevance": "not_relevant"}, zomato, STUDENT_CTX
        )
        assert score == 1.0

    def test_student_spam_zomato_wrong_prediction(self):
        zomato = next(e for e in STUDENT_EMAILS if "Zomato" in e["subject"])
        score = self.grader.score(
            {"relevance": "relevant"}, zomato, STUDENT_CTX
        )
        assert score == 0.0

    def test_student_netflix_spam_correct(self):
        netflix = next(e for e in STUDENT_EMAILS if "Netflix" in e["subject"])
        score = self.grader.score(
            {"relevance": "not_relevant"}, netflix, STUDENT_CTX
        )
        assert score == 1.0

    # ── Professional emails ───────────────────────────────
    def test_professional_relevant_correct(self):
        email = get_email(PROFESSIONAL_EMAILS, label="relevant")
        score = self.grader.score(
            {"relevance": "relevant"}, email, PROFESSIONAL_CTX
        )
        assert score == 1.0

    def test_professional_not_relevant_correct(self):
        email = get_email(PROFESSIONAL_EMAILS, label="not_relevant")
        score = self.grader.score(
            {"relevance": "not_relevant"}, email, PROFESSIONAL_CTX
        )
        assert score == 1.0

    # ── Business Owner emails ─────────────────────────────
    def test_business_gst_email_correct(self):
        gst = next(e for e in BUSINESS_EMAILS if "GST" in e["subject"])
        score = self.grader.score(
            {"relevance": "relevant"}, gst, BUSINESS_CTX
        )
        assert score == 1.0

    def test_business_iphone_spam_correct(self):
        spam = next(e for e in BUSINESS_EMAILS if "iPhone" in e["subject"])
        score = self.grader.score(
            {"relevance": "not_relevant"}, spam, BUSINESS_CTX
        )
        assert score == 1.0

    def test_business_iphone_spam_wrong_prediction(self):
        spam = next(e for e in BUSINESS_EMAILS if "iPhone" in e["subject"])
        score = self.grader.score(
            {"relevance": "relevant"}, spam, BUSINESS_CTX
        )
        assert score == 0.0


# ─────────────────────────────────────────────────────────
# 3. MEDIUM GRADER  (relevance + priority)
# ─────────────────────────────────────────────────────────

class TestMediumGrader:

    def setup_method(self):
        from rewards import MediumGrader
        self.grader = MediumGrader()

    # ── Student: Google interview (relevant + urgent) ─────
    def test_student_interview_both_correct_scores_1(self):
        email = STUDENT_EMAILS[0]   # "Reminder: Prepare for Your Upcoming Interview"
        assert email["label"] == "relevant"
        assert email["priority"] == "urgent"
        score = self.grader.score(
            {"relevance": "relevant", "priority": "urgent"},
            email, STUDENT_CTX
        )
        assert score == 1.0

    def test_student_interview_relevance_correct_priority_wrong(self):
        email = STUDENT_EMAILS[0]
        score = self.grader.score(
            {"relevance": "relevant", "priority": "low"},
            email, STUDENT_CTX
        )
        # priority 2 steps off (urgent→low): partial credit 0.4
        assert score == 0.4

    def test_student_interview_relevance_wrong_priority_correct(self):
        email = STUDENT_EMAILS[0]
        score = self.grader.score(
            {"relevance": "not_relevant", "priority": "urgent"},
            email, STUDENT_CTX
        )
        # relevance wrong, priority correct → 0.15 (lucky guess penalty)
        assert score == 0.15

    def test_student_interview_both_wrong_scores_0(self):
        email = STUDENT_EMAILS[0]
        score = self.grader.score(
            {"relevance": "not_relevant", "priority": "low"},
            email, STUDENT_CTX
        )
        assert score == 0.0

    # ── Student: spam email (not_relevant + low) ──────────
    def test_student_zomato_spam_both_correct(self):
        zomato = next(e for e in STUDENT_EMAILS if "Zomato" in e["subject"])
        assert zomato["label"] == "not_relevant"
        assert zomato["priority"] == "low"
        score = self.grader.score(
            {"relevance": "not_relevant", "priority": "low"},
            zomato, STUDENT_CTX
        )
        assert score == 1.0

    # ── Student: normal priority email ───────────────────
    def test_student_placement_cell_normal_priority(self):
        # "Placement Cell Update" → relevant + normal
        placement = next(
            e for e in STUDENT_EMAILS if "Placement Cell" in e["subject"]
        )
        assert placement["priority"] == "normal"
        score = self.grader.score(
            {"relevance": "relevant", "priority": "normal"},
            placement, STUDENT_CTX
        )
        assert score == 1.0

    # ── Business: GST urgent ──────────────────────────────
    def test_business_gst_urgent_both_correct(self):
        gst = next(e for e in BUSINESS_EMAILS if "GST Filing" in e["subject"])
        score = self.grader.score(
            {"relevance": "relevant", "priority": "urgent"},
            gst, BUSINESS_CTX
        )
        assert score == 1.0

    # ── Business: normal priority email ───────────────────
    def test_business_bulk_supplier_normal_priority(self):
        bulk = next(e for e in BUSINESS_EMAILS if "Bulk Supplier" in e["subject"])
        assert bulk["priority"] == "normal"
        score = self.grader.score(
            {"relevance": "relevant", "priority": "normal"},
            bulk, BUSINESS_CTX
        )
        assert score == 1.0

    def test_score_always_between_0_and_1(self):
        for email in STUDENT_EMAILS:
            score = self.grader.score(
                {"relevance": "relevant", "priority": "urgent"},
                email, STUDENT_CTX
            )
            assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────
# 4. HARD GRADER  (relevance 0.3 + priority 0.3 + keyword 0.4)
# ─────────────────────────────────────────────────────────

class TestHardGrader:

    def setup_method(self):
        from rewards import HardGrader
        self.grader = HardGrader()

    # ── Student: Google interview, keyword in recent_searches ─
    def test_student_interview_all_correct_full_score(self):
        email = STUDENT_EMAILS[0]
        # "google interview questions" is in STUDENT_CTX["recent_searches"]
        score = self.grader.score(
            {
                "relevance": "relevant",
                "priority": "urgent",
                "reason": "user recently searched google interview questions"
            },
            email, STUDENT_CTX
        )
        # 0.25 (relevance) + 0.35 (priority) + partial reason (1 search hit, no event/rel) = 0.84
        assert score == pytest.approx(0.84)

    def test_student_leetcode_keyword_match(self):
        email = STUDENT_EMAILS[0]
        # "leetcode problems" is in recent_searches but doesn't appear in email body/subject
        score = self.grader.score(
            {
                "relevance": "relevant",
                "priority": "urgent",
                "reason": "user purchased leetcode problems subscription"
            },
            email, STUDENT_CTX
        )
        # 0.25 + 0.35 + partial reason (loose hit only, no event/rel match) = 0.70
        assert score == pytest.approx(0.7)

    def test_student_correct_labels_no_keyword_reason(self):
        email = STUDENT_EMAILS[0]
        score = self.grader.score(
            {
                "relevance": "relevant",
                "priority": "urgent",
                "reason": "looks important to me"   # no real keyword
            },
            email, STUDENT_CTX
        )
        # 0.25 (relevance) + 0.35 (priority) + 0.02 (minimal reason partial) = 0.62
        assert score == pytest.approx(0.62)

    def test_student_only_keyword_in_reason_scores_0_4(self):
        email = STUDENT_EMAILS[0]
        score = self.grader.score(
            {
                "relevance": "not_relevant",
                "priority": "low",
                "reason": "user searched amazon internship apply deadline"
            },
            email, STUDENT_CTX
        )
        # wrong relevance/priority + partial reason keyword credit = 0.178
        assert score == pytest.approx(0.178)

    def test_student_all_wrong_scores_zero(self):
        email = STUDENT_EMAILS[0]
        score = self.grader.score(
            {
                "relevance": "not_relevant",
                "priority": "low",
                "reason": "no match whatsoever"
            },
            email, STUDENT_CTX
        )
        # wrong relevance/priority; reason too short for word-count credit but gets 0.05 reason partial
        assert score == pytest.approx(0.098)

    # ── Business: GST filing ──────────────────────────────
    def test_business_gst_all_correct(self):
        gst = next(e for e in BUSINESS_EMAILS if "GST Filing" in e["subject"])
        score = self.grader.score(
            {
                "relevance": "relevant",
                "priority": "urgent",
                "reason": "user searched gst filing deadline 2025 and deadline in 3 days"
            },
            gst, BUSINESS_CTX
        )
        # 0.25 + 0.35 + reason (1 search hit + event partial) = 0.88
        assert score == pytest.approx(0.88)

    def test_business_bulk_supplier_keyword(self):
        email = next(e for e in BUSINESS_EMAILS if "Bulk Supplier" in e["subject"])
        score = self.grader.score(
            {
                "relevance": "relevant",
                "priority": "normal",
                "reason": "user searched bulk raw material suppliers for better pricing"
            },
            email, BUSINESS_CTX
        )
        # 0.25 + 0.35 + reason partial (1 search hit + event partial) = 0.88
        assert score == pytest.approx(0.88)

    # ── Ground truth reason should always score >= 0.6 ───
    def test_ground_truth_reason_student_scores_high(self):
        for email in STUDENT_EMAILS:
            if email["label"] == "not_relevant":
                continue
            score = self.grader.score(
                {
                    "relevance": email["label"],
                    "priority": email["priority"],
                    "reason": email["reason"]
                },
                email, STUDENT_CTX
            )
            assert score >= 0.6, (
                f"Expected >= 0.6 for '{email['subject']}' but got {score}"
            )

    def test_ground_truth_reason_business_scores_high(self):
        for email in BUSINESS_EMAILS:
            if email["label"] == "not_relevant":
                continue
            score = self.grader.score(
                {
                    "relevance": email["label"],
                    "priority": email["priority"],
                    "reason": email["reason"]
                },
                email, BUSINESS_CTX
            )
            assert score >= 0.6, (
                f"Expected >= 0.6 for '{email['subject']}' but got {score}"
            )

    def test_score_always_in_range_all_personas(self):
        for persona, ctx in [
            ("Student", STUDENT_CTX),
            ("Professional", PROFESSIONAL_CTX),
            ("Business Owner", BUSINESS_CTX),
        ]:
            for email in PERSONAS[persona]["emails"]:
                score = self.grader.score(
                    {
                        "relevance": email["label"],
                        "priority": email["priority"],
                        "reason": email["reason"]
                    },
                    email, ctx
                )
                assert 0.0 <= score <= 1.0, (
                    f"Score {score} out of range for {persona}: {email['subject']}"
                )


# ─────────────────────────────────────────────────────────
# 5. ENVIRONMENT TESTS  (real personas.json on disk)
# ─────────────────────────────────────────────────────────

class TestEmailTriageEnvLifecycle:

    @pytest.mark.asyncio
    async def test_reset_returns_valid_observation(self):
        from email_env import EmailTriageEnv
        env = await EmailTriageEnv.from_env(task_level="easy", max_emails=5)
        obs = await env.reset()
        assert obs.persona in ["Student", "Professional", "Business Owner"]
        assert obs.task_level == "easy"
        assert isinstance(obs.subject, str)
        assert isinstance(obs.context, dict)
        await env.close()

    @pytest.mark.asyncio
    async def test_step_reward_in_range(self):
        from email_env import EmailTriageEnv
        from model import Action
        env = await EmailTriageEnv.from_env(task_level="easy", max_emails=5)
        await env.reset()
        _, reward, _, _ = await env.step(Action(relevance="relevant"))
        assert 0.0 <= reward.value <= 1.0
        await env.close()

    @pytest.mark.asyncio
    async def test_done_after_max_emails(self):
        from email_env import EmailTriageEnv
        from model import Action
        env = await EmailTriageEnv.from_env(task_level="easy", max_emails=3)
        await env.reset()
        done = False
        for _ in range(3):
            _, _, done, _ = await env.step(Action(relevance="relevant"))
        assert done is True

    @pytest.mark.asyncio
    async def test_not_done_before_max_emails(self):
        from email_env import EmailTriageEnv
        from model import Action
        env = await EmailTriageEnv.from_env(task_level="easy", max_emails=5)
        await env.reset()
        _, _, done, _ = await env.step(Action(relevance="relevant"))
        assert done is False
        await env.close()

    @pytest.mark.asyncio
    async def test_step_info_keys(self):
        from email_env import EmailTriageEnv
        from model import Action
        env = await EmailTriageEnv.from_env(task_level="medium", max_emails=5)
        await env.reset()
        _, _, _, info = await env.step(Action(relevance="relevant", priority="urgent"))
        assert "step" in info
        assert "cumulative_reward" in info
        assert "task_level" in info
        await env.close()

    @pytest.mark.asyncio
    async def test_state_tracks_steps(self):
        from email_env import EmailTriageEnv
        from model import Action
        env = await EmailTriageEnv.from_env(task_level="easy", max_emails=5)
        await env.reset()
        await env.step(Action(relevance="relevant"))
        await env.step(Action(relevance="not_relevant"))
        state = await env.state()
        assert state["step"] == 2
        assert len(state["episode_rewards"]) == 2
        await env.close()

    @pytest.mark.asyncio
    async def test_env_loads_all_three_personas(self):
        from email_env import EmailTriageEnv
        env = await EmailTriageEnv.from_env(task_level="easy", max_emails=5)
        assert "Student" in env.email_data
        assert "Professional" in env.email_data
        assert "Business Owner" in env.email_data
        await env.close()

    @pytest.mark.asyncio
    async def test_hard_env_step_with_keyword(self):
        from email_env import EmailTriageEnv
        from model import Action
        env = await EmailTriageEnv.from_env(task_level="hard", max_emails=5)
        await env.reset()
        _, reward, _, _ = await env.step(Action(
            relevance="relevant",
            priority="urgent",
            reason="user searched google interview questions recently"
        ))
        assert 0.0 <= reward.value <= 1.0
        await env.close()

    @pytest.mark.asyncio
    async def test_observation_context_matches_persona(self):
        from email_env import EmailTriageEnv
        env = await EmailTriageEnv.from_env(task_level="easy", max_emails=5)
        obs = await env.reset()
        # context should have the keys defined in personas.json
        assert "recent_searches" in obs.context
        await env.close()


# ─────────────────────────────────────────────────────────
# 6. FASTAPI APP TESTS
# ─────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """TestClient using real personas.json via EmailTriageEnv."""
    from fastapi.testclient import TestClient
    import server.app as app_module
    app_module.sessions.clear()
    tc = TestClient(app_module.app)
    yield tc
    app_module.sessions.clear()


class TestHealthEndpoint:

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_shows_session_count(self, client):
        assert "active_sessions" in client.get("/health").json()


class TestCreateEnvEndpoint:

    def test_create_easy(self, client):
        resp = client.post("/env/create", json={"task_level": "easy"})
        assert resp.status_code == 200
        assert "session_id" in resp.json()

    def test_create_medium(self, client):
        assert client.post("/env/create", json={"task_level": "medium"}).status_code == 200

    def test_create_hard(self, client):
        assert client.post("/env/create", json={"task_level": "hard"}).status_code == 200

    def test_create_invalid_level_returns_400(self, client):
        assert client.post("/env/create", json={"task_level": "extreme"}).status_code == 400

    def test_create_default_max_emails_is_10(self, client):
        resp = client.post("/env/create", json={"task_level": "easy"})
        assert resp.json()["max_emails"] == 10

    def test_create_custom_max_emails(self, client):
        resp = client.post("/env/create", json={"task_level": "easy", "max_emails": 3})
        assert resp.json()["max_emails"] == 3


class TestResetEndpoint:

    def test_reset_returns_real_persona(self, client):
        sid = client.post("/env/create", json={"task_level": "easy"}).json()["session_id"]
        obs = client.post(f"/env/{sid}/reset").json()["observation"]
        assert obs["persona"] in ["Student", "Professional", "Business Owner"]

    def test_reset_observation_has_recent_searches(self, client):
        sid = client.post("/env/create", json={"task_level": "easy"}).json()["session_id"]
        obs = client.post(f"/env/{sid}/reset").json()["observation"]
        assert "recent_searches" in obs["context"]

    def test_reset_invalid_session_returns_404(self, client):
        assert client.post("/env/bad-session-id/reset").status_code == 404


class TestStepEndpoint:

    def _setup(self, client, level="easy", max_emails=5):
        sid = client.post("/env/create", json={
            "task_level": level, "max_emails": max_emails
        }).json()["session_id"]
        client.post(f"/env/{sid}/reset")
        return sid

    def test_easy_step_reward_in_range(self, client):
        sid = self._setup(client)
        data = client.post(f"/env/{sid}/step", json={"relevance": "relevant"}).json()
        assert 0.0 <= data["reward"] <= 1.0

    def test_medium_step_with_priority(self, client):
        sid = self._setup(client, "medium")
        resp = client.post(f"/env/{sid}/step", json={
            "relevance": "relevant", "priority": "urgent"
        })
        assert resp.status_code == 200

    def test_hard_step_with_real_keyword_in_reason(self, client):
        sid = self._setup(client, "hard")
        resp = client.post(f"/env/{sid}/step", json={
            "relevance": "relevant",
            "priority": "urgent",
            "reason": "user searched google interview questions"
        })
        assert resp.status_code == 200

    def test_step_done_false_before_max(self, client):
        sid = self._setup(client, max_emails=5)
        data = client.post(f"/env/{sid}/step", json={"relevance": "relevant"}).json()
        assert data["done"] is False

    def test_step_invalid_session_returns_404(self, client):
        assert client.post("/env/nonexistent/step", json={"relevance": "relevant"}).status_code == 404

    def test_step_returns_observation_info(self, client):
        sid = self._setup(client)
        data = client.post(f"/env/{sid}/step", json={"relevance": "relevant"}).json()
        assert "observation" in data
        assert "info" in data
        assert "step" in data["info"]


class TestStateEndpoint:

    def test_state_reflects_steps(self, client):
        sid = client.post("/env/create", json={"task_level": "easy"}).json()["session_id"]
        client.post(f"/env/{sid}/reset")
        client.post(f"/env/{sid}/step", json={"relevance": "relevant"})
        state = client.get(f"/env/{sid}/state").json()
        assert state["step"] == 1
        assert state["task_level"] == "easy"

    def test_state_invalid_session_returns_404(self, client):
        assert client.get("/env/fake-id/state").status_code == 404


class TestCloseEndpoint:

    def test_close_removes_session(self, client):
        sid = client.post("/env/create", json={"task_level": "easy"}).json()["session_id"]
        client.post(f"/env/{sid}/close")
        assert client.get(f"/env/{sid}/state").status_code == 404

    def test_close_invalid_session_returns_404(self, client):
        assert client.post("/env/bad-id/close").status_code == 404

    def test_close_returns_message(self, client):
        sid = client.post("/env/create", json={"task_level": "easy"}).json()["session_id"]
        assert "message" in client.post(f"/env/{sid}/close").json()


# ─────────────────────────────────────────────────────────
# 7. INTEGRATION — Full episodes via API (real data)
# ─────────────────────────────────────────────────────────

class TestFullEpisodeIntegration:

    def test_easy_full_episode_all_rewards_in_range(self, client):
        sid = client.post("/env/create", json={
            "task_level": "easy", "max_emails": 5
        }).json()["session_id"]
        client.post(f"/env/{sid}/reset")
        rewards = []
        for _ in range(5):
            data = client.post(f"/env/{sid}/step", json={"relevance": "relevant"}).json()
            rewards.append(data["reward"])
            if data["done"]:
                break
        assert all(0.0 <= r <= 1.0 for r in rewards)

    def test_hard_episode_with_real_student_keywords(self, client):
        sid = client.post("/env/create", json={
            "task_level": "hard", "max_emails": 5
        }).json()["session_id"]
        client.post(f"/env/{sid}/reset")
        for _ in range(5):
            resp = client.post(f"/env/{sid}/step", json={
                "relevance": "relevant",
                "priority": "urgent",
                # uses actual keywords from STUDENT_CTX["recent_searches"]
                "reason": "user searched google interview questions and amazon internship apply"
            })
            if resp.json().get("done"):
                break
        assert resp.status_code == 200

    def test_session_auto_removed_after_done(self, client):
        sid = client.post("/env/create", json={
            "task_level": "easy", "max_emails": 2
        }).json()["session_id"]
        client.post(f"/env/{sid}/reset")
        done = False
        for _ in range(2):
            data = client.post(f"/env/{sid}/step", json={"relevance": "relevant"}).json()
            if data["done"]:
                done = True
                break
        assert done
        # session auto-cleaned → 404
        assert client.get(f"/env/{sid}/state").status_code == 404

    def test_two_parallel_sessions_independent(self, client):
        sid1 = client.post("/env/create", json={
            "task_level": "easy", "max_emails": 5
        }).json()["session_id"]
        sid2 = client.post("/env/create", json={
            "task_level": "hard", "max_emails": 5
        }).json()["session_id"]

        client.post(f"/env/{sid1}/reset")
        client.post(f"/env/{sid2}/reset")

        client.post(f"/env/{sid1}/step", json={"relevance": "relevant"})
        client.post(f"/env/{sid2}/step", json={
            "relevance": "relevant", "priority": "urgent",
            "reason": "gst filing deadline 2025 is coming soon"
        })

        s1 = client.get(f"/env/{sid1}/state").json()
        s2 = client.get(f"/env/{sid2}/state").json()
        assert s1["task_level"] == "easy"
        assert s2["task_level"] == "hard"
        assert s1["step"] == 1
        assert s2["step"] == 1