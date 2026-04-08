"""
rewards.py — Grading system for Email Triage Environment
=========================================================

Graders:
  EasyGrader   → relevance only         (binary, with input validation)
  MediumGrader → relevance + priority   (weighted, adjacent priority partial credit)
  HardGrader   → relevance + priority + context-aware reason quality
                 (multi-signal, not gameable)

Score range: [0.0, 1.0] for all graders.
"""

from __future__ import annotations
import re
from typing import Optional


# ── Constants ──────────────────────────────────────────────────────────────

VALID_RELEVANCE = {"relevant", "not_relevant"}
VALID_PRIORITY  = {"urgent", "normal", "low"}

# Priority adjacency — being one step off is less wrong than two steps off
PRIORITY_ORDER = ["urgent", "normal", "low"]


# ── Shared Utilities ───────────────────────────────────────────────────────

def _normalize(value: Optional[str]) -> str:
    """Lowercase + strip; returns empty string if None."""
    return (value or "").strip().lower()


def _validate_relevance(value: Optional[str]) -> Optional[str]:
    """Returns normalized relevance or None if invalid."""
    v = _normalize(value)
    return v if v in VALID_RELEVANCE else None


def _validate_priority(value: Optional[str]) -> Optional[str]:
    """Returns normalized priority or None if invalid."""
    v = _normalize(value)
    return v if v in VALID_PRIORITY else None


def _priority_distance(predicted: str, actual: str) -> int:
    """
    Returns how many steps apart two priorities are.
    urgent→normal = 1, urgent→low = 2, normal→normal = 0
    """
    try:
        return abs(PRIORITY_ORDER.index(predicted) - PRIORITY_ORDER.index(actual))
    except ValueError:
        return 2  # max penalty if either is unknown


def _extract_reason_keywords(reason: str, context: dict) -> dict:
    """
    Checks how well the reason matches context signals.
    Returns a dict with match counts and quality score.

    Signals checked:
      - recent_searches  : highest weight (agent should cite these)
      - upcoming_events  : medium weight
      - active_relationships : medium weight
      - recent_purchases : lower weight
    """
    reason_text = _normalize(reason)

    # Split multi-word keywords into individual tokens for flexible matching
    def keyword_hit(keyword: str) -> bool:
        kw = _normalize(keyword)
        # Multi-word: all significant words must appear in reason
        words = [w for w in kw.split() if len(w) > 3]  # skip tiny words like "how", "to"
        return all(w in reason_text for w in words) if words else kw in reason_text

    def email_relevance(keyword: str, email: dict) -> bool:
        """Check if the keyword relates to the email content too."""
        combined = _normalize(email.get("subject", "") + " " + email.get("body", ""))
        words = [w for w in _normalize(keyword).split() if len(w) > 3]
        return any(w in combined for w in words) if words else False

    return {
        "recent_searches":     context.get("recent_searches", []),
        "upcoming_events":     context.get("upcoming_events", []),
        "active_relationships": context.get("active_relationships", []),
        "recent_purchases":    context.get("recent_purchases", []),
        "keyword_hit":         keyword_hit,
        "email_relevance":     email_relevance,
    }


def _reason_quality_score(agent_response: dict, email: dict, context: dict) -> float:
    """
    Scores the quality of the agent's reason out of 1.0.

    Criteria:
      1. Cites at least one relevant recent_search that relates to the email  → 0.40
      2. Mentions an upcoming_event or active_relationship relevant to email  → 0.30
      3. Reason is specific (not generic filler) and min length               → 0.20
      4. Reason aligns with the correct relevance decision                    → 0.10

    Total: 1.0
    This score is then scaled to the weight assigned in HardGrader (0.4 of total).
    """
    reason = _normalize(agent_response.get("reason", ""))
    signals = _extract_reason_keywords(reason, context)
    kw_hit = signals["keyword_hit"]
    em_rel = signals["email_relevance"]

    score = 0.0

    # ── Criterion 1: recent_searches cited AND relate to email (0.40) ─────
    search_hits = [
        kw for kw in signals["recent_searches"]
        if kw_hit(kw) and em_rel(kw, email)
    ]
    if len(search_hits) >= 2:
        score += 0.40
    elif len(search_hits) == 1:
        score += 0.25
    else:
        # Partial: keyword in reason but doesn't relate to email
        loose_hits = [kw for kw in signals["recent_searches"] if kw_hit(kw)]
        if loose_hits:
            score += 0.10

    # ── Criterion 2: upcoming_event or relationship cited (0.30) ──────────
    event_hits = [e for e in signals["upcoming_events"] if kw_hit(e)]
    rel_hits   = [r for r in signals["active_relationships"] if kw_hit(r)]

    if event_hits and rel_hits:
        score += 0.30          # cited both → full marks
    elif event_hits or rel_hits:
        score += 0.15          # cited one → half marks

    # ── Criterion 3: reason is specific and long enough (0.20) ───────────
    word_count = len(reason.split())
    filler_phrases = [
        "this email", "the email", "not relevant", "is relevant",
        "unrelated", "related to", "because"
    ]
    filler_count = sum(1 for f in filler_phrases if f in reason)
    unique_words = len(set(reason.split()))

    if word_count >= 8 and unique_words >= 6 and filler_count <= 1:
        score += 0.20
    elif word_count >= 5:
        score += 0.10

    # ── Criterion 4: reason matches relevance decision (0.10) ────────────
    predicted_relevance = _normalize(agent_response.get("relevance", ""))
    actual_relevance    = _normalize(email.get("label", ""))
    if predicted_relevance == actual_relevance:
        # Reason should support the decision, not contradict it
        if actual_relevance == "relevant" and any(
            w in reason for w in ["deadline", "interview", "urgent", "filing",
                                   "meeting", "exam", "delivery", "launch",
                                   "recruiter", "supplier", "client", "professor"]
        ):
            score += 0.10
        elif actual_relevance == "not_relevant" and any(
            w in reason for w in ["unrelated", "irrelevant", "spam", "offer",
                                   "promotion", "discount", "sale", "no relation"]
        ):
            score += 0.10
        else:
            score += 0.05   # correct decision but reason doesn't justify it well

    return round(min(score, 1.0), 4)


# ── EasyGrader ─────────────────────────────────────────────────────────────

class EasyGrader:
    """
    Task: Classify email as relevant or not_relevant.
    Score: 1.0 (correct) | 0.0 (wrong) | -0.1 (invalid input)

    Fixes over original:
      - Accepts dict OR string for agent_response
      - Validates input values — penalises garbage output
      - Clear docstring and normalized comparison
    """

    def score(
        self,
        agent_response: dict | str,
        email: dict,
        context: dict
    ) -> float:

        # ── Normalize input — accept dict or bare string ───────────────
        if isinstance(agent_response, str):
            predicted = _normalize(agent_response)
        elif isinstance(agent_response, dict):
            predicted = _normalize(agent_response.get("relevance", ""))
        else:
            return 0.0   # completely unexpected type

        actual = _normalize(email.get("label", ""))

        # ── Validate predicted value ───────────────────────────────────
        if predicted not in VALID_RELEVANCE:
            return 0.0   # malformed output — no partial credit

        # ── Score ──────────────────────────────────────────────────────
        return 1.0 if predicted == actual else 0.0


# ── MediumGrader ───────────────────────────────────────────────────────────

class MediumGrader:
    """
    Task: Classify relevance AND priority.
    Score breakdown:
      - Both correct                          → 1.0
      - Relevance correct, priority 1 off     → 0.6  (adjacent penalty)
      - Relevance correct, priority 2 off     → 0.4
      - Relevance correct, priority invalid   → 0.35
      - Relevance wrong, priority correct     → 0.15 (lucky guess on priority)
      - Relevance wrong, priority 1 off       → 0.05
      - Both wrong / both invalid             → 0.0

    Fixes over original:
      - Adjacent priority partial credit (urgent vs normal is less wrong than urgent vs low)
      - Input validation for both fields
      - Relevance-wrong-but-priority-correct lowered (was 0.3, too generous)
    """

    def score(
        self,
        agent_response: dict,
        email: dict,
        context: dict
    ) -> float:

        predicted_rel  = _validate_relevance(agent_response.get("relevance"))
        predicted_pri  = _validate_priority(agent_response.get("priority"))
        actual_rel     = _normalize(email.get("label", ""))
        actual_pri     = _normalize(email.get("priority", ""))

        relevance_correct = (predicted_rel == actual_rel)
        priority_dist     = _priority_distance(predicted_pri or "", actual_pri)

        # ── Score matrix ───────────────────────────────────────────────
        if relevance_correct:
            if priority_dist == 0:
                return 1.0
            elif priority_dist == 1:
                return 0.6   # one step off (urgent vs normal)
            elif priority_dist == 2:
                return 0.4   # two steps off (urgent vs low)
            else:
                return 0.35  # invalid priority but relevance right
        else:
            if priority_dist == 0:
                return 0.15  # correct priority but wrong relevance
            elif priority_dist == 1:
                return 0.05
            else:
                return 0.0


# ── HardGrader ─────────────────────────────────────────────────────────────

class HardGrader:
    """
    Task: Relevance + Priority + Context-aware Reason.
    Score breakdown (total = 1.0):

      Relevance  : 0.25  (correct label)
      Priority   : 0.35  (with adjacency partial credit)
      Reason     : 0.40  (multi-signal quality score — not gameable)

    Priority sub-scores:
      exact match     → 0.35
      1 step off      → 0.20
      2 steps off     → 0.10
      invalid         → 0.00

    Reason sub-scores (see _reason_quality_score for full breakdown):
      Cites search keywords that relate to email  → up to 0.40
      Mentions event or relationship              → up to 0.30
      Specific and long enough                   → up to 0.20
      Justifies the relevance decision            → up to 0.10

    Fixes over original:
      - Keyword match requires relation to email (not gameable by stuffing all keywords)
      - Multi-signal reason scoring (searches + events + relationships)
      - Partial credit at every level
      - Input validation throughout
      - Reason=None handled gracefully (no crash)
    """

    # Weights
    RELEVANCE_WEIGHT = 0.25
    PRIORITY_WEIGHT  = 0.35
    REASON_WEIGHT    = 0.40

    def score(
        self,
        agent_response: dict,
        email: dict,
        context: dict
    ) -> float:

        score = 0.0

        # ── 1. Relevance (0.25) ────────────────────────────────────────
        predicted_rel = _validate_relevance(agent_response.get("relevance"))
        actual_rel    = _normalize(email.get("label", ""))

        if predicted_rel == actual_rel:
            score += self.RELEVANCE_WEIGHT

        # ── 2. Priority (0.35) ────────────────────────────────────────
        predicted_pri = _validate_priority(agent_response.get("priority"))
        actual_pri    = _normalize(email.get("priority", ""))
        dist          = _priority_distance(predicted_pri or "", actual_pri)

        if dist == 0:
            score += self.PRIORITY_WEIGHT          # 0.35
        elif dist == 1:
            score += self.PRIORITY_WEIGHT * 0.57   # 0.20
        elif dist == 2:
            score += self.PRIORITY_WEIGHT * 0.28   # 0.10
        # else 0 (invalid)

        # ── 3. Reason quality (0.40) ───────────────────────────────────
        reason_raw = agent_response.get("reason")
        if reason_raw:
            rq = _reason_quality_score(agent_response, email, context)
            score += rq * self.REASON_WEIGHT
        # else: no reason provided → 0 for this component

        return round(min(score, 1.0), 4)


# ── Grader Factory ─────────────────────────────────────────────────────────

def get_grader(task_level: str) -> EasyGrader | MediumGrader | HardGrader:
    """
    Returns the correct grader instance for a given task level.
    Use this in EmailTriageEnv.__init__() to avoid creating a new
    grader object on every step.

    Usage:
        self.grader = get_grader(self.task_level)
        ...
        return self.grader.score(agent_response, self.current_email, context)
    """
    graders = {
        "easy":   EasyGrader(),
        "medium": MediumGrader(),
        "hard":   HardGrader(),
    }
    if task_level not in graders:
        raise ValueError(
            f"Unknown task_level '{task_level}'. "
            f"Must be one of {list(graders.keys())}"
        )
    return graders[task_level]


# ── Quick self-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_email = {
        "subject": "Google Interview Confirmed for Tomorrow 10 AM",
        "body": "Your interview with Google is confirmed for tomorrow at 10 AM IST.",
        "label": "relevant",
        "priority": "urgent",
        "reason": "User searched google interview questions and has interview tomorrow"
    }
    sample_context = {
        "recent_searches": ["google interview questions", "system design basics", "leetcode problems"],
        "upcoming_events": ["Google interview tomorrow", "Amazon internship deadline today"],
        "active_relationships": ["Google HR", "Amazon recruiter"],
        "recent_purchases": ["leetcode premium"]
    }

    print("=== EasyGrader ===")
    eg = EasyGrader()
    print(f"Correct:   {eg.score({'relevance': 'relevant'}, sample_email, sample_context)}")       # 1.0
    print(f"Wrong:     {eg.score({'relevance': 'not_relevant'}, sample_email, sample_context)}")   # 0.0
    print(f"String in: {eg.score('relevant', sample_email, sample_context)}")                       # 1.0
    print(f"Invalid:   {eg.score({'relevance': 'maybe'}, sample_email, sample_context)}")           # 0.0

    print("\n=== MediumGrader ===")
    mg = MediumGrader()
    print(f"Both correct:        {mg.score({'relevance': 'relevant', 'priority': 'urgent'}, sample_email, sample_context)}")   # 1.0
    print(f"Rel OK, pri 1 off:   {mg.score({'relevance': 'relevant', 'priority': 'normal'}, sample_email, sample_context)}")   # 0.6
    print(f"Rel OK, pri 2 off:   {mg.score({'relevance': 'relevant', 'priority': 'low'}, sample_email, sample_context)}")     # 0.4
    print(f"Rel wrong, pri OK:   {mg.score({'relevance': 'not_relevant', 'priority': 'urgent'}, sample_email, sample_context)}") # 0.15
    print(f"Both wrong:          {mg.score({'relevance': 'not_relevant', 'priority': 'low'}, sample_email, sample_context)}")  # 0.0

    print("\n=== HardGrader ===")
    hg = HardGrader()
    perfect = {
        "relevance": "relevant",
        "priority": "urgent",
        "reason": "user searched google interview questions and system design basics, has Google interview tomorrow with Google HR"
    }
    partial = {
        "relevance": "relevant",
        "priority": "normal",
        "reason": "google interview"
    }
    wrong = {
        "relevance": "not_relevant",
        "priority": "low",
        "reason": "unrelated email"
    }
    print(f"Perfect response:  {hg.score(perfect, sample_email, sample_context)}")   # ~0.95+
    print(f"Partial response:  {hg.score(partial, sample_email, sample_context)}")   # ~0.45-0.60
    print(f"Wrong response:    {hg.score(wrong, sample_email, sample_context)}")     # ~0.05-0.15

    print("\n=== Factory ===")
    for level in ["easy", "medium", "hard"]:
        g = get_grader(level)
        print(f"{level}: {type(g).__name__}")