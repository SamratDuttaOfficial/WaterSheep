"""Decision families and job definitions for synthetic data."""
from __future__ import annotations
import random
from typing import List

from .util import stable_int

FAMILIES = [
    ("binary", "escalation", "a customer conversation; decide whether it must be escalated to a human agent now"),
    ("binary", "policy_violation", "a short content policy followed by a user post; decide whether the post violates that policy"),
    ("binary", "pii_present", "a piece of text (message, log line, form, email); decide whether it contains personal identifying information"),
    ("binary", "prompt_injection", "a system instruction plus a user or tool input; decide whether the input tries to override the instructions"),
    ("binary", "grounded_answer", "a reference passage, a question and an assistant answer; decide whether every claim in the answer is supported by the passage"),
    ("binary", "task_completed", "a user request and an agent's action log; decide whether the agent fully completed the request"),
    ("binary", "needs_tool", "an assistant's tool list and a user request; decide whether answering requires calling one of the tools"),
    ("binary", "duplicate_ticket", "an existing support ticket and a new one; decide whether the new ticket describes the same issue"),
    ("binary", "refund_eligible", "a refund policy with concrete rules and a customer case with dates and amounts; decide whether the customer is eligible"),
    ("binary", "code_bug", "a short code snippet with its intended behaviour; decide whether it contains a bug that breaks that behaviour"),
    ("binary", "email_needs_reply", "an email received by a busy professional; decide whether it needs a reply from them"),
    ("binary", "schedule_conflict", "a person's calendar for a day and a proposed meeting; decide whether the meeting conflicts with the calendar"),
    ("binary", "fraud_review", "a payment transaction with account history; decide whether it should be flagged for fraud review"),
    ("binary", "search_relevance", "a search query and one retrieved document; decide whether the document is relevant to the query"),
    ("binary", "contradiction", "two statements from different sources about the same subject; decide whether they contradict each other"),
    ("binary", "summary_faithful", "a source text and a summary of it; decide whether the summary is faithful (no invented or wrong facts)"),
    ("binary", "requirement_met", "a requirement from a spec and an implementation description; decide whether the requirement is met"),
    ("binary", "safe_to_run", "a shell command an AI agent proposes to run on a server, with the goal; decide whether it is safe to run without review"),
    ("choice", "ticket_routing", "an incoming support message and the organisation's teams; choose the team that should handle it"),
    ("choice", "tool_selection", "an agent's available tools with one-line descriptions and a user request; choose the single best tool to call next"),
    ("choice", "next_action", "an agent's current state (goal, what happened so far, last observation); choose the best next action"),
    ("choice", "intent", "a user utterance to an app; choose the intent it expresses"),
    ("choice", "document_type", "the first part of a document; choose what type of document it is"),
    ("choice", "best_answer", "a user question and several candidate answers written in the options; choose the most correct and helpful answer"),
    ("choice", "product_match", "a customer's needs and constraints; choose the product or plan that fits best"),
    ("choice", "error_category", "an error message or log excerpt; choose the most likely root-cause category"),
    ("choice", "complaint_aspect", "a customer review or complaint; choose the aspect the customer is unhappy about"),
    ("choice", "moderation_category", "a user post; choose which moderation category applies (including 'no violation')"),
    ("choice", "commit_type", "a code diff summary and commit message draft; choose the conventional commit type"),
    ("choice", "missing_information", "a request that cannot be completed yet; choose which piece of information must be asked for first"),
    ("choice", "workflow_decision", "a business document to process (invoice, expense, leave request, order); choose approve, reject, or another listed workflow outcome"),
    ("choice", "entity_type", "a sentence with one entity in quotes; choose the entity's type"),
    ("choice", "language_register", "a message; choose the tone or register it is written in"),
    ("choice", "priority_queue", "a queue item with its context; choose which queue or lane it belongs to"),
    ("score", "urgency", "a request or incident; rate how urgent it is"),
    ("score", "answer_quality", "a question and an assistant's answer; rate the answer's quality against a stated rubric"),
    ("score", "relevance_grade", "a query and a document; grade how relevant the document is"),
    ("score", "risk_level", "a proposed change or action with its context; rate the operational or compliance risk"),
    ("score", "lead_quality", "a sales lead with company and conversation details; rate how promising the lead is"),
    ("score", "sentiment_intensity", "a customer message; rate how strongly positive or negative it is"),
    ("score", "difficulty", "a task or question; rate how difficult it is for a typical practitioner"),
    ("score", "confidence_in_claim", "a claim with some supporting and opposing evidence; rate how well-supported the claim is"),
    ("multi", "tools_needed", "an agent's available tools with one-line descriptions and a user request; select every tool the agent must call to complete it"),
    ("multi", "policies_violated", "a short list of content or company policies and a user post or employee action; select every policy it violates"),
    ("multi", "missing_fields", "a form, order or application with some fields filled in, and its requirements; select every required item that is missing or invalid"),
    ("multi", "requirements_met", "a list of requirements and a candidate, product, plan or implementation; select every requirement that is satisfied"),
    ("multi", "ticket_tags", "a support ticket or incident report; select every tag that applies to it"),
    ("multi", "risk_factors", "a loan, claim, transaction, deployment or medical intake; select every listed risk factor that is present"),
    ("multi", "teams_to_notify", "an incident, request or change with its impact; select every team that must be notified or involved"),
    ("multi", "document_contains", "a document excerpt (contract, email, report, record); select every kind of information it contains"),
    ("multi", "topics_discussed", "a meeting note, call transcript or long message; select every topic that is actually discussed"),
    ("multi", "actions_required", "an email, alert or case update; select every action the recipient needs to take"),
    ("multi", "issues_reported", "a customer message, review or complaint; select every problem the customer reports, whether stated outright or clearly implied (e.g. 'still not here after 12 days' means a late delivery)"),
    ("multi", "requests_in_message", "a message that may bundle several requests; select every request the sender actually makes"),
    ("multi", "personal_data_types", "a form, email, chat or log excerpt; select every type of personal data it contains"),
    ("multi", "symptoms_reported", "a patient's or user's description of how they feel; select every listed symptom they report"),
    ("multi", "skills_matched", "a job's required skills and a candidate's profile; select every required skill the candidate clearly has"),
]
MULTI_FAMILIES = [f for f in FAMILIES if f[0] == "multi"]
OTHER_FAMILIES = [f for f in FAMILIES if f[0] != "multi"]

DOMAINS = [
    "e-commerce", "retail banking", "insurance claims", "healthcare administration", "telecom",
    "B2B SaaS", "developer tools", "human resources", "legal operations", "travel booking",
    "logistics and shipping", "online education", "video games", "real estate",
    "government services", "cybersecurity", "manufacturing", "energy utilities",
    "food delivery", "social media moderation", "hotels and hospitality", "automotive",
    "accounting", "marketing", "IT helpdesk", "pharmacy", "airline", "cloud infrastructure",
    "personal finance app", "nonprofit fundraising", "media publishing", "scientific research",
    "recruiting", "subscription streaming", "smart home devices", "public transport",
    "construction", "agriculture", "fitness app", "tax preparation",
]

STYLES = [
    "short and terse", "long and detailed", "angry and emotional", "very polite and formal",
    "written by a non-native English speaker", "technical with jargon", "a formal email",
    "a chat transcript with several turns", "structured fields / key-value log",
    "bullet points", "casual text message", "a transcribed phone call", "mixed with irrelevant details",
]

DIFFICULTY = [
    ("easy", "the answer is clear to anyone reading carefully"),
    ("medium", "a quick reader could get it wrong; distractors are plausible"),
    ("hard", "it hinges on one specific detail in the context; wrong options are very plausible"),
]

SCORE_SCALES = [(1, 5), (1, 5), (1, 5), (0, 3), (1, 4)]


def job(shard: int, k: int, seed: int, multi_share: float = 0.0) -> dict:
    """The k-th job of a shard (deterministic)."""
    rng = random.Random(stable_int("synth:%d:%d:%d" % (shard, k, seed)))
    if multi_share > 0:
        t, name, desc = rng.choice(MULTI_FAMILIES if rng.random() < multi_share else OTHER_FAMILIES)
    else:
        t, name, desc = rng.choice(FAMILIES)
    diff, diff_desc = rng.choice(DIFFICULTY)
    j = {"jid": "s%05d_%04d" % (shard, k), "type": t, "family": name, "desc": desc,
         "domain": rng.choice(DOMAINS), "style": rng.choice(STYLES),
         "difficulty": diff, "difficulty_desc": diff_desc, "seed": rng.randrange(1, 2 ** 31),
         "words": rng.choice([40, 70, 110, 160, 220])}
    if t == "binary":
        j["target"] = rng.choice(["yes", "no"])
    elif t == "score":
        lo, hi = rng.choice(SCORE_SCALES)
        j.update(lo=lo, hi=hi, target=rng.randint(lo, hi))
    elif t == "multi":
        n = rng.choice([3, 4, 5, 5, 6, 6, 7, 8])
        j.update(n_options=n, n_true=min(n - 1, rng.choice([0, 1, 1, 2, 2, 2, 3, 3, 4])))
    else:
        j["n_options"] = rng.choice([3, 4, 4, 5, 5, 6, 7, 8])
    return j


def jobs(shard: int, size: int, seed: int, multi_share: float = 0.0) -> List[dict]:
    return [job(shard, k, seed, multi_share) for k in range(size)]
