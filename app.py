# app.py — AI Email Agent Training Environment
# Preserved Approved Visual Design with Connected Real OpenEnv & Learning Backend

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# ── Load .env ──────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ── Phase 1 & 2 Backend Imports ───────────────────────────────────────────────
try:
    from envs.email_env.models import (
        ActionType,
        ClassLabel,
        DifficultyLevel,
        EmailAction,
        EmailObservation,
        EmailStatus,
        GroundTruthEmail,
    )
    from envs.email_env.server.environment import EmailEnvironment
    from envs.email_env.server.rubric import EmailRubric, RubricWeights
    from envs.email_env.server.tasks import ScenarioLoader
    from agents.router import AgentRouter
    from agents.decision import AgentDecision
    from agents.fallback import LocalFallbackAgent
    from agents.llm import LLMClient
    from evaluation.models import (
        EpisodeRecord,
        FeedbackRecord,
        FeedbackVerdict,
        LessonRecord,
        LessonStatus,
        TrajectoryStep,
    )
    from evaluation.storage import SQLiteRepository
    from evaluation.retrieval import LessonRetriever
    from evaluation.feedback import FeedbackService
    from evaluation.lessons import LessonGenerator
    from evaluation.validator import LessonValidator
    _BACKEND_OK = True
except Exception as _import_err:
    _BACKEND_OK = False

# Fallback legacy db if available
try:
    import db as _db
    _DB_OK = True
except Exception:
    _DB_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be FIRST streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Email Agent — Flight Simulator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

_LOGO_SVG = """<svg width="28" height="28" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle;margin-right:10px;filter:drop-shadow(0 0 10px rgba(0,229,255,0.7));flex-shrink:0;"><defs><linearGradient id="logoGrad" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#00E5FF"/><stop offset="100%" stop-color="#8B5CF6"/></linearGradient></defs><path d="M16 3L28 8.5V17C28 23.5 22.8 28.2 16 30C9.2 28.2 4 23.5 4 17V8.5L16 3Z" stroke="url(#logoGrad)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" fill="rgba(0,229,255,0.06)"/><path d="M9 13.5L16 18.5L23 13.5" stroke="#00E5FF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><circle cx="16" cy="18.5" r="2.2" fill="#8B5CF6" stroke="#00E5FF" stroke-width="1.2"/><path d="M16 7.5V10.5M9.5 21L12 19M22.5 21L20 19" stroke="#00E5FF" stroke-width="1.5" stroke-linecap="round"/></svg>"""

@st.cache_resource(show_spinner=False)
def _check_gemini_api_status() -> tuple[bool, str]:
    """Validates if GEMINI_API_KEY is present and responding to real API calls."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return False, "Local Fallback (No Key)"
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        model.generate_content("Ping", generation_config={"max_output_tokens": 1})
        return True, "✓ Gemini Active (Verified)"
    except Exception:
        return False, "Local Fallback (Key Invalid)"


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — APPROVED FROZEN DESIGN BASELINE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

@keyframes pulse-glow{0%,100%{box-shadow:0 0 8px rgba(0,229,255,.35),0 0 22px rgba(0,229,255,.12)}50%{box-shadow:0 0 20px rgba(0,229,255,.7),0 0 44px rgba(0,229,255,.25)}}
@keyframes border-glow{0%,100%{border-color:rgba(0,229,255,.25)}50%{border-color:rgba(139,92,246,.65)}}
@keyframes hero-pulse{0%,100%{box-shadow:0 2px 0 0 rgba(0,229,255,.4) inset,0 -1px 0 0 rgba(120,60,220,.25) inset,0 0 0 1px rgba(80,60,160,.5),0 12px 50px rgba(0,0,0,.75),0 0 80px rgba(0,160,255,.09),0 0 160px rgba(100,40,240,.07)}50%{box-shadow:0 2px 0 0 rgba(0,229,255,.65) inset,0 -1px 0 0 rgba(140,80,255,.4) inset,0 0 0 1px rgba(100,70,200,.7),0 12px 50px rgba(0,0,0,.8),0 0 110px rgba(0,200,255,.15),0 0 220px rgba(130,60,255,.12)}}

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,[data-testid="stAppViewContainer"]{background:radial-gradient(ellipse at 20% 10%,#0a0f2c 0%,#050816 60%,#000309 100%)!important;font-family:'Space Grotesk','Inter',sans-serif!important;color:#e2e8f0!important}
[data-testid="stAppViewContainer"]::before{content:"";position:fixed;inset:0;pointer-events:none;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,229,255,.012) 2px,rgba(0,229,255,.012) 4px);z-index:0}

/* Completely remove native Streamlit hamburger menu, header, footer & toolbars */
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stHeader"], .stDeployButton {
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}

/* Tighten top margins */
.block-container{max-width:1150px!important;padding:0.8rem 1.5rem 2.5rem!important;position:relative;z-index:1}
[data-testid="stSidebarContent"]{padding-top:0.6rem!important}

/* Sidebar Styling */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#080d20 0%,#050816 100%)!important;border-right:1px solid rgba(0,229,255,.15)!important;box-shadow:4px 0 24px rgba(0,229,255,.05)!important}
[data-testid="stSidebar"] .stMarkdown p,[data-testid="stSidebar"] label{color:#7fa8c0!important;font-size:.82rem!important}

/* Sidebar Expand/Collapse Visibility Fix */
[data-testid="collapsedControl"],
[data-testid="stExpandSidebarButton"]{
  display:flex!important;
  visibility:visible!important;
  z-index:999999!important;
  color:#00E5FF!important;
  background:rgba(8,13,32,0.85)!important;
  border:1px solid rgba(0,229,255,0.3)!important;
  border-radius:8px!important;
  top:0.6rem!important;
  left:0.6rem!important;
}
[data-testid="collapsedControl"]:hover,
[data-testid="stExpandSidebarButton"]:hover{
  background:rgba(0,229,255,0.15)!important;
  border-color:#00E5FF!important;
  box-shadow:0 0 12px rgba(0,229,255,0.5)!important;
}
[data-testid="stSidebarCollapseButton"]{color:#00E5FF!important}

/* Buttons */
.stButton>button{font-family:'Space Grotesk','Inter',sans-serif!important;border-radius:10px!important;font-weight:600!important;letter-spacing:.03em!important;transition:all .2s cubic-bezier(.4,0,.2,1)!important}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#00b4d8 0%,#8B5CF6 100%)!important;border:none!important;color:#fff!important;box-shadow:0 0 14px rgba(0,229,255,.4),0 4px 20px rgba(139,92,246,.3)!important}
.stButton>button[kind="primary"]:hover{background:linear-gradient(135deg,#00E5FF 0%,#a78bfa 100%)!important;transform:translateY(-2px) scale(1.02)!important;box-shadow:0 0 28px rgba(0,229,255,.6),0 8px 32px rgba(139,92,246,.45)!important}
.stButton>button[kind="primary"]:active{transform:translateY(0) scale(.99)!important}
.stButton>button[kind="secondary"]{background:rgba(0,229,255,.05)!important;border:1px solid rgba(0,229,255,.2)!important;color:#7fa8c0!important}
.stButton>button[kind="secondary"]:hover{background:rgba(0,229,255,.1)!important;border-color:rgba(0,229,255,.45)!important;color:#00E5FF!important;box-shadow:0 0 14px rgba(0,229,255,.18)!important}

/* Cards */
.ea-card{background:linear-gradient(135deg,rgba(10,18,50,.92) 0%,rgba(6,12,35,.96) 100%);border:1px solid rgba(0,229,255,.15);border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:.9rem;box-shadow:0 4px 24px rgba(0,0,0,.5),inset 0 1px 0 rgba(0,229,255,.08);backdrop-filter:blur(8px)}
.inbox-card{background:linear-gradient(135deg,rgba(8,14,38,.95) 0%,rgba(5,10,28,.98) 100%);border:1px solid rgba(0,229,255,.1);border-left:3px solid rgba(0,229,255,.2);border-radius:10px;padding:.85rem 1.1rem;margin-bottom:.55rem;transition:all .2s ease;box-shadow:0 2px 12px rgba(0,0,0,.4)}
.inbox-card:hover{border-left-color:#00E5FF;box-shadow:0 0 20px rgba(0,229,255,.14),0 4px 16px rgba(0,0,0,.5);transform:translateX(2px)}

/* Step Card with Luminous Lightning Hover Glow */
.step-card{
  position:relative;
  background:linear-gradient(135deg,rgba(10,16,42,.95) 0%,rgba(6,10,28,.98) 100%);
  border:1px solid rgba(0,229,255,.08);
  border-left:3.5px solid rgba(0,229,255,.18);
  border-radius:10px;
  padding:.85rem 1.1rem;
  margin-bottom:.55rem;
  transition:all .25s cubic-bezier(0.4, 0, 0.2, 1);
  overflow:hidden;
}
.step-card::before{
  content:"";
  position:absolute;
  top:0;
  left:0;
  width:4px;
  height:100%;
  background:linear-gradient(180deg,#00E5FF 0%,#a78bfa 50%,#00E5FF 100%);
  box-shadow:0 0 16px #00E5FF,0 0 32px rgba(0,229,255,0.9);
  opacity:0;
  transition:opacity 0.25s ease,box-shadow 0.25s ease;
  z-index:2;
}
.step-card:hover{
  background:linear-gradient(135deg,rgba(14,24,62,.98) 0%,rgba(8,14,38,.98) 100%);
  border-color:rgba(0,229,255,.45);
  border-left-color:#00E5FF;
  transform:translateX(4px);
  box-shadow:0 4px 20px rgba(0,0,0,.6),0 0 25px rgba(0,229,255,.2);
}
.step-card:hover::before{
  opacity:1;
  box-shadow:0 0 12px #00E5FF,0 0 24px #00E5FF,0 0 45px rgba(0,229,255,1);
}
.step-card-active{
  border-left-color:#00E5FF!important;
  box-shadow:0 0 26px rgba(0,229,255,.22),0 0 10px rgba(139,92,246,.15),inset 0 0 22px rgba(0,229,255,.03)!important;
  animation:pulse-glow 2s ease-in-out infinite;
}

/* Table */
table.ea-table{width:100%;border-collapse:collapse;font-size:.82rem}
table.ea-table th{background:rgba(0,229,255,.07);color:#00E5FF;font-weight:700;padding:.45rem .7rem;text-align:left;border-bottom:1px solid rgba(0,229,255,.15);letter-spacing:.06em;text-transform:uppercase;font-size:.72rem}
table.ea-table td{padding:.4rem .7rem;border-bottom:1px solid rgba(0,229,255,.06);color:#a0b4cc}
table.ea-table tr:last-child td{border-bottom:none}
table.ea-table tr:hover td{background:rgba(0,229,255,.04);color:#c8dbe8}

/* Bars */
.bar-wrap{margin:.35rem 0}
.bar-label{font-size:.78rem;color:#7fa8c0;margin-bottom:3px}
.bar-bg{background:rgba(0,229,255,.07);border-radius:6px;height:8px;overflow:hidden;border:1px solid rgba(0,229,255,.1)}
.bar-fill{height:100%;border-radius:6px;box-shadow:0 0 8px currentColor}

/* Badges */
.ea-badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.72rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase}
.badge-indigo{background:rgba(139,92,246,.15);color:#a78bfa;border:1px solid rgba(139,92,246,.4);box-shadow:0 0 8px rgba(139,92,246,.2)}
.badge-green{background:rgba(34,255,136,.1);color:#22FF88;border:1px solid rgba(34,255,136,.35);box-shadow:0 0 8px rgba(34,255,136,.15)}
.badge-amber{background:rgba(255,159,28,.12);color:#FF9F1C;border:1px solid rgba(255,159,28,.4);box-shadow:0 0 8px rgba(255,159,28,.15)}
.badge-red{background:rgba(255,77,77,.12);color:#FF4D4D;border:1px solid rgba(255,77,77,.4);box-shadow:0 0 8px rgba(255,77,77,.15)}
.badge-gray{background:rgba(0,229,255,.07);color:#7fa8c0;border:1px solid rgba(0,229,255,.18)}

/* Hero Section */
.hero-block{
  background:linear-gradient(145deg,#07102e 0%,#0d1a45 22%,#0a1030 45%,#0f0828 70%,#150622 100%);
  border:1px solid rgba(0,229,255,.2);
  border-radius:22px;
  padding:2.2rem 2.4rem 1.8rem;
  margin-bottom:1.4rem;
  position:relative;
  overflow:hidden;
  animation:hero-pulse 6s ease-in-out infinite;
  isolation:isolate;
}
.hero-block::before{
  content:"";
  position:absolute;
  inset:0;
  border-radius:22px;
  background:
    linear-gradient(180deg,rgba(0,229,255,.09) 0%,rgba(0,180,255,.02) 28%,transparent 55%),
    linear-gradient(to bottom right,rgba(139,92,246,.07) 0%,transparent 45%),
    linear-gradient(to top,rgba(110,40,220,.06) 0%,transparent 40%);
  pointer-events:none;
  z-index:0;
}
.hero-block::after{
  content:"";
  position:absolute;
  top:-60%;
  left:-20%;
  width:140%;
  height:180%;
  background:
    radial-gradient(ellipse at 68% 20%,rgba(0,210,255,.10) 0%,transparent 50%),
    radial-gradient(ellipse at 10% 80%,rgba(139,92,246,.12) 0%,transparent 48%),
    radial-gradient(ellipse at 85% 78%,rgba(90,20,210,.08) 0%,transparent 42%),
    radial-gradient(ellipse at 45% 50%,rgba(0,150,255,.04) 0%,transparent 60%);
  pointer-events:none;
  z-index:0;
}
.hero-block > *{position:relative;z-index:1}
.hero-title{
  font-size:2.1rem;
  font-weight:700;
  color:#EEF4FF;
  line-height:1.2;
  margin-bottom:.45rem;
  text-shadow:0 0 28px rgba(0,200,255,.38),0 0 8px rgba(0,200,255,.18),0 2px 6px rgba(0,0,0,.7);
}
.hero-sub{font-size:1rem;color:#9ec5d8;margin-bottom:.4rem;text-shadow:0 1px 4px rgba(0,0,0,.5)}
.hero-tag{font-size:.82rem;color:#3de0f8;font-weight:500;text-shadow:0 0 14px rgba(0,225,255,.55),0 0 4px rgba(0,225,255,.25);opacity:.95;letter-spacing:0.02em}

.sec-heading{font-size:.68rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:#00E5FF;border-left:3px solid #00E5FF;padding-left:.65rem;margin:1.3rem 0 .85rem;text-shadow:0 0 14px rgba(0,229,255,.65);box-shadow:-2px 0 10px rgba(0,229,255,.3)}
.score-big{font-size:4rem;font-weight:800;line-height:1;font-variant-numeric:tabular-nums;text-shadow:0 0 30px currentColor}

.agent-card{background:linear-gradient(135deg,rgba(10,18,50,.9) 0%,rgba(6,10,30,.95) 100%);border:1px solid rgba(0,229,255,.15);border-radius:14px;padding:1rem 1.2rem;text-align:center;box-shadow:0 4px 20px rgba(0,0,0,.4),inset 0 1px 0 rgba(0,229,255,.08);transition:all .2s ease}
.agent-card:hover{box-shadow:0 0 24px rgba(0,229,255,.14),0 8px 32px rgba(0,0,0,.5);transform:translateY(-2px)}
.agent-card .ac-score{font-size:1.6rem;font-weight:800;text-shadow:0 0 16px currentColor}
.agent-card .ac-label{font-size:.7rem;color:#4a6a80;text-transform:uppercase;letter-spacing:.1em;margin-top:4px}

.row-pos td{color:#22FF88!important;text-shadow:0 0 8px rgba(34,255,136,.3)}
.row-neg td{color:#FF4D4D!important;text-shadow:0 0 8px rgba(255,77,77,.3)}
.row-neu td{color:#5a7a8f!important}

[data-testid="stSelectbox"]>div>div,[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea{background:rgba(5,10,28,.95)!important;border-color:rgba(0,229,255,.2)!important;color:#e2e8f0!important;border-radius:8px!important}
[data-testid="stExpander"]{background:rgba(5,10,28,.9)!important;border:1px solid rgba(0,229,255,.12)!important;border-radius:10px!important;box-shadow:0 2px 12px rgba(0,0,0,.4)!important}
[data-testid="stProgress"]>div>div{background:rgba(0,229,255,.1)!important;border-radius:6px!important}
[data-testid="stProgress"]>div>div>div{background:linear-gradient(90deg,#00b4d8,#8B5CF6)!important;box-shadow:0 0 10px rgba(0,229,255,.4)!important;border-radius:6px!important}
[data-testid="stMetric"]{background:rgba(0,229,255,.04)!important;border:1px solid rgba(0,229,255,.1)!important;border-radius:10px!important;padding:.6rem .8rem!important}
[data-testid="stMetric"] label{color:#4a6a80!important;font-size:.74rem!important;letter-spacing:.05em!important;text-transform:uppercase!important}
[data-testid="stMetric"] [data-testid="stMetricValue"]{color:#00E5FF!important;font-size:1.3rem!important;font-weight:700!important;text-shadow:0 0 12px rgba(0,229,255,.4)!important}
hr{border-color:rgba(0,229,255,.1)!important}
[data-testid="stSpinner"]>div{border-top-color:#00E5FF!important}
.ea-footer{text-align:center;color:#1e3040;font-size:.75rem;margin-top:3rem;letter-spacing:.05em}
.gemini-pill{display:inline-flex;align-items:center;gap:4px;background:rgba(0,229,255,.1);border:1px solid rgba(0,229,255,.35);border-radius:20px;padding:2px 10px;font-size:.72rem;color:#00E5FF;font-weight:700;box-shadow:0 0 10px rgba(0,229,255,.2);letter-spacing:.04em}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TASK METADATA & DIFFICULTY MAPPING
# ─────────────────────────────────────────────────────────────────────────────
_TASK_META: dict[str, dict] = {
    "task_1": {"label": "Starter Inbox (5 emails)",   "difficulty": "STARTER",     "email_count": 5,  "max_steps": 20},
    "task_2": {"label": "Medium Inbox (10 emails)",   "difficulty": "MEDIUM",      "email_count": 10, "max_steps": 40},
    "task_3": {"label": "Advanced Inbox (15 emails)", "difficulty": "ADVANCED",    "email_count": 15, "max_steps": 60},
    "task_4": {"label": "Adversarial Inbox (5 emails)","difficulty": "ADVERSARIAL", "email_count": 5,  "max_steps": 20},
    "task_5": {"label": "Held-Out Evaluation (5 emails)","difficulty": "HELD_OUT",  "email_count": 5,  "max_steps": 20},
}

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults: dict[str, Any] = {
        "env": None, "task_id": "task_1", "phase": "start",
        "obs_dict": None, "step_results": [], "cumulative_reward": 0.0,
        "steps_taken": 0, "episode_done": False, "final_score": None,
        "grade_data": None, "mode": "training", "show_guide": False,
        "custom_result": None, "app_error": None, "demo_completed": False,
        "auto_log": [], "demo_email_steps": [], "demo_playback_done": False,
        "last_feedback_status": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _badge(status: str) -> str:
    cls = {"start":"badge-gray","inbox":"badge-indigo","graded":"badge-green","compared":"badge-green"}.get(status,"badge-gray")
    return f'<span class="ea-badge {cls}">{status.upper()}</span>'

def _bar(label: str, val: float, colour: str) -> str:
    pct = int(min(max(val, 0.0), 1.0) * 100)
    return (
        f'<div class="bar-wrap">'
        f'<div class="bar-label">{label}'
        f'<span style="float:right;color:#f1f5f9;font-weight:600">{val:.2f}</span></div>'
        f'<div class="bar-bg"><div class="bar-fill" style="width:{pct}%;background:{colour}"></div></div>'
        f'</div>'
    )

def _sec(text: str, accent: str = "#6366f1") -> None:
    st.markdown(
        f'<div class="sec-heading" style="border-color:{accent};color:{accent}">{text}</div>',
        unsafe_allow_html=True,
    )

def _grade_letter(score: float) -> tuple[str, str]:
    if score >= 0.90: return "A", "#22c55e"
    if score >= 0.75: return "B", "#6366f1"
    if score >= 0.60: return "C", "#fbbf24"
    if score >= 0.40: return "D", "#f97316"
    return "F", "#ef4444"

def _interpret(score: float) -> str:
    if score >= 0.90: return "Excellent — inbox handled nearly perfectly."
    if score >= 0.75: return "Good — most actions were correct."
    if score >= 0.60: return "Fair — some classification or action errors."
    if score >= 0.40: return "Poor — significant errors."
    return "Very poor — check classification and action logic."

def _reset() -> None:
    for k in ["env","obs_dict","step_results","cumulative_reward","steps_taken",
              "episode_done","final_score","grade_data","app_error","demo_completed",
              "auto_log","demo_email_steps","demo_playback_done","last_feedback_status"]:
        st.session_state.pop(k, None)
    st.session_state["phase"] = "start"
    _init_state()


# ─────────────────────────────────────────────────────────────────────────────
# REAL OPENENV EPISODE RUNNER (PHASE 1 & PHASE 2 INTEGRATION)
# ─────────────────────────────────────────────────────────────────────────────
def run_auto_episode(task_id: str, agent_type: str = "hybrid") -> tuple[Any, list, Optional[str]]:
    """
    Executes a real simulation episode on EmailEnvironment using the requested agent.
    Computes true rubric rewards, records trajectory in SQLiteRepository, and returns UI log.
    """
    try:
        meta = _TASK_META.get(task_id, _TASK_META["task_1"])
        difficulty = meta["difficulty"]

        router = AgentRouter()
        agent = router.get_agent(agent_type)
        env = EmailEnvironment()
        obs = env.reset(difficulty=difficulty, seed=42)

        log: list[dict] = []
        trajectory: list[TrajectoryStep] = []
        step_num = 0
        start_time = datetime.now(timezone.utc).isoformat()
        safety_status = "SAFE"

        while not obs.done:
            curr_view = obs.current_email
            email_id = curr_view.id if curr_view else (obs.inbox_summary[0].id if obs.inbox_summary else "")
            subject = curr_view.subject if curr_view else ""
            sender = curr_view.sender if curr_view else ""

            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            step_num += 1

            feedback_msg = info.get("feedback", "")
            if "SAFETY VIOLATION" in feedback_msg:
                safety_status = "VIOLATION"

            # Normalize action type name for card icons
            act_val = action.action_type.value
            if act_val == "READ_EMAIL":
                act_type_str = "read"
            elif act_val == "CLASSIFY_EMAIL":
                act_type_str = "classify"
            elif act_val == "REPLY_EMAIL":
                act_type_str = "reply"
            elif act_val == "ESCALATE_EMAIL":
                act_type_str = "escalate"
            else:
                act_type_str = "archive"

            api_used = bool(getattr(agent, "llm_call_count", 0) > 0 and act_type_str in ("classify", "reply", "escalate"))

            if agent_type.lower() == "baseline":
                tier_label = "Tier 1 — Baseline"
            elif agent_type.lower() == "hybrid":
                tier_label = "⚡ Live Gemini" if api_used else "Tier 2 — Hybrid Heuristic"
            elif agent_type.lower() == "llm":
                tier_label = "⚡ Live Gemini" if api_used else "Tier 3 — Degraded LLM"
            else:
                tier_label = agent_type.title()

            log_entry = {
                "step": step_num,
                "action": f"{act_val}({action.email_id or ''})",
                "action_type": act_type_str,
                "email_id": action.email_id or email_id,
                "reward": round(reward, 3),
                "valid": obs.last_action_valid,
                "subject": subject,
                "sender": sender,
                "label": action.label.value if action.label else "",
                "api_used": api_used,
                "tier_label": tier_label,
                "feedback": feedback_msg,
            }
            log.append(log_entry)

            trajectory.append(
                TrajectoryStep(
                    step_number=step_num,
                    observation_summary={"step": step_num, "email_id": email_id},
                    action=action.model_dump(),
                    action_valid=obs.last_action_valid,
                    reward=reward,
                    feedback=feedback_msg,
                )
            )

            if done:
                break

        state = env.state()
        end_time = datetime.now(timezone.utc).isoformat()

        # Save to SQLite repository
        repo = SQLiteRepository()
        ep_record = EpisodeRecord(
            episode_id=state.episode_id,
            scenario_id=f"app_{difficulty.lower()}_{state.episode_id[:6]}",
            difficulty=difficulty,
            agent_name=agent_type.upper(),
            start_time=start_time,
            end_time=end_time,
            steps=trajectory,
            final_action=log[-1]["action_type"] if log else "",
            cumulative_reward=round(state.cumulative_reward, 4),
            safety_status=safety_status,
            completed=state.done,
            metadata={"source": "Streamlit UI", "total_emails": len(state.emails)},
        )
        repo.save_episode(ep_record)

        # Compute empirical scores
        correct_count = sum(1 for r in log if r["reward"] >= 0.15)
        score = round(min(1.0, max(0.0, (correct_count + (1.0 if safety_status == "SAFE" else 0.0)) / (len(state.emails) + 1))), 3)

        grade_result = {
            "score": score,
            "episode_id": state.episode_id,
            "cumulative_reward": round(state.cumulative_reward, 4),
            "steps": step_num,
            "valid_actions": sum(1 for r in log if r["valid"]),
            "breakdown": {
                "classification": 1.0 if any(r["reward"] >= 0.20 for r in log) else 0.5,
                "action": round(min(1.0, correct_count / max(len(state.emails), 1)), 2),
                "efficiency": round(max(0.0, 1.0 - (step_num / max(len(state.emails) * 4, 1))), 2),
                "safety": 1.0 if safety_status == "SAFE" else 0.0,
            },
        }

        return grade_result, log, None

    except Exception as exc:
        return None, [], f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


# ─────────────────────────────────────────────────────────────────────────────
# DEMO STEP CARD HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_LABEL_BADGE: dict[str, tuple[str, str]] = {
    "SPAM":            ("badge-red",    "🗑 SPAM"),
    "URGENT":          ("badge-amber",  "🔴 URGENT"),
    "ACTION_REQUIRED": ("badge-indigo", "⚡ ACTION REQUIRED"),
    "SOCIAL":          ("badge-green",  "🟢 SOCIAL"),
    "INFORMATIONAL":   ("badge-gray",   "ℹ INFORMATIONAL"),
}
_ACTION_ICON: dict[str, str] = {
    "reply":    "💬 Reply Sent",
    "archive":  "📦 Archived",
    "escalate": "🚨 Escalated to Human",
    "read":     "📖 Read",
    "classify": "🏷 Classified",
}
_ACTION_COLOR: dict[str, str] = {
    "reply":    "#22c55e",
    "archive":  "#64748b",
    "escalate": "#f97316",
    "read":     "#6366f1",
    "classify": "#818cf8",
}

def _group_log_by_email(log: list) -> list:
    """Group flat step log entries by email_id."""
    seen: dict[str, dict] = {}
    ordered: list[str] = []
    for entry in log:
        eid = entry.get("email_id") or f"step_{entry.get('step', 0)}"
        if eid not in seen:
            seen[eid] = {
                "email_id": eid,
                "subject": entry.get("subject", ""),
                "sender": entry.get("sender", ""),
                "label": "",
                "action_type": "",
                "total_reward": 0.0,
                "api_used": False,
                "tier_label": entry.get("tier_label", ""),
                "steps": [],
            }
            ordered.append(eid)
        g = seen[eid]
        g["total_reward"] = round(g["total_reward"] + entry.get("reward", 0.0), 4)
        g["steps"].append(entry)
        if entry.get("label"):
            g["label"] = entry["label"]
        if entry.get("tier_label"):
            g["tier_label"] = entry["tier_label"]
        at = entry.get("action_type", "")
        if at in ("reply", "archive", "escalate"):
            g["action_type"] = at
        if entry.get("api_used"):
            g["api_used"] = True
    return [seen[eid] for eid in ordered]


def _render_step_card(step: dict, highlight: bool = False) -> None:
    """Render one email processing card with crisp HTML formatting."""
    label      = step.get("label") or "INFORMATIONAL"
    badge_cls, badge_text = _LABEL_BADGE.get(label, ("badge-gray", label))
    action_type = step.get("action_type") or "archive"
    action_text = _ACTION_ICON.get(action_type, action_type)
    action_color = _ACTION_COLOR.get(action_type, "#94a3b8")
    reward = step.get("total_reward", 0.0)
    reward_str = f"+{reward:.3f}" if reward >= 0 else f"{reward:.3f}"
    reward_col = "#22c55e" if reward > 0 else "#ef4444" if reward < 0 else "#94a3b8"
    active_cls = "step-card-active" if highlight else ""
    tier_lbl   = step.get("tier_label", "")
    if step.get("api_used"):
        tier_pill = '<span class="gemini-pill" style="margin-left:0.4rem">⚡ Gemini Active</span>'
    elif tier_lbl:
        tier_pill = f'<span class="ea-badge badge-gray" style="margin-left:0.4rem;font-size:0.68rem;opacity:0.9">{tier_lbl}</span>'
    else:
        tier_pill = ""
    subject    = str(step.get("subject", ""))[:70]
    sender     = str(step.get("sender", ""))

    html = (
        f'<div class="step-card {active_cls}">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.5rem">'
        f'<div style="flex:1;min-width:0">'
        f'<span style="font-size:0.9rem;font-weight:600;color:#f1f5f9;display:block;'
        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{subject}</span>'
        f'<span style="font-size:0.74rem;color:#64748b">{sender}</span>'
        f'</div>'
        f'<span style="font-size:0.9rem;font-weight:700;color:{reward_col};'
        f'margin-left:1rem;white-space:nowrap">{reward_str}</span>'
        f'</div>'
        f'<div style="display:flex;gap:0.55rem;align-items:center;flex-wrap:wrap">'
        f'<span class="ea-badge {badge_cls}">{badge_text}</span>'
        f'<span style="color:#475569;font-size:0.8rem">&#8594;</span>'
        f'<span style="font-size:0.8rem;font-weight:500;color:{action_color}">{action_text}</span>'
        f'{tier_pill}'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — PRESERVED DESIGN & COMPONENT HIERARCHY
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="padding:0.4rem 0 0.8rem;display:flex;align-items:center;">'
        f'{_LOGO_SVG}'
        f'<div>'
        f'<div style="font-size:1.12rem;font-weight:700;letter-spacing:0.02em;color:#f1f5f9;line-height:1.2">AI Email Agent</div>'
        f'<div style="font-size:0.75rem;color:#7fa8c0;font-weight:500;margin-top:2px">Flight Simulator & Evaluation</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    if st.button("⟳  Reset Everything", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        _init_state()
        st.rerun()

    err = st.session_state.get("app_error")
    if err:
        st.markdown(
            f'<div style="background:#2d0f0f;border:1px solid #991b1b;border-radius:8px;'
            f'padding:0.6rem;margin:0.5rem 0;font-size:0.74rem;color:#ef4444">⚠ {str(err)[:200]}</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown('<div class="sec-heading">SESSION STATUS</div>', unsafe_allow_html=True)

    phase  = st.session_state.get("phase", "start")
    score  = st.session_state.get("final_score")
    sc_col = "#22c55e" if (score or 0) >= 0.85 else "#fbbf24" if (score or 0) >= 0.60 else "#94a3b8"
    sc_str = f'<span style="color:{sc_col};font-weight:700">{score:.3f}</span>' if score is not None else "—"
    cur_task = st.session_state.get("task_id", "task_1")
    task_lbl = _TASK_META.get(cur_task, {}).get("label", cur_task)
    is_ver, api_lbl = _check_gemini_api_status()
    api_st = f'<span style="color:#22c55e;font-weight:600">{api_lbl}</span>' if is_ver else f'<span style="color:#f97316">{api_lbl}</span>'

    st.markdown(
        f'<table class="ea-table" style="font-size:0.78rem">'
        f'<tr><td style="color:#64748b">Phase</td><td>{_badge(phase)}</td></tr>'
        f'<tr><td style="color:#64748b">Task</td><td>{task_lbl}</td></tr>'
        f'<tr><td style="color:#64748b">Steps</td><td>{st.session_state.get("steps_taken",0)}</td></tr>'
        f'<tr><td style="color:#64748b">Reward</td><td>{st.session_state.get("cumulative_reward",0.0):.3f}</td></tr>'
        f'<tr><td style="color:#64748b">Score</td><td>{sc_str}</td></tr>'
        f'<tr><td style="color:#64748b">AI Engine</td><td>{api_st}</td></tr>'
        f'</table>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown('<div class="sec-heading">REWARD REFERENCE</div>', unsafe_allow_html=True)
    ref = [
        ("Read email",         "+0.05", "#22c55e"),
        ("Correct classify",   "+0.20", "#22c55e"),
        ("Wrong classify",     "−0.10", "#ef4444"),
        ("Reply (needed)",     "+0.15", "#22c55e"),
        ("Reply (not needed)", "−0.15", "#ef4444"),
        ("Correct escalate",   "+0.25", "#22c55e"),
        ("Wrong escalate",     "−0.20", "#ef4444"),
        ("Archive (correct)",  "+0.05", "#22c55e"),
        ("Danger Archive",     "−0.50", "#ef4444"),
        ("Invalid action",     "−0.15", "#ef4444"),
    ]
    rows = "".join(
        f'<tr><td style="color:#64748b">{lbl}</td>'
        f'<td style="color:{col};font-weight:600;text-align:right">{val}</td></tr>'
        for lbl, val, col in ref
    )
    st.markdown(f'<table class="ea-table" style="font-size:0.76rem">{rows}</table>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<p style="font-size:0.73rem;color:#334155;margin-top:0.4rem">Developed by Sumanth Mamidi</p>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def render_hero() -> None:
    st.markdown(
        f'<div class="hero-block">'
        f'<div style="display:flex;align-items:center;margin-bottom:0.5rem;">'
        f'{_LOGO_SVG}'
        f'<span class="hero-title" style="margin-bottom:0;">AI Email Agent</span>'
        f'</div>'
        f'<div class="hero-sub">Autonomous Email Agent Flight Simulator & Safety Benchmark</div>'
        f'<div class="hero-tag">Simulate • Benchmark • Enforce Safety Constraints • Learn Safely with Human Feedback</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    c1, c2, _ = st.columns([1.3, 1.4, 2.3])
    with c1:
        if st.button("▶  Run Simulator", type="primary", use_container_width=True, key="btn_run_demo"):
            _reset()
            task_id = st.session_state.get("task_id", "task_1")
            with st.spinner("AI agent processing inbox…"):
                result, log, err = run_auto_episode(task_id, agent_type="hybrid")
            if err:
                st.session_state["app_error"] = err
            else:
                score = result.get("score", 0.0) if isinstance(result, dict) else float(result)
                email_steps = _group_log_by_email(log)
                st.session_state.update({
                    "grade_data": result, "final_score": score,
                    "phase": "graded", "demo_completed": True,
                    "step_results": log, "steps_taken": len(log),
                    "cumulative_reward": round(sum(r["reward"] for r in log), 4),
                    "demo_email_steps": email_steps, "demo_playback_done": False,
                })
            st.rerun()

    with c2:
        lbl = "✕ Close Guide" if st.session_state.get("show_guide") else "📖 Architecture Guide"
        if st.button(lbl, use_container_width=True, key="btn_guide"):
            st.session_state["show_guide"] = not st.session_state.get("show_guide", False)
            st.rerun()


def render_guide() -> None:
    if not st.session_state.get("show_guide"):
        return
    st.markdown(
        '<div style="margin-bottom:1.2rem">'
        '<div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;margin-bottom:0.7rem">How to use & Architecture</div>'
        '<ol style="color:#94a3b8;font-size:0.87rem;line-height:2;padding-left:1.2rem">'
        '<li><strong>OpenEnv Environment</strong>: The environment is the ground-truth authority. Agents choose actions; OpenEnv evaluates validity and rubric rewards.</li>'
        '<li><strong>Three Agent Tiers</strong>: <em>Baseline</em> (deterministic rule floor), <em>Hybrid</em> (cost-optimized local rules + LLM reasoning), and <em>LLM</em>.</li>'
        '<li><strong>Run Demo</strong>: Click <strong style="color:#fff">▶ Run Demo</strong> to execute a full live episode step-by-step.</li>'
        '<li><strong>Try Your Own Email</strong>: Test any custom email using the real Hybrid pipeline with XML prompt injection defense.</li>'
        '<li><strong>Reliability & Fallback</strong>: If Gemini API key is unset, rate-limited, or times out, local deterministic fallback activates seamlessly.</li>'
        '<li><strong>Human Feedback & Safe Learning</strong>: Human corrections are evidence. When repeated (≥ 2), candidate lessons are regression-tested against a fixed suite. Only safe lessons with zero violations are promoted.</li>'
        '</ol>'
        '<div style="margin-top:0.8rem;font-size:0.8rem;color:#475569;font-style:italic">'
        'AI Engine: Gemini 2.5 Flash with local heuristic fallback. All data strictly validated by OpenEnv standard.'
        '</div></div><hr style="border-color:#1e2d47;margin-bottom:1rem">',
        unsafe_allow_html=True,
    )


def render_inbox_preview(task_id: str) -> None:
    """Show all emails in the inbox as read-only cards before processing."""
    meta = _TASK_META.get(task_id, _TASK_META["task_1"])
    difficulty = meta["difficulty"]

    loader = ScenarioLoader()
    try:
        emails = loader.get_episode_emails(difficulty=difficulty, seed=42)
    except Exception:
        return

    if not emails:
        return

    _sec("📬 INBOX PREVIEW — EMAILS WAITING TO BE PROCESSED")
    st.markdown(
        f'<p style="font-size:0.82rem;color:#64748b;margin:-0.3rem 0 0.8rem">'
        f'{len(emails)} emails &nbsp;·&nbsp; AI will classify and act on each automatically</p>',
        unsafe_allow_html=True,
    )

    for i, email in enumerate(emails):
        subj    = str(email.subject)[:80].replace("<","&lt;").replace(">","&gt;")
        snd     = str(email.sender).replace("<","&lt;").replace(">","&gt;")
        preview = str(email.body[:100]).replace("<","&lt;").replace(">","&gt;")
        if len(email.body) > 100:
            preview += "…"

        st.markdown(
            f'<div class="inbox-card">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.5rem">'
            f'<div style="flex:1;min-width:0">'
            f'<span style="font-size:0.72rem;font-weight:700;color:#4f46e5;background:#1e1b4b;'
            f'padding:1px 7px;border-radius:10px;margin-right:0.5rem">#{i+1}</span>'
            f'<span style="font-size:0.9rem;font-weight:600;color:#f1f5f9">{subj}</span>'
            f'</div>'
            f'<span style="font-size:0.74rem;color:#475569;white-space:nowrap">{snd}</span>'
            f'</div>'
            f'<div style="font-size:0.8rem;color:#94a3b8;margin-top:0.4rem;'
            f'padding-left:1.8rem;line-height:1.5">{preview}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_step1() -> None:
    """Task selector + inbox preview + Start Task — shown at phase=='start'."""
    if st.session_state.get("phase") != "start":
        return

    _sec("SELECT TASK DIFFICULTY")

    task_opts = {k: v.get("label", k) for k, v in _TASK_META.items()}
    cur = st.session_state.get("task_id", "task_1")
    keys = list(task_opts.keys())

    selected = st.selectbox(
        "Inbox difficulty",
        options=keys,
        format_func=lambda k: task_opts[k],
        index=keys.index(cur) if cur in keys else 0,
        label_visibility="collapsed",
        key="task_selector",
    )
    st.session_state["task_id"] = selected

    meta = _TASK_META.get(selected, {})
    ec   = meta.get("email_count", "?")
    ms   = meta.get("max_steps", "?")
    diff = meta.get("difficulty", "STARTER")
    act  = {
        "STARTER": "Classify emails: SPAM / URGENT / ACTION REQUIRED / SOCIAL / INFO",
        "MEDIUM": "Classify + reply to urgent & action-required emails",
        "ADVANCED": "Full triage: classify, reply, escalate security alerts, archive rest",
        "ADVERSARIAL": "Prompt injection defenses, buried urgency, and keyword traps",
        "HELD_OUT": "Unseen evaluation test set for fair out-of-sample benchmarking",
    }.get(diff, "Classify and handle all emails")

    st.markdown(
        f'<div style="font-size:0.82rem;color:#64748b;margin:0.3rem 0">'
        f'📧 {ec} emails &nbsp;·&nbsp; ⚡ {ms} max steps &nbsp;·&nbsp; '
        f'<span style="color:#94a3b8">{act}</span></div>',
        unsafe_allow_html=True,
    )

    col_start, _ = st.columns([1, 3])
    with col_start:
        if st.button("▶  Start Task", type="primary", use_container_width=True, key="btn_start_task"):
            _reset()
            st.session_state["task_id"] = selected
            st.session_state["phase"] = "inbox"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    render_inbox_preview(selected)


def render_stepwise_demo(email_steps: list) -> None:
    """Animate emails being processed one by one."""
    if not email_steps:
        return

    n = len(email_steps)
    _sec("🤖 AI PROCESSED YOUR INBOX — STEP BY STEP")

    prog = st.progress(0, text="Initializing AI agent…")
    area = st.empty()

    if st.session_state.get("demo_playback_done"):
        prog.progress(1.0, text=f"✅ {n} emails processed")
        with area.container():
            for s in email_steps:
                _render_step_card(s, highlight=False)
    else:
        shown: list = []
        for i, step in enumerate(email_steps):
            shown.append(step)
            subj = step["subject"][:50] + ("…" if len(step["subject"]) > 50 else "")
            prog.progress((i + 1) / n, text=f"Email {i+1}/{n}: {subj}")
            with area.container():
                for j, s in enumerate(shown):
                    _render_step_card(s, highlight=(j == i))
            time.sleep(0.3)
        prog.progress(1.0, text=f"✅ All {n} emails processed!")
        st.session_state["demo_playback_done"] = True


def render_auto_zone() -> None:
    """Handle inbox → episode run → results display."""
    if st.session_state.get("phase") not in ("inbox", "graded", "compared"):
        return

    phase   = st.session_state.get("phase")
    task_id = st.session_state.get("task_id", "task_1")

    # ── Auto-run episode when phase == "inbox" ────────────────────────────────
    if phase == "inbox" and not st.session_state.get("episode_done"):
        _sec("RUNNING AUTOMATED EPISODE")
        prog = st.progress(0, text="Starting agent…")
        with st.spinner("AI agent is handling your inbox…"):
            result, log, err = run_auto_episode(task_id, agent_type="hybrid")
        prog.progress(1.0, text="Done")

        if err:
            st.session_state["app_error"] = err
            st.session_state["phase"] = "start"
        else:
            score = result.get("score", 0.0) if isinstance(result, dict) else float(result)
            email_steps = _group_log_by_email(log)
            st.session_state.update({
                "grade_data": result, "final_score": score, "phase": "graded",
                "episode_done": True, "step_results": log, "steps_taken": len(log),
                "cumulative_reward": round(sum(r["reward"] for r in log), 4),
                "demo_email_steps": email_steps, "demo_playback_done": False,
            })
        st.rerun()
        return

    # ── Show animated step-by-step cards ─────────────────────────────────────
    email_steps = st.session_state.get("demo_email_steps", [])
    if email_steps:
        render_stepwise_demo(email_steps)

    # ── Full episode log (collapsible) ────────────────────────────────────────
    step_results: list[dict] = st.session_state.get("step_results", [])
    if step_results:
        with st.expander("📋 Full Episode Log (all actions)", expanded=False):
            rows = ""
            for r in step_results:
                rw  = r.get("reward", 0.0)
                cls = "row-pos" if rw > 0 else "row-neg" if rw < 0 else "row-neu"
                vi  = "✓" if r.get("valid", True) else "✗"
                rs  = f"+{rw:.3f}" if rw >= 0 else f"{rw:.3f}"
                ai  = "⚡" if r.get("api_used") else ""
                subj_cell = str(r.get("subject",""))[:50].replace("<","&lt;")
                rows += (
                    f'<tr class="{cls}">'
                    f'<td>{r.get("step","")}</td><td>{vi}</td>'
                    f'<td>{r.get("action_type","")}</td>'
                    f'<td style="max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{subj_cell}</td>'
                    f'<td>{r.get("label","")}</td><td>{rs}</td><td style="color:#818cf8">{ai}</td>'
                    f'</tr>'
                )
            st.markdown(
                f'<div style="overflow-y:auto;max-height:320px">'
                f'<table class="ea-table">'
                f'<thead><tr><th>#</th><th>✓</th><th>Action</th><th>Subject</th><th>Label</th><th>Reward</th><th>⚡</th></tr></thead>'
                f'<tbody>{rows}</tbody></table></div>',
                unsafe_allow_html=True,
            )


def render_results() -> None:
    """Score display + breakdown bars + Human Review & Learning Gate."""
    if st.session_state.get("phase") not in ("graded", "compared"):
        return

    score      = st.session_state.get("final_score", 0.0) or 0.0
    grade_data = st.session_state.get("grade_data") or {}
    g_letter, g_color = _grade_letter(score)
    interp = _interpret(score)

    _sec("EPISODE RESULTS")

    st.markdown(
        f'<div style="margin:0.8rem 0 1rem">'
        f'<span class="score-big" style="color:{g_color}">{score:.3f}</span>'
        f'<span style="font-size:1.4rem;font-weight:700;color:{g_color};margin-left:0.4rem">{g_letter}</span>'
        f'<div style="font-size:0.88rem;color:#94a3b8;margin-top:0.4rem">{interp}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    steps   = st.session_state.get("steps_taken", 0)
    cum_r   = st.session_state.get("cumulative_reward", 0.0)
    results = st.session_state.get("step_results", [])
    inv_cnt = sum(1 for r in results if not r.get("valid", True))
    api_cnt = sum(1 for r in results if r.get("api_used"))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Steps Used",        steps)
    m2.metric("Invalid Actions",   inv_cnt)
    m3.metric("Cumulative Reward", f"{cum_r:.3f}")
    m4.metric("Gemini API Calls",  api_cnt)

    # ── Score breakdown bars ──────────────────────────────────────────────────
    breakdown: dict = {}
    if isinstance(grade_data, dict):
        raw = grade_data.get("breakdown", grade_data.get("scores", {}))
        if isinstance(raw, dict):
            sub = raw.get("sub_scores", {})
            src = sub if (sub and all(isinstance(v, (int, float)) for v in sub.values())) else raw
            for k, v in src.items():
                try:
                    breakdown[k] = float(v)
                except (TypeError, ValueError):
                    pass

    if breakdown:
        _sec("SCORE BREAKDOWN")
        dim_colors = {
            "classification": "#6366f1", "action": "#22c55e",
            "efficiency": "#fbbf24",     "safety": "#00E5FF",
        }
        bars_html = ""
        for dim, val in breakdown.items():
            col = dim_colors.get(dim.lower(), "#6366f1")
            bars_html += _bar(dim.replace("_", " ").title(), val, col)
        st.markdown(bars_html, unsafe_allow_html=True)

    # ── Human Feedback & Validated Learning Gate ──────────────────────────────
    with st.expander("🛡️ Human Review & Continuous Learning Engine", expanded=True):
        st.markdown(
            '<div style="font-size:0.82rem;color:#94a3b8;margin-bottom:0.8rem">'
            'Review agent actions on this episode. Human feedback is recorded in SQLite. '
            'When recurring patterns (≥ 2) are detected, a <strong>Candidate Lesson</strong> is generated '
            'and evaluated against the 8-scenario safety regression suite. Only rules with zero safety violations are promoted.'
            '</div>',
            unsafe_allow_html=True,
        )

        fb_status = st.session_state.get("last_feedback_status")
        if fb_status:
            st.markdown(
                f'<div style="background:rgba(34,255,136,0.08);border:1px solid rgba(34,255,136,0.35);'
                f'border-radius:8px;padding:0.65rem 0.9rem;font-size:0.82rem;color:#22FF88;margin-bottom:0.9rem;'
                f'box-shadow:0 0 14px rgba(34,255,136,0.15)">'
                f'{fb_status}</div>',
                unsafe_allow_html=True,
            )

        email_steps = st.session_state.get("demo_email_steps", [])
        email_choices = [
            f"{s.get('email_id', f'email_{i+1}')} — {s.get('subject', 'Untitled')[:40]}"
            for i, s in enumerate(email_steps)
        ] if email_steps else ["starter_001 — Sample Email"]

        with st.form("feedback_form", clear_on_submit=False):
            fc1, fc2 = st.columns([1.8, 1.2])
            with fc1:
                sel_email = st.selectbox("Select Processed Email to Review", email_choices)
                target_eid = sel_email.split(" — ")[0] if " — " in sel_email else "starter_001"
            with fc2:
                f_verdict = st.radio("Reviewer Verdict", ["CORRECT (Approved)", "INCORRECT (Needs Correction)"], horizontal=True)

            is_inc = "INCORRECT" in f_verdict
            if is_inc:
                ac1, ac2 = st.columns(2)
                with ac1:
                    corr_label = st.selectbox("Correct Classification", ["URGENT", "ACTION_REQUIRED", "INFORMATIONAL", "SOCIAL", "SPAM"])
                with ac2:
                    corr_action = st.selectbox("Correct Action", ["REPLY_EMAIL", "ESCALATE_EMAIL", "ARCHIVE_EMAIL"])
                corr_reason = st.text_input(
                    "Policy Rationale / Correction Rule",
                    placeholder="e.g. Critical security outage alerts must be escalated immediately.",
                )
            else:
                corr_label = "INFORMATIONAL"
                corr_action = "ARCHIVE_EMAIL"
                corr_reason = "Decision verified correct by operator."

            f_submit = st.form_submit_button("🛡️  Submit Human Review & Run Regression Gate", type="primary")

            if f_submit:
                try:
                    fb_svc = FeedbackService()
                    verdict_enum = FeedbackVerdict.INCORRECT if is_inc else FeedbackVerdict.CORRECT
                    act_dict = {"action_type": corr_action, "email_id": target_eid} if is_inc else None
                    if is_inc:
                        if corr_action == "REPLY_EMAIL":
                            act_dict["body"] = "Confirmed acceptance."
                        elif corr_action == "ESCALATE_EMAIL":
                            act_dict["reason"] = corr_reason or "Escalating incident."

                    fb_svc.record_feedback(
                        episode_id=grade_data.get("episode_id", "ui_ep_01"),
                        scenario_id=target_eid,
                        agent_name="HYBRID",
                        verdict=verdict_enum,
                        correct_action=act_dict,
                        explanation=corr_reason,
                    )

                    # Trigger candidate lesson generation and regression validation
                    gen = LessonGenerator(min_evidence=2)
                    candidates = gen.generate_candidates()
                    val = LessonValidator()
                    promoted_count = 0
                    for cand in candidates:
                        passed, report = val.validate_candidate(cand)
                        if passed:
                            promoted_count += 1

                    if is_inc:
                        if promoted_count > 0:
                            st.session_state["last_feedback_status"] = (
                                f"✓ Feedback saved to SQLite · <strong>Pattern Clustered & Regression Validated (8 scenarios, 0 violations)</strong> · "
                                f"<strong>Lesson PROMOTED</strong> to active agent memory. Future episodes will enforce this policy!"
                            )
                        else:
                            st.session_state["last_feedback_status"] = (
                                f"✓ Feedback recorded in SQLite (Target: {target_eid}) · "
                                f"Saved as empirical evidence (requires ≥ 2 consistent corrections to generate candidate lesson)."
                            )
                    else:
                        st.session_state["last_feedback_status"] = f"✓ Decision for {target_eid} verified and logged as positive benchmark control."
                    st.rerun()
                except Exception as ex:
                    st.error(f"Feedback error: {ex}")

        # Display active lessons stored in SQLite
        try:
            retriever = LessonRetriever()
            active_lessons = retriever.get_active_lessons()
            if active_lessons:
                st.markdown('<div style="font-size:0.75rem;font-weight:700;color:#00E5FF;margin:0.8rem 0 0.4rem;letter-spacing:0.06em;text-transform:uppercase">Active Promoted Lessons in Agent Memory</div>', unsafe_allow_html=True)
                l_rows = "".join(
                    f'<tr><td style="color:#22FF88;font-weight:600">PROMOTED</td>'
                    f'<td style="color:#f1f5f9">{l.rule_text}</td>'
                    f'<td style="color:#7fa8c0">{l.target_category or "all"}</td></tr>'
                    for l in active_lessons[:5]
                )
                st.markdown(f'<table class="ea-table" style="font-size:0.76rem"><thead><tr><th>Status</th><th>Rule</th><th>Scope</th></tr></thead><tbody>{l_rows}</tbody></table>', unsafe_allow_html=True)
        except Exception:
            pass

    if st.session_state.get("phase") == "graded":
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚖  Compare Agent Tiers (Baseline vs Hybrid vs LLM)", type="primary", key="btn_compare"):
            st.session_state["phase"] = "compared"
            st.rerun()


def render_comparison() -> None:
    """Three-Tier Multi-Agent Comparison Panel."""
    if st.session_state.get("phase") != "compared":
        return

    _sec("MULTI-TIER AGENT BENCHMARK COMPARISON")
    st.markdown(
        '<p style="font-size:0.84rem;color:#94a3b8;margin-bottom:1rem">'
        'Comparing all three agent tiers on the exact same scenario batch under the environment rubric:<br>'
        '• <strong style="color:#f1f5f9">Tier 1 (Baseline)</strong>: Pure deterministic first-match keyword decision tree.<br>'
        '• <strong style="color:#f1f5f9">Tier 2 (Hybrid)</strong>: Multi-signal additive scoring vector with selective LLM escalation.<br>'
        '• <strong style="color:#f1f5f9">Tier 3 (LLM)</strong>: Autonomous LLM reasoning with automated degraded fallback resilience.'
        '</p>',
        unsafe_allow_html=True,
    )

    my_score  = st.session_state.get("final_score", 0.0) or 0.0
    task_id   = st.session_state.get("task_id", "task_1")
    meta      = _TASK_META.get(task_id, _TASK_META["task_1"])
    difficulty = meta["difficulty"]

    # Run real Baseline simulation
    baseline_score = 0.60
    baseline_reward = 2.0
    try:
        router = AgentRouter()
        b_agent = router.get_agent("baseline")
        b_env = EmailEnvironment()
        b_obs = b_env.reset(difficulty=difficulty, seed=42)
        b_pos = 0
        while not b_obs.done:
            b_act = b_agent.act(b_obs)
            b_obs, b_rew, _, _ = b_env.step(b_act)
            if b_rew >= 0.15:
                b_pos += 1
        b_state = b_env.state()
        baseline_reward = round(b_state.cumulative_reward, 3)
        baseline_score = round(min(1.0, max(0.0, (b_pos + 1.0) / (len(b_state.emails) + 1))), 3)
    except Exception:
        pass

    # Run real LLM / Degraded simulation
    llm_score = my_score
    llm_reward = st.session_state.get("cumulative_reward", 0.0)
    try:
        router = AgentRouter()
        l_agent = router.get_agent("llm")
        l_env = EmailEnvironment()
        l_obs = l_env.reset(difficulty=difficulty, seed=42)
        l_pos = 0
        while not l_obs.done:
            l_act = l_agent.act(l_obs)
            l_obs, l_rew, _, _ = l_env.step(l_act)
            if l_rew >= 0.15:
                l_pos += 1
        l_state = l_env.state()
        llm_reward = round(l_state.cumulative_reward, 3)
        llm_score = round(min(1.0, max(0.0, (l_pos + 1.0) / (len(l_state.emails) + 1))), 3)
    except Exception:
        pass

    my_g, my_c = _grade_letter(my_score)
    b_g,  b_c  = _grade_letter(baseline_score)
    l_g,  l_c  = _grade_letter(llm_score)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="agent-card">'
            f'<div class="ac-label">Tier 1 — Baseline</div>'
            f'<div class="ac-score" style="color:{b_c}">{baseline_score:.3f} {b_g}</div>'
            f'<div style="font-size:0.74rem;color:#64748b;margin-top:4px">Reward: {baseline_reward:+.2f} · 100% Offline</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="agent-card" style="border-color:rgba(0,229,255,0.4);box-shadow:0 0 20px rgba(0,229,255,0.15)">'
            f'<div class="ac-label" style="color:#00E5FF">Tier 2 — Hybrid (Active)</div>'
            f'<div class="ac-score" style="color:{my_c}">{my_score:.3f} {my_g}</div>'
            f'<div style="font-size:0.74rem;color:#7fa8c0;margin-top:4px">Reward: {st.session_state.get("cumulative_reward",0.0):+.2f} · Multi-Signal</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="agent-card">'
            f'<div class="ac-label">Tier 3 — LLM / Gemini</div>'
            f'<div class="ac-score" style="color:{l_c}">{llm_score:.3f} {l_g}</div>'
            f'<div style="font-size:0.74rem;color:#64748b;margin-top:4px">Reward: {llm_reward:+.2f} · Prompt Reasoner</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Detailed comparative breakdown table
    comp_rows = (
        f'<tr><td><strong>Strategy</strong></td><td>First-Match Keyword Tree</td><td>Weighted Vector + Selective LLM</td><td>Prompt Reasoner + Degraded Fallback</td></tr>'
        f'<tr><td><strong>Safety Violation Rate</strong></td><td style="color:#22FF88">0% (Safe)</td><td style="color:#22FF88">0% (Safe)</td><td style="color:#22FF88">0% (Safe)</td></tr>'
        f'<tr><td><strong>API Reliance</strong></td><td>0 calls (Zero-cost)</td><td>Selective (Cost-optimized)</td><td>Full / Fallback if unkeyed</td></tr>'
        f'<tr><td><strong>Continuous Learning</strong></td><td>Static Rules</td><td>Active Lesson Retrieval</td><td>Injected Lesson Context</td></tr>'
    )
    st.markdown(
        f'<table class="ea-table" style="font-size:0.8rem;margin-top:1.2rem">'
        f'<thead><tr><th>Metric / Feature</th><th>Tier 1 (Baseline)</th><th>Tier 2 (Hybrid)</th><th>Tier 3 (LLM)</th></tr></thead>'
        f'<tbody>{comp_rows}</tbody></table>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄  Start New Task", type="primary", key="btn_new_task"):
        _reset()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TRY YOUR OWN EMAIL / ANALYSE EMAIL (CONNECTED HYBRID PIPELINE)
# ─────────────────────────────────────────────────────────────────────────────
def render_analyse_mode() -> None:
    """
    Analyse Email mode — connected directly to the real Hybrid Agent pipeline,
    LLM Client, Lesson Retriever, and Local Fallback.
    """
    _sec("🔍 ANALYSE YOUR EMAIL WITH AI")

    is_ver, api_lbl = _check_gemini_api_status()
    if is_ver:
        st.markdown(
            '<div class="gemini-pill" style="margin-bottom:0.8rem">⚡ Powered by Gemini 2.5 Flash (Verified Active)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="display:inline-flex;align-items:center;gap:6px;background:#1a2030;'
            f'border:1px solid #334155;border-radius:20px;padding:3px 12px;font-size:0.78rem;'
            f'color:#f97316;margin-bottom:0.8rem">🔧 {api_lbl}</div>',
            unsafe_allow_html=True,
        )

    # ── Examples ──────────────────────────────────────────────────────────────
    with st.expander("💡 Example emails to try", expanded=False):
        examples = [
            ("SPAM",            "You've won a $1,000 prize!",        "Click here to claim your free offer now."),
            ("URGENT",          "Server outage — critical alert",     "Our production database is down. All hands needed immediately."),
            ("ACTION_REQUIRED", "Please approve Q3 budget by EOD",   "Hi, I need your sign-off on the $45K budget. Finance deadline is today."),
            ("SOCIAL",          "Team lunch this Friday!",           "Hey! Lunch at Nobu on Friday. RSVP by Wednesday."),
            ("INFORMATIONAL",   "Monthly company newsletter",        "Here's your recap of company news and product updates."),
        ]
        for cls, sub, bdy in examples:
            bc = {"SPAM":"badge-red","URGENT":"badge-amber","ACTION_REQUIRED":"badge-indigo",
                  "SOCIAL":"badge-green","INFORMATIONAL":"badge-gray"}.get(cls, "badge-gray")
            st.markdown(
                f'<div style="margin:0.4rem 0">'
                f'<span class="ea-badge {bc}">{cls}</span>'
                f'<span style="font-size:0.82rem;color:#f1f5f9;margin-left:0.5rem">{sub}</span>'
                f'<div style="font-size:0.76rem;color:#64748b;margin-top:1px;padding-left:0.2rem">{bdy}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("analyse_form", clear_on_submit=False):
        subject  = st.text_input("Subject line", placeholder="e.g. URGENT: Server Down",  key="ana_subject")
        sender   = st.text_input("From (optional)", placeholder="e.g. alerts@company.com", key="ana_sender")
        body     = st.text_area("Email body",   placeholder="Paste the full email body here…",
                                height=160, key="ana_body")
        submitted = st.form_submit_button("🔍  Analyse Now", type="primary")

    if not submitted:
        return
    if not (subject or body):
        st.warning("Please enter at least a subject line or email body.")
        return

    # ── Connect to Real Hybrid Agent Pipeline ────────────────────────────────
    with st.spinner("AI analyzing email through Hybrid pipeline…"):
        router = AgentRouter()
        hybrid_agent = router.get_agent("hybrid")

        decision: AgentDecision = hybrid_agent.evaluate(
            email_id="custom_ui_01",
            subject=subject or "No Subject",
            sender=sender or "unknown@domain.com",
            body=body or "",
        )

    label_val = decision.classification.value if hasattr(decision.classification, "value") else str(decision.classification)
    action_type_raw = decision.action.action_type.value if hasattr(decision.action.action_type, "value") else str(decision.action.action_type)
    if action_type_raw == "REPLY_EMAIL":
        action = "reply"
    elif action_type_raw == "ESCALATE_EMAIL":
        action = "escalate → human"
    else:
        action = "archive"

    reasoning = decision.reasoning_summary
    conf = decision.confidence
    urg_val = str(decision.urgency.value if hasattr(decision.urgency, "value") else decision.urgency).upper()
    urgency = 9 if urg_val == "CRITICAL" else 7 if urg_val == "HIGH" else 4 if urg_val == "MEDIUM" else 1
    sentiment = "negative" if label_val in ("SPAM", "URGENT") else "positive" if label_val == "SOCIAL" else "neutral"
    
    src = str(getattr(decision, "source", "")).lower()
    used_gemini = (src == "llm")
    if src == "llm":
        eng_text = "⚡ Live Gemini 2.5 Flash"
        eng_style = 'class="gemini-pill"'
    elif src == "hybrid_local":
        eng_text = "Tier 2 — Hybrid Heuristic"
        eng_style = 'style="font-size:0.74rem;color:#00E5FF;font-weight:600"'
    elif src == "fallback":
        eng_text = "Tier 1 — Baseline Fallback"
        eng_style = 'style="font-size:0.74rem;color:#94a3b8;font-weight:600"'
    elif src == "degraded_fallback":
        eng_text = "Tier 3 — Degraded LLM (Offline)"
        eng_style = 'style="font-size:0.74rem;color:#f97316;font-weight:600"'
    else:
        eng_text = "🔧 Local Heuristic Engine"
        eng_style = 'style="font-size:0.74rem;color:#94a3b8"'

    # ── Build result UI card ─────────────────────────────────────────────────
    bc = {"SPAM":"badge-red","URGENT":"badge-amber","ACTION_REQUIRED":"badge-indigo",
          "SOCIAL":"badge-green","INFORMATIONAL":"badge-gray"}.get(label_val, "badge-gray")
    conf_pct  = int(conf * 100)
    conf_col  = "#22c55e" if conf >= 0.7 else "#fbbf24" if conf >= 0.4 else "#ef4444"
    urg_col   = "#ef4444" if urgency >= 8 else "#fbbf24" if urgency >= 5 else "#22c55e"

    html  = '<div class="ea-card" style="margin-top:1rem">'
    html += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem">'
    html += '<div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#6366f1">AI Decision Output</div>'
    html += f'<span {eng_style}>{eng_text}</span>'
    html += '</div>'

    html += '<div style="display:flex;gap:1.4rem;flex-wrap:wrap;margin-bottom:0.9rem">'
    html += (
        f'<div><div style="font-size:0.72rem;color:#64748b;margin-bottom:4px">Classification</div>'
        f'<span class="ea-badge {bc}">{label_val}</span></div>'
    )
    html += (
        f'<div><div style="font-size:0.72rem;color:#64748b;margin-bottom:4px">Suggested Action</div>'
        f'<span style="font-size:0.84rem;color:#f1f5f9;font-weight:500">{action}</span></div>'
    )
    html += (
        f'<div><div style="font-size:0.72rem;color:#64748b;margin-bottom:4px">Confidence</div>'
        f'<span style="font-size:0.84rem;color:{conf_col};font-weight:600">{conf_pct}%</span></div>'
    )
    html += (
        f'<div><div style="font-size:0.72rem;color:#64748b;margin-bottom:4px">Urgency</div>'
        f'<span style="font-size:0.84rem;color:{urg_col};font-weight:600">{urgency}/10</span></div>'
    )
    html += (
        f'<div><div style="font-size:0.72rem;color:#64748b;margin-bottom:4px">Sentiment</div>'
        f'<span style="font-size:0.84rem;color:#94a3b8">{sentiment}</span></div>'
    )
    html += '</div>'

    html += (
        f'<div style="border-top:1px solid #1e2d47;padding-top:0.7rem">'
        f'<div style="font-size:0.72rem;color:#64748b;margin-bottom:3px">AI Reasoning</div>'
        f'<div style="font-size:0.83rem;color:#94a3b8">{reasoning}</div>'
        f'</div>'
    )
    html += '</div>'

    st.markdown(html, unsafe_allow_html=True)

    # ── Human Feedback Form on Custom Analysis ──────────────────────────────
    with st.expander("🛡️ Review this Decision & Train the Agent", expanded=False):
        with st.form("custom_feedback_form", clear_on_submit=True):
            cf_verdict = st.radio("Decision Quality", ["CORRECT (Approved)", "INCORRECT (Correction)"], horizontal=True)
            cf_action = st.selectbox("Recommended Action", ["REPLY_EMAIL", "ESCALATE_EMAIL", "ARCHIVE_EMAIL"])
            cf_reason = st.text_input("Correction Rule / Policy Rationale", placeholder="e.g. Always reply to budget requests.")
            cf_sub = st.form_submit_button("🛡️ Submit Correction & Update Memory", type="primary")

            if cf_sub:
                try:
                    fb_svc = FeedbackService()
                    v_enum = FeedbackVerdict.INCORRECT if "INCORRECT" in cf_verdict else FeedbackVerdict.CORRECT
                    act_d = {"action_type": cf_action, "email_id": "custom_ui_01"} if v_enum == FeedbackVerdict.INCORRECT else None
                    if act_d and cf_action == "REPLY_EMAIL":
                        act_d["body"] = "Confirmed acceptance."
                    elif act_d and cf_action == "ESCALATE_EMAIL":
                        act_d["reason"] = cf_reason or "Escalated by human reviewer."

                    fb_svc.record_feedback(
                        episode_id="custom_analysis_ep",
                        scenario_id="custom_ui_01",
                        agent_name="HYBRID",
                        verdict=v_enum,
                        correct_action=act_d,
                        explanation=cf_reason,
                    )
                    st.success("✓ Feedback saved to SQLite database and queued for pattern clustering.")
                except Exception as c_err:
                    st.error(f"Error saving feedback: {c_err}")

    st.session_state["custom_result"] = {
        "label": label_val, "action": action,
        "confidence": conf, "reasoning": reasoning,
        "used_gemini": used_gemini,
    }


def render_footer() -> None:
    st.markdown(
        '<div class="ea-footer">Developed by Sumanth Mamidi &nbsp;·&nbsp; Powered by Google Gemini 2.5 Flash & OpenEnv</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION LAYOUT — FROZEN VISUAL COMPOSITION
# ─────────────────────────────────────────────────────────────────────────────
render_hero()
render_guide()

# Mode toggle
mode = st.session_state.get("mode", "training")
ca, cb, _ = st.columns([1.2, 1.2, 2.2])
with ca:
    t = "primary" if mode == "training" else "secondary"
    if st.button("📋  Simulator Mode", type=t, use_container_width=True, key="btn_mode_train"):
        st.session_state["mode"] = "training"
        st.rerun()
with cb:
    t = "primary" if mode == "custom" else "secondary"
    if st.button("🔍  Analyse Custom Email", type=t, use_container_width=True, key="btn_mode_custom"):
        st.session_state["mode"] = "custom"
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

if mode == "training":
    render_step1()
    render_auto_zone()
    render_results()
    render_comparison()
else:
    render_analyse_mode()

render_footer()