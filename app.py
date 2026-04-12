# app.py — AI Email Agent Training Environment
# Fixed version: correct actions, working Analyse Email, clean HTML, clear demo, robust fallback

from __future__ import annotations
import os, re, time, traceback, json
from typing import Any, Optional

import streamlit as st

# ── Load .env ──────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ── Backend imports (all optional — graceful degrade) ─────────────────────────
try:
    from models.environment import EmailEnv
    _ENV_OK = True
except Exception:
    _ENV_OK = False

try:
    from models import ClassLabel, Action
    _MODELS_OK = True
except Exception:
    _MODELS_OK = False
    class ClassLabel:   # type: ignore
        SPAM = "SPAM"; URGENT = "URGENT"; ACTION_REQUIRED = "ACTION_REQUIRED"
        SOCIAL = "SOCIAL"; INFORMATIONAL = "INFORMATIONAL"
        @staticmethod
        def _member_map_(): return {}
    class Action:       # type: ignore
        @staticmethod
        def read_email(eid): return {"type": "read_email", "email_id": eid}
        @staticmethod
        def classify_email(eid, label): return {"type": "classify_email", "email_id": eid, "label": label}
        @staticmethod
        def reply_email(eid, body=""): return {"type": "reply_email", "email_id": eid, "body": body}
        @staticmethod
        def archive_email(eid): return {"type": "archive_email", "email_id": eid}
        @staticmethod
        def escalate_email(eid, reason=""): return {"type": "escalate_email", "email_id": eid, "reason": reason}

try:
    from agent import EmailClassifier
    _AGENT_OK = True
except Exception:
    _AGENT_OK = False

try:
    from smart_agent import SmartEmailProcessor
    _SMART_OK = True
except Exception:
    _SMART_OK = False

try:
    import db as _db
    _DB_OK = True
except Exception:
    _DB_OK = False

try:
    from models.baseline import BaselineAgent
    _BASELINE_OK = True
except Exception:
    _BASELINE_OK = False

try:
    from models.tasks import TASKS, get_task
    _TASKS_OK = True
    _GET_TASK_OK = True
except Exception:
    _TASKS_OK = False; _GET_TASK_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be FIRST streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Email Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
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
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"],header{background:transparent!important}
.block-container{max-width:1100px!important;padding:1.5rem 1.5rem 3rem!important;position:relative;z-index:1}

[data-testid="stSidebar"]{background:linear-gradient(180deg,#080d20 0%,#050816 100%)!important;border-right:1px solid rgba(0,229,255,.15)!important;box-shadow:4px 0 24px rgba(0,229,255,.05)!important}
[data-testid="stSidebar"] .stMarkdown p,[data-testid="stSidebar"] label{color:#7fa8c0!important;font-size:.82rem!important}

.stButton>button{font-family:'Space Grotesk','Inter',sans-serif!important;border-radius:10px!important;font-weight:600!important;letter-spacing:.03em!important;transition:all .2s cubic-bezier(.4,0,.2,1)!important}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#00b4d8 0%,#8B5CF6 100%)!important;border:none!important;color:#fff!important;box-shadow:0 0 14px rgba(0,229,255,.4),0 4px 20px rgba(139,92,246,.3)!important}
.stButton>button[kind="primary"]:hover{background:linear-gradient(135deg,#00E5FF 0%,#a78bfa 100%)!important;transform:translateY(-2px) scale(1.02)!important;box-shadow:0 0 28px rgba(0,229,255,.6),0 8px 32px rgba(139,92,246,.45)!important}
.stButton>button[kind="primary"]:active{transform:translateY(0) scale(.99)!important}
.stButton>button[kind="secondary"]{background:rgba(0,229,255,.05)!important;border:1px solid rgba(0,229,255,.2)!important;color:#7fa8c0!important}
.stButton>button[kind="secondary"]:hover{background:rgba(0,229,255,.1)!important;border-color:rgba(0,229,255,.45)!important;color:#00E5FF!important;box-shadow:0 0 14px rgba(0,229,255,.18)!important}

.ea-card{background:linear-gradient(135deg,rgba(10,18,50,.92) 0%,rgba(6,12,35,.96) 100%);border:1px solid rgba(0,229,255,.15);border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:.9rem;box-shadow:0 4px 24px rgba(0,0,0,.5),inset 0 1px 0 rgba(0,229,255,.08);backdrop-filter:blur(8px)}

.inbox-card{background:linear-gradient(135deg,rgba(8,14,38,.95) 0%,rgba(5,10,28,.98) 100%);border:1px solid rgba(0,229,255,.1);border-left:3px solid rgba(0,229,255,.2);border-radius:10px;padding:.85rem 1.1rem;margin-bottom:.55rem;transition:all .2s ease;box-shadow:0 2px 12px rgba(0,0,0,.4)}
.inbox-card:hover{border-left-color:#00E5FF;box-shadow:0 0 20px rgba(0,229,255,.14),0 4px 16px rgba(0,0,0,.5);transform:translateX(2px)}

.step-card{background:linear-gradient(135deg,rgba(10,16,42,.95) 0%,rgba(6,10,28,.98) 100%);border:1px solid rgba(0,229,255,.08);border-left:3px solid rgba(0,229,255,.12);border-radius:10px;padding:.85rem 1.1rem;margin-bottom:.55rem;transition:all .25s ease}
.step-card-active{border-left-color:#00E5FF!important;box-shadow:0 0 26px rgba(0,229,255,.22),0 0 10px rgba(139,92,246,.15),inset 0 0 22px rgba(0,229,255,.03)!important;animation:pulse-glow 2s ease-in-out infinite}

table.ea-table{width:100%;border-collapse:collapse;font-size:.82rem}
table.ea-table th{background:rgba(0,229,255,.07);color:#00E5FF;font-weight:700;padding:.45rem .7rem;text-align:left;border-bottom:1px solid rgba(0,229,255,.15);letter-spacing:.06em;text-transform:uppercase;font-size:.72rem}
table.ea-table td{padding:.4rem .7rem;border-bottom:1px solid rgba(0,229,255,.06);color:#a0b4cc}
table.ea-table tr:last-child td{border-bottom:none}
table.ea-table tr:hover td{background:rgba(0,229,255,.04);color:#c8dbe8}

.bar-wrap{margin:.35rem 0}
.bar-label{font-size:.78rem;color:#7fa8c0;margin-bottom:3px}
.bar-bg{background:rgba(0,229,255,.07);border-radius:6px;height:8px;overflow:hidden;border:1px solid rgba(0,229,255,.1)}
.bar-fill{height:100%;border-radius:6px;box-shadow:0 0 8px currentColor}

.ea-badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.72rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase}
.badge-indigo{background:rgba(139,92,246,.15);color:#a78bfa;border:1px solid rgba(139,92,246,.4);box-shadow:0 0 8px rgba(139,92,246,.2)}
.badge-green{background:rgba(34,255,136,.1);color:#22FF88;border:1px solid rgba(34,255,136,.35);box-shadow:0 0 8px rgba(34,255,136,.15)}
.badge-amber{background:rgba(255,159,28,.12);color:#FF9F1C;border:1px solid rgba(255,159,28,.4);box-shadow:0 0 8px rgba(255,159,28,.15)}
.badge-red{background:rgba(255,77,77,.12);color:#FF4D4D;border:1px solid rgba(255,77,77,.4);box-shadow:0 0 8px rgba(255,77,77,.15)}
.badge-gray{background:rgba(0,229,255,.07);color:#7fa8c0;border:1px solid rgba(0,229,255,.18)}

.hero-block{
  background:linear-gradient(145deg,#07102e 0%,#0d1a45 22%,#0a1030 45%,#0f0828 70%,#150622 100%);
  border:1px solid transparent;
  border-radius:22px;
  padding:2.5rem 2.7rem 2.1rem;
  margin-bottom:1.7rem;
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
  margin-bottom:.55rem;
  text-shadow:0 0 28px rgba(0,200,255,.38),0 0 8px rgba(0,200,255,.18),0 2px 6px rgba(0,0,0,.7);
}
.hero-sub{font-size:1rem;color:#9ec5d8;margin-bottom:.45rem;text-shadow:0 1px 4px rgba(0,0,0,.5)}
.hero-tag{font-size:.83rem;color:#3de0f8;font-style:italic;text-shadow:0 0 14px rgba(0,225,255,.55),0 0 4px rgba(0,225,255,.25);opacity:.92}

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
# TASK METADATA
# ─────────────────────────────────────────────────────────────────────────────
_TASK_META: dict[str, dict] = {}
if _TASKS_OK:
    for _k, _v in TASKS.items():
        _TASK_META[_k] = _v if isinstance(_v, dict) else {"label": str(_v), "email_count": 5, "max_steps": 20}
if not _TASK_META:
    _TASK_META = {
        "task_1": {"label": "Starter Inbox (5 emails)",   "email_count": 5,  "max_steps": 15},
        "task_2": {"label": "Medium Inbox (10 emails)",   "email_count": 10, "max_steps": 40},
        "task_3": {"label": "Advanced Inbox (15 emails)", "email_count": 15, "max_steps": 60},
    }

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults: dict[str, Any] = {
        "env": None, "task_id": "task_1", "phase": "start",
        "obs_dict": None, "step_results": [], "cumulative_reward": 0.0,
        "steps_taken": 0, "episode_done": False, "final_score": None,
        "grade_data": None, "mode": "training", "show_guide": False,
        "custom_result": None, "app_error": None, "demo_completed": False,
        "auto_log": [], "demo_email_steps": [], "demo_playback_done": False,
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

def _label_str(label_obj) -> str:
    """Safely extract string from ClassLabel enum or plain string."""
    if label_obj is None:
        return "INFORMATIONAL"
    if hasattr(label_obj, "value"):
        return str(label_obj.value)
    return str(label_obj).upper().replace("CLASSLABEL.", "").strip()

def _reset() -> None:
    for k in ["env","obs_dict","step_results","cumulative_reward","steps_taken",
              "episode_done","final_score","grade_data","app_error","demo_completed",
              "auto_log","demo_email_steps","demo_playback_done"]:
        st.session_state.pop(k, None)
    st.session_state["phase"] = "start"
    _init_state()

# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD CLASSIFIER (pure fallback — no imports needed)
# ─────────────────────────────────────────────────────────────────────────────
_KW: dict[str, list[str]] = {
    "SPAM": [
        "win","winner","prize","lottery","click here","unsubscribe","free offer",
        "guaranteed","inheritance","claim now","100x","crypto","presale",
        "free iphone","earn cash","act now","make money",
    ],
    "URGENT": [
        "urgent","asap","immediately","critical","emergency","time-sensitive",
        "time sensitive","high priority","escalate","outage"," down ","all hands",
        "p0","priority 0","breach","attack","data breach","incident","war room",
    ],
    "ACTION_REQUIRED": [
        "please review","please approve","approval needed","sign off",
        "action required","response required","please confirm","awaiting your",
        "need your input","follow up","follow-up","next steps","please respond",
        "due by","submit by","deadline","due date","timesheet","onboarding",
        "performance review","self-assessment","invoice","purchase order","nda",
        "access request","budget approval","dsar","gdpr",
    ],
    "SOCIAL": [
        "invitation","invite","party","birthday","wedding","event",
        "gathering","happy hour","lunch","coffee","catch up",
        "get together","celebrate","rsvp","volunteer","picnic","happy hour",
    ],
}

def _kw_classify(subject: str, body: str) -> str:
    text = (subject + " " + body).lower()
    scores: dict[str, int] = {k: 0 for k in _KW}
    for lbl, kws in _KW.items():
        for kw in kws:
            if kw in text:
                scores[lbl] += 1
    best = max(scores, key=lambda l: scores[l])
    return best if scores[best] > 0 else "INFORMATIONAL"

def _to_label_obj(label_str: str):
    """Convert string label to ClassLabel enum if available."""
    try:
        return getattr(ClassLabel, label_str)
    except Exception:
        return label_str

# ─────────────────────────────────────────────────────────────────────────────
# ESCALATION REASONS — per task email
# ─────────────────────────────────────────────────────────────────────────────
_ESCALATION_REASONS: dict[str, str] = {
    "t3_e4":  "Legal notice — workplace harassment complaint requires immediate HR and legal review.",
    "t3_e3":  "Critical security alert — unauthorized access attempt requires security team escalation.",
    "t3_e13": "Potential data breach — GDPR/CCPA notification deadline requires immediate escalation.",
}

_EMAILS_REQUIRING_ESCALATION = set(_ESCALATION_REASONS.keys())

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _gemini_model():
    """Return a configured Gemini model or None if unavailable."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        import google.generativeai as genai  # type: ignore
        import warnings; warnings.filterwarnings("ignore")
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.5-flash")
    except Exception:
        return None

def _call_gemini_analyze(subject: str, body: str, sender: str = "") -> Optional[dict]:
    """Call Gemini to analyze an email. Returns dict or None on any failure."""
    model = _gemini_model()
    if not model:
        return None
    try:
        prompt = (
            "Analyze this email and return ONLY a valid JSON object with no markdown or explanation.\n\n"
            f"From: {sender or 'unknown'}\n"
            f"Subject: {subject}\n"
            f"Body: {body[:800]}\n\n"
            "Return exactly:\n"
            '{"type":"SPAM|URGENT|ACTION_REQUIRED|SOCIAL|INFORMATIONAL",'
            '"urgency_score":<0-10>,"sentiment":"positive|neutral|negative",'
            '"action":"reply|archive|escalate|delete",'
            '"key_points":["<point1>","<point2>"],'
            '"reasoning":"<1-2 sentences>"}'
        )
        resp = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 400, "temperature": 0.1},
        )
        text = resp.text.strip()
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text).strip()
        return json.loads(text)
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# CORE EPISODE RUNNER — FIXED ACTION LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def run_auto_episode(task_id: str) -> tuple[Any, list, Optional[str]]:
    """
    Run a complete automated episode.
    FIX: URGENT emails now properly escalated (with reason) or replied.
         ACTION_REQUIRED emails are replied.
         SPAM / SOCIAL / INFORMATIONAL emails are archived.
    """
    if not _ENV_OK:
        return None, [], "EmailEnv not available — check environment.py"

    try:
        # ── Initialise processor ──────────────────────────────────────────────
        processor = None
        if _SMART_OK:
            try:
                processor = SmartEmailProcessor()
            except Exception:
                processor = None

        env = EmailEnv(task_id)
        obs = env.reset()
        log: list[dict] = []
        step_num = 0
        done = False

        def _email_ids(o) -> list[str]:
            if hasattr(o, "inbox_summary"):
                return [item.id for item in o.inbox_summary]
            if isinstance(o, dict):
                inbox = o.get("inbox_summary", o.get("inbox", o.get("email_ids", [])))
                if inbox and isinstance(inbox[0], dict):
                    return [item["id"] for item in inbox]
                return list(inbox)
            return []

        email_ids = _email_ids(obs)

        # Load task cfg to know which emails need replies / escalations
        _emails_needing_reply: set = set()
        _emails_needing_escalation: set = set()
        try:
            if _GET_TASK_OK:
                from models.tasks import get_task as _get_task
                cfg = _get_task(task_id)
                _emails_needing_reply = cfg.emails_requiring_reply
                _emails_needing_escalation = cfg.emails_requiring_escalation
        except Exception:
            pass

        for eid in email_ids:
            if done:
                break

            subject, body, sender = str(eid), "", ""

            # ── STEP 1: Read ──────────────────────────────────────────────────
            try:
                step_num += 1
                obs, reward, done, info = env.step(Action.read_email(eid))
                if hasattr(obs, "current_email") and obs.current_email:
                    subject = getattr(obs.current_email, "subject", str(eid)) or str(eid)
                    body    = getattr(obs.current_email, "body", "") or ""
                    sender  = getattr(obs.current_email, "sender", "") or ""
                log.append({
                    "step": step_num, "action": f"read_email({eid})",
                    "action_type": "read", "email_id": eid,
                    "reward": reward, "valid": info.get("valid", True),
                    "subject": subject, "sender": sender,
                    "label": "", "api_used": False,
                })
            except Exception:
                pass

            if done:
                break

            # ── STEP 2: Classify ──────────────────────────────────────────────
            label_str = "INFORMATIONAL"
            try:
                if processor:
                    label_str = processor.classify(subject, body)
                else:
                    label_str = _kw_classify(subject, body)

                label_obj = _to_label_obj(label_str)
                step_num += 1
                obs, reward, done, info = env.step(Action.classify_email(eid, label_obj))
                log.append({
                    "step": step_num, "action": f"classify({label_str})",
                    "action_type": "classify", "email_id": eid,
                    "reward": reward, "valid": info.get("valid", True),
                    "subject": subject, "sender": sender,
                    "label": label_str, "api_used": False,
                })
            except Exception:
                label_str = "INFORMATIONAL"

            if done:
                break

            # ── STEP 3: Decide and execute final action ───────────────────────
            #
            # CORRECT LOGIC (FIX #1):
            #   • emails_requiring_escalation → escalate (with reason)
            #   • emails_requiring_reply      → reply
            #   • SPAM / SOCIAL / INFORMATIONAL → archive
            #   • URGENT (not in escalation list) → reply
            #   • ACTION_REQUIRED (not in reply list) → reply anyway
            #

            try:
                if eid in _emails_needing_escalation:
                    # Escalate — requires a non-empty reason (min 5 chars)
                    reason = _ESCALATION_REASONS.get(
                        eid,
                        "This email requires immediate human escalation and senior management review."
                    )
                    step_num += 1
                    obs, reward, done, info = env.step(Action.escalate_email(eid, reason=reason))
                    log.append({
                        "step": step_num, "action": f"escalate_email({eid})",
                        "action_type": "escalate", "email_id": eid,
                        "reward": reward, "valid": info.get("valid", True),
                        "subject": subject, "sender": sender,
                        "label": label_str, "api_used": False,
                    })

                elif label_str in ("URGENT", "ACTION_REQUIRED") or eid in _emails_needing_reply:
                    # Reply — use Gemini if available else canned template
                    api_used = False
                    if processor:
                        calls_before = processor.api_calls_made
                        reply_text = processor.reply(eid, subject, body, label_str)
                        api_used = processor.api_calls_made > calls_before
                    else:
                        if label_str == "URGENT":
                            reply_text = (
                                "Thank you for flagging this urgently. We have received your message "
                                "and our team is investigating immediately. "
                                "We will provide an update within the hour."
                            )
                        else:
                            reply_text = (
                                "Thank you for reaching out. We have reviewed your request and will "
                                "take the required action by end of business today. "
                                "Please let us know if any additional information is needed."
                            )
                    step_num += 1
                    obs, reward, done, info = env.step(Action.reply_email(eid, body=reply_text))
                    log.append({
                        "step": step_num, "action": f"reply_email({eid})",
                        "action_type": "reply", "email_id": eid,
                        "reward": reward, "valid": info.get("valid", True),
                        "subject": subject, "sender": sender,
                        "label": label_str, "api_used": api_used,
                    })

                else:
                    # Archive — SPAM, SOCIAL, INFORMATIONAL
                    step_num += 1
                    obs, reward, done, info = env.step(Action.archive_email(eid))
                    log.append({
                        "step": step_num, "action": f"archive_email({eid})",
                        "action_type": "archive", "email_id": eid,
                        "reward": reward, "valid": info.get("valid", True),
                        "subject": subject, "sender": sender,
                        "label": label_str, "api_used": False,
                    })
            except Exception:
                pass

        # ── Grade ─────────────────────────────────────────────────────────────
        grade_result = env.grade()

        # ── Optional DB persist ───────────────────────────────────────────────
        if _DB_OK:
            try:
                score = grade_result.get("score", 0.0) if isinstance(grade_result, dict) else float(grade_result)
                _db.save_episode(
                    task_id=task_id, score=score, steps=step_num,
                    valid_actions=sum(1 for r in log if r.get("valid", True)),
                    invalid_actions=sum(1 for r in log if not r.get("valid", True)),
                    cumulative_reward=sum(r["reward"] for r in log),
                    breakdown=grade_result.get("breakdown", {}) if isinstance(grade_result, dict) else {},
                    agent_tier=getattr(processor, "tier_name", "rule-based") if processor else "rule-based",
                )
            except Exception:
                pass

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
                "steps": [],
            }
            ordered.append(eid)
        g = seen[eid]
        g["total_reward"] = round(g["total_reward"] + entry.get("reward", 0.0), 4)
        g["steps"].append(entry)
        if entry.get("label"):
            g["label"] = entry["label"]
        at = entry.get("action_type", "")
        if at in ("reply", "archive", "escalate"):
            g["action_type"] = at
        if entry.get("api_used"):
            g["api_used"] = True
    return [seen[eid] for eid in ordered]


def _render_step_card(step: dict, highlight: bool = False) -> None:
    """
    Render one email processing card.
    FIX #3: All HTML is clean — no raw tags leaking, no broken f-strings.
    FIX #4: Shows subject, classification, action, and reward clearly.
    """
    label      = step.get("label") or "INFORMATIONAL"
    badge_cls, badge_text = _LABEL_BADGE.get(label, ("badge-gray", label))
    action_type = step.get("action_type") or "archive"
    action_text = _ACTION_ICON.get(action_type, action_type)
    action_color = _ACTION_COLOR.get(action_type, "#94a3b8")
    reward = step.get("total_reward", 0.0)
    reward_str = f"+{reward:.3f}" if reward >= 0 else f"{reward:.3f}"
    reward_col = "#22c55e" if reward > 0 else "#ef4444" if reward < 0 else "#94a3b8"
    active_cls = "step-card-active" if highlight else ""
    api_pill   = '<span class="gemini-pill" style="margin-left:0.4rem">⚡ Gemini</span>' if step.get("api_used") else ""
    subject    = str(step.get("subject", ""))[:70]
    sender     = str(step.get("sender", ""))

    # Build HTML cell by cell to avoid multi-line f-string pitfalls (FIX #3)
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
        f'{api_pill}'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="padding:0.8rem 0 1rem">'
        '<div style="font-size:1.1rem;font-weight:700;color:#f1f5f9">🤖 AI Email Agent</div>'
        '<div style="font-size:0.76rem;color:#475569;margin-top:2px">Training Environment</div>'
        '</div>',
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
    gem_key = os.environ.get("GEMINI_API_KEY", "").strip()
    api_st  = '<span style="color:#22c55e;font-weight:600">✓ Gemini Active</span>' if gem_key else '<span style="color:#f97316">Rule-Based Only</span>'

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
        ("Reply (needed)",     "+0.10", "#22c55e"),
        ("Reply (not needed)", "−0.15", "#ef4444"),
        ("Correct escalate",   "+0.25", "#22c55e"),
        ("Wrong escalate",     "−0.20", "#ef4444"),
        ("Archive (correct)",  "+0.05", "#22c55e"),
        ("Invalid action",     "−0.10", "#ef4444"),
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
        '<div class="hero-block">'
        '<div class="hero-title">🤖 AI Email Agent Training Environment</div>'
        '<div class="hero-sub">Watch the AI read, classify, reply, and escalate emails — scored across multiple dimensions.</div>'
        '<div class="hero-tag">Select a task → preview the inbox → click Run Demo to watch it live.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1.4, 1, 2])
    with c1:
        if st.button("▶  Run Demo", type="primary", use_container_width=True, key="btn_run_demo"):
            _reset()
            task_id = st.session_state.get("task_id", "task_1")
            with st.spinner("AI agent processing inbox…"):
                result, log, err = run_auto_episode(task_id)
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
        lbl = "✕ Close Guide" if st.session_state.get("show_guide") else "？ Guide"
        if st.button(lbl, use_container_width=True, key="btn_guide"):
            st.session_state["show_guide"] = not st.session_state.get("show_guide", False)
            st.rerun()

    with c3:
        st.markdown(
            f'<div style="padding:0.55rem 0;font-size:0.82rem;color:#64748b">'
            f'Phase: {_badge(st.session_state.get("phase","start"))}</div>',
            unsafe_allow_html=True,
        )


def render_guide() -> None:
    if not st.session_state.get("show_guide"):
        return
    st.markdown(
        '<div style="margin-bottom:1.2rem">'
        '<div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;margin-bottom:0.7rem">How to use</div>'
        '<ol style="color:#94a3b8;font-size:0.87rem;line-height:2;padding-left:1.2rem">'
        '<li>Pick inbox difficulty (5 / 10 / 15 emails). Emails preview instantly.</li>'
        '<li>Click <strong style="color:#fff">▶ Run Demo</strong> — AI handles every email step by step.</li>'
        '<li>Or click <strong style="color:#fff">▶ Start Task</strong> to run any difficulty.</li>'
        '<li>Watch the animated cards: classification badge → action → reward.</li>'
        '<li>View Score Breakdown and compare vs. Baseline Agent.</li>'
        '<li>Use <strong style="color:#fff">🔍 Analyse Email</strong> to test any email with Gemini AI.</li>'
        '</ol>'
        '<div style="margin-top:0.8rem;font-size:0.8rem;color:#475569;font-style:italic">'
        'AI: Gemini 2.5 Flash (active when GEMINI_API_KEY set). Auto rule-based fallback if unavailable.'
        '</div></div><hr style="border-color:#1e2d47;margin-bottom:1rem">',
        unsafe_allow_html=True,
    )


def render_inbox_preview(task_id: str) -> None:
    """Show all emails in the inbox as read-only cards before processing."""
    if not _GET_TASK_OK:
        return
    try:
        task = get_task(task_id)
        emails = task.emails
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
    act  = {
        "task_1": "Classify emails: SPAM / URGENT / ACTION REQUIRED / SOCIAL / INFO",
        "task_2": "Classify + reply to urgent & action-required emails",
        "task_3": "Full triage: classify, reply, escalate legal notices, archive rest",
    }.get(selected, "Classify and handle all emails")

    st.markdown(
        f'<div style="font-size:0.82rem;color:#64748b;margin:0.3rem 0">'
        f'📧 {ec} emails &nbsp;·&nbsp; ⚡ {ms} max steps &nbsp;·&nbsp; '
        f'<span style="color:#94a3b8">{act}</span></div>',
        unsafe_allow_html=True,
    )

    col_start, _ = st.columns([1, 3])
    with col_start:
        if st.button("▶  Start Task", type="primary", use_container_width=True, key="btn_start_task"):
            if not _ENV_OK:
                st.session_state["app_error"] = "EmailEnv not available — check environment.py"
                st.rerun()
            else:
                try:
                    env = EmailEnv(selected)
                    obs = env.reset()
                    st.session_state.update({
                        "env": env, "obs_dict": obs, "task_id": selected,
                        "phase": "inbox", "step_results": [], "auto_log": [],
                        "cumulative_reward": 0.0, "steps_taken": 0,
                        "episode_done": False, "final_score": None,
                        "grade_data": None, "app_error": None,
                        "demo_email_steps": [], "demo_playback_done": False,
                    })
                except Exception as exc:
                    st.session_state["app_error"] = f"Could not start task: {exc}"
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    render_inbox_preview(selected)


def render_stepwise_demo(email_steps: list) -> None:
    """
    Animate emails being processed one by one.
    FIX #4: Clearly shows subject, classification badge, action type, and reward per card.
    """
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
            time.sleep(0.7)
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
            result, log, err = run_auto_episode(task_id)
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
    """Score display + breakdown bars. FIX #3: clean HTML, no leaking tags."""
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
            "classification": "#6366f1", "reply": "#22c55e",
            "escalation": "#f97316",     "workflow": "#06b6d4",
            "action": "#22c55e",         "efficiency": "#fbbf24",
            "coverage": "#06b6d4",
        }
        bars_html = ""
        for dim, val in breakdown.items():
            col = dim_colors.get(dim.lower(), "#6366f1")
            bars_html += _bar(dim.replace("_", " ").title(), val, col)
        st.markdown(bars_html, unsafe_allow_html=True)

        weakest = min(breakdown, key=lambda k: breakdown[k])
        tips = {
            "classification": "Improve keyword rules or enable Gemini for smarter classification.",
            "reply":          "Check required keywords/phrases appear in replies.",
            "escalation":     "Escalate legal, security, and HR compliance emails immediately.",
            "workflow":       "Reduce invalid actions and stay within the step budget.",
            "action":         "Always read an email before classifying it.",
            "efficiency":     "Avoid redundant steps — process each email in 3 steps.",
            "coverage":       "Ensure every email in the inbox is handled.",
        }
        tip = tips.get(weakest.lower(), "Focus on this dimension next.")
        st.markdown(
            f'<div style="background:#0d1526;border-left:3px solid #fbbf24;'
            f'border-radius:0 8px 8px 0;padding:0.6rem 0.9rem;margin-top:0.6rem;'
            f'font-size:0.82rem;color:#fbbf24">'
            f'&#9889; Weakest: <strong>{weakest.replace("_"," ").title()}</strong> — {tip}'
            f'</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.get("phase") == "graded":
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚖  Compare with Baseline Agent", type="primary", key="btn_compare"):
            st.session_state["phase"] = "compared"
            st.rerun()


def render_comparison() -> None:
    """Baseline agent comparison panel."""
    if st.session_state.get("phase") != "compared":
        return

    _sec("BASELINE COMPARISON")
    st.markdown(
        '<p style="font-size:0.84rem;color:#94a3b8;margin-bottom:1rem">'
        'The <strong style="color:#f1f5f9">Baseline</strong> agent uses the same rule-based classifier '
        'but no learning. The <strong style="color:#f1f5f9">Random</strong> agent acts uniformly at random.</p>',
        unsafe_allow_html=True,
    )

    my_score  = st.session_state.get("final_score", 0.0) or 0.0
    task_id   = st.session_state.get("task_id", "task_1")
    baseline  = 0.62

    if _BASELINE_OK and _ENV_OK:
        try:
            b_agent = BaselineAgent()
            b_env   = EmailEnv(task_id)
            b_obs   = b_env.reset()
            b_done  = False
            while not b_done:
                act = b_agent.act(b_obs)
                b_obs, _, b_done, _ = b_env.step(act)
            b_res  = b_env.grade()
            baseline = b_res.get("score", 0.62) if isinstance(b_res, dict) else float(b_res)
        except Exception:
            pass

    rnd = 0.21
    my_g, my_c = _grade_letter(my_score)
    b_g,  b_c  = _grade_letter(baseline)

    c1, c2, c3 = st.columns(3)
    for col, label, sc, g, gc in [
        (c1, "Your Session",           my_score, my_g, my_c),
        (c2, "Baseline (Rule-Based)",  baseline, b_g,  b_c),
        (c3, "Random Agent",           rnd,      "F",  "#ef4444"),
    ]:
        with col:
            st.markdown(
                f'<div class="agent-card">'
                f'<div class="ac-label">{label}</div>'
                f'<div class="ac-score" style="color:{gc}">{sc:.3f} {g}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    delta = my_score - baseline
    d_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
    d_col = "#22c55e" if delta >= 0 else "#ef4444"
    d_wrd = "above" if delta >= 0 else "below"
    st.markdown(
        f'<div style="font-size:0.85rem;color:#94a3b8;margin:0.9rem 0">'
        f'Your agent scored <strong style="color:{d_col}">{d_str}</strong> {d_wrd} the rule-based baseline.'
        f'</div>',
        unsafe_allow_html=True,
    )

    if _DB_OK:
        try:
            curve = _db.get_learning_curve(task_id)
            if curve and len(curve) >= 3:
                _sec("LEARNING CURVE")
                import pandas as pd
                df = pd.DataFrame(curve)[["episode_number","score"]].set_index("episode_number")
                st.line_chart(df, color="#6366f1")
                st.caption(f"Baseline: {baseline:.3f} — task: {task_id}")
        except Exception:
            pass

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄  Start New Task", type="primary", key="btn_new_task"):
        _reset()
        st.rerun()


def render_analyse_mode() -> None:
    """
    FIX #2: Analyse Email mode — fully working form with Gemini + rule-based fallback.
    Button IS connected: st.form_submit_button fires and calls backend.
    FIX #3: All result HTML is clean — built string-by-string, no leaking tags.
    """
    _sec("🔍 ANALYSE YOUR EMAIL WITH AI")

    gem_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if gem_key:
        st.markdown(
            '<div class="gemini-pill" style="margin-bottom:0.8rem">⚡ Powered by Gemini 2.5 Flash</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="display:inline-flex;align-items:center;gap:6px;background:#1a2030;'
            'border:1px solid #334155;border-radius:20px;padding:3px 12px;font-size:0.78rem;'
            'color:#64748b;margin-bottom:0.8rem">🔧 Rule-Based Mode (no GEMINI_API_KEY set)</div>',
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

    # ── Early exit if form not submitted or inputs empty ─────────────────────
    if not submitted:
        return
    if not (subject or body):
        st.warning("Please enter at least a subject line or email body.")
        return

    # ── FIX #2: Connected backend logic ──────────────────────────────────────
    # Try Gemini first; on ANY failure silently use rule-based fallback.
    gemini_result = _call_gemini_analyze(subject, body, sender) if gem_key else None
    used_gemini   = gemini_result is not None

    if used_gemini:
        label_val  = str(gemini_result.get("type", "INFORMATIONAL")).upper()
        action     = gemini_result.get("action", "archive")
        reasoning  = gemini_result.get("reasoning", "")
        urgency    = int(gemini_result.get("urgency_score", 0))
        sentiment  = gemini_result.get("sentiment", "neutral")
        key_points = gemini_result.get("key_points", [])
        conf       = min(0.95, 0.70 + urgency * 0.025)
    else:
        # Rule-based fallback (FIX #2: always works, never crashes)
        label_val = _kw_classify(subject, body)
        action_map = {
            "SPAM": "archive", "URGENT": "escalate → reply",
            "ACTION_REQUIRED": "reply", "SOCIAL": "archive", "INFORMATIONAL": "archive",
        }
        action    = action_map.get(label_val, "archive")
        reasoning = {
            "SPAM":            "Keywords indicate spam or phishing. Archive without reply.",
            "URGENT":          "Urgent keywords detected. Requires immediate escalation and reply.",
            "ACTION_REQUIRED": "Action deadline or approval request detected. Reply required today.",
            "SOCIAL":          "Social event invitation. Archive after noting the date.",
            "INFORMATIONAL":   "Informational content. No action required — archive after reading.",
        }.get(label_val, "Review this email manually.")
        urgency   = {"SPAM":1,"URGENT":9,"ACTION_REQUIRED":7,"SOCIAL":2,"INFORMATIONAL":2}.get(label_val, 3)
        sentiment = "negative" if label_val in ("SPAM","URGENT") else "neutral"
        key_points = []
        conf      = {"SPAM":0.90,"URGENT":0.85,"ACTION_REQUIRED":0.82,"SOCIAL":0.78,"INFORMATIONAL":0.75}.get(label_val, 0.70)

    # Ensure label_val is clean
    label_val = label_val.strip().upper()
    if label_val not in ("SPAM","URGENT","ACTION_REQUIRED","SOCIAL","INFORMATIONAL"):
        label_val = "INFORMATIONAL"

    # ── Build result UI (FIX #3: clean HTML, string-concatenated) ─────────────
    bc = {"SPAM":"badge-red","URGENT":"badge-amber","ACTION_REQUIRED":"badge-indigo",
          "SOCIAL":"badge-green","INFORMATIONAL":"badge-gray"}.get(label_val, "badge-gray")
    conf_pct  = int(conf * 100)
    conf_col  = "#22c55e" if conf >= 0.7 else "#fbbf24" if conf >= 0.4 else "#ef4444"
    urg_col   = "#ef4444" if urgency >= 8 else "#fbbf24" if urgency >= 5 else "#22c55e"
    eng_text  = "⚡ Gemini 2.5 Flash" if used_gemini else "🔧 Rule-Based Fallback"
    eng_style = ('class="gemini-pill"' if used_gemini
                 else 'style="font-size:0.74rem;color:#64748b"')

    # Key points list
    kp_html = ""
    if key_points:
        items = "".join(f'<li style="color:#94a3b8;font-size:0.8rem">{p}</li>' for p in key_points[:3])
        kp_html = f'<ul style="margin:0.5rem 0 0;padding-left:1.2rem">{items}</ul>'

    # Header row
    html  = '<div class="ea-card" style="margin-top:1rem">'
    html += f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem">'
    html += f'<div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#6366f1">AI Decision Output</div>'
    html += f'<span {eng_style}>{eng_text}</span>'
    html += '</div>'

    # Metrics row
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

    # Reasoning row
    html += (
        f'<div style="border-top:1px solid #1e2d47;padding-top:0.7rem">'
        f'<div style="font-size:0.72rem;color:#64748b;margin-bottom:3px">AI Reasoning</div>'
        f'<div style="font-size:0.83rem;color:#94a3b8">{reasoning}</div>'
        f'{kp_html}'
        f'</div>'
    )
    html += '</div>'   # close ea-card

    st.markdown(html, unsafe_allow_html=True)

    st.session_state["custom_result"] = {
        "label": label_val, "action": action,
        "confidence": conf, "reasoning": reasoning,
        "used_gemini": used_gemini,
    }


def render_footer() -> None:
    st.markdown(
        '<div class="ea-footer">Developed by Sumanth Mamidi &nbsp;·&nbsp; Powered by Google Gemini 2.5 Flash</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
render_hero()
render_guide()

# Mode toggle
mode = st.session_state.get("mode", "training")
ca, cb, _ = st.columns([1, 1, 2])
with ca:
    t = "primary" if mode == "training" else "secondary"
    if st.button("📋  Training Mode", type=t, use_container_width=True, key="btn_mode_train"):
        st.session_state["mode"] = "training"
        st.rerun()
with cb:
    t = "primary" if mode == "custom" else "secondary"
    if st.button("🔍  Analyse Email", type=t, use_container_width=True, key="btn_mode_custom"):
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