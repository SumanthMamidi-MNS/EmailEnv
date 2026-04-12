# === inference.py ===
"""
AI Email Agent - standalone inference script.

Implements the smart API-usage strategy:
  - Rule-based classifier handles ALL emails (zero API cost)
  - Groq LLM used ONLY for URGENT/ACTION_REQUIRED reply generation
  - All API calls wrapped in try/except with silent fallback
  - Per-email cache prevents duplicate API calls
  - System never crashes regardless of API availability

Run with: python inference.py
"""

from __future__ import annotations

import io
import sys
import traceback

# Force UTF-8 on Windows to prevent codec errors
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# Load .env silently
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


# ── Backend imports ───────────────────────────────────────────────────────────

try:
    from models.environment import EmailEnv
    _ENV_OK = True
except Exception as _e:
    print(f"[ERROR] Could not import EmailEnv: {_e}")
    _ENV_OK = False

try:
    from models import ClassLabel, Action
    _MODELS_OK = True
except Exception as _e:
    print(f"[ERROR] Could not import models: {_e}")
    _MODELS_OK = False

    class ClassLabel:  # type: ignore
        SPAM = "SPAM"; URGENT = "URGENT"; ACTION_REQUIRED = "ACTION_REQUIRED"
        SOCIAL = "SOCIAL"; INFORMATIONAL = "INFORMATIONAL"

    class Action:  # type: ignore
        @staticmethod
        def read_email(eid): return {"type": "read_email", "email_id": eid}
        @staticmethod
        def classify_email(eid, label): return {"type": "classify_email", "email_id": eid, "label": label}
        @staticmethod
        def archive_email(eid): return {"type": "archive_email", "email_id": eid}
        @staticmethod
        def escalate_email(eid): return {"type": "escalate_email", "email_id": eid}
        @staticmethod
        def reply_email(eid, text=""): return {"type": "reply_email", "email_id": eid, "text": text}

try:
    from smart_agent import SmartEmailProcessor
    _SMART_OK = True
except Exception:
    _SMART_OK = False


# ── Smart processor singleton ─────────────────────────────────────────────────

def _make_processor():
    if _SMART_OK:
        try:
            return SmartEmailProcessor()
        except Exception:
            pass

    # Minimal inline fallback if smart_agent can't be imported
    class _MinimalProcessor:
        tier_name = "Tier 1 - Keyword Fallback (inline)"
        api_calls_made = 0

        _kw = {
            "SPAM": ["win", "winner", "prize", "lottery", "click here", "free offer", "unsubscribe", "claim now"],
            "URGENT": ["urgent", "asap", "immediately", "critical", "emergency", "outage", "all hands"],
            "ACTION_REQUIRED": ["please review", "approval", "sign off", "action required", "response required",
                                "please confirm", "follow up", "awaiting your"],
            "SOCIAL": ["invitation", "invite", "party", "birthday", "rsvp", "event", "gathering"],
        }

        def classify(self, subj, body):
            text = (subj + " " + body).lower()
            for label, kws in self._kw.items():
                if any(k in text for k in kws):
                    return label
            return "INFORMATIONAL"

        def classify_as_label(self, subj, body):
            return self.classify(subj, body)

        def decide_action(self, label):
            return "reply" if label in ("URGENT", "ACTION_REQUIRED") else "archive"

        def reply(self, eid, subj, body, label):
            if label == "URGENT":
                return "Thank you for flagging this urgently. We are investigating and will update you within the hour."
            return "Thank you for reaching out. We will take the required action by end of business today."

    return _MinimalProcessor()


# ── Email ID extraction ───────────────────────────────────────────────────────

def _email_ids(obs) -> list:
    if hasattr(obs, "inbox_summary"):
        return [item.id for item in obs.inbox_summary]
    if isinstance(obs, dict):
        inbox = obs.get("inbox_summary", obs.get("inbox", obs.get("email_ids", [])))
        if inbox and isinstance(inbox[0], dict):
            return [item["id"] for item in inbox]
        return list(inbox)
    return []


# ── ASCII display helpers ──────────────────────────────────────────────────────

_COL_STEP    = 6
_COL_VALID   = 3
_COL_ACTION  = 38
_COL_REWARD  = 10
_COL_SUBJECT = 42

def _fmt_row(step: int, valid: bool, action: str, reward: float, subject: str) -> str:
    icon = "+" if valid else "!"
    rstr = f"+{reward:.4f}" if reward >= 0 else f"{reward:.4f}"
    subj = subject[:_COL_SUBJECT - 1] if len(subject) >= _COL_SUBJECT else subject
    return f"  [{step:>3}]  {icon}  {action:<{_COL_ACTION}}  {rstr:>{_COL_REWARD}}  {subj}"

def _grade_letter(score: float) -> str:
    if score >= 0.90: return "A"
    if score >= 0.75: return "B"
    if score >= 0.60: return "C"
    if score >= 0.40: return "D"
    return "F"

def _divider(ch: str = "-", width: int = 72) -> str:
    return ch * width


# ── Main episode runner ────────────────────────────────────────────────────────

def run() -> None:
    # ── Init processor (non-blocking) ─────────────────────────────────────────
    processor = _make_processor()

    print()
    print(_divider("="))
    print("  AI EMAIL AGENT - Inference Run")
    print(f"  Strategy : Smart tiered API usage")
    print(f"  Classify : {processor.tier_name}")
    print(f"  API calls: Only for URGENT / ACTION_REQUIRED replies (if key set)")
    print(f"  Task     : task_1")
    print(_divider("="))
    print()

    if not _ENV_OK:
        print("[FATAL] EmailEnv unavailable. Check models/environment.py")
        sys.exit(1)

    # ── Init environment ──────────────────────────────────────────────────────
    try:
        env = EmailEnv("task_1")
        obs = env.reset()
    except Exception as exc:
        print(f"[FATAL] Could not initialise EmailEnv: {exc}")
        traceback.print_exc()
        sys.exit(1)

    email_ids = _email_ids(obs)

    if not email_ids:
        # Try direct env.state fallback
        try:
            email_ids = [e.id for e in env.state.inbox]
        except Exception:
            pass

    if not email_ids:
        print("[WARN] Inbox is empty — nothing to process.")
    else:
        print(f"  Inbox: {len(email_ids)} email(s).")
        print()

    # ── Column header ─────────────────────────────────────────────────────────
    print(f"  {'#':>5}  {'V':>1}  {'Action':<{_COL_ACTION}}  {'Reward':>{_COL_REWARD}}  Subject")
    print("  " + _divider("-", 70))

    # ── Episode tracking ──────────────────────────────────────────────────────
    step_num: int = 0
    cum_reward: float = 0.0
    valid_count: int = 0
    invalid_count: int = 0
    done: bool = False
    info: dict = {}
    api_calls_this_run: int = 0

    # ── Process each email ────────────────────────────────────────────────────
    for eid in email_ids:
        if done:
            break

        # ── STEP: Read ────────────────────────────────────────────────────────
        try:
            step_num += 1
            obs, reward, done, info = env.step(Action.read_email(eid))
            cum_reward += reward
            valid = info.get("action_valid", info.get("valid", True))
            if valid: valid_count += 1
            else: invalid_count += 1

            # Extract email content from observation
            subject, body = "", ""
            if hasattr(obs, "current_email") and obs.current_email:
                subject = obs.current_email.subject
                body    = obs.current_email.body

            print(_fmt_row(step_num, valid, f"read_email({eid})", reward, subject))
        except Exception as exc:
            print(f"  [ERR] read_email({eid}): {exc}")
            continue

        if done:
            break

        # ── STEP: Classify (ALWAYS rule-based — zero API cost) ────────────────
        try:
            label_str   = processor.classify(subject, body)      # instant, no API
            label_obj   = processor.classify_as_label(subject, body)
            step_num   += 1
            obs, reward, done, info = env.step(Action.classify_email(eid, label_obj))
            cum_reward += reward
            valid = info.get("action_valid", info.get("valid", True))
            if valid: valid_count += 1
            else: invalid_count += 1
            print(_fmt_row(step_num, valid, f"classify -> {label_str}", reward, subject))
        except Exception as exc:
            print(f"  [ERR] classify({eid}): {exc}")
            label_str = "INFORMATIONAL"

        if done:
            break

        # ── STEP: Follow-up (reply or archive) ────────────────────────────────
        action_type = processor.decide_action(label_str)

        try:
            if action_type == "reply":
                # API only called here, only for URGENT/ACTION_REQUIRED, with full fallback
                reply_text = processor.reply(eid, subject, body, label_str)
                api_used = processor.api_calls_made > api_calls_this_run
                api_calls_this_run = processor.api_calls_made

                step_num += 1
                obs, reward, done, info = env.step(Action.reply_email(eid, reply_text))
                cum_reward += reward
                valid = info.get("action_valid", info.get("valid", True))
                if valid: valid_count += 1
                else: invalid_count += 1
                api_tag = " [LLM]" if api_used else " [tmpl]"
                print(_fmt_row(step_num, valid, f"reply_email({eid}){api_tag}", reward, subject))

                if not done:
                    step_num += 1
                    obs, reward, done, info = env.step(Action.archive_email(eid))
                    cum_reward += reward
                    valid = info.get("action_valid", info.get("valid", True))
                    if valid: valid_count += 1
                    else: invalid_count += 1
                    print(_fmt_row(step_num, valid, f"archive_email({eid})", reward, subject))
            else:
                # SPAM / SOCIAL / INFORMATIONAL — archive immediately, NO API
                step_num += 1
                obs, reward, done, info = env.step(Action.archive_email(eid))
                cum_reward += reward
                valid = info.get("action_valid", info.get("valid", True))
                if valid: valid_count += 1
                else: invalid_count += 1
                print(_fmt_row(step_num, valid, f"archive_email({eid})", reward, subject))
        except Exception as exc:
            print(f"  [ERR] follow-up({eid}): {exc}")

    # ── Grade ─────────────────────────────────────────────────────────────────
    print()
    print("  " + _divider("-", 70))

    try:
        result = env.grade()
        score = float(result.get("score", 0.0)) if isinstance(result, dict) else float(result)
        grade = _grade_letter(score)
    except Exception as exc:
        print(f"  [ERROR] Grading failed: {exc}")
        score, grade, result = 0.0, "?", {}

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print(_divider("="))
    print("  EPISODE SUMMARY")
    print(_divider("-"))
    print(f"  Final Score       :  {score:.4f}  ({grade})")
    print(f"  Steps Taken       :  {step_num}")
    print(f"  Valid Actions     :  {valid_count}")
    print(f"  Invalid Actions   :  {invalid_count}")
    print(f"  Cumulative Reward :  {cum_reward:+.4f}")
    print(f"  API Calls (LLM)   :  {processor.api_calls_made}  (0 for SPAM/SOCIAL/INFO)")

    # ── Score breakdown ───────────────────────────────────────────────────────
    if isinstance(result, dict):
        raw = result.get("breakdown", {})
        display = {}
        if isinstance(raw, dict):
            sub = raw.get("sub_scores", {})
            if sub and all(isinstance(v, (int, float)) for v in sub.values()):
                display = sub
            else:
                for k, v in raw.items():
                    try: display[k] = float(v)
                    except (TypeError, ValueError): pass

        if display:
            print()
            print("  SCORE BREAKDOWN")
            print(_divider("-"))
            for dim, val in display.items():
                try:
                    fval = float(val)
                    bar_len = int(fval * 20)
                    bar = "#" * bar_len + "." * (20 - bar_len)
                    print(f"  {dim.replace('_',' ').title():<18}  [{bar}]  {fval:.3f}")
                except (TypeError, ValueError):
                    pass

    print()
    print(_divider("="))
    print("  Developed by Sumanth Mamidi")
    print(_divider("="))
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[FATAL] {exc}")
        traceback.print_exc()
        sys.exit(1)