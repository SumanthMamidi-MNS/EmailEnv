# 🚀 AI Email Agent — A Flight Simulator for Autonomous Email Agents

**One-Line Description**: An OpenEnv-compliant simulation environment, objective evaluation harness, and safe learning engine designed to test, benchmark, and harden autonomous email agents against operational risks before production deployment.

---

## 🎯 Problem
Deploying autonomous LLM agents directly to enterprise inboxes carries severe operational risks:
- **Catastrophic Action Failures**: Archiving critical P0 server outage alerts or data breach warnings disguised behind conversational text.
- **Prompt Injection Vulnerabilities**: Malicious instructions embedded in email bodies hijacking the agent's workflow.
- **Hallucinated Commitments**: Unprompted or non-compliant replies to external vendors and clients.
- **Unvalidated Feedback Loops**: Human corrections blindly modifying agent rules without regression safety checks.

---

## 💡 Solution
**AI Email Agent** provides a sandboxed "flight simulator" for email agents built on the OpenEnv standard. Agents interact with realistic, multi-intent, and adversarial inboxes through strictly typed observations and actions while an environment-owned rubric grades their performance against hidden ground truth.

---

## 🏗️ Technical Architecture & Key Highlights

1. **OpenEnv Standard Simulation (`envs/email_env/`)**:
   - Implements strict `reset`, `step`, and `state` lifecycle with hidden ground-truth isolation (`GroundTruthEmail` vs `EmailView`).
   - Deterministic state machine with environment-owned rubric applying severe penalties (`-0.50`) for archiving critical outages.

2. **Three-Tier Agent Hierarchy (`agents/`)**:
   - **Tier 1 (Baseline)**: Deterministic first-match keyword decision tree providing a zero-dependency baseline floor.
   - **Tier 2 (Hybrid)**: Weighted additive multi-signal scoring vector $(S_{\text{spam}}, S_{\text{urgent}}, S_{\text{action}}, S_{\text{social}})$ with selective LLM escalation for ambiguous emails.
   - **Tier 3 (LLM)**: Autonomous triage engine powered by Google Gemini 2.5 Flash with XML prompt injection defense boundaries (`<untrusted_email_content>`).

3. **Fault-Tolerant Degraded Fallback Pipeline**:
   - Transparent mid-call exception handling recovering from HTTP 400 (invalid credentials), HTTP 429 (rate limits), and network timeouts without crashing.
   - 100% offline functionality for zero-cost reproducibility.

4. **Continuous Learning with Regression Protection (`evaluation/`)**:
   - Human feedback is treated as empirical evidence: $\ge 2$ consistent corrections generate a `CandidateLesson`.
   - Candidates are subjected to automated regression evaluation against a fixed 8-scenario benchmark (`regression.jsonl`).
   - Only rules that cause zero safety violations and maintain/improve cumulative reward are marked `PROMOTED` and retrieved into runtime agent context.

5. **Multi-Difficulty Scenario Benchmark (`data/scenarios/`)**:
   - 48 scenarios across 6 difficulty tiers: Starter, Medium, Advanced, Adversarial (prompt injections, buried urgency, keyword traps), Held-Out, and Regression.

---

## 📊 Measurable Empirical Results

- **Automated Test Coverage**: 65/65 unit and integration tests passing (`pytest tests/`).
- **Safety Violation Rate**: 0% dangerous archives of critical outages across all benchmark tiers (100% safety rate).
- **API Failure Resilience**: 100% episode completion rate with zero unhandled exceptions under simulated provider failure.

---

## 🛠️ Tech Stack
- **Core Runtime**: Python 3.12, OpenEnv, FastAPI, Pydantic v2, SQLite3
- **LLM Integration**: Google Gemini 2.5 Flash (`google-generativeai`), Mock Providers
- **Interface**: Streamlit Interactive Flight Simulator
- **Deployment**: Multi-Stage Docker, Hugging Face Spaces (Port 7860, Non-Root User)

---

## 🔗 Repository & Live Demo
- **GitHub**: [github.com/your-username/EmailEnv](https://github.com/your-username/EmailEnv)
- **Hugging Face Spaces**: [huggingface.co/spaces/your-username/ai-email-agent](https://huggingface.co/spaces/your-username/ai-email-agent)
