# ══════════════════════════════════════════════════════════════════════════════
# AI Email Agent — Optimized Dockerfile
#
# Build time target : < 3 minutes on cold cache, < 30 seconds on warm cache
# Image size target : ~400 MB  (vs old ~2.5 GB with sentence-transformers/torch)
# Runtime           : streamlit run app.py on port 7860
# Compatible with   : Docker, Hugging Face Spaces
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: dependency builder ───────────────────────────────────────────────
# Isolating pip install in its own stage means:
#   • changing app.py NEVER triggers pip re-install
#   • only requirements.txt changes break this cache
FROM python:3.12-slim AS deps

WORKDIR /deps

# Install gcc only — needed by a handful of slim packages (e.g. pydantic v2)
# Single RUN = single layer; rm -rf keeps the layer size minimal
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements FIRST so Docker can cache this layer independently
COPY requirements.txt .

# Install all runtime packages into the system site-packages
# --no-cache-dir    → smaller layer (no pip wheel cache)
# --upgrade pip     → avoids legacy-resolver warnings
# Install from requirements.txt so Dockerfile and requirements.txt are always in sync
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.12-slim

# Create a non-root user for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Pull installed packages from builder stage — no compiler or build tools needed
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application source — kept last so code changes don't bust the pip cache
# .dockerignore excludes: venv/, __pycache__, .env, .git, tests/, openenv.db
COPY . .

# Streamlit telemetry off (avoids first-run interactive prompt in containers)
# Streamlit — disable telemetry & interactive prompts in containers
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true
# GEMINI_API_KEY is intentionally NOT set here.
# Pass it at runtime only:
#   docker run -e GEMINI_API_KEY=your_key ...
# or via Hugging Face Spaces → Settings → Repository Secrets.

# Hugging Face Spaces uses port 7860
EXPOSE 7860

# Set correct ownership
RUN chown -R appuser:appuser /app
USER appuser

# Streamlit flags passed as CMD args — no separate config file needed
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]