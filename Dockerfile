# Explicit build for Railway.
#
# Railway takes a Dockerfile over auto-detection, whichever builder the service
# is set to. That matters here: the repo root has BOTH a pyproject.toml (the
# engine's own package manifest) and a requirements.txt (the app's deps), which
# is exactly the ambiguity that makes auto-detection guess — and guess wrong.
# Everything below is stated outright, so the build doesn't depend on which
# builder Railway happens to default to.
FROM python:3.11-slim

# cairo is what cairosvg needs to rasterize the pattern for the vision input.
# Without it the agent still runs (see agent.py:_render_png, which falls back
# to state-only reasoning), so this is a capability, not a hard requirement.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libcairo2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so edits to application code don't re-run pip.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The engine is NOT pip-installed — it goes on PYTHONPATH below. Installing it
# would mean resolving pyproject.toml's build backend for no benefit.
COPY src/ ./src/
COPY backend/ ./backend/
COPY app/ ./app/
# Not test data despite the path: the app serves the bundled sample pattern and
# reads the default measurement table from here (app.py: FIXTURES, DEFAULT_MEAS).
COPY tests/fixtures/ ./tests/fixtures/

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

# serve.py reads $PORT itself, so this needs no shell and can use exec form
# (signals reach the server directly instead of going to a wrapping shell).
CMD ["python", "backend/serve.py"]
