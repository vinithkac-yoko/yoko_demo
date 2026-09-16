"""Production entry point.

Reads ``$PORT`` itself rather than relying on the start command being run
through a shell. Railway execs the start command directly, so an argument
like ``--port ${PORT:-8000}`` arrives at uvicorn as that literal string and
is rejected ("not a valid integer"). Doing the lookup in Python makes the
command work the same under every builder, and whether it's exec'd or run by
a shell.

Also puts both import roots on ``sys.path`` (``backend/`` for ``app``,
``src/`` for ``seamly_engine``), so no PYTHONPATH prefix is required either.

    python backend/serve.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
for path in (str(_HERE), str(_SRC)):
    if Path(path).is_dir() and path not in sys.path:
        sys.path.insert(0, path)


def port() -> int:
    """$PORT, tolerating it being unset or empty (an empty value would
    otherwise raise instead of falling back)."""
    raw = (os.getenv("PORT") or "").strip()
    try:
        return int(raw)
    except ValueError:
        return 8000


def main() -> None:
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=port())


if __name__ == "__main__":
    main()
