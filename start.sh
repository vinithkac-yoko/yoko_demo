#!/bin/sh
# Railpack looks for this by convention ("Script start.sh not found" in its
# build output). serve.py reads $PORT itself, so nothing here depends on
# parameter expansion — but this runs under a shell anyway, so both work.
#
# exec so the server replaces this shell and receives signals directly,
# rather than having them delivered to a wrapper that ignores them.
exec python backend/serve.py
