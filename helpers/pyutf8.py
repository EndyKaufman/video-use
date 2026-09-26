"""Force UTF-8 on stdout/stderr for the helper scripts.

Windows gives a redirected (piped) Python stdio the legacy ANSI codepage — on a
Russian locale that is cp1251. Every helper prints em dashes and arrows in its
progress lines, so a piped run aborts with
`'charmap' codec can't encode character '\\u2192'` before it does any work.
Setuptools of any kind won't help; the fix belongs at the stream.

Import for the side effect, as early as possible:

    import pyutf8  # noqa: F401

`PYTHONUTF8=1` (see env.sh / env.ps1) does the same thing for the whole
interpreter. This module is the belt-and-braces version: it works when someone
runs `python helpers/render.py` in a shell that never sourced the env file.
File IO is *not* covered here — read/write calls that can carry non-ASCII
content pass `encoding="utf-8"` explicitly.
"""

from __future__ import annotations

import sys

_UTF8 = "utf8"


def _needs_fix(stream) -> bool:
    encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
    return encoding != _UTF8


def ensure_utf8_stdio() -> bool:
    """Reconfigure stdout/stderr to UTF-8. Returns True if anything changed."""
    changed = False
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None or not _needs_fix(stream):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
            changed = True
        except (AttributeError, ValueError, OSError):
            # Frozen / replaced streams (pytest capture, embeds) — not fatal.
            pass
    return changed


ensure_utf8_stdio()
