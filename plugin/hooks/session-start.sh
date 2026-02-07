#!/bin/bash
# SCS SessionStart: refresh index + inject project map + agent workflow
SCS="${CLAUDE_PLUGIN_ROOT}/bin/scs"

# Refresh index (auto-embeds in background by default)
"$SCS" refresh --quiet 2>/dev/null

# Generate project map
"$SCS" map 2>/dev/null

# Agent workflow rules (separated from map data, directive format)
cat <<RULES

<scs-rules>
RULE: Use scs commands before falling back to Read/Grep/Glob for code exploration.
SCS saves 90%+ tokens compared to reading files directly.

scs="$SCS"

REQUIRED workflow for navigating unfamiliar code:
1. scs map --area <dir>       → zoom into a module (see files + symbols)
2. scs map --area <file>      → see all functions in a file
3. scs lookup "<SymbolName>"  → get exact file:line location
4. scs search "<concept>"     → find code by meaning (semantic search)
5. Read file:line             → read ONLY the lines you need, AFTER locating via scs

NEVER start with Read/Grep/Glob when you can use scs lookup or scs search first.
The project map above gives you the starting orientation — drill down from there.
</scs-rules>
RULES
