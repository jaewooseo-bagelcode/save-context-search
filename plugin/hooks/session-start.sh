#!/bin/bash
# SCS SessionStart: refresh index + inject project map + agent workflow
SCS="${CLAUDE_PLUGIN_ROOT}/bin/scs"

# Refresh index (quiet, no embedding)
"$SCS" refresh --quiet 2>/dev/null

# Generate project map
"$SCS" map 2>/dev/null

# Agent workflow guide (injected as context)
cat <<'WORKFLOW'

// ── SCS Agent Workflow ──────────────────────────────────────
// Navigate: scs map --area <dir>          → zoom into module
//           scs map --area <file>         → see all functions
// Locate:   scs lookup "<SymbolName>"     → exact file:line
//           scs search "<concept>"        → find by meaning
// Read:     Read file:line_start-line_end → only needed lines
//
// Flow: map (above) → zoom (map --area) → locate → read
// Binary: ${CLAUDE_PLUGIN_ROOT}/bin/scs
WORKFLOW
