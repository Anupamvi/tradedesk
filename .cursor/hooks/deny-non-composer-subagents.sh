#!/usr/bin/env bash
INPUT=$(cat)
MODEL=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('subagent_model',''))")
TYPE=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('subagent_type',''))")
lower=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')
allow=false
if echo "$lower" | grep -qE 'composer|cursor-grok'; then allow=true; fi
if echo "$lower" | grep -q 'fast'; then allow=false; fi
if [ "$allow" = true ]; then echo '{"permission":"allow"}'; else echo "{\"permission\":\"deny\",\"user_message\":\"Blocked $TYPE $MODEL\"}"; fi
