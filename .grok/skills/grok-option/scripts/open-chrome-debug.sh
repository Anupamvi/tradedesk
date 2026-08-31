#!/bin/zsh
# Launch a separate Chrome with CDP on 9222 for grok-option.
# Log into x.com in this window. Does not take over daily Chrome.
set -euo pipefail
PROFILE="${HOME}/.grok/browser/chrome-debug"
mkdir -p "$PROFILE"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [[ ! -x "$CHROME" ]]; then
  echo "Google Chrome not found at $CHROME" >&2
  exit 1
fi
echo "CDP: http://127.0.0.1:9222"
echo "Profile: $PROFILE"
exec "$CHROME" \
  --remote-debugging-port=9222 \
  --user-data-dir="$PROFILE" \
  "https://x.com"
