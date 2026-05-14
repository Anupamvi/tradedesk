#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/anuppamvi/uw_root/tradedesk"
WORKBOOK="/Users/anuppamvi/Library/Mobile Documents/com~apple~CloudDocs/Latest_Options.numbers"
PYTHON_BIN="${PYTHON_BIN:-/Users/anuppamvi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3}"
TODAY="$(date +%Y-%m-%d)"
STAMP="$(date +%Y%m%d-%H%M%S)"

START_DATE="${START_DATE:-2026-04-01}"
END_DATE="${END_DATE:-$TODAY}"
BACKFILL_DAYS="${BACKFILL_DAYS:-365}"
SHEET_NAME="${SHEET_NAME:-Codex Schwab Numbers}"

EXPORT_XLSX="$ROOT/out/codex_schwab_numbers_inspect.xlsx"
SCHWAB_JSON="$ROOT/out/schwab_positions_${TODAY}_schwab_numbers.json"
ROWS_JSON="$ROOT/out/codex_schwab_numbers_rows.json"

cp -p "$WORKBOOK" "$WORKBOOK.backup-schwab-numbers-$STAMP"

osascript <<APPLESCRIPT
set sourcePath to "$WORKBOOK"
set exportPath to "$EXPORT_XLSX"
tell application "Numbers"
	open (POSIX file sourcePath)
	delay 1
	export front document to (POSIX file exportPath) as Microsoft Excel
end tell
APPLESCRIPT

(
	cd "$ROOT"
	python3 schwab_pull.py \
		--force-backfill \
		--first-backfill-days "$BACKFILL_DAYS" \
		--backfill-chunk-days 30 \
		--output "$SCHWAB_JSON"
)

"$PYTHON_BIN" "$ROOT/scripts/update_latest_options_numbers.py" \
	--workbook "$WORKBOOK" \
	--export-xlsx "$EXPORT_XLSX" \
	--schwab-json "$SCHWAB_JSON" \
	--state-db "$ROOT/out/schwab_pull_state/schwab_pull_state.sqlite" \
	--sheet-name "$SHEET_NAME" \
	--start-date "$START_DATE" \
	--end-date "$END_DATE" \
	--out-json "$ROWS_JSON"

echo "Updated $SHEET_NAME in $WORKBOOK"
