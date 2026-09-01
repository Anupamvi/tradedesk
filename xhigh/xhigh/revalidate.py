from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from xhigh.config import OUT_DIR
from xhigh.dates import add_days
from xhigh.gates import load_gates
from xhigh.geometry import spot_from_quote
from xhigh import schwab
from xhigh.pipeline import _legal
from xhigh.report import write_run
from xhigh.schwab import use_live_schwab


def apply(date: str, out_root: Optional[Path] = None) -> Dict[str, Any]:
    dest = (Path(out_root) if out_root else OUT_DIR) / date
    tickets_path = dest / "tickets.json"
    if not tickets_path.is_file():
        return {"date": date, "out": str(dest / "board.md"), "revalidate": False}
    tickets = json.loads(tickets_path.read_text(encoding="utf-8"))
    gates = load_gates()
    live = use_live_schwab(date, live_flag=True, no_schwab=False)
    names = sorted({str(r.get("ticker") or "").upper() for r in tickets if r.get("ticker")})
    quotes = schwab.quotes_many(names, date) if live else {}
    from_d = add_days(date, int(gates.get("dte_min") or 25) - 2) or date
    to_d = add_days(date, int(gates.get("dte_max") or 45) + 2) or date
    keep = []
    drop = []
    for row in tickets:
        name = str(row.get("ticker") or "").upper()
        last = spot_from_quote(quotes.get(name) or {})
        if last is None or not _legal(row, last, gates):
            row = dict(row)
            row["action"] = "WATCH"
            row["note"] = "revalidate failed geometry"
            drop.append(row)
            continue
        row = dict(row)
        row["last"] = round(last, 2)
        keep.append(row)
    watch = json.loads((dest / "watch.json").read_text(encoding="utf-8")) if (dest / "watch.json").is_file() else []
    skips = json.loads((dest / "skips.json").read_text(encoding="utf-8")) if (dest / "skips.json").is_file() else []
    man = json.loads((dest / "manifest.json").read_text(encoding="utf-8")) if (dest / "manifest.json").is_file() else {}
    write_run(
        dest,
        date=date,
        tickets=keep,
        watch=watch + drop,
        x_queue=[r.get("ticker") for r in keep],
        gates=gates,
        skips=skips,
        manifest=man,
        macro=man.get("macro") if isinstance(man, dict) else {},
    )
    return {"date": date, "out": str(dest / "board.md"), "kept": len(keep), "dropped": len(drop)}
