from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from xhigh.config import CODE_DIR, OUT_DIR
from xhigh.report import overlay_intel


def apply(date: str, out_root: Optional[Path] = None) -> Dict[str, Any]:
    dest = (Path(out_root) if out_root else OUT_DIR) / date
    folder = CODE_DIR / "var" / "intel" / date
    names = []
    macro = {}
    if folder.is_dir():
        for path in sorted(folder.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if path.name == "macro.json":
                macro = payload if isinstance(payload, dict) else {}
                continue
            if isinstance(payload, dict) and payload.get("ticker"):
                names.append(payload)
            elif isinstance(payload, dict) and isinstance(payload.get("names"), list):
                names.extend([n for n in payload["names"] if isinstance(n, dict)])
    hot = {"asof": date, "source": "intel", "names": names, "macro": macro}
    overlay_intel(dest, hot)
    return {"date": date, "out": str(dest / "board.md"), "intel": True, "n": len(names)}
