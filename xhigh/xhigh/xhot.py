from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from xhigh.config import CODE_DIR, OUT_DIR
from xhigh.report import overlay_x


def apply(date: str, out_root: Optional[Path] = None) -> Dict[str, Any]:
    dest = (Path(out_root) if out_root else OUT_DIR) / date
    hot_path = CODE_DIR / "var" / "xhot" / date / "hot.json"
    if not hot_path.is_file():
        return {"date": date, "out": str(dest / "board.md"), "xhot": False}
    hot = json.loads(hot_path.read_text(encoding="utf-8"))
    overlay_x(dest, hot)
    return {"date": date, "out": str(dest / "board.md"), "xhot": True}
