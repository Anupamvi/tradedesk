import json
from pathlib import Path
from typing import Any, Dict

from xhigh.config import GATES_PATH


def load_gates(path: Path = None) -> Dict[str, Any]:
    src = Path(path) if path else GATES_PATH
    return json.loads(src.read_text(encoding="utf-8"))
