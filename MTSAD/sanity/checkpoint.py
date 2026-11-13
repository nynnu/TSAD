import json
import os
import time
from pathlib import Path


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    last_exc = None
    for _ in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(0.1)
    raise last_exc


def upsert_checkpoint(path: Path, case_id: str, row: dict) -> None:
    data = load_checkpoint(path)
    data[case_id] = row
    save_checkpoint(path, data)


def should_skip(entry: dict | None) -> bool:
    return bool(entry and entry.get("status") == "OK")


def checkpoint_to_rows(data: dict) -> list[dict]:
    rows = []
    for case_id, row in data.items():
        full = {"case_id": case_id}
        full.update(row)
        rows.append(full)
    return rows
