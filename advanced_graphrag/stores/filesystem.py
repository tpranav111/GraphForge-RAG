"""Filesystem artifact store."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from advanced_graphrag.exceptions import StorageError


class FileSystemArtifactStore:
    """Atomic JSON artifact store for checkpoints, manifests, and audits."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put_json(self, key: str, value: dict[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            os.replace(tmp_name, path)
        except Exception as exc:
            try:
                os.remove(tmp_name)
            except OSError:
                pass
            raise StorageError(f"failed to write artifact {key}") from exc

    def get_json(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            raise StorageError(f"failed to read artifact {key}") from exc

    def _path(self, key: str) -> Path:
        safe = key.strip("/").replace("\\", "/")
        if ".." in Path(safe).parts:
            raise StorageError("artifact keys cannot contain parent directory traversal")
        return self.root / f"{safe}.json"

