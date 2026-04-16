from __future__ import annotations

import ast
import pathlib
import shutil
import tempfile
from collections import Counter
from pathlib import Path


def _patch_path_exists_for_ultralytics() -> None:
    """GitRepo() walks parents for `.git`; on some Windows paths stat raises OSError."""
    original_exists = pathlib.Path.exists

    def _exists(self, *args, **kwargs):
        try:
            return original_exists(self, *args, **kwargs)
        except OSError:
            return False

    pathlib.Path.exists = _exists  # type: ignore[method-assign]


_patch_path_exists_for_ultralytics()

MODEL_CANDIDATES = [
    Path(__file__).resolve().parent / "model" / "fridge_best.pt",
    Path(__file__).resolve().parent / "model" / "best.pt",
    Path(__file__).resolve().parent / "backend" / "runs" / "detect" / "smart_fridge_train" / "weights" / "best.pt",
]

DATASET_CONFIG_CANDIDATES = [
    Path(__file__).resolve().parent / "FridgeVision.yolov8" / "data.local.yaml",
    Path(__file__).resolve().parent / "FridgeVision.yolov8" / "data.yaml",
]


def get_fridge_dataset_config_path() -> Path | None:
    for candidate in DATASET_CONFIG_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def get_fridge_dataset_info() -> dict[str, object]:
    config_path = get_fridge_dataset_config_path()
    if config_path is None:
        return {"available": False, "config_path": None, "class_names": []}

    class_names: list[str] = []
    base_path: str | None = None

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("path:"):
            base_path = line.split(":", 1)[1].strip()
        elif line.startswith("names:"):
            names_literal = line.split(":", 1)[1].strip()
            parsed = ast.literal_eval(names_literal)
            class_names = [str(name).strip() for name in parsed]

    dataset_root = config_path.parent
    if base_path:
        dataset_root = (Path(__file__).resolve().parent / base_path).resolve()

    return {
        "available": True,
        "config_path": str(config_path),
        "dataset_root": str(dataset_root),
        "class_count": len(class_names),
        "class_names": class_names,
    }


def get_fridge_model_path() -> Path | None:
    for candidate in MODEL_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def detect_items_from_upload(upload_file, conf: float = 0.25) -> list[dict[str, int | str]]:
    model_path = get_fridge_model_path()
    if model_path is None:
        raise FileNotFoundError(
            "Fridge model not found. Expected one of: "
            + ", ".join(str(path) for path in MODEL_CANDIDATES)
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed, so the fridge model cannot run") from exc

    suffix = Path(upload_file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        upload_file.file.seek(0)
        shutil.copyfileobj(upload_file.file, temp_file)

    try:
        model = YOLO(str(model_path))
        results = model.predict(source=str(temp_path), conf=conf, verbose=False)
        if not results:
            return []

        result = results[0]
        names = result.names
        counter = Counter()

        if result.boxes is not None and result.boxes.cls is not None:
            for cls_idx in result.boxes.cls.tolist():
                idx = int(cls_idx)
                name = names.get(idx, str(idx)) if isinstance(names, dict) else names[idx]
                counter[str(name)] += 1

        return [
            {"name": name, "count": int(count)}
            for name, count in sorted(counter.items(), key=lambda item: item[0])
        ]
    finally:
        temp_path.unlink(missing_ok=True)
