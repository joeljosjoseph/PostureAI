from __future__ import annotations

import ast
import os
import pathlib
import shutil
import tempfile
from functools import lru_cache
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

BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / ".runtime"
MODEL_PATH_ENV_VARS = ("FRIDGE_MODEL_PATH", "YOLO_MODEL_PATH")
PERSISTENT_MODEL_CANDIDATES = [
    Path("/var/data/fridge_best.pt"),
    Path("/var/data/best.pt"),
]


def _configure_ultralytics_env() -> None:
    """Keep Ultralytics config inside the project to avoid Windows permission issues."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME_DIR))


_configure_ultralytics_env()

MODEL_CANDIDATES = [
    *PERSISTENT_MODEL_CANDIDATES,
    BASE_DIR / "models" / "fridge_best.pt",
    BASE_DIR / "models" / "best.pt",
    BASE_DIR / "model" / "fridge_best.pt",
    BASE_DIR / "model" / "best.pt",
    BASE_DIR / "best.pt",
    BASE_DIR / "yolov8n.pt",
    BASE_DIR / "backend" / "runs" / "detect" / "smart_fridge_train" / "weights" / "best.pt",
]

BUNDLED_MODEL_SOURCES = [
    BASE_DIR / "models" / "fridge_best.pt",
    BASE_DIR / "models" / "best.pt",
    BASE_DIR / "best.pt",
    RUNTIME_DIR / "repacked" / "best.pt",
]

DATASET_CONFIG_CANDIDATES = [
    BASE_DIR / "FridgeVision.yolov8" / "data.local.yaml",
    BASE_DIR / "FridgeVision.yolov8" / "data.yaml",
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
        dataset_root = (BASE_DIR / base_path).resolve()

    return {
        "available": True,
        "config_path": str(config_path),
        "dataset_root": str(dataset_root),
        "class_count": len(class_names),
        "class_names": class_names,
    }


def get_fridge_model_path() -> Path | None:
    _seed_persistent_model_if_needed()

    for env_var in MODEL_PATH_ENV_VARS:
        raw_path = os.getenv(env_var, "").strip()
        if not raw_path:
            continue

        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = (BASE_DIR / candidate).resolve()

        if candidate.is_file():
            return candidate

        unpacked = _resolve_unpacked_checkpoint(candidate)
        if unpacked is not None:
            return unpacked

    for candidate in MODEL_CANDIDATES:
        if candidate.is_file():
            return candidate
        unpacked = _resolve_unpacked_checkpoint(candidate)
        if unpacked is not None:
            return unpacked
    return None


def get_fridge_model_debug_info() -> dict[str, object]:
    env_values: dict[str, str | None] = {}
    for env_var in MODEL_PATH_ENV_VARS:
        raw_value = os.getenv(env_var, "").strip()
        env_values[env_var] = raw_value or None

    model_path = get_fridge_model_path()
    return {
        "available": model_path is not None,
        "resolved_model_path": str(model_path) if model_path else None,
        "env_vars": env_values,
        "candidate_paths": [str(path) for path in MODEL_CANDIDATES],
    }


def _find_unpacked_checkpoint_root(candidate: Path) -> Path | None:
    if not candidate.is_dir():
        return None

    nested = candidate / candidate.stem
    if (nested / "data.pkl").is_file():
        return nested
    if (candidate / "data.pkl").is_file():
        return candidate
    return None


def _build_repacked_checkpoint(checkpoint_root: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    archive_base = output_path.parent / output_path.stem
    zip_path = Path(
        shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=str(checkpoint_root.parent),
            base_dir=checkpoint_root.name,
        )
    )
    shutil.copyfile(zip_path, output_path)
    return output_path


def _get_bundled_model_file() -> Path | None:
    for candidate in BUNDLED_MODEL_SOURCES:
        if candidate.is_file():
            return candidate
        unpacked = _find_unpacked_checkpoint_root(candidate)
        if unpacked is not None:
            return _build_repacked_checkpoint(unpacked, RUNTIME_DIR / "repacked" / f"{candidate.stem}.pt")
    return None


def _seed_persistent_model_if_needed() -> None:
    target_paths: list[Path] = []

    for env_var in MODEL_PATH_ENV_VARS:
        raw_path = os.getenv(env_var, "").strip()
        if not raw_path:
            continue
        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            target_paths.append(candidate)

    for candidate in PERSISTENT_MODEL_CANDIDATES:
        if candidate not in target_paths:
            target_paths.append(candidate)

    source_file = _get_bundled_model_file()
    if source_file is None:
        return

    for target in target_paths:
        if target.exists() or target.suffix.lower() != ".pt":
            continue
        if not target.parent.exists():
            continue
        try:
            shutil.copyfile(source_file, target)
            return
        except OSError:
            continue


@lru_cache(maxsize=8)
def _resolve_unpacked_checkpoint(candidate: Path) -> Path | None:
    checkpoint_root = _find_unpacked_checkpoint_root(candidate)
    if checkpoint_root is None:
        return None

    cache_dir = RUNTIME_DIR / "repacked"
    cache_dir.mkdir(parents=True, exist_ok=True)
    repacked_path = cache_dir / f"{candidate.stem}.pt"

    if repacked_path.is_file():
        return repacked_path

    return _build_repacked_checkpoint(checkpoint_root, repacked_path)


def detect_items_from_upload(upload_file, conf: float = 0.25) -> list[dict[str, int | str]]:
    model_path = get_fridge_model_path()
    if model_path is None:
        raise FileNotFoundError(
            "Fridge model not found. Set FRIDGE_MODEL_PATH or place the checkpoint at one of: "
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
