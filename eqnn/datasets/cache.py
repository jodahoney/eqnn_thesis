from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from eqnn.datasets.heisenberg import DatasetBundle, HeisenbergDatasetConfig, generate_dataset
from eqnn.datasets.io import load_dataset_bundle, save_dataset_bundle
from eqnn.utils.timing import RuntimeProfile, timed


CACHE_NAMESPACE = "heisenberg_dataset_v1"


def dataset_cache_key(config: HeisenbergDatasetConfig) -> str:
    payload = {
        "namespace": CACHE_NAMESPACE,
        "config": asdict(config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def load_or_generate_cached_dataset(
    config: HeisenbergDatasetConfig,
    *,
    cache_dir: str | Path,
    profile: RuntimeProfile | None = None,
    force_rebuild: bool = False,
) -> tuple[DatasetBundle, Path, bool]:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    key = dataset_cache_key(config)
    target_dir = cache_root / key

    if target_dir.exists() and not force_rebuild:
        with timed(profile, "cache.dataset_load"):
            bundle = load_dataset_bundle(target_dir)
        return bundle, target_dir, True

    with timed(profile, "cache.dataset_generate"):
        bundle = generate_dataset(config, profile=profile)

    with timed(profile, "cache.dataset_save"):
        save_dataset_bundle(bundle, target_dir)

    return bundle, target_dir, False