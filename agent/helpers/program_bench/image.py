"""Docker image pull helpers used by both prepare.py and the runner."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from programbench.constants import DOCKER_EXECUTABLE


def pull_image(image_ref: str, *, timeout: int = 600) -> tuple[str, bool, str]:
    """Pull one image. Returns (image_ref, ok, message)."""
    try:
        result = subprocess.run(
            [DOCKER_EXECUTABLE, "pull", image_ref],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return image_ref, False, f"pull timed out after {timeout}s"
    if result.returncode != 0:
        return image_ref, False, (result.stderr or result.stdout or "").strip()
    return image_ref, True, "ok"


def pull_images_parallel(
    image_refs: list[str],
    *,
    workers: int = 4,
    timeout: int = 600,
) -> dict[str, tuple[bool, str]]:
    """Pull a batch of images in parallel. Returns {image_ref: (ok, message)}."""
    results: dict[str, tuple[bool, str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(pull_image, ref, timeout=timeout): ref for ref in image_refs}
        for fut in as_completed(futures):
            ref, ok, msg = fut.result()
            results[ref] = (ok, msg)
    return results
