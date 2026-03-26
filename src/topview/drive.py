from __future__ import annotations

from pathlib import Path


def maybe_mount_google_drive() -> None:
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        return

    if not Path("/content/drive").exists():
        drive.mount("/content/drive")
