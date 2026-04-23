from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path


def pytest_configure(config) -> None:
    temp_logs = Path(tempfile.mkdtemp(prefix="virtual_audio_test_logs_"))
    os.environ["VIRTUAL_AUDIO_LOG_DIR"] = str(temp_logs)


def pytest_unconfigure(config) -> None:
    logger = logging.getLogger("virtual_audio_simple")
    for handler in list(logger.handlers):
        try:
            handler.close()
        except Exception:
            pass
        logger.removeHandler(handler)
