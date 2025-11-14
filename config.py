"""Application configuration for the Image Transcription tool."""

from __future__ import annotations

from pathlib import Path

# Base paths
BASE_DIR: Path = Path(__file__).resolve().parent
INPUT_DIR: Path = BASE_DIR / "input"
PROCESSED_DIR: Path = BASE_DIR / "processed"
FAILED_DIR: Path = BASE_DIR / "failed"
LOG_DIR: Path = BASE_DIR / "logs"
LOG_FILE: Path = LOG_DIR / "process.log"
OUTPUT_FILE: Path = BASE_DIR / "output.csv"
PROGRESS_FILE: Path = BASE_DIR / "progress.json"

# Model settings
ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
OPENAI_MODEL: str = "gpt-4o"
MAX_TOKENS: int = 4096
API_RETRY_ATTEMPTS: int = 3
RETRY_DELAY_SECONDS: int = 2

# Processing settings
BATCH_SIZE: int = 1  # Process one image at a time
MOVE_PROCESSED_FILES: bool = True  # Move to processed/ folder
MOVE_FAILED_FILES: bool = True  # Move to failed/ folder

# Logging
LOG_LEVEL: str = "INFO"  # INFO, DEBUG, ERROR

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".pdf")
ESTIMATED_COST_PER_IMAGE: float = 0.02
INPUT_TOKEN_COST_PER_1K: float = 0.003  # USD per 1K input tokens (estimate)
OUTPUT_TOKEN_COST_PER_1K: float = 0.015  # USD per 1K output tokens (estimate)
