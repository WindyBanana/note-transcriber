"""Application configuration for Note Transcriber."""

from __future__ import annotations

from pathlib import Path

# Base paths
BASE_DIR: Path = Path(__file__).resolve().parent
INPUT_DIR: Path = BASE_DIR / "input"
PROCESSED_DIR: Path = BASE_DIR / "processed"
FAILED_DIR: Path = BASE_DIR / "failed"
REVIEW_DIR: Path = BASE_DIR / "review"  # For partial successes/suspicious results
LOG_DIR: Path = BASE_DIR / "logs"
LOG_FILE: Path = LOG_DIR / "process.log"
OUTPUT_FILE: Path = BASE_DIR / "output.csv"
PROGRESS_FILE: Path = BASE_DIR / "progress.json"
WARNINGS_FILE: Path = BASE_DIR / "warnings.log"

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
MOVE_REVIEW_FILES: bool = True  # Move suspicious results to review/ folder

# Validation settings
WARN_ON_EMPTY_EXTRACTION: bool = True  # Warn if no notes extracted
WARN_ON_MOSTLY_NULL: bool = True  # Warn if >50% fields are null
WARN_ON_INCOMPLETE_SPLIT: bool = True  # Warn if split pattern not applied
MIN_EXPECTED_NOTES: int = 1  # Minimum notes expected per image (0 = no minimum)
MAX_NULL_FIELDS_RATIO: float = 0.5  # If >50% fields are null, flag as suspicious

# Partial failure handling
ACCEPT_PARTIAL_EXTRACTION: bool = True  # Accept images even if some notes might be missing
REQUIRE_ALL_FIELDS: bool = False  # Fail if any required field is missing

# Logging
LOG_LEVEL: str = "INFO"  # INFO, DEBUG, ERROR

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".pdf")
ESTIMATED_COST_PER_IMAGE: float = 0.02
INPUT_TOKEN_COST_PER_1K: float = 0.003  # USD per 1K input tokens (estimate)
OUTPUT_TOKEN_COST_PER_1K: float = 0.015  # USD per 1K output tokens (estimate)

# Image compression settings
MAX_IMAGE_SIZE_MB: float = 2.5  # Max size in MB before base64 encoding (Claude limit ~5MB after encoding)
MAX_IMAGE_DIMENSION: int = 2560  # Max width/height in pixels (balances quality and size)
JPEG_QUALITY: int = 90  # JPEG compression quality (1-100) - higher preserves text better
