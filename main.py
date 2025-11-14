"""CLI entry point for the Image Transcription tool."""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import pandas as pd
from anthropic import Anthropic, APIError
from openai import OpenAI
from openai import OpenAIError
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import config

PROMPT_TEMPLATE = (
    "You are analyzing an image that may contain one or multiple separate notes, labels, "
    "receipts, or text snippets.\n"
    "Your task:\n\n"
    "Identify how many separate notes/items are in this image\n"
    "For each note, extract the text and structure it according to this template:\n"
    "{template_columns}\n\n"
    "Return a JSON array with one object per note:\n"
    "[\n"
    "{{\n"
    '"note_number": 1,\n'
    '"field1": "extracted value",\n'
    '"field2": "extracted value",\n'
    "...\n"
    "}},\n"
    "{{\n"
    '"note_number": 2,\n'
    "...\n"
    "}}\n"
    "]\n\n"
    "Rules:\n\n"
    "If a field is not visible or unclear, use null\n"
    "Be precise with numbers, dates, and amounts\n"
    "If text is ambiguous, use your best interpretation\n"
    "Number notes from top-left to bottom-right"
)


@dataclass
class ProgressState:
    """Represents persisted progress for resuming runs."""

    last_processed: str | None = None
    total_processed: int = 0
    total_notes: int = 0
    processed_files: set[str] = field(default_factory=set)
    timestamp: str | None = None

    @classmethod
    def from_file(cls, path: Path) -> ProgressState | None:
        if not path.exists() or path.stat().st_size == 0:
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError:
            logging.warning("Progress file is corrupt. Starting fresh.")
            return None
        if not data:
            return None
        processed_files = set(data.get("processed_files", []))
        return cls(
            last_processed=data.get("last_processed"),
            total_processed=int(data.get("total_processed", 0)),
            total_notes=int(data.get("total_notes", 0)),
            processed_files=processed_files,
            timestamp=data.get("timestamp"),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "last_processed": self.last_processed,
            "total_processed": self.total_processed,
            "total_notes": self.total_notes,
            "processed_files": sorted(self.processed_files),
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        }


@dataclass
class RunStats:
    processed_images: int = 0
    failed_images: int = 0
    total_notes: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def update_tokens(self, usage: dict[str, Any]) -> None:
        input_tokens = usage.get("input_tokens")
        if input_tokens is None:
            input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("output_tokens")
        if output_tokens is None:
            output_tokens = usage.get("completion_tokens", 0)
        self.total_input_tokens += int(input_tokens or 0)
        self.total_output_tokens += int(output_tokens or 0)


@dataclass
class ProviderClient:
    """Represents an API provider client and selected model."""

    name: Literal["anthropic", "openai"]
    client: Any
    model: str


def configure_logging() -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.FileHandler(config.LOG_FILE, encoding="utf-8")]
    # Mirror essential logs to stdout for user feedback
    handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def ensure_directories() -> None:
    for path in (config.INPUT_DIR, config.PROCESSED_DIR, config.FAILED_DIR, config.LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)
    if not config.PROGRESS_FILE.exists():
        config.PROGRESS_FILE.write_text("{}\n", encoding="utf-8")
    if not config.OUTPUT_FILE.exists():
        config.OUTPUT_FILE.touch()


def load_provider_client() -> ProviderClient:
    load_dotenv()
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    providers: dict[str, str] = {}
    if anthropic_key:
        providers["anthropic"] = anthropic_key
    if openai_key:
        providers["openai"] = openai_key

    if not providers:
        raise RuntimeError(
            "Missing API key. Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY in your .env file."
        )

    provider_name = next(iter(providers))
    if len(providers) > 1:
        provider_name = prompt_provider_choice()

    if provider_name == "anthropic":
        client = Anthropic(api_key=providers["anthropic"])
        return ProviderClient(name="anthropic", client=client, model=config.ANTHROPIC_MODEL)

    client = OpenAI(api_key=providers["openai"])
    return ProviderClient(name="openai", client=client, model=config.OPENAI_MODEL)


def prompt_provider_choice() -> Literal["anthropic", "openai"]:
    print("Multiple providers detected. Choose which API to use for this run:")
    print("[1] Anthropic Claude")
    print("[2] OpenAI GPT")
    while True:
        selection = input("Select provider: ").strip()
        if selection == "1":
            return "anthropic"
        if selection == "2":
            return "openai"
        print("Invalid choice. Enter 1 or 2.")


def prompt_for_template_columns() -> tuple[Path, list[str]]:
    while True:
        template_input = input("Please provide template file path: ").strip().strip('"')
        template_path = Path(template_input).expanduser().resolve()
        if not template_path.exists():
            print("Template not found. Please try again.")
            continue
        try:
            columns = load_template_columns(template_path)
        except ValueError as exc:
            print(f"Template error: {exc}. Please try again.")
            continue
        return template_path, columns


def load_template_columns(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise ValueError("Template must be CSV or Excel (.csv/.xlsx/.xls)")
    if suffix == ".csv":
        df = pd.read_csv(path, nrows=0, encoding="utf-8")
    else:
        df = pd.read_excel(path, nrows=0)
    columns: list[str] = [str(col).strip() for col in df.columns if str(col).strip()]
    if not columns:
        raise ValueError("Template file must include at least one column header")
    return columns


def scan_input_images() -> list[Path]:
    return sorted(
        [
            path
            for path in config.INPUT_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in config.SUPPORTED_EXTENSIONS
        ]
    )


def prompt_mode(allow_resume: bool) -> str:
    print("Step 2: Select Mode")
    print("[1] Dry run (no API calls, just estimate)")
    print("[2] Preview (first 5 images)")
    print("[3] Full batch (all images)")
    if allow_resume:
        print("[4] Resume previous run")
    while True:
        choice = input("Select: ").strip()
        if choice in {"1", "2", "3"}:
            return {"1": "dry_run", "2": "preview", "3": "full"}[choice]
        if choice == "4" and allow_resume:
            return "resume"
        print("Invalid selection. Please choose a valid option.")


def perform_dry_run(images: Sequence[Path]) -> None:
    print("Dry Run Results:")
    print(f"Images to process: {len(images)}")
    estimated_cost = len(images) * config.ESTIMATED_COST_PER_IMAGE
    estimated_time = max(len(images) * 0.2, 1)
    print(f"Estimated cost: ~${estimated_cost:.2f} ({len(images)} images × ${config.ESTIMATED_COST_PER_IMAGE:.2f})")
    print(f"Estimated time: ~{estimated_time:.0f} minutes")
    print("No API calls made. Ready to run for real!")


def calculate_resume_images(images: list[Path], progress: ProgressState | None) -> list[Path]:
    if not progress:
        return images
    if progress.processed_files:
        processed = {name.lower() for name in progress.processed_files}
        return [image for image in images if image.name.lower() not in processed]
    return images[progress.total_processed :]


def backup_previous_run() -> None:
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    backup_dir = config.BASE_DIR / f"backup_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in (config.OUTPUT_FILE, config.PROGRESS_FILE):
        if path.exists() and path.stat().st_size > 0:
            shutil.copy2(path, backup_dir / path.name)
    config.PROGRESS_FILE.write_text("{}\n", encoding="utf-8")
    print(f"Previous artifacts backed up to {backup_dir}.")


def validate_image(image_path: Path) -> None:
    if image_path.suffix.lower() == ".pdf":
        return
    try:
        with Image.open(image_path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Corrupt or unsupported image") from exc


def build_prompt(template_columns: Sequence[str]) -> str:
    template_listing = "\n".join(f"- {column}" for column in template_columns)
    return PROMPT_TEMPLATE.format(template_columns=template_listing)


def guess_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".pdf":
        return "application/pdf"
    return "application/octet-stream"


def call_model_api(
    provider: ProviderClient, image_path: Path, template_columns: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt = build_prompt(template_columns)
    image_bytes = image_path.read_bytes()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    media_type = guess_media_type(image_path)
    if provider.name == "anthropic":
        return call_anthropic_api(provider, prompt, encoded_image, media_type, image_path)
    return call_openai_api(provider, prompt, encoded_image, media_type, image_path)


def call_anthropic_api(
    provider: ProviderClient,
    prompt: str,
    encoded_image: str,
    media_type: str,
    image_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    message = provider.client.messages.create(
        model=provider.model,
        max_tokens=config.MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded_image,
                        },
                    },
                ],
            }
        ],
    )
    response_text = extract_text_response(message)
    notes = parse_notes_array(response_text, image_path)
    usage = getattr(message, "usage", {}) or {}
    return notes, usage


def call_openai_api(
    provider: ProviderClient,
    prompt: str,
    encoded_image: str,
    media_type: str,
    image_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    response = provider.client.responses.create(
        model=provider.model,
        max_output_tokens=config.MAX_TOKENS,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": {"url": f"data:{media_type};base64,{encoded_image}"},
                    },
                ],
            }
        ],
    )
    response_text = extract_openai_text(response)
    notes = parse_notes_array(response_text, image_path)
    usage_obj = getattr(response, "usage", None)
    usage = {
        "input_tokens": getattr(usage_obj, "input_tokens", 0)
        or getattr(usage_obj, "prompt_tokens", 0)
        or 0,
        "output_tokens": getattr(usage_obj, "output_tokens", 0)
        or getattr(usage_obj, "completion_tokens", 0)
        or 0,
    }
    return notes, usage


def parse_notes_array(response_text: str, image_path: Path) -> list[dict[str, Any]]:
    try:
        notes = json.loads(response_text)
    except json.JSONDecodeError as exc:
        logging.error("Invalid JSON for %s: %s", image_path.name, response_text)
        raise ValueError("Model response was not valid JSON") from exc
    if not isinstance(notes, list):
        raise ValueError("Model response must be a JSON array")
    return notes


def extract_text_response(message: Any) -> str:
    content = getattr(message, "content", [])
    texts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            texts.append(block.get("text", ""))
    combined = "\n".join(texts).strip()
    if not combined:
        raise ValueError("Claude response did not include any text content")
    return combined


def extract_openai_text(response: Any) -> str:
    outputs = getattr(response, "output", [])
    texts: list[str] = []
    for item in outputs:
        if isinstance(item, dict) and item.get("type") == "message":
            for block in item.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
    combined = "\n".join(texts).strip()
    if not combined:
        raise ValueError("OpenAI response did not include any text content")
    return combined


def normalize_notes(
    notes: list[dict[str, Any]], template_columns: Sequence[str]
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, raw_note in enumerate(notes, start=1):
        note_number = raw_note.get("note_number", index)
        row: dict[str, Any] = {
            "note_number": note_number,
        }
        for column in template_columns:
            row[column] = raw_note.get(column)
        normalized.append(row)
    return normalized


def append_rows_to_csv(
    rows: Iterable[dict[str, Any]], template_columns: Sequence[str], source_filename: str
) -> None:
    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    expanded_rows: list[dict[str, Any]] = []
    for row in rows:
        expanded = {
            "source_filename": source_filename,
            "note_number": row.get("note_number"),
            "timestamp": timestamp,
        }
        for column in template_columns:
            expanded[column] = row.get(column)
        expanded_rows.append(expanded)
    columns = ["source_filename", "note_number", *template_columns, "timestamp"]
    df = pd.DataFrame(expanded_rows, columns=columns)
    header_needed = config.OUTPUT_FILE.stat().st_size == 0
    df.to_csv(
        config.OUTPUT_FILE,
        mode="a",
        header=header_needed,
        index=False,
        encoding="utf-8",
    )


def move_file(source: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / source.name
    shutil.move(str(source), target)


def save_progress(progress: ProgressState) -> None:
    payload = progress.to_payload()
    with config.PROGRESS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def process_images(
    images: Sequence[Path],
    template_columns: Sequence[str],
    provider: ProviderClient,
    progress: ProgressState | None,
) -> RunStats:
    stats = RunStats()
    progress_state = progress or ProgressState()
    for image_path in tqdm(images, desc="Processing images"):
        try:
            validate_image(image_path)
            notes, usage = invoke_with_retry(provider, image_path, template_columns)
            normalized = normalize_notes(notes, template_columns)
            append_rows_to_csv(normalized, template_columns, image_path.name)
            stats.processed_images += 1
            stats.total_notes += len(normalized)
            progress_state.last_processed = image_path.name
            progress_state.total_processed += 1
            progress_state.total_notes += len(normalized)
            progress_state.processed_files.add(image_path.name)
            if usage:
                stats.update_tokens(usage)
            save_progress(progress_state)
            if config.MOVE_PROCESSED_FILES:
                move_file(image_path, config.PROCESSED_DIR)
        except Exception as exc:  # pylint: disable=broad-except
            logging.error("Failed to process %s: %s", image_path.name, exc)
            stats.failed_images += 1
            if config.MOVE_FAILED_FILES:
                try:
                    move_file(image_path, config.FAILED_DIR)
                except Exception as move_exc:  # pragma: no cover
                    logging.error("Failed to move %s to failed/: %s", image_path.name, move_exc)
    return stats


def invoke_with_retry(
    provider: ProviderClient, image_path: Path, template_columns: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for attempt in range(1, config.API_RETRY_ATTEMPTS + 1):
        try:
            return call_model_api(provider, image_path, template_columns)
        except (APIError, OpenAIError, ValueError) as exc:
            logging.warning(
                "Attempt %s/%s failed for %s: %s",
                attempt,
                config.API_RETRY_ATTEMPTS,
                image_path.name,
                exc,
            )
            if attempt == config.API_RETRY_ATTEMPTS:
                raise
            delay = config.RETRY_DELAY_SECONDS * attempt
            time.sleep(delay)
    raise RuntimeError("Exhausted retries")


def summarize_run(stats: RunStats) -> None:
    print("\nSummary:")
    print(f"Images processed: {stats.processed_images}")
    print(f"Images failed: {stats.failed_images}")
    print(f"Notes extracted: {stats.total_notes}")
    if stats.total_input_tokens or stats.total_output_tokens:
        input_cost = (stats.total_input_tokens / 1000) * config.INPUT_TOKEN_COST_PER_1K
        output_cost = (stats.total_output_tokens / 1000) * config.OUTPUT_TOKEN_COST_PER_1K
        total_cost = input_cost + output_cost
        print(
            "Token usage: "
            f"{stats.total_input_tokens} input / {stats.total_output_tokens} output"
        )
        print(f"Estimated API cost: ${total_cost:.2f}")
    else:
        estimated_cost = stats.processed_images * config.ESTIMATED_COST_PER_IMAGE
        print(f"Estimated API cost: ~${estimated_cost:.2f}")


def main() -> None:
    configure_logging()
    ensure_directories()
    print("Image Transcription Tool")
    try:
        provider = load_provider_client()
    except RuntimeError as exc:
        print(exc)
        return
    provider_label = "Anthropic Claude" if provider.name == "anthropic" else "OpenAI GPT"
    print(f"✓ API key loaded ({provider_label}, model {provider.model})")

    images = scan_input_images()
    print(f"✓ Found {len(images)} images in input/")
    if not images:
        print("Add images to the input/ folder and rerun the program.")
        return

    template_path, template_columns = prompt_for_template_columns()
    print(
        f"✓ Template loaded from {template_path.name}: {len(template_columns)} columns ({', '.join(template_columns)})"
    )

    progress_state = ProgressState.from_file(config.PROGRESS_FILE)
    if progress_state and progress_state.total_processed:
        print(
            f"Previous run found. Last processed image: {progress_state.last_processed}"
        )
    mode = prompt_mode(allow_resume=progress_state is not None)

    if mode == "dry_run":
        perform_dry_run(images)
        return

    if mode == "resume" and progress_state:
        resume = input(
            f"Previous run found ({progress_state.total_processed} images). Resume? (y/n): "
        ).strip().lower()
        if resume != "y":
            backup_previous_run()
            progress_state = None
    elif mode != "resume":
        progress_state = None

    images_to_process = images
    if mode == "preview":
        images_to_process = images[:5]
    elif mode == "full":
        images_to_process = images
    elif mode == "resume":
        images_to_process = calculate_resume_images(images, progress_state)

    if not images_to_process:
        print("No images to process with the current selection.")
        return

    print(f"Processing {len(images_to_process)} images...")
    start_time = time.time()
    stats = process_images(images_to_process, template_columns, provider, progress_state)
    elapsed_minutes = (time.time() - start_time) / 60
    summarize_run(stats)
    print(f"Time elapsed: {elapsed_minutes:.2f} minutes")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. You can resume using option 4 next time.")
