"""CLI entry point for Note Transcriber."""

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
from rich.console import Console
from rich.table import Table

import config

console = Console()

PROMPT_TEMPLATE = (
    "You are analyzing an image that may contain one or multiple separate notes, labels, "
    "receipts, or text snippets.\n\n"
    "TASK:\n"
    "1. Identify how many SEPARATE, DISTINCT notes/items are in this image\n"
    "   - Look for physical boundaries (separate sticky notes, cards, papers)\n"
    "   - Look for significant whitespace or visual separation\n"
    "   - Look for different sections or groupings\n\n"
    "2. For EACH separate note, extract the text and structure it according to these fields:\n"
    "{template_columns}\n\n"
    "FIELD EXTRACTION GUIDELINES:\n"
    "{field_guidelines}\n\n"
    "OUTPUT FORMAT:\n"
    "Return a JSON array with one object per note:\n"
    "[\n"
    "  {{\n"
    '    "note_number": 1,\n'
    '{field_example}'
    "  }},\n"
    "  {{\n"
    '    "note_number": 2,\n'
    "    ...\n"
    "  }}\n"
    "]\n\n"
    "RULES:\n"
    "• If a field is not visible or unclear, use null\n"
    "• Be precise with numbers, dates, and amounts\n"
    "• Match the expected format/pattern for each field\n"
    "• If text is ambiguous, use your best interpretation\n"
    "• Number notes sequentially from top-left to bottom-right\n"
    "• Each visually distinct note should be a separate object in the array"
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
    review_images: int = 0  # Images with warnings/suspicious results
    total_notes: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    warnings: list[str] = field(default_factory=list)  # Warnings collected during run

    def update_tokens(self, usage: dict[str, Any]) -> None:
        input_tokens = usage.get("input_tokens")
        if input_tokens is None:
            input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("output_tokens")
        if output_tokens is None:
            output_tokens = usage.get("completion_tokens", 0)
        self.total_input_tokens += int(input_tokens or 0)
        self.total_output_tokens += int(output_tokens or 0)

    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)
        logging.warning(warning)


@dataclass
class ProviderClient:
    """Represents an API provider client and selected model."""

    name: Literal["anthropic", "openai"]
    client: Any
    model: str


@dataclass
class ValidationResult:
    """Result of validating extracted notes."""

    is_valid: bool = True  # Overall validation status
    should_review: bool = False  # Should be moved to review folder
    warnings: list[str] = field(default_factory=list)  # Specific issues found
    notes_count: int = 0  # Number of notes extracted


@dataclass
class FieldMetadata:
    """Metadata describing a template column/field."""

    name: str
    data_type: str = "text"  # text, number, rating, date, etc.
    pattern: str | None = None  # e.g., "x-y" for ratings
    description: str | None = None  # e.g., "Usually a sentence"
    required: bool = False
    split_pattern: str | None = None  # e.g., "x" for first part of "x-y" split
    split_partner: str | None = None  # Name of the partner field in split pattern

    def to_prompt_description(self) -> str:
        """Generate AI prompt description for this field."""
        parts = [f'"{self.name}"']
        if self.description:
            parts.append(f"({self.description})")
        if self.pattern:
            parts.append(f"- Expected format: {self.pattern}")
        if self.data_type != "text":
            parts.append(f"- Type: {self.data_type}")
        return " ".join(parts)


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
    for path in (config.INPUT_DIR, config.PROCESSED_DIR, config.FAILED_DIR, config.REVIEW_DIR, config.LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)
    if not config.PROGRESS_FILE.exists():
        config.PROGRESS_FILE.write_text("{}\n", encoding="utf-8")
    if not config.OUTPUT_FILE.exists():
        config.OUTPUT_FILE.touch()
    if not config.WARNINGS_FILE.exists():
        config.WARNINGS_FILE.touch()


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


def prompt_for_template_columns() -> tuple[Path, list[FieldMetadata]]:
    """Prompt user for template file and configure field mappings."""
    console.print("\n[bold cyan]📋 Template Configuration[/bold cyan]")
    console.print("\n[yellow]Requirements:[/yellow]")
    console.print("  • Template must be CSV or Excel (.csv/.xlsx/.xls)")
    console.print("  • Column headers must be in Row 1")
    console.print("  • Each column represents a field to extract from notes")

    while True:
        template_input = input("\n📁 Template file path: ").strip().strip('"')
        template_path = Path(template_input).expanduser().resolve()

        if not template_path.exists():
            console.print("[red]❌ File not found. Please try again.[/red]")
            continue

        try:
            columns = load_template_columns(template_path)
            console.print(f"\n[green]✓ Found {len(columns)} columns in template[/green]")

            # Display detected columns
            display_template_structure(columns)

            # Interactive field mapping
            fields = configure_field_mappings(columns)

            return template_path, fields

        except ValueError as exc:
            console.print(f"[red]❌ {exc}[/red]")
            continue


def load_template_columns(path: Path) -> list[str]:
    """Load column names from template file."""
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

    # Validate no duplicate columns
    if len(columns) != len(set(columns)):
        duplicates = [col for col in columns if columns.count(col) > 1]
        raise ValueError(f"Duplicate column names found: {', '.join(set(duplicates))}")

    return columns


def display_template_structure(columns: list[str]) -> None:
    """Display template structure as a table."""
    table = Table(title="Template Structure", show_header=True, header_style="bold magenta")
    table.add_column("Column", style="cyan", width=20)
    table.add_column("Letter", style="yellow", justify="center", width=10)

    for idx, col in enumerate(columns):
        letter = chr(65 + idx)  # A, B, C, etc.
        table.add_row(col, letter)

    console.print(table)


def configure_field_mappings(columns: list[str]) -> list[FieldMetadata]:
    """Interactively configure field mappings or use smart defaults."""
    console.print("\n[bold cyan]🔧 Field Configuration[/bold cyan]")
    console.print("Configure how AI should interpret each field.\n")

    # Ask if user wants to configure manually or use defaults
    choice = input("Configure fields manually? (y/N): ").strip().lower()

    if choice == "y":
        return configure_fields_interactive(columns)
    else:
        return configure_fields_auto(columns)


def configure_fields_interactive(columns: list[str]) -> list[FieldMetadata]:
    """Interactive field configuration."""
    fields = []
    console.print("\n[yellow]For each field, specify its characteristics:[/yellow]")

    # First pass: collect basic info
    field_data = []
    for col in columns:
        console.print(f"\n[bold]{col}[/bold]")

        # Data type
        console.print("  Type: [1] Text [2] Number [3] Rating/Score [4] Date")
        type_choice = input("  Choose (1-4, default=1): ").strip() or "1"
        type_map = {"1": "text", "2": "number", "3": "rating", "4": "date"}
        data_type = type_map.get(type_choice, "text")

        # Pattern
        pattern = None
        if data_type == "rating":
            pattern = input("  Expected format (e.g., 'x-y', '1-5', default=x-y): ").strip() or "x-y"

        # Description
        description = input("  Description (optional, e.g., 'usually a sentence'): ").strip() or None

        field_data.append({
            "name": col,
            "data_type": data_type,
            "pattern": pattern,
            "description": description
        })

    # Ask about split patterns
    console.print("\n[bold cyan]Split Pattern Configuration[/bold cyan]")
    console.print("Do any two adjacent fields share a single 'x-y' value on the note?")
    console.print("(e.g., '3-5' where 3→Field1, 5→Field2)")
    has_split = input("Configure split pattern? (y/N): ").strip().lower() == "y"

    split_field1 = None
    split_field2 = None
    if has_split:
        console.print("\nAvailable fields:")
        for idx, col in enumerate(columns):
            console.print(f"  [{idx+1}] {col}")

        try:
            idx1 = int(input("First field (number before dash): ").strip()) - 1
            idx2 = int(input("Second field (number after dash): ").strip()) - 1

            if 0 <= idx1 < len(columns) and 0 <= idx2 < len(columns):
                split_field1 = columns[idx1]
                split_field2 = columns[idx2]
                console.print(f"[green]✓ Split pattern: {split_field1} (x) ← 'x-y' → {split_field2} (y)[/green]")
        except (ValueError, IndexError):
            console.print("[red]Invalid selection, skipping split pattern[/red]")

    # Create field metadata with split info
    for idx, data in enumerate(field_data):
        col = columns[idx]
        split_pattern = None
        split_partner = None

        if split_field1 and split_field2:
            if col == split_field1:
                split_pattern = "x"
                split_partner = split_field2
                data["data_type"] = "number"  # Override to number for split fields
            elif col == split_field2:
                split_pattern = "y"
                split_partner = split_field1
                data["data_type"] = "number"  # Override to number for split fields

        fields.append(FieldMetadata(
            name=data["name"],
            data_type=data["data_type"],
            pattern=data["pattern"],
            description=data["description"],
            required=False,
            split_pattern=split_pattern,
            split_partner=split_partner
        ))

    return fields


def configure_fields_auto(columns: list[str]) -> list[FieldMetadata]:
    """Automatic field configuration with smart defaults."""
    fields = []

    # Detect split pattern pairs (e.g., Verdivurdering and Gjennomførbarhet)
    verdi_idx = None
    gjennom_idx = None

    for idx, col in enumerate(columns):
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ["verdivurdering", "verdi"]) and "business" not in col_lower:
            verdi_idx = idx
        elif any(keyword in col_lower for keyword in ["gjennomførbarhet", "feasibility"]):
            gjennom_idx = idx

    # Check if they're adjacent (typical pattern)
    has_split_pattern = (
        verdi_idx is not None and
        gjennom_idx is not None and
        abs(verdi_idx - gjennom_idx) == 1
    )

    for idx, col in enumerate(columns):
        col_lower = col.lower()

        # Smart pattern detection based on common Norwegian/English field names
        if any(keyword in col_lower for keyword in ["verdivurdering", "verdi"]) and "business" not in col_lower:
            # Value assessment - may be part of split pattern
            if has_split_pattern and idx == verdi_idx:
                fields.append(FieldMetadata(
                    name=col,
                    data_type="number",
                    description="First number from x-y pattern on note",
                    split_pattern="x",
                    split_partner=columns[gjennom_idx]
                ))
            else:
                fields.append(FieldMetadata(
                    name=col,
                    data_type="rating",
                    pattern="x-y",
                    description="Numeric value or range"
                ))
        elif any(keyword in col_lower for keyword in ["gjennomførbarhet", "feasibility", "complexity"]):
            # Feasibility - may be part of split pattern
            if has_split_pattern and idx == gjennom_idx:
                fields.append(FieldMetadata(
                    name=col,
                    data_type="number",
                    description="Second number from x-y pattern on note",
                    split_pattern="y",
                    split_partner=columns[verdi_idx]
                ))
            else:
                fields.append(FieldMetadata(
                    name=col,
                    data_type="rating",
                    pattern="x-y",
                    description="Numeric score or range"
                ))
        elif any(keyword in col_lower for keyword in ["risk", "risiko"]):
            # Risk - could be text or number
            fields.append(FieldMetadata(
                name=col,
                data_type="text",
                description="Risk level or description"
            ))
        elif any(keyword in col_lower for keyword in ["beskrivelse", "description", "business"]):
            # Description fields - longer text
            fields.append(FieldMetadata(
                name=col,
                data_type="text",
                description="Usually a sentence or paragraph"
            ))
        elif any(keyword in col_lower for keyword in ["bedrift", "company", "organization", "kunde", "customer"]):
            # Entity names
            fields.append(FieldMetadata(
                name=col,
                data_type="text",
                description="Name or identifier"
            ))
        else:
            # Default: plain text
            fields.append(FieldMetadata(
                name=col,
                data_type="text"
            ))

    console.print("\n[green]✓ Using smart defaults for field configuration[/green]")

    # Check if we detected a split pattern
    has_split = any(f.split_pattern for f in fields)
    if has_split:
        split_fields = [f for f in fields if f.split_pattern]
        if len(split_fields) == 2:
            console.print(f"\n[bold yellow]📊 Split Pattern Detected:[/bold yellow]")
            console.print(f"  On the note: A single value like '[cyan]3-5[/cyan]' will be split:")
            console.print(f"    • [green]{split_fields[0].name}[/green] = 3 (first number)")
            console.print(f"    • [green]{split_fields[1].name}[/green] = 5 (second number)")

    # Display configured fields
    table = Table(title="Field Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Field", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Split", style="green")
    table.add_column("Description", style="white")

    for field in fields:
        split_info = field.split_pattern if field.split_pattern else "-"
        table.add_row(
            field.name,
            field.data_type,
            split_info,
            field.description or "-"
        )

    console.print(table)

    return fields


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


def build_prompt(fields: Sequence[FieldMetadata]) -> str:
    """Build AI prompt with field metadata."""
    # Column listing
    template_listing = "\n".join(f"- {field.name}" for field in fields)

    # Check for split patterns
    split_pairs = []
    for field in fields:
        if field.split_pattern == "x" and field.split_partner:
            split_pairs.append((field.name, field.split_partner))

    # Detailed field guidelines
    field_guidelines = []
    for field in fields:
        guideline = f'• "{field.name}"'
        if field.description:
            guideline += f" - {field.description}"

        # Special handling for split pattern fields
        if field.split_pattern and field.split_partner:
            if field.split_pattern == "x":
                guideline += f'\n  IMPORTANT: This field is paired with "{field.split_partner}" and shares a single value on the note.'
                guideline += f'\n  Extract ONLY the FIRST number and put it in this field.'
                guideline += f'\n  Common formats you might see:'
                guideline += f'\n    • "3-5" → {field.name}=3, {field.split_partner}=5'
                guideline += f'\n    • "V3 F5" or "v3 f5" → {field.name}=3, {field.split_partner}=5'
                guideline += f'\n    • "v:3 f:5" or "V:3 F:5" → {field.name}=3, {field.split_partner}=5'
                guideline += f'\n    • "2/4" → {field.name}=2, {field.split_partner}=4'
                guideline += f'\n    • "v2-v3" → {field.name}=2, {field.split_partner}=3'
                guideline += f'\n  Look for TWO numbers on the note and extract the FIRST one here.'
            elif field.split_pattern == "y":
                guideline += f'\n  IMPORTANT: Extract the SECOND number from the paired value.'
                guideline += f'\n  This is paired with "{field.split_partner}" - extract the second/right number.'
        elif field.pattern:
            guideline += f"\n  Expected format: {field.pattern}"

        if field.data_type != "text":
            guideline += f"\n  Type: {field.data_type}"

        field_guidelines.append(guideline)

    guidelines_text = "\n".join(field_guidelines)

    # Add special note about split patterns
    if split_pairs:
        split_note = "\n\n⚠️ CRITICAL SPLIT PATTERN RULE:\n"
        for first_field, second_field in split_pairs:
            split_note += f'The note will show TWO numbers representing "{first_field}" and "{second_field}".\n'
            split_note += f'These can be written in VARIOUS formats:\n'
            split_note += f'  • "3-5" → {first_field}=3, {second_field}=5\n'
            split_note += f'  • "V3 F5" or "v3 f5" → {first_field}=3, {second_field}=5\n'
            split_note += f'  • "v:3 f:5" → {first_field}=3, {second_field}=5\n'
            split_note += f'  • "2/4" → {first_field}=2, {second_field}=4\n'
            split_note += f'  • "v2-v3" → {first_field}=2, {second_field}=3\n'
            split_note += f'\nALWAYS split into separate values - NEVER put the entire string in one field!\n'
            split_note += f'If you see "V:3 F:5", output: {first_field}=3, {second_field}=5 (NOT {first_field}="V:3 F:5")\n'
        guidelines_text += split_note

    # Example field structure
    field_examples = []
    for field in fields:
        if field.split_pattern == "x":
            example_value = "3"  # First number from split
        elif field.split_pattern == "y":
            example_value = "5"  # Second number from split
        elif field.pattern:
            example_value = field.pattern  # e.g., "x-y" for ratings
        elif field.data_type == "number":
            example_value = "123"
        else:
            example_value = "extracted text"
        field_examples.append(f'    "{field.name}": "{example_value}"')

    field_example_text = ",\n".join(field_examples)
    if field_example_text:
        field_example_text = f"    {field_example_text},\n"

    return PROMPT_TEMPLATE.format(
        template_columns=template_listing,
        field_guidelines=guidelines_text,
        field_example=field_example_text
    )


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
    provider: ProviderClient, image_path: Path, fields: Sequence[FieldMetadata]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt = build_prompt(fields)
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


def validate_extraction(
    notes: list[dict[str, Any]],
    fields: Sequence[FieldMetadata],
    image_path: Path,
) -> ValidationResult:
    """Validate extraction quality and detect potential issues."""
    result = ValidationResult(notes_count=len(notes))
    template_columns = [field.name for field in fields]

    # Check 1: Empty extraction
    if len(notes) == 0:
        if config.WARN_ON_EMPTY_EXTRACTION:
            result.warnings.append(f"{image_path.name}: No notes extracted (empty array)")
            result.should_review = True
        if config.MIN_EXPECTED_NOTES > 0:
            result.is_valid = False
            result.warnings.append(f"{image_path.name}: Expected at least {config.MIN_EXPECTED_NOTES} notes")
        return result

    # Check 2: Minimum expected notes
    if config.MIN_EXPECTED_NOTES > 0 and len(notes) < config.MIN_EXPECTED_NOTES:
        result.warnings.append(
            f"{image_path.name}: Only {len(notes)} notes extracted (expected at least {config.MIN_EXPECTED_NOTES})"
        )
        result.should_review = True

    # Check 3: Mostly null fields (suspicious quality)
    for note_idx, note in enumerate(notes, start=1):
        null_count = sum(1 for col in template_columns if not note.get(col))
        null_ratio = null_count / len(template_columns) if template_columns else 0

        if config.WARN_ON_MOSTLY_NULL and null_ratio > config.MAX_NULL_FIELDS_RATIO:
            result.warnings.append(
                f"{image_path.name} note #{note_idx}: {null_count}/{len(template_columns)} fields are null ({null_ratio:.0%})"
            )
            result.should_review = True

    # Check 4: Split pattern validation
    split_fields = [(f.name, f.split_partner, f.split_pattern) for f in fields if f.split_pattern]
    if split_fields and config.WARN_ON_INCOMPLETE_SPLIT:
        for note_idx, note in enumerate(notes, start=1):
            for field_name, partner_name, pattern_type in split_fields:
                if pattern_type == "x":  # Only check from the first field to avoid duplicate warnings
                    val1 = note.get(field_name)
                    val2 = note.get(partner_name) if partner_name else None

                    # Check if split was applied (both should be single numbers, not compound patterns)
                    if val1 and isinstance(val1, str):
                        # Check for various unsplit patterns
                        suspicious_patterns = [
                            "-",           # "3-5"
                            "/",           # "3/5"
                            " ",           # "3 5" or "V3 F5"
                            ":",           # "v:3 f:5"
                            "v",           # "v2-v3" or "V3"
                            "V",           # "V3 F5"
                            "f",           # "f5"
                            "F",           # "F5"
                        ]

                        has_suspicious = any(p in val1.lower() for p in suspicious_patterns)
                        # But allow if it's just a single digit/number
                        is_simple_number = val1.strip().isdigit()

                        if has_suspicious and not is_simple_number:
                            result.warnings.append(
                                f"{image_path.name} note #{note_idx}: '{field_name}' contains '{val1}' (looks like unsplit pattern - should be two separate numbers)"
                            )
                            result.should_review = True

                    # Check if both fields are populated
                    if not val1 or not val2:
                        if val1 or val2:  # One is filled but not the other
                            result.warnings.append(
                                f"{image_path.name} note #{note_idx}: Incomplete split pattern ({field_name}={val1}, {partner_name}={val2})"
                            )
                            result.should_review = True

    # Check 5: Required fields (if configured)
    if config.REQUIRE_ALL_FIELDS:
        required_fields = [f.name for f in fields if f.required]
        for note_idx, note in enumerate(notes, start=1):
            missing = [col for col in required_fields if not note.get(col)]
            if missing:
                result.warnings.append(
                    f"{image_path.name} note #{note_idx}: Missing required fields: {', '.join(missing)}"
                )
                result.should_review = True

    return result


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
    fields: Sequence[FieldMetadata],
    provider: ProviderClient,
    progress: ProgressState | None,
) -> RunStats:
    stats = RunStats()
    progress_state = progress or ProgressState()
    # Extract column names for CSV operations
    template_columns = [field.name for field in fields]

    for image_path in tqdm(images, desc="Processing images"):
        try:
            validate_image(image_path)
            notes, usage = invoke_with_retry(provider, image_path, fields)

            # Validate extraction quality
            validation = validate_extraction(notes, fields, image_path)

            # Handle validation failures
            if not validation.is_valid and not config.ACCEPT_PARTIAL_EXTRACTION:
                raise ValueError(f"Validation failed: {'; '.join(validation.warnings)}")

            # Log warnings
            for warning in validation.warnings:
                stats.add_warning(warning)

            # Write extracted data to CSV
            normalized = normalize_notes(notes, template_columns)
            if len(normalized) > 0:  # Only write if we got some data
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

            # Move file to appropriate folder
            if validation.should_review and config.MOVE_REVIEW_FILES:
                move_file(image_path, config.REVIEW_DIR)
                stats.review_images += 1
                logging.info("Moved %s to review/ (suspicious extraction quality)", image_path.name)
            elif config.MOVE_PROCESSED_FILES:
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
    provider: ProviderClient, image_path: Path, fields: Sequence[FieldMetadata]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for attempt in range(1, config.API_RETRY_ATTEMPTS + 1):
        try:
            return call_model_api(provider, image_path, fields)
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
    """Display run summary with validation warnings."""
    console.print("\n[bold cyan]📊 Processing Summary[/bold cyan]")

    # Success/failure breakdown
    total_images = stats.processed_images + stats.failed_images
    success_rate = (stats.processed_images / total_images * 100) if total_images > 0 else 0

    console.print(f"\n[bold]Images:[/bold]")
    console.print(f"  ✓ Processed: [green]{stats.processed_images}[/green]")
    if stats.review_images > 0:
        console.print(f"  ⚠ Review needed: [yellow]{stats.review_images}[/yellow] (suspicious quality)")
    if stats.failed_images > 0:
        console.print(f"  ✗ Failed: [red]{stats.failed_images}[/red]")
    console.print(f"  Success rate: {success_rate:.1f}%")

    console.print(f"\n[bold]Notes extracted:[/bold] {stats.total_notes}")
    if stats.processed_images > 0:
        avg_notes = stats.total_notes / stats.processed_images
        console.print(f"  Average per image: {avg_notes:.1f}")

    # Token usage and cost
    if stats.total_input_tokens or stats.total_output_tokens:
        input_cost = (stats.total_input_tokens / 1000) * config.INPUT_TOKEN_COST_PER_1K
        output_cost = (stats.total_output_tokens / 1000) * config.OUTPUT_TOKEN_COST_PER_1K
        total_cost = input_cost + output_cost
        console.print(
            f"\n[bold]Token usage:[/bold] "
            f"{stats.total_input_tokens:,} input / {stats.total_output_tokens:,} output"
        )
        console.print(f"[bold]Estimated API cost:[/bold] ${total_cost:.2f}")
    else:
        estimated_cost = stats.processed_images * config.ESTIMATED_COST_PER_IMAGE
        console.print(f"\n[bold]Estimated API cost:[/bold] ~${estimated_cost:.2f}")

    # Warnings
    if stats.warnings:
        console.print(f"\n[bold yellow]⚠ Warnings ({len(stats.warnings)}):[/bold yellow]")
        # Group warnings by type
        warning_types: dict[str, int] = {}
        for warning in stats.warnings:
            if "No notes extracted" in warning:
                warning_types["Empty extraction"] = warning_types.get("Empty extraction", 0) + 1
            elif "fields are null" in warning:
                warning_types["Mostly null fields"] = warning_types.get("Mostly null fields", 0) + 1
            elif "Incomplete split pattern" in warning:
                warning_types["Incomplete split pattern"] = warning_types.get("Incomplete split pattern", 0) + 1
            elif "contains" in warning and "expected split" in warning:
                warning_types["Split not applied"] = warning_types.get("Split not applied", 0) + 1
            elif "Missing required fields" in warning:
                warning_types["Missing required fields"] = warning_types.get("Missing required fields", 0) + 1
            else:
                warning_types["Other"] = warning_types.get("Other", 0) + 1

        for warning_type, count in warning_types.items():
            console.print(f"  • {warning_type}: {count}")

        console.print(f"\n[yellow]Details saved to: {config.WARNINGS_FILE}[/yellow]")

        # Save warnings to file
        with config.WARNINGS_FILE.open("a", encoding="utf-8") as f:
            f.write(f"\n=== Run at {datetime.utcnow().isoformat()} ===\n")
            for warning in stats.warnings:
                f.write(f"{warning}\n")

    # Folder locations
    console.print(f"\n[bold]Output locations:[/bold]")
    console.print(f"  • Extracted data: [cyan]{config.OUTPUT_FILE}[/cyan]")
    if stats.processed_images > 0:
        console.print(f"  • Processed images: [cyan]{config.PROCESSED_DIR}/[/cyan]")
    if stats.review_images > 0:
        console.print(f"  • Review needed: [yellow]{config.REVIEW_DIR}/[/yellow] (check these manually)")
    if stats.failed_images > 0:
        console.print(f"  • Failed images: [red]{config.FAILED_DIR}/[/red]")

    console.print()


def main() -> None:
    configure_logging()
    ensure_directories()
    print("Note Transcriber")
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

    template_path, fields = prompt_for_template_columns()
    column_names = [field.name for field in fields]
    console.print(
        f"\n[green]✓ Template configured: {len(fields)} fields ({', '.join(column_names)})[/green]\n"
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
    stats = process_images(images_to_process, fields, provider, progress_state)
    elapsed_minutes = (time.time() - start_time) / 60
    summarize_run(stats)
    print(f"Time elapsed: {elapsed_minutes:.2f} minutes")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. You can resume using option 4 next time.")
