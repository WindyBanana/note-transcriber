"""Interactive setup assistant for the Image Transcription tool."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

BASE_DIR = Path(__file__).resolve().parent
console = Console()

StatusResult = Tuple[bool, str]


def _status(success: bool, message: str) -> StatusResult:
    return success, message


def check_python_version() -> StatusResult:
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 12):
        return _status(False, "Python 3.12+ required. Use pyenv/uv to select the bundled version.")
    return _status(True, f"Python {version.major}.{version.minor}.{version.micro} detected")


def check_uv_cli() -> StatusResult:
    uv_path = shutil.which("uv")
    if uv_path:
        return _status(True, f"uv found at {uv_path}")
    instructions = (
        "uv command not found. Install it via:\n"
        "- macOS: brew install uv\n"
        "- Linux: curl -LsSf https://astral.sh/uv/install.sh | sh\n"
        "- Windows: powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\""
    )
    return _status(False, instructions)


def ensure_env_file() -> StatusResult:
    env_file = BASE_DIR / ".env"
    example_file = BASE_DIR / ".env.example"
    if not env_file.exists():
        if example_file.exists():
            content = example_file.read_text(encoding="utf-8")
        else:
            content = "ANTHROPIC_API_KEY=\nOPENAI_API_KEY=\n"
        env_file.write_text(content, encoding="utf-8")
        console.print("Created .env from template.")
    else:
        console.print(".env already present. You can update keys below or skip to keep existing values.")

    updated: list[str] = []
    if Confirm.ask("Add/Update Anthropic API key now?", default=False):
        key = Prompt.ask("Enter Anthropic API key (leave blank to skip)", default="", password=True)
        if key:
            _set_env_value(env_file, "ANTHROPIC_API_KEY", key)
            updated.append("Anthropic key saved")
    if Confirm.ask("Add/Update OpenAI API key now?", default=False):
        key = Prompt.ask("Enter OpenAI API key (leave blank to skip)", default="", password=True)
        if key:
            _set_env_value(env_file, "OPENAI_API_KEY", key)
            updated.append("OpenAI key saved")

    if updated:
        return _status(True, ", ".join(updated))
    return _status(True, "Populate .env with ANTHROPIC_API_KEY and/or OPENAI_API_KEY before running main.py")


def _set_env_value(env_path: Path, key: str, value: str) -> None:
    lines = []
    found = False
    if env_path.exists():
        existing = env_path.read_text(encoding="utf-8").splitlines()
    else:
        existing = []
    for line in existing:
        if line.startswith(f"{key}="):
            lines.append(f"{key}={value}")
            found = True
        else:
            lines.append(line)
    if not found:
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def ensure_directories_step() -> StatusResult:
    created: list[str] = []
    for folder in ("input", "processed", "failed", "logs"):
        path = BASE_DIR / folder
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(folder)
    message = "Ensured required folders exist"
    if created:
        message += f" (created: {', '.join(created)})"
    return _status(True, message)


def show_next_steps() -> StatusResult:
    msg = (
        "Run `uv sync` to install Python dependencies, then execute `uv run python main.py` to start the tool.\n"
        "You can run everything from a normal macOS/Linux terminal (Terminal.app, iTerm, GNOME Terminal, etc.)."
    )
    console.print(Panel(msg, title="What next?", expand=False))
    return _status(True, "Next steps displayed")


@dataclass
class SetupStep:
    name: str
    action: Callable[[], StatusResult]
    status: str = "pending"
    detail: str = ""

    def run(self) -> None:
        try:
            success, detail = self.action()
        except Exception as exc:  # pylint: disable=broad-except
            success = False
            detail = str(exc)
        self.status = "done" if success else "error"
        self.detail = detail


STEPS = [
    SetupStep("Check Python version", check_python_version),
    SetupStep("Check uv CLI availability", check_uv_cli),
    SetupStep("Ensure .env file", ensure_env_file),
    SetupStep("Ensure required folders", ensure_directories_step),
    SetupStep("Review next steps", show_next_steps),
]


def render_step_table() -> None:
    table = Table(title="Setup Progress", box=box.SIMPLE_HEAVY)
    table.add_column("Step", justify="left", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Details", justify="left")
    status_icons = {
        "pending": "[yellow]●[/]",
        "done": "[green]✔[/]",
        "error": "[red]✖[/]",
    }
    for step in STEPS:
        icon = status_icons.get(step.status, "●")
        table.add_row(step.name, icon + " " + step.status.title(), step.detail or "")
    console.print(table)


def run_setup() -> None:
    console.print(Panel("Welcome to the Image Transcription setup assistant", style="bold cyan"))
    for step in STEPS:
        console.print(f"\n[bold]→ {step.name}[/]")
        step.run()
        render_step_table()
        if step.status == "error":
            console.print(
                "[red]Fix the issue above and rerun `uv run python setup_cli.py` to continue.[/]"
            )
            break
    else:
        console.print(
            Panel(
                "Setup complete! Run `uv sync` if you haven't already, then `uv run python main.py`.",
                style="bold green",
            )
        )


if __name__ == "__main__":
    run_setup()
