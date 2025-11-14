# Image Transcription Tool

Batch process images containing handwritten or printed notes using Claude or OpenAI vision models.

## Requirements

- Python 3.12 (managed via `.python-version`)
- [UV](https://docs.astral.sh/uv/) for dependency management
- Anthropic and/or OpenAI API key with image-reading capabilities

## Installation

### macOS
```bash
brew install uv
```

### Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Setup

1. Clone or download this repository and move into the `claudeocr` directory.
2. (Optional but recommended) Run the TUI setup helper:
   ```bash
   uv run python setup_cli.py
   ```
   This script checks your Python version, confirms the `uv` CLI is installed, ensures `.env` exists, and creates any missing folders. It runs in any regular macOS/Linux terminal (Terminal.app, iTerm2, GNOME Terminal, etc.).
3. Install Python dependencies:
   ```bash
   uv sync
   ```
4. Configure your API keys (skip if you already filled them in from the TUI):
   - Copy `.env.example` to `.env`
   - Add whichever of these you plan to use: `ANTHROPIC_API_KEY=...`, `OPENAI_API_KEY=...`
5. Prepare your assets:
   - Place source images inside the `input/` folder (jpg, jpeg, png, pdf)
   - Create a CSV or Excel template containing the columns you want in the output

## Usage

Run the CLI:
```bash
uv run python main.py
```

The program guides you through:
1. Loading your template file
2. Selecting a mode (dry run, preview, full batch, or resume)
3. Monitoring progress via terminal prompts and logs

Tip: A standard terminal session (macOS Terminal/iTerm, GNOME Terminal, Windows PowerShell) is sufficient—no container or VM is required once `uv` is installed.

## Output

- `output.csv`: Consolidated transcription results (appended per run)
- `processed/`: Images successfully processed and moved out of `input/`
- `failed/`: Images that failed after retries
- `logs/process.log`: Detailed processing log
- `progress.json`: Resume state for interrupted runs

## Notes

- All paths are handled with `pathlib` for cross-platform compatibility.
- The tool requires UTF-8 encoded templates; ensure Excel/CSV exports adhere to UTF-8.
- For Windows terminals, run commands inside a supported shell (PowerShell or Git Bash).

## Model Recommendations

- **Anthropic Claude Sonnet 4 (default)** – Best balance of accuracy and speed for mixed handwriting/print OCR, good at structured JSON outputs.
- **Anthropic Claude Haiku 4** – Faster/cheaper if handwriting is very clear and you need lower cost; update `ANTHROPIC_MODEL` in `config.py` to use it.
- **OpenAI GPT-4o (default)** – Strong OCR quality, works well on receipts/labels with diagrams; lower latency than legacy GPT-4.
- **OpenAI GPT-4o Mini** – Budget-friendly alternative when perfect accuracy is not critical.

You can change the defaults by editing `ANTHROPIC_MODEL` or `OPENAI_MODEL` in `config.py`. During runtime, the CLI automatically selects the provider whose API key is available (or prompts you when both keys exist).
