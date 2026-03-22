# 1. Installation & Configuration

**What you'll do:** Clone the repository, install dependencies, and configure API keys.

## Install Dependencies

```bash
git clone https://github.com/hherb/biasbuster.git
cd biasbuster
uv sync
```

This creates a `.venv/` virtual environment and installs all Python dependencies.

For Apple Silicon (MLX training), also install the MLX group:

```bash
uv sync --group mlx
```

## Configure

Copy the example configuration and edit it:

```bash
cp config.example.py config.py
```

Open `config.py` and set the required values:

### Required Settings

| Setting | Description |
|---------|-------------|
| `crossref_mailto` | Your email address. Required by Crossref for polite-pool access. |

### API Keys (set at least one annotation backend)

| Setting | Environment Variable | Description |
|---------|---------------------|-------------|
| `anthropic_api_key` | `ANTHROPIC_API_KEY` | For Claude-based annotation (default backend) |
| `deepseek_api_key` | `DEEPSEEK_API_KEY` | For DeepSeek annotation (alternative backend) |
| `ncbi_api_key` | -- | Optional. Increases PubMed rate limit from 3/s to 10/s |

Environment variables take precedence over config file values for API keys.

### Key Defaults You May Want to Adjust

```python
# Where the SQLite database lives
db_path: str = "dataset/biasbuster.db"

# Collection limits (reduce for initial testing)
retraction_watch_max: int = 2000
spin_screening_max: int = 5000
cochrane_rob_max: int = 1000

# Annotation model
annotation_model: str = "claude-sonnet-4-6"
```

For a quick test run, consider reducing the collection limits to 50-100 papers per source.

## Verify Setup

```bash
uv run python -c "from config import Config; c = Config(); print(f'DB: {c.db_path}'); print(f'Domains: {len(c.focus_domains)}')"
```

You should see output like:

```
DB: dataset/biasbuster.db
Domains: 7
```

## Directory Structure

After setup, the project directory looks like this:

```
biasbuster/
├── config.py              # Your configuration (gitignored)
├── config.example.py      # Template
├── pipeline.py            # Main pipeline orchestrator
├── database.py            # SQLite database interface
├── export.py              # Training data exporter
├── prompts.py             # Single source of truth for all prompts
├── seed_database.py       # Post-collection cleanup & enrichment
├── collectors/            # Data source collectors
├── enrichers/             # Heuristic analysis modules
├── annotators/            # LLM annotation backends
├── schemas/               # Bias taxonomy dataclasses & enums
├── evaluation/            # Model evaluation framework
├── training/              # LoRA fine-tuning code (PyTorch + MLX)
├── agent/                 # Verification agent (tool-use loop)
├── gui/                   # Fine-Tuning Workbench (NiceGUI)
├── utils/                 # Review GUI, training monitor, retry helpers
├── dataset/               # Created at runtime (DB + exports)
└── training_output/       # Created during training
```

## Next Step

[Harvesting Training Data](02_data_collection.md) -- collect abstracts from PubMed, Retraction Watch, and Cochrane reviews.
