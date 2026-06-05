# FinSight-RAG

A starter retrieval-augmented generation (RAG) project for financial insight workflows.

## Project structure

- `data/` - raw and processed dataset assets.
- `notebooks/` - exploratory analysis and model experimentation.
- `src/` - reusable Python modules for data processing, vector indexing, and model integration.

## Setup

1. Create and activate a Python virtual environment:

```bash
cd /Users/girishrao/Desktop/FinSight-RAG
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the CLI helper:

```bash
python src/app.py --help
```
