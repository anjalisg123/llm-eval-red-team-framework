.PHONY: install ingest serve eval dashboard test lint

install:
	python3.12 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"

ingest:      ## Chunk + embed the corpus into Chroma
	python -m target.ingest

serve:       ## Run the target RAG system (black box under test)
	uvicorn target.server:app --reload --port 8000

eval:        ## Run the full evaluation suite against the target
	python -m evals.runner

dashboard:   ## Launch the Streamlit results dashboard
	streamlit run dashboard/app.py

test:
	pytest

lint:
	ruff check .
