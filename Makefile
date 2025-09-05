PY?=$(shell (command -v python3.11 >/dev/null 2>&1 && echo python3.11) || echo python3)
VENV?=.venv
BIN=$(VENV)/bin

.SILENT:  # keep output clean

.PHONY: install run test ui lint clean ingest ingest-sample

install:
	[ -d $(VENV) ] || $(PY) -m venv $(VENV)
	$(BIN)/python -c 'import sys,platform; v=sys.version_info; print(f"Using Python {v.major}.{v.minor}.{v.micro} ({platform.platform()})")'
	@if ! $(BIN)/python -c 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) else 1)'; then \
	  echo 'ERROR: Python 3.11 is required for this project.'; \
	  echo 'Recreate the venv with: make clean && make install PY=python3.11'; \
	  exit 1; \
	fi
	$(BIN)/python -m pip install -r requirements.txt

ingest-sample:
	PYTHONPATH=. $(BIN)/python -m app.ingest --file samples/troubleshooting_guide.md --title "Troubleshooting Guide"

ingest:
	@if [ -z "$(FILE)" ] || [ -z "$(TITLE)" ]; then \
	  echo "Usage: make ingest FILE=path/to/file TITLE='Your Title'"; \
	  exit 1; \
	fi
	PYTHONPATH=. $(BIN)/python -m app.ingest --file $(FILE) --title "$(TITLE)"

test:
	PYTHONPATH=. $(BIN)/pytest -q

run:
	PYTHONPATH=. $(BIN)/python -m app.main

ui:
	PYTHONPATH=. $(BIN)/streamlit run app/ui_streamlit.py

clean:
	rm -rf __pycache__ .pytest_cache index $(VENV)
