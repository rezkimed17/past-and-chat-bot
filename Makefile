PY?=$(shell (command -v python3.11 >/dev/null 2>&1 && echo python3.11) || echo python3)
VENV?=.venv
BIN=$(VENV)/bin

.SILENT:  # keep output clean

.PHONY: install run test ui lint clean

install:
	[ -d $(VENV) ] || $(PY) -m venv $(VENV)
	$(BIN)/python -c 'import sys; import platform; v=sys.version_info; print(f"Using Python {v.major}.{v.minor}.{v.micro} ({platform.platform()})")'
	$(BIN)/python -c 'import sys; import sys; import textwrap; \

v=sys.version_info[:2]; \
required=(3,11); \
import sys as _s; \
(_s.exit(0) if v==required else (print(textwrap.dedent("""
ERROR: Python 3.11 is required for this project (wheels for tiktoken/faiss).
Recreate the venv with Python 3.11, e.g.:
  make clean
  make install PY=python3.11
""")), _s.exit(1)))'
	$(BIN)/python -m pip install -r requirements.txt

test:
	PYTHONPATH=. $(BIN)/pytest -q

run:
	PYTHONPATH=. $(BIN)/python app/main.py

ui:
	PYTHONPATH=. $(BIN)/streamlit run app/ui_streamlit.py

clean:
	rm -rf __pycache__ .pytest_cache index $(VENV)
