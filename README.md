# Paste and Chat Support Bot

Small local chatbot that answers questions using your pasted product notes or FAQs. It grounds every answer in your notes, cites the matching chunk, and remembers conversation context. If the answer is not found, it refuses and lists what info is missing.

Goals
- Paste short notes or index a local file, then chat.
- Show a cited quote from the matched chunk.
- Refuse to guess when unsupported and list missing details.
- Remember conversation across turns.

Tech
- Python 3.11
- LangChain (core pieces), `langchain-openai` for Chat Completions and Embeddings
- FAISS in-memory vector store with local persistence
- No external services beyond OpenAI

Quick Start
1) Setup
   - `cp .env.example .env` and set `OPENAI_API_KEY`.
   - Optionally adjust models and thresholds in `.env`.

2) Install
   - Requires Python 3.11. Other versions (e.g., 3.12/3.13) can fail to build `tiktoken` or `faiss`.
   - `make install` (creates local `.venv` and installs deps there)
   - If needed, select Python explicitly: `make install PY=python3.11`

3) Index notes
   - Paste text: `python app/ingest.py --stdin --title "Troubleshooting Guide" < samples/troubleshooting_guide.md`
   - Or from file: `python app/ingest.py --file samples/troubleshooting_guide.md --title "Troubleshooting Guide"`

4) CLI chat
   - `make run`
   - Ask: `How do I reset the connector?`
   - Follow-up: `What if that fails?`

5) UI (local)
   - `make ui`
   - Paste notes or upload a file in the sidebar, then chat. Sources appear in the right panel.

Example Session
- Q: How do I reset the connector?
- A: Brief steps, plus a quote and source like: "Power-cycle the connector by unplugging for 10 seconds …" From 'Reset Steps', lines 12–18
- Q: What if that fails?
- A: Uses chat history to return the fallback step with citation.
- Q: Are there warranty terms?
- A: I could not find this in your notes. Missing details to add: warranty duration, coverage, and RMA process.

Acceptance Checks
- Runs locally via CLI and Streamlit UI.
- Answers reset question using sample with correct citation.
- Follow-up uses chat history for fallback.
- Out-of-scope questions are declined with missing info listed.
- `pip install -r requirements.txt` installs only used deps.
- No icons, badges, bylines, or generated markers in repo.

Troubleshooting
- Missing key: Set `OPENAI_API_KEY` in `.env`.
- Index empty: Run `app/ingest.py` to build the index.
- Threshold too strict: Lower `RELEVANCE_THRESHOLD` in `.env`.
- Externally managed environment: Use `make install` (creates `.venv`).
- Build errors for `tiktoken` or `faiss`: ensure Python 3.11. Recreate env:
  - `make clean`
  - `make install PY=python3.11`

Commands
- Build/Install: `make install`
- Tests: `make test`
- CLI: `make run`
- UI: `make ui`
- Clean index: `make clean`

Notes
- If you hit “externally-managed-environment”, don’t use system Python. The Makefile now creates and uses `.venv` automatically.

License
MIT (see `LICENSE`).
