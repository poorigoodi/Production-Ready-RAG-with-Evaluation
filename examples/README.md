# Examples

Runnable demos for interview walkthroughs and local smoke checks.

| File | Requires API? | Description |
|---|---|---|
| [`example_offline_pipeline.py`](example_offline_pipeline.py) | No | Ingest → hybrid search → grounded chat in-process |
| [`example_api_usage.py`](example_api_usage.py) | Yes | Health, upload, search, chat, follow-up via HTTP |
| [`curl_examples.sh`](curl_examples.sh) | Yes | Same flows with curl |
| [`sample_outputs.json`](sample_outputs.json) | No | Captured mock-mode request/response samples |

```bash
# Offline (recommended first)
python examples/example_offline_pipeline.py

# With API running
uvicorn app.main:app --host 127.0.0.1 --port 8000
python examples/example_api_usage.py
```
