#!/usr/bin/env bash
# Example curl calls against a local API (MOCK_MODELS=true recommended).
set -euo pipefail
API="${API_URL:-http://127.0.0.1:8000}"

echo "=== Health ==="
curl -s "$API/health" | python -m json.tool

echo "=== Upload ==="
curl -s -F "file=@data/documents/sample/company_policy.txt" \
  "$API/documents/upload" | python -m json.tool

echo "=== Search ==="
curl -s -X POST "$API/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the refund period?","retrieval_mode":"hybrid","top_k":3,"rerank":true}' \
  | python -m json.tool

echo "=== Chat ==="
curl -s -X POST "$API/chat" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the refund policy?","retrieval_mode":"hybrid","top_k":3,"rerank":true}' \
  | python -m json.tool
