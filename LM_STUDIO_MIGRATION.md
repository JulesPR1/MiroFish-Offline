# Migration Guide: Ollama → LM Studio

Complete guide to converting MiroFish-Offline from Ollama to LM Studio backend.

## Overview

This document explains all changes made to run MiroFish-Offline with **LM Studio** instead of **Ollama**. The key difference is:

- **Ollama**: Docker service on port 11434 with proprietary `/api/embed` endpoint
- **LM Studio**: Host machine application on port 1234 with OpenAI-compatible `/v1/*` endpoints

---

## File-by-File Changes

### 1. `.env.example` — Configuration Template

**What changed**: Environment variables now point to LM Studio instead of Ollama.

#### Before (Ollama):
```env
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL_NAME=qwen2.5:32b

EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_BASE_URL=http://localhost:11434

OPENAI_API_KEY=ollama
OPENAI_API_BASE_URL=http://localhost:11434/v1
```

#### After (LM Studio):
```env
LLM_API_KEY=API_USELESS_TOKEN
LLM_BASE_URL=http://127.0.0.1:1234/v1
LLM_MODEL_NAME=qwen3.5-35b-a3b

EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_BASE_URL=http://127.0.0.1:1234

OPENAI_API_KEY=API_USELESS_TOKEN
OPENAI_API_BASE_URL=http://127.0.0.1:1234
```

#### Docker Mode Changes:
**Before:**
```env
# For Docker: use service name "ollama" instead of "localhost"
LLM_BASE_URL=http://ollama:11434/v1
EMBEDDING_BASE_URL=http://ollama:11434
OPENAI_API_BASE_URL=http://ollama:11434/v1
```

**After:**
```env
# LM Studio runs on your HOST machine, not in Docker
LLM_BASE_URL=http://host.docker.internal:1234/v1
EMBEDDING_BASE_URL=http://host.docker.internal:1234
OPENAI_API_BASE_URL=http://host.docker.internal:1234
```

**Why**: LM Studio doesn't run in Docker — it's a native desktop application. From Docker containers, you reach the host via `host.docker.internal` (Mac/Windows) or manually configured DNS (Linux).

---

### 2. `docker-compose.yml` — Container Orchestration

**What changed**: Removed Ollama service, added host networking for LM Studio.

#### Before:
```yaml
services:
  mirofish:
    depends_on:
      ollama:
        condition: service_started

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  ollama_data:
```

#### After:
```yaml
services:
  mirofish:
    extra_hosts:
      - "host.docker.internal:host-gateway"
    # (ollama service removed)

# (ollama_data volume removed)
```

**Why:**
- **Removed Ollama service**: LM Studio runs on the host, not in Docker
- **Added `extra_hosts`**: Enables containers to reach `host.docker.internal` (especially important on Linux where it's not automatic)
- **Simplified dependencies**: No more wait for Ollama startup

---

### 3. `backend/app/storage/embedding_service.py` — Embeddings API

**What changed**: Changed from Ollama's proprietary `/api/embed` to OpenAI-compatible `/v1/embeddings`.

#### Endpoint Change:
```python
# Before (Ollama)
self._embed_url = f"{self.base_url}/api/embed"

# After (OpenAI-compatible)
self._embed_url = f"{self.base_url}/v1/embeddings"
```

#### Response Format Change:
Ollama uses: `{"embeddings": [...]}`
OpenAI format uses: `{"data": [{"embedding": [...], "index": 0}, ...]}`

**Code before:**
```python
embeddings = data.get("embeddings", [])
return embeddings
```

**Code after:**
```python
# OpenAI-compatible format
items = data.get("data", [])
items.sort(key=lambda x: x.get("index", 0))
embeddings = [item["embedding"] for item in items]
return embeddings
```

**Why:** LM Studio exposes embeddings via the OpenAI `/v1/embeddings` standard endpoint.

#### Logging Improvements:
- Added initialization logging: `"EmbeddingService initialized: url=..., model=..., timeout=..."`
- Added detailed request/response logging at DEBUG level
- Better error messages (generic "embedding server" instead of "Ollama")

---

### 4. `backend/app/utils/llm_client.py` — LLM API Wrapper

**What changed**:
1. Removed `json_schema` response format (LM Studio doesn't support it)
2. Added comprehensive logging
3. Made code more generic (supports Ollama, LM Studio, OpenAI, etc.)

#### Response Format Removal:
```python
# Before: attempted to use json_schema
response_format={"type": "json_schema"}

# After: rely on prompt to guide JSON output
# (no response_format parameter sent)
```

**Why:** LM Studio's OpenAI compatibility layer doesn't support the newer `json_schema` format. The prompt is explicit enough that the model outputs valid JSON anyway.

#### Logging Added:
```python
logger.debug(f"Chat request: model={self.model}, num_messages={len(messages)}")
logger.debug(f"Sending request to LLM at {self.base_url}...")
logger.debug(f"LLM response received, length: {len(content)} characters")
```

**Why:** Better visibility into what's happening during LLM calls.

#### Generic Comments:
```python
# Before: "Supports Ollama num_ctx parameter to prevent prompt truncation"
# After: "Supports Ollama num_ctx parameter to prevent prompt truncation when using Ollama"
```

This clarifies that `num_ctx` is Ollama-specific (LM Studio doesn't need it).

---

### 5. `backend/app/services/ontology_generator.py` — Ontology Generation

**What changed**: Added logging and error handling.

#### Before:
```python
result = self.llm_client.chat_json(
    messages=messages,
    temperature=0.3,
    max_tokens=8192
)
```

#### After:
```python
try:
    logger.info(f"Starting ontology generation with {len(document_texts)} document(s)")
    # ... build messages ...
    logger.info("Sending request to LLM for ontology generation...")
    result = self.llm_client.chat_json(...)
    logger.info(f"LLM returned response, validating...")
    result = self._validate_and_process(result)
    logger.info(f"Ontology generation completed: {len(result.get('entity_types', []))} entity types, {len(result.get('edge_types', []))} edge types")
    return result
except Exception as e:
    logger.error(f"Ontology generation failed: {error_msg}")
    logger.error(f"Traceback:\n{tb}")
    raise
```

**Why:** When errors occur, having detailed logs helps identify whether it's:
- Configuration issue (wrong model name, port, etc.)
- Network issue (LM Studio not running)
- LLM parsing error (invalid JSON output)

---

### 6. `backend/app/api/graph.py` — API Endpoints

**What changed**: Enhanced error logging for the ontology generation endpoint.

#### Added debugging prints and logging:
```python
print("[GENERATE_ONTOLOGY] START", flush=True)
try:
    print("[GENERATE_ONTOLOGY] In try block", flush=True)
    # ... process ...
except Exception as e:
    print(f"[EXCEPTION] Ontology generation failed: {error_msg}", flush=True)
    print(f"[TRACEBACK]\n{tb}", flush=True)
    logger.error(f"Ontology generation failed: {error_msg}")
```

**Why:**
- Print statements appear immediately in Docker logs (no buffering)
- Helps debug when logging level is not set correctly
- Both logger and print ensure visibility

---

### 7. `backend/app/utils/logger.py` — Logging Configuration

**What changed**: Changed console logging level from INFO to DEBUG.

```python
# Before
console_handler.setLevel(logging.INFO)

# After
console_handler.setLevel(logging.DEBUG)
```

**Why:** During development/debugging, seeing DEBUG logs in the console is essential to understand what's happening. File logs still capture everything.

---

### 8. `README.md` — Documentation

**What changed**: Updated all user-facing documentation and examples.

#### Key sections updated:
1. **Features table**: "Ollama" → "LM Studio"
2. **Prerequisites**: Added "LM Studio installed and running with models loaded"
3. **Quick Start - Step 0**: New section explaining how to start LM Studio
4. **Option A (Docker)**: Added Docker-specific instructions about `host.docker.internal`
5. **Option B (Manual)**: Simplified (no more Ollama pull commands)
6. **Configuration section**: Updated defaults to LM Studio
7. **Architecture diagram**: Updated comments to reference LM Studio
8. **Credits**: Updated version numbers (Neo4j 5.15 → 5.18)

---

## Key Technical Differences

### Port Numbers
| Service | Ollama | LM Studio |
|---------|--------|-----------|
| Chat API | 11434 | 1234 |
| Endpoint path | `/v1/chat/completions` | `/v1/chat/completions` |
| Embeddings | `/api/embed` | `/v1/embeddings` |

### API Formats
| Component | Ollama | LM Studio |
|-----------|--------|-----------|
| Chat | OpenAI-compatible ✓ | OpenAI-compatible ✓ |
| Embeddings | Ollama proprietary `/api/embed` | OpenAI-compatible `/v1/embeddings` |
| Response Format | Supports `json_schema` | Does NOT support `json_schema` |
| Context Window | Via `extra_body: {options: {num_ctx}}` | Not applicable |

### Docker Networking
| Scenario | Ollama (Docker service) | LM Studio (Host app) |
|----------|------------------------|----------------------|
| From container | Service name `ollama` | `host.docker.internal` |
| From host | `localhost:11434` | `localhost:1234` |

---

## Migration Checklist

If migrating an existing Ollama setup:

- [ ] Stop Ollama: `docker compose down` or `killall ollama`
- [ ] Download/install LM Studio from https://lmstudio.ai/
- [ ] Open LM Studio and download models:
  - Chat model (e.g., `qwen3.5-35b-a3b`, `llama2`, `mistral`)
  - Embedding model: `nomic-embed-text`
- [ ] Go to **Local Server** tab and start server on port 1234
- [ ] Update `.env`:
  - `LLM_BASE_URL=http://127.0.0.1:1234/v1` (or `host.docker.internal` for Docker)
  - `LLM_MODEL_NAME=<exact model name from LM Studio>`
  - `EMBEDDING_BASE_URL=http://127.0.0.1:1234`
  - `EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5` (exact name from LM Studio)
- [ ] Update `docker-compose.yml` if using Docker
- [ ] Rebuild Docker image: `docker compose build --no-cache`
- [ ] Start services: `docker compose up -d`
- [ ] Test: Upload a document and check logs for errors

---

## Troubleshooting

### "Connection refused" errors
- Check LM Studio is running: `curl http://localhost:1234/v1/models`
- Check port 1234 is not blocked by firewall
- Check correct IP/hostname in `.env`

### "Model not found" error
- Check exact model name in LM Studio's **Local Server** page
- Copy-paste the exact name into `LLM_MODEL_NAME` in `.env`
- Rebuild Docker image after changing `.env`

### "Invalid JSON" errors
- Check `EMBEDDING_MODEL` matches exact name from LM Studio
- Ensure embedding model is actually loaded (shows in Local Server page)
- Check LLM model supports JSON output (most modern models do)

### Docker container can't reach host
- On **Linux**: Add `extra_hosts` to `docker-compose.yml` (already done)
- On **Mac/Windows**: `host.docker.internal` should work automatically
- Test from container: `docker exec <container-name> curl http://host.docker.internal:1234/v1/models`

---

## Summary

The migration maintains 100% API compatibility while switching the backend:

| Layer | Ollama | LM Studio | Impact |
|-------|--------|-----------|--------|
| Chat API | OpenAI `/v1/chat/completions` | OpenAI `/v1/chat/completions` | None — same interface |
| Embeddings | Ollama `/api/embed` | OpenAI `/v1/embeddings` | Response format changed |
| Response Format | Supports `json_schema` | Doesn't support it | Use prompt guidance instead |
| Logging | Basic | Comprehensive | Better debugging |
| Docker | Container service | Host application | Updated networking |

All changes are **backwards compatible** — the code can still use Ollama if configured to do so.
