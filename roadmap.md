Here’s a compact “living document” you can drop into the repo as `ROADMAP.md`.  
Everything is grouped by stage so you can track what is already done and what still needs research or design decisions.

---

# Notebook-Next Roadmap

## 0 — Overall Vision
A cross-platform knowledge workspace that lets a researcher  
• ingest & chat with collections of papers  
• attribute each idea to its cited source  
• iterate on thoughts until they crystallise into slide decks / text docs  
• run **offline-first** as a desktop app

---

## 2 — Researcher Alpha

Todo:
-fix model selection: route hf_cache to local dir
-improve chat
--drop-in contexts
--latex support
--add stop-generate button
--pdf display for sources

-notepad features
--latex
--exports

-add google scholar search
-improve chat context window: use truncated/compressed embeddings and pinned contexts.
-scratchpad micro generations: one-off generation requests output directly into scratchpad, can attach contexts

---

## 3 — Public Beta (Cloud functions)

| New Capability | Description | Infra Upgrades |
|----------------|-------------|----------------|
| Cloud sync & sharing | Push notebooks, invite users | • S3 / R2 bucket<br>• Postgres row-level ACL |
| Citation manager | Zotero integration, auto-format refs | • OAuth flow<br>• CSL processing lib |
| Scheduled jobs | Nightly re-embedding, re-ranking, summarisation | • Celery beat + Flower dashboard |
| Observability | Tracing, metrics, error logs | • Prometheus + Grafana |

---

## Tech Stack Summary

• Core services: Python 3.11, FastAPI, sqlite3  
• Vector DB: FAISS 
• Storage: local FS
• Front-end:  React 18 + Vite + Tauri Shell (Rust)    
• LLM/RAG: huggingface, sentence-transformers, enterprise APIs  

---

## Next Steps

2. Groom Alpha stories: create GitHub issues linked to **Section 2** table.  
4. Plan upgrade of the PDF viewer stack – migrate `react-pdf` **7.7.3 → 9.x** (moves to ESM-only build and `pdfjs-dist` 4.3+).

## Interesting research questions

1. Can we obfuscate local data before sending requests to OpenAI, and still retain high quality of responses?
