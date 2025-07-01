# Notebook — Production Desktop Edition

> Cross-platform, offline-first knowledge workspace for reading, chatting with, and organising research papers.

---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Getting Started](#getting-started)
4. [Folder Layout](#folder-layout)
5. [Security Notes](#security-notes)
6. [Roadmap](#roadmap)

---

## Features

• Chat with a collection of PDFs using RAG (retrieval-augmented generation).  
• Inline citations with one-click **PDF preview** (eye-icon) and full-document viewer.  
• Notebook editor (rich-text, markdown).  
• Local vector store & metadata DB — works fully offline.  
• Ships as a tiny Tauri desktop app (Rust core ≈ 3 MB).

---

## Architecture

```
┌──────────────────────┐        ┌─────────────────┐
│   React 18 + Vite    │  HTTP  │   FastAPI (Python) │
│  UI (ui/ folder)     │ ─────▶ │  app/ services   │
└─────────▲────────────┘        └────────┬────────┘
          │  Filesystem (Tauri FS)         │
          │                                │
          │            FAISS + SQLite      │
          └───────────────────────────────▶│
```

* **ui/** — Vite/React front-end wrapped by Tauri.  
* **app/** — Python package with ingest, chat, and storage services.  
* **rag_server/** — FastAPI application (entry-point `rag_server.main:app`) & background workers.  
* **src-tauri/** — Rust glue that bundles the UI into a desktop binary.

---

## Getting Started

### 1 · System requirements

* **Node ≥ 18** (for front-end build)
* **Python 3.11**
* **Rust toolchain** + `cargo` (for Tauri shell)  
  `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
* `npm` (or `pnpm`/`yarn`) & `pip`

### 2 · Clone & set up

```bash
# clone
$ git clone https://github.com/…/notebook.git
$ cd notebook

# Python env
$ python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
$ pip install -r requirements.txt

# Front-end deps
$ cd ui && npm install && cd ..
```

### 3 · Development mode (hot-reload)

```bash
# terminal 1 – API & background jobs
$ uvicorn rag_server.main:app --reload --port 8000

# terminal 2 – React dev server
$ cd ui && npm run dev

# optional — run the desktop shell (needs @tauri-app/cli)
$ cargo tauri dev  # or: npm run tauri dev
```

Navigate to http://localhost:5173 (default Vite port).

### 4 · Production build

```bash
# Build front-end assets
$ cd ui && npm run build && cd ..

# Package the desktop app (macOS / Windows / Linux)
$ cargo tauri build
```

---

## Folder Layout

```
.
├─ app/            # Python core (ingest, chat, DB)
├─ ui/             # React front-end
│   └─ src/components/PdfModal.tsx
├─ src-tauri/      # Rust + Tauri configuration
├─ data/           # Local user data & ingested PDFs
├─ roadmap.md      # Long-term plan
└─ README.md       # (this file)
```

---

## Security Notes

Current viewer stack: **react-pdf 7.7.3 + pdfjs-dist 4.2.67**.  
This removes the CVE-2024-4367 exploit by disabling `eval` inside the
worker, but `npm audit` may still flag the nested pdfjs-dist 3.x copy.

> We plan to migrate to **react-pdf 9.x** (ESM-only & pdfjs-dist 4.3+) in
a future sprint once the build pipeline is moved to native ESM.  See
`ROADMAP.md` for the tracked task.

The Quill XSS advisory is mitigated by DOMPurify; we'll revisit when
upgrading to react-quill 2.x.

+ **Debug-only Clear-DB endpoint** — For local development there is a POST `/debug/clear_db` API (exposed via the "Clear Database" button in *Library View*).  It **deletes** the `db/metadata.db` file and recreates it empty on next access.

This is strictly a convenience tool for debugging desktop/offline builds.  **Never enable it on deployments where the front-end speaks to a shared or remote server.**  In the long run we should either remove the endpoint entirely or scope it so it only affects files on the user's own machine.

---

This project exists to build useful tools that maintain ownership of your data. AI is a powerful productivity multiplier, and we should aim to arm users with the right to modify, repair, and understand the systems they use. 

In an age of Everything-as-a-Service and proprietary black boxes, it's more important than ever to keep the software commons alive. This project is licensed under the **AGPLv3** to ensure that anyone who improves or deploys this code contributes back to the community.

### How You Can Help

If you find this project useful:

* **Share it** with others who might benefit.
* **Use it** in your work or projects—it's here for that.
* **Credit it** when appropriate, especially in research, documentation, or public tools.

R. Belaire
