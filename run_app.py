from app.ingest import ingest_pdfs
from app.insights_store import InsightsStore
import streamlit as st
import os
import subprocess
import sys
from typing import Optional
from datetime import datetime
from app.metadata_db import (
    get_connection,
    list_papers,
    list_collections,
    create_collection,
    add_papers_to_collection,
    get_filenames_for_collection,
    upsert_paper,
    replace_chunks,
    upsert_paper_embedding,
)  # noqa: E501
from importlib import import_module
from sentence_transformers import SentenceTransformer
from app.task_manager import submit as submit_task, status as task_status, exception as task_exception

# -----------------------------------------------------------------------------
# Helper: generate short title using the loaded LLM (defined early so it exists
# when widgets are evaluated later).
# -----------------------------------------------------------------------------


def generate_short_title(text: str, rag):  # noqa: ANN001
    """Return a 2–4 word title summarizing *text* using *rag*'s LLM.

    Falls back to the first 50 characters if generation fails or *rag* is None.
    """

    if rag is None:
        return text[:50]

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Respond with ONLY the 2-4 word title, no extra text."},
        {"role": "user", "content": text},
    ]

    try:
        prompt = rag.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        out = rag.generator(
            prompt,
            max_new_tokens=12,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.05,
            return_full_text=False,
        )[0]["generated_text"]
        cleaned = out.strip().split("\n")[0]
        # Remove common preambles
        for prefix in ("Here is", "Sure", "Certainly", "A concise", "The title is", "Here\u2019s", "Here is a concise 2-4 word title is", "Here is a concise 2-4 word title"):
            if cleaned.lower().startswith(prefix.lower()):
                # try after colon or first ':' if present
                if ":" in cleaned:
                    cleaned = cleaned.split(":", 1)[1].strip()
                else:
                    cleaned = cleaned[len(prefix):].strip()
        return cleaned[:100]
    except Exception:
        return text[:50]

st.set_page_config(page_title="Research RAG Assistant", page_icon="", layout="wide")

# -----------------------------------------------------------------------------
# Custom CSS + keyframes for pin alignment & flash animation
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Shrink default button inside the right-hand pin column */
    .pin-col button[data-baseweb="button"] {
        padding: 2px 6px;
        border-radius: 6px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    /* One-shot flash animation after pin */
    @keyframes highlightFade {
        0%   {background-color: rgba(255, 230, 0, 0.25);} /* yellowish */
        100% {background-color: transparent;}
    }
    .flash-once {
        animation: highlightFade 800ms ease-out 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag" not in st.session_state:
    st.session_state.rag = None
if "db_loaded" not in st.session_state:
    st.session_state.db_loaded = False
if "loaded_model_id" not in st.session_state:
    st.session_state.loaded_model_id = None
if "insights_store" not in st.session_state:
    st.session_state.insights_store = InsightsStore()

with st.sidebar:
    st.header("Data Ingestion")
    pdf_dir = st.text_input("PDF Directory", value="data/papers")

    # Track currently running ingestion task
    ingest_state = st.session_state.get("ingest_task_id")

    if ingest_state:
        stat = task_status(ingest_state)
        if stat in ("pending", "running"):
            st.markdown("⏳ Ingestion running in background…")
        elif stat == "done":
            st.success("Ingestion complete!")
            # Refresh vector store on next chat
            st.session_state.db_loaded = False
            del st.session_state.ingest_task_id
        elif stat == "error":
            st.error(f"Ingestion failed: {task_exception(ingest_state)}")
            del st.session_state.ingest_task_id

    if st.button("Ingest PDFs") and not st.session_state.get("ingest_task_id"):
        # Schedule ingestion
        task_id = submit_task(ingest_pdfs, pdf_dir=pdf_dir)
        st.session_state.ingest_task_id = task_id
        st.markdown("⏳ Ingestion started; you can continue using the app.")

    # ---------------------------------------------------------------------
    # Button to open the target directory in the OS file explorer
    # ---------------------------------------------------------------------
    if st.button("📂 Open Folder"):
        try:
            # Ensure the directory exists
            os.makedirs(pdf_dir, exist_ok=True)

            # Cross-platform folder opening
            if sys.platform.startswith("win"):
                os.startfile(os.path.abspath(pdf_dir))  # type: ignore[attr-defined]
            elif sys.platform.startswith("darwin"):
                subprocess.Popen(["open", pdf_dir])
            else:
                subprocess.Popen(["xdg-open", pdf_dir])
        except Exception as e:
            st.error(f"Could not open folder: {e}")

    # ---------------------------------------------------------------------
    # Model selection
    # ---------------------------------------------------------------------
    st.header("Model Selection")
    st.text("Specify any public or gated HuggingFace model id accessible with your token.")
    st.text_input(
        "HF Model ID",
        value="meta-llama/Llama-3.2-1B-Instruct",
        key="model_id",
    )

    # ---------------------------------------------------------------------
    # Insights management (search & list)
    # ---------------------------------------------------------------------
    st.header("📌 Insights")
    if "insights_store" in st.session_state:
        insight_query = st.text_input("Search insights", key="insight_search")
        if insight_query:
            matches = st.session_state.insights_store.search(insight_query, k=5)
        else:
            # Show recent insights if no query
            matches = [(d, 0.0) for d in st.session_state.insights_store.list_all()[::-1][:5]]

        for meta, dist in matches:
            display_label = meta.get("title", meta["text"])
            st.markdown(f"- {display_label}")
            # Delete button inline
            if st.button("🗑️ Delete", key=f"del_{meta['id']}"):
                try:
                    st.session_state.insights_store.delete_insight(meta["id"])
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

st.title("📚 Research Chat with HuggingFace Llama")

# -----------------------------------------------------------------------------
# Lazy-load / reload the RAG backend when the vector DB is ready *and* the
# requested model identifier changes.
# -----------------------------------------------------------------------------
model_id = st.session_state.get("model_id", "meta-llama/Llama-3.2-1B-Instruct")

if (
    st.session_state.rag is None
    or not st.session_state.db_loaded
    or st.session_state.loaded_model_id != model_id
):
    try:
        # -----------------------------------------------------------------------------
        # Safe import of RAGChat – exit if the module itself has errors
        # -----------------------------------------------------------------------------
        from app.rag_chat import RAGChat
        st.session_state.rag = RAGChat(model_id=model_id)
        st.session_state.db_loaded = True
        st.session_state.loaded_model_id = model_id

        # ------------------------------------------------------------------
        # Apply context pool restriction after RAGChat instantiation
        # ------------------------------------------------------------------
        ctx_pool_name = st.session_state.get("ctx_pool", "All papers")
        if ctx_pool_name != "All papers":
            # Find collection
            conn_ctx = get_connection()
            all_cols = list_collections(conn_ctx)
            sel_col = next((c for c in all_cols if c["name"] == ctx_pool_name), None)
            if sel_col is not None:
                allowed = get_filenames_for_collection(conn_ctx, sel_col["id"])
                st.session_state.rag.set_allowed_sources(allowed)
            conn_ctx.close()
        else:
            st.session_state.rag.set_allowed_sources(None)
    except FileNotFoundError as e:
        st.info(str(e))
    except Exception as import_err:
        import traceback, sys

        # Print the full traceback to the terminal for debugging.
        traceback.print_exc()

        # Show a human-readable error in the Streamlit UI (if it manages to init).
        st.error(f"Fatal error while importing RAGChat: {import_err}")

        # Immediately terminate the Streamlit script runner so the process exits
        # instead of hanging in a broken state.
        sys.exit(1)

# -----------------------------------------------------------------------------
# Render existing chat and provide floating input at the bottom
# -----------------------------------------------------------------------------
for idx, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        # Determine if this message should flash (immediately after pin)
        flash_class: str = "flash-once" if st.session_state.get("flash_idx") == idx else ""

        # --- Layout & content --------------------------------------------------
        col_msg, col_pin = st.columns([0.95, 0.05])
        with col_msg:
            st.markdown(
                f'<div class="{flash_class}">{message["content"]}</div>',
                unsafe_allow_html=True,
            )

            # Render citation expanders if present (for reruns)
            if "citations" in message and "contexts" in message:
                for cidx, meta in enumerate(message["citations"]):
                    label_parts = []
                    if meta.get('authors'):
                        label_parts.append(meta['authors'])
                    if meta.get('title'):
                        label_parts.append(f"“{meta['title']}”")
                    if meta.get('year'):
                        label_parts.append(str(meta['year']))
                    if not label_parts:
                        label_parts.append(meta.get('source',''))
                    label = ' '.join(label_parts)
                    with st.expander(f"[{cidx+1}] {label}"):
                        st.markdown(message["contexts"][cidx])

        # Only allow pinning of assistant messages (the ones with context)
        if message["role"] == "assistant":
            with col_pin:
                col_pin.markdown("", unsafe_allow_html=True)  # ensure container exists

                store = st.session_state.get("insights_store")
                if store is not None:
                    pin_label = "📌"

                    # Determine pinned status --------------------------------------------------
                    insight_id = message.get("insight_id")
                    is_pinned = False
                    if insight_id:
                        # Verify still exists in store
                        try:
                            # list_all is small; fine to scan
                            is_pinned = any(d["id"] == insight_id for d in store.list_all())
                        except Exception:
                            is_pinned = False

                    button_type = "primary" if is_pinned else "secondary"

                    with st.container():
                        st.markdown("<div class='pin-col'>", unsafe_allow_html=True)
                        if st.button(pin_label, key=f"pin_{idx}", type=button_type):
                            contexts = message.get("contexts", [])
                            try:
                                if is_pinned:
                                    # Unpin
                                    store.delete_insight(insight_id)  # type: ignore[arg-type]
                                    message.pop("insight_id", None)
                                    st.toast("Insight removed", icon="🗑️")
                                else:
                                    # Pin with generated title
                                    title = generate_short_title(message["content"], st.session_state.rag)
                                    try:
                                        new_id = store.add_insight(message["content"], contexts, title=title)
                                    except TypeError as te:
                                        # Handle older store instance that lacks the *title* param
                                        if "unexpected keyword argument 'title'" in str(te):
                                            store = InsightsStore()  # reload fresh implementation
                                            st.session_state.insights_store = store
                                            new_id = store.add_insight(message["content"], contexts, title=title)
                                        else:
                                            raise
                                    message["insight_id"] = new_id
                                    st.toast("Insight pinned!", icon="📌")
                                    st.session_state["flash_idx"] = idx
                                # Persist change to session_state
                                st.session_state.chat_history[idx] = message
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))
                        st.markdown("</div>", unsafe_allow_html=True)

# Clear the flash flag so it only triggers once
if "flash_idx" in st.session_state:
    del st.session_state["flash_idx"]

# Floating input; hitting Enter submits automatically.
user_query = st.chat_input("Enter your question")

if user_query:
    if st.session_state.rag is None:
        st.warning("Vector store not available. Please ingest PDFs first.")
    else:
        # Show the user message immediately
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("Generating response…"):
            # Update allowed_sources based on any changes to context pool mid-session
            ctx_pool_name = st.session_state.get("ctx_pool", "All papers")
            if ctx_pool_name != "All papers":
                conn_ctx2 = get_connection()
                sel_col2 = next((c for c in list_collections(conn_ctx2) if c["name"] == ctx_pool_name), None)
                if sel_col2 is not None:
                    allowed2 = get_filenames_for_collection(conn_ctx2, sel_col2["id"])
                    st.session_state.rag.set_allowed_sources(allowed2)
                conn_ctx2.close()
            else:
                st.session_state.rag.set_allowed_sources(None)

            assistant_reply = st.session_state.rag.chat(user_query, st.session_state.chat_history)
            # Retrieve contexts from the RAG object if available
            contexts = getattr(st.session_state.rag, "last_contexts", [])
            citation_meta = getattr(st.session_state.rag, "last_citation_meta", [])

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": assistant_reply,
                "contexts": contexts,
                "citations": citation_meta,
            })

        with st.chat_message("assistant"):
            st.markdown(assistant_reply)
            # Render citation viewers
            for idx_c, meta in enumerate(citation_meta):
                label_parts = []
                if meta.get('authors'):
                    label_parts.append(meta['authors'])
                if meta.get('title'):
                    label_parts.append(f"“{meta['title']}”")
                if meta.get('year'):
                    label_parts.append(str(meta['year']))
                if not label_parts:
                    label_parts.append(meta.get('source',''))
                label = ' '.join(label_parts)
                with st.expander(f"[{idx_c+1}] {label}"):
                    st.markdown(contexts[idx_c])

# -----------------------------------------------------------------------------
# 📝 Scratch Notepad Tab (moved into dedicated tab)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------
# Layout tabs: Chat (default), Notepad, and Library
# -----------------------------------------------------------------
chat_tab, note_tab, library_tab = st.tabs(["💬 Chat", "📝 Notepad", "📂 Library"])

# Ensure scratch pad state exists early so widgets preserve content across reruns
if "scratch_pad_html" not in st.session_state:
    st.session_state.scratch_pad_html = ""

with note_tab:
    st.subheader("Scratch Pad (Rich Text)")

    # Dynamically import quill component if available
    HAVE_QUILL = False
    try:
        st_quill_mod = import_module("streamlit_quill")
        st_quill = getattr(st_quill_mod, "st_quill")
        HAVE_QUILL = True
    except Exception:
        HAVE_QUILL = False

    if HAVE_QUILL:
        scratch_html = st_quill(html=True, value=st.session_state.scratch_pad_html, key="scratch_editor")
        if scratch_html is not None:
            st.session_state.scratch_pad_html = scratch_html
    else:
        st.warning("streamlit-quill not installed; falling back to plain textarea.")
        scratch_plain = st.text_area("Notes (plain text)", key="scratch_pad_fallback", height=300)
        st.session_state.scratch_pad_html = f"<p>{scratch_plain}</p>"  # naive wrap

    if st.button("Export to Markdown", key="export_md"):
        try:
            notes_dir = "notes"
            os.makedirs(notes_dir, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"note_{timestamp}.md"

            # Convert HTML to Markdown
            try:
                from markdownify import markdownify as mdify
                md_content = mdify(st.session_state.scratch_pad_html or "")
            except Exception:
                md_content = st.session_state.scratch_pad_html  # fallback html

            # Write to disk
            with open(os.path.join(notes_dir, filename), "w", encoding="utf-8") as f:
                f.write(md_content)

            # ----------------------------------------------
            # Persist note as a searchable paper-like entity
            # ----------------------------------------------
            try:
                conn_note = get_connection()

                # Derive title: first non-empty line or filename
                first_line = next((line.strip('# ').strip() for line in md_content.splitlines() if line.strip()), filename)
                paper_id = upsert_paper(
                    conn_note,
                    filename=filename,
                    title=first_line,
                )

                # Store full note as single chunk for potential retrieval
                replace_chunks(conn_note, paper_id, [md_content])

                # Compute and persist embedding
                cleaned_note = _clean_text_for_embedding(md_content)

                # Offload embedding to background so UI returns quickly
                def _embed_and_store():
                    from sentence_transformers import SentenceTransformer as _ST
                    vec = _ST("all-MiniLM-L6-v2").encode(cleaned_note)
                    conn_bg = get_connection()
                    upsert_paper_embedding(conn_bg, paper_id, vec)
                    conn_bg.close()

                submit_task(_embed_and_store)
            except Exception as embed_err:
                st.warning(f"Note saved but async embedding scheduling failed: {embed_err}")

            st.success(f"Notes exported to {os.path.join(notes_dir, filename)}")
        except Exception as e:
            st.error(str(e))

# -----------------------------------------------------------------
# 📂 Library Tab: View papers and manage collections
# -----------------------------------------------------------------

with library_tab:
    st.subheader("Paper Library")

    try:
        conn = get_connection()

        # ------------------------------------------------------------------
        # Fetch papers & render selection checkboxes
        # ------------------------------------------------------------------
        papers = list_papers(conn)
        if not papers:
            st.info("No papers ingested yet. Use the sidebar to ingest PDFs.")
        else:
            st.markdown("**Select papers:**")
            selected_ids: list[int] = []
            for row in papers:
                label_parts = []
                if row.get("title"):
                    label_parts.append(f"{row['title']}")
                label_parts.append(f"({row['filename']})")
                checkbox_key = f"paper_sel_{row['id']}"
                checked = st.checkbox(" ".join(label_parts), key=checkbox_key)
                if checked:
                    selected_ids.append(row["id"])

            st.divider()

            # ------------------------------------------------------------------
            # Create new collection
            # ------------------------------------------------------------------
            st.markdown("### Create New Collection")
            new_coll_name = st.text_input("Collection name", key="new_collection_name")
            if st.button("Create collection", key="btn_create_collection"):
                try:
                    if not new_coll_name.strip():
                        st.error("Please provide a collection name.")
                    else:
                        cid = create_collection(conn, new_coll_name.strip())
                        st.success(f"Collection '{new_coll_name}' created (id {cid}).")
                        st.session_state.new_collection_name = ""  # clear
                        st.rerun()
                except Exception as e:
                    st.error(str(e))

            st.divider()

            # ------------------------------------------------------------------
            # Add selected papers to an existing collection
            # ------------------------------------------------------------------
            st.markdown("### Add Papers to Collection")
            collections = list_collections(conn)
            if not collections:
                st.info("No collections yet – create one first.")
            else:
                coll_options = {f"{c['name']} (id {c['id']})": c["id"] for c in collections}
                selected_coll_label = st.selectbox(
                    "Choose collection",
                    list(coll_options.keys()),
                    key="sel_collection",
                )
                sel_coll_id = coll_options[selected_coll_label]

                if st.button("Add to collection", key="btn_add_to_collection"):
                    try:
                        if not selected_ids:
                            st.error("No papers selected.")
                        else:
                            add_papers_to_collection(conn, sel_coll_id, selected_ids)
                            st.success("Papers added to collection.")
                            # Optionally clear selections
                            for pid in selected_ids:
                                st.session_state[f"paper_sel_{pid}"] = False
                    except Exception as e:
                        st.error(str(e))

        conn.close()
    except Exception as e:
        st.error(str(e))

# -----------------------------------------------------------------
# 📂 Sidebar: show collections list
# -----------------------------------------------------------------

with st.sidebar:
    st.header("📂 Collections")
    try:
        conn_sidebar = get_connection()
        sidebar_colls = list_collections(conn_sidebar)
        conn_sidebar.close()

        if sidebar_colls:
            for c in sidebar_colls:
                st.markdown(f"- {c['name']}")
        else:
            st.caption("No collections yet.")

        # ------------------------------------------------------------------
        # Context pool selection (All vs specific collection)
        # ------------------------------------------------------------------
        st.subheader("🎯 Context Pool")
        context_options = ["All papers"] + [c["name"] for c in sidebar_colls]
        ctx_current = st.session_state.get("ctx_pool", "All papers")
        # Determine index safely
        try:
            current_idx = context_options.index(ctx_current)
        except ValueError:
            current_idx = 0
        selected_pool = st.radio(
            "Active context", context_options, index=current_idx, key="ctx_pool_radio"
        )
        st.session_state.ctx_pool = selected_pool

        # Apply selection immediately to existing RAG object (if any)
        rag_obj = st.session_state.get("rag")
        if rag_obj is not None:
            # If running instance lacks the new API, force reload.
            if not hasattr(rag_obj, "set_allowed_sources"):
                st.session_state.rag = None  # trigger re-instantiation on rerun
                st.rerun()
            else:
                if selected_pool != "All papers":
                    conn_apply = get_connection()
                    sel_col_apply = next((c for c in sidebar_colls if c["name"] == selected_pool), None)
                    if sel_col_apply is not None:
                        allowed_apply = get_filenames_for_collection(conn_apply, sel_col_apply["id"])
                        rag_obj.set_allowed_sources(allowed_apply)
                    conn_apply.close()
                else:
                    rag_obj.set_allowed_sources(None)
    except Exception as sidebar_err:
        st.error(str(sidebar_err)) 