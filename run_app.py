from app.ingest import ingest_pdfs
import streamlit as st
import os
import subprocess
import sys

st.set_page_config(page_title="Research RAG Assistant", page_icon="📚", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag" not in st.session_state:
    st.session_state.rag = None
if "db_loaded" not in st.session_state:
    st.session_state.db_loaded = False
if "loaded_model_id" not in st.session_state:
    st.session_state.loaded_model_id = None

with st.sidebar:
    st.header("Data Ingestion")
    pdf_dir = st.text_input("PDF Directory", value="data/papers")
    if st.button("Ingest PDFs"):
        try:
            with st.spinner("Ingesting PDFs…"):
                ingest_pdfs(pdf_dir=pdf_dir)
                st.success("Ingestion complete!")
                st.session_state.db_loaded = False
        except Exception as e:
            st.error(str(e))

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
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
            assistant_reply = st.session_state.rag.chat(user_query, st.session_state.chat_history)

        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
        with st.chat_message("assistant"):
            st.markdown(assistant_reply) 