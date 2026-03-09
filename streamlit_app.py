import streamlit as st
import requests
from typing import Optional, Dict, List
import uuid
from datetime import datetime
import json
import sseclient  # pip install sseclient-py

# =============================================================================
# CONFIGURATION
# =============================================================================
try:
    API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")
except Exception:
    API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Multi-Agent Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS — smooth fade-in for messages
st.markdown("""
<style>
[data-testid="stChatMessage"] {
    animation: fadeIn 0.3s ease-in;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

[data-testid="stChatInput"] textarea {
    border-radius: 12px !important;
    font-size: 15px !important;
}
.stButton > button { border-radius: 8px; }
.stAlert            { border-radius: 8px; }

/* Suggestion chip styling */
div[data-testid="stHorizontalBlock"] .stButton > button {
    background-color: #f0f2f6;
    color: #262730;
    border: 1px solid #d0d3db;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 13px;
    transition: background-color 0.2s ease;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background-color: #e0e3ea;
    border-color: #a0a3ab;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================
def initialize_session_state():
    defaults = {
        'bot_id':             None,
        'session_id':         str(uuid.uuid4()),
        'chat_history':       [],
        'bot_metadata':       None,
        'suggestions':        [],      # holds the 3 suggested questions
        'clicked_suggestion': None,    # holds a suggestion the user clicked
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# API HELPERS
# =============================================================================
def upload_and_create_bot(team_name: str, bot_name: str, files: List) -> Optional[Dict]:
    """Upload files and create a new bot via API."""
    try:
        form_data  = {'team_name': team_name, 'bot_name': bot_name}
        files_data = [('files', (f.name, f.getvalue(), f.type)) for f in files if f]
        response   = requests.post(
            f"{API_BASE_URL}/bot/upload",
            data=form_data, files=files_data, timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"Error creating bot: {response.json().get('detail', str(e))}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the API server.")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
    return None


def stream_bot_response(bot_id: str, question: str, session_id: str):
    """
    Generator — yields either:
      - str  : a text chunk to append to the response
      - dict : {"suggestions": [...]} when suggested questions arrive
      - dict : {"error": "message"}   when something goes wrong

    IMPORTANT: No st.* calls inside this function.
    All Streamlit UI calls must happen in the caller (render_chat_interface).
    """
    payload = {"bot_id": bot_id, "question": question, "session_id": session_id}

    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/stream",
            json=payload, stream=True, timeout=(5, 120),
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get('detail', str(e)) if e.response else str(e)
        yield {"error": f"❌ Server error: {detail}"}
        return
    except requests.exceptions.ConnectionError:
        yield {"error": "❌ Could not connect to the API server."}
        return
    except Exception as e:
        yield {"error": f"❌ Unexpected error: {e}"}
        return

    client = sseclient.SSEClient(response)

    for event in client.events():
        raw = event.data
        if not raw:
            continue

        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue

        event_type = parsed.get("type", "")

        if event_type == "done":
            break
        elif event_type == "error":
            yield {"error": f"Bot error: {parsed.get('message', 'Unknown error')}"}
            break
        elif event_type == "suggestions":
            # Yield the suggestions list as a dict so the caller can store it
            questions = parsed.get("questions", [])
            if questions:
                yield {"suggestions": questions}
        elif event_type == "token":
            value = parsed.get("value", "")
            if value:
                yield value


def check_health() -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar():
    with st.sidebar:
        st.header("📤 Create New Bot")
        team_name      = st.text_input("Team Name", placeholder="e.g., Engineering Team", key="team_name")
        bot_name       = st.text_input("Bot Name",  placeholder="e.g., HR Assistant",     key="bot_name")
        uploaded_files = st.file_uploader(
            "Upload Documents", type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True, key="file_uploader",
        )

        if st.button("🚀 Create Bot", type="primary", use_container_width=True):
            if not team_name.strip() or not bot_name.strip():
                st.error("Please provide both team name and bot name.")
            elif not uploaded_files:
                st.error("Please upload at least one file.")
            else:
                with st.spinner("Creating bot… This may take a moment."):
                    meta = upload_and_create_bot(team_name, bot_name, uploaded_files)
                    if meta:
                        st.session_state.bot_id       = meta['bot_id']
                        st.session_state.bot_metadata = meta
                        st.session_state.chat_history = []
                        st.session_state.suggestions  = []
                        st.success(f"✅ Bot '{bot_name}' created successfully!")
                        st.rerun()

        st.markdown("---")
        st.subheader("📊 Current Bot")

        if st.session_state.bot_metadata:
            m = st.session_state.bot_metadata
            st.info(
                f"**Team:** {m['team_name']}  \n"
                f"**Bot:** {m['bot_name']}  \n"
                f"**Files:** {len(m['file_names'])} document(s)  \n"
                f"**Chunks:** {m['chunk_count']}"
            )
            with st.expander("📁 Uploaded Files"):
                for fn in m['file_names']:
                    st.text(f"• {fn}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 New Session", use_container_width=True):
                    st.session_state.session_id   = str(uuid.uuid4())
                    st.session_state.chat_history = []
                    st.session_state.suggestions  = []
                    st.success("New session started!")
                    st.rerun()
            with col2:
                if st.button("🗑️ Reset Bot", use_container_width=True):
                    st.session_state.bot_id       = None
                    st.session_state.bot_metadata = None
                    st.session_state.chat_history = []
                    st.session_state.suggestions  = []
                    st.success("Bot reset!")
                    st.rerun()
        else:
            st.warning("No bot created yet. Upload files to get started!")

        st.markdown("---")
        st.caption(f"Session: `{st.session_state.session_id[:8]}…`")
        st.markdown("---")
        st.success("🟢 API: Online") if check_health() else st.error("🔴 API: Offline")


# =============================================================================
# CHAT INTERFACE
# =============================================================================
def render_chat_interface():
    if not st.session_state.bot_id:
        st.info("👈 Please create a bot using the sidebar to start chatting!")
        st.markdown("""
### How to Use
1. **Create a Bot** — Provide team/bot names, upload documents, click "Create Bot".
2. **Start Chatting** — Ask anything about your documents.
3. **New Session** — Resets conversation memory without deleting the bot.
4. **Reset Bot** — Clears everything and starts fresh.
        """)
        return

    st.subheader("💬 Chat with Your Bot")

    # ── Render previous messages ──────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(msg['question'])
        with st.chat_message("assistant"):
            st.markdown(msg['answer'])

    # ── Render suggestion chips from previous response ────────────────────────
    # These sit just above the chat input so they feel like quick-reply buttons
    if st.session_state.suggestions:
        st.markdown("**💡 Suggested Questions:**")
        cols = st.columns(len(st.session_state.suggestions))
        for i, (col, question) in enumerate(zip(cols, st.session_state.suggestions)):
            with col:
                # Use a unique key combining index + truncated last question
                # to avoid key collisions across reruns
                last_q = st.session_state.chat_history[-1]['question'][:15] if st.session_state.chat_history else "init"
                if st.button(question, key=f"sugg_{i}_{last_q}", use_container_width=True):
                    st.session_state.clicked_suggestion = question
                    st.session_state.suggestions        = []
                    st.rerun()

    # ── Pick up a clicked suggestion or wait for manual input ─────────────────
    prefill_question = st.session_state.pop("clicked_suggestion", None)
    user_question    = st.chat_input("Ask a question about your documents…")

    # Clicked suggestion takes priority over typed input
    user_question = prefill_question or user_question
    if not user_question:
        return

    # Clear suggestions whenever a new question is submitted
    st.session_state.suggestions = []

    # Show user bubble immediately
    with st.chat_message("user"):
        st.markdown(user_question)

    # ── Stream assistant response ─────────────────────────────────────────────
    with st.chat_message("assistant"):
        placeholder   = st.empty()
        full_response = ""
        error_message = None

        for chunk in stream_bot_response(
            bot_id=st.session_state.bot_id,
            question=user_question,
            session_id=st.session_state.session_id,
        ):
            # Error sentinel
            if isinstance(chunk, dict) and "error" in chunk:
                error_message = chunk["error"]
                break

            # Suggestions sentinel — store for rendering after rerun
            if isinstance(chunk, dict) and "suggestions" in chunk:
                st.session_state.suggestions = chunk["suggestions"]
                continue

            # Normal token — accumulate and stream with cursor
            full_response += chunk
            placeholder.markdown(full_response + "▌")

        # Final render — remove blinking cursor
        if error_message:
            placeholder.error(error_message)
        elif full_response:
            placeholder.markdown(full_response)
        else:
            placeholder.warning("⚠️ No response received.")

    # ── Persist to history ────────────────────────────────────────────────────
    if full_response and not error_message:
        st.session_state.chat_history.append({
            'question':  user_question,
            'answer':    full_response,
            'timestamp': datetime.now().isoformat(),
        })

    st.rerun()


# =============================================================================
# MAIN
# =============================================================================
def main():
    initialize_session_state()
    st.title("🤖 Multi-Agent Chatbot")
    st.markdown("---")
    render_sidebar()
    render_chat_interface()
    st.markdown("---")
    st.caption("Multi-Agent Chatbot System | Powered by LangChain, ChromaDB & Google Gemini")


if __name__ == "__main__":
    main()