"""
Streamlit Frontend for Multi-Agent Chatbot System.

This application provides a user-friendly interface for:
- Uploading documents and creating bots
- Chatting with created bots
- Managing conversation sessions
"""

import streamlit as st
import requests
from typing import Optional, Dict, List
import uuid
from datetime import datetime
import json


# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'bot_id' not in st.session_state:
        st.session_state.bot_id = None
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'bot_metadata' not in st.session_state:
        st.session_state.bot_metadata = None
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []


# =============================================================================
# API FUNCTIONS
# =============================================================================

def upload_and_create_bot(team_name: str, bot_name: str, files: List) -> Optional[Dict]:
    """
    Upload files and create a new bot.
    
    Args:
        team_name: Name of the team/organization
        bot_name: Name of the bot
        files: List of uploaded files
        
    Returns:
        Bot metadata dictionary or None if failed
    """
    try:
        # Prepare form data
        form_data = {
            'team_name': team_name,
            'bot_name': bot_name
        }
        
        # Prepare files for upload
        files_data = []
        for uploaded_file in files:
            files_data.append(
                ('files', (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type))
            )
        
        # Make API request
        response = requests.post(
            f"{API_BASE_URL}/bot/upload",
            data=form_data,
            files=files_data,
            timeout=60
        )
        
        if response.status_code == 201:
            return response.json()
        else:
            st.error(f"Error creating bot: {response.json().get('detail', 'Unknown error')}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the API server. Please ensure it's running.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


def chat_with_bot(bot_id: str, question: str, session_id: str) -> Optional[Dict]:
    """
    Send a question to the bot and get a response.
    
    Args:
        bot_id: Unique bot identifier
        question: User's question
        session_id: Session identifier for conversation history
        
    Returns:
        Chat response dictionary or None if failed
    """
    try:
        payload = {
            "bot_id": bot_id,
            "question": question,
            "session_id": session_id
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the API server.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


def check_health() -> bool:
    """
    Check if the API server is healthy.
    
    Returns:
        True if server is healthy, False otherwise
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_header():
    """Render the application header."""
    st.title("🤖 Multi-Agent Chatbot")
    st.markdown("---")


def render_sidebar():
    """Render the sidebar with bot creation interface."""
    with st.sidebar:
        st.header("📤 Create New Bot")
        
        # Team and bot information
        team_name = st.text_input(
            "Team Name",
            placeholder="e.g., Engineering Team",
            help="Name of your team or organization"
        )
        
        bot_name = st.text_input(
            "Bot Name",
            placeholder="e.g., HR Assistant",
            help="Name for your chatbot"
        )
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files"
        )
        
        # Create bot button
        if st.button("🚀 Create Bot", type="primary", use_container_width=True):
            if not team_name or not bot_name:
                st.error("Please provide both team name and bot name")
            elif not uploaded_files:
                st.error("Please upload at least one file")
            else:
                with st.spinner("Creating bot... This may take a moment."):
                    bot_metadata = upload_and_create_bot(team_name, bot_name, uploaded_files)
                    
                    if bot_metadata:
                        # Save bot_id and metadata to session state
                        st.session_state.bot_id = bot_metadata['bot_id']
                        st.session_state.bot_metadata = bot_metadata
                        st.session_state.chat_history = []  # Reset chat history
                        
                        st.success("✅ Bot created successfully!")
                        st.rerun()
        
        # Display current bot info
        st.markdown("---")
        st.subheader("📊 Current Bot")
        
        if st.session_state.bot_metadata:
            metadata = st.session_state.bot_metadata
            
            st.info(f"""
            **Team:** {metadata['team_name']}  
            **Bot:** {metadata['bot_name']}  
            **Files:** {len(metadata['file_names'])} document(s)  
            **Chunks:** {metadata['chunk_count']}
            """)
            
            # Show uploaded files
            with st.expander("📁 Uploaded Files"):
                for file_name in metadata['file_names']:
                    st.text(f"• {file_name}")
            
            # New session button
            if st.button("🔄 New Session", use_container_width=True):
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.chat_history = []
                st.success("New session started!")
                st.rerun()
            
            # Reset bot button
            if st.button("🗑️ Reset Bot", use_container_width=True):
                st.session_state.bot_id = None
                st.session_state.bot_metadata = None
                st.session_state.chat_history = []
                st.success("Bot reset. You can create a new one.")
                st.rerun()
        else:
            st.warning("No bot created yet. Upload files to get started!")
        
        # Session info
        st.markdown("---")
        st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")
        
        # Server status
        st.markdown("---")
        if check_health():
            st.success("🟢 API Server: Online")
        else:
            st.error("🔴 API Server: Offline")


def render_chat_interface():
    """Render the main chat interface."""
    if not st.session_state.bot_id:
        # Show welcome message if no bot is created
        st.info("👈 Please create a bot using the sidebar to start chatting!")
        
        st.markdown("""
        ### How to Use:
        
        1. **Create a Bot** (Sidebar)
           - Enter your team name and bot name
           - Upload PDF, DOCX, or TXT files
           - Click "Create Bot"
        
        2. **Start Chatting**
           - Ask questions about your uploaded documents
           - The bot will answer based on the content
        
        3. **Session Management**
           - Conversations are maintained within a session
           - Start a new session to reset the conversation
        
        ### Features:
        - 🔒 Multi-Agent isolation (each bot has its own data)
        - 💬 Conversation history within sessions
        - 📄 Support for multiple file formats
        - 🎯 Context-aware responses
        """)
        
    else:
        # Chat interface
        st.subheader("💬 Chat with Your Bot")
        
        # Display chat history
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_history:
                # User message
                with st.chat_message("user"):
                    st.write(message['question'])
                
                # Bot response
                with st.chat_message("assistant"):
                    st.write(message['answer'])
                    
                    # Show source documents in expander
                    if message.get('source_documents'):
                        with st.expander("📚 View Sources"):
                            for idx, doc in enumerate(message['source_documents'], 1):
                                st.markdown(f"**Source {idx}:**")
                                st.text(doc['page_content'][:200] + "...")
                                st.markdown("---")
        
        # Chat input
        user_question = st.chat_input("Ask a question about your documents...")
        
        if user_question:
            # Add user message to chat history (optimistic update)
            with st.chat_message("user"):
                st.write(user_question)
            
            # Get bot response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_with_bot(
                        bot_id=st.session_state.bot_id,
                        question=user_question,
                        session_id=st.session_state.session_id
                    )
                
                if response:
                    st.write(response['answer'])
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': user_question,
                        'answer': response['answer'],
                        'source_documents': response.get('source_documents', []),
                        'timestamp': response.get('timestamp', datetime.now().isoformat())
                    })
                    
                    # Show sources
                    if response.get('source_documents'):
                        with st.expander("📚 View Sources"):
                            for idx, doc in enumerate(response['source_documents'], 1):
                                st.markdown(f"**Source {idx}:**")
                                st.text(doc['page_content'][:200] + "...")
                                st.markdown("---")
                    
                    st.rerun()


def render_footer():
    """Render the application footer."""
    st.markdown("---")
    st.caption("Multi-Agent Chatbot System | Powered by LangChain, ChromaDb, and Google Gemini")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()
    
    # Render UI components
    render_header()
    render_sidebar()
    render_chat_interface()
    render_footer()


if __name__ == "__main__":
    main()