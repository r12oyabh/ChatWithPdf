"""
Chat service for handling conversations with bots.
"""
from Utills.llm import LLMManager
from typing import Dict, Optional
from datetime import datetime
from Config.logger import logger

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_retrieval_chain          # ✅ Main package
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import mlflow

from Config.settings import settings
from Utills.chroma import chromadb_service
from Utills.file_utills import generate_namespace
from Utills.evaluation import evaluation_service



class ChatService:
    """Service class for handling chat interactions with bots."""
    def __init__(self):
        """Initialize chat service."""
        mlflow.gemini.autolog()
        self.llm=LLMManager().llm
        self.sessions: Dict[str, ChatMessageHistory] = {}
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. Use the following context to answer the question.
            
Context: {context}

Important:
- If you don't know the answer based on the context, say so clearly.
- Be concise and accurate in your responses.
- Only use information from the provided context."""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}")
        ])
    @mlflow.trace
    def chat(self, bot_id: str, question: str, session_id: Optional[str] = None) -> Dict:
        """
        Chat with a specific bot using its namespace.
        
        Args:
            bot_id: Unique bot identifier
            question: User's question
            session_id: Optional session ID for conversation history
            
        Returns:
            Dictionary containing answer and source documents
            
        Raises:
            Exception: If chat interaction fails
        """
        try:
            mlflow.update_current_trace(
            metadata={
                "mlflow.trace.user": bot_id,          # bot acts as the "user"
                "mlflow.trace.session": session_id,   # conversation/session id
                "bot_id": bot_id,                     # optional custom metadata
                "base_vector_db": "ChromaDB",
                "retriever_k": 3
            })
            
            # Create vector store with user isolation
            vectorstore = chromadb_service.create_vectorstore(user_id=bot_id)
            retriever = vectorstore.as_retriever(
                search_kwargs={"k":3}
            )
            # Create retrieval chain
            retrieval_chain = create_retrieval_chain(
                retriever,
                self.prompt | self.llm
            )
            # Use session ID or create default one
            if session_id is None:
                session_id = f"bot_{bot_id}_default"
            # Create conversational chain with memory
            conversational_chain = RunnableWithMessageHistory(
                retrieval_chain,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
            # Get response
            response = conversational_chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": session_id}}
            )
            answer = response.get('answer', '')
            if hasattr(answer, 'content'):
                # If it's an AIMessage object, extract the content
                answer = answer.content
            elif not isinstance(answer, str):
                # If it's any other object, convert to string
                answer = str(answer)
            # Format source documents
            source_docs = []
            context_texts = []
            for doc in response.get('context', []):
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                source_docs.append({
                    'page_content': content,
                    'metadata': doc.metadata if hasattr(doc, 'metadata') else {}
                })
                context_texts.append(content)
            
            # Run evaluation (asynchronously would be better, but doing sync for now as requested)
            try:
                logger.info("Starting evaluation...")
                eval_metrics = evaluation_service.evaluate(
                    question=question,
                    answer=answer,
                    context=context_texts
                )
                logger.info(f"Evaluation metrics: {eval_metrics}")
            except Exception as e:
                logger.error(f"Failed to run evaluation: {e}")

            return {
                'bot_id': bot_id,
                'question': question,
                'answer': answer,
                'source_documents': source_docs,
                'timestamp': datetime.now()
            }
        except Exception as e:
            raise Exception(f"Error during chat interaction: {str(e)}")
    
    @mlflow.trace(name="Get_Session_History", span_type="MEMORY")
    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        """
        Get or create chat history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ChatMessageHistory instance
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatMessageHistory()
        return self.sessions[session_id]
    
    def clear_session(self, session_id: str) -> None:
        """
        Clear chat history for a session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"✅ Session '{session_id}' cleared")
    
    def get_session_count(self) -> int:
        """
        Get the number of active sessions.
        
        Returns:
            Number of active sessions
        """
        return len(self.sessions)

# Global chat service instance
chat_service = ChatService()