"""
Chat service for handling conversations with bots.
"""
from Utills.llm import LLMManager
from typing import Dict, Optional
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_retrieval_chain          # ✅ Main package
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document


from Utills.chroma import chromadb_service
from Utills.evaluation import evaluation_service
from Config.logger import logger
from Config.telemetry import tracer, meter

# Define metrics
chat_requests_counter = meter.create_counter(
    "rag.chat.requests",
    unit="1",
    description="Number of chat requests handled"
)

# Initialize tracer



class ChatService:
    """Service class for handling chat interactions with bots."""
    def __init__(self):
        """Initialize chat service."""
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
    def retrieve_documents(self, bot_id: str, question: str, k: int = 3):
        """
        Retrieve relevant documents from ChromaDB.
        """
        with tracer.start_as_current_span("retrieve_documents") as span:
            span.set_attribute("bot_id", bot_id)
            span.set_attribute("retrieval.k", k)
            
            vectorstore = chromadb_service.create_vectorstore(user_id=bot_id)
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            docs = retriever.invoke(question)
            
            span.set_attribute("retrieval.count", len(docs))
            return docs


    def chat(self, bot_id: str, question: str, session_id: Optional[str] = None, context: Optional[list] = None) -> Dict:
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
        chat_requests_counter.add(1, {"bot_id": bot_id})

        with tracer.start_as_current_span("chat_flow") as span:
            span.set_attribute("bot_id", bot_id)
            if session_id:
                span.set_attribute("session_id", session_id)

            try:
                # Get context (either provided or retrieved)
                if context is None:
                    # Create vector store with user isolation
                    vectorstore = chromadb_service.create_vectorstore(user_id=bot_id)
                    retriever = vectorstore.as_retriever(
                        search_kwargs={"k":3}
                    )
                else:
                    # Use provided context - create a simple retriever that returns it

                    class FixedRetriever(BaseRetriever):
                        docs: list
                        def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
                            return self.docs
                    
                    retriever = FixedRetriever(docs=context)

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
                for doc in response.get('context', []):
                    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    source_docs.append({
                        'page_content': content,
                        'metadata': doc.metadata if hasattr(doc, 'metadata') else {}
                    })

                span.set_attribute("answer_length", len(answer))
                span.set_attribute("source_count", len(source_docs))

                return {
                    'bot_id': bot_id,
                    'question': question,
                    'answer': answer,
                    'source_documents': source_docs,
                    'timestamp': datetime.now()
                }

            except Exception as e:
                span.record_exception(e)
                raise Exception(f"Error during chat interaction: {str(e)}")

    
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

    def evaluate_chat_response(self, question: str, answer: str, source_documents: list):
        """
        Run evaluation for a chat response in the background.
        Args:
            question: User's question
            answer: Generated answer
            source_documents: List of source documents
        """
        with tracer.start_as_current_span("background_evaluation") as span:
            try:
                logger.info("🎬 Starting background evaluation...")
                context_texts = [doc['page_content'] for doc in source_documents]
                # Use evaluation service
                eval_metrics = evaluation_service.evaluate(
                    question=question,
                    answer=answer,
                    context=context_texts,
                )
                
                # Record evaluation metrics in span
                if eval_metrics:
                    for metric, score in eval_metrics.items():
                        span.set_attribute(f"evaluation.{metric}", score)

                logger.info(f"✅ Background evaluation metrics: {eval_metrics}")
            except Exception as e:
                span.record_exception(e)
                logger.error(f"❌ Failed to run background evaluation: {e}")


# Global chat service instance
chat_service = ChatService()