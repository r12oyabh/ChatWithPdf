"""
Chat service for handling conversations with bots.
"""
import json

from Utills.llm import LLMManager
from typing import Dict, Optional,AsyncGenerator
from datetime import datetime
from Config.logger import logger

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_retrieval_chain          # ✅ Main package
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from fastapi.responses import StreamingResponse
from langchain_core.output_parsers import StrOutputParser
import re
import mlflow


from Config.settings import settings
from Utills.chroma import chromadb_service
from Utills.file_utills import generate_namespace
from Utills.evaluation import evaluation_service



class ChatService:
    """Service class for handling chat interactions with bots."""
    def __init__(self):
        """Initialize chat service."""
        mlflow.openai.autolog()
        self.llm=LLMManager().llm
        self.sessions: Dict[str, ChatMessageHistory] = {}
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. Use the following context to answer the question.
            
Context: {context}

Important:
- If you don't know the answer based on the context, say so clearly.
- Be concise and accurate in your responses.
- Only use information from the provided context. """),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}")
        ])
    @mlflow.trace
    async def chat(self, bot_id: str, question: str, session_id: Optional[str] = None) -> Dict:
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
                "retriever_k": "3",
                "question": question

            })

            with mlflow.start_run(run_name=f"chat_{bot_id}_{session_id}", nested=True):
            # Log input parameters
                mlflow.log_param("bot_id", bot_id)
                mlflow.log_param("session_id", session_id)
                mlflow.log_param("retriever_k", 3)
                mlflow.log_param("vector_db", "ChromaDB")
                mlflow.log_text(question, "input_question.txt")
            # Log the input question
            # Create vector store with user isolation
            vectorstore = chromadb_service.create_vectorstore(user_id=bot_id)
            retriever = vectorstore.as_retriever(
                search_kwargs={"k":15},search_type="mmr"
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
            response = await conversational_chain.ainvoke(
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

            return {
                'bot_id': bot_id,
                'question': question,
                'answer': answer,
                'source_documents': source_docs,
                'timestamp': datetime.now()
            }
        except Exception as e:
            raise Exception(f"Error during chat interaction: {str(e)}")
    
    @mlflow.trace
    async def stream_chat(
        self,
        bot_id: str,
        question: str,
        session_id: Optional[str] = None,
    ) -> StreamingResponse:
        """
        Stream chat response as Server-Sent Events (SSE).

        Each token is sent as a JSON object:
            data: {"type": "token", "value": "word "}\n\n

        Final sentinel:
            data: {"type": "done", "session_id": "...", "bot_id": "..."}\n\n

        Error event:
            data: {"type": "error", "message": "..."}\n\n
        """
        try:
            # ── MLflow tracing ────────────────────────────────────────────────
            mlflow.update_current_trace(
                metadata={
                    "mlflow.trace.user":    bot_id,
                    "mlflow.trace.session": session_id,
                    "bot_id":               bot_id,
                    "base_vector_db":       "ChromaDB",
                    "retriever_k":          "3",
                    "question":             question,
                }
            )

            with mlflow.start_run(run_name=f"stream_{bot_id}_{session_id}", nested=True):
                mlflow.log_param("bot_id",      bot_id)
                mlflow.log_param("session_id",  session_id)
                mlflow.log_param("retriever_k", 3)
                mlflow.log_param("vector_db",   "ChromaDB")
                mlflow.log_text(question, "input_question.txt")

            # ── Setup ─────────────────────────────────────────────────────────
            if session_id is None:
                session_id = f"bot_{bot_id}_default"

            vectorstore = chromadb_service.create_vectorstore(user_id=bot_id)
            retriever   = vectorstore.as_retriever(search_kwargs={"k": 3},search_type="mmr")
            docs        = retriever.invoke(question)

            context = "\n\n".join(
                doc.page_content if hasattr(doc, "page_content") else str(doc)
                for doc in docs
            )

            history         = self._get_session_history(session_id)
            chat_history    = history.messages
            streaming_chain = self.prompt | self.llm | StrOutputParser()

            suggestion_prompt = ChatPromptTemplate.from_template("""
            Based on the AI response and context below, suggest 3 short and relevant 
            follow-up questions the user might want to ask next.
            Be specific to the content. Return ONLY a valid JSON array of 3 strings.
            No explanation, no markdown, just the raw JSON array.

            AI Response: {answer}
            Context: {context}
            """)

            suggestion_chain = suggestion_prompt | self.llm | StrOutputParser()
            # ── Generator ─────────────────────────────────────────────────────
            async def event_generator() -> AsyncGenerator[str, None]:
                full_answer = ""
                buffer      = ""          # accumulates text until we hit a word boundary

                def make_event(payload: dict) -> str:
                    """Serialize a dict to a valid SSE line."""
                    return f"data: {json.dumps(payload)}\n\n"

                try:
                    async for chunk in streaming_chain.astream({
                        "input":        question,
                        "context":      context,
                        "chat_history": chat_history,
                    }):
                        full_answer += chunk
                        buffer      += chunk

                        # ── Flush complete "words" from the buffer ────────────
                        # A word boundary is any whitespace character.
                        # We keep the last segment (which may be incomplete) in
                        # the buffer and emit everything before it.
                        parts = re.split(r"(\s+)", buffer)   # keeps the delimiters

                        # parts alternates: [word, space, word, space, ..., tail]
                        # The last element is always an incomplete word or "".
                        # We emit everything except the last element.
                        to_emit = parts[:-1]   # complete word+space pairs
                        buffer  = parts[-1]    # incomplete tail — hold for next chunk

                        for part in to_emit:
                            if part:           # skip empty strings from re.split
                                yield make_event({"type": "token", "value": part})

                    # ── Flush whatever remains in the buffer ──────────────────
                    if buffer:
                        yield make_event({"type": "token", "value": buffer})

                    # ── Persist to memory ─────────────────────────────────────
                    history.add_user_message(question)
                    history.add_ai_message(full_answer)

                    raw_suggestions = await suggestion_chain.ainvoke({
                        "answer":  full_answer,
                        "context": context,
                    })
                    cleaned = (
                        raw_suggestions
                        .strip()
                        .removeprefix("```json")
                        .removeprefix("```")
                        .removesuffix("```")
                        .strip()
                    )

                    suggestions = json.loads(cleaned)

                    if isinstance(suggestions, list) and len(suggestions) > 0:
                        yield make_event({
                            "type":      "suggestions",
                            "questions": suggestions[:3],
                        })

                    # ── Done sentinel ─────────────────────────────────────────
                    yield make_event({
                        "type":       "done",
                        "bot_id":     bot_id,
                        "session_id": session_id,
                    })

                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    yield make_event({"type": "error", "message": str(e)})
                    yield make_event({"type": "done",  "bot_id": bot_id, "session_id": session_id})

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control":     "no-cache",
                    "Connection":        "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        except Exception as e:
            logger.error(f"Failed to initialize stream: {e}")
            raise Exception(f"Error during streaming chat: {str(e)}")

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
            logger.info(f"Session '{session_id}' cleared")
    
    def get_session_count(self) -> int:
        """
        Get the number of active sessions.
        
        Returns:
            Number of active sessions
        """
        return len(self.sessions)

    @mlflow.trace(name="Background_RAG_Evaluation", span_type="PARSER")
    def evaluate_chat_response(self, question: str, answer: str, source_documents: list):
        """
        Run evaluation for a chat response in the background.
        Args:
            question: User's question
            answer: Generated answer
            source_documents: List of source documents
        """
        try:
            logger.info("Starting background evaluation...")
            context_texts = [doc['page_content'] for doc in source_documents]
            
            # Use evaluation service
            eval_metrics = evaluation_service.evaluate(
                question=question,
                answer=answer,
                context=context_texts
            )
            logger.info(f"✅ Background evaluation metrics: {eval_metrics}")
        except Exception as e:
            logger.error(f"Failed to run background evaluation: {e}")

# Global chat service instance
chat_service = ChatService()