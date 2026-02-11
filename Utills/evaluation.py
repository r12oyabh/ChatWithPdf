"""
Evaluation service for MLflow GenAI metrics.
"""
import mlflow
import pandas as pd
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from Config.settings import settings
from Config.logger import logger
from Utills.llm import LLMManager

class EvaluationService:
    """Service for evaluating RAG responses using GPT-4o for accurate scoring."""

    def __init__(self):
        """Initialize evaluation service with GPT-4o."""
        # Enable OpenAI autologging for tracking GPT-4o calls
        mlflow.openai.autolog()
        
        # Use GPT-4o specifically for evaluation (more accurate for scoring)
        try:
            self.llm = LLMManager().llm
            logger.info(f"✅ Evaluation service initialized with GPT-4o (Azure OpenAI)")
        except Exception as e:
            logger.error(f"❌ Failed to initialize GPT-4o for evaluation: {e}")
            raise e

    @mlflow.trace(name="Evaluate_Faithfulness", span_type="CHAT_MODEL")
    def _evaluate_faithfulness(self, question: str, answer: str, context: list[str]) -> float:
        """Evaluate faithfulness: Is the answer derived from the context?
           **Check the Context and Answer are related to each other **
        Args:
            question: The user's question
            answer: The generated answer
            context: List of retrieved document chunks
        
        Returns:
            Faithfulness score (0.0 to 1.0)
        """
        prompt = ChatPromptTemplate.from_template("""
        You are an expert evaluator. Rate the Faithfulness of the answer on a scale of 0.0 to 1.0.
        Faithfulness measures if the answer is derived *only* from the context.
        
        Question: {question}
        Context: {context}
        Answer: {answer}
        
        Return ONLY the numeric score (0.0 to 1.0).
        """)
        try:
            with mlflow.start_span(name="LLM_Faithfulness_Scoring", span_type="CHAT_MODEL") as span:
                span.set_inputs({
                    "question": question[:200],
                    "answer": answer[:200],
                    "context_chunks": len(context)
                })
                
                chain = prompt | self.llm
                result = chain.invoke({"question": question, "context": "\n".join(context), "answer": answer})
                score = float(result.content.strip())
                
                span.set_outputs({"faithfulness_score": score})
                span.set_attributes({
                    "metric_type": "faithfulness",
                    "model": "gpt-4o"
                })
                
                return score
        except Exception as e:
            logger.warning(f"Faithfulness evaluation failed: {e}")
            return 0.0

    @mlflow.trace(name="Evaluate_Relevance", span_type="CHAT_MODEL")
    def _evaluate_relevance(self, question: str, answer: str) -> float:
        """Evaluate relevance: Does the answer address the question?"""
        prompt = ChatPromptTemplate.from_template("""
        You are an expert evaluator. Rate the Relevance of the answer on a scale of 0.0 to 1.0.
        Relevance measures if the answer actually addresses the user's question.
        
        Question: {question}
        Answer: {answer}
        
        Return ONLY the numeric score (0.0 to 1.0).
        """)
        try:
            with mlflow.start_span(name="LLM_Relevance_Scoring", span_type="CHAT_MODEL") as span:
                span.set_inputs({
                    "question": question[:200],
                    "answer": answer[:200]
                })
                
                chain = prompt | self.llm
                result = chain.invoke({"question": question, "answer": answer})
                score = float(result.content.strip())
                
                span.set_outputs({"relevance_score": score})
                span.set_attributes({
                    "metric_type": "relevance",
                    "model": "gpt-4o"
                })
                
                return score
        except Exception as e:
            logger.warning(f"Relevance evaluation failed: {e}")
            return 0.0

    @mlflow.trace(name="RAG_Evaluation", span_type="PARSER")
    def evaluate(self, question: str, answer: str, context: list[str], ground_truth: str = None) -> dict:
        """
        Evaluate a single RAG response.

        Args:
            question: The user's question
            answer: The generated answer
            context: List of retrieved document chunks
            ground_truth: The correct answer (optional)

        Returns:
            Dictionary of metric scores
        """
        try:
            with mlflow.start_span(name="Calculate_Quality_Metrics", span_type="PARSER") as eval_span:
                eval_span.set_inputs({
                    "question": question[:100],
                    "answer_length": len(answer),
                    "context_chunks": len(context),
                    "has_ground_truth": ground_truth is not None
                })
                
                # Calculate metrics manually using our LLM
                faithfulness_score = self._evaluate_faithfulness(question, answer, context)
                relevance_score = self._evaluate_relevance(question, answer)
                
                # Calculate average score
                average_score = (faithfulness_score + relevance_score) / 2
                
                metrics = {
                    "faithfulness_score": faithfulness_score,
                    "relevance_score": relevance_score,
                    "average_quality_score": average_score
                }
                
                eval_span.set_outputs(metrics)
                eval_span.set_attributes({
                    "evaluation_type": "RAG_quality",
                    "metrics_computed": ["faithfulness", "relevance"],
                    "passed_quality_threshold": average_score >= 0.7
                })
            
            # Log metrics to MLflow
            # We are likely inside a trace or run initiated by the chat service context
            # tracing might not be an active run in the traditional sense, so we check.
            if mlflow.active_run():
                 mlflow.log_metrics(metrics)
            else:
                 with mlflow.start_run(run_name="evaluation", nested=True):
                     mlflow.log_metrics(metrics)
            
            logger.info(f"✅ Evaluation completed: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"❌ Error during evaluation: {str(e)}")
            return {}

evaluation_service = EvaluationService()