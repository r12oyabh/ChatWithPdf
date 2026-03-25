"""
Production-ready evaluation service for MLflow GenAI metrics.
Uses MLflow's built-in evaluation framework with Azure OpenAI as the judge LLM.
"""
import os
import mlflow
import pandas as pd
from typing import Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from ragas import evaluate as ragas_evaluate
from ragas.metrics import context_precision, context_recall
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

from Config.settings import settings
from Config.logger import logger
from Utills.llm import LLMManager


class EvaluationService:
    """Production-ready service for evaluating RAG responses using MLflow's built-in metrics."""

    def __init__(self):
        """Initialize evaluation service with Azure OpenAI configuration."""
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_BASE"] = settings.AZURE_OPENAI_ENDPOINT
        os.environ["OPENAI_API_KEY"] = settings.OPEN_API_KEY
        os.environ["OPENAI_API_VERSION"] = settings.AZURE_OPENAI_API_VERSION
        os.environ["OPENAI_DEPLOYMENT_NAME"] = settings.AZURE_OPENAI_DEPLOYMENT
        
        mlflow.openai.autolog()
        
        try:
            self.llm = LLMManager().llm
            logger.info(f"Evaluation service initialized with Azure OpenAI (GPT-4o)")
            logger.info(f"Endpoint: {settings.AZURE_OPENAI_ENDPOINT}")
            logger.info(f"Deployment: {settings.AZURE_OPENAI_DEPLOYMENT}")
        except Exception as e:
            logger.error(f"Failed to initialize evaluation service: {e}")
            raise e

    @mlflow.trace(name="MLflow_Evaluation", span_type="PARSER")
    def evaluate(
        self, 
        question: str, 
        answer: str, 
        context: List[str], 
        ground_truth: Optional[str] = None
    ) -> Dict:
        """
        Evaluate a single RAG response using MLflow's built-in metrics.
        
        This method maintains backward compatibility with the existing interface
        while using MLflow's standardized evaluation framework internally.
        
        Args:
            question: User's question
            answer: Generated answer
            context: List of retrieved document chunks
            ground_truth: Optional ground truth answer for additional metrics
            
        Returns:
            Dictionary of evaluation metrics (normalized to 0.0-1.0 scale for compatibility)
        """
        try:
            # Convert single example to batch format for MLflow evaluation
            results = self.evaluate_batch(
                questions=[question],
                answers=[answer],
                contexts=[context],
                ground_truths=[ground_truth] if ground_truth else None
            )
            
            # Extract metrics from the first (and only) row
            if results:
                return results
            else:
                logger.warning("Evaluation returned no results")
                return {}
                
        except Exception as e:
            logger.error(f"Error during single evaluation: {str(e)}")
            return {}

    @mlflow.trace(name="MLflow_Batch_Evaluation", span_type="PARSER")
    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate a batch of RAG responses using MLflow's evaluation framework.
        
        Args:
            questions: List of user questions
            answers: List of generated answers
            contexts: List of context lists (retrieved documents)
            ground_truths: Optional list of ground truth answers
            
        Returns:
            Dictionary of aggregated evaluation metrics
        """
        try:
            with mlflow.start_span(name="Prepare_Evaluation_Data", span_type="PARSER") as span:
                # Prepare data in MLflow's expected format
                eval_data = pd.DataFrame({
                    "inputs": questions,
                    "predictions": answers,  # MLflow expects 'predictions' column
                    "context": ["\n".join(ctx) for ctx in contexts],
                })
                
                if ground_truths:
                    eval_data["ground_truth"] = ground_truths
                
                span.set_outputs({
                    "num_examples": len(questions),
                    "has_ground_truth": ground_truths is not None
                })
            
            # Import MLflow metrics
            from mlflow.metrics.genai import faithfulness, answer_relevance
            
            # Create metrics with Azure OpenAI as judge
            faithfulness_metric = faithfulness(model=f"openai:/{settings.AZURE_OPENAI_DEPLOYMENT}")
            answer_relevance_metric = answer_relevance(model=f"openai:/{settings.AZURE_OPENAI_DEPLOYMENT}")
            
            logger.info(f"Starting MLflow evaluation for {len(questions)} example(s)")
            
            # Run MLflow evaluation
            with mlflow.start_span(name="Run_MLflow_Evaluate", span_type="PARSER"):
                results = mlflow.evaluate(
                    data=eval_data,
                    model_type="question-answering",
                    targets="ground_truth" if ground_truths else None,
                    predictions="predictions",
                    extra_metrics=[faithfulness_metric, answer_relevance_metric],
                )
            
            # Extract and normalize metrics for backward compatibility
            metrics = {}
            
            # MLflow metrics are on 1-5 scale, normalize to 0.0-1.0 for backward compatibility
            if "faithfulness/v1/mean" in results.metrics:
                # Normalize from 1-5 scale to 0.0-1.0 scale
                faithfulness_raw = results.metrics["faithfulness/v1/mean"]
                metrics["faithfulness_score"] = (faithfulness_raw - 1) / 4  # Convert 1-5 to 0-1
                metrics["faithfulness_raw"] = faithfulness_raw  # Keep original for reference
            
            if "answer_relevance/v1/mean" in results.metrics:
                relevance_raw = results.metrics["answer_relevance/v1/mean"]
                metrics["relevance_score"] = (relevance_raw - 1) / 4  # Convert 1-5 to 0-1
                metrics["answer_relevance_raw"] = relevance_raw  # Keep original for reference
            
            # Add RAGAS metrics if ground truth is provided
            if ground_truths:
                try:
                    ragas_scores = self._evaluate_ragas_metrics(
                        questions[0], answers[0], contexts[0], ground_truths[0]
                    )
                    metrics.update(ragas_scores)
                except Exception as e:
                    logger.warning(f"RAGAS evaluation failed: {e}")
            
            # Calculate average quality score
            valid_scores = [
                metrics.get("faithfulness_score", 0),
                metrics.get("relevance_score", 0)
            ]
            metrics["average_quality_score"] = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            
            # Log metrics to MLflow
            if mlflow.active_run():
                mlflow.log_metrics(metrics)
            
            logger.info(f"✅ Evaluation completed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"MLflow batch evaluation failed: {str(e)}")
            # Fallback to custom evaluation if MLflow fails
            logger.info("Attempting fallback to custom evaluation...")
            return self._fallback_evaluation(questions[0], answers[0], contexts[0])

    @mlflow.trace(name="RAGAS_Evaluation", span_type="PARSER")
    def _evaluate_ragas_metrics(
        self, 
        question: str, 
        answer: str, 
        context: List[str], 
        ground_truth: str
    ) -> Dict:
        """
        Evaluate using RAGAS library metrics (context precision and recall).
        
        This is kept for additional retrieval quality metrics when ground truth is available.
        """
        try:
            # Prepare data for RAGAS
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [context],
                "ground_truth": [ground_truth]
            }
            dataset = Dataset.from_dict(data)
            
            # Configure RAGAS to use our Azure LLM
            ragas_llm = LangchainLLMWrapper(self.llm)
            ragas_embeddings = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(
                google_api_key=settings.GEMINI_API_KEY,
            ))

            # Update metrics with our LLM/Embeddings
            context_precision.llm = ragas_llm
            context_recall.llm = ragas_llm
            
            # Run evaluation
            with mlflow.start_span(name="Ragas_Library_Call", span_type="PARSER") as span:
                results = ragas_evaluate(
                    dataset=dataset,
                    metrics=[context_precision, context_recall],
                    llm=ragas_llm, 
                    embeddings=ragas_embeddings 
                )
                
                scores = {
                    "context_precision_score": results["context_precision"],
                    "context_recall_score": results["context_recall"]
                }
                span.set_outputs(scores)
                return scores

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {}

    @mlflow.trace(name="Fallback_Evaluation", span_type="CHAT_MODEL")
    def _fallback_evaluation(self, question: str, answer: str, context: List[str]) -> Dict:
        """
        Fallback evaluation using custom LLM prompts if MLflow evaluation fails.
        
        This ensures the system continues to work even if MLflow has issues.
        """
        try:
            logger.info("Using fallback custom evaluation")
            
            # Simple faithfulness check
            faithfulness_prompt = ChatPromptTemplate.from_template("""
            Rate the faithfulness of the answer on a scale of 0.0 to 1.0.
            Faithfulness measures if the answer is derived only from the context.
            
            Question: {question}
            Context: {context}
            Answer: {answer}
            
            Return ONLY the numeric score (0.0 to 1.0).
            """)
            # Simple relevance check
            relevance_prompt = ChatPromptTemplate.from_template("""
            Rate the relevance of the answer on a scale of 0.0 to 1.0.
            Relevance measures if the answer addresses the question.
            Question: {question}
            Answer: {answer}
            Return ONLY the numeric score (0.0 to 1.0).
            """)
            
            faithfulness_chain = faithfulness_prompt | self.llm
            relevance_chain = relevance_prompt | self.llm
            
            faithfulness_result = faithfulness_chain.invoke({
                "question": question, 
                "context": "\n".join(context), 
                "answer": answer
            })
            relevance_result = relevance_chain.invoke({
                "question": question, 
                "answer": answer
            })
            
            faithfulness_score = float(faithfulness_result.content.strip())
            relevance_score = float(relevance_result.content.strip())
            
            metrics = {
                "faithfulness_score": faithfulness_score,
                "relevance_score": relevance_score,
                "average_quality_score": (faithfulness_score + relevance_score) / 2,
                "evaluation_method": "fallback"
            }
            
            # Log to MLflow if in active run
            if mlflow.active_run():
                mlflow.log_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Fallback evaluation also failed: {e}")
            return {
                "faithfulness_score": 0.0,
                "relevance_score": 0.0,
                "average_quality_score": 0.0,
                "evaluation_method": "failed"
            }
# Global evaluation service instance
evaluation_service = EvaluationService()