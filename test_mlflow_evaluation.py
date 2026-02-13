"""
Test script to verify MLflow evaluation with Azure OpenAI configuration.
This demonstrates the fix for the 'NoneType' object has no attribute 'get_endpoint' error.
"""
import os
import pandas as pd
import mlflow
from Config.settings import settings

# ✅ FIX: Configure Azure OpenAI environment variables BEFORE calling mlflow.evaluate()
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = settings.AZURE_OPENAI_ENDPOINT
os.environ["OPENAI_API_KEY"] = settings.OPEN_API_KEY
os.environ["OPENAI_API_VERSION"] = settings.AZURE_OPENAI_API_VERSION
os.environ["OPENAI_DEPLOYMENT_NAME"] = settings.AZURE_OPENAI_DEPLOYMENT

print("✅ Azure OpenAI environment variables configured")
print(f"   Endpoint: {os.environ['OPENAI_API_BASE']}")
print(f"   Deployment: {os.environ['OPENAI_DEPLOYMENT_NAME']}")

# Prepare test data (from your example)
# ✅ MLflow expects 'predictions' column, not 'outputs'
eval_data = pd.DataFrame({
    "inputs": ["I wanted to know about my email and phone number?"],
    "context": [
        "--- Content from 20260210_223520_AbhinandanResume (1).pdf ---\n"
        "Abhinandan Kumar\nSoftware Engineer - AI/ML | 2+ Years Experience\n"
        "ak0590810@gmail.com | +91 6239561703 | LinkedIn | GitHub"
    ],
    "predictions": ["Your email is `ak0590810@gmail.com` and your phone number is `+91 6239561703`."],
    "ground_truth": ["Email: ak0590810@gmail.com, Phone: +91 6239561703"]
})

print("\n📊 Evaluation Data:")
print(eval_data)

# Set MLflow tracking
mlflow.set_tracking_uri(settings.ML_FLOW_TRACKING_URL)
mlflow.set_experiment(settings.ML_FLOW_EXPERIMENT_NAME)

print(f"\n🔬 Starting MLflow evaluation...")

try:
    with mlflow.start_run(run_name="test_evaluation_fix"):
        # ✅ Use the NEW MLflow 3.0 GenAI evaluation API
        # Define metrics using make_genai_metric
        from mlflow.metrics.genai import faithfulness, answer_relevance
        
        # Create metrics with Azure OpenAI as judge
        faithfulness_metric = faithfulness(model=f"openai:/{settings.AZURE_OPENAI_DEPLOYMENT}")
        answer_relevance_metric = answer_relevance(model=f"openai:/{settings.AZURE_OPENAI_DEPLOYMENT}")
        
        print(f"\n📊 Using metrics: faithfulness, answer_relevance")
        
        # Use the new mlflow.genai.evaluate API (MLflow 3.0+)
        results = mlflow.evaluate(
            data=eval_data,
            model_type="question-answering",
            targets="ground_truth",  # Specify ground truth column
            predictions="predictions",  # Specify predictions column
            extra_metrics=[faithfulness_metric, answer_relevance_metric],
        )
        
        print("\n✅ Evaluation completed successfully!")
        print("\n📈 Metrics:")
        for metric_name, metric_value in results.metrics.items():
            print(f"   {metric_name}: {metric_value}")
        
        print("\n📊 Detailed Results:")
        if hasattr(results, 'tables') and 'eval_results_table' in results.tables:
            print(results.tables['eval_results_table'])
        
except Exception as e:
    print(f"\n❌ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
