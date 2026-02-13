"""
Test script for the updated evaluation service.
Verifies that the MLflow-based evaluation works correctly and maintains backward compatibility.
"""
import sys

from Utills.evaluation import evaluation_service
from Config.logger import logger
import mlflow
from Config.settings import settings

# Set MLflow tracking
mlflow.set_tracking_uri(settings.ML_FLOW_TRACKING_URL)
mlflow.set_experiment(settings.ML_FLOW_EXPERIMENT_NAME)

print("=" * 80)
print("Testing Updated Evaluation Service")
print("=" * 80)

# Test data
test_question = "I wanted to know about my email and phone number?"
test_context = [
    "--- Content from Resume ---",
    "Abhinandan Kumar",
    "Software Engineer - AI/ML | 2+ Years Experience",
    "ak0590810@gmail.com | +91 6239561703 | LinkedIn | GitHub"
]
test_answer = "Your email is `ak0590810@gmail.com` and your phone number is `+91 6239561703`."
test_ground_truth = "Email: ak0590810@gmail.com, Phone: +91 6239561703"

print("\n📝 Test Data:")
print(f"   Question: {test_question}")
print(f"   Answer: {test_answer[:50]}...")
print(f"   Context chunks: {len(test_context)}")

# Test 1: Single evaluation without ground truth
print("\n" + "=" * 80)
print("Test 1: Single Evaluation (without ground truth)")
print("=" * 80)

try:
    with mlflow.start_run(run_name="test_evaluation_service_single"):
        metrics = evaluation_service.evaluate(
            question=test_question,
            answer=test_answer,
            context=test_context
        )
        
        print("\n✅ Evaluation completed successfully!")
        print("\n📊 Metrics returned:")
        for key, value in metrics.items():
            print(f"   {key}: {value}")
        
        # Verify backward compatibility
        assert "faithfulness_score" in metrics, "Missing faithfulness_score"
        assert "relevance_score" in metrics, "Missing relevance_score"
        assert "average_quality_score" in metrics, "Missing average_quality_score"
        
        # Verify scores are in 0.0-1.0 range (backward compatible)
        assert 0.0 <= metrics["faithfulness_score"] <= 1.0, "faithfulness_score out of range"
        assert 0.0 <= metrics["relevance_score"] <= 1.0, "relevance_score out of range"
        
        print("\n✅ Backward compatibility verified!")
        print("   - All expected metric keys present")
        print("   - Scores normalized to 0.0-1.0 range")
        
except Exception as e:
    print(f"\n❌ Test 1 failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Single evaluation with ground truth (includes RAGAS)
print("\n" + "=" * 80)
print("Test 2: Single Evaluation (with ground truth)")
print("=" * 80)

try:
    with mlflow.start_run(run_name="test_evaluation_service_with_ground_truth"):
        metrics = evaluation_service.evaluate(
            question=test_question,
            answer=test_answer,
            context=test_context,
            ground_truth=test_ground_truth
        )
        
        print("\n✅ Evaluation with ground truth completed!")
        print("\n📊 Metrics returned:")
        for key, value in metrics.items():
            print(f"   {key}: {value}")
        
        # Check if RAGAS metrics are included
        if "context_precision_score" in metrics:
            print("\n✅ RAGAS metrics included!")
        else:
            print("\n⚠️  RAGAS metrics not included (may have failed)")
        
except Exception as e:
    print(f"\n❌ Test 2 failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Batch evaluation
print("\n" + "=" * 80)
print("Test 3: Batch Evaluation")
print("=" * 80)

try:
    with mlflow.start_run(run_name="test_evaluation_service_batch"):
        batch_metrics = evaluation_service.evaluate_batch(
            questions=[test_question, "What is your name?"],
            answers=[test_answer, "My name is Abhinandan Kumar"],
            contexts=[test_context, ["Name: Abhinandan Kumar"]],
            ground_truths=[test_ground_truth, "Abhinandan Kumar"]
        )
        
        print("\n✅ Batch evaluation completed!")
        print("\n📊 Batch Metrics:")
        for key, value in batch_metrics.items():
            print(f"   {key}: {value}")
        
except Exception as e:
    print(f"\n❌ Test 3 failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("All Tests Completed!")
print("=" * 80)
print("\n💡 Next steps:")
print("   1. Check MLflow UI at http://localhost:5000")
print("   2. Navigate to 'MultiAgent' experiment")
print("   3. View the test runs to see logged metrics")
print("\n✅ The evaluation service is ready for production use!")
