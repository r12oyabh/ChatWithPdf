# MLflow Tracing Guide for RAG Application

## 📊 Overview

This RAG application now has **comprehensive MLflow tracing** implemented to provide full visibility into:
- **What's happening**: Every step of the workflow is tracked
- **Why you get certain responses**: See retrieved documents, prompts, and evaluation scores
- **Where time is spent**: Identify bottlenecks in your pipeline

## 🎯 What Gets Traced

### 1. **Chat Pipeline** (`chatservice.py`)
The main RAG chat flow has 6 traced steps:

| Span Name | Type | What It Tracks |
|-----------|------|----------------|
| `RAG_Chat_Pipeline` | `CHAIN` | The entire chat request from start to finish |
| `Load_Conversation_History` | `MEMORY` | Loading previous chat history for context |
| `Document_Retrieval` | `RETRIEVER` | Searching ChromaDB for relevant documents |
| `Build_RAG_Chain` | `CHAIN` | Constructing the retrieval + LLM chain |
| `Generate_Answer` | `CHAT_MODEL` | LLM generating the final answer |
| `RAG_Evaluation` | `PARSER` | Quality metrics (faithfulness, relevance) |

### 2. **Bot Creation Pipeline** (`botservice.py`)
When creating a new bot, these steps are traced:

| Span Name | Type | What It Tracks |
|-----------|------|----------------|
| `Create_Bot_Pipeline` | `CHAIN` | The entire bot creation process |
| `Extract_Text_From_Files` | `PARSER` | Extracting text from uploaded documents |
| `Chunk_Documents` | `PARSER` | Splitting text into chunks |
| `Store_In_VectorDB` | `EMBEDDING` | Creating embeddings and storing in ChromaDB |

### 3. **Database Operations** (`chroma.py`)

| Span Name | Type | What It Tracks |
|-----------|------|----------------|
| `ChromaDB.create_vectorstore` | `RETRIEVER` | Setting up the vector database connection |
| `Collection_Stats` | `RETRIEVER` | Fetching collection metadata |
| `ChromaDB.store_documents` | `EMBEDDING` | Embedding and storing documents |
| `Create_Embeddings` | `EMBEDDING` | The actual embedding generation process |

### 4. **Evaluation** (`evaluation.py`)

| Span Name | Type | What It Tracks |
|-----------|------|----------------|
| `RAG_Evaluation` | `PARSER` | Overall evaluation orchestration |
| `Evaluate_Faithfulness` | `CHAT_MODEL` | LLM scoring faithfulness (0.0-1.0) |
| `Evaluate_Relevance` | `CHAT_MODEL` | LLM scoring relevance (0.0-1.0) |
| `Calculate_Quality_Metrics` | `PARSER` | Aggregating all quality metrics |

## 🔍 What Information Is Captured

### Trace-Level Metadata
Every trace includes:
```python
{
    "mlflow.trace.user": "bot_id",
    "mlflow.trace.session": "session_id",
    "bot_id": "acme_support_bot",
    "base_vector_db": "ChromaDB",
    "retriever_k": 3,
    "model_type": "RAG",
    "pipeline_version": "v1.0"
}
```

### Span-Level Data
Each span captures:

**Inputs**: What went into the operation
```json
{
  "query": "What is MLflow tracing?",
  "bot_id": "acme_support_bot",
  "k": 3
}
```

**Outputs**: What came out
```json
{
  "num_documents_retrieved": 3,
  "answer_length": 245,
  "faithfulness_score": 0.92,
  "relevance_score": 0.88
}
```

**Attributes**: Additional metadata
```json
{
  "vector_db": "ChromaDB",
  "model": "google-gemini",
  "temperature": 0.7,
  "embedding_model": "models/embedding-001"
}
```

## 📈 How to View Traces in MLflow UI

### Step 1: Start MLflow UI
```bash
cd c:\Users\AbhinandanKumar\Rag
mlflow ui
```

The UI will open at `http://localhost:5000`

### Step 2: Navigate to Traces

1. **Experiments Tab** → Select your experiment
2. **Traces Tab** → See all traces
3. Click on a specific trace to dive deep

### Step 3: Analyze the Trace

You'll see a **waterfall diagram** showing:

```
RAG_Chat_Pipeline (3.5s total)
├─ Load_Conversation_History (0.05s)
├─ Document_Retrieval (0.8s)  ← Bottleneck identified!
├─ Build_RAG_Chain (0.02s)
├─ Generate_Answer (2.1s)     ← LLM call taking time
└─ RAG_Evaluation (0.53s)
   ├─ Calculate_Quality_Metrics
   ├─ Evaluate_Faithfulness (0.3s)
   └─ Evaluate_Relevance (0.23s)
```

### Step 4: Click on Individual Spans

For each span, you can see:
- ✅ **Inputs**: The exact query, parameters used
- ✅ **Outputs**: Retrieved documents, generated answer, scores
- ✅ **Timing**: Start time, end time, duration
- ✅ **Attributes**: Model used, temperature, token counts
- ✅ **Status**: OK or ERROR

### Step 5: Viewing Custom Metadata (Bot ID, Team, etc.)

To see the custom fields (like `bot_id`, `team_name`, `question`, `num_files`) that we added:

1. **In the Trace List View**:
   - Click functionality **Columns** button (usually top-right of the table).
   - Search for `mlflow.trace.metadata.*`.
   - Select columns like `mlflow.trace.metadata.bot_id`, `mlflow.trace.metadata.team_name`.
   - These will now appear as columns in your main trace list!

2. **In the Span Details (Right Panel)**:
   - Click on the **Root Span** (the top-most bar in the waterfall, usually named `Chat: {bot_id}` or `Create Bot: {name}`).
   - Look at the **Attributes** or **Tags** tab.
   - You will see all the custom metadata there:
     - `mlflow.trace.metadata.bot_id`
     - `mlflow.trace.metadata.question`
     - `mlflow.trace.inputs`
     - `mlflow.trace.outputs`

## 💼 For Stakeholders (Non-Technical View)

### What This Means for You

1. **Transparency**: You can see exactly what the system is doing
2. **Quality Assurance**: Every response gets scored on:
   - **Faithfulness**: Is the answer based on the documents?
   - **Relevance**: Does the answer address the question?
3. **Performance Monitoring**: Identify slow steps
4. **Debugging**: When something goes wrong, see exactly where

### Example: Understanding a Response

**Question**: "What are the benefits of MLflow?"

**Trace Shows You**:
1. **Memory**: Loaded 3 previous messages for context
2. **Retrieval**: Found 3 relevant documents in 0.8s
3. **Documents Preview**: 
   - "MLflow provides experiment tracking..."
   - "MLflow offers model registry..."
4. **Generation**: LLM took 2.1s to generate answer
5. **Quality Scores**:
   - Faithfulness: 0.92/1.0 ✅ (answer is based on docs)
   - Relevance: 0.88/1.0 ✅ (answer addresses question)
   - Average: 0.90/1.0 ✅ (high quality)

**Conclusion**: The system retrieved good documents and generated a high-quality, trustworthy answer.

## 🎨 Visual Trace Example

```
┌─────────────────────────────────────────────────────┐
│ Trace: RAG_Chat_Pipeline                            │
│ Duration: 3.5s                                      │
│ Status: ✅ OK                                       │
│ User: acme_support_bot                             │
│ Session: session_abc123                            │
└─────────────────────────────────────────────────────┘

Time →
0s      1s      2s      3s      4s
├───────┼───────┼───────┼───────┤
│ Memory│                         (0.05s)
        │ Retrieval              │ (0.8s) ← SLOW
        │ Chain│                  (0.02s)
        │      │ LLM Generation  │ (2.1s) ← SLOW
        │      │                 │ Eval│ (0.53s)

Insights:
🔴 Retrieval: 0.8s - Consider optimizing vector search
🔴 LLM: 2.1s - Normal for Gemini, consider caching
🟢 Evaluation: 0.53s - Good performance
```

## 🚀 Advanced Usage

### Finding Slow Requests
1. Go to MLflow UI → Traces
2. Sort by `execution_duration` (descending)
3. Identify traces taking >5 seconds
4. Drill into their spans to find bottlenecks

### Comparing Quality Over Time
1. Filter traces by date range
2. Export evaluation metrics:
   - `faithfulness_score`
   - `relevance_score`
   - `average_quality_score`
3. Plot trends to see if quality improves

### Debugging Failed Requests
1. Filter traces by `state: ERROR`
2. Check which span failed
3. View the error message and stack trace
4. See exact inputs that caused the failure

## 📊 Key Metrics to Monitor

### Performance Metrics
- **Total Response Time**: Target <3 seconds
- **Retrieval Time**: Target <1 second
- **LLM Generation Time**: Target <2 seconds
- **Evaluation Time**: Target <0.5 seconds

### Quality Metrics
- **Faithfulness Score**: Target >0.8
- **Relevance Score**: Target >0.8
- **Average Quality**: Target >0.8
- **Pass Rate**: % of responses with quality >0.7

### Volume Metrics
- **Requests per Hour**: Track usage
- **Error Rate**: Target <1%
- **Average Documents Retrieved**: Typically 3

## 🛠️ Troubleshooting

### Q: I don't see any traces
**A**: Make sure you're running your chat/bot service. Traces are only created when you make requests.

### Q: Traces are missing some spans
**A**: Check that `mlflow.gemini.autolog()` is enabled in `chatservice.py`

### Q: Evaluation scores are missing
**A**: Ensure evaluation is not failing silently. Check logs for evaluation errors.

### Q: Trace taking too long
**A**: Look at the waterfall diagram to identify which span is slow. Common culprits:
- **Retrieval**: Too many documents or slow vector DB
- **LLM**: Large prompts or complex generation
- **Evaluation**: Double LLM calls for scoring

## 📞 Support

For questions about tracing:
1. Check this guide first
2. Review MLflow official docs: https://mlflow.org/docs/latest/llms/tracing/index.html
3. Check trace metadata and span outputs for debugging

---

**Last Updated**: 2026-02-11
**Version**: 1.0
