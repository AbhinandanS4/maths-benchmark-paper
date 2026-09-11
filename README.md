# LLM Math Benchmark & RAG Suite
A modular, researcher-grade Python benchmarking toolkit built to evaluate open-source Large Language Models (LLMs) on complex mathematical problem-solving.

This framework systematically benchmarks model performance across 5 experimental conditions:

Chain of Thought (CoT): Baseline step-by-step reasoning.

Dense RAG: Vector search over textbook context using ChromaDB.

Hybrid RAG: BM25 sparse search fused with dense vector embeddings via Reciprocal Rank Fusion (RRF).

Tools Only: Python REPL code execution for algebraic and numeric verification (SymPy/NumPy).

RAG + Tools: Full retrieval-augmented generation paired with dynamic tool execution.

# System Architecture
The repository is organized into a clean 5-module structure to ensure modularity and reproducibility:

├── config.py             # Central configuration (Prompts, Tool Specs, Model Routing) 
├── tools.py              # Isolated Python REPL execution environment (SymPy / NumPy)
├── rag_dense.py          # PDF loading, chunking, and ChromaDB vector store management
├── rag_hybrid.py         # Custom math tokenizer + BM25 & Dense Reciprocal Rank Fusion (RRF)
├── evaluator.py          # Latex \boxed{} extraction and symbolic equivalence checker
├── main_benchmark.py     # Main benchmark orchestrator and latency/accuracy recorder
├── requirements.txt      # Project dependencies


# Benchmarking Assistant Setup Guide
Quick start guide for configuring and running the evaluation benchmarks.

# 1. Environment Setup
Create a .env file in the root directory (or update your existing one) and add your API key and LiteLLM settings:
# Your API Key for the LLM Provider
CUSTOM_API_KEY="your-api-key-here"

# (Optional) If using a custom base URL for LiteLLM
LITELLM_BASE_URL="https://your-custom-endpoint.com/v1"

# 2. Configure Model Target (config.py)
Open config.py and set your desired target model using the provider/model-name format expected by LiteLLM:

# Specify your target model. MUST start with the provider prefix (e.g., openai/, huggingface/, groq/, anthropic/)
MODEL_NAME = "huggingface/deepseek-ai/DeepSeek-V4.1-Flash:novita"

Common Provider Examples:

Hugging Face: huggingface/your-organization/your-model

OpenAI: openai/gpt-4o

Groq: groq/llama-3.3-70b-versatile

Anthropic: anthropic/claude-3-5-sonnet-20241022

# 3. Configure LiteLLM Client (main_benchmark.py)
Open main_benchmark.py to configure your API key, target model, and base URL overrides:

import os
import litellm
from dotenv import load_dotenv
from config import MODEL_NAME

# 1. Load environment variables
load_dotenv()

# 2. Assign API Key and optional Base URL
api_key = os.getenv("CUSTOM_API_KEY")
base_url = os.getenv("LITELLM_BASE_URL")  # Set if using a custom gateway/proxy

# 3. Configure LiteLLM parameters or call completion
response = litellm.completion(
    model=MODEL_NAME,
    api_key=api_key,
    base_url=base_url,  # Optional: pass base_url if connecting to a custom endpoint
    messages=[{"role": "user", "content": "Hello!"}]
)

# 4. Run the Benchmark
Execute the benchmark entry point directly from your terminal:

python3 main_benchmark.py
