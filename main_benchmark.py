import os
import json
import time
from typing import Dict, Any, List
from dotenv import load_dotenv
import litellm

# Explicitly load .env file parameters
load_dotenv(".env")
litellm.api_base = "https://router.huggingface.co/v1"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
litellm.api_key = GROQ_API_KEY
# Import project modules
import config
from tools import execute_python_code
from rag_dense import DenseRAG
from rag_hybrid import HybridRAG
from evaluator import evaluate_solution

# LiteLLM configuration
litellm.drop_params = True

class MathBenchmarkRunner:
    def __init__(self, model_name: str, pdf_path: str):
        self.model_name = model_name
        
        print(f"[Runner] Loading Dense RAG index from: {pdf_path}")
        self.dense_rag = DenseRAG(pdf_path)
        
        print("[Runner] Building Hybrid RAG index...")
        self.hybrid_rag = HybridRAG(self.dense_rag)

    def run_cot(self, question: str) -> Dict[str, Any]:
        """Condition 1: Chain of Thought (Base Prompting)"""
        prompt = f"{config.PROMPTS['cot']}\n\nProblem: {question}"
        start = time.time()
        
        response = litellm.completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            api_base="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN"]
        )
        latency = time.time() - start
        output = response.choices[0].message.content or ""
        return {"output": output, "latency": latency}

    def run_rag_dense(self, question: str) -> Dict[str, Any]:
        """Condition 2: RAG Only (Dense Vector Search)"""
        context = self.dense_rag.retrieve(question, top_k=3)
        prompt = config.PROMPTS["rag"].format(context=context, question=question)
        start = time.time()
        
        response = litellm.completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            api_base="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN"]
        )
        latency = time.time() - start
        output = response.choices[0].message.content or ""
        return {"output": output, "latency": latency}

    def run_rag_hybrid(self, question: str) -> Dict[str, Any]:
        """Condition 3: RAG (Hybrid BM25 + Vector)"""
        context = self.hybrid_rag.retrieve(question, top_k=3)
        prompt = config.PROMPTS["rag"].format(context=context, question=question)
        start = time.time()
        
        response = litellm.completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            api_base="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN"]
        )
        latency = time.time() - start
        output = response.choices[0].message.content or ""
        return {"output": output, "latency": latency}

    def run_tools_only(self, question: str) -> Dict[str, Any]:
        """Condition 4: Tools Only (Python REPL)"""
        messages = [
            {"role": "system", "content": config.PROMPTS["tool"]},
            {"role": "user", "content": question}
        ]
        start = time.time()
        
        response = litellm.completion(
            model=self.model_name,
            messages=messages,
            tools=config.PYTHON_REPL_SPEC,
            tool_choice="auto",
            temperature=0.0,
            api_base="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN"]
        )
        msg = response.choices[0].message
        
        # Execute tool calls if requested by model
        if msg.get("tool_calls"):
            messages.append(msg)
            for tool_call in msg["tool_calls"]:
                try:
                    args = json.loads(tool_call.function.arguments)
                    code = args.get("code", "")
                except json.JSONDecodeError:
                    code = ""
                    
                exec_result = execute_python_code(code)
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": "execute_python_code",
                    "content": exec_result
                })
            
            final_res = litellm.completion(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                api_base="https://router.huggingface.co/v1",
                api_key=os.environ["HF_TOKEN"]
            )
            output = final_res.choices[0].message.content or ""
        else:
            output = msg.content or ""

        latency = time.time() - start
        return {"output": output, "latency": latency}

    def run_rag_plus_tools(self, question: str) -> Dict[str, Any]:
        """Condition 5: RAG + Tools (Full Augmentation)"""
        context = self.hybrid_rag.retrieve(question, top_k=3)
        system_msg = f"{config.PROMPTS['tool']}\n\nReference Context:\n{context}"
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question}
        ]
        start = time.time()
        
        response = litellm.completion(
            model=self.model_name,
            messages=messages,
            tools=config.PYTHON_REPL_SPEC,
            tool_choice="auto",
            temperature=0.0,
            api_base="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN"]
        )
        msg = response.choices[0].message
        
        if msg.get("tool_calls"):
            messages.append(msg)
            for tool_call in msg["tool_calls"]:
                try:
                    args = json.loads(tool_call.function.arguments)
                    code = args.get("code", "")
                except json.JSONDecodeError:
                    code = ""
                    
                exec_result = execute_python_code(code)
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": "execute_python_code",
                    "content": exec_result
                })
            
            final_res = litellm.completion(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                api_base="https://router.huggingface.co/v1",
                api_key=os.environ["HF_TOKEN"]
            )
            output = final_res.choices[0].message.content or ""
        else:
            output = msg.content or ""

        latency = time.time() - start
        return {"output": output, "latency": latency}


# ==========================================
# EXECUTION PIPELINE
# ==========================================

if __name__ == "__main__":
    PDF_PATH = "Class_11_Mathematics_Textbook.pdf"
    
    # Dataset item
    test_problem = {
        "question": "In an Arithmetic Progression, if the 5th term is 19 and the 11th term is 43, find the 20th term and the sum of the first 20 terms.",
        "answer": "\\boxed{S_{20} = 820, a_{20} = 79}"
    }

    runner = MathBenchmarkRunner(model_name=config.MODEL_NAME, pdf_path=PDF_PATH)

    print("\n--- Running Evaluation Conditions ---")
    
    eval_runs = {
        "1. CoT (Base)": runner.run_cot(test_problem["question"]),
        "2. RAG (Dense)": runner.run_rag_dense(test_problem["question"]),
        "3. RAG (Hybrid)": runner.run_rag_hybrid(test_problem["question"]),
        "4. Tools Only": runner.run_tools_only(test_problem["question"]),
        "5. RAG + Tools": runner.run_rag_plus_tools(test_problem["question"]),
    }

    # Summary table output
    print("\n" + "=" * 62)
    print(f"{'Condition':<20} | {'Correct':<8} | {'Latency':<10}")
    print("-" * 62)
    
    for condition_name, result in eval_runs.items():
        is_correct = evaluate_solution(result["output"], test_problem["answer"])
        print(f"{condition_name:<20} | {str(is_correct):<8} | {result['latency']:.2f}s")
        
    print("=" * 62)