import os

# Central model definition
MODEL_NAME = "huggingface/deepseek-ai/DeepSeek-V4.1-Flash:novita"  # Change this to your desired model


# System prompts for experimental conditions
PROMPTS = {
    "cot": (
        "Solve the following mathematical problem step by step.\n"
        "Show your logical steps clearly and place your final numeric or symbolic "
        "answer inside \\boxed{{}} at the very end."  # Fixed: Escaped {{}}
    ),
    "rag": (
        "You are given excerpts from a Class 11 Mathematics textbook for context.\n"
        "Textbook Context:\n{context}\n\n"
        "Problem: {question}\n\n"
        "Use the relevant formulas from the context to solve the problem step by step. "
        "Wrap your final answer inside \\boxed{{}}."  # Fixed: Escaped {{}}
    ),
    "tool": (
        "You are an expert mathematical assistant with access to a Python REPL tool.\n"
        "Use the `execute_python_code` tool for multi-step calculations, equation solving, "
        "or verifications. Always output the final answer inside \\boxed{{}}."  # Fixed: Escaped {{}}
    )
}

# Function spec for LiteLLM / OpenAI tool calling
PYTHON_REPL_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "execute_python_code",
            "description": "Executes a Python snippet using SymPy or NumPy to perform precise arithmetic or algebraic solving.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Assign the final answer to a local variable named 'result'."
                    }
                },
                "required": ["code"]
            }
        }
    }
]