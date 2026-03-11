import os
from dotenv import load_dotenv
load_dotenv()
os.getenv("HF_TOKEN")

from langchain_classic.evaluation.qa import QAEvalChain

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

llm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b"))

eval_chain = QAEvalChain.from_llm(llm)

examples = [
    {
        "query": "What is the capital of France?",
        "answer": "Paris"
    }
]

predictions = [
    {
        "result": "Paris"
    }
]

graded_outputs = eval_chain.evaluate(
    examples,
    predictions
)

print(graded_outputs)