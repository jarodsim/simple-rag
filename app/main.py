from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import get_answer

app = FastAPI(title="RAG com NVIDIA NIM")


class Query(BaseModel):
    question: str


@app.post("/query")
def query_rag(query: Query):
    answer = get_answer(query.question)
    return {"answer": answer}
