from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chain import ask
import sys
import os

sys.path.append(os.path.dirname(__file__))

app = FastAPI(title="Indian AI Lawyer")

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── REQUEST MODEL ────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str

# ── ROUTES ───────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Indian AI Lawyer is running"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    answer = ask(request.question)
    return {"question": request.question, "answer": answer}