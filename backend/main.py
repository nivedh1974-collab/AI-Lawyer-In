# ═══ 1. IMPORTS ═══════════════════════════════════════════════
from fastapi import FastAPI                              # the web framework that creates the server
from fastapi.middleware.cors import CORSMiddleware       # controls which websites may call the server
from pydantic import BaseModel                           # defines and checks the shape of incoming data
from chain import ask                                    # your RAG function from chain.py (question in, answer out)
import sys
import os

sys.path.append(os.path.dirname(__file__))               # lets Python find chain.py in this folder, wherever you launch from

# ═══ 2. CREATE THE APP ════════════════════════════════════════
app = FastAPI(title="Indian AI Lawyer")                  # the server object; the title shows in the auto-generated /docs page

# ═══ 3. CORS (who may call this server) ═══════════════════════
# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                                 # any website may call it (fine for a demo, too open for production)
    allow_methods=["*"],                                 # any HTTP method
    allow_headers=["*"],                                 # any header
)

# ═══ 4. REQUEST SHAPE ═════════════════════════════════════════
class QuestionRequest(BaseModel):
    question: str                                        # the body must be {"question": "..."}; anything else is rejected automatically

# ═══ 5. ROUTES (the two URLs the server offers) ═══════════════
@app.get("/")                                            # visiting the plain address
def root():
    return {"status": "Indian AI Lawyer is running"}     # a health check: confirms the server is up

@app.post("/ask")                                        # the real endpoint
def ask_question(request: QuestionRequest):
    answer = ask(request.question)                       # runs the whole RAG chain
    return {"question": request.question, "answer": answer}   # sends back the question and the answer as JSON
