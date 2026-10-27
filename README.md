# ⚖️ Indian AI Lawyer

A chatbot that answers questions about Indian law using only the text of the Acts. It finds the passages that match your question, hands them to an LLM, and replies with a short answer, the law it relies on, and the section. If the Acts don't cover the question, it says so and points you to a licensed advocate instead of guessing. I built it to learn how retrieval-augmented generation (RAG) works end to end, and how to test one honestly. It is general information, not legal advice.

🔗 Live demo: https://huggingface.co/spaces/nivedh1974/indian-ai-lawyer

## 📦 Technologies

- Python
- LangChain
- ChromaDB
- sentence-transformers (all-MiniLM-L6-v2)
- BM25 (optional keyword search)
- Groq (LLM, currently openai/gpt-oss-120b)
- Streamlit
- FastAPI
- HTML and CSS
- Docker
- Hugging Face Spaces

## 🦄 Features

Here's what you can do with Indian AI Lawyer:

**Ask in plain language:** Type a question like "What is the punishment for theft?" or "Can WhatsApp chats be used as evidence?" and get an answer in a fixed format: short answer, law, related provisions, procedure and a note.

**See the source:** Every answer names the Act and section and quotes the key wording. A sources panel lists the passages the model actually saw, so you can check them.

**Current criminal law:** Answers use the BNS, BNSS and BSA, not the old IPC, CrPC and Evidence Act.

**Honest refusals:** If the context doesn't support an answer, the app replies with one fixed sentence. It does not make something up. Questions about income tax, GST or "which lawyer should I hire" are refused on purpose.

**14 Acts covered:**

- Bharatiya Nyaya Sanhita (BNS)
- Bharatiya Nagarik Suraksha Sanhita (BNSS)
- Bharatiya Sakshya Adhiniyam (BSA)
- Constitution of India
- Indian Contract Act 1872
- Transfer of Property Act 1882
- Negotiable Instruments Act 1881
- Hindu Marriage Act 1955
- Protection of Women from Domestic Violence Act 2005
- Consumer Protection Act 2019
- Right to Information Act 2005
- Information Technology Act 2000
- Prevention of Corruption Act 1988
- Kerala Motor Vehicles Act

## 🎯 How it works

1. Load the Act PDFs and tag every page with its Act.
2. Split the text into chunks of 800 characters with 100 overlap (5,101 chunks).
3. Embed each chunk with all-MiniLM-L6-v2 and store it in ChromaDB.
4. Embed the question and fetch the closest chunks. Label each one with its Act and page, like [BNSS, p. 67].
5. Send the labelled chunks and the question to the LLM with a strict prompt: answer only from the context, name the Act exactly as labelled, refuse if the context falls short.
6. Show the answer and the sources.

## 👩🏽‍🍳 The Process

I started with the data. I loaded the PDFs, tagged each page with its Act and split them into chunks. Then I put the chunks in a vector database and wired up retrieval, so a question could return the closest passages.

Next I wrote the prompt. The model had to answer only from the context, use the new criminal laws, and refuse when it couldn't support an answer. I added a fixed refusal sentence so a refusal is always the same clean line.

After that I built the interface and deployed it on Hugging Face Spaces, with a sources panel so users can check the answer.

Then came the part I learned the most from: testing. I wrote 37 questions, scored the answers by hand, and ran the same questions through different settings, changing one thing at a time. I wrote a script that runs every question through the same code as the app and saves a CSV, so each run is repeatable.

Along the way I kept notes on what failed and why, and logged each problem as a bug in Jira. Writing it down changed how I saw the project. My first guess was that retrieval was the weak spot. The results said otherwise.

## 🧪 Test results

This is an internal developer test. There are no outside users, and the scores are my own reading of the answers, so treat them as provisional. Out of 37 questions, six are out of scope on purpose and should be refused.

| 37 questions | Run A (hybrid, k=8, FIR wording) | Test 1 (vector, k=4) | Test 2 (vector, k=8) | Test 3 (hybrid, k=8) |
|---|---|---|---|---|
| Correct | 19 | 16 | 14 | 14 |
| Borderline | 5 | 6 | 9 | 7 |
| Wrong | 2 | 2 | 3 | 4 |
| Declined (in scope) | 5 | 7 | 5 | 5 |
| Correct refusal | 6 | 6 | 6 | 5 |
| Error | 0 | 0 | 0 | 2 |

## 📚 What I Learned

**🧠 Test one thing at a time**

- My first v2 build changed several things at once, so when it scored well I couldn't say why. Running one change per test showed that keyword search (BM25) and a bigger k did not beat plain vector search.

**🔎 Where the errors really are**

- Most mistakes come when the model writes the citation, not when it finds the passage. It sometimes turns a page number into a section number, guesses an Act name from a raw file tag, or invents related provisions.
- Three questions fail in every run: how to file an FIR, whether a helmet is compulsory in Kerala, and the punishment under IPC section 302. They have three different causes: a retrieval miss, a model that declines despite the right chunks, and a missing old-to-new section mapping.

**📏 Measuring honestly**

- 37 questions is a small test. A swing of two or three answers can be noise. I score the answer and the law line, log wrong related provisions as a separate bug, and don't claim an accuracy number I can't back up.

**🏷️ Labels matter**

- The model can only name an Act correctly if the label says it. Raw file tags like "A187209" made it guess. Full Act names fixed several citations.

**⏱️ Free-tier limits are real**

- Groq's free tier allows 200,000 tokens a day, about two full test runs. Rate limits caused slow answers and a few errors, so I plan runs around the quota.

## ⚠️ Known limitations

- Answers can cite the wrong section number. Check the sources panel and the Act.
- Related provisions are sometimes invented.
- It doesn't map old IPC or CrPC section numbers to the new ones.
- Some in-scope questions are refused (FIR, Kerala helmet rule, defamation).
- It covers only the 14 Acts above. No case law, no later amendments, no rules or notifications.

## 💭 How can it be improved?

- Show full Act names in every label and word the section rule more softly, so the model cites a section when it knows it.
- Route each question to its Act before searching, and test that on fresh questions.
- Add an old-to-new section mapping (IPC to BNS, CrPC to BNSS).
- Fix the keyword search so it stops returning the same off-topic chunks, then retest hybrid search.
- Write a new set of unseen questions and check the answer key against the PDFs.
- Add more Acts and amendments.

## 🗂️ Project structure

- `backend/` API and retrieval code
- `frontend/` web interface
- `data/` Act PDFs
- `eval/` test questions and evaluation scripts
- `chunks.pkl` saved chunks used for keyword search
- `requirements.txt` Python packages

## 🚀 Run it locally

1. Clone the repo: `git clone https://github.com/nivedh1974-collab/AI-Lawyer-In.git`
2. Create a virtual environment and install the packages: `pip install -r requirements.txt`
3. Get a free Groq API key from console.groq.com and set it: `export GROQ_API_KEY=your_key`
4. Start the app.

Never commit your API key.

## 📜 Disclaimer

This project gives general information from public legal texts. It is not legal advice and does not replace a licensed advocate. Please don't enter personal details.

## 👋 Author

Nivedh Mohanan. Portfolio: https://nivedhmohanan.vercel.app
