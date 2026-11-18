from fastapi import FastAPI
from pydantic import BaseModel
from Rag import retriever, llm   # change to 'rag' if your file is rag.py

app = FastAPI()

# Request model
class QueryRequest(BaseModel):
    query: str


# Home route
@app.get("/")
def home():
    return {"message": "RAG API is running"}


# Ask route
@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        query = request.query
        print("Query:", query)

        # Retrieve documents
        retrieved_docs = retriever.invoke(query)

        # Create context
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        print("Context:", context[:200])

        # Improved prompt
        prompt = f"""
You are a helpful IT support assistant.

Based on the context, give a clear and short solution.

Context:
{context}

User Issue:
{query}

Solution:
"""

        # LLM response
        response = llm.invoke(prompt)
        answer = str(response).strip()

        print("RAW RESPONSE:", answer)

        # Clean output
        if "Solution:" in answer:
            answer = answer.split("Solution:")[-1].strip()

        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()

        # Fallback if empty
        if not answer:
            answer = "Please reset your password and clear browser cache."

        print("FINAL ANSWER:", answer)

        return {"answer": answer}

    except Exception as e:
        print("ERROR:", e)
        return {"error": str(e)}