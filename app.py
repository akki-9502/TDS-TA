import os
import json
import logging
import sqlite3
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import aiohttp
from dotenv import load_dotenv

load_dotenv()

# AI Proxy configuration
RAW_TOKEN = os.getenv("AIPROXY_TOKEN")  # should be just the raw token in .env
AIPROXY_TOKEN = f"Bearer {RAW_TOKEN}"

AIPROXY_URL = "https://aiproxy.sanand.workers.dev/openai"
DB_PATH = "knowledge_base.db"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_URL = f"{AIPROXY_URL}/v1/embeddings"
COMPLETION_URL = f"{AIPROXY_URL}/v1/chat/completions"

# Logging
logging.basicConfig(level=logging.INFO)

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response schemas
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

class Link(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[Link]

# Cosine similarity
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Get embedding from AI proxy
async def get_embedding(text):
    headers = {
        "Authorization": AIPROXY_TOKEN,
        "Content-Type": "application/json"
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(EMBEDDING_URL, headers=headers, json=payload) as response:
            data = await response.json()
            print("🔴 Embedding API raw response:", data)
            if "data" not in data:
                raise ValueError(f"Embedding API failed: {data}")
            return data["data"][0]["embedding"]

# Find top similar chunks from DB
async def find_similar_content(query_embedding, conn, top_k=5):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM markdown_chunks WHERE embedding IS NOT NULL")
    rows = cursor.fetchall()
    scored_chunks = []
    for row in rows:
        _, url, content, embedding_json, _ = row
        embedding = json.loads(embedding_json)
        sim = cosine_similarity(query_embedding, embedding)
        scored_chunks.append((sim, url, content))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return scored_chunks[:top_k]

# Call GPT model via AI proxy with context
async def query_openai_with_context(context, question):
    headers = {
        "Authorization": AIPROXY_TOKEN,
        "Content-Type": "application/json"
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful virtual TA for the TDS course. Use only the provided context. "
                "Always respond in the following JSON format:\n\n"
                "{\n"
                "  \"answer\": \"<concise answer — ideally under 3 lines>\",\n"
                "  \"links\": [\n"
                "    {\"url\": \"<link_url>\", \"text\": \"<short description>\"},\n"
                "    ... up to 2 links\n"
                "  ]\n"
                "}\n\n"
                "Do NOT include explanations, markdown, or extra text outside this JSON format. "
                "If no links are relevant, return an empty links list. Keep the answer short and specific."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]

    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": 0.2
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(COMPLETION_URL, headers=headers, json=payload) as response:
            result = await response.json()
            return result["choices"][0]["message"]["content"]

# Parse GPT response into structured format
def clean_gpt_response(text: str) -> dict:
    try:
        if text.strip().startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip()
        if text.strip().startswith("{") and text.strip().endswith("}"):
            parsed = json.loads(text)
            for link in parsed.get("links", []):
                if not link.get("text"):
                    link["text"] = link.get("url", "Reference")
            return {
                "answer": parsed.get("answer", "").strip(),
                "links": parsed.get("links", [])
            }
    except Exception:
        logging.warning("Failed to parse GPT response as JSON", exc_info=True)

    return {
        "answer": text.strip() or "⚠️ No answer generated.",
        "links": []
    }

# Main API route
@app.post("/api/", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    try:
        logging.info(f"🔍 Received question: {request.question}")
        if request.image:
            logging.info(f"🖼️ Received base64 image (length {len(request.image)})")

        query_embedding = await get_embedding(request.question)
        conn = sqlite3.connect(DB_PATH)
        top_chunks = await find_similar_content(query_embedding, conn)
        context = "\n\n".join(chunk[2] for chunk in top_chunks)

        fallback_links = []
        seen_urls = set()
        for chunk in top_chunks:
            url = chunk[1]
            if "discourse.onlinedegree.iitm.ac.in" in url and url not in seen_urls:
                seen_urls.add(url)
                fallback_links.append({"url": url, "text": "Refer to this related discussion."})
            if len(fallback_links) == 2:
                break

        raw_answer = await query_openai_with_context(context, request.question)
        print("\n🔹 Raw LLM response:\n", raw_answer)

        parsed = clean_gpt_response(raw_answer)
        if not parsed.get("answer"):
            parsed["answer"] = "⚠️ No answer generated."

        links = parsed.get("links", []) if parsed.get("links") else fallback_links

        response = {
            "answer": parsed["answer"],
            "links": links
        }

        print("\n✅ Final API response:\n", json.dumps(response, indent=2))
        return QueryResponse(answer=response["answer"], links=response["links"])

    except Exception:
        logging.error("Error in /api/", exc_info=True)
        return QueryResponse(answer="⚠️ Failed to get an answer.", links=[])
