import os
import re
import json
import time
import random
import logging
from typing import List, Optional, Dict, Any, TypedDict
import numpy as np
from datetime import datetime
import requests

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from rank_bm25 import BM25Okapi
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# -----------------------------------------------
# Configuration
# -----------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
DOC_SUMMARY_PATH = os.path.join(VECTORSTORE_DIR, "doc_summary.json")
EMBEDDING_MODEL = os.path.join(BASE_DIR, "all-mini_v12")
MODEL_PATH = os.path.join(BASE_DIR, "qwen_4b")
TOP_K_CHUNKS = 6
TOP_K_DOCS = 3

# AUTH_API_URL = "http://192.168.1.52:4050"
# NUMBER_CHECK_API = (
#     f"{AUTH_API_URL}/auth/chat/mob_verification"  # API 1: DB check
# )
# AUTH_API_URL = "https://d37a4cpfzgh2bx.cloudfront.net/mca-api"    #Stagging
# AUTH_API_URL = "https://pminternship.mca.gov.in/mca-api"  # Deployement
AUTH_API_URL = "http://localhost:8900"
NUMBER_CHECK_API = f"{AUTH_API_URL}/auth/chat/mob_verification"
SIGN_IN_URL = f"{AUTH_API_URL}/auth/chat/sign-in"  # API 2: OTP verify
CANDIDATE_URL = f"{AUTH_API_URL}/chatbot/candidate_details_internship"  # API 3
FP = "test-device"
OTP_EXPIRY_SECONDS = 300
SESSION_EXPIRY_SECONDS = 3600
MAX_CONVERSATION_MEMORY = 25

print(f"[INFO] Vectorstore directory: {VECTORSTORE_DIR}")

# RAG_PROMPT = """<|system|>
# You are an expert in answering questions related to the program.
# Given the context and the question, please provide a clear, concise, and accurate answer.
# You are a helpful assistant that answers based only on the provided context.
# Use {current_datetime} for age related questions.
# Avoid repeating phrases, emojis and provide a complete, single answer.
# Do not repeate the answer and also do not give tags at any cost like </|assistant|>, </endl>, "Final note", "Answer ends here", "End of output", "clearly and concisely." in your answer.
# </|system|>
# <|user|>
# Context:
# {context}

# Question:
# {question}
# </|user|>

# """

# RAG_PROMPT = """You are a helpful assistant for the program.
# Answer the question using ONLY the context below. Be direct and natural.
# Do not add summaries, closing lines, or extra commentary.
# Use {current_datetime} only when the question involves age or dates.
# Avoid repeating phrases, emojis and provide a complete, single answer.
# Context:
# {context}

# Question:
# {question}

# Answer:"""

"""
best 23 march 2026
"""
# RAG_PROMPT = """<|im_start|>system
# You are a helpful assistant for the program.
# Answer only from the given context. Be brief and natural.
# Do not add closing remarks, summaries, or meta-commentary.
# Use {current_datetime} only for age or date-related questions.
# Avoid repeating phrases, emojis and provide a complete, single answer.
# <|im_end|>
# <|im_start|>user
# Context:
# {context}

# Question:
# {question}
# <|im_end|>
# <|im_start|>assistant
# """

RAG_PROMPT = """<|im_start|>system
You are a helpful assistant for The Internship Program.
Answer only from the given context. Be brief and natural.
If answer is not in Context, just say that I don't have information reagrding this, Please try with a different question, don't try to make up an answer.
Do not add closing remarks, summaries, or meta-commentary.
If the user input contains only symbols, punctuation, emojis, or signs with no real text or numbers or valid question, respond only with: "Please ask a valid question."
Use {current_datetime} only for age or date-related questions.
Avoid repeating phrases, emojis and provide a complete, single answer.
<|im_end|>
<|im_start|>user
Context:
{context}

Question:
{question}
<|im_end|>
<|im_start|>assistant
"""

logger = logging.getLogger("api")
logging.basicConfig(level=logging.WARNING)  # suppress INFO from terminal
# Keep full INFO logging for uvicorn / file handlers if configured
logger.setLevel(logging.INFO)

# -----------------------------------------------
# FastAPI App
# -----------------------------------------------
app = FastAPI(title="RAG API with OTP Authentication", version="5.0")

import os
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# -----------------------------------------------
# In-Memory Storage
# -----------------------------------------------
faiss_db = None
doc_summary_store = {}
embedder = None
bm25_index = None
bm25_corpus_texts = []
bm25_docs_metadata = []
model = None
tokenizer = None
device = None

# thread_id -> list of {"role", "content", "timestamp"}
conversation_memory: Dict[str, List[Dict[str, str]]] = {}

# thread_id -> {"is_authenticated", "awaiting_phone", "awaiting_otp",
#               "phone_number", "user_id", "created_at", "last_activity"}
sessions: Dict[str, Dict[str, Any]] = {}

# thread_id -> {"data": user_data_dict, "cached_at": timestamp}
user_cache: Dict[str, Dict[str, Any]] = {}

# phone_number -> {"otp": str, "generated_at": float, "attempts": int}
otp_store: Dict[str, Dict[str, Any]] = {}

# thread_id -> pending query string (saved while user was not logged in)
pending_queries: Dict[str, str] = {}


# -----------------------------------------------
# LLM Helper
# -----------------------------------------------
def call_llm(prompt: str, max_tokens: int = 1024, temperature: float = 0.1) -> str:
    """Call the local Qwen LLM and return generated text."""
    if model is None or tokenizer is None:
        return ""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
                # stop_sequences=["<|im_end|>", "<|im_start|>", "Final note", "Answer ends here", "End of output", "</|assistant|>", "</endl>", "clearly and concisely."]
            )
        # ✅ FIX
        input_length = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][
            input_length - 1 :
        ]  # include the last token of the input to avoid losing the first word if the model starts generating immediately after the prompt
        # new_tokens = output_ids[0][input_length:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return ""


# -----------------------------------------------
# Date & Time Tool
# -----------------------------------------------
class DateTimeTool:
    """Provides current date/time information."""

    @staticmethod
    def get_current_datetime() -> Dict[str, Any]:
        now = datetime.now()
        return {
            "current_datetime": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day_name": now.strftime("%A"),
            "month_name": now.strftime("%B"),
            "year": now.year,
            "formatted": now.strftime("%B %d, %Y at %I:%M %p"),
        }

    @staticmethod
    def check_date_query(query: str) -> bool:
        """Check if the query is asking about date/time."""
        date_keywords = [
            "today",
            "date",
            "time",
            "when",
            "day",
            "month",
            "year",
            "current",
            "now",
            "what day",
            "what time",
            "what date",
        ]
        return any(kw in query.lower() for kw in date_keywords)


datetime_tool = DateTimeTool()


def _lookup_greeting(text: str) -> Optional[str]:
    """
    Check full text against greeting cache (longest match first, word-boundary safe).
    Returns the canned reply string, or None if not a greeting.
    """
    normalized = text.strip().lower()
    if normalized in GREETING_CACHE:
        return GREETING_CACHE[normalized]
    # Prefix match — sorted by length descending so "thank you so much" beats "thank you"
    for phrase, reply in sorted(GREETING_CACHE.items(), key=lambda x: -len(x[0])):
        if normalized.startswith(phrase):
            rest = normalized[len(phrase):]
            # Must be end-of-string OR followed by non-letter (word boundary)
            if not rest or not rest[0].isalpha():
                return reply
    return None


def validate_query(query: str) -> Dict[str, Any]:
    """
    Validate a raw query string BEFORE it enters the LangGraph pipeline.

    Returns a dict with:
      skip_graph  : bool   — True means return immediately without hitting the graph
      response    : str    — pre-built reply to send back (only when skip_graph=True)

    Decision table:
      Empty / whitespace only          → "Please ask a valid question."
      Symbols / emojis / punct only    → "Please ask a valid question."
      Matches greeting cache           → canned greeting reply
      Single real word (non-greeting)  → "Please elaborate your question."
      Two or more real words           → valid, pass to graph
    """
    stripped = query.strip()

    # 1. Empty / blank
    if not stripped:
        return {"skip_graph": True, "response": "Please ask a valid question."}

    # 2. No real alphanumeric content at all (only symbols / emojis / punctuation)
    if not _REAL_CONTENT_RE.search(stripped):
        return {"skip_graph": True, "response": "Please ask a valid question."}

    # 3. Greeting / appreciation cache (checked BEFORE single-word gate)
    greeting_reply = _lookup_greeting(stripped)
    if greeting_reply:
        return {"skip_graph": True, "response": greeting_reply}

    # 4. Single real word (non-greeting) → ask to elaborate
    real_words = [w for w in stripped.split() if _REAL_CONTENT_RE.search(w)]
    if len(real_words) == 1:
        return {"skip_graph": True, "response": "Please elaborate your question."}

    # 5. Valid query — pass to graph
    return {"skip_graph": False, "response": ""}


# -----------------------------------------------
# Intent Classification Tool
# -----------------------------------------------
class IntentTool:
    """Classifies whether a query is general or user-specific using the LLM."""

    @staticmethod
    def classify_intent(query: str) -> Dict[str, Any]:
        # - "What is my application status?" -> USER_SPECIFIC
        # - "My stipend amount?" -> USER_SPECIFIC
        # - "Which company am I assigned to?" -> USER_SPECIFIC

        # Respond ONLY with valid JSON (no extra text):
        # {{"intent": "GENERAL or USER_SPECIFIC", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
        # </|system|>
        # <|user|>
        # Query: {query}
        # </|user|>
        # <|assistant|>
        # """

        prompt = f"""<|im_start|>system
        You are an intent classifier for chatbot.
        Classify the query into exactly one of: GENERAL or USER_SPECIFIC or NOT_VALID_QUERY

        GENERAL = Questions about the program in general (eligibility, process, guidelines, dates, deadlines)
        USER_SPECIFIC = Questions about a specific user's personal data (their application status, internship, company assigned, stipend, mentor)
        NOT_VALID_QUERY = Input that contains only symbols, punctuation, emojis, signs (@#$%^&*!?.,=+-), or is empty/blank with no real words or numbers. Greetings like hi, hello, bye, thank you, good morning are VALID and should be GENERAL, not NOT_VALID_QUERY.
        
        Examples:
        - "What is the program?" -> GENERAL
        - "When is the application deadline?" -> GENERAL
        - "Tell me about Internship" -> GENERAL
        - "Internship" -> NOT_VALID_QUERY
        - "intern" -> NOT_VALID_QUERY
        - "hi" -> GENERAL
        - "hello" -> GENERAL  
        - "thank you" -> GENERAL
        - "good morning" -> GENERAL
        - "bye" -> GENERAL
        - "What is my application status?" -> USER_SPECIFIC
        - "My stipend amount?" -> USER_SPECIFIC
        - "Which company am I assigned to?" -> USER_SPECIFIC
        - "???" -> NOT_VALID_QUERY
        - "!!!" -> NOT_VALID_QUERY
        - "@#$%" -> NOT_VALID_QUERY
        - "😀🔥" -> NOT_VALID_QUERY
        - "..." -> NOT_VALID_QUERY
        - "   " -> NOT_VALID_QUERY

        Respond ONLY with valid JSON (no extra text):
        {{"intent": "GENERAL or USER_SPECIFIC or NOT_VALID_QUERY", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
        <|im_end|>
        <|im_start|>user
        Query: {query}
        <|im_end|>
        <|im_start|>assistant
        """
        response = call_llm(prompt, max_tokens=1024, temperature=0.1)

        try:
                data = json.loads(response)
                intent = str(data.get("intent", "GENERAL")).upper()

                # ── Map intent string to flags ────────────────────────────────────
                if "NOT_VALID" in intent or "NOT VALID" in intent:
                    return {
                        "is_user_specific": False,
                        "intent": "not_valid_query",
                        "confidence": data.get("confidence", 0.9),
                        "reasoning": data.get("reasoning", "LLM: not valid query"),
                    }

                is_user_specific = "USER" in intent or "SPECIFIC" in intent
                return {
                    "is_user_specific": is_user_specific,
                    "intent": "user_specific" if is_user_specific else "general",
                    "confidence": data.get("confidence", 0.8),
                    "reasoning": data.get("reasoning", "LLM classification"),
                }

        except Exception:
                # Heuristic fallback
                q = f" {query.lower()} "
                # Check if query has any real alphanumeric content
                has_real = bool(re.search(r"[a-zA-Z0-9\u0900-\u097F]", query))
                if not has_real:
                    return {
                        "is_user_specific": False,
                        "intent": "not_valid_query",
                        "confidence": 0.9,
                        "reasoning": "Heuristic: no real alphanumeric content found",
                    }
                is_user_specific = " my " in q or query.lower().startswith("my ")
                return {
                    "is_user_specific": is_user_specific,
                    "intent": "user_specific" if is_user_specific else "general",
                    "confidence": 0.6,
                    "reasoning": "Heuristic fallback",
                }


intent_tool = IntentTool()


# -----------------------------------------------
# Session Management Tool
# -----------------------------------------------
class SessionTool:
    """Manages user sessions stored in the in-memory `sessions` dict."""

    @staticmethod
    def get_or_create_session(thread_id: str) -> Dict[str, Any]:
        """Return existing session or create a new one."""
        if thread_id not in sessions:
            sessions[thread_id] = {
                "thread_id": thread_id,
                "created_at": time.time(),
                "last_activity": time.time(),
                "is_authenticated": False,
                "awaiting_phone": False,
                "awaiting_otp": False,
                "phone_number": None,
                "user_id": None,
            }
            logger.info(f"New session created: {thread_id}")
        else:
            sessions[thread_id]["last_activity"] = time.time()
        return sessions[thread_id]

    @staticmethod
    def update_session(thread_id: str, **kwargs):
        """Update specific fields in a session."""
        if thread_id in sessions:
            sessions[thread_id].update(kwargs)
            sessions[thread_id]["last_activity"] = time.time()

    @staticmethod
    def is_session_valid(thread_id: str) -> bool:
        """Return True if the session exists and has not expired."""
        if thread_id not in sessions:
            return False
        session = sessions[thread_id]
        elapsed = time.time() - session.get("last_activity", 0)
        if elapsed > SESSION_EXPIRY_SECONDS:
            logger.info(f"Session expired: {thread_id}")
            # Reset auth so the user has to log in again
            sessions[thread_id]["is_authenticated"] = False
            sessions[thread_id]["awaiting_phone"] = False
            sessions[thread_id]["awaiting_otp"] = False
            return False
        return True

    @staticmethod
    def get_cached_user_data(thread_id: str) -> Optional[Dict[str, Any]]:
        """Return cached user data if it exists and is not too old."""
        if thread_id in user_cache:
            cached = user_cache[thread_id]
            if time.time() - cached.get("cached_at", 0) < SESSION_EXPIRY_SECONDS:
                return cached.get("data")
        return None

    @staticmethod
    def cache_user_data(thread_id: str, user_data: Dict[str, Any]):
        """Store user data in the cache keyed by thread_id."""
        user_cache[thread_id] = {"data": user_data, "cached_at": time.time()}


session_tool = SessionTool()


# -----------------------------------------------
# Authentication Tool
# -----------------------------------------------
class AuthTool:
    """Handles the 3-API auth flow."""

    # ── API 1: check if number is in DB ──────────────────────────────────────
    @staticmethod
    def check_number_in_db(mobile: str) -> Dict[str, Any]:
        """
        POST http://192.168.1.52:4050/auth/chat/mob_verification
        Body: {"mobile": "<10-digit number>"}
        Returns {"present": true/false}.
        Called BEFORE asking for OTP so unregistered numbers are rejected early.
        """
        try:
            resp = requests.post(
                NUMBER_CHECK_API,
                headers={"Content-Type": "application/json"},
                json={"mobile": mobile},
                timeout=5,
            )

            # API returns HTTP 400 with JSON body for invalid/unregistered numbers
            # API returns HTTP 200 with {"code": true, "message": "OTP sent successfully."} for valid numbers
            if resp.status_code == 400:
                data = resp.json()
                error_msg = data.get("error", "Invalid mobile number")
                logger.info(f"API 1 – mobile={mobile} not in DB: {error_msg}")
                return {"success": True, "present": False, "mobile": mobile}

            resp.raise_for_status()
            data = resp.json()
            # Check the "code" field: true = number found in DB and OTP sent
            present = bool(data.get("code", False))
            logger.info(f"API 1 – mobile={mobile} present={present}, response={data}")
            return {"success": True, "present": present, "mobile": mobile}

        except requests.exceptions.ConnectionError:
            logger.error(f"API 1 – Cannot connect to {NUMBER_CHECK_API}")
            return {
                "success": False,
                "present": False,
                "error": "Number-check service is unreachable (192.168.1.52:4050).",
            }
        except requests.exceptions.Timeout:
            logger.error("API 1 – Timeout")
            return {
                "success": False,
                "present": False,
                "error": "Number-check service timed out.",
            }
        except Exception as e:
            logger.error(f"API 1 – Error: {e}")
            return {"success": False, "present": False, "error": str(e)}

    # ── Register phone for OTP (pre-API-2 step) ───────────────────────────────
    @staticmethod
    def register_phone_for_otp(phone_number: str):
        """Store phone in otp_store and mark it as waiting for OTP input."""
        otp_store[phone_number] = {
            "generated_at": time.time(),
            "attempts": 0,
        }
        logger.info(f"Phone {phone_number} registered – awaiting OTP entry")

    # ── API 2: verify OTP ─────────────────────────────────────────────────────
    @staticmethod
    def verify_otp(phone_number: str, otp: str) -> Dict[str, Any]:
        """
        POST http://192.168.1.52:4050/auth/chat/sign-in
        body: {"mobile": "...", "otp": "...", "fp": "test-device"}
        Returns JWT token on success.
        """
        if phone_number not in otp_store:
            return {
                "success": False,
                "error": "Session expired. Please enter your phone number again.",
            }

        stored = otp_store[phone_number]

        if time.time() - stored["generated_at"] > OTP_EXPIRY_SECONDS:
            del otp_store[phone_number]
            return {
                "success": False,
                "error": "OTP timed out. Please enter your phone number again.",
            }

        stored["attempts"] += 1
        if stored["attempts"] > 3:
            del otp_store[phone_number]
            return {
                "success": False,
                "error": "Too many incorrect attempts. Please enter your phone number again.",
            }

        try:
            body = {"mobile": phone_number, "otp": otp, "fp": FP}
            resp = requests.post(
                SIGN_IN_URL,
                headers={"Content-Type": "application/json"},
                json=body,
                timeout=10,
            )
            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": f"Login failed (HTTP {resp.status_code}). Check your OTP.",
                }

            token = resp.json().get("token")
            if not token:
                return {
                    "success": False,
                    "error": "Auth API returned no token. Please try again.",
                }

            del otp_store[phone_number]
            logger.info(f"API 2 – Login OK for {phone_number}")
            return {"success": True, "token": token}

        except requests.exceptions.Timeout:
            return {"success": False, "error": "Login API timed out. Please try again."}
        except Exception as e:
            logger.error(f"API 2 – Error: {e}")
            return {"success": False, "error": str(e)}


auth_tool = AuthTool()


# -----------------------------------------------
# Sub-Intent Classification
# -----------------------------------------------

SUB_INTENTS = [
    "INTERNSHIP_COUNT",
    "INTERNSHIP_LIST",
    "INTERNSHIP_STATUS",
    "INTERNSHIP_DATES",
    "INTERNSHIP_STIPEND",
    "INTERNSHIP_DURATION",
    "DBT_COUNT",
    "DBT_MONTHS",
    "DBT_LAST_PAYMENT",
    "DBT_TOTAL_AMOUNT",
    "DBT_DETAILS",
    "GRIEVANCE_COUNT",
    "GRIEVANCE_STATUS",
    "GRIEVANCE_LIST",
    "COMPLAINT_COUNT",
    "COMPLAINT_STATUS",
    "COMPLAINT_LIST",
    "CANDIDATE_PROFILE",
    "ALL_DATA",
]

SUB_INTENT_EXAMPLES = """
INTERNSHIP_COUNT    : "how many internships did I apply to", "kitni internship apply ki", "total internships"
INTERNSHIP_LIST     : "which internships did I apply to", "kis kis internship mein apply kiya", "internship ke naam batao"
INTERNSHIP_STATUS   : "what is my internship status", "internship ka status", "which ones are completed", "konsa active hai"
INTERNSHIP_DATES    : "internship start date kab hai", "when does internship end", "start aur end date"
INTERNSHIP_STIPEND  : "kitna stipend milta hai", "what is my stipend", "salary per month"
INTERNSHIP_DURATION : "internship kitne din ki hai", "how long is my internship", "kitne mahine ki internship", "internship duration"
DBT_COUNT           : "how many payments did I receive", "kitne payment mila", "payments count"
DBT_MONTHS          : "kitne mahine ka payment mila", "how many months of stipend", "months of payment"
DBT_LAST_PAYMENT    : "last payment kab aya", "when was my last payment", "latest payment date"
DBT_TOTAL_AMOUNT    : "kitna total paisa mila", "total amount received", "total stipend amount"
DBT_DETAILS         : "show my payments", "payment details dikhao", "payment ka detail"
GRIEVANCE_COUNT     : "kitni grievance hai", "how many grievances", "complaint count"
GRIEVANCE_STATUS    : "grievance status kya hai", "complaint resolved hai kya"
GRIEVANCE_LIST      : "grievances dikhao", "show my complaints", "list grievances"
COMPLAINT_COUNT     : "kitni complaints hai", "how many complaints", "grievance count"
COMPLAINT_STATUS    : "complaint status kya hai", "grievance resolved hai kya"
COMPLAINT_LIST      : "complaints dikhao", "show my complaints", "list complaints"
CANDIDATE_PROFILE   : "mera profile", "personal details", "my information", "candidate details"
ALL_DATA            : "sara data dikha", "show everything", "full details", "all my data"
"""


def classify_sub_intent(query: str, conversation_context: str = "") -> str:
    """
    Step 1 LLM call: classify the user query into one of the predefined sub-intents.
    Falls back to heuristic keyword matching if the LLM is not loaded or fails.
    """
    if model is not None:
        ctx_section = (
            f"\nConversation context:\n{conversation_context}\n"
            if conversation_context
            else ""
        )
        #         prompt = f"""<|system|>
        # You are a sub-intent classifier for a chatbot.
        # The user is already authenticated. Classify their query into exactly one of these sub-intents:

        # {chr(10).join(SUB_INTENTS)}

        # Examples:
        # {SUB_INTENT_EXAMPLES}

        # Rules:
        # - If the user asks about duration or length of internship, use INTERNSHIP_DURATION (calculate from DBT payment dates).
        # - If the user asks for a count, use the COUNT variant.
        # - If the user asks to list or name things, use the LIST variant.
        # - If the user asks about payment months or last payment, use the DBT_ variants.
        # - If unclear, prefer the most specific match.
        # - Respond ONLY with valid JSON, no extra text:
        #   {{"sub_intent": "ONE_OF_THE_INTENTS_ABOVE", "confidence": 0.0-1.0}}
        # </|system|>
        # <|user|>
        # {ctx_section}
        # Query: {query}
        # </|user|>
        # <|assistant|>
        # """

        prompt = f"""<|im_start|>system
        You are a sub-intent classifier for a chatbot assistant.
        The user is already authenticated. Classify their query into exactly one of these sub-intents:

        {chr(10).join(SUB_INTENTS)}

        Examples:
        {SUB_INTENT_EXAMPLES}

        Rules:
        - If the user asks about duration or length of internship, use INTERNSHIP_DURATION (calculate from DBT payment dates).
        - If the user asks for a count, use the COUNT variant.
        - If the user asks to list or name things, use the LIST variant.
        - If the user asks about payment months or last payment, use the DBT_ variants.
        - If unclear, prefer the most specific match.
        - Respond ONLY with valid JSON, no extra text:
        {{"sub_intent": "ONE_OF_THE_INTENTS_ABOVE", "confidence": 0.0-1.0}}
        <|im_end|>
        <|im_start|>user
        {ctx_section}
        Query: {query}
        <|im_end|>
        <|im_start|>assistant
        """
        response = call_llm(prompt, max_tokens=1024, temperature=0.0)
        logger.info(f"[SUB-INTENT LLM] raw: {response!r}")

        try:
            clean = response.strip()
            if clean.startswith("```"):
                clean = re.sub(r"```[a-z]*", "", clean).strip("` \n")
            data = json.loads(clean)
            intent = str(data.get("sub_intent", "")).strip().upper()
            if intent in SUB_INTENTS:
                logger.info(
                    f"[SUB-INTENT LLM] classified: {intent} conf={data.get('confidence')}"
                )
                return intent
        except Exception as e:
            logger.warning(f"[SUB-INTENT LLM] parse failed: {e}. Using heuristic.")

    return _heuristic_sub_intent(query)


def _heuristic_sub_intent(query: str) -> str:
    """
    Heuristic fallback: keyword-based sub-intent detection.
    Covers English, Hindi, and Hinglish phrasing.
    """
    q = query.lower()

    dbt_words = [
        "dbt",
        "pfms",
        "payment",
        "paisa",
        "stipend status",
        "stipend info",
        "amount",
        "transfer",
        "credit",
        "bank",
        "stipend",
        "stipend received",
        "stipend payment",
        "stipend amount",
        "stipend mila",
        "stipend get",
        "stipend",
        "fund",
        "fund transfer",
        "fund received",
        "fund amount",
        "dbt mila",
        "dbt payment",
        "dbt amount",
        "dbt received",
        "dbt transfer",
        "dbt details",
        "dbt status",
        "dbt kab mila",
        "dbt kab aaya",
        "paise kab aya",
    ]
    grievance_words = [
        "grievance",
        "complaint",
        "issue",
        "problem",
        "dispute",
        "ticket",
        "appeal",
        "concern",
        "report",
        "feedback",
        "samasya",
        "shikayat",
        "arj",
        "grievance status",
        "complaint status",
        "grievance count",
        "complaint count",
        "grievance list",
        "complaint list",
        "grievances dikhao",
        "complaints dikhao",
        "grievences batao",
        "complaints batao",
        "grievance details",
        "complaint details",
        "grievance status kya hai",
        "complaint status kya hai",
    ]
    profile_words = [
        "profile",
        "personal",
        "candidate",
        "details",
        "info",
        "name",
        "address",
        "education",
    ]
    all_words = ["all", "everything", "sara", "poora", "full", "summary", "overview"]
    count_words = ["how many", "kitni", "kitna", "count", "total", "number of"]
    list_words = [
        "list",
        "which",
        "konsa",
        "konsi",
        "kis",
        "naam",
        "name",
        "show",
        "batao",
    ]
    status_words = [
        "status",
        "active",
        "completed",
        "pending",
        "current",
        "kya hai",
        "hua",
    ]
    date_words = ["date", "when", "kab", "start", "end", "begin"]
    stipend_words = ["stipend", "salary", "pay", "kitna milta", "per month"]
    duration_words = [
        "duration",
        "din",
        "days",
        "kitne mahine",
        "how long",
        "kitne din",
        "kitne time",
    ]
    last_pay_words = [
        "last payment",
        "aakhri payment",
        "latest payment",
        "pichla payment",
        "recent payment",
    ]
    months_pay_words = [
        "kitne mahine",
        "how many months",
        "months of payment",
        "mahine ka payment",
    ]

    is_dbt = any(w in q for w in dbt_words)
    is_grievance = any(w in q for w in grievance_words)
    is_profile = any(w in q for w in profile_words)
    is_all = any(w in q for w in all_words)
    is_count = any(w in q for w in count_words)
    is_list = any(w in q for w in list_words)
    is_status = any(w in q for w in status_words)
    is_date = any(w in q for w in date_words)
    is_stipend = any(w in q for w in stipend_words)
    is_duration = any(w in q for w in duration_words)
    is_last_pay = any(w in q for w in last_pay_words)
    is_months = any(w in q for w in months_pay_words)

    if is_all:
        return "ALL_DATA"
    if is_profile:
        return "CANDIDATE_PROFILE"
    if is_grievance:
        if is_count:
            return "GRIEVANCE_COUNT"
        if is_status:
            return "GRIEVANCE_STATUS"
        return "GRIEVANCE_LIST"
    if is_dbt:
        if is_last_pay:
            return "DBT_LAST_PAYMENT"
        if is_months:
            return "DBT_MONTHS"
        if is_count:
            return "DBT_COUNT"
        if "total" in q:
            return "DBT_TOTAL_AMOUNT"
        if is_status:
            return "DBT_DETAILS"
        if is_stipend and not any(
            w in q for w in ["mila", "received", "amount", "total", "kitna aaya"]
        ):
            return "INTERNSHIP_STIPEND"
        return "DBT_DETAILS"
    if is_duration:
        return "INTERNSHIP_DURATION"
    if is_stipend:
        return "INTERNSHIP_STIPEND"
    if is_date:
        return "INTERNSHIP_DATES"
    if is_status:
        return "INTERNSHIP_STATUS"
    if is_count:
        return "INTERNSHIP_COUNT"
    if is_list:
        return "INTERNSHIP_LIST"
    return "ALL_DATA"


# -----------------------------------------------
# Field Resolution Helpers
# -----------------------------------------------


def _get_role(item: dict) -> str:
    return (
        item.get("job_role_name")
        or item.get("sector_name")
        or item.get("title")
        or item.get("role")
        or "N/A"
    )


def _get_status(item: dict) -> str:
    return (
        item.get("applied_internship_status")
        or item.get("internship_status")
        or item.get("status")
        or "N/A"
    )


def _get_company(item: dict) -> str:
    return (
        item.get("company_name")
        or item.get("organization")
        or item.get("company")
        or "N/A"
    )


def _get_start(item: dict) -> str:
    return (
        item.get("internship_start_date")
        or item.get("start_date")
        or item.get("start")
        or "N/A"
    )


def _get_end(item: dict) -> str:
    return (
        item.get("internship_end_date")
        or item.get("end_date")
        or item.get("end")
        or "N/A"
    )


def _get_stipend(item: dict) -> str:
    return str(item.get("stipend_amount") or item.get("stipend") or "N/A")


def _get_dbt_date(item: dict) -> str:
    return (
        item.get("payment_date")
        or item.get("transaction_date")
        or item.get("date")
        or item.get("month")
        or item.get("credit_date")
        or "N/A"
    )


def _get_dbt_amount(item: dict) -> float:
    raw = (
        item.get("amount")
        or item.get("stipend")  # <-- yeh add karo
        or item.get("stipend_amount")
        or item.get("credited_amount")
        or 0
    )
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _parse_date(date_str: str):
    """Parse a date string into a datetime. Returns None on failure."""
    if not date_str or date_str == "N/A":
        return None
    formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%b %Y",
        "%B %Y",
        "%Y-%m",
        "%m-%Y",
        "%d %b %Y",
        "%d %B %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


# -----------------------------------------------
# Context Extractors
# Each extractor returns a tuple:
#   (context_for_llm: str, fallback_answer: str)
#
# context_for_llm  - focused, pre-computed facts from the JSON relevant
#                    to the sub-intent; passed directly into the LLM prompt
# fallback_answer  - formatted template answer returned when LLM is not loaded
# -----------------------------------------------


def _extract_internship_count(internships: list) -> tuple:
    n = len(internships)
    context = f"Total internships applied: {n}"
    if internships:
        names = [_get_role(i) for i in internships]
        context += f"\nInternship names: {', '.join(names)}"
    fallback = f"You applied to {n} internship(s)."
    return context, fallback


def _extract_internship_list(internships: list) -> tuple:
    n = len(internships)
    if not internships:
        return "No internship records found.", "No internship records found."
    lines = [f"Total internships applied: {n}\n"]
    for i, item in enumerate(internships, 1):
        lines.append(
            f"  {i}. Role: {_get_role(item)} | Company: {_get_company(item)} | Status: {_get_status(item)}"
        )
    context = "\n".join(lines)
    fallback = context
    return context, fallback


def _extract_internship_status(internships: list) -> tuple:
    if not internships:
        return "No internship records found.", "No internship records found."
    lines = [f"Internship status for {len(internships)} application(s):\n"]
    for i, item in enumerate(internships, 1):
        lines.append(f"  {i}. {_get_role(item)}: status = {_get_status(item)}")
    context = "\n".join(lines)
    fallback = context
    return context, fallback


def _extract_internship_dates(internships: list) -> tuple:
    if not internships:
        return "No internship records found.", "No internship records found."
    lines = [f"Internship date information ({len(internships)} record(s)):\n"]
    for i, item in enumerate(internships, 1):
        lines.append(
            f"  {i}. {_get_role(item)}: start = {_get_start(item)}, end = {_get_end(item)}"
        )
    context = "\n".join(lines)
    fallback = context
    return context, fallback


def _extract_internship_stipend(internships: list) -> tuple:
    if not internships:
        return "No internship records found.", "No internship records found."
    lines = [f"Stipend information ({len(internships)} internship(s)):\n"]
    for i, item in enumerate(internships, 1):
        lines.append(
            f"  {i}. {_get_role(item)}: stipend = Rs {_get_stipend(item)} per month"
        )
    context = "\n".join(lines)
    fallback = context
    return context, fallback


def _extract_internship_duration(internships: list, dbt_records: list) -> tuple:
    """
    Primary source: sort DBT payment dates, compute first to last inclusive months.
    Fallback: use start and end dates from internship records.
    """
    if dbt_records:
        dated = []
        for p in dbt_records:
            d = _parse_date(_get_dbt_date(p))
            if d:
                dated.append(d)

        if len(dated) >= 2:
            dated.sort()
            first_pay = dated[0]
            last_pay = dated[-1]
            months = (
                (last_pay.year - first_pay.year) * 12
                + (last_pay.month - first_pay.month)
                + 1
            )
            days = (last_pay - first_pay).days + 1
            context = (
                f"Internship duration calculated from DBT payment records:\n"
                f"  First payment month : {first_pay.strftime('%B %Y')}\n"
                f"  Last payment month  : {last_pay.strftime('%B %Y')}\n"
                f"  Total months (inclusive): {months}\n"
                f"  Total days (first to last): {days}\n"
                f"  Total DBT payments received: {len(dbt_records)}"
            )
            fallback = (
                f"Based on your DBT payment records:\n"
                f"  First payment : {first_pay.strftime('%B %Y')}\n"
                f"  Last payment  : {last_pay.strftime('%B %Y')}\n"
                f"  Duration      : {months} month(s) ({days} day(s))"
            )
            return context, fallback

        if len(dated) == 1:
            context = f"Only one DBT payment found on {_get_dbt_date(dbt_records[0])}. Cannot compute full duration."
            fallback = context
            return context, fallback

    if internships:
        lines = ["Internship duration from internship records:\n"]
        fallback_lines = ["Internship duration:\n"]
        for i, item in enumerate(internships, 1):
            start_str = _get_start(item)
            end_str = _get_end(item)
            start_dt = _parse_date(start_str)
            end_dt = _parse_date(end_str)
            if start_dt and end_dt:
                days = (end_dt - start_dt).days + 1
                months = round(days / 30, 1)
                line = (
                    f"  {i}. {_get_role(item)}: {start_str} to {end_str} "
                    f"= {days} day(s) (~{months} month(s))"
                )
            else:
                line = (
                    f"  {i}. {_get_role(item)}: start = {start_str}, end = {end_str} "
                    f"(could not calculate)"
                )
            lines.append(line)
            fallback_lines.append(line)
        return "\n".join(lines), "\n".join(fallback_lines)

    msg = "No internship or DBT records available to calculate duration."
    return msg, msg


def _extract_dbt_count(dbt_records: list) -> tuple:
    n = len(dbt_records)
    context = f"Total DBT/PFMS payments received: {n}"
    fallback = f"You have received {n} payment(s) via DBT/PFMS."
    return context, fallback


def _extract_dbt_months(dbt_records: list) -> tuple:
    if not dbt_records:
        msg = "No DBT/PFMS payment records found."
        return msg, msg

    months = [_get_dbt_date(p) for p in dbt_records if _get_dbt_date(p) != "N/A"]
    if not months:
        msg = f"You have {len(dbt_records)} payment record(s) but no date information is available."
        return msg, msg

    lines = [f"Months of stipend payment received: {len(months)}\n"]
    for i, m in enumerate(months, 1):
        lines.append(f"  {i}. {m}")
    context = "\n".join(lines)
    fallback = context
    return context, fallback


def _extract_dbt_last_payment(dbt_records: list) -> tuple:
    if not dbt_records:
        msg = "No DBT/PFMS payment records found."
        return msg, msg

    dated = []
    undated = []
    for p in dbt_records:
        d = _parse_date(_get_dbt_date(p))
        if d:
            dated.append((d, p))
        else:
            undated.append(p)

    if dated:
        dated.sort(key=lambda x: x[0], reverse=True)
        _, last = dated[0]
        amount = _get_dbt_amount(last)
        date_str = _get_dbt_date(last)
        lines = ["Most recent payment details:\n"]
        lines.append(f"  Payment date   : {date_str}")
        lines.append(
            f"  Amount         : Rs {amount:,.0f}" if amount else "  Amount : N/A"
        )
        for k, v in last.items():
            if k not in {
                "amount",
                "stipend_amount",
                "credited_amount",
                "payment_date",
                "transaction_date",
                "date",
                "month",
                "credit_date",
            }:
                lines.append(f"  {str(k).replace('_', ' ').title()}: {v}")
        context = "\n".join(lines)
        fallback = context
        return context, fallback

    last = dbt_records[-1]
    lines = ["Last payment record (date could not be parsed):\n"]
    for k, v in last.items():
        lines.append(f"  {str(k).replace('_', ' ').title()}: {v}")
    context = "\n".join(lines)
    fallback = context
    return context, fallback


def _extract_dbt_total_amount(dbt_records: list) -> tuple:
    if not dbt_records:
        msg = "No DBT/PFMS payment records found."
        return msg, msg

    total = sum(_get_dbt_amount(p) for p in dbt_records)
    amounts_list = [f"Rs {_get_dbt_amount(p):,.0f}" for p in dbt_records]
    context = (
        f"DBT/PFMS total amount summary:\n"
        f"  Number of payments   : {len(dbt_records)}\n"
        f"  Individual payments  : {', '.join(amounts_list)}\n"
        f"  Total amount credited: Rs {total:,.0f}"
    )
    fallback = context
    return context, fallback


def _extract_dbt_details(dbt_records: list) -> tuple:
    if not dbt_records:
        msg = "No DBT/PFMS payment records found."
        return msg, msg

    lines = [f"All DBT/PFMS stipend payments ({len(dbt_records)} record(s)):\n"]
    total = 0
    for i, item in enumerate(dbt_records, 1):
        lines.append(f"  Payment {i}:")
        amount = (
            item.get("stipend")
            or item.get("stipend_amount")
            or item.get("amount")
            or "N/A"
        )
        lines.append(f"      Stipend Amount: Rs {amount}")
        lines.append(f"      Status: {item.get('status', 'N/A')}")
        lines.append(
            f"      Payment Date: {item.get('stipend_date') or item.get('payment_date', 'N/A')}"
        )
        lines.append(f"      Month: {item.get('year_month', 'N/A')}")
        lines.append(f"      Bank: {item.get('bankname', 'N/A')}")
        lines.append(f"      Stage: {item.get('current_stage', 'N/A')}")
        try:
            total += float(amount)
        except (TypeError, ValueError):
            pass
    lines.append(f"\n  Total Stipend Credited: Rs {total:,.0f}")
    context = "\n".join(lines)
    return context, context


def _extract_grievance_count(grievances: list) -> tuple:
    n = len(grievances)
    context = f"Total grievances filed: {n}"
    fallback = f"You have {n} grievance(s) on record."
    return context, fallback


def _extract_grievance_status(grievances: list) -> tuple:
    if not grievances:
        msg = "You have no grievances on record."
        return msg, msg

    lines = [f"Grievance status ({len(grievances)} record(s)):\n"]
    for i, item in enumerate(grievances, 1):
        status = (
            item.get("grievance_status")
            or item.get("status")
            or item.get("complaint_status")
            or "N/A"
        )
        title = (
            item.get("grievance_title")
            or item.get("subject")
            or item.get("title")
            or str(item.get("description", ""))[:60]
            or "N/A"
        )
        lines.append(f"  {i}. {title}: status = {status}")
    context = "\n".join(lines)
    fallback = context
    return context, fallback


def _extract_grievance_list(grievances: list) -> tuple:
    if not grievances:
        msg = "You have no grievances on record."
        return msg, msg

    lines = [f"All grievances filed ({len(grievances)} total):\n"]
    for i, item in enumerate(grievances, 1):
        lines.append(f"  Grievance {i}:")
        for k, v in item.items():
            lines.append(f"      {str(k).replace('_', ' ').title()}: {v}")
    context = "\n".join(lines)
    fallback = context
    return context, fallback


def _extract_candidate_profile(data: dict) -> tuple:
    exclude = {"internships", "pfmsdbt", "grievances"}
    profile = {k: v for k, v in data.items() if k not in exclude}
    if not profile:
        msg = "No personal profile fields found in the API response."
        return msg, msg

    lines = ["Candidate personal profile:\n"]
    for k, v in profile.items():
        if isinstance(v, dict):
            lines.append(f"  {str(k).replace('_', ' ').title()}:")
            for sk, sv in v.items():
                lines.append(f"      {str(sk).replace('_', ' ').title()}: {sv}")
        elif isinstance(v, list):
            lines.append(
                f"  {str(k).replace('_', ' ').title()}: {', '.join(str(x) for x in v)}"
            )
        else:
            lines.append(f"  {str(k).replace('_', ' ').title()}: {v}")
    context = "\n".join(lines)
    fallback = context
    return context, fallback


def _extract_all_data(data: dict) -> tuple:
    internships = data.get("internships", [])
    dbt_records = data.get("pfmsdbt", [])
    grievances = data.get("grievances", [])

    profile_ctx, _ = _extract_candidate_profile(data)
    internship_ctx, _ = (
        _extract_internship_list(internships)
        if internships
        else ("No internship records.", "")
    )
    status_ctx, _ = _extract_internship_status(internships) if internships else ("", "")
    dbt_ctx, _ = _extract_dbt_details(dbt_records)
    grievance_ctx, _ = _extract_grievance_list(grievances)

    context = (
        f"=== CANDIDATE PROFILE ===\n{profile_ctx}\n\n"
        f"=== INTERNSHIPS ===\n{internship_ctx}\n\n"
        f"=== INTERNSHIP STATUS ===\n{status_ctx}\n\n"
        f"=== DBT / PFMS PAYMENTS ===\n{dbt_ctx}\n\n"
        f"=== GRIEVANCES ===\n{grievance_ctx}"
    )
    fallback = context
    return context, fallback


# -----------------------------------------------
# Semantic Search on User JSON Data
# -----------------------------------------------


def _json_record_to_text(record: dict, section: str, index: int) -> str:
    """
    Convert a single JSON record (internship, DBT payment, or grievance)
    into a flat human-readable text string suitable for embedding.
    Keeps ALL fields so nothing is lost during semantic search.
    """
    parts = [f"[{section} record {index}]"]
    for k, v in record.items():
        if v is not None and v != "" and v != "N/A":
            label = str(k).replace("_", " ")
            parts.append(f"{label}: {v}")
    return "  ".join(parts)


def _build_user_data_chunks(user_data: dict) -> list:
    """
    Convert the full cached user JSON into a flat list of text chunks.
    Each chunk is a dict: {"text": str, "section": str, "index": int, "record": dict}

    Sections covered:
      - internships  (one chunk per internship record)
      - pfmsdbt      (one chunk per DBT payment record)
      - grievances   (one chunk per grievance record)
      - profile      (one chunk for all top-level candidate fields)
    """
    chunks = []

    # Internship records
    for i, item in enumerate(user_data.get("internships", []), 1):
        text = _json_record_to_text(item, "INTERNSHIP", i)
        chunks.append(
            {"text": text, "section": "internship", "index": i, "record": item}
        )

    # DBT / PFMS payment records
    for i, item in enumerate(user_data.get("pfmsdbt", []), 1):
        text = _json_record_to_text(item, "DBT_PAYMENT", i)
        chunks.append({"text": text, "section": "dbt", "index": i, "record": item})

    # Grievance records
    for i, item in enumerate(user_data.get("grievances", []), 1):
        text = _json_record_to_text(item, "GRIEVANCE", i)
        chunks.append(
            {"text": text, "section": "grievance", "index": i, "record": item}
        )

    # Candidate profile (top-level fields only, no nested lists)
    exclude = {"internships", "pfmsdbt", "grievances"}
    profile = {
        k: v
        for k, v in user_data.items()
        if k not in exclude and not isinstance(v, list)
    }
    if profile:
        profile_text = "[CANDIDATE_PROFILE]  " + "  ".join(
            f"{str(k).replace('_', ' ')}: {v}" for k, v in profile.items() if v
        )
        chunks.append(
            {"text": profile_text, "section": "profile", "index": 0, "record": profile}
        )

    return chunks


def semantic_search_on_user_data(
    query: str,
    user_data: dict,
    sub_intent: str,
    top_k: int = 5,
) -> list:
    """
    Perform semantic (embedding-based) search over the user's cached JSON data.

    Process:
      1. Convert every JSON record into a text chunk.
      2. Embed all chunks using the same HuggingFace embedder already loaded.
      3. Embed the user query.
      4. Rank chunks by cosine similarity.
      5. Return top_k most relevant chunk texts.

    If the embedder is not available, returns an empty list (caller uses extractor
    context only).

    The sub_intent is used to optionally pre-filter chunks to the most relevant
    section before running embedding, reducing noise.
    """
    chunks = _build_user_data_chunks(user_data)
    if not chunks:
        logger.warning("[SEMANTIC] No chunks built from user data.")
        return []

    # Section pre-filter based on sub_intent so the LLM gets focused context
    section_filter_map = {
        "INTERNSHIP_COUNT": {"internship"},
        "INTERNSHIP_LIST": {"internship"},
        "INTERNSHIP_STATUS": {"internship"},
        "INTERNSHIP_DATES": {"internship"},
        "INTERNSHIP_STIPEND": {"internship"},
        "INTERNSHIP_DURATION": {"internship", "dbt"},
        "DBT_COUNT": {"dbt"},
        "DBT_MONTHS": {"dbt"},
        "DBT_LAST_PAYMENT": {"dbt"},
        "DBT_TOTAL_AMOUNT": {"dbt"},
        "DBT_DETAILS": {"dbt"},
        "GRIEVANCE_COUNT": {"grievance"},
        "GRIEVANCE_STATUS": {"grievance"},
        "GRIEVANCE_LIST": {"grievance"},
        "CANDIDATE_PROFILE": {"profile"},
        "COMPLAINT_STATUS": {"grievance"},
        "COMPLAINT_LIST": {"grievance"},
        "COMPLAINT_COUNT": {"grievance"},
        "ALL_DATA": None,  # no filter, search everything
    }

    allowed_sections = section_filter_map.get(sub_intent)
    if allowed_sections is not None:
        filtered = [c for c in chunks if c["section"] in allowed_sections]
        # If filter removed everything (edge case), fall back to all chunks
        search_chunks = filtered if filtered else chunks
    else:
        search_chunks = chunks

    if not search_chunks:
        return []

    try:
        emb = get_embedder()

        # Embed query
        query_vec = np.array(emb.embed_query(query), dtype=np.float32)
        qnorm = np.linalg.norm(query_vec)
        if qnorm == 0:
            logger.warning("[SEMANTIC] Query embedding is zero vector.")
            return [c["text"] for c in search_chunks[:top_k]]

        # Embed all chunks
        chunk_texts = [c["text"] for c in search_chunks]
        chunk_vecs = np.array(emb.embed_documents(chunk_texts), dtype=np.float32)

        # Cosine similarity for each chunk
        scores = []
        for i, vec in enumerate(chunk_vecs):
            vnorm = np.linalg.norm(vec)
            score = float(np.dot(query_vec, vec) / (qnorm * vnorm + 1e-8))
            scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)

        top_chunks = [chunk_texts[i] for _, i in scores[:top_k]]

        log_lines = "\n".join(
            f"  {s:.3f}: {chunk_texts[i][:120]}" for s, i in scores[:top_k]
        )
        logger.info(f"[SEMANTIC] top chunks:\n{log_lines}")

        return top_chunks

    except Exception as e:
        logger.warning(f"[SEMANTIC] Embedding failed: {e}. Skipping semantic search.")
        return [c["text"] for c in search_chunks[:top_k]]


# -----------------------------------------------
# Natural Response Generator (Step 3 LLM call)
# -----------------------------------------------


def generate_natural_response(
    query: str,
    sub_intent: str,
    extracted_context: str,
    semantic_chunks: list,
    conversation_context: str,
    fallback_answer: str,
) -> str:
    """
    Final LLM call. Combines two context sources:
      extracted_context : structured pre-computed facts from the extractor
      semantic_chunks   : top-K most relevant raw JSON records via semantic search

    The LLM sees both and writes a natural, conversational answer.
    Falls back to fallback_answer when model is not loaded or generation fails.
    """
    if model is None:
        logger.info("[GENERATE] Model not loaded. Returning fallback.")
        return fallback_answer

    current_dt = datetime_tool.get_current_datetime()

    conv_section = ""
    if conversation_context:
        conv_section = (
            "Previous conversation (for context-aware follow-ups):\n"
            + conversation_context
            + "\n\n"
        )

    if semantic_chunks:
        sem_lines = "\n".join(f"  - {chunk}" for chunk in semantic_chunks)
        semantic_section = (
            f"Semantically relevant records ({len(semantic_chunks)} result(s) ranked by similarity):\n"
            + sem_lines
        )
    else:
        semantic_section = "No additional semantic matches found."

    # prompt = (
    #     "<|system|>\n"
    #     "You are a helpful chatbot assistant.\n"
    #     "The user is logged in. Their data has been retrieved from the government portal.\n"
    #     "Answer the user question naturally and conversationally using ONLY the data below.\n"
    #     "Do not invent or assume any information not present in the data.\n"
    #     "Avoid repeating phrases, emojis and provide a complete, single answer.\n"
    #     "The user may make spelling mistakes, typing errors, or use informal wording.\n"
    #     "You MUST understand the intended meaning and respond with the closest correct answer.\n"
    #     "Always match the most relevant information even if the question is misspelled.\n"
    #     "Do not mention spelling mistakes in your response.\n"
    #     "Stipend is often mentioned as 'payment' in the data, but the user may ask about 'payment' or 'salary' or 'stipend'.\n"
    #     "Focus on the specific sub-intent of the user's question to find the relevant facts.\n"
    #     f"Sub-intent: {sub_intent}\n"
    #     f"Current date/time: {current_dt['formatted']}\n"
    #     "</|system|>\n"
    #     "<|user|>\n"
    #     + conv_section
    #     + "--- Structured extracted facts ---\n"
    #     + extracted_context
    #     + "\n\n--- Semantic search results (most relevant records) ---\n"
    #     + semantic_section
    #     + "\n\nUser question: "
    #     + query
    #     + "\n</|user|>\n<|assistant|>\n"
    # )

    prompt = (
        "<|im_start|>system\n"
        "You are a helpful chatbot assistant.\n"
        "The user is logged in. Their data has been retrieved from the government portal.\n"
        "Answer the user question naturally and conversationally using ONLY the data below.\n"
        "Do not invent or assume any information not present in the data.\n"
        "Avoid repeating phrases, emojis and provide a complete, single answer.\n"
        "The user may make spelling mistakes, typing errors, or use informal wording.\n"
        "You MUST understand the intended meaning and respond with the closest correct answer.\n"
        "Always match the most relevant information even if the question is misspelled.\n"
        "Do not mention spelling mistakes in your response.\n"
        "Stipend is often mentioned as 'payment' in the data, but the user may ask about 'payment' or 'salary' or 'stipend'.\n"
        "Focus on the specific sub-intent of the user's question to find the relevant facts.\n"
        f"Sub-intent: {sub_intent}\n"
        f"Current date/time: {current_dt['formatted']}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        + conv_section
        + "--- Structured extracted facts ---\n"
        + extracted_context
        + "\n\n--- Semantic search results ---\n"
        + semantic_section
        + "\n\nQuestion: "
        + query
        + "\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    logger.info(
        f"[GENERATE] sub_intent={sub_intent} query={query}\n"
        f"Extracted:\n{extracted_context}\n"
        f"Semantic ({len(semantic_chunks)} hits):\n"
        + "\n".join(f"  {c[:120]}" for c in semantic_chunks)
    )

    # print(f"\n  You > {query}\n  Thinking\u2026", flush=True)
    # response = call_llm(prompt, max_tokens=1024, temperature=0.2)
    # input_length = response["input_ids"].shape[1]
    # new_tokens = out[0][input_length:]
    # if new_tokens and len(response.strip()) > 15:
    #     logger.info(
    #         f"[GENERATE] LLM response ({len(response)} chars): {response[:200]}"
    #     )
    #     answer_lines = "\n    ".join(response.strip().splitlines())
    #     print(
    #         f"{SEP}\n  Bot:\n    {answer_lines}\n{SEP}\n"
    #         f"  [USER-DATA | intent: user_specific | sub_intent: {sub_intent}]\n",
    #         flush=True,
    #     )
    #     return response.strip()

    # logger.warning("[GENERATE] LLM empty or too short. Using fallback.")
    # return fallback_answer
    print(f"\n  You > {query}\n  Thinking…", flush=True)
    response = call_llm(prompt, max_tokens=512, temperature=0.2)
    # response is already a clean string here — no slicing needed

    if response and len(response.strip()) > 15:
        logger.info(
            f"[GENERATE] LLM response ({len(response)} chars): {response[:200]}"
        )
        SEP = "─" * 60
        answer_lines = "\n    ".join(response.strip().splitlines())
        print(
            f"{SEP}\n  Bot:\n    {answer_lines}\n{SEP}\n"
            f"  [USER-DATA | intent: user_specific | sub_intent: {sub_intent}]\n",
            flush=True,
        )
        return response.strip()

    logger.warning("[GENERATE] LLM empty or too short. Using fallback.")
    return fallback_answer


# -----------------------------------------------
# Response Generation Tool
# -----------------------------------------------
class ResponseTool:
    """
    Orchestrates the full user-data answering pipeline:
      1. Fetch and cache full candidate JSON from API 3 (done once on login).
      2. For each user question:
         a. classify_sub_intent()    - Step 1 LLM call, identifies what the user wants
         b. context extractor        - pulls only the relevant JSON slice and pre-computes facts
         c. generate_natural_response() - Step 2 LLM call, produces conversational answer from context
    """

    @staticmethod
    def fetch_candidate_data(token: str) -> Dict[str, Any]:
        """
        Call API 3 with the auth token to fetch the full candidate JSON.
        The raw response is cached without any modification.
        """
        try:
            resp = requests.get(
                CANDIDATE_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                    "fp": "test-device",
                },
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                logger.info(
                    f"[API 3] Candidate data OK.\nRaw JSON:\n"
                    f"{json.dumps(data, indent=2, ensure_ascii=False)}"
                )
                return {"success": True, "data": data}
            logger.error(f"[API 3] HTTP {resp.status_code}: {resp.text[:300]}")
            return {"success": False, "error": f"API 3 returned {resp.status_code}"}
        except requests.exceptions.Timeout:
            return {"success": False, "error": "API 3 timed out."}
        except Exception as e:
            logger.error(f"[API 3] Error: {e}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def generate_from_user_data(
        query: str,
        user_data: Dict,
        conversation_context: str = "",
    ) -> str:
        """
        Main answer pipeline for any user-specific question.

        Step 1: classify_sub_intent()          - LLM identifies what the user wants
        Step 2: context extractor              - pulls the relevant JSON slice + computes facts
        Step 3: semantic_search_on_user_data() - finds top-K most relevant records via embeddings
        Step 4: generate_natural_response()    - LLM writes natural answer from both sources
        """
        internships = user_data.get("internships", [])
        dbt_records = user_data.get("pfmsdbt", [])
        grievances = user_data.get("grievances", [])

        logger.info(
            f"[CONTEXT] query={query}\n"
            f"Cached JSON:\n{json.dumps(user_data, indent=2, ensure_ascii=False)}"
        )

        # Step 1: classify sub-intent via LLM (heuristic fallback if model absent)
        sub_intent = classify_sub_intent(query, conversation_context)
        logger.info(f"[SUB-INTENT] {sub_intent}")

        # Step 2: extract structured pre-computed facts relevant to this sub-intent
        extractor_map = {
            "INTERNSHIP_COUNT": lambda: _extract_internship_count(internships),
            "INTERNSHIP_LIST": lambda: _extract_internship_list(internships),
            "INTERNSHIP_STATUS": lambda: _extract_internship_status(internships),
            "INTERNSHIP_DATES": lambda: _extract_internship_dates(internships),
            "INTERNSHIP_STIPEND": lambda: _extract_internship_stipend(internships),
            "INTERNSHIP_DURATION": lambda: _extract_internship_duration(
                internships, dbt_records
            ),
            "DBT_COUNT": lambda: _extract_dbt_count(dbt_records),
            "DBT_MONTHS": lambda: _extract_dbt_months(dbt_records),
            "DBT_LAST_PAYMENT": lambda: _extract_dbt_last_payment(dbt_records),
            "DBT_TOTAL_AMOUNT": lambda: _extract_dbt_total_amount(dbt_records),
            "DBT_DETAILS": lambda: _extract_dbt_details(dbt_records),
            "GRIEVANCE_COUNT": lambda: _extract_grievance_count(grievances),
            "GRIEVANCE_STATUS": lambda: _extract_grievance_status(grievances),
            "GRIEVANCE_LIST": lambda: _extract_grievance_list(grievances),
            "COMPLAINT_STATUS": lambda: _extract_grievance_status(grievances),
            "COMPLAINT_LIST": lambda: _extract_grievance_list(grievances),
            "COMPLAINT_COUNT": lambda: _extract_grievance_count(grievances),
            "CANDIDATE_PROFILE": lambda: _extract_candidate_profile(user_data),
            "ALL_DATA": lambda: _extract_all_data(user_data),
        }

        extractor = extractor_map.get(sub_intent, lambda: _extract_all_data(user_data))
        extracted_context, fallback_answer = extractor()

        # Step 3: semantic search over the user JSON records
        # This finds which records are most semantically similar to the user query,
        # using the same HuggingFace embedder loaded at startup.
        semantic_chunks = semantic_search_on_user_data(
            query=query,
            user_data=user_data,
            sub_intent=sub_intent,
            top_k=5,
        )

        # Step 4: LLM generates natural response from extractor context + semantic hits
        return generate_natural_response(
            query=query,
            sub_intent=sub_intent,
            extracted_context=extracted_context,
            semantic_chunks=semantic_chunks,
            conversation_context=conversation_context,
            fallback_answer=fallback_answer,
        )


response_tool = ResponseTool()


# -----------------------------------------------
# Conversation Memory Helpers
# -----------------------------------------------
def add_to_conversation_memory(thread_id: str, role: str, content: str):
    """Append a message to the thread's conversation history."""
    if thread_id not in conversation_memory:
        conversation_memory[thread_id] = []
    conversation_memory[thread_id].append(
        {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
    )
    # Trim to keep only the last MAX_CONVERSATION_MEMORY messages
    if len(conversation_memory[thread_id]) > MAX_CONVERSATION_MEMORY:
        conversation_memory[thread_id] = conversation_memory[thread_id][
            -MAX_CONVERSATION_MEMORY:
        ]
    logger.info(
        f"Memory updated: thread={thread_id}, "
        f"total_messages={len(conversation_memory[thread_id])}"
    )


def get_conversation_memory(thread_id: str) -> List[Dict]:
    """Return full conversation history for a thread."""
    return conversation_memory.get(thread_id, [])


def get_conversation_context(thread_id: str, last_n: int = 5) -> str:
    """Return the last N messages as a formatted string for LLM context."""
    history = get_conversation_memory(thread_id)
    if not history:
        return ""
    recent = history[-last_n:]
    parts = []
    for msg in recent:
        role = msg["role"].upper()
        content = (
            msg["content"][:100] + "..."
            if len(msg["content"]) > 100
            else msg["content"]
        )
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


# -----------------------------------------------
# Model Loading
# -----------------------------------------------
def load_model():
    """Load the Qwen LLM from disk."""
    global model, tokenizer, device
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at: {MODEL_PATH}")
        return False
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model on {device.upper()}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            model = model.to(device)
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        return False


def get_embedder():
    """Lazy-load the sentence embedding model on CPU or CUDA automatically."""
    global embedder
    if embedder:
        return embedder
    embed_device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading embedding model on {embed_device.upper()}...")
    embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": embed_device},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embedder


def load_faiss():
    """Load FAISS vector store and build BM25 keyword index."""
    global faiss_db, doc_summary_store, bm25_index, bm25_corpus_texts, bm25_docs_metadata
    if not os.path.exists(VECTORSTORE_DIR):
        logger.error(f"Vectorstore not found at: {VECTORSTORE_DIR}")
        return False
    try:
        emb = get_embedder()
        logger.info("Loading FAISS index...")
        faiss_db = FAISS.load_local(
            VECTORSTORE_DIR, emb, allow_dangerous_deserialization=True
        )
        docs = list(faiss_db.docstore._dict.values())
        bm25_corpus_texts = [d.page_content for d in docs]
        bm25_docs_metadata = [d.metadata for d in docs]
        tokenized = [txt.lower().split() for txt in bm25_corpus_texts]
        bm25_index = BM25Okapi(tokenized)
        if os.path.exists(DOC_SUMMARY_PATH):
            with open(DOC_SUMMARY_PATH, "r") as f:
                doc_summary_store.update(json.load(f))
        logger.info("FAISS and BM25 loaded successfully")
        return True
    except Exception as e:
        logger.error(f"FAISS loading error: {e}")
        return False


# -----------------------------------------------
# LangGraph State Definition
# -----------------------------------------------
class RAGState(TypedDict):
    # Input
    query: str
    thread_id: str  # Unique thread/conversation identifier
    k: int
    metadata_filters: Optional[Dict[str, str]]

    # Intent classification results
    intent: str  # "general" or "user_specific" or "not_valid_query"
    is_user_specific: bool
    is_not_valid_query: bool
    has_date_query: bool
    current_datetime: Dict[str, Any]
    intent_confidence: float
    entities: Optional[Dict[str, Any]]

    # Session and authentication state
    session_info: Optional[Dict[str, Any]]
    is_authenticated: bool
    awaiting_phone: bool
    awaiting_otp: bool
    phone_number: Optional[str]
    user_data: Optional[Dict[str, Any]]

    # RAG pipeline components
    top_docs: List[str]
    semantic_docs: List[Document]
    keyword_docs: List[Document]
    merged_docs: List[Document]
    context: str

    # Final output
    answer: str
    error: Optional[str]
    metadata: Dict[str, Any]
    response_type: str  # "general", "user_api", "auth", "no_auth"


# -----------------------------------------------
# LangGraph Agent Nodes
# -----------------------------------------------


def agent_initialize_session(state: RAGState) -> RAGState:
    """
    Node 1: Initialize or restore session state.
    - Checks if session token is still valid (not expired)
    - Restores awaiting_phone / awaiting_otp flags
    - Restores cached user data for authenticated sessions
    """
    thread_id = state.get("thread_id", "anonymous")

    # Create session if first time, or refresh last_activity timestamp
    session = session_tool.get_or_create_session(thread_id)

    # Validate session (checks expiry and resets auth if expired)
    is_valid = session_tool.is_session_valid(thread_id)
    session = sessions[thread_id]  # Re-read after potential expiry reset

    # Restore cached user data for authenticated sessions
    cached_user = session_tool.get_cached_user_data(thread_id)

    state["session_info"] = session
    state["is_authenticated"] = session.get("is_authenticated", False) and is_valid
    state["awaiting_phone"] = session.get("awaiting_phone", False)
    state["awaiting_otp"] = session.get("awaiting_otp", False)
    state["phone_number"] = session.get("phone_number", None)

    # Restore user data from cache if user is authenticated
    if state["is_authenticated"] and cached_user and not state.get("user_data"):
        state["user_data"] = cached_user

    state["metadata"] = state.get("metadata", {})
    logger.info(
        f"Session initialized: id={thread_id}, "
        f"authenticated={state['is_authenticated']}, "
        f"awaiting_phone={state['awaiting_phone']}, "
        f"awaiting_otp={state['awaiting_otp']}"
    )
    return state


_REAL_WORD_RE = re.compile(r"[a-zA-Z0-9\u0900-\u097F]+")  # Latin + Devanagari


def agent_not_valid_query(state: RAGState) -> RAGState:
    """
    Post-validation node — the sole validator for bad input.
    Reached only when the LLM classifies intent as NOT_VALID_QUERY.

    Decision table:
      Empty / blank                        → "Please ask a valid question."
      Only symbols / emojis / signs        → "Please ask a valid question."
      Exactly one real word                → "Please elaborate your question."
      Two or more real words (edge case)   → "Please elaborate your question."
    """
    query = state.get("query", "").strip()

    if not query:
        msg = "Please ask a valid question."
    else:
        real_words = _REAL_WORD_RE.findall(query)

        if len(real_words) == 0:
            # Only symbols / emojis / punctuation / signs
            msg = "Please ask a valid question."
        else:
            # One real word OR the LLM flagged a multi-word input as invalid
            msg = "Please elaborate your question."

    state["answer"] = msg
    state["response_type"] = "not_valid_query"
    state["metadata"]["source"] = "not_valid_query"
    return state


def agent_classify_intent(state: RAGState) -> RAGState:
    """
    Node 2: Classify query intent using LLM.
    Sets is_user_specific, intent, confidence, and date-query flag.
    """
    classification = intent_tool.classify_intent(state["query"])

    state["intent"] = classification["intent"]
    state["is_user_specific"] = classification["is_user_specific"]
    state["is_not_valid_query"] = classification["intent"] == "not_valid_query"
    state["intent_confidence"] = classification["confidence"]
    state["has_date_query"] = datetime_tool.check_date_query(state["query"])
    state["current_datetime"] = datetime_tool.get_current_datetime()
    state["metadata"] = {
        "intent": classification["intent"],
        "confidence": classification["confidence"],
        "has_date_query": state["has_date_query"],
    }
    logger.info(
        f"Intent classified: {classification['intent']} "
        f"(confidence={classification['confidence']:.2f}), "
        f"user_specific={classification['is_user_specific']}"
    )
    return state


def route_after_session_init(state: RAGState) -> str:
    """
    Routing function immediately after session initialization.

    IMPORTANT: This runs BEFORE the LLM intent classifier.
    If the user is in the middle of the auth flow (entered phone or OTP),
    we skip the LLM entirely and route directly to the right handler.
    The LLM is only called when we actually need to classify a real question.

    Routes:
    - awaiting_phone = True  -> handle_phone   (no LLM needed)
    - awaiting_otp   = True  -> verify_otp      (no LLM needed)
    - otherwise              -> classify_intent (LLM used here)
    """
    if state.get("awaiting_phone"):
        return "handle_phone"
    if state.get("awaiting_otp"):
        return "verify_otp"
    return "classify_intent"


def route_after_intent(state: RAGState) -> str:
    """
    Routing function after intent classification.
    At this point we know awaiting_phone and awaiting_otp are both False
    (handled by route_after_session_init above), so we only route
    on query intent.

    Routes:
    - is_user_specific = True  -> user_specific  (check auth, answer or start login)
    - otherwise                -> document_filter (RAG pipeline)
    """
    if state.get("is_not_valid_query"):
        return "not_valid_query"
    if state.get("is_user_specific"):
        return "user_specific"
    return "document_filter"


def agent_user_specific_response(state: RAGState) -> RAGState:
    """
    Node 3a: Handle user-specific queries.

    Flow:
    - If user IS authenticated (valid session token) -> fetch user data and answer
    - If user is NOT authenticated:
        * Save query to pending_queries
        * Ask user to provide phone number for OTP login
    """
    thread_id = state.get("thread_id", "anonymous")

    if state.get("is_authenticated") and state.get("user_data"):
        # User is logged in – always pull from the full cached JSON
        conversation_context = get_conversation_context(thread_id, last_n=5)
        cached = session_tool.get_cached_user_data(thread_id)
        user_data = cached if cached else state.get("user_data", {})

        answer = ResponseTool.generate_from_user_data(
            state["query"], user_data, conversation_context
        )
        state["answer"] = answer
        state["metadata"]["source"] = "user_api"
        state["metadata"]["user_id"] = sessions[thread_id].get("user_id")
        state["response_type"] = "user_api"
        logger.info(f"User-specific answer generated for thread: {thread_id}")

    else:
        # User is not authenticated -> save query and ask for phone number
        pending_queries[thread_id] = state["query"]
        logger.info(f"Pending query saved for thread '{thread_id}': {state['query']}")

        session_tool.update_session(thread_id, awaiting_phone=True)
        state["awaiting_phone"] = True

        state["answer"] = (
            "To access your personal information, please verify your identity.\n\n"
            "Please enter your registered 10-digit mobile number:"
        )
        state["metadata"]["source"] = "no_auth"
        state["metadata"]["pending_query_saved"] = True
        state["response_type"] = "no_auth"

    return state


def agent_handle_phone_input(state: RAGState) -> RAGState:
    """
    Node 3b: Handle phone number submitted by the user.

    Steps:
      1. Extract 10-digit number from input
      2. Call API 1 (192.168.1.52:4050) to check if number exists in DB
      3. If present → register for OTP and move to awaiting_otp
      4. If not present → reject and free the user (no loop trap)
    """
    logger.info("Agent: Handle Phone Input")
    query = state["query"].strip()
    thread_id = state["thread_id"]

    # ── Step 1: extract 10-digit number ──────────────────────────────────────
    match = re.search(r"\b\d{10}\b", query)
    phone = match.group() if match else None

    if not phone:
        session_tool.update_session(thread_id, awaiting_phone=False)
        state["awaiting_phone"] = False
        state["answer"] = (
            "That doesn't look like a valid 10-digit mobile number.\n\n"
            "Please enter your 10-digit registered number (e.g. 9876543210).\n"
            "Or feel free to ask any general question about the program."
        )
        state["response_type"] = "auth"
        return state

    # ── Step 2: API 1 – check if number is in DB ──────────────────────────────
    check = auth_tool.check_number_in_db(phone)

    if not check["success"]:
        # API unreachable – inform user, free them
        session_tool.update_session(thread_id, awaiting_phone=False)
        state["awaiting_phone"] = False
        state["answer"] = (
            f"Could not verify {phone}: {check.get('error', 'Service unavailable')}.\n"
            "Please try again later."
        )
        state["response_type"] = "auth"
        return state

    if not check["present"]:
        # ── Not in DB → reject ────────────────────────────────────────────────
        session_tool.update_session(thread_id, awaiting_phone=False)
        state["awaiting_phone"] = False
        state["answer"] = (
            # f"Mobile number {phone} is not registered in the system.\n\n"
            "Please double-check your number and try again, or ask any "
            "general question about the program."
        )
        state["response_type"] = "auth"
        return state

    # ── Step 3: Number is in DB → register for OTP ───────────────────────────
    auth_tool.register_phone_for_otp(phone)

    session_tool.update_session(
        thread_id,
        awaiting_phone=False,
        awaiting_otp=True,
        phone_number=phone,
        user_id=phone,
    )
    state["awaiting_phone"] = False
    state["awaiting_otp"] = True
    state["phone_number"] = phone

    state["answer"] = (
        # f"Mobile {phone} verified.\n\n"
        "An OTP has been sent to your registered number.\n"
        "Please enter the 6-digit OTP to continue:"
    )
    state["response_type"] = "auth"
    return state


def agent_verify_otp(state: RAGState) -> RAGState:
    """
    Node 3c: Verify OTP submitted by the user.
    - If OTP is correct:
        * Authenticate the session
        * Cache user data
        * Auto-answer any pending query from before login
    - If OTP is wrong: return error message
    """
    logger.info("Agent: Verify OTP")
    query = state["query"].strip()
    thread_id = state["thread_id"]

    # Extract 6-digit OTP from input
    otp = None
    if re.match(r"^\d{6}$", query):
        otp = query
    else:
        match = re.search(r"\b\d{6}\b", query)
        if match:
            otp = match.group()

    if not otp:
        state["answer"] = "Invalid OTP format. Please enter a 6-digit code."
        state["response_type"] = "auth"
        return state

    phone = state.get("phone_number")
    if not phone:
        # Phone number lost from session - restart auth
        session_tool.update_session(thread_id, awaiting_otp=False, awaiting_phone=True)
        state["awaiting_otp"] = False
        state["awaiting_phone"] = True
        state["answer"] = "Session error. Please enter your phone number again:"
        state["response_type"] = "auth"
        return state

    # Verify the OTP
    verify_result = auth_tool.verify_otp(phone, otp)
    if not verify_result["success"]:
        error_msg = verify_result["error"]

        # If OTP expired or too many attempts, the otp_store entry was deleted.
        # Reset the session so the user is free - they can retry by sending
        # their number again or just ask a general question.
        if "expired" in error_msg.lower() or "too many" in error_msg.lower():
            session_tool.update_session(
                thread_id,
                awaiting_otp=False,
                awaiting_phone=False,
                phone_number=None,
            )
            state["awaiting_otp"] = False
            state["answer"] = (
                f"OTP verification failed: {error_msg}\n\n"
                "To try again, ask a personal question (e.g. 'my application status') "
                "and you will be prompted to enter your number.\n\n"
                "Or feel free to ask any general question about the program."
            )
        else:
            # Wrong OTP but attempts still remaining - stay in OTP step
            state["answer"] = (
                f"Incorrect OTP: {error_msg}\n" "Please enter the correct 6-digit OTP."
            )
        state["response_type"] = "auth"
        return state

    # OTP verified - get token and fetch real candidate data
    token = verify_result["token"]

    data_result = ResponseTool.fetch_candidate_data(token)
    if data_result["success"]:
        user_data = data_result["data"]
    else:
        # Data fetch failed but login succeeded - continue with empty data
        logger.warning(f"Candidate data fetch failed: {data_result['error']}")
        user_data = {}

    user_id = user_data.get("user_id") or user_data.get("id") or phone

    # Mark session as authenticated and cache the FULL raw JSON
    session_tool.update_session(
        thread_id,
        is_authenticated=True,
        user_id=user_id,
        awaiting_otp=False,
        awaiting_phone=False,
        token=token,
    )
    session_tool.cache_user_data(thread_id, user_data)

    state["is_authenticated"] = True
    state["user_data"] = user_data
    state["awaiting_otp"] = False

    logger.info(
        f"User authenticated: user_id={user_id}, thread={thread_id}\n"
        f"Cached JSON:\n{json.dumps(user_data, indent=2, ensure_ascii=False)}"
    )

    # Auto-answer any pending query saved before login
    pending_query = pending_queries.pop(thread_id, None)

    if pending_query:
        logger.info(f"Auto-answering pending query: {pending_query}")
        pending_answer = ResponseTool.generate_from_user_data(
            query=pending_query,
            user_data=user_data,
            conversation_context="",
        )
        add_to_conversation_memory(
            thread_id, "user", f"[AUTO-ANSWERED] {pending_query}"
        )
        add_to_conversation_memory(thread_id, "assistant", pending_answer)

        state["answer"] = (
            # f'Your earlier question: "{pending_query}"\n\n'
            f"{pending_answer}"
        )
        state["metadata"]["pending_query_auto_answered"] = True
        state["metadata"]["pending_query"] = pending_query
    else:
        state["answer"] = (
            "Login successful!\n\n"
            "You can now ask about your internships, payments, grievances, and more."
        )

    state["response_type"] = "auth"
    return state


# -----------------------------------------------
# RAG Pipeline Nodes
# -----------------------------------------------


def doc_filter_node(state: RAGState) -> RAGState:
    """Filter documents by semantic similarity to the query (document-level)."""
    top_docs = []
    if doc_summary_store:
        try:
            emb = get_embedder()
            q_emb = emb.embed_query(state["query"])
            qv = np.array(q_emb, dtype=np.float32)
            sims = []
            for docid, info in doc_summary_store.items():
                v = np.array(info["summary_vector"], dtype=np.float32)
                sim = float(
                    np.dot(qv, v) / (np.linalg.norm(qv) * np.linalg.norm(v) + 1e-8)
                )
                sims.append((docid, sim))
            sims = sorted(sims, key=lambda x: x[1], reverse=True)[:TOP_K_DOCS]
            top_docs = [d for d, _ in sims]
        except Exception as e:
            logger.warning(f"Doc filter error: {e}")
    state["top_docs"] = top_docs
    return state


def semantic_search_node(state: RAGState) -> RAGState:
    """Retrieve relevant chunks using FAISS semantic (dense) search."""
    if not faiss_db:
        state["semantic_docs"] = []
        return state
    try:
        docs = faiss_db.similarity_search(state["query"], k=state["k"])
        # If we have a document-level filter, apply it; otherwise keep all results
        if state.get("top_docs"):
            filtered = [
                d for d in docs if d.metadata.get("source") in state["top_docs"]
            ]
            docs = filtered if filtered else docs[: state["k"]]
        state["semantic_docs"] = docs
    except Exception as e:
        logger.warning(f"Semantic search error: {e}")
        state["semantic_docs"] = []
    return state


def keyword_search_node(state: RAGState) -> RAGState:
    """Retrieve relevant chunks using BM25 keyword (sparse) search."""
    if not bm25_index:
        state["keyword_docs"] = []
        return state
    try:
        tokens = state["query"].lower().split()
        scores = bm25_index.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][: state["k"]]
        docs = []
        all_docs = [] 
        for idx in top_idx:
            if scores[idx] > 0:
                doc = Document(
                    page_content=bm25_corpus_texts[idx],
                    metadata=bm25_docs_metadata[idx],
                )
                all_docs.append(doc)
                if (
                    state.get("top_docs")
                    and bm25_docs_metadata[idx].get("source") not in state["top_docs"]
                ):
                    continue
                docs.append(doc)

        state["keyword_docs"] = (docs if docs else all_docs)[: state["k"]]
    except Exception as e:
        logger.warning(f"Keyword search error: {e}")
        state["keyword_docs"] = []
    return state
    return state


def merge_results_node(state: RAGState) -> RAGState:
    """
    Merge semantic and keyword results using Reciprocal Rank Fusion (RRF).
    De-duplicates chunks and re-ranks them by combined score.
    """
    sem = state.get("semantic_docs", [])
    kw = state.get("keyword_docs", [])
    scores = {}
    for rank, doc in enumerate(sem, 1):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0.0) + 1.0 / (rank + 60)
    for rank, doc in enumerate(kw, 1):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0.0) + 1.0 / (rank + 60)
    seen = set()
    merged = []
    for doc in sem + kw:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            merged.append((scores[key], doc))
    merged = sorted(merged, key=lambda x: x[0], reverse=True)
    state["merged_docs"] = [d for _, d in merged[: state["k"]]]
    return state


def assemble_context_node(state: RAGState) -> RAGState:
    """Build the context string for the LLM from merged document chunks."""
    docs = state.get("merged_docs", [])
    if not docs:
        state["context"] = ""
        return state
    parts = []
    total_chars = 0
    char_limit = 3500
    for d in docs:
        if total_chars >= char_limit:
            break
        src = d.metadata.get("source", "unknown")
        chunk = f"[{src}]\n{d.page_content.strip()}"
        if total_chars + len(chunk) > char_limit:
            break
        parts.append(chunk)
        total_chars += len(chunk)
    state["context"] = "\n\n".join(parts)
    return state


def generate_answer_node(state: RAGState) -> RAGState:
    """Generate the final answer using the LLM and the assembled context."""
    context = state.get("context", "")
    query = state.get("query", "").strip().lower()
    docs = state.get("merged_docs", [])

    logger.info(
        f"RAG context for query '{state.get('query','')}' "
        f"({len(docs)} chunks):\n{context}"
    )

    # greetings = {
    #     "hi",
    #     "hello",
    #     "hey",
    #     "bye",
    #     "goodbye",
    #     "thank you",
    #     "thanks",
    #     "good morning",
    #     "good evening",
    #     "good night",
    #     "good afternoon",
    #     "thankyou",
    #     "ok",
    #     "okay",
    #     "welcome",
    # }
    # if any(query == g or query.startswith(g) for g in greetings):
    #     state["answer"] = "Hello! How can I help you today?"
    #     return state

    if not context:
        state["answer"] = (
            "I don't have information regarding this. "
            "Please try with a different question."
        )
        return state

    if not model:
        # Model not loaded - return raw context as fallback
        state["answer"] = f"[Model not loaded - raw context]\n\n{context}"
        return state
    try:
        current_dt = state.get("current_datetime", {})
        prompt = RAG_PROMPT.format(
            context=context,
            question=state["query"],
            current_datetime=current_dt.get("formatted", ""),
        )
        SEP = "─" * 60
        intent_label = state.get("intent", "general")
        confidence_pct = int(state.get("intent_confidence", 1.0) * 100)
        sources_set = {d.metadata.get("source", "?") for d in docs}
        sources_str = ", ".join(sorted(sources_set))

        print(f"\n  You > {state['query']}\n  Thinking\u2026", flush=True)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_length = inputs["input_ids"].shape[1]
        new_tokens = out[0][
            input_length - 1 :
        ]  # Include the last token of the prompt to ensure we capture the full answer, in case the model doesn't generate any new tokens (e.g. if it thinks the prompt already contains the answer). This is a common edge case with some models where they might just return the input if they think it's sufficient. By including the last token, we ensure that we at least get something back to decode, even if it's just the prompt repeated.
        # new_tokens = out[0][input_length :]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        state["answer"] = answer if answer else "Could not generate an answer."
        state["metadata"]["chunks"] = len(docs)
        state["metadata"]["sources"] = list(
            {d.metadata.get("source", "?") for d in docs}
        )
        answer_lines = "\n    ".join(state["answer"].splitlines())
        print(
            f"{SEP}\n  Bot:\n    {answer_lines}\n{SEP}\n"
            f"  [RAG | intent: {intent_label} ({confidence_pct}%) | chunks: {len(docs)} | docs: {sources_str}]\n",
            flush=True,
        )

    except Exception as e:
        logger.error(f"Answer generation error: {e}")
        state["answer"] = f"Error during answer generation: {e}"
    return state


# -----------------------------------------------
# Build LangGraph Workflow
# -----------------------------------------------
def build_rag_graph():
    """
    Build the RAG + Auth workflow as a LangGraph StateGraph.

    Flow:
        initialize_session
            -> [route_after_session_init]   <-- NO LLM here
                -> handle_phone   (END)     <-- user entered phone number
                -> verify_otp     (END)     <-- user entered OTP
                -> classify_intent          <-- real question: LLM runs here
                    -> [route_after_intent]
                        -> user_specific    (END)   <-- personal question
                        -> document_filter          <-- general question
                            -> semantic_search
                            -> keyword_search
                            -> merge_results
                            -> assemble_context
                            -> generate_answer  (END)

    Key design: The LLM (classify_intent) is ONLY called when the user sends
    a real question. Phone numbers and OTPs are handled with pure regex/dict
    lookups - no model inference needed.
    """
    workflow = StateGraph(RAGState)

    # Register all nodes
    workflow.add_node("initialize_session", agent_initialize_session)
    workflow.add_node("classify_intent", agent_classify_intent)
    workflow.add_node("handle_phone", agent_handle_phone_input)
    workflow.add_node("not_valid_query", agent_not_valid_query)
    workflow.add_node("verify_otp", agent_verify_otp)
    workflow.add_node("user_specific", agent_user_specific_response)
    workflow.add_node("document_filter", doc_filter_node)
    workflow.add_node("semantic_search", semantic_search_node)
    workflow.add_node("keyword_search", keyword_search_node)
    workflow.add_node("merge_results", merge_results_node)
    workflow.add_node("assemble_context", assemble_context_node)
    workflow.add_node("generate_answer", generate_answer_node)

    # Entry point
    workflow.set_entry_point("initialize_session")

    # Step 1: After session init, route BEFORE calling the LLM
    # Phone/OTP inputs bypass classify_intent entirely
    workflow.add_conditional_edges(
        "initialize_session",
        route_after_session_init,
        {
            "handle_phone": "handle_phone",  # No LLM
            "verify_otp": "verify_otp",  # No LLM
            "classify_intent": "classify_intent",  # LLM runs here
        },
    )

    # Step 2: After LLM intent classification, route to answer handler
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "not_valid_query": "not_valid_query",
            "user_specific": "user_specific",
            "document_filter": "document_filter",
        },
    )

    # Terminal nodes
    workflow.add_edge("not_valid_query", END)

    workflow.add_edge("handle_phone", END)
    workflow.add_edge("verify_otp", END)
    workflow.add_edge("user_specific", END)

    # RAG pipeline (sequential to avoid InvalidUpdateError)
    workflow.add_edge("document_filter", "semantic_search")
    workflow.add_edge("semantic_search", "keyword_search")
    workflow.add_edge("keyword_search", "merge_results")
    workflow.add_edge("merge_results", "assemble_context")
    workflow.add_edge("assemble_context", "generate_answer")
    workflow.add_edge("generate_answer", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


graph = None


# -----------------------------------------------
# API Request / Response Models
# -----------------------------------------------
class ChatRequest(BaseModel):
    query: str
    thread_id: str  # Unique thread identifier - each new user/chat gets a new thread_id


class ChatResponse(BaseModel):
    answer: str
    metadata: Dict[str, Any]
    conversation_history: Optional[List[Dict]] = None
    pending_query_answered: Optional[str] = None


# -----------------------------------------------
# API Endpoints
# -----------------------------------------------
@app.on_event("startup")
async def startup():
    """Load FAISS, BM25, and LLM on server start."""
    global graph
    logger.info("Server starting up...")
    load_faiss()
    load_model()
    graph = build_rag_graph()
    logger.info("Server ready.")


@app.post("/chat")
def chat(req: ChatRequest):

    if not graph:
        raise HTTPException(status_code=503, detail="System not ready. Please wait.")

    raw_query = req.query.strip()
    if not raw_query:
        raise HTTPException(status_code=400, detail="Kindly provide a question.")

    # Guardrails check
    restricted_terms = ["prime minister", "pmis", "prime minister internship scheme"]
    query_lower = raw_query.lower()
    if any(term in query_lower for term in restricted_terms):
         raise HTTPException(status_code=400, detail="Your query contains restricted or sensitive terms and cannot be processed.")

    thread_id = req.thread_id

    # ── All validation now happens inside the graph via agent_not_valid_query ──

    try:
        state = {
            "query": raw_query,
            "thread_id": thread_id,
            "k": TOP_K_CHUNKS,
            "metadata_filters": None,
            "intent": "general",
            "is_user_specific": False,
            "is_not_valid_query": False,
            "has_date_query": False,
            "current_datetime": {},
            "intent_confidence": 0.0,
            "entities": {},
            "session_info": None,
            "is_authenticated": False,
            "awaiting_phone": False,
            "awaiting_otp": False,
            "phone_number": None,
            "user_data": None,
            "top_docs": [],
            "semantic_docs": [],
            "keyword_docs": [],
            "merged_docs": [],
            "context": "",
            "answer": "",
            "error": None,
            "metadata": {},
            "response_type": "general",
        }

        config = {"configurable": {"thread_id": thread_id}}
        final = graph.invoke(state, config)

        answer = final.get("answer", "No answer generated.")
        metadata = final.get("metadata", {})

        add_to_conversation_memory(thread_id, "user", raw_query)
        add_to_conversation_memory(thread_id, "assistant", answer)

        history = get_conversation_memory(thread_id)

        return JSONResponse(
            content={
                "answer": answer,
                "metadata": metadata,
                "conversation_history": history,
                "pending_query_answered": metadata.get("pending_query"),
            },
            status_code=200,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/history/{thread_id}")
def get_history(thread_id: str):
    """Return the full conversation history for a session."""
    history = get_conversation_memory(thread_id)
    return {
        "thread_id": thread_id,
        "message_count": len(history),
        "messages": history,
    }

@app.get("/debug/faiss")
def debug_faiss():
    return {
        "faiss_loaded": faiss_db is not None,
        "bm25_loaded": bm25_index is not None,
        "model_loaded": model is not None,
        "vectorstore_dir_exists": os.path.exists(VECTORSTORE_DIR),
        "vectorstore_dir": VECTORSTORE_DIR,
        "doc_count": len(bm25_corpus_texts) if bm25_corpus_texts else 0,
    }

@app.delete("/history/{thread_id}")
def clear_history(thread_id: str):
    """Clear conversation history for a session."""
    if thread_id in conversation_memory:
        conversation_memory[thread_id] = []
    return {"message": "History cleared.", "thread_id": thread_id}


@app.get("/pending/{thread_id}")
def check_pending(thread_id: str):
    """Check if a session has a saved pending query."""
    pending = pending_queries.get(thread_id)
    return {
        "thread_id": thread_id,
        "has_pending": pending is not None,
        "pending_query": pending,
    }


@app.get("/session/{thread_id}")
def get_session_status(thread_id: str):
    """Return current session status (for debugging)."""
    session = sessions.get(thread_id, {})
    return {
        "thread_id": thread_id,
        "exists": bool(session),
        "is_authenticated": session.get("is_authenticated", False),
        "awaiting_phone": session.get("awaiting_phone", False),
        "awaiting_otp": session.get("awaiting_otp", False),
        "user_id": session.get("user_id"),
        "has_cached_data": thread_id in user_cache,
    }


@app.get("/datetime")
def get_datetime():
    """Return the current server date and time."""
    return datetime_tool.get_current_datetime()


@app.get("/health")
def health():
    """Return server health status."""
    return {
        "status": "healthy",
        "faiss_loaded": faiss_db is not None,
        "model_loaded": model is not None,
        "graph_loaded": graph is not None,
        "total_sessions": len(sessions),
        "authenticated_sessions": sum(
            1 for s in sessions.values() if s.get("is_authenticated")
        ),
        "pending_queries_count": len(pending_queries),
    }


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RAG API v5.0 - OTP Authentication + Session Management")
    print("=" * 70)
    print("\nAuthentication Flow:")
    print("  1. User asks a user-specific question (e.g., 'my application status')")
    print("  2. Not logged in -> question saved, bot asks for phone number")
    print("  3. User enters phone (XXXXXXXXXX) -> OTP generated")
    print(f"  4. OTP is sent by the real API to the user's phone")
    print("  5. User enters OTP -> verified, pending question auto-answered")
    print("  6. Session stays active; future questions answered directly")
    print("  7. Session expires after 1 hour of inactivity -> re-auth required")
    print("\nEndpoints:")
    print("  POST   /chat                  - Main chat endpoint")
    print("  GET    /history/{thread_id}  - Conversation history")
    print("  DELETE /history/{thread_id}  - Clear history")
    print("  GET    /pending/{thread_id}  - Check pending query")
    print("  GET    /session/{thread_id}  - Session status (debug)")
    print("  GET    /datetime              - Current date/time")
    print("  GET    /health                - System health")
    print("=" * 70 + "\n")

    uvicorn.run(
        "agents:app",
        host="0.0.0.0",
        port=8634,
        reload=False,
        log_level="info",
    )
