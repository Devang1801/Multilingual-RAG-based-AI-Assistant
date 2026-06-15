"""
test.py  –  PMIS Interactive Chat Client
─────────────────────────────────────────────────────────────────────────────
Connects to agents.py running on port 8634.
The user can type ANYTHING – general PMIS questions, personal questions,
phone numbers, OTPs – the server handles all routing automatically.

Start the server first:
    python agents.py

Then run this client:
    python test.py

Commands:
    /health   – Check server + API health
    /session  – Show your session / auth status
    /pending  – Check if you have a saved unanswered question
    /history  – Show last 10 messages in this session
    /clear    – Clear conversation history
    /exit     – Quit

Example questions you can ask:
    General:
        What is PMIS?
        Who is eligible for PMIS?
        What is the application deadline?
        How do I apply for internship?

    Personal (triggers login flow):
        What is my internship status?
        Show my payment details
        How many payments have I received?
        Do I have any grievances?
        Which company am I assigned to?
        Show all my details
        What is my stipend?
        When does my internship start?
"""

import requests
import uuid

BASE_URL = "http://127.0.0.1:8634"

# Each run gets a unique thread ID so sessions don't overlap between test runs
THREAD_ID = str(uuid.uuid4())


# ─────────────────────────────────────────────────────────────────────────────
# Core Chat Function
# ─────────────────────────────────────────────────────────────────────────────
def chat(query: str) -> dict:
    """Send one message to /chat and return the response dict."""
    try:
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"query": query, "thread_id": THREAD_ID},
            timeout=520,
        )
        if resp.status_code == 200:
            return resp.json()
        return {
            "answer": f"Server error {resp.status_code}: {resp.text[:200]}",
            "metadata": {},
        }
    except requests.exceptions.ConnectionError:
        return {
            "answer": "Cannot connect to the server. Make sure agents.py is running on port 8634.",
            "metadata": {},
        }
    except requests.exceptions.Timeout:
        return {
            "answer": "Request timed out – model is still processing. Try again in a moment.",
            "metadata": {},
        }
    except Exception as e:
        return {"answer": f"Unexpected error: {e}", "metadata": {}}


# ─────────────────────────────────────────────────────────────────────────────
# Response Printer
# ─────────────────────────────────────────────────────────────────────────────
def print_response(result: dict):
    answer = result.get("answer", "No answer returned.")
    meta = result.get("metadata", {})

    print("\n" + "─" * 60)
    print("  Bot:\n")
    for line in answer.split("\n"):
        print(f"    {line}")
    print()
    print("─" * 60)

    # ── One-line status hint ───────────────────────────────────────────────
    source = meta.get("source", "")
    intent = meta.get("intent", "")
    confidence = meta.get("confidence", 0)
    chunks = meta.get("chunks", 0)
    sources = meta.get("sources", [])

    if source == "user_api":
        uid = meta.get("user_id", "")
        print(f"  [✓ Answered from your API data | user: {uid}]")

    elif source == "no_auth":
        print("  [→ Not logged in – question saved]")
        print("  [→ Enter your 10-digit registered mobile number to log in]")

    elif meta.get("pending_query_auto_answered"):
        saved = meta.get("pending_query", "")
        print(f'  [✓ Auto-answered saved question: "{saved}"]')

    elif intent == "general" and chunks:
        pct = int(confidence * 100) if confidence <= 1.0 else int(confidence)
        docs = ", ".join(sources) if sources else "N/A"
        print(f"  [RAG | intent: general ({pct}%) | chunks: {chunks} | docs: {docs}]")

    elif intent:
        pct = int(confidence * 100) if confidence <= 1.0 else int(confidence)
        print(f"  [intent: {intent} ({pct}%)]")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Command Handlers
# ─────────────────────────────────────────────────────────────────────────────
def cmd_health():
    """Check all 3 API services + server health."""
    print()
    try:
        data = requests.get(f"{BASE_URL}/health", timeout=10).json()
        print(f"  Server status       : {data.get('status')}")
        print(f"  FAISS loaded        : {data.get('faiss_loaded')}")
        print(f"  Model loaded        : {data.get('model_loaded')}")
        print(f"  Graph loaded        : {data.get('graph_loaded')}")
        print(f"  Active sessions     : {data.get('total_sessions', 0)}")
        print(f"  Authenticated       : {data.get('authenticated_sessions', 0)}")
        print(f"  Pending queries     : {data.get('pending_queries_count', 0)}")
    except Exception as e:
        print(f"  Server health check failed: {e}")

    # Quick ping to API 1
    print()
    try:
        resp = requests.post(
            "http://localhost:8900/verify_number/0000000000", timeout=3
        )
        print(f"  API 1 (number check): ✓ reachable (HTTP {resp.status_code})")
    except Exception:
        print("  API 1 (number check): ✗ unreachable (localhost:8900)")

    # Quick ping to API 2/3 base
    try:
        resp = requests.get("http://192.168.1.52:4050", timeout=3)
        print(f"  API 2/3 (auth+data) : ✓ reachable (HTTP {resp.status_code})")
    except Exception:
        print("  API 2/3 (auth+data) : ✗ unreachable (192.168.1.52:4050)")
    print()


def cmd_session():
    try:
        data = requests.get(f"{BASE_URL}/session/{THREAD_ID}", timeout=10).json()
        print(f"\n  Thread ID           : {data.get('thread_id')}")
        print(f"  Authenticated       : {data.get('is_authenticated')}")
        print(f"  Awaiting phone      : {data.get('awaiting_phone')}")
        print(f"  Awaiting OTP        : {data.get('awaiting_otp')}")
        print(f"  User ID             : {data.get('user_id', 'N/A')}")
        print(f"  User data cached    : {data.get('has_cached_data')}")
        print()
    except Exception as e:
        print(f"\n  Session check failed: {e}\n")


def cmd_pending():
    try:
        data = requests.get(f"{BASE_URL}/pending/{THREAD_ID}", timeout=10).json()
        if data.get("has_pending"):
            print(f'\n  Saved question: "{data.get("pending_query")}"\n')
        else:
            print("\n  No pending question.\n")
    except Exception as e:
        print(f"\n  Pending check failed: {e}\n")


def cmd_history():
    try:
        data = requests.get(f"{BASE_URL}/history/{THREAD_ID}", timeout=10).json()
        messages = data.get("messages", [])
        total = data.get("message_count", 0)
        print(f"\n  History ({total} messages, showing last 10):\n")
        for msg in messages[-10:]:
            role = "  You " if msg["role"] == "user" else "  Bot "
            content = msg["content"]
            if len(content) > 90:
                content = content[:90] + "..."
            print(f"  {role}: {content}")
        print()
    except Exception as e:
        print(f"\n  History failed: {e}\n")


def cmd_clear():
    try:
        requests.delete(f"{BASE_URL}/history/{THREAD_ID}", timeout=10)
        print("\n  Conversation history cleared.\n")
    except Exception as e:
        print(f"\n  Clear failed: {e}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Startup Banner
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  PMIS Interactive Chat Client")
print("=" * 60)
print(f"  Server    : {BASE_URL}")
print(f"  Thread ID : {THREAD_ID}")
print("=" * 60)
print("  Commands  : /health  /session  /pending  /history  /clear  /exit")
print("=" * 60)
print()
print("  You can ask ANYTHING, for example:")
print()
print("  General questions (no login needed):")
print("    What is PMIS?")
print("    Who is eligible?")
print("    What is the application deadline?")
print()
print("  Personal questions (triggers login flow):")
print("    What is my internship status?")
print("    Show my payment details")
print("    How many payments have I received?")
print("    Do I have any grievances?")
print("    Show all my details")
print()
print("  During login:")
print("    Step 1 – Enter your 10-digit mobile number (e.g. 9876543210)")
print("    Step 2 – Enter the 6-digit OTP sent to your phone")
print()

# Quick server health on start
try:
    h = requests.get(f"{BASE_URL}/health", timeout=5).json()
    print(
        f"  Server   : {h.get('status','unknown')}  |  "
        f"FAISS: {h.get('faiss_loaded')}  |  "
        f"Model: {h.get('model_loaded')}  |  "
        f"Graph: {h.get('graph_loaded')}"
    )
    if not h.get("graph_loaded"):
        print("  WARNING: Graph not loaded – server may still be starting.")
except requests.exceptions.ConnectionError:
    print(f"  WARNING: Cannot connect to {BASE_URL}")
    print("  Start the server with:  python agents.py")
print()


# ─────────────────────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────────────────────
while True:
    try:
        user_input = input("  You > ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n\n  Goodbye.\n")
        break

    if not user_input:
        continue

    cmd = user_input.lower()

    if cmd in ("/exit", "/quit", "exit", "quit"):
        print("\n  Goodbye.\n")
        break

    elif cmd == "/health":
        cmd_health()

    elif cmd == "/session":
        cmd_session()

    elif cmd == "/pending":
        cmd_pending()

    elif cmd == "/history":
        cmd_history()

    elif cmd == "/clear":
        cmd_clear()

    else:
        print("  Thinking…", end="\r", flush=True)
        result = chat(user_input)
        print_response(result)
