"""
mock_user_api.py  –  Mock Number-Check Service
─────────────────────────────────────────────────────────────────────────────
Simulates the DB lookup and OTP verification.

Run:
    python mock_user_api.py
"""

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

app = FastAPI(title="Internship Mock API")

# Simulated DB of registered mobile numbers
REGISTERED_NUMBERS = {
    "7668455121",  # primary test account
    "8881164451",  # secondary test account
    "9999999999",  # tertiary test account
}

class MobileRequest(BaseModel):
    mobile: str

class SignInRequest(BaseModel):
    mobile: str
    otp: str
    fp: str = ""

@app.post("/auth/chat/mob_verification")
def mob_verification(req: MobileRequest):
    """
    Called by agents.py (API 1) BEFORE sending OTP.
    """
    present = req.mobile in REGISTERED_NUMBERS
    print(f"[API 1] check: mobile={req.mobile}  present={present}")
    if present:
        return {"code": True, "message": "OTP sent successfully."}
    else:
        # FastAPI can return custom status code if needed, but returning dict is fine if agents.py checks 400
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail={"error": "Invalid mobile number"})

@app.post("/auth/chat/sign-in")
def sign_in(req: SignInRequest):
    if req.otp == "123456" and req.mobile in REGISTERED_NUMBERS:
        return {"token": "mock_jwt_token_123456"}
    from fastapi import HTTPException
    raise HTTPException(status_code=400, detail={"error": "Invalid OTP"})

@app.post("/chatbot/candidate_details_internship")
def candidate_details():
    return {
        "candidate_name": "Test User",
        "internships": [
            {"job_role_name": "Software Developer", "company_name": "Tech Corp", "status": "Active", "start_date": "2026-01-01", "end_date": "2026-06-01", "stipend_amount": 10000}
        ],
        "pfmsdbt": [],
        "grievances": []
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "registered_numbers": len(REGISTERED_NUMBERS),
    }

@app.get("/")
def root():
    return {
        "service": "Internship Mock API",
        "endpoints": [
            "POST /auth/chat/mob_verification",
            "POST /auth/chat/sign-in",
            "POST /chatbot/candidate_details_internship"
        ],
        "test_numbers": list(REGISTERED_NUMBERS),
    }

if __name__ == "__main__":
    print("=" * 60)
    print("  Mock API for Internship")
    print("=" * 60)
    print(f"\n  Listening on  : http://localhost:8900")
    print(f"\n  Registered test numbers:")
    for n in sorted(REGISTERED_NUMBERS):
        print(f"    {n}")
    print("\n  Use OTP '123456' for sign-in.")
    print()

    uvicorn.run(
        "mock_user_api:app",
        host="localhost",
        port=8900,
        reload=False,
        log_level="info",
    )
