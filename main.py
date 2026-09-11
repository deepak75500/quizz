from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import traceback

# Your existing question extraction module
from pp import extract_questions


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="UGC NET Question Extraction API",
    description="API for extracting UGC NET questions",
    version="1.0.0",
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ExtractRequest(BaseModel):
    years: str
    number: int = Field(default=10, ge=1, le=100)
    paper: str = "Political Science"
    difficulty: str = "mixed"


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "UGC NET Question Extraction API",
        "runtime": "Cloudflare Python Worker",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "ok"
    }


# ============================================================
# EXTRACT QUESTIONS
# ============================================================

@app.post("/extract")
async def extract(request: ExtractRequest):

    print("=" * 70)
    print("EXTRACT REQUEST RECEIVED")
    print(f"years      = {request.years}")
    print(f"number     = {request.number}")
    print(f"paper      = {request.paper}")
    print(f"difficulty = {request.difficulty}")
    print("=" * 70)

    try:

        result = await extract_questions(
            years=request.years,
            number=request.number,
            paper=request.paper,
            difficulty=request.difficulty,
        )

        questions = []

        if isinstance(result, dict):
            questions = result.get("questions", [])

        print("EXTRACTION COMPLETED")
        print(f"Questions returned: {len(questions)}")

        return result

    except Exception as e:

        print("=" * 70)
        print("EXTRACTION ERROR")
        print(repr(e))
        traceback.print_exc()
        print("=" * 70)

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================
# CLOUDFLARE ASGI ENTRYPOINT
# ============================================================

from workers import asgi

Default = asgi.entrypoint(app)
