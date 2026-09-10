from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os

os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "/opt/render/project/src/.playwright"

from pp import ...
import traceback
from pathlib import Path
from typing import Any
import os
import re
import json
import asyncio



from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field, field_validator

os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "/tmp/ms-playwright"
from playwright.async_api import (
    async_playwright,
    TimeoutError as PlaywrightTimeoutError
)
from pp import extract_questions

app = FastAPI()


class ExtractRequest(BaseModel):
    years: str
    number: int = Field(default=10, ge=1, le=100)
    paper: str = "Political Science"
    difficulty: str = "mixed"


@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "UGC NET Question Extraction API"
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


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
            difficulty=request.difficulty
        )

        print("EXTRACTION COMPLETED")
        print(f"Questions returned: {len(result.get('questions', []))}")

        return result

    except Exception as e:
        print("=" * 70)
        print("EXTRACTION ERROR")
        print(repr(e))
        traceback.print_exc()
        print("=" * 70)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
