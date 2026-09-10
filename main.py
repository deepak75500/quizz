from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pp import extract_questions

from pathlib import Path
from typing import Any
import os
import re
import json
import asyncio

os.environ.setdefault(
    "PLAYWRIGHT_BROWSERS_PATH",
    "/opt/render/project/src/.playwright"
)

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
app = FastAPI(
    title="UGC NET Question Extractor API",
    version="1.0.0"
)


class ExtractRequest(BaseModel):
    years: str = Field(
        ...,
        example="2019"
    )

    number: int = Field(
        default=10,
        ge=1,
        le=100
    )

    paper: str = Field(
        default="Paper 1",
        example="Paper 1"
    )

    difficulty: str = Field(
        default="mixed",
        example="mixed"
    )


@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "UGC NET Question Extractor"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }


@app.post("/extract")
async def extract(request: ExtractRequest):

    try:

        result = await extract_questions(
            years=request.years,
            number=request.number,
            paper=request.paper,
            difficulty=request.difficulty
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000
    )
