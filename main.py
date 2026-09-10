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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from pp import extract_questions


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
        default="Paper 1"
    )

    difficulty: str = Field(
        default="mixed"
    )

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, value):

        value = value.strip().lower()

        if value not in {
            "easy",
            "medium",
            "hard",
            "mixed"
        }:
            raise ValueError(
                "difficulty must be easy, medium, hard, or mixed"
            )

        return value


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
