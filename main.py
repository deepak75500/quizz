import os
import re
import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import os

os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "/tmp/ms-playwright"

from playwright.async_api import (
    async_playwright,
    TimeoutError as PlaywrightTimeoutError
)
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field, field_validator
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "UGC-NET Web Research API"

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = DATA_DIR / "output"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("UGCNET_API_KEY", "")

PAGE_TIMEOUT = int(os.getenv("PAGE_TIMEOUT", "60000"))

research_lock = asyncio.Lock()


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title=APP_NAME,
    version="1.0.0",
    description="URL-based UGC-NET research API using Playwright"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ResearchRequest(BaseModel):
    topic: str = Field(
        ...,
        min_length=1,
        max_length=500
    )

    year: str = Field(
        default="any",
        min_length=1,
        max_length=100
    )

    category: str | list[str] = Field(
        default="MCQ"
    )

    count: int = Field(
        default=20,
        ge=1,
        le=500
    )

    url: str = Field(
        ...,
        min_length=5,
        max_length=5000
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str):
        value = value.strip()

        if not re.match(
            r"^https?://",
            value,
            re.IGNORECASE
        ):
            raise ValueError(
                "URL must start with http:// or https://"
            )

        return value


# ============================================================
# RESPONSE MODEL
# ============================================================

class ResearchResponse(BaseModel):
    success: bool
    request: dict[str, Any]
    page: dict[str, Any]
    questions: list[dict[str, Any]]
    count: int
    generated_at: str


# ============================================================
# CATEGORY NORMALIZATION
# ============================================================

def normalize_categories(
    category: str | list[str]
) -> list[str]:

    if isinstance(category, str):
        values = [category]
    else:
        values = category

    result = []

    for value in values:
        value = str(value).strip()

        if value and value not in result:
            result.append(value)

    return result or ["MCQ"]


# ============================================================
# YEAR NORMALIZATION
# ============================================================

def normalize_year(year: str) -> str:

    year = year.strip()

    if not year:
        return "any"

    return year


# ============================================================
# CREATE RESEARCH DESCRIPTION
# ============================================================

def create_request_description(
    request: ResearchRequest
) -> dict[str, Any]:

    return {
        "topic": request.topic.strip(),
        "year": normalize_year(request.year),
        "category": normalize_categories(request.category),
        "count": request.count,
        "url": request.url.strip()
    }


# ============================================================
# PLAYWRIGHT PAGE EXTRACTION
# ============================================================

async def extract_page_content(
    url: str
) -> dict[str, Any]:

    async with async_playwright() as playwright:

        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--no-zygote",
                "--single-process"
            ]
        )

        context = await browser.new_context(
            viewport={
                "width": 1440,
                "height": 900
            },
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/139.0.0.0 Safari/537.36"
            )
        )

        page = await context.new_page()

        try:

            response = await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=PAGE_TIMEOUT
            )

            # Give dynamic pages some time
            try:
                await page.wait_for_load_state(
                    "networkidle",
                    timeout=15000
                )
            except PlaywrightTimeoutError:
                pass

            title = await page.title()

            # Remove elements that normally contain
            # non-content information.
            await page.evaluate(
                """
                () => {
                    const selectors = [
                        'script',
                        'style',
                        'noscript',
                        'svg',
                        'canvas',
                        'iframe'
                    ];

                    for (const selector of selectors) {
                        document
                            .querySelectorAll(selector)
                            .forEach(el => el.remove());
                    }
                }
                """
            )

            text = await page.locator("body").inner_text(
                timeout=15000
            )

            text = clean_text(text)

            html = await page.content()

            final_url = page.url

            status_code = None

            if response:
                status_code = response.status

            return {
                "requested_url": url,
                "final_url": final_url,
                "title": title,
                "status_code": status_code,
                "text": text,
                "text_length": len(text),
                "html_length": len(html)
            }

        finally:

            await context.close()
            await browser.close()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:

    text = text.replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# QUESTION EXTRACTION
# ============================================================

def extract_questions_from_text(
    text: str,
    requested_count: int,
    topic: str,
    year: str,
    categories: list[str],
    source_url: str
) -> list[dict[str, Any]]:

    questions = []

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # --------------------------------------------------------
    # Pattern 1:
    # 1. Question text
    # --------------------------------------------------------

    question_pattern = re.compile(
        r"^(?:Q(?:uestion)?\s*)?(\d{1,4})[\.\):\-]\s*(.+)$",
        re.IGNORECASE
    )

    current_question = None

    for line in lines:

        match = question_pattern.match(line)

        if match:

            if current_question:

                questions.append(
                    current_question
                )

            number = int(match.group(1))
            question_text = match.group(2).strip()

            current_question = {
                "id": number,
                "question": question_text,
                "options": {},
                "correct_answer": None,
                "explanation": None,
                "year": year,
                "category": categories,
                "topic": topic,
                "source": {
                    "url": source_url,
                    "name": source_url
                },
                "verified": False
            }

            continue

        # ----------------------------------------------------
        # Options
        # ----------------------------------------------------

        option_match = re.match(
            r"^\(?([A-Da-d])\)?[\.\):\-]\s*(.+)$",
            line
        )

        if option_match and current_question:

            option_letter = option_match.group(1).upper()

            option_text = option_match.group(2).strip()

            current_question["options"][
                option_letter
            ] = option_text

            continue

    if current_question:

        questions.append(current_question)

    # --------------------------------------------------------
    # Remove duplicate questions
    # --------------------------------------------------------

    unique = []

    seen = set()

    for question in questions:

        normalized = re.sub(
            r"\s+",
            " ",
            question["question"].lower()
        ).strip()

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)

        unique.append(question)

    return unique[:requested_count]


# ============================================================
# FIND ANSWERS FROM PAGE
# ============================================================

def find_answer_for_question(
    question: dict[str, Any],
    text: str
) -> str | None:

    question_number = question.get("id")

    # Look for common answer formats
    patterns = [
        rf"{question_number}\s*[-:.]?\s*answer\s*[:\-]\s*([A-D])",
        rf"question\s*{question_number}\s*answer\s*[:\-]\s*([A-D])",
        rf"q\.?\s*{question_number}\s*answer\s*[:\-]\s*([A-D])"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(1).upper()

    return None


# ============================================================
# VALIDATE QUESTION
# ============================================================

def validate_question(
    question: dict[str, Any]
) -> bool:

    if not question.get("question"):
        return False

    options = question.get("options")

    if not isinstance(options, dict):
        return False

    if len(options) < 2:
        return False

    return True


# ============================================================
# ENRICH QUESTIONS
# ============================================================

def enrich_questions(
    questions: list[dict[str, Any]],
    page_text: str
) -> list[dict[str, Any]]:

    for question in questions:

        answer = find_answer_for_question(
            question,
            page_text
        )

        if answer:
            question["correct_answer"] = answer
            question["verified"] = True

        question["question_type"] = (
            "MCQ"
        )

        question["graph_data"] = None
        question["table_data"] = None
        question["diagram_data"] = None

    return questions


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(
    result: dict[str, Any]
) -> str:

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        f"ugcnet_research_{timestamp}.json"
    )

    filepath = OUTPUT_DIR / filename

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2
        )

    latest_file = OUTPUT_DIR / "latest.json"

    with open(
        latest_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2
        )

    return str(filepath)


# ============================================================
# AUTHENTICATION
# ============================================================

def check_api_key(
    authorization: str | None
):

    # If no UGCNET_API_KEY is configured,
    # authentication is disabled.
    if not API_KEY:
        return

    if not authorization:

        raise HTTPException(
            status_code=401,
            detail="Authorization header required"
        )

    if not authorization.startswith(
        "Bearer "
    ):

        raise HTTPException(
            status_code=401,
            detail="Use Bearer authentication"
        )

    supplied_key = authorization[
        len("Bearer "):
    ].strip()

    if supplied_key != API_KEY:

        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )


# ============================================================
# RESEARCH
# ============================================================

async def perform_research(
    request: ResearchRequest
) -> dict[str, Any]:

    categories = normalize_categories(
        request.category
    )

    year = normalize_year(
        request.year
    )

    # --------------------------------------------------------
    # Open URL
    # --------------------------------------------------------

    page_data = await extract_page_content(
        request.url
    )

    page_text = page_data["text"]

    if not page_text:

        raise RuntimeError(
            "No readable text was found on the supplied URL"
        )

    # --------------------------------------------------------
    # Extract questions
    # --------------------------------------------------------

    questions = extract_questions_from_text(
        text=page_text,
        requested_count=request.count,
        topic=request.topic,
        year=year,
        categories=categories,
        source_url=page_data["final_url"]
    )

    # --------------------------------------------------------
    # Enrich
    # --------------------------------------------------------

    questions = enrich_questions(
        questions,
        page_text
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    valid_questions = []

    for question in questions:

        if validate_question(question):

            valid_questions.append(question)

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {
        "success": True,

        "request": {
            "topic": request.topic,
            "year": year,
            "category": categories,
            "count": request.count,
            "url": request.url
        },

        "page": {
            "requested_url": page_data[
                "requested_url"
            ],
            "final_url": page_data[
                "final_url"
            ],
            "title": page_data[
                "title"
            ],
            "status_code": page_data[
                "status_code"
            ],
            "text_length": page_data[
                "text_length"
            ]
        },

        "questions": valid_questions,

        "count": len(
            valid_questions
        ),

        "generated_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    save_result(result)

    return result


# ============================================================
# API ENDPOINT
# ============================================================

@app.post(
    "/research",
    response_model=ResearchResponse
)
async def research(
    request: ResearchRequest,
    authorization: str | None = Header(
        default=None
    )
):

    check_api_key(
        authorization
    )

    # Prevent multiple Chromium instances
    # from running simultaneously.
    async with research_lock:

        try:

            result = await perform_research(
                request
            )

            return result

        except PlaywrightTimeoutError:

            raise HTTPException(
                status_code=504,
                detail=(
                    "The supplied URL took too long "
                    "to load."
                )
            )

        except Exception as exc:

            raise HTTPException(
                status_code=500,
                detail=str(exc)
            )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "service": APP_NAME,
        "playwright": True
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "service": APP_NAME,
        "status": "running",
        "endpoints": {
            "research": "POST /research",
            "health": "GET /health"
        }
    }


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(
            os.getenv("PORT", "8000")
        ),
        reload=False
    )
