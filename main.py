import asyncio
import html
import json
import os
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright, Page


# ============================================================
# CONFIGURATION
# ============================================================

CHATGPT_URL = os.getenv(
    "CHATGPT_URL",
    "https://chatgpt.com/"
)

DATA_DIR = Path(
    os.getenv("DATA_DIR", "./cloud_data")
)

OUTPUT_DIR = DATA_DIR / "ugcnet_output"
PROFILE_DIR = DATA_DIR / "chatgpt_profile"

STORAGE_STATE = Path(
    os.getenv(
        "STORAGE_STATE",
        str(DATA_DIR / "storage_state.json")
    )
)

MAX_WAIT_SECONDS = int(
    os.getenv("MAX_WAIT_SECONDS", "600")
)

NAVIGATION_TIMEOUT = int(
    os.getenv("NAVIGATION_TIMEOUT", "60000")
)

API_KEY = os.getenv(
    "UGCNET_API_KEY",
    ""
)

DONE_MARKER = "<UGCNET_JSON_DONE>"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PROFILE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="UGC-NET Original Question Research API",
    description=(
        "Searches the web through ChatGPT and extracts "
        "verified original UGC-NET previous-year questions."
    ),
    version="2.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ResearchRequest(BaseModel):

    topic: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="UGC-NET topic or subject area"
    )

    year: str = Field(
        default="any",
        min_length=1,
        max_length=100,
        description=(
            "Year specification. Examples: "
            "2019, 2020-2025, 2019,2021,2023, any"
        )
    )

    category: str | list[str] = Field(
        default="MCQ",
        description=(
            "Question category. Can be a single category "
            "or multiple categories."
        )
    )

    count: int = Field(
        default=20,
        ge=1,
        le=500,
        description="Maximum number of verified questions"
    )


# ============================================================
# GLOBAL LOCK
# ============================================================

# Prevent two requests from using the same ChatGPT session
# simultaneously.
research_lock = asyncio.Lock()


# ============================================================
# API SECURITY
# ============================================================

def check_api_key(
    authorization: str | None
) -> bool:

    # If UGCNET_API_KEY is not configured,
    # authentication is disabled.
    if not API_KEY:
        return True

    if not authorization:
        return False

    expected = f"Bearer {API_KEY}"

    return authorization.strip() == expected


# ============================================================
# CATEGORY NORMALIZATION
# ============================================================

def normalize_categories(
    category: str | list[str]
) -> list[str]:

    if isinstance(category, list):

        result = []

        for item in category:

            value = str(item).strip()

            if value:
                result.append(value)

        return result

    if isinstance(category, str):

        value = category.strip()

        if not value:
            return ["MCQ"]

        # Allow:
        # "MCQ, Graph Based"
        # "MCQ;Graph Based"
        if "," in value:
            parts = value.split(",")

        elif ";" in value:
            parts = value.split(";")

        else:
            parts = [value]

        result = []

        for item in parts:

            item = item.strip()

            if item:
                result.append(item)

        return result or ["MCQ"]

    return ["MCQ"]


# ============================================================
# YEAR NORMALIZATION
# ============================================================

def normalize_years(
    year: str
) -> list[str]:

    value = str(year).strip()

    if not value:
        return ["any"]

    if value.lower() in {
        "any",
        "all",
        "any year",
        "all years"
    }:
        return ["any"]

    parts = re.split(
        r"[,;]",
        value
    )

    result = []

    for part in parts:

        part = part.strip()

        if part:
            result.append(part)

    return result or ["any"]


# ============================================================
# USER REQUEST DESCRIPTION
# ============================================================

def create_request_description(
    request: ResearchRequest
) -> str:

    categories = normalize_categories(
        request.category
    )

    years = normalize_years(
        request.year
    )

    return (
        f"Topic: {request.topic}\n"
        f"Year: {', '.join(years)}\n"
        f"Category: {', '.join(categories)}\n"
        f"Count: {request.count}"
    )


# ============================================================
# RESEARCH PROMPT
# ============================================================

def build_prompt(
    request: ResearchRequest
) -> str:

    categories = normalize_categories(
        request.category
    )

    years = normalize_years(
        request.year
    )

    output_schema = {
        "request": {
            "original_request": "",
            "years_requested": [],
            "sessions_requested": [],
            "subjects_requested": [],
            "papers_requested": [],
            "topics_requested": [],
            "question_types_requested": [],
            "difficulty_requested": [],
            "question_count_requested": 0
        },
        "questions": [
            {
                "id": 1,
                "question": "",
                "question_type": "",
                "topic": "",
                "subject": "",
                "paper": "",
                "year": "",
                "exam_date": "",
                "session": "",
                "shift": "",
                "language": "",
                "options": {
                    "A": "",
                    "B": "",
                    "C": "",
                    "D": ""
                },
                "correct_answer": "",
                "correct_option_text": "",
                "explanation": "",
                "source_name": "",
                "source_url": "",
                "source_type": "",
                "verified": True,
                "verification_notes": "",
                "graph_data": None,
                "table_data": None,
                "diagram_data": None
            }
        ],
        "statistics": {
            "requested": 0,
            "verified_returned": 0,
            "excluded_unverified": 0,
            "duplicates_removed": 0,
            "shortfall": 0
        }
    }

    schema_text = json.dumps(
        output_schema,
        ensure_ascii=False,
        indent=2
    )

    prompt = f"""
You are an expert research agent for UGC-NET previous-year
question papers.

Your task is NOT to generate questions.

Your task is to SEARCH THE WEB and find ACTUAL ORIGINAL
UGC-NET QUESTIONS that were previously asked in real
UGC-NET examinations.

============================================================
USER REQUEST
============================================================

Topic:
{request.topic}

Requested year:
{request.year}

Normalized years:
{json.dumps(years, ensure_ascii=False)}

Requested categories:
{json.dumps(categories, ensure_ascii=False)}

Requested count:
{request.count}

============================================================
CORE REQUIREMENT
============================================================

Return ONLY questions that can be verified as genuine
previous-year UGC-NET questions.

DO NOT:

- invent questions
- generate similar questions
- rewrite questions
- paraphrase questions
- create synthetic options
- guess the correct answer
- invent year
- invent shift
- invent exam date
- invent source URL
- fabricate graph data
- fabricate table data
- fabricate diagram information
- claim a question is original without evidence

If an exact original question cannot be verified,
DO NOT return it.

It is better to return fewer verified questions than
fabricated questions.

============================================================
WEB RESEARCH
============================================================

You MUST perform web research.

Search multiple sources where necessary.

Prioritize authoritative sources such as:

1. NTA UGC-NET official website
2. UGC official previous-question-paper archive
3. Official NTA answer keys
4. Official examination documents
5. Reliable copies of original question papers
6. Reliable educational archives only when necessary

Useful official domains include:

ugcnet.nta.nic.in
ugcnet.nta.nic.in/archive/
ugcnet.nta.nic.in/documents/
ugcnetonline.in

Do not assume that a search-result snippet proves
that a question is genuine.

Open and inspect the source.

============================================================
YEAR REQUIREMENT
============================================================

The year comes from the user's request.

Requested year:
{request.year}

If the request is a range such as:

2019-2025

search across that range.

Do not silently replace the requested range with another
range.

If the user specifies:

2021

do not return 2020 or 2022 questions.

If year is "any", search across available years.

============================================================
CATEGORY REQUIREMENT
============================================================

Requested category:

{json.dumps(categories, ensure_ascii=False)}

Possible categories include:

- MCQ
- Graph Based
- Table Based
- Data Interpretation
- Assertion Reason
- Statement Based
- Multiple Statement
- Matching
- Numerical
- Case Based
- Communication
- Teaching Aptitude
- Research Aptitude
- ICT
- Logical Reasoning
- Mathematical Reasoning
- Higher Education
- Environment
- People and Development
- Reading Comprehension
- Other

The category must describe the actual question.

Do not change a question into the requested category.

For example, if the source question is graph-based,
preserve it as graph-based.

============================================================
EXACT QUESTION
============================================================

Preserve the original wording as closely as possible.

Do not improve grammar.

Do not simplify wording.

Do not rewrite the question.

Preserve:

- numbers
- percentages
- mathematical symbols
- statements
- option wording
- sequences
- table values
- graph values
- labels

Minor OCR correction is allowed only when the source clearly
contains an OCR error and the original source confirms it.

============================================================
OPTIONS
============================================================

Capture ALL original options.

For normal four-option questions:

A
B
C
D

must all be present.

Do not invent a missing option.

For matching questions or questions with special option
structures, preserve the actual structure.

============================================================
ANSWER
============================================================

Find the correct answer from:

- official answer key
- official response/answer record
- reliable question paper with answer
- multiple reliable sources

Do not guess.

If the answer cannot be verified,
exclude the question.

============================================================
EXPLANATION
============================================================

Provide an explanation only after verifying the answer.

The explanation should explain why the correct option is
correct.

Do not invent an explanation that contradicts the source.

For numerical questions, show the reasoning.

For graph questions, explain the data.

============================================================
GRAPH QUESTIONS
============================================================

If the original question contains a graph/chart:

DO NOT generate an image.

Represent the graph as structured JSON data.

Example:

"graph_data": {{
    "chart_type": "bar",
    "title": "Example",
    "x_axis": ["A", "B", "C"],
    "y_axis": [10, 20, 30],
    "series": [
        {{
            "name": "Value",
            "data": [10, 20, 30]
        }}
    ]
}}

The values must come from the original question.

Do not estimate values.

If the graph cannot be reliably reconstructed,
preserve the question and set graph_data to null,
with an explanation in verification_notes.

============================================================
TABLE QUESTIONS
============================================================

If the original question contains a table:

Represent it as:

"table_data": {{
    "columns": ["Column 1", "Column 2"],
    "rows": [
        ["A", "10"],
        ["B", "20"]
    ]
}}

Use only values present in the source.

============================================================
DIAGRAM QUESTIONS
============================================================

Do not invent a diagram.

If the diagram contains identifiable structured data,
represent it in:

"diagram_data"

Otherwise preserve the question and explain the limitation
in verification_notes.

============================================================
SOURCE INFORMATION
============================================================

Every question MUST contain:

source_name
source_url
source_type
verified

source_type should be one of:

- official_nta
- official_ugc
- official_answer_key
- reliable_question_paper
- reliable_archive

source_url must be a real HTTP/HTTPS URL.

Do not use fake URLs.

============================================================
DUPLICATES
============================================================

Remove duplicate questions.

The same question appearing on multiple websites is still
one question.

Use the strongest available source.

Do not count duplicates as separate questions.

============================================================
VERIFICATION
============================================================

Every returned question must have:

"verified": true

If a candidate cannot be verified:

EXCLUDE IT.

Do not return:

"verified": false

as a question.

============================================================
IMPORTANT ANTI-HALLUCINATION RULE
============================================================

If you find only 17 verified questions while the user asked
for 50:

return 17.

DO NOT manufacture the remaining 33.

Set:

statistics.requested = 50
statistics.verified_returned = 17
statistics.shortfall = 33

============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

Do not use Markdown.

Do not write explanations outside JSON.

Do not write:

Here are the questions.

Do not write:

```json

Return exactly one JSON object.

The required structure is:

{schema_text}

============================================================
FINAL CHECK BEFORE RETURNING
============================================================

For every question verify:

1. It is an actual UGC-NET question.
2. The requested year matches.
3. The topic/category matches.
4. Original wording is preserved.
5. All available options are captured.
6. Correct answer is verified.
7. Source URL is real.
8. Source information is included.
9. It is not a duplicate.
10. Graph/table/diagram data is not fabricated.
11. verified is true.

At the very end of the response, after the JSON object,
append exactly:

{DONE_MARKER}

Nothing else.
"""

    return prompt


# ============================================================
# FIND CHATGPT INPUT
# ============================================================

async def find_chatgpt_input(
    page: Page
):

    selectors = [
        'textarea[aria-label="Chat with ChatGPT"]',
        '#mobile-composer-prompt',
        'textarea.wm-composer-textarea',
        'textarea[name="prompt"]',
        'textarea[placeholder="Ask ChatGPT"]',
        'textarea',
        '[contenteditable="true"]',
        '[role="textbox"]'
    ]

    for selector in selectors:

        try:

            locator = page.locator(
                selector
            )

            count = await locator.count()

            for i in range(count):

                element = locator.nth(i)

                try:

                    if await element.is_visible():

                        return element

                except Exception:
                    continue

        except Exception:
            continue

    return None


# ============================================================
# ASSISTANT MESSAGE LOCATOR
# ============================================================

async def get_assistant_messages(
    page: Page
):

    selectors = [
        'li[data-message-role="assistant"] [data-assistant-markdown]',
        '[data-message-author-role="assistant"]',
        'div[data-message-author-role="assistant"]',
        'article [data-message-author-role="assistant"]'
    ]

    for selector in selectors:

        try:

            locator = page.locator(
                selector
            )

            if await locator.count() > 0:

                return locator

        except Exception:
            continue

    return None


# ============================================================
# GET ASSISTANT MESSAGE COUNT
# ============================================================

async def get_assistant_message_count(
    page: Page
) -> int:

    locator = await get_assistant_messages(
        page
    )

    if locator is None:
        return 0

    try:
        return await locator.count()

    except Exception:
        return 0


# ============================================================
# GET LATEST RESPONSE
# ============================================================

async def get_latest_response(
    page: Page
) -> str:

    locator = await get_assistant_messages(
        page
    )

    if locator is None:
        return ""

    try:

        count = await locator.count()

        if count == 0:
            return ""

        latest = locator.nth(
            count - 1
        )

        text = await latest.inner_text()

        return html.unescape(
            text or ""
        )

    except Exception:

        return ""


# ============================================================
# GENERATING CHECK
# ============================================================

async def is_generating(
    page: Page
) -> bool:

    selectors = [
        'button[aria-label*="Stop"]',
        'button[aria-label*="stop"]',
        'button[data-testid*="stop"]',
        '[data-testid*="stop-generating"]'
    ]

    for selector in selectors:

        try:

            locator = page.locator(
                selector
            )

            if await locator.count() > 0:

                for i in range(
                    await locator.count()
                ):

                    try:

                        if await locator.nth(i).is_visible():

                            return True

                    except Exception:
                        pass

        except Exception:
            pass

    return False


# ============================================================
# SEND PROMPT
# ============================================================

async def send_prompt(
    page: Page,
    prompt: str
) -> int:

    # IMPORTANT:
    # Record assistant-message count BEFORE sending.
    # This prevents an old answer from being mistaken
    # for the new research response.

    baseline_count = (
        await get_assistant_message_count(
            page
        )
    )

    textarea = await find_chatgpt_input(
        page
    )

    if textarea is None:

        raise RuntimeError(
            "ChatGPT input box was not found. "
            "The login session may have expired."
        )

    await textarea.click()

    await textarea.fill(
        prompt
    )

    print(
        "Prompt entered."
    )

    await textarea.press(
        "Enter"
    )

    print(
        "Prompt sent."
    )

    return baseline_count


# ============================================================
# WAIT FOR RESPONSE
# ============================================================

async def wait_for_response(
    page: Page,
    baseline_count: int
) -> str:

    start = asyncio.get_running_loop().time()

    last_response = ""

    stable_count = 0

    while True:

        elapsed = (
            asyncio.get_running_loop().time()
            - start
        )

        if elapsed > MAX_WAIT_SECONDS:

            raise TimeoutError(
                f"ChatGPT response timeout after "
                f"{MAX_WAIT_SECONDS} seconds."
            )

        current_count = (
            await get_assistant_message_count(
                page
            )
        )

        # Wait until a NEW assistant message exists.
        if current_count <= baseline_count:

            await page.wait_for_timeout(
                1000
            )

            continue

        response = await get_latest_response(
            page
        )

        if not response:

            await page.wait_for_timeout(
                1000
            )

            continue

        print(
            f"Response received: {len(response)} characters"
        )

        # Preferred completion condition.
        if DONE_MARKER in response:

            print(
                "Completion marker found."
            )

            return response

        generating = await is_generating(
            page
        )

        # If ChatGPT is still generating,
        # keep waiting.
        if generating:

            last_response = response
            stable_count = 0

            await page.wait_for_timeout(
                1000
            )

            continue

        # Some UI versions don't expose a generating button.
        # Detect a stable response.
        if response == last_response:

            stable_count += 1

        else:

            stable_count = 0
            last_response = response

        if stable_count >= 5:

            print(
                "Response appears stable."
            )

            return response

        await page.wait_for_timeout(
            1000
        )


# ============================================================
# EXTRACT JSON
# ============================================================

def extract_json(
    response: str
) -> dict[str, Any]:

    if not response:

        raise ValueError(
            "Empty ChatGPT response."
        )

    cleaned = html.unescape(
        response
    ).strip()

    # Remove completion marker.
    cleaned = cleaned.replace(
        DONE_MARKER,
        ""
    ).strip()

    # --------------------------------------------------------
    # 1. Try complete response
    # --------------------------------------------------------

    try:

        data = json.loads(
            cleaned
        )

        if isinstance(data, dict):
            return data

    except Exception:
        pass

    # --------------------------------------------------------
    # 2. Markdown JSON code block
    # --------------------------------------------------------

    code_block_pattern = re.compile(
        r"```(?:json)?\s*(.*?)\s*```",
        re.IGNORECASE | re.DOTALL
    )

    blocks = code_block_pattern.findall(
        cleaned
    )

    for block in blocks:

        try:

            data = json.loads(
                block
            )

            if isinstance(data, dict):
                return data

        except Exception:
            continue

    # --------------------------------------------------------
    # 3. Balanced JSON object extraction
    # --------------------------------------------------------

    start_index = cleaned.find(
        "{"
    )

    if start_index == -1:

        raise ValueError(
            "No JSON object found in ChatGPT response."
        )

    depth = 0
    in_string = False
    escaped = False

    for index in range(
        start_index,
        len(cleaned)
    ):

        char = cleaned[index]

        if in_string:

            if escaped:

                escaped = False

            elif char == "\\":

                escaped = True

            elif char == '"':

                in_string = False

            continue

        if char == '"':

            in_string = True

        elif char == "{":

            depth += 1

        elif char == "}":

            depth -= 1

            if depth == 0:

                candidate = cleaned[
                    start_index:index + 1
                ]

                try:

                    data = json.loads(
                        candidate
                    )

                    if isinstance(
                        data,
                        dict
                    ):
                        return data

                except json.JSONDecodeError as e:

                    raise ValueError(
                        "JSON was found but could not "
                        f"be decoded: {e}"
                    )

    raise ValueError(
        "Could not extract a valid JSON object."
    )


# ============================================================
# NORMALIZE REQUEST METADATA
# ============================================================

def normalize_request_metadata(
    data: dict[str, Any],
    request: ResearchRequest
) -> dict[str, Any]:

    categories = normalize_categories(
        request.category
    )

    years = normalize_years(
        request.year
    )

    data["request"] = {
        "original_request": create_request_description(
            request
        ),
        "years_requested": years,
        "sessions_requested": [],
        "subjects_requested": [],
        "papers_requested": [],
        "topics_requested": [
            request.topic
        ],
        "question_types_requested": categories,
        "difficulty_requested": [],
        "question_count_requested": request.count
    }

    return data


# ============================================================
# NORMALIZE ANSWER
# ============================================================

def normalize_answer(
    answer: Any
) -> Any:

    if answer is None:
        return None

    if isinstance(
        answer,
        str
    ):

        value = answer.strip()

        if not value:
            return None

        match = re.fullmatch(
            r"(?:OPTION\s*)?([A-D])",
            value,
            re.IGNORECASE
        )

        if match:

            return match.group(
                1
            ).upper()

        return value

    return answer


# ============================================================
# NORMALIZE QUESTION TYPE
# ============================================================

def normalize_question_type(
    value: Any
) -> str:

    if value is None:
        return "MCQ"

    value = str(
        value
    ).strip()

    return value or "MCQ"


# ============================================================
# VALIDATE URL
# ============================================================

def valid_source_url(
    value: Any
) -> bool:

    if not isinstance(
        value,
        str
    ):
        return False

    value = value.strip()

    return (
        value.startswith("https://")
        or value.startswith("http://")
    )


# ============================================================
# VALIDATE QUESTIONS
# ============================================================

def validate_questions(
    data: dict[str, Any],
    request: ResearchRequest
) -> dict[str, Any]:

    if not isinstance(
        data,
        dict
    ):

        raise ValueError(
            "Root JSON must be an object."
        )

    questions = data.get(
        "questions"
    )

    if not isinstance(
        questions,
        list
    ):

        raise ValueError(
            "JSON must contain a questions array."
        )

    valid_questions = []

    rejected_count = 0

    duplicate_count = 0

    seen = set()

    standard_mcq_types = {
        "mcq",
        "graph based",
        "graph-based",
        "table based",
        "table-based",
        "data interpretation",
        "statement based",
        "statement-based",
        "multiple statement",
        "multiple-statement",
        "assertion reason",
        "assertion-reason"
    }

    for question in questions:

        if not isinstance(
            question,
            dict
        ):

            rejected_count += 1
            continue

        # ----------------------------------------------------
        # Question text
        # ----------------------------------------------------

        question_text = str(
            question.get(
                "question",
                ""
            )
        ).strip()

        if not question_text:

            rejected_count += 1
            continue

        # ----------------------------------------------------
        # Verification
        # ----------------------------------------------------

        verified = question.get(
            "verified"
        )

        if verified is not True:

            rejected_count += 1
            continue

        # ----------------------------------------------------
        # Source
        # ----------------------------------------------------

        source_url = question.get(
            "source_url"
        )

        if not valid_source_url(
            source_url
        ):

            rejected_count += 1
            continue

        source_name = str(
            question.get(
                "source_name",
                ""
            )
        ).strip()

        if not source_name:

            rejected_count += 1
            continue

        source_type = str(
            question.get(
                "source_type",
                ""
            )
        ).strip()

        if not source_type:

            rejected_count += 1
            continue

        # ----------------------------------------------------
        # Options
        # ----------------------------------------------------

        options = question.get(
            "options"
        )

        if not isinstance(
            options,
            dict
        ):

            rejected_count += 1
            continue

        if len(options) == 0:

            rejected_count += 1
            continue

        # ----------------------------------------------------
        # Question type
        # ----------------------------------------------------

        question_type = normalize_question_type(
            question.get(
                "question_type"
            )
        )

        type_key = question_type.lower()

        # Normal MCQ questions should have A-D.
        if type_key in standard_mcq_types:

            missing = [
                option
                for option in [
                    "A",
                    "B",
                    "C",
                    "D"
                ]
                if option not in options
            ]

            if missing:

                rejected_count += 1
                continue

        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        correct_answer = normalize_answer(
            question.get(
                "correct_answer"
            )
        )

        if correct_answer is None:

            rejected_count += 1
            continue

        # ----------------------------------------------------
        # Year
        # ----------------------------------------------------

        year = str(
            question.get(
                "year",
                ""
            )
        ).strip()

        if not year:

            rejected_count += 1
            continue

        # ----------------------------------------------------
        # Duplicate detection
        # ----------------------------------------------------

        normalized_text = re.sub(
            r"\s+",
            " ",
            question_text.lower()
        )

        duplicate_key = (
            normalized_text,
            year,
            str(
                question.get(
                    "session",
                    ""
                )
            ).strip().lower(),
            str(
                question.get(
                    "shift",
                    ""
                )
            ).strip().lower()
        )

        if duplicate_key in seen:

            duplicate_count += 1
            continue

        seen.add(
            duplicate_key
        )

        # ----------------------------------------------------
        # Clean question
        # ----------------------------------------------------

        clean_question = dict(
            question
        )

        clean_question[
            "question"
        ] = question_text

        clean_question[
            "question_type"
        ] = question_type

        clean_question[
            "verified"
        ] = True

        clean_question[
            "correct_answer"
        ] = correct_answer

        clean_question[
            "source_url"
        ] = source_url.strip()

        clean_question[
            "source_name"
        ] = source_name

        clean_question[
            "source_type"
        ] = source_type

        # ----------------------------------------------------
        # Ensure structured fields exist
        # ----------------------------------------------------

        for field in [
            "graph_data",
            "table_data",
            "diagram_data"
        ]:

            if field not in clean_question:

                clean_question[field] = None

            elif (
                clean_question[field] is not None
                and not isinstance(
                    clean_question[field],
                    dict
                )
            ):

                clean_question[field] = None

        valid_questions.append(
            clean_question
        )

        # ----------------------------------------------------
        # Stop at requested count
        # ----------------------------------------------------

        if len(valid_questions) >= request.count:

            break

    # --------------------------------------------------------
    # Reassign IDs
    # --------------------------------------------------------

    for index, question in enumerate(
        valid_questions,
        start=1
    ):

        question["id"] = index

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    returned_count = len(
        valid_questions
    )

    shortfall = max(
        0,
        request.count - returned_count
    )

    data["questions"] = (
        valid_questions
    )

    data["statistics"] = {
        "requested": request.count,
        "verified_returned": returned_count,
        "excluded_unverified": rejected_count,
        "duplicates_removed": duplicate_count,
        "shortfall": shortfall
    }

    return data


# ============================================================
# SAVE JSON
# ============================================================

def timestamp() -> str:

    return datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


def save_json(
    data: dict[str, Any]
) -> Path:

    file_name = (
        f"ugcnet_questions_"
        f"{timestamp()}.json"
    )

    output_file = (
        OUTPUT_DIR / file_name
    )

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )

    # Also maintain latest.json
    latest_file = (
        OUTPUT_DIR / "latest.json"
    )

    with latest_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"JSON saved: {output_file}"
    )

    return output_file


# ============================================================
# SAVE RAW RESPONSE
# ============================================================

def save_raw_response(
    response: str
) -> Path:

    file_name = (
        f"raw_response_"
        f"{timestamp()}.txt"
    )

    output_file = (
        OUTPUT_DIR / file_name
    )

    output_file.write_text(
        response,
        encoding="utf-8"
    )

    return output_file


# ============================================================
# CREATE BROWSER
# ============================================================

async def create_browser(
    playwright
):

    chromium = playwright.chromium

    browser = await chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu"
        ]
    )

    # --------------------------------------------------------
    # Authenticated session
    # --------------------------------------------------------

    if STORAGE_STATE.exists():

        print(
            f"Using storage state: {STORAGE_STATE}"
        )

        context = await browser.new_context(
            storage_state=str(
                STORAGE_STATE
            ),
            viewport={
                "width": 1400,
                "height": 900
            }
        )

    else:

        print(
            "WARNING: storage_state.json not found."
        )

        context = await browser.new_context(
            viewport={
                "width": 1400,
                "height": 900
            }
        )

    return browser, context


# ============================================================
# RESEARCH
# ============================================================

async def research_ugcnet(
    request: ResearchRequest
):

    prompt = build_prompt(
        request
    )

    async with async_playwright() as p:

        browser, context = (
            await create_browser(p)
        )

        try:

            # ------------------------------------------------
            # Page
            # ------------------------------------------------

            if context.pages:

                page = context.pages[0]

            else:

                page = await context.new_page()

            page.set_default_timeout(
                30000
            )

            print(
                "Opening ChatGPT..."
            )

            await page.goto(
                CHATGPT_URL,
                wait_until="domcontentloaded",
                timeout=NAVIGATION_TIMEOUT
            )

            await page.wait_for_timeout(
                5000
            )

            # ------------------------------------------------
            # Check login / input
            # ------------------------------------------------

            textarea = await find_chatgpt_input(
                page
            )

            if textarea is None:

                screenshot = (
                    OUTPUT_DIR
                    / "chatgpt_login_error.png"
                )

                try:

                    await page.screenshot(
                        path=str(
                            screenshot
                        ),
                        full_page=True
                    )

                except Exception:
                    pass

                raise RuntimeError(
                    "ChatGPT input was not found. "
                    "Your storage_state.json may be "
                    "missing or expired. "
                    "Create a fresh authenticated "
                    "Playwright storage state."
                )

            # ------------------------------------------------
            # Send
            # ------------------------------------------------

            baseline_count = (
                await send_prompt(
                    page,
                    prompt
                )
            )

            # ------------------------------------------------
            # Wait
            # ------------------------------------------------

            response = await wait_for_response(
                page,
                baseline_count
            )

            print(
                f"Raw response length: {len(response)}"
            )

            # ------------------------------------------------
            # Save raw
            # ------------------------------------------------

            raw_file = save_raw_response(
                response
            )

            # ------------------------------------------------
            # Extract JSON
            # ------------------------------------------------

            data = extract_json(
                response
            )

            # ------------------------------------------------
            # Force request metadata from API input.
            # Do not trust the LLM to reproduce these fields.
            # ------------------------------------------------

            data = normalize_request_metadata(
                data,
                request
            )

            # ------------------------------------------------
            # Validate
            # ------------------------------------------------

            data = validate_questions(
                data,
                request
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            output_file = save_json(
                data
            )

            return {
                "success": True,
                "questions": len(
                    data["questions"]
                ),
                "requested": request.count,
                "output_file": str(
                    output_file
                ),
                "raw_response_file": str(
                    raw_file
                ),
                "data": data
            }

        finally:

            try:
                await context.close()
            except Exception:
                pass

            try:
                await browser.close()
            except Exception:
                pass


# ============================================================
# API ENDPOINT
# ============================================================

@app.post(
    "/research"
)
async def research(
    request: ResearchRequest,
    authorization: str | None = Header(
        default=None
    )
):

    # --------------------------------------------------------
    # Authentication
    # --------------------------------------------------------

    if not check_api_key(
        authorization
    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid API key."
        )

    # --------------------------------------------------------
    # Validate topic
    # --------------------------------------------------------

    if not request.topic.strip():

        raise HTTPException(
            status_code=400,
            detail="topic cannot be empty."
        )

    # --------------------------------------------------------
    # Validate category
    # --------------------------------------------------------

    categories = normalize_categories(
        request.category
    )

    if not categories:

        raise HTTPException(
            status_code=400,
            detail="category cannot be empty."
        )

    # --------------------------------------------------------
    # Prevent concurrent browser sessions
    # --------------------------------------------------------

    if research_lock.locked():

        raise HTTPException(
            status_code=429,
            detail=(
                "Another UGC-NET research job is "
                "currently running. Please try again later."
            )
        )

    async with research_lock:

        try:

            print(
                "================================================"
            )

            print(
                "NEW UGC-NET RESEARCH REQUEST"
            )

            print(
                json.dumps(
                    request.model_dump(),
                    ensure_ascii=False,
                    indent=2
                )
            )

            print(
                "================================================"
            )

            result = await research_ugcnet(
                request
            )

            return result

        except HTTPException:

            raise

        except Exception as e:

            print(
                "ERROR:",
                repr(e)
            )

            raise HTTPException(
                status_code=500,
                detail=str(e)
            )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "service":
            "UGC-NET Original Question Researcher",

        "version":
            "2.0.0",

        "status":
            "running",

        "web_research":
            True,

        "input_format":
            {
                "topic": "string",
                "year": "string",
                "category": "string or array",
                "count": "integer"
            },

        "example": {
            "topic": "Communication",
            "year": "2019-2025",
            "category": [
                "MCQ",
                "Graph Based"
            ],
            "count": 20
        }
    }


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health"
)
async def health():

    return {
        "status": "ok",
        "chatgpt_session_configured":
            STORAGE_STATE.exists(),
        "storage_state":
            str(STORAGE_STATE),
        "output_directory":
            str(OUTPUT_DIR)
    }


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv(
            "PORT",
            "8000"
        )
    )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
