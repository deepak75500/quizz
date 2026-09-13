import os
import re
import io
import gc
import json
import time
import hashlib
import logging
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from groq import Groq

from paddleocr import PaddleOCR


# ============================================================
# CONFIG
# ============================================================

SOURCE_URL = os.getenv(
    "SOURCE_URL",
    "https://humanperitus.in/ugc-net-paper-1-previous-papers/"
)

GROQ_API_KEY = os.environ["GROQ_API_KEY"]

# Recommended current Groq model
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

CF_ACCOUNT_ID = os.environ["CF_ACCOUNT_ID"]
CF_D1_DATABASE_ID = os.environ["CF_D1_DATABASE_ID"]
CF_API_TOKEN = os.environ["CF_API_TOKEN"]

# Optional:
# Process only selected years, e.g. "2009,2010,2011"
YEARS = os.getenv("YEARS", "").strip()

# Maximum number of papers per run.
# Empty = all.
MAX_PAPERS = int(os.getenv("MAX_PAPERS", "0"))

# OCR DPI
DPI = int(os.getenv("OCR_DPI", "150"))

# Render memory is limited, so don't use huge images.
MAX_IMAGE_WIDTH = int(os.getenv("MAX_IMAGE_WIDTH", "1800"))

# Request timeout
HTTP_TIMEOUT = 60

# D1 endpoint
D1_URL = (
    f"https://api.cloudflare.com/client/v4/accounts/"
    f"{CF_ACCOUNT_ID}/d1/database/"
    f"{CF_D1_DATABASE_ID}/query"
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger("ugc-net")


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/139 Safari/537.36"
    )
})


# ============================================================
# GROQ
# ============================================================

groq_client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# PADDLE OCR
# ============================================================

log.info("Loading PaddleOCR...")

ocr = PaddleOCR(
    lang="en"
)

log.info("PaddleOCR loaded.")


# ============================================================
# CLOUDFLARE D1
# ============================================================

def d1_query(sql, params=None):
    """
    Execute SQL against Cloudflare D1.
    """

    payload = {
        "sql": sql
    }

    if params:
        payload["params"] = params

    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        D1_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"D1 HTTP {response.status_code}: {response.text[:1000]}"
        )

    data = response.json()

    if not data.get("success"):
        raise RuntimeError(
            f"D1 error: {data}"
        )

    return data


# ============================================================
# D1 INSERT
# ============================================================

def question_hash(question):
    normalized = re.sub(
        r"\s+",
        " ",
        question.lower().strip()
    )

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def save_question(q, source_url, pdf_name):

    question = q.get("question", "").strip()

    if not question:
        return False

    options = q.get("options", {})

    if not isinstance(options, dict):
        options = {}

    option_a = str(options.get("A", "")).strip()
    option_b = str(options.get("B", "")).strip()
    option_c = str(options.get("C", "")).strip()
    option_d = str(options.get("D", "")).strip()

    correct_answer = str(
        q.get("correct_answer", "")
    ).strip().upper()

    if correct_answer not in ["A", "B", "C", "D"]:
        correct_answer = ""

    year = q.get("year")

    if year:
        try:
            year = int(year)
        except Exception:
            year = None

    session_name = str(
        q.get("session", "")
    ).strip()

    category = str(
        q.get("category", "")
    ).strip()

    difficulty = str(
        q.get("difficulty", "")
    ).strip().lower()

    explanation = str(
        q.get("explanation", "")
    ).strip()

    graph_data = q.get("graph_data")

    table_data = q.get("table_data")

    if graph_data is not None:
        graph_data = json.dumps(
            graph_data,
            ensure_ascii=False
        )

    if table_data is not None:
        table_data = json.dumps(
            table_data,
            ensure_ascii=False
        )

    qhash = question_hash(question)

    sql = """
    INSERT OR IGNORE INTO questions
    (
        question,
        option_a,
        option_b,
        option_c,
        option_d,
        correct_answer,
        explanation,
        year,
        session,
        category,
        difficulty,
        graph_data,
        table_data,
        source_url,
        source_pdf,
        question_hash
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    params = [
        question,
        option_a,
        option_b,
        option_c,
        option_d,
        correct_answer,
        explanation,
        year,
        session_name,
        category,
        difficulty,
        graph_data,
        table_data,
        source_url,
        pdf_name,
        qhash
    ]

    result = d1_query(
        sql,
        params
    )

    return True


# ============================================================
# DOWNLOAD PAGE
# ============================================================

def get_source_page():

    log.info(
        "Downloading source page: %s",
        SOURCE_URL
    )

    response = session.get(
        SOURCE_URL,
        timeout=HTTP_TIMEOUT
    )

    response.raise_for_status()

    return response.text


# ============================================================
# FIND PDF LINKS
# ============================================================

def find_pdf_links(html):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    results = []

    for a in soup.find_all("a", href=True):

        href = a["href"].strip()

        absolute = urljoin(
            SOURCE_URL,
            href
        )

        parsed = urlparse(absolute)

        if parsed.scheme not in ["http", "https"]:
            continue

        text = a.get_text(
            " ",
            strip=True
        )

        lower = absolute.lower()

        if (
            ".pdf" in lower
            or "question" in text.lower()
            or "paper 1" in text.lower()
        ):
            results.append({
                "url": absolute,
                "text": text
            })

    # Remove duplicates
    unique = {}

    for item in results:
        unique[item["url"]] = item

    results = list(unique.values())

    # Exclude obvious answer keys
    filtered = []

    for item in results:

        combined = (
            item["url"] + " " +
            item["text"]
        ).lower()

        if (
            "answer key" in combined
            or "answer-key" in combined
            or "answer_key" in combined
        ):
            continue

        filtered.append(item)

    return filtered


# ============================================================
# YEAR FILTER
# ============================================================

def allowed_year(url, text):

    if not YEARS:
        return True

    requested = {
        x.strip()
        for x in YEARS.split(",")
        if x.strip()
    }

    combined = f"{url} {text}"

    found_years = set(
        re.findall(
            r"\b20(?:0[9]|1[0-9]|2[0-9])\b",
            combined
        )
    )

    if not found_years:
        return True

    return bool(
        requested.intersection(found_years)
    )


# ============================================================
# DOWNLOAD PDF
# ============================================================

def download_pdf(url):

    log.info("Downloading PDF: %s", url)

    response = session.get(
        url,
        timeout=HTTP_TIMEOUT,
        stream=True
    )

    response.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=False
    )

    path = tmp.name

    try:

        for chunk in response.iter_content(
            chunk_size=1024 * 64
        ):

            if chunk:
                tmp.write(chunk)

    finally:
        tmp.close()

    return path


# ============================================================
# EXTRACT EXISTING PDF TEXT
# ============================================================

def extract_native_text(page):

    try:

        text = page.get_text(
            "text"
        )

        if text and len(text.strip()) >= 80:
            return text.strip()

    except Exception:
        pass

    return ""


# ============================================================
# PDF PAGE -> IMAGE
# ============================================================

def page_to_png(page):

    scale = DPI / 72

    matrix = fitz.Matrix(
        scale,
        scale
    )

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    png_bytes = pix.tobytes(
        "png"
    )

    del pix

    return png_bytes


# ============================================================
# PADDLE OCR
# ============================================================

def paddle_ocr_page(png_bytes):

    """
    OCR one page only.

    This is deliberately page-by-page to reduce RAM.
    """

    result = ocr.predict(
        input=png_bytes
    )

    texts = []

    try:

        for res in result:

            # PaddleOCR versions differ slightly.
            # Try common output structures.

            if hasattr(res, "json"):
                data = res.json
                if callable(data):
                    data = data()

            elif isinstance(res, dict):
                data = res

            else:
                continue

            if not isinstance(data, dict):
                continue

            # Common PaddleOCR structure
            inner = data.get("res", data)

            if not isinstance(inner, dict):
                continue

            rec_texts = inner.get(
                "rec_texts",
                []
            )

            if rec_texts:
                texts.extend(
                    str(x)
                    for x in rec_texts
                    if str(x).strip()
                )

    except Exception as e:

        log.warning(
            "OCR output parsing issue: %s",
            e
        )

    return "\n".join(texts)


# ============================================================
# OCR ENTIRE PDF
# ============================================================

def pdf_to_text(pdf_path):

    log.info(
        "OCR PDF: %s",
        pdf_path
    )

    document = fitz.open(
        pdf_path
    )

    all_pages = []

    try:

        for page_number in range(
            len(document)
        ):

            log.info(
                "Page %s/%s",
                page_number + 1,
                len(document)
            )

            page = document[
                page_number
            ]

            # First try normal PDF text.
            native = extract_native_text(
                page
            )

            if native:

                text = native

            else:

                png = page_to_png(
                    page
                )

                text = paddle_ocr_page(
                    png
                )

                del png

                gc.collect()

            if text.strip():

                all_pages.append(
                    f"\n--- PAGE {page_number + 1} ---\n"
                    f"{text}"
                )

    finally:

        document.close()

    text = "\n".join(
        all_pages
    )

    return text


# ============================================================
# CLEAN OCR TEXT
# ============================================================

def clean_text(text):

    text = text.replace(
        "\x00",
        ""
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{4,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# SPLIT TEXT INTO CHUNKS
# ============================================================

def split_text(text, max_chars=30000):

    """
    Do NOT send an entire 100+ page paper blindly.
    Split on page boundaries.
    """

    pages = re.split(
        r"\n--- PAGE \d+ ---\n",
        text
    )

    chunks = []

    current = ""

    for page in pages:

        page = page.strip()

        if not page:
            continue

        if len(current) + len(page) + 100 <= max_chars:

            current += (
                "\n--- PAGE ---\n" +
                page
            )

        else:

            if current.strip():
                chunks.append(
                    current.strip()
                )

            current = (
                "\n--- PAGE ---\n" +
                page
            )

    if current.strip():
        chunks.append(
            current.strip()
        )

    return chunks


# ============================================================
# GROQ EXTRACTION
# ============================================================

SYSTEM_PROMPT = r"""
You are an expert UGC-NET Paper 1 question extraction system.

You receive OCR text from an original UGC-NET Paper 1 question paper.

Your job is ONLY to extract questions that are actually present
in the supplied text.

Do NOT invent questions.

Do NOT paraphrase questions.

Do NOT improve wording.

Do NOT correct the question.

Preserve the original wording as closely as possible.

Every extracted MCQ should have:

question
options A-D
correct_answer
year
session
category
difficulty
explanation
graph_data
table_data

Rules:

1. Extract only actual questions.
2. Every question must have A, B, C and D where available.
3. If a question is incomplete, do not invent missing information.
4. correct_answer must be A, B, C or D only if the answer can be
   reliably determined from the supplied answer information.
5. If no answer information is supplied, correct_answer must be "".
6. category should be one of the UGC-NET Paper 1 topics when clear.
7. difficulty must be easy, medium or hard.
8. Keep equations and special terminology.
9. If there is a table, preserve its contents in table_data.
10. If there is a graph/diagram, describe only what is actually visible
    in graph_data.
11. Do not create explanations unless the answer is reasonably
    supported by the supplied source.
12. Return JSON only.

Expected format:

{
  "questions": [
    {
      "question": "...",
      "options": {
        "A": "...",
        "B": "...",
        "C": "...",
        "D": "..."
      },
      "correct_answer": "A",
      "year": 2025,
      "session": "Morning",
      "category": "Teaching Aptitude",
      "difficulty": "medium",
      "explanation": "...",
      "graph_data": null,
      "table_data": null
    }
  ]
}
"""


def groq_extract(
    text,
    metadata
):

    user_prompt = f"""
SOURCE FILE:
{metadata.get("pdf_name", "")}

SOURCE URL:
{metadata.get("source_url", "")}

YEAR:
{metadata.get("year", "")}

SESSION:
{metadata.get("session", "")}

Extract all UGC-NET Paper 1 questions from this OCR text.

IMPORTANT:
The text may contain OCR errors.
Do not hallucinate missing questions.

OCR TEXT:
{text}
"""

    for attempt in range(4):

        try:

            log.info(
                "Sending chunk to Groq, attempt %s/4",
                attempt + 1
            )

            completion = (
                groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],
                    temperature=0,
                    max_completion_tokens=12000,
                    response_format={
                        "type": "json_object"
                    }
                )
            )

            raw = (
                completion
                .choices[0]
                .message
                .content
            )

            data = json.loads(
                raw
            )

            questions = data.get(
                "questions",
                []
            )

            if not isinstance(
                questions,
                list
            ):
                return []

            # Add metadata when model misses it
            for q in questions:

                if not q.get("year"):
                    q["year"] = metadata.get(
                        "year"
                    )

                if not q.get("session"):
                    q["session"] = metadata.get(
                        "session"
                    )

            return questions

        except Exception as e:

            log.warning(
                "Groq error: %s",
                e
            )

            if attempt < 3:
                time.sleep(
                    2 ** attempt
                )

    return []


# ============================================================
# METADATA
# ============================================================

def extract_metadata(
    text,
    pdf_url
):

    combined = (
        text[:5000] +
        " " +
        pdf_url
    )

    years = re.findall(
        r"\b20(?:0[9]|1[0-9]|2[0-9])\b",
        combined
    )

    year = None

    if years:
        # Prefer the first year
        year = int(
            years[0]
        )

    session = ""

    lower = combined.lower()

    if "morning" in lower:
        session = "Morning"

    elif "evening" in lower:
        session = "Evening"

    elif "shift 1" in lower:
        session = "Shift 1"

    elif "shift 2" in lower:
        session = "Shift 2"

    return {
        "year": year,
        "session": session
    }


# ============================================================
# PROCESS ONE PDF
# ============================================================

def process_pdf(
    pdf_url,
    link_text
):

    pdf_path = None

    pdf_name = (
        Path(
            urlparse(pdf_url).path
        ).name
        or "ugc_net_paper1.pdf"
    )

    log.info(
        "=" * 70
    )

    log.info(
        "PROCESSING: %s",
        pdf_name
    )

    try:

        pdf_path = download_pdf(
            pdf_url
        )

        # OCR
        text = pdf_to_text(
            pdf_path
        )

        text = clean_text(
            text
        )

        if len(text) < 100:

            log.warning(
                "Very little text extracted from %s",
                pdf_name
            )

            return

        metadata = extract_metadata(
            text,
            pdf_url
        )

        metadata["pdf_name"] = pdf_name
        metadata["source_url"] = pdf_url

        chunks = split_text(
            text,
            max_chars=30000
        )

        log.info(
            "Text split into %s Groq chunks",
            len(chunks)
        )

        total = 0

        for index, chunk in enumerate(
            chunks
        ):

            log.info(
                "Groq chunk %s/%s",
                index + 1,
                len(chunks)
            )

            questions = groq_extract(
                chunk,
                metadata
            )

            log.info(
                "Groq extracted %s questions",
                len(questions)
            )

            for q in questions:

                try:

                    saved = save_question(
                        q,
                        pdf_url,
                        pdf_name
                    )

                    if saved:
                        total += 1

                except Exception as e:

                    log.error(
                        "D1 insert failed: %s",
                        e
                    )

            # Release memory
            del chunk
            del questions

            gc.collect()

        log.info(
            "Saved %s questions from %s",
            total,
            pdf_name
        )

        # Important:
        # Save progress AFTER the whole PDF succeeds.
        mark_paper_processed(
            pdf_url,
            pdf_name,
            total
        )

    except Exception as e:

        log.exception(
            "Failed PDF: %s",
            e
        )

    finally:

        # Delete temporary PDF
        if pdf_path:

            try:
                os.remove(
                    pdf_path
                )
            except Exception:
                pass

        gc.collect()


# ============================================================
# PROCESS TRACKING
# ============================================================

def ensure_progress_table():

    sql = """
    CREATE TABLE IF NOT EXISTS processed_papers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pdf_url TEXT UNIQUE NOT NULL,
        pdf_name TEXT,
        question_count INTEGER DEFAULT 0,
        processed_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """

    d1_query(sql)


def already_processed(
    pdf_url
):

    sql = """
    SELECT id
    FROM processed_papers
    WHERE pdf_url = ?
    LIMIT 1
    """

    result = d1_query(
        sql,
        [pdf_url]
    )

    results = (
        result.get("result", [{}])[0]
        .get("results", [])
    )

    return len(results) > 0


def mark_paper_processed(
    pdf_url,
    pdf_name,
    question_count
):

    sql = """
    INSERT OR REPLACE INTO processed_papers
    (
        pdf_url,
        pdf_name,
        question_count,
        processed_at
    )
    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
    """

    d1_query(
        sql,
        [
            pdf_url,
            pdf_name,
            question_count
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    log.info(
        "UGC-NET Paper 1 scraper starting..."
    )

    ensure_progress_table()

    html = get_source_page()

    links = find_pdf_links(
        html
    )

    log.info(
        "Found %s possible paper links",
        len(links)
    )

    selected = []

    for item in links:

        if not allowed_year(
            item["url"],
            item["text"]
        ):
            continue

        selected.append(
            item
        )

    log.info(
        "After year filtering: %s",
        len(selected)
    )

    if MAX_PAPERS > 0:

        selected = selected[
            :MAX_PAPERS
        ]

    processed = 0

    for item in selected:

        url = item["url"]

        try:

            if already_processed(
                url
            ):

                log.info(
                    "Already processed: %s",
                    url
                )

                continue

        except Exception as e:

            log.warning(
                "Could not check progress: %s",
                e
            )

        process_pdf(
            url,
            item["text"]
        )

        processed += 1

        # Allow memory to settle
        gc.collect()

    log.info(
        "Finished. Papers processed this run: %s",
        processed
    )


if __name__ == "__main__":
    main()
