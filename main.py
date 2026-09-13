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
import pymupdf
from bs4 import BeautifulSoup
from groq import Groq
from paddleocr import PaddleOCR


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_URL = os.getenv(
    "SOURCE_URL",
    "https://humanperitus.in/ugc-net-paper-1-previous-papers/"
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID")
CF_D1_DATABASE_ID = os.getenv("CF_D1_DATABASE_ID")
CF_API_TOKEN = os.getenv("CF_API_TOKEN")

# Optional year filter
# Example:
# YEARS=2020,2021,2022
YEARS = os.getenv(
    "YEARS",
    ""
).strip()

# 0 = process all papers
MAX_PAPERS = int(
    os.getenv(
        "MAX_PAPERS",
        "0"
    )
)

# PDF rendering resolution
OCR_DPI = int(
    os.getenv(
        "OCR_DPI",
        "120"
    )
)

HTTP_TIMEOUT = 90

# D1 endpoint
D1_URL = (
    f"https://api.cloudflare.com/client/v4/"
    f"accounts/{CF_ACCOUNT_ID}/d1/database/"
    f"{CF_D1_DATABASE_ID}/query"
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger(
    "ugc-net-processor"
)


# ============================================================
# VALIDATE ENVIRONMENT
# ============================================================

def validate_environment():

    missing = []

    if not GROQ_API_KEY:
        missing.append(
            "GROQ_API_KEY"
        )

    if not CF_ACCOUNT_ID:
        missing.append(
            "CF_ACCOUNT_ID"
        )

    if not CF_D1_DATABASE_ID:
        missing.append(
            "CF_D1_DATABASE_ID"
        )

    if not CF_API_TOKEN:
        missing.append(
            "CF_API_TOKEN"
        )

    if missing:

        raise RuntimeError(
            "Missing Render environment variables: "
            + ", ".join(missing)
            + "\n"
            "Go to Render → Environment and add them."
        )


validate_environment()


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/139.0 Safari/537.36"
    ),
    "Accept": "*/*"
})


# ============================================================
# GROQ CLIENT
# ============================================================

groq_client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# PADDLE OCR
# ============================================================

log.info(
    "Loading PaddleOCR..."
)

ocr = PaddleOCR(
    lang="en"
)

log.info(
    "PaddleOCR loaded successfully."
)


# ============================================================
# CLOUDFLARE D1
# ============================================================

def d1_query(
    sql,
    params=None,
    retries=3
):

    headers = {
        "Authorization": (
            f"Bearer {CF_API_TOKEN}"
        ),
        "Content-Type": (
            "application/json"
        )
    }

    payload = {
        "sql": sql
    }

    if params is not None:
        payload["params"] = params

    last_error = None

    for attempt in range(retries):

        try:

            response = requests.post(
                D1_URL,
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code != 200:

                raise RuntimeError(
                    f"D1 HTTP "
                    f"{response.status_code}: "
                    f"{response.text[:1000]}"
                )

            data = response.json()

            if not data.get(
                "success",
                False
            ):

                raise RuntimeError(
                    f"D1 returned error: "
                    f"{data}"
                )

            return data

        except Exception as e:

            last_error = e

            log.warning(
                "D1 attempt %s/%s failed: %s",
                attempt + 1,
                retries,
                e
            )

            if attempt < retries - 1:

                time.sleep(
                    2 ** attempt
                )

    raise last_error


# ============================================================
# CREATE TABLES
# ============================================================

def create_tables():

    log.info(
        "Checking Cloudflare D1 tables..."
    )

    questions_sql = """
    CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,

        question TEXT NOT NULL,

        option_a TEXT,
        option_b TEXT,
        option_c TEXT,
        option_d TEXT,

        correct_answer TEXT,

        explanation TEXT,

        year INTEGER,

        session TEXT,

        category TEXT,

        difficulty TEXT,

        graph_data TEXT,

        table_data TEXT,

        source_url TEXT,

        source_pdf TEXT,

        question_hash TEXT UNIQUE,

        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """

    d1_query(
        questions_sql
    )

    indexes = [

        """
        CREATE INDEX IF NOT EXISTS
        idx_questions_year
        ON questions(year);
        """,

        """
        CREATE INDEX IF NOT EXISTS
        idx_questions_category
        ON questions(category);
        """,

        """
        CREATE INDEX IF NOT EXISTS
        idx_questions_hash
        ON questions(question_hash);
        """,

        """
        CREATE INDEX IF NOT EXISTS
        idx_questions_difficulty
        ON questions(difficulty);
        """

    ]

    for sql in indexes:
        d1_query(sql)

    progress_sql = """
    CREATE TABLE IF NOT EXISTS processed_papers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,

        pdf_url TEXT UNIQUE NOT NULL,

        pdf_name TEXT,

        question_count INTEGER DEFAULT 0,

        processed_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """

    d1_query(
        progress_sql
    )

    log.info(
        "D1 tables ready."
    )


# ============================================================
# QUESTION HASH
# ============================================================

def make_question_hash(
    question
):

    normalized = re.sub(
        r"\s+",
        " ",
        question.lower().strip()
    )

    return hashlib.sha256(
        normalized.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# CHECK DUPLICATE
# ============================================================

def question_exists(
    qhash
):

    sql = """
    SELECT id
    FROM questions
    WHERE question_hash = ?
    LIMIT 1;
    """

    result = d1_query(
        sql,
        [qhash]
    )

    try:

        rows = (
            result
            ["result"][0]
            ["results"]
        )

        return len(rows) > 0

    except Exception:

        return False


# ============================================================
# SAVE QUESTION TO D1
# ============================================================

def save_question(
    question_data,
    source_url,
    pdf_name,
    metadata
):

    question = str(
        question_data.get(
            "question",
            ""
        )
    ).strip()

    if not question:
        return False

    options = question_data.get(
        "options",
        {}
    )

    if not isinstance(
        options,
        dict
    ):
        options = {}

    option_a = str(
        options.get(
            "A",
            ""
        )
    ).strip()

    option_b = str(
        options.get(
            "B",
            ""
        )
    ).strip()

    option_c = str(
        options.get(
            "C",
            ""
        )
    ).strip()

    option_d = str(
        options.get(
            "D",
            ""
        )
    ).strip()

    # Require all four options
    if not all([
        option_a,
        option_b,
        option_c,
        option_d
    ]):

        log.warning(
            "Skipping incomplete question: %s",
            question[:100]
        )

        return False

    correct_answer = str(
        question_data.get(
            "correct_answer",
            ""
        )
    ).strip().upper()

    if correct_answer not in [
        "A",
        "B",
        "C",
        "D"
    ]:

        correct_answer = ""

    explanation = str(
        question_data.get(
            "explanation",
            ""
        )
    ).strip()

    category = str(
        question_data.get(
            "category",
            "Other"
        )
    ).strip()

    difficulty = str(
        question_data.get(
            "difficulty",
            "medium"
        )
    ).strip().lower()

    if difficulty not in [
        "easy",
        "medium",
        "hard"
    ]:

        difficulty = "medium"

    year = (
        question_data.get(
            "year"
        )
        or metadata.get(
            "year"
        )
    )

    try:

        if year:
            year = int(year)

    except Exception:

        year = None

    session_name = str(
        question_data.get(
            "session",
            ""
        )
        or metadata.get(
            "session",
            ""
        )
    ).strip()

    graph_data = question_data.get(
        "graph_data"
    )

    table_data = question_data.get(
        "table_data"
    )

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

    qhash = make_question_hash(
        question
    )

    # Extra duplicate protection
    if question_exists(
        qhash
    ):

        log.info(
            "Duplicate skipped: %s",
            question[:100]
        )

        return False

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
    VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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

    d1_query(
        sql,
        params
    )

    log.info(
        "Saved question [%s] %s",
        category,
        question[:100]
    )

    return True


# ============================================================
# FIND PROCESSED PDF
# ============================================================

def already_processed(
    pdf_url
):

    sql = """
    SELECT id
    FROM processed_papers
    WHERE pdf_url = ?
    LIMIT 1;
    """

    result = d1_query(
        sql,
        [pdf_url]
    )

    try:

        rows = (
            result
            ["result"][0]
            ["results"]
        )

        return len(rows) > 0

    except Exception:

        return False


# ============================================================
# MARK PDF PROCESSED
# ============================================================

def mark_processed(
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
    VALUES
    (?, ?, ?, CURRENT_TIMESTAMP);
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
# DOWNLOAD SOURCE PAGE
# ============================================================

def get_source_page():

    log.info(
        "Opening HumanPeritus page..."
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

def find_pdf_links(
    html
):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    found = {}

    for a in soup.find_all(
        "a",
        href=True
    ):

        href = a.get(
            "href",
            ""
        ).strip()

        if not href:
            continue

        url = urljoin(
            SOURCE_URL,
            href
        )

        text = a.get_text(
            " ",
            strip=True
        )

        combined = (
            url + " " + text
        ).lower()

        # Ignore answer keys
        if any(
            x in combined
            for x in [
                "answer key",
                "answer-key",
                "answer_key",
                "answerkey"
            ]
        ):
            continue

        # Only PDF/question-paper links
        if (
            ".pdf" in url.lower()
            or "question paper" in combined
            or "question-paper" in combined
            or "paper 1" in combined
        ):

            found[url] = {
                "url": url,
                "text": text
            }

    return list(
        found.values()
    )


# ============================================================
# YEAR FILTER
# ============================================================

def allowed_year(
    url,
    text
):

    if not YEARS:
        return True

    requested = {
        x.strip()
        for x in YEARS.split(",")
        if x.strip()
    }

    combined = (
        url + " " + text
    )

    years = set(
        re.findall(
            r"\b20(?:0[9]|1[0-9]|2[0-9])\b",
            combined
        )
    )

    if not years:
        return True

    return bool(
        requested.intersection(
            years
        )
    )


# ============================================================
# DOWNLOAD PDF
# ============================================================

def download_pdf(
    url
):

    log.info(
        "Downloading PDF: %s",
        url
    )

    response = session.get(
        url,
        timeout=HTTP_TIMEOUT,
        stream=True
    )

    response.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(
        prefix="ugcnet_",
        suffix=".pdf",
        delete=False
    )

    path = tmp.name

    try:

        for chunk in response.iter_content(
            chunk_size=64 * 1024
        ):

            if chunk:
                tmp.write(
                    chunk
                )

    finally:

        tmp.close()

    log.info(
        "Temporary PDF created: %s",
        path
    )

    return path


# ============================================================
# EXTRACT NATIVE PDF TEXT
# ============================================================

def extract_native_text(
    page
):

    try:

        text = page.get_text(
            "text"
        )

        if (
            text
            and
            len(text.strip()) >= 80
        ):

            return text.strip()

    except Exception as e:

        log.warning(
            "Native PDF text error: %s",
            e
        )

    return ""


# ============================================================
# PAGE -> PNG BYTES
# ============================================================

def page_to_png(
    page
):

    matrix = pymupdf.Matrix(
        OCR_DPI / 72,
        OCR_DPI / 72
    )

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    png_bytes = pix.tobytes(
        "png"
    )

    # Immediately release image memory
    pix = None

    return png_bytes


# ============================================================
# PADDLE OCR
# ============================================================

def paddle_ocr_page(
    png_bytes
):

    texts = []

    try:

        result = ocr.predict(
            input=png_bytes
        )

        for res in result:

            data = None

            # New PaddleOCR result
            if hasattr(
                res,
                "json"
            ):

                data = res.json

                if callable(data):
                    data = data()

            elif isinstance(
                res,
                dict
            ):

                data = res

            if not isinstance(
                data,
                dict
            ):
                continue

            inner = data.get(
                "res",
                data
            )

            if not isinstance(
                inner,
                dict
            ):
                continue

            rec_texts = inner.get(
                "rec_texts",
                []
            )

            for item in rec_texts:

                item = str(
                    item
                ).strip()

                if item:
                    texts.append(
                        item
                    )

    except Exception as e:

        log.exception(
            "PaddleOCR failed: %s",
            e
        )

    finally:

        # Important for Render memory
        gc.collect()

    return "\n".join(
        texts
    )


# ============================================================
# PDF -> TEXT
# ============================================================

def pdf_to_text(
    pdf_path
):

    log.info(
        "Processing PDF: %s",
        pdf_path
    )

    document = pymupdf.open(
        pdf_path
    )

    pages_text = []

    try:

        total_pages = len(
            document
        )

        log.info(
            "PDF has %s pages",
            total_pages
        )

        for index in range(
            total_pages
        ):

            page_number = index + 1

            log.info(
                "Processing page %s/%s",
                page_number,
                total_pages
            )

            page = document[
                index
            ]

            # ------------------------------------------------
            # FIRST: Try normal PDF text
            # ------------------------------------------------

            native_text = (
                extract_native_text(
                    page
                )
            )

            if native_text:

                log.info(
                    "Page %s: native PDF text",
                    page_number
                )

                page_text = native_text

            else:

                # ------------------------------------------------
                # SECOND: OCR scanned page
                # ------------------------------------------------

                log.info(
                    "Page %s: using PaddleOCR",
                    page_number
                )

                png_bytes = None

                try:

                    png_bytes = page_to_png(
                        page
                    )

                    page_text = (
                        paddle_ocr_page(
                            png_bytes
                        )
                    )

                finally:

                    # Never keep image in memory
                    png_bytes = None

                    gc.collect()

            if page_text.strip():

                pages_text.append(
                    "\n"
                    f"--- PAGE {page_number} ---\n"
                    f"{page_text.strip()}\n"
                )

            # Release page references
            page = None

            gc.collect()

    finally:

        document.close()

        gc.collect()

    return "\n".join(
        pages_text
    )


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(
    text
):

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
# SPLIT TEXT
# ============================================================

def split_text(
    text,
    max_chars=28000
):

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

        candidate = (
            current
            + "\n--- PAGE ---\n"
            + page
        )

        if (
            len(candidate)
            <= max_chars
        ):

            current = candidate

        else:

            if current.strip():

                chunks.append(
                    current.strip()
                )

            current = (
                "\n--- PAGE ---\n"
                + page
            )

    if current.strip():

        chunks.append(
            current.strip()
        )

    return chunks


# ============================================================
# UGC-NET PAPER 1 CATEGORIES
# ============================================================

UGC_NET_CATEGORIES = [
    "Teaching Aptitude",
    "Research Aptitude",
    "Comprehension",
    "Communication",
    "Mathematical Reasoning and Aptitude",
    "Logical Reasoning",
    "Data Interpretation",
    "Information and Communication Technology (ICT)",
    "People, Development and Environment",
    "Higher Education System",
    "Other"
]


# ============================================================
# GROQ SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a highly accurate UGC-NET Paper 1 question extraction
and classification system.

You receive OCR text extracted from an ORIGINAL UGC-NET Paper 1
question paper.

Your task is to extract ONLY questions that actually occur in
the supplied source text.

============================================================
STRICT ANTI-HALLUCINATION RULES
============================================================

1. NEVER invent a question.

2. NEVER create a missing option.

3. NEVER paraphrase the original question.

4. Preserve the wording from the source as closely as possible.

5. OCR may contain spelling mistakes. Correct only obvious OCR
   recognition errors when the intended original text is clear.

6. If a question cannot be reconstructed reliably, SKIP it.

7. Every accepted MCQ should have:
   A
   B
   C
   D

8. If one or more options are genuinely unavailable in the source,
   SKIP that question instead of inventing the option.

============================================================
ANSWER RULES
============================================================

9. Determine correct_answer only when it is supported by:
   - an answer key supplied in the input, OR
   - an unambiguous factual/logical answer.

10. correct_answer must be exactly:
    A
    B
    C
    D

11. If the answer cannot be reliably determined:
    correct_answer = ""

12. Never guess an answer simply to fill the field.

============================================================
UGC-NET PAPER 1 CATEGORY
============================================================

Classify every question into EXACTLY ONE of these categories:

Teaching Aptitude

Research Aptitude

Comprehension

Communication

Mathematical Reasoning and Aptitude

Logical Reasoning

Data Interpretation

Information and Communication Technology (ICT)

People, Development and Environment

Higher Education System

Other

Choose the category based on the actual question.

Examples:

Teaching methods, learner characteristics,
evaluation, teaching levels:
Teaching Aptitude

Research methodology, hypothesis,
sampling, research ethics:
Research Aptitude

Passage-based question:
Comprehension

Communication models, barriers, verbal/non-verbal communication:
Communication

Number series, percentages, ratios, averages,
arithmetic:
Mathematical Reasoning and Aptitude

Arguments, syllogisms, statements,
logical inference:
Logical Reasoning

Charts, graphs, tables, datasets:
Data Interpretation

Internet, computer networks, operating systems,
ICT tools, digital technologies:
Information and Communication Technology (ICT)

Climate change, pollution, biodiversity,
sustainable development, environmental issues:
People, Development and Environment

Universities, UGC, accreditation, NEP,
higher education governance:
Higher Education System

============================================================
DIFFICULTY
============================================================

Classify each question as exactly one:

easy
medium
hard

Do not use any other value.

============================================================
TABLES AND GRAPHS
============================================================

If a question contains a table:

table_data = structured description of the table.

If a question contains a graph/chart/diagram:

graph_data = structured description of what is visible.

Never invent graph/table values.

If there is no graph:
graph_data = null

If there is no table:
table_data = null

============================================================
EXPLANATION
============================================================

Provide a short explanation only when the answer is supported.

Do not invent reasoning.

============================================================
YEAR / SESSION
============================================================

Use year and session from the supplied metadata when available.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Do not return Markdown.

Do not return ```json.

Use exactly this structure:

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
      "session": "Shift 2",
      "category": "Teaching Aptitude",
      "difficulty": "medium",
      "explanation": "...",
      "graph_data": null,
      "table_data": null
    }
  ]
}

If there are no reliable questions:

{
  "questions": []
}
"""


# ============================================================
# GROQ EXTRACTION
# ============================================================

def groq_extract(
    text,
    metadata
):

    prompt = f"""
SOURCE PDF:
{metadata.get("pdf_name", "")}

SOURCE URL:
{metadata.get("source_url", "")}

YEAR:
{metadata.get("year", "")}

SESSION:
{metadata.get("session", "")}

UGC-NET PAPER:
Paper 1

Extract all complete UGC-NET Paper 1 MCQs from
the OCR text below.

Important:

- Extract only actual questions.
- Do not invent.
- Do not paraphrase.
- Keep original options.
- All A-D options must be present.
- Classify every question into the correct UGC-NET Paper 1 category.
- Determine answer only when reliable.

OCR TEXT:

{text}
"""

    for attempt in range(
        4
    ):

        try:

            log.info(
                "Groq request attempt %s/4",
                attempt + 1
            )

            response = (
                groq_client
                .chat
                .completions
                .create(

                    model=GROQ_MODEL,

                    messages=[
                        {
                            "role": "system",
                            "content":
                                SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content":
                                prompt
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
                response
                .choices[0]
                .message
                .content
            )

            if not raw:
                return []

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

            # Add source metadata
            for q in questions:

                if not isinstance(
                    q,
                    dict
                ):
                    continue

                if not q.get(
                    "year"
                ):

                    q["year"] = (
                        metadata.get(
                            "year"
                        )
                    )

                if not q.get(
                    "session"
                ):

                    q["session"] = (
                        metadata.get(
                            "session",
                            ""
                        )
                    )

            log.info(
                "Groq returned %s questions",
                len(questions)
            )

            return questions

        except json.JSONDecodeError as e:

            log.warning(
                "Groq returned invalid JSON: %s",
                e
            )

        except Exception as e:

            log.warning(
                "Groq request failed: %s",
                e
            )

        if attempt < 3:

            time.sleep(
                2 ** attempt
            )

    return []


# ============================================================
# EXTRACT YEAR / SESSION
# ============================================================

def extract_metadata(
    text,
    pdf_url
):

    sample = (
        text[:8000]
        + " "
        + pdf_url
    )

    years = re.findall(
        r"\b20(?:0[9]|1[0-9]|2[0-9])\b",
        sample
    )

    year = None

    if years:

        # Pick the first valid year
        year = int(
            years[0]
        )

    lower = sample.lower()

    session = ""

    if (
        "shift 1" in lower
    ):

        session = "Shift 1"

    elif (
        "shift 2" in lower
    ):

        session = "Shift 2"

    elif (
        "morning" in lower
    ):

        session = "Morning"

    elif (
        "evening" in lower
    ):

        session = "Evening"

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

    parsed_path = urlparse(
        pdf_url
    )

    pdf_name = Path(
        parsed_path.path
    ).name

    if not pdf_name:

        pdf_name = (
            "ugc_net_paper1.pdf"
        )

    log.info(
        "=================================================="
    )

    log.info(
        "START PDF: %s",
        pdf_name
    )

    successful = False

    total_saved = 0

    try:

        # ------------------------------------------------
        # DOWNLOAD
        # ------------------------------------------------

        pdf_path = download_pdf(
            pdf_url
        )

        # ------------------------------------------------
        # OCR / TEXT
        # ------------------------------------------------

        text = pdf_to_text(
            pdf_path
        )

        text = clean_text(
            text
        )

        if len(text) < 100:

            raise RuntimeError(
                "Very little text extracted"
            )

        # ------------------------------------------------
        # METADATA
        # ------------------------------------------------

        metadata = extract_metadata(
            text,
            pdf_url
        )

        metadata[
            "pdf_name"
        ] = pdf_name

        metadata[
            "source_url"
        ] = pdf_url

        # ------------------------------------------------
        # SPLIT
        # ------------------------------------------------

        chunks = split_text(
            text,
            max_chars=28000
        )

        log.info(
            "Created %s Groq chunks",
            len(chunks)
        )

        # ------------------------------------------------
        # GROQ
        # ------------------------------------------------

        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):

            log.info(
                "Processing Groq chunk %s/%s",
                chunk_number,
                len(chunks)
            )

            questions = groq_extract(
                chunk,
                metadata
            )

            # ------------------------------------------------
            # D1
            # ------------------------------------------------

            for question in questions:

                if not isinstance(
                    question,
                    dict
                ):
                    continue

                try:

                    saved = save_question(
                        question,
                        pdf_url,
                        pdf_name,
                        metadata
                    )

                    if saved:
                        total_saved += 1

                except Exception as e:

                    log.exception(
                        "Question D1 insertion failed: %s",
                        e
                    )

                    # Do NOT mark PDF successful
                    # if a D1 question write failed.
                    raise

            # Release chunk data
            questions = None
            chunk = None

            gc.collect()

        # ------------------------------------------------
        # ONLY AFTER ALL QUESTIONS SUCCESSFULLY SENT
        # ------------------------------------------------

        mark_processed(
            pdf_url,
            pdf_name,
            total_saved
        )

        successful = True

        log.info(
            "SUCCESS: %s | %s questions saved",
            pdf_name,
            total_saved
        )

    except Exception as e:

        log.exception(
            "FAILED: %s | %s",
            pdf_name,
            e
        )

    finally:

        # ====================================================
        # DELETE PDF ONLY AFTER SUCCESS
        # ====================================================

        if successful:

            if (
                pdf_path
                and
                os.path.exists(
                    pdf_path
                )
            ):

                try:

                    os.remove(
                        pdf_path
                    )

                    log.info(
                        "Deleted PDF: %s",
                        pdf_path
                    )

                except Exception as e:

                    log.warning(
                        "Could not delete PDF: %s",
                        e
                    )

        else:

            log.warning(
                "Keeping failed PDF for retry: %s",
                pdf_path
            )

        # Release everything
        pdf_path = None

        gc.collect()

    return successful


# ============================================================
# MAIN
# ============================================================

def main():

    log.info(
        "=============================================="
    )

    log.info(
        "UGC-NET PAPER 1 PROCESSOR"
    )

    log.info(
        "=============================================="
    )

    log.info(
        "Groq model: %s",
        GROQ_MODEL
    )

    log.info(
        "Source: %s",
        SOURCE_URL
    )

    # ------------------------------------------------
    # D1
    # ------------------------------------------------

    create_tables()

    # ------------------------------------------------
    # SOURCE PAGE
    # ------------------------------------------------

    html = get_source_page()

    # ------------------------------------------------
    # PDF LINKS
    # ------------------------------------------------

    links = find_pdf_links(
        html
    )

    log.info(
        "Found %s candidate links",
        len(links)
    )

    selected = []

    for item in links:

        if allowed_year(
            item["url"],
            item["text"]
        ):

            selected.append(
                item
            )

    log.info(
        "Selected %s PDFs",
        len(selected)
    )

    if MAX_PAPERS > 0:

        selected = selected[
            :MAX_PAPERS
        ]

        log.info(
            "MAX_PAPERS applied: %s",
            MAX_PAPERS
        )

    # ------------------------------------------------
    # PROCESS
    # ------------------------------------------------

    success_count = 0
    skipped_count = 0
    failed_count = 0

    for number, item in enumerate(
        selected,
        start=1
    ):

        url = item["url"]

        log.info(
            "=============================================="
        )

        log.info(
            "PAPER %s/%s",
            number,
            len(selected)
        )

        # ------------------------------------------------
        # Skip already completed papers
        # ------------------------------------------------

        try:

            if already_processed(
                url
            ):

                log.info(
                    "Already processed: %s",
                    url
                )

                skipped_count += 1

                continue

        except Exception as e:

            log.warning(
                "Progress check failed: %s",
                e
            )

        # ------------------------------------------------
        # Process
        # ------------------------------------------------

        success = process_pdf(
            url,
            item["text"]
        )

        if success:

            success_count += 1

        else:

            failed_count += 1

        # Important memory cleanup
        gc.collect()

    # ------------------------------------------------
    # FINAL
    # ------------------------------------------------

    log.info(
        "=============================================="
    )

    log.info(
        "PROCESSING FINISHED"
    )

    log.info(
        "Successful: %s",
        success_count
    )

    log.info(
        "Skipped: %s",
        skipped_count
    )

    log.info(
        "Failed: %s",
        failed_count
    )

    log.info(
        "=============================================="
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
