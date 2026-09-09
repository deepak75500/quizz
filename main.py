import io
import re
import time
import hashlib
from typing import Optional, List, Dict, Any, Tuple

import requests
import fitz  # PyMuPDF
import pytesseract

from bs4 import BeautifulSoup
from PIL import Image, ImageOps, ImageFilter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "UGC NET Paper 1 Research API"

DEFAULT_SOURCE_URL = (
    "https://humanperitus.in/ugc-net-paper-1-previous-papers/"
)

REQUEST_TIMEOUT = 40

# OCR settings
OCR_DPI = 150

# Maximum PDFs that will be inspected for one request.
# Increase if you want to search more years/shifts.
MAX_PAPERS_TO_SCAN = 20

# Maximum pages per PDF.
MAX_PAGES_PER_PDF = 60

# User agent
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title=APP_NAME,
    version="2.0.0",
    description=(
        "UGC NET Paper 1 question research API using "
        "Human Peritus + PDF extraction + local OCR."
    ),
)


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=1)

    # "2025", "2024", etc.
    # "any" searches multiple years.
    year: str = "2025"

    category: str = "MCQ"

    count: int = Field(
        default=20,
        ge=1,
        le=200
    )

    # Optional override.
    # Normally the user does NOT need to send this.
    url: str = DEFAULT_SOURCE_URL


class Question(BaseModel):
    question_number: Optional[int] = None
    question: str

    options: Dict[str, str]

    answer: Optional[str] = None

    year: Optional[str] = None
    paper: Optional[str] = None

    source_url: Optional[str] = None

    score: float = 0.0


class ResearchResponse(BaseModel):
    success: bool
    topic: str
    year: str
    requested_count: int
    count: int

    source: str

    papers_scanned: int
    pages_scanned: int

    questions: List[Question]

    message: str


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()
session.headers.update(HEADERS)


# ============================================================
# TEXT UTILITIES
# ============================================================

def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\r", "\n")

    # Normalize common OCR artifacts
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("’", "'")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_for_search(text: str) -> str:
    text = text.lower()

    replacements = {
        "communication": "communication",
        "communicative": "communication",
        "communicator": "communication",
        "communications": "communication",
        "non verbal": "nonverbal",
        "non-verbal": "nonverbal",
    }

    for a, b in replacements.items():
        text = text.replace(a, b)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# TOPIC KEYWORDS
# ============================================================

TOPIC_ALIASES = {

    "communication": [
        "communication",
        "communicator",
        "communicative",
        "message",
        "sender",
        "receiver",
        "feedback",
        "encoding",
        "decoding",
        "channel",
        "noise",
        "barrier",
        "barriers",
        "verbal",
        "nonverbal",
        "non verbal",
        "interpersonal",
        "mass communication",
        "medium",
        "media",
        "semantic",
        "context",
        "communication model",
        "information flow",
    ],

    "teaching": [
        "teaching",
        "teacher",
        "learner",
        "learning",
        "pedagogy",
        "instruction",
        "classroom",
        "teaching method",
        "teaching methods",
        "evaluation",
    ],

    "research": [
        "research",
        "researcher",
        "hypothesis",
        "sampling",
        "sample",
        "population",
        "validity",
        "reliability",
        "research methodology",
        "qualitative",
        "quantitative",
        "research design",
        "variable",
    ],

    "ict": [
        "ict",
        "information technology",
        "computer",
        "internet",
        "network",
        "digital",
        "software",
        "hardware",
        "database",
        "web",
        "technology",
        "information and communication technology",
    ],

    "environment": [
        "environment",
        "pollution",
        "climate",
        "ecosystem",
        "biodiversity",
        "sustainable development",
        "greenhouse",
        "carbon",
        "environmental",
    ],

    "higher education": [
        "higher education",
        "university",
        "college",
        "ugc",
        "naac",
        "academic",
        "education policy",
        "institution",
        "accreditation",
    ],

    "logical reasoning": [
        "logical reasoning",
        "reasoning",
        "argument",
        "syllogism",
        "fallacy",
        "proposition",
        "inference",
        "logic",
    ],

    "data interpretation": [
        "data interpretation",
        "data",
        "table",
        "graph",
        "chart",
        "percentage",
        "ratio",
        "average",
        "bar graph",
        "pie chart",
    ],
}


def get_topic_terms(topic: str) -> List[str]:

    topic_clean = normalize_for_search(topic)

    for key, values in TOPIC_ALIASES.items():
        if key in topic_clean:
            return values

    # Generic topic
    words = topic_clean.split()

    words = [
        w for w in words
        if len(w) >= 3
    ]

    return words


# ============================================================
# FETCH WEB PAGE
# ============================================================

def fetch_html(url: str) -> str:

    response = session.get(
        url,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    return response.text


# ============================================================
# DISCOVER PDF LINKS
# ============================================================

def discover_paper_links(
    source_url: str,
    requested_year: str
) -> List[Dict[str, str]]:

    html = fetch_html(source_url)

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    papers = []

    anchors = soup.find_all("a")

    for a in anchors:

        href = a.get("href")

        if not href:
            continue

        text = clean_text(
            a.get_text(" ", strip=True)
        )

        href_lower = href.lower()

        # Must be PDF
        if ".pdf" not in href_lower:
            continue

        # Answer key is handled separately
        if "answer" in text.lower() or "answer-key" in href_lower:
            continue

        # Must look like Paper 1
        combined = (
            text.lower()
            + " "
            + href_lower
        )

        if "paper" not in combined:
            continue

        if "paper 1" not in combined and "paper-1" not in combined:
            continue

        # Year filter
        if requested_year.lower() != "any":

            if requested_year not in combined:
                continue

        papers.append({
            "title": text,
            "url": href
        })

    # Resolve relative URLs
    from urllib.parse import urljoin

    for p in papers:
        p["url"] = urljoin(
            source_url,
            p["url"]
        )

    # Deduplicate
    unique = []

    seen = set()

    for p in papers:

        if p["url"] in seen:
            continue

        seen.add(p["url"])

        unique.append(p)

    return unique


# ============================================================
# DOWNLOAD PDF
# ============================================================

def download_pdf(url: str) -> bytes:

    response = session.get(
        url,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "content-type",
        ""
    ).lower()

    data = response.content

    if len(data) < 1000:
        raise ValueError(
            "Downloaded PDF appears to be empty."
        )

    # Most PDFs begin with %PDF
    if not data.startswith(b"%PDF"):

        # Some servers don't return the correct content type
        if "pdf" not in content_type:

            raise ValueError(
                f"URL did not return a PDF: {url}"
            )

    return data


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_page_text_normal(
    page: fitz.Page
) -> str:

    try:

        text = page.get_text(
            "text",
            sort=True
        )

        return clean_text(text)

    except Exception:
        return ""


# ============================================================
# OCR IMAGE
# ============================================================

def render_page_for_ocr(
    page: fitz.Page,
    dpi: int = OCR_DPI
) -> Image.Image:

    zoom = dpi / 72

    matrix = fitz.Matrix(
        zoom,
        zoom
    )

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    img = Image.frombytes(
        "RGB",
        [
            pix.width,
            pix.height
        ],
        pix.samples
    )

    return img


def preprocess_image(
    image: Image.Image
) -> Image.Image:

    # Convert to grayscale
    image = ImageOps.grayscale(image)

    # Improve contrast
    image = ImageOps.autocontrast(image)

    # Light sharpening
    image = image.filter(
        ImageFilter.SHARPEN
    )

    return image


def ocr_page(
    page: fitz.Page
) -> str:

    try:

        image = render_page_for_ocr(
            page
        )

        image = preprocess_image(
            image
        )

        text = pytesseract.image_to_string(
            image,
            config="--psm 6"
        )

        return clean_text(text)

    except Exception as exc:

        print(
            "OCR error:",
            repr(exc)
        )

        return ""


# ============================================================
# DETERMINE WHETHER OCR IS NEEDED
# ============================================================

def needs_ocr(
    text: str
) -> bool:

    if not text:
        return True

    normalized = normalize_for_search(
        text
    )

    # If only PDF metadata is extracted,
    # OCR is probably required.
    question_markers = [
        "question number",
        "question id",
        "options",
        "option",
    ]

    marker_count = sum(
        1
        for marker in question_markers
        if marker in normalized
    )

    # If there is metadata but no actual
    # readable question-like content,
    # OCR anyway.
    if marker_count >= 1 and len(text) < 1500:
        return True

    # Very short page
    if len(text) < 300:
        return True

    return False


# ============================================================
# EXTRACT ALL PDF PAGES
# ============================================================

def extract_pdf_pages(
    pdf_bytes: bytes
) -> Tuple[List[str], int]:

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    page_texts = []

    pages_scanned = 0

    total_pages = min(
        len(document),
        MAX_PAGES_PER_PDF
    )

    for page_index in range(
        total_pages
    ):

        page = document[
            page_index
        ]

        text = extract_page_text_normal(
            page
        )

        # Important:
        #
        # Human Peritus PDFs can contain
        # scanned/image question content.
        #
        # Therefore OCR is used when
        # normal extraction isn't sufficient.
        if needs_ocr(text):

            ocr_text = ocr_page(
                page
            )

            if len(ocr_text) > len(text):

                text = ocr_text

            elif ocr_text:

                text = (
                    text
                    + "\n"
                    + ocr_text
                )

        page_texts.append(
            text
        )

        pages_scanned += 1

    document.close()

    return page_texts, pages_scanned


# ============================================================
# QUESTION NUMBER DETECTION
# ============================================================

QUESTION_NUMBER_PATTERNS = [

    re.compile(
        r"question\s*number\s*[:\-]?\s*(\d+)",
        re.I
    ),

    re.compile(
        r"question\s*no\.?\s*[:\-]?\s*(\d+)",
        re.I
    ),

    re.compile(
        r"^\s*(\d{1,3})\s*[\.\)]\s+",
        re.I
    ),

    re.compile(
        r"^\s*Q(?:uestion)?\s*(\d{1,3})\s*[\.\):\-]?",
        re.I
    ),
]


def find_question_number(
    line: str
) -> Optional[int]:

    line = line.strip()

    for pattern in QUESTION_NUMBER_PATTERNS:

        match = pattern.search(
            line
        )

        if match:

            try:
                number = int(
                    match.group(1)
                )

                if 1 <= number <= 200:
                    return number

            except Exception:
                pass

    return None


# ============================================================
# OPTION DETECTION
# ============================================================

OPTION_PATTERNS = [

    re.compile(
        r"^\s*\(?([A-Da-d])\)?\s*[\.\:\-]\s*(.+)$"
    ),

    re.compile(
        r"^\s*\(?([1-4])\)?\s*[\.\:\-]\s*(.+)$"
    ),

    re.compile(
        r"^\s*\(?([A-Da-d])\)?\s+(.+)$"
    ),
]


def normalize_option_key(
    key: str
) -> str:

    key = key.strip().upper()

    mapping = {
        "1": "A",
        "2": "B",
        "3": "C",
        "4": "D",
    }

    return mapping.get(
        key,
        key
    )


def parse_option(
    line: str
) -> Optional[Tuple[str, str]]:

    line = line.strip()

    for pattern in OPTION_PATTERNS:

        match = pattern.match(
            line
        )

        if not match:
            continue

        key = normalize_option_key(
            match.group(1)
        )

        value = clean_text(
            match.group(2)
        )

        if key in {
            "A",
            "B",
            "C",
            "D"
        } and value:

            return key, value

    return None


# ============================================================
# QUESTION BLOCK SPLITTING
# ============================================================

def split_question_blocks(
    page_texts: List[str]
) -> List[Dict[str, Any]]:

    blocks = []

    current = None

    for page_text in page_texts:

        lines = page_text.splitlines()

        for raw_line in lines:

            line = clean_text(
                raw_line
            )

            if not line:
                continue

            question_number = find_question_number(
                line
            )

            # ------------------------------------------------
            # New question
            # ------------------------------------------------

            if question_number is not None:

                # Save previous
                if current:

                    blocks.append(
                        current
                    )

                # Remove question number
                question_text = line

                for pattern in QUESTION_NUMBER_PATTERNS:

                    question_text = pattern.sub(
                        "",
                        question_text,
                        count=1
                    )

                current = {
                    "question_number":
                        question_number,

                    "question_lines": [],

                    "options": {},

                    "raw": []
                }

                if question_text.strip():

                    current[
                        "question_lines"
                    ].append(
                        question_text.strip()
                    )

                current[
                    "raw"
                ].append(line)

                continue

            # ------------------------------------------------
            # No current question
            # ------------------------------------------------

            if current is None:
                continue

            # ------------------------------------------------
            # Option
            # ------------------------------------------------

            option = parse_option(
                line
            )

            if option:

                key, value = option

                current[
                    "options"
                ][key] = value

                current[
                    "raw"
                ].append(line)

                continue

            # ------------------------------------------------
            # Skip metadata
            # ------------------------------------------------

            lower = line.lower()

            metadata_prefixes = (
                "question id",
                "question type",
                "question type:",
                "question category",
                "display question",
                "marks",
                "options:",
                "section:",
                "group number",
                "groupid",
                "status:",
            )

            if lower.startswith(
                metadata_prefixes
            ):
                continue

            # ------------------------------------------------
            # Normal question text
            # ------------------------------------------------

            current[
                "question_lines"
            ].append(line)

            current[
                "raw"
            ].append(line)

    if current:
        blocks.append(
            current
        )

    return blocks


# ============================================================
# IMPROVE OCR QUESTION BLOCKS
# ============================================================

def clean_question_block(
    block: Dict[str, Any]
) -> Optional[Dict[str, Any]]:

    question = clean_text(
        " ".join(
            block.get(
                "question_lines",
                []
            )
        )
    )

    options = block.get(
        "options",
        {}
    )

    # Remove obvious metadata
    question = re.sub(
        r"Question\s*(?:Number|No\.?)?\s*[:\-]?\s*\d+",
        "",
        question,
        flags=re.I
    )

    question = clean_text(
        question
    )

    # A valid MCQ should normally have
    # at least two options.
    if len(options) < 2:
        return None

    # OCR sometimes duplicates text.
    # Remove repeated whitespace.
    question = re.sub(
        r"\s+",
        " ",
        question
    )

    if len(question) < 10:
        return None

    return {
        "question_number":
            block.get(
                "question_number"
            ),

        "question":
            question,

        "options":
            options,
    }


# ============================================================
# ANSWER KEY EXTRACTION
# ============================================================

def parse_answer_key_text(
    text: str
) -> Dict[int, str]:

    answers = {}

    text = clean_text(
        text
    )

    # --------------------------------------------------------
    # Format examples handled:
    #
    # 1 3
    # 1 | 3
    # 1  |  3
    # Question 1 - 3
    # Q1 3
    # 1 : 3
    # --------------------------------------------------------

    patterns = [

        re.compile(
            r"^\s*(\d{1,3})\s*[\|\:\-]?\s*([1-4])\s*$",
            re.I
        ),

        re.compile(
            r"^\s*question\s*(\d{1,3})\s*"
            r"[\|\:\-]?\s*([1-4])\s*$",
            re.I
        ),

        re.compile(
            r"^\s*q\s*(\d{1,3})\s*"
            r"[\|\:\-]?\s*([1-4])\s*$",
            re.I
        ),

        re.compile(
            r"^\s*(\d{1,3})\s+([1-4])\s*$",
            re.I
        ),
    ]

    for line in text.splitlines():

        line = clean_text(
            line
        )

        for pattern in patterns:

            match = pattern.match(
                line
            )

            if not match:
                continue

            try:

                q_no = int(
                    match.group(1)
                )

                answer_number = int(
                    match.group(2)
                )

                if 1 <= q_no <= 200:

                    answers[
                        q_no
                    ] = normalize_option_key(
                        str(answer_number)
                    )

                    break

            except Exception:
                pass

    # --------------------------------------------------------
    # Fallback:
    # Search text globally.
    # --------------------------------------------------------

    if not answers:

        matches = re.findall(
            r"\b(\d{1,3})\s*[\|\:\-]\s*([1-4])\b",
            text
        )

        for q_no, ans in matches:

            q_no = int(q_no)

            if 1 <= q_no <= 200:

                answers[
                    q_no
                ] = normalize_option_key(
                    ans
                )

    return answers


def extract_answer_key(
    pdf_bytes: bytes
) -> Dict[int, str]:

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    all_text = []

    for page in document:

        text = extract_page_text_normal(
            page
        )

        if len(text) < 50:

            ocr_text = ocr_page(
                page
            )

            if len(ocr_text) > len(text):

                text = ocr_text

        all_text.append(
            text
        )

    document.close()

    return parse_answer_key_text(
        "\n".join(
            all_text
        )
    )


# ============================================================
# FIND ANSWER KEY URL
# ============================================================

def normalize_filename(
    url: str
) -> str:

    url = url.lower()

    url = url.replace(
        "_",
        "-"
    )

    url = re.sub(
        r"[^a-z0-9\-]",
        "",
        url
    )

    return url


def find_best_answer_key(
    paper_url: str,
    answer_links: List[Dict[str, str]]
) -> Optional[str]:

    paper_norm = normalize_filename(
        paper_url
    )

    # Extract filename
    paper_name = paper_url.split(
        "/"
    )[-1]

    paper_name = paper_name.lower()

    # Remove question related words
    target_tokens = set(
        re.findall(
            r"[a-z0-9]+",
            paper_name
        )
    )

    target_tokens -= {
        "question",
        "questions",
        "paper",
        "ugc",
        "net",
        "1",
        "pdf",
    }

    best_url = None
    best_score = -1

    for answer in answer_links:

        ans_url = answer["url"]

        ans_name = ans_url.lower()

        ans_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                ans_name
            )
        )

        score = len(
            target_tokens
            &
            ans_tokens
        )

        # Same date / shift is important.
        for token in [
            "morning",
            "evening",
            "shift",
        ]:

            if token in paper_norm and token in ans_name:

                score += 5

        # Year match
        years = re.findall(
            r"20\d{2}",
            paper_url
        )

        for year in years:

            if year in ans_name:

                score += 10

        if score > best_score:

            best_score = score
            best_url = ans_url

    return best_url


# ============================================================
# DISCOVER ANSWER LINKS
# ============================================================

def discover_answer_links(
    source_url: str,
    requested_year: str
) -> List[Dict[str, str]]:

    html = fetch_html(
        source_url
    )

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    from urllib.parse import urljoin

    results = []

    for a in soup.find_all("a"):

        href = a.get(
            "href"
        )

        if not href:
            continue

        text = clean_text(
            a.get_text(
                " ",
                strip=True
            )
        )

        combined = (
            text.lower()
            + " "
            + href.lower()
        )

        if ".pdf" not in href.lower():
            continue

        if "answer" not in combined:
            continue

        if (
            requested_year.lower()
            != "any"
            and requested_year
            not in combined
        ):
            continue

        results.append({
            "title": text,
            "url": urljoin(
                source_url,
                href
            )
        })

    unique = []

    seen = set()

    for item in results:

        if item["url"] in seen:
            continue

        seen.add(
            item["url"]
        )

        unique.append(
            item
        )

    return unique


# ============================================================
# TOPIC SCORING
# ============================================================

def score_question(
    question: str,
    options: Dict[str, str],
    topic: str
) -> float:

    text = normalize_for_search(
        question
        + " "
        + " ".join(
            options.values()
        )
    )

    topic_normalized = normalize_for_search(
        topic
    )

    terms = get_topic_terms(
        topic
    )

    score = 0.0

    # Exact topic phrase
    if topic_normalized in text:

        score += 15.0

    # Individual terms
    for term in terms:

        term_normalized = normalize_for_search(
            term
        )

        if not term_normalized:
            continue

        occurrences = text.count(
            term_normalized
        )

        if occurrences:

            # First occurrence has highest value.
            score += min(
                occurrences,
                3
            ) * 3.0

    # More precise matching
    if (
        "communication"
        in topic_normalized
    ):

        special_terms = [
            "feedback",
            "encoding",
            "decoding",
            "barrier",
            "noise",
            "sender",
            "receiver",
            "channel",
            "message",
            "nonverbal",
            "interpersonal",
        ]

        for term in special_terms:

            if term in text:

                score += 2.5

    return score


# ============================================================
# DEDUPLICATION
# ============================================================

def question_fingerprint(
    question: str,
    options: Dict[str, str]
) -> str:

    raw = (
        normalize_for_search(
            question
        )
        + "|"
        + "|".join(
            normalize_for_search(
                options.get(
                    key,
                    ""
                )
            )
            for key in [
                "A",
                "B",
                "C",
                "D",
            ]
        )
    )

    return hashlib.sha256(
        raw.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# PAPER PROCESSING
# ============================================================

def process_paper(
    paper: Dict[str, str],
    answer_key_url: Optional[str],
    topic: str
) -> Tuple[
    List[Dict[str, Any]],
    int
]:

    print(
        f"Processing paper: {paper['title']}"
    )

    print(
        f"PDF: {paper['url']}"
    )

    try:

        pdf_bytes = download_pdf(
            paper["url"]
        )

    except Exception as exc:

        print(
            "Paper download failed:",
            exc
        )

        return [], 0

    try:

        page_texts, pages_scanned = (
            extract_pdf_pages(
                pdf_bytes
            )
        )

    except Exception as exc:

        print(
            "PDF extraction failed:",
            exc
        )

        return [], 0

    blocks = split_question_blocks(
        page_texts
    )

    questions = []

    for block in blocks:

        cleaned = clean_question_block(
            block
        )

        if not cleaned:
            continue

        score = score_question(
            cleaned["question"],
            cleaned["options"],
            topic
        )

        cleaned["score"] = score

        cleaned["paper"] = paper[
            "title"
        ]

        cleaned["source_url"] = paper[
            "url"
        ]

        questions.append(
            cleaned
        )

    # --------------------------------------------------------
    # Answer key
    # --------------------------------------------------------

    answers = {}

    if answer_key_url:

        print(
            f"Answer key: {answer_key_url}"
        )

        try:

            answer_bytes = download_pdf(
                answer_key_url
            )

            answers = extract_answer_key(
                answer_bytes
            )

        except Exception as exc:

            print(
                "Answer key failed:",
                exc
            )

    # Add answers
    for question in questions:

        q_no = question.get(
            "question_number"
        )

        question["answer"] = (
            answers.get(
                q_no
            )
            if q_no is not None
            else None
        )

    return questions, pages_scanned


# ============================================================
# YEAR EXTRACTION
# ============================================================

def extract_year(
    text: str
) -> Optional[str]:

    match = re.search(
        r"\b(20\d{2})\b",
        text
    )

    if match:

        return match.group(1)

    return None


# ============================================================
# MAIN RESEARCH
# ============================================================

def perform_research(
    request: ResearchRequest
) -> ResearchResponse:

    source_url = (
        request.url
        or DEFAULT_SOURCE_URL
    )

    print("=" * 70)

    print(
        "TOPIC:",
        request.topic
    )

    print(
        "YEAR:",
        request.year
    )

    print(
        "COUNT:",
        request.count
    )

    print(
        "SOURCE:",
        source_url
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Discover papers
    # --------------------------------------------------------

    try:

        papers = discover_paper_links(
            source_url,
            request.year
        )

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                "Could not read Human Peritus source page: "
                + str(exc)
            )
        )

    if not papers:

        return ResearchResponse(
            success=False,
            topic=request.topic,
            year=request.year,
            requested_count=request.count,
            count=0,
            source=source_url,
            papers_scanned=0,
            pages_scanned=0,
            questions=[],
            message=(
                "No Paper 1 PDF links were found "
                "for the requested year."
            )
        )

    # --------------------------------------------------------
    # Discover answer keys
    # --------------------------------------------------------

    try:

        answer_links = discover_answer_links(
            source_url,
            request.year
        )

    except Exception:

        answer_links = []

    print(
        f"Found {len(papers)} paper PDFs."
    )

    print(
        f"Found {len(answer_links)} answer-key PDFs."
    )

    # Limit scanning
    papers = papers[
        :MAX_PAPERS_TO_SCAN
    ]

    all_questions = []

    total_pages = 0

    # --------------------------------------------------------
    # Process papers
    # --------------------------------------------------------

    for paper in papers:

        answer_url = find_best_answer_key(
            paper["url"],
            answer_links
        )

        questions, pages = process_paper(
            paper,
            answer_url,
            request.topic
        )

        total_pages += pages

        all_questions.extend(
            questions
        )

        # ----------------------------------------------------
        # If we already have enough strong matches,
        # don't unnecessarily OCR more PDFs.
        # ----------------------------------------------------

        if len(all_questions) >= (
            request.count * 3
        ):

            # enough candidates
            break

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    unique_questions = []

    seen = set()

    for q in all_questions:

        fp = question_fingerprint(
            q["question"],
            q["options"]
        )

        if fp in seen:
            continue

        seen.add(fp)

        unique_questions.append(
            q
        )

    # --------------------------------------------------------
    # Sort by relevance
    # --------------------------------------------------------

    unique_questions.sort(
        key=lambda x: (
            x.get(
                "score",
                0
            ),
            len(
                x.get(
                    "question",
                    ""
                )
            )
        ),
        reverse=True
    )

    # --------------------------------------------------------
    # If topic matching gives no results,
    # return a useful fallback.
    # --------------------------------------------------------

    matched = [
        q
        for q in unique_questions
        if q.get(
            "score",
            0
        ) > 0
    ]

    if matched:

        final_questions = matched[
            :request.count
        ]

    else:

        final_questions = unique_questions[
            :request.count
        ]

    # --------------------------------------------------------
    # Add year
    # --------------------------------------------------------

    final_output = []

    for q in final_questions:

        item = dict(q)

        item["year"] = (
            extract_year(
                item.get(
                    "paper",
                    ""
                )
            )
            or request.year
        )

        final_output.append(
            item
        )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    if not final_output:

        message = (
            "No questions could be extracted. "
            "The source PDFs may require OCR or "
            "their layout may have changed."
        )

    elif len(final_output) < request.count:

        message = (
            f"Found only {len(final_output)} "
            f"matching/extractable questions, "
            f"while {request.count} were requested."
        )

    else:

        message = (
            f"Successfully extracted "
            f"{len(final_output)} questions."
        )

    return ResearchResponse(
        success=True,
        topic=request.topic,
        year=request.year,
        requested_count=request.count,
        count=len(final_output),
        source=source_url,
        papers_scanned=len(papers),
        pages_scanned=total_pages,
        questions=final_output,
        message=message
    )


# ============================================================
# API ENDPOINT
# ============================================================

@app.post(
    "/research",
    response_model=ResearchResponse
)
def research(
    request: ResearchRequest
):

    start = time.time()

    try:

        result = perform_research(
            request
        )

        elapsed = time.time() - start

        print(
            f"Completed in {elapsed:.2f}s"
        )

        return result

    except HTTPException:
        raise

    except Exception as exc:

        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    try:

        version = fitz.__doc__

    except Exception:

        version = None

    try:

        tesseract_version = (
            pytesseract.get_tesseract_version()
        )

        tesseract_available = True

    except Exception:

        tesseract_version = None
        tesseract_available = False

    return {
        "status": "healthy",

        "service": APP_NAME,

        "source": DEFAULT_SOURCE_URL,

        "ocr": {
            "available":
                tesseract_available,

            "version":
                str(tesseract_version)
                if tesseract_version
                else None
        },

        "pymupdf":
            version,

        "gemini_required":
            False,

        "openai_required":
            False,

        "database_required":
            False
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "service": APP_NAME,

        "status": "running",

        "source": DEFAULT_SOURCE_URL,

        "usage": {
            "method": "POST",
            "endpoint": "/research"
        },

        "example": {
            "topic": "communication",
            "year": "2025",
            "category": "MCQ",
            "count": 20
        },

        "features": [
            "Human Peritus automatic source",
            "PDF discovery",
            "Answer key discovery",
            "PDF text extraction",
            "Local OCR",
            "Topic matching",
            "Question deduplication",
            "No Gemini API",
            "No OpenAI API",
            "No database"
        ]
    }


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
