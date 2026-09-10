import asyncio
import json
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from playwright.async_api import async_playwright


# ============================================================
# CONFIG
# ============================================================

CHATGPT_URL = "https://chatgpt.com/"
OUTPUT_FILE = "ugcnet_questions.json"

# Persistent browser profile.
# Login to ChatGPT manually the first time.
PROFILE_DIR = Path("./chatgpt_profile")

MAX_QUESTIONS_PER_REQUEST = 25
MAX_RETRIES = 4

# How long to wait for ChatGPT response.
RESPONSE_TIMEOUT = 180_000


# ============================================================
# USER INPUT
# ============================================================

def parse_years(value: str) -> List[int]:
    """
    Supports:

        2019
        2018,2019,2020
        2015-2020
        2015-2017,2019,2021-2022
    """

    years = set()

    value = value.strip()

    for part in value.split(","):
        part = part.strip()

        if not part:
            continue

        if "-" in part:
            pieces = part.split("-", 1)

            try:
                start = int(pieces[0])
                end = int(pieces[1])
            except ValueError:
                continue

            if start > end:
                start, end = end, start

            for year in range(start, end + 1):
                years.add(year)

        else:
            try:
                years.add(int(part))
            except ValueError:
                pass

    return sorted(years)


def get_user_config():

    print("=" * 70)
    print("UGC-NET VERIFIED PREVIOUS-YEAR QUESTION EXTRACTOR")
    print("CHATGPT + PLAYWRIGHT")
    print("=" * 70)

    years_input = input(
        "\nYears "
        "(example: 2019 / 2018,2019 / 2015-2020): "
    ).strip()

    years = parse_years(years_input)

    if not years:
        raise ValueError("Invalid year input.")

    number_input = input(
        "Number of questions (1-100): "
    ).strip()

    try:
        number_of_questions = int(number_input)
    except ValueError:
        raise ValueError("Question count must be a number.")

    if number_of_questions < 1:
        raise ValueError("Question count must be at least 1.")

    if number_of_questions > 100:
        number_of_questions = 100

    paper = input(
        "Paper/Subject "
        "(example: Paper 1, Computer Science): "
    ).strip()

    if not paper:
        paper = "Paper 1"

    difficulty = input(
        "Difficulty "
        "(easy / medium / hard / mixed): "
    ).strip().lower()

    if difficulty not in {
        "easy",
        "medium",
        "hard",
        "mixed"
    }:
        difficulty = "mixed"

    return {
        "years": years,
        "number": number_of_questions,
        "paper": paper,
        "difficulty": difficulty,
    }


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    years: List[int],
    paper: str,
    difficulty: str,
    count: int,
    existing_questions: List[Dict[str, Any]],
    batch_number: int
) -> str:

    years_text = ", ".join(str(y) for y in years)

    existing_text = ""

    if existing_questions:
        existing_text = json.dumps(
            [
                q.get("question", "")
                for q in existing_questions
                if q.get("question")
            ],
            ensure_ascii=False
        )

    prompt = f"""
You are a STRICT UGC-NET PREVIOUS-YEAR QUESTION
EXTRACTION AND VERIFICATION SYSTEM.

THIS IS NOT A QUESTION-GENERATION TASK.

You MUST NOT create new UGC-NET-style questions.

You MUST ONLY extract questions that were ACTUALLY ASKED
in a real UGC-NET examination.

============================================================
EXAM
============================================================

UGC NET

============================================================
PAPER / SUBJECT
============================================================

{paper}

============================================================
REQUESTED YEARS
============================================================

{years_text}

============================================================
DIFFICULTY
============================================================

{difficulty}

============================================================
BATCH
============================================================

{batch_number}

============================================================
NUMBER REQUESTED
============================================================

{count}

============================================================
MOST IMPORTANT RULE
============================================================

DO NOT HALLUCINATE.

A question is acceptable ONLY if you can confidently establish
that the question was ACTUALLY ASKED in UGC-NET.

DO NOT create a question merely because it looks like a
UGC-NET question.

DO NOT create a question from general knowledge.

DO NOT assign a requested year to a question unless that
question actually belongs to that year.

============================================================
STRICT VERIFICATION RULES
============================================================

For every question you return, verify internally:

1. Was this question actually asked in UGC-NET?

2. What exact year was this question asked?

3. Does that year belong to the requested years?

4. Does the question belong to the requested paper/subject?

5. Can the year be confidently established?

6. Can the original wording be established?

7. Can the original options be established?

8. Can the correct answer be confidently established?

9. Is this question different from questions already returned?

If ANY of these cannot be established confidently:

REJECT THE QUESTION.

DO NOT GUESS.

============================================================
YEAR RULE
============================================================

The "year" field MUST be the actual examination year.

NEVER do this:

Question known from UGC-NET
+
Requested year = 2019
=
year 2019

That is NOT acceptable.

The year must correspond to the actual occurrence of
that exact question.

If you are uncertain whether a question belongs to
2018 or 2019:

REJECT IT.

If you know the question but do not know its exact year:

REJECT IT.

============================================================
SESSION / SHIFT RULE
============================================================

If the original evidence identifies:

- Session
- Shift
- Date
- Paper
- Question ID

preserve that information.

If it is not available, use:

"session": null

DO NOT GUESS SESSION OR SHIFT.

============================================================
ORIGINAL WORDING RULE
============================================================

Preserve the original question wording.

DO NOT:

- paraphrase
- rewrite
- simplify
- summarize
- combine
- improve grammar
- change terminology
- change numbers
- change names
- change statements

If the exact wording cannot be confidently established:

REJECT THE QUESTION.

============================================================
OPTIONS RULE
============================================================

Preserve the original four options.

Every accepted question must contain:

A
B
C
D

DO NOT invent missing options.

DO NOT replace an option with a newly generated option.

If the original options cannot be confidently established:

REJECT THE QUESTION.

============================================================
ANSWER RULE
============================================================

The answer must be exactly:

"A"
"B"
"C"
or
"D"

Only provide an answer when it can be confidently verified.

If the correct answer is uncertain:

REJECT THE ENTIRE QUESTION.

============================================================
DIFFICULTY RULE
============================================================

Difficulty is secondary to historical verification.

DO NOT invent a question to satisfy the requested difficulty.

Difficulty must be:

easy
medium
hard

For "mixed", return a reasonable mixture ONLY among
already verified questions.

============================================================
NO HALLUCINATION COUNT RULE
============================================================

You are asked for {count} questions.

THIS DOES NOT MEAN YOU MUST RETURN {count} QUESTIONS.

If only 8 questions can be verified:

return 8.

If only 3 can be verified:

return 3.

If zero can be verified:

return an empty questions array.

NEVER invent additional questions to reach {count}.

FEWER VERIFIED QUESTIONS ARE ALWAYS BETTER THAN
HALLUCINATED QUESTIONS.

============================================================
DUPLICATE RULE
============================================================

Do not return:

- exact duplicates
- paraphrased duplicates
- slightly modified duplicates
- the same question multiple times

Questions already accepted:

{existing_text if existing_text else "NONE"}

============================================================
VERIFICATION FIELD
============================================================

Every returned question MUST contain:

"verification": {{
    "verified": true,
    "reason": "..."
}}

The reason must explain why the question's occurrence
and year can be confidently established.

Do NOT set verified=true for an uncertain question.

If you cannot verify it:

DO NOT OUTPUT IT.

============================================================
REQUIRED JSON FORMAT
============================================================

{{
  "questions": [
    {{
      "year": 2019,
      "session": null,
      "paper": "{paper}",
      "difficulty": "medium",
      "question": "EXACT ORIGINAL QUESTION",
      "options": {{
        "A": "ORIGINAL OPTION A",
        "B": "ORIGINAL OPTION B",
        "C": "ORIGINAL OPTION C",
        "D": "ORIGINAL OPTION D"
      }},
      "answer": "B",
      "verification": {{
        "verified": true,
        "reason": "The exact question occurrence and examination year were confidently established."
      }}
    }}
  ]
}}

============================================================
OUTPUT RULES
============================================================

Return ONLY valid JSON.

NO markdown.

NO ```json.

NO explanation outside JSON.

NO comments.

NO text before JSON.

NO text after JSON.

DO NOT GENERATE NEW QUESTIONS.

DO NOT HALLUCINATE YEARS.

DO NOT HALLUCINATE SOURCES.

DO NOT HALLUCINATE QUESTION IDs.

DO NOT HALLUCINATE SESSIONS.

ONLY RETURN VERIFIED PREVIOUSLY ASKED UGC-NET QUESTIONS.

If verification is insufficient, return fewer questions.

If verification is impossible, return:

{{
  "questions": []
}}

"""

    return prompt


# ============================================================
# JSON EXTRACTION
# ============================================================

def clean_json_text(text: str) -> str:

    text = text.strip()

    # Remove markdown fences.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    # Remove ChatGPT UI artifacts that may surround JSON.
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    # Find outer JSON object.
    first = text.find("{")
    last = text.rfind("}")

    if first >= 0 and last > first:
        text = text[first:last + 1]

    return text.strip()


def extract_json(text: str) -> Dict[str, Any]:

    cleaned = clean_json_text(text)

    try:
        data = json.loads(cleaned)

        if isinstance(data, dict):
            return data

    except json.JSONDecodeError:
        pass

    # More robust balanced-brace extraction.
    start_positions = [
        m.start()
        for m in re.finditer(r"\{", text)
    ]

    for start in start_positions:

        depth = 0
        in_string = False
        escaped = False

        for i in range(start, len(text)):

            char = text[i]

            if escaped:
                escaped = False
                continue

            if char == "\\" and in_string:
                escaped = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                depth += 1

            elif char == "}":
                depth -= 1

                if depth == 0:

                    candidate = text[start:i + 1]

                    try:
                        data = json.loads(candidate)

                        if isinstance(data, dict):
                            return data

                    except json.JSONDecodeError:
                        break

    raise ValueError(
        "Could not parse valid JSON from ChatGPT response."
    )


# ============================================================
# NORMALIZATION / HASHING
# ============================================================

def normalize_question(text: str) -> str:

    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text
    )

    return text


def question_hash(question: str) -> str:

    normalized = normalize_question(question)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


# ============================================================
# STRICT VALIDATION
# ============================================================

def validate_question(
    q: Dict[str, Any],
    allowed_years: List[int],
    expected_paper: str
) -> bool:

    if not isinstance(q, dict):
        return False

    question = q.get("question")
    options = q.get("options")
    answer = q.get("answer")
    year = q.get("year")
    difficulty = q.get("difficulty")

    # --------------------------------------------------------
    # VERIFICATION IS MANDATORY
    # --------------------------------------------------------

    verification = q.get("verification")

    if not isinstance(verification, dict):
        return False

    if verification.get("verified") is not True:
        return False

    reason = verification.get("reason")

    if not isinstance(reason, str):
        return False

    if len(reason.strip()) < 10:
        return False

    # --------------------------------------------------------
    # QUESTION
    # --------------------------------------------------------

    if not isinstance(question, str):
        return False

    question = question.strip()

    if len(question) < 10:
        return False

    # --------------------------------------------------------
    # OPTIONS
    # --------------------------------------------------------

    if not isinstance(options, dict):
        return False

    if set(options.keys()) != {"A", "B", "C", "D"}:
        return False

    for option in ["A", "B", "C", "D"]:

        if not isinstance(options[option], str):
            return False

        if not options[option].strip():
            return False

    # --------------------------------------------------------
    # ANSWER
    # --------------------------------------------------------

    if answer not in {"A", "B", "C", "D"}:
        return False

    # --------------------------------------------------------
    # YEAR
    # --------------------------------------------------------

    try:
        year = int(year)
    except Exception:
        return False

    if year not in allowed_years:
        return False

    # --------------------------------------------------------
    # DIFFICULTY
    # --------------------------------------------------------

    if difficulty not in {
        "easy",
        "medium",
        "hard"
    }:
        return False

    # --------------------------------------------------------
    # PAPER
    # --------------------------------------------------------

    paper = q.get("paper")

    if paper is not None:

        if not isinstance(paper, str):
            return False

        if not paper.strip():
            return False

    # --------------------------------------------------------
    # SESSION
    # --------------------------------------------------------

    session = q.get("session")

    if session is not None:

        if not isinstance(session, str):
            return False

        if not session.strip():
            return False

    return True


# ============================================================
# VALIDATE + DEDUPLICATE
# ============================================================

def validate_and_deduplicate(
    data: Dict[str, Any],
    allowed_years: List[int],
    expected_paper: str,
    existing: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    questions = data.get("questions", [])

    if not isinstance(questions, list):
        return []

    existing_hashes = {
        question_hash(q["question"])
        for q in existing
        if q.get("question")
    }

    result = []

    for q in questions:

        if not validate_question(
            q,
            allowed_years,
            expected_paper
        ):
            continue

        q_hash = question_hash(
            q["question"]
        )

        if q_hash in existing_hashes:
            continue

        existing_hashes.add(q_hash)

        result.append({
            "year": int(q["year"]),

            "session": q.get(
                "session",
                None
            ),

            "paper": q.get(
                "paper",
                expected_paper
            ),

            "difficulty": q["difficulty"],

            "question": q["question"].strip(),

            "options": {
                "A": q["options"]["A"].strip(),
                "B": q["options"]["B"].strip(),
                "C": q["options"]["C"].strip(),
                "D": q["options"]["D"].strip(),
            },

            "answer": q["answer"],

            "verification": {
                "verified": True,
                "reason": q["verification"]["reason"].strip()
            }
        })

    return result


# ============================================================
# CHATGPT UI HELPERS
# ============================================================

async def find_textarea(page):

    selectors = [
        'textarea',
        'textarea[placeholder*="Message"]',
        'textarea[placeholder*="message"]',
        '[contenteditable="true"]',
    ]

    for selector in selectors:

        try:

            locator = page.locator(
                selector
            ).last

            if await locator.count() > 0:

                if await locator.is_visible():
                    return locator

        except Exception:
            pass

    return None


async def send_message(
    page,
    message: str
):

    textarea = await find_textarea(page)

    if textarea is None:
        raise RuntimeError(
            "Could not find ChatGPT message input."
        )

    await textarea.click()

    await textarea.fill(message)

    await page.wait_for_timeout(500)

    await textarea.press("Enter")

    await page.wait_for_timeout(2000)


# ============================================================
# GET LAST ASSISTANT RESPONSE
# ============================================================

async def get_last_assistant_text(page) -> str:

    selectors = [
        '[data-message-author-role="assistant"]',
        '[data-testid*="conversation-turn"]',
        '[data-assistant-stream-block]',
    ]

    for selector in selectors:

        try:

            locator = page.locator(
                selector
            )

            count = await locator.count()

            if count > 0:

                last = locator.last

                text = await last.inner_text()

                if text and text.strip():
                    return text.strip()

        except Exception:
            pass

    # --------------------------------------------------------
    # IMPORTANT:
    # ChatGPT may expose the response as:
    #
    # <p data-assistant-stream-block="">
    #
    # inner_text() normally gives the visible text without
    # the HTML tags.
    # --------------------------------------------------------

    try:

        blocks = page.locator(
            '[data-assistant-stream-block]'
        )

        count = await blocks.count()

        if count > 0:

            texts = []

            for i in range(count):

                try:

                    txt = await blocks.nth(
                        i
                    ).inner_text()

                    if txt:
                        texts.append(txt)

                except Exception:
                    pass

            if texts:
                return "\n".join(texts).strip()

    except Exception:
        pass

    # Fallback.
    try:

        body_text = await page.locator(
            "body"
        ).inner_text()

        return body_text[-50000:]

    except Exception:
        return ""


# ============================================================
# WAIT FOR RESPONSE
# ============================================================

async def wait_for_response(
    page,
    old_text: str
):

    last_text = old_text

    stable_count = 0

    start = asyncio.get_event_loop().time()

    while True:

        elapsed = (
            asyncio.get_event_loop().time()
            - start
        )

        if elapsed > RESPONSE_TIMEOUT / 1000:

            raise TimeoutError(
                "ChatGPT response timed out."
            )

        await page.wait_for_timeout(1500)

        try:

            current = await get_last_assistant_text(
                page
            )

        except Exception:
            continue

        if not current:
            continue

        if current != last_text:

            last_text = current

            stable_count = 0

        else:

            stable_count += 1

        # Response appears finished when unchanged
        # several times.
        if stable_count >= 4:

            if len(current.strip()) > 20:
                return current

    return last_text


# ============================================================
# GENERATE / EXTRACT ONE BATCH
# ============================================================

async def generate_batch(
    page,
    years,
    paper,
    difficulty,
    count,
    existing,
    batch_number
):

    prompt = build_prompt(
        years=years,
        paper=paper,
        difficulty=difficulty,
        count=count,
        existing_questions=existing,
        batch_number=batch_number
    )

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        print(
            f"\nBatch {batch_number} | "
            f"Attempt {attempt}/{MAX_RETRIES}"
        )

        old_text = await get_last_assistant_text(
            page
        )

        await send_message(
            page,
            prompt
        )

        try:

            response = await wait_for_response(
                page,
                old_text
            )

        except Exception as e:

            print(
                "Response error:",
                e
            )

            if attempt < MAX_RETRIES:

                await page.wait_for_timeout(
                    3000
                )

                continue

            return []

        print(
            "Received response:",
            len(response),
            "characters"
        )

        # Debug preview.
        print(
            "\nResponse preview:\n",
            response[:1500]
        )

        try:

            data = extract_json(
                response
            )

            questions = validate_and_deduplicate(
                data=data,
                allowed_years=years,
                expected_paper=paper,
                existing=existing
            )

            print(
                "Verified new questions:",
                len(questions)
            )

            if questions:
                return questions

            # If ChatGPT returned zero verified questions,
            # do NOT force generation.
            print(
                "No sufficiently verified questions "
                "were returned."
            )

        except Exception as e:

            print(
                "JSON validation error:",
                e
            )

        # ----------------------------------------------------
        # STRICT RECOVERY PROMPT
        # ----------------------------------------------------

        prompt = f"""
Your previous response was rejected.

This is a STRICT HISTORICAL UGC-NET EXTRACTION TASK.

DO NOT GENERATE QUESTIONS.

Requested years:
{", ".join(map(str, years))}

Paper:
{paper}

Difficulty:
{difficulty}

Return up to {count} ONLY if you can confidently verify
that they were actually asked in UGC-NET.

For EVERY question:

- exact historical occurrence must be known
- exact year must be known
- year must be one of the requested years
- paper must match
- original question wording must be known
- original options must be known
- correct answer must be known

If ANYTHING is uncertain, reject that question.

DO NOT GUESS THE YEAR.

DO NOT ASSIGN A REQUESTED YEAR TO AN UNKNOWN QUESTION.

DO NOT CREATE UGC-NET-STYLE QUESTIONS.

DO NOT PARAPHRASE.

DO NOT INVENT OPTIONS.

DO NOT INVENT ANSWERS.

DO NOT INVENT SESSION INFORMATION.

It is completely acceptable to return fewer questions.

If no question can be confidently verified, return:

{{
  "questions": []
}}

Return ONLY valid JSON.

Required format:

{{
  "questions": [
    {{
      "year": 2019,
      "session": null,
      "paper": "{paper}",
      "difficulty": "medium",
      "question": "Exact original question",
      "options": {{
        "A": "Original option A",
        "B": "Original option B",
        "C": "Original option C",
        "D": "Original option D"
      }},
      "answer": "B",
      "verification": {{
        "verified": true,
        "reason": "The question occurrence and year were confidently established."
      }}
    }}
  ]
}}
"""

    return []


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    questions,
    config
):

    # Never exceed requested number.
    questions = questions[
        :config["number"]
    ]

    output = {
        "exam": "UGC NET",

        "paper": config["paper"],

        "years": config["years"],

        "requested_questions": config["number"],

        "total_questions": len(
            questions
        ),

        "difficulty": config["difficulty"],

        "verification_policy": (
            "Only questions marked as verified are "
            "included. Unverified questions are rejected."
        ),

        "questions": []
    }

    for index, q in enumerate(
        questions,
        start=1
    ):

        item = {
            "id": index,

            "year": q["year"],

            "session": q.get(
                "session",
                None
            ),

            "paper": q.get(
                "paper",
                config["paper"]
            ),

            "difficulty": q["difficulty"],

            "question": q["question"],

            "options": q["options"],

            "answer": q["answer"],

            "verification": q["verification"]
        }

        output["questions"].append(
            item
        )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=" * 70
    )

    print(
        "Requested:",
        config["number"]
    )

    print(
        "Verified:",
        len(questions)
    )

    print(
        "Output:",
        OUTPUT_FILE
    )

    if len(questions) < config["number"]:

        print(
            "\nWARNING:"
        )

        print(
            "Only verified questions were saved."
        )

        print(
            "The program intentionally did NOT "
            "invent questions to reach the requested count."
        )

    else:

        print(
            "\nSUCCESS:"
        )

        print(
            "Requested number of verified questions obtained."
        )


# ============================================================
# MAIN
# ============================================================

async def main():

    config = get_user_config()

    print(
        "\nConfiguration:"
    )

    print(
        json.dumps(
            config,
            indent=2
        )
    )

    PROFILE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    async with async_playwright() as p:

        print(
            "\nLaunching Chromium..."
        )

        context = (
            await p.chromium.launch_persistent_context(
                user_data_dir=str(
                    PROFILE_DIR
                ),

                headless=False,

                viewport={
                    "width": 1440,
                    "height": 900
                },

                args=[
                    "--disable-blink-features=AutomationControlled"
                ]
            )
        )

        pages = context.pages

        if pages:

            page = pages[0]

        else:

            page = await context.new_page()

        print(
            "Opening ChatGPT..."
        )

        await page.goto(
            CHATGPT_URL,
            wait_until="domcontentloaded",
            timeout=120_000
        )

        await page.wait_for_timeout(
            5000
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "CHATGPT LOGIN"
        )

        print(
            "=" * 70
        )

        print(
            "If ChatGPT asks you to log in, "
            "log in manually."
        )

        print(
            "The browser profile will remember "
            "your session."
        )

        print(
            "Do NOT put your ChatGPT password "
            "into this script."
        )

        # ----------------------------------------------------
        # Wait for ChatGPT input.
        # ----------------------------------------------------

        for _ in range(120):

            textarea = await find_textarea(
                page
            )

            if textarea is not None:
                break

            await page.wait_for_timeout(
                1000
            )

        textarea = await find_textarea(
            page
        )

        if textarea is None:

            print(
                "\nCould not find ChatGPT input."
            )

            print(
                "Check the browser and log in manually."
            )

            await context.close()

            return

        print(
            "\nChatGPT is ready."
        )

        all_questions = []

        target = config["number"]

        batch_number = 1

        # ----------------------------------------------------
        # Generate only verified questions.
        # ----------------------------------------------------

        while len(all_questions) < target:

            remaining = (
                target
                - len(all_questions)
            )

            batch_size = min(
                remaining,
                MAX_QUESTIONS_PER_REQUEST
            )

            print(
                "\n" + "-" * 70
            )

            print(
                "Progress:",
                f"{len(all_questions)}/{target}"
            )

            print(
                "Requesting up to:",
                batch_size,
                "VERIFIED questions"
            )

            new_questions = await generate_batch(
                page=page,

                years=config["years"],

                paper=config["paper"],

                difficulty=config["difficulty"],

                count=batch_size,

                existing=all_questions,

                batch_number=batch_number
            )

            if not new_questions:

                print(
                    "\nNo verified questions "
                    "received from this batch."
                )

                # ------------------------------------------------
                # Recovery with smaller batch.
                # ------------------------------------------------

                if batch_size > 5:

                    smaller_size = max(
                        5,
                        batch_size // 2
                    )

                    print(
                        "Trying smaller verified batch:",
                        smaller_size
                    )

                    new_questions = await generate_batch(
                        page=page,

                        years=config["years"],

                        paper=config["paper"],

                        difficulty=config["difficulty"],

                        count=smaller_size,

                        existing=all_questions,

                        batch_number=batch_number
                    )

                if not new_questions:

                    print(
                        "\nUnable to obtain another "
                        "verified batch."
                    )

                    print(
                        "Stopping rather than "
                        "inventing questions."
                    )

                    break

            # ----------------------------------------------------
            # Global duplicate protection.
            # ----------------------------------------------------

            existing_hashes = {
                question_hash(
                    q["question"]
                )
                for q in all_questions
            }

            added = 0

            for q in new_questions:

                h = question_hash(
                    q["question"]
                )

                if h not in existing_hashes:

                    all_questions.append(q)

                    existing_hashes.add(h)

                    added += 1

                if len(all_questions) >= target:
                    break

            print(
                "Verified questions added:",
                added
            )

            print(
                "Total verified questions:",
                len(all_questions)
            )

            # ----------------------------------------------------
            # Prevent endless loop if ChatGPT repeatedly
            # produces no new verified questions.
            # ----------------------------------------------------

            if added == 0:

                print(
                    "\nNo new unique verified "
                    "questions were added."
                )

                print(
                    "Stopping to prevent "
                    "repeated requests."
                )

                break

            batch_number += 1

            await page.wait_for_timeout(
                2000
            )

        # ----------------------------------------------------
        # Save.
        # ----------------------------------------------------

        save_json(
            all_questions,
            config
        )

        print(
            "\nBrowser will remain open."
        )

        print(
            "Press Ctrl+C to stop."
        )

        try:

            while True:

                await asyncio.sleep(
                    3600
                )

        except KeyboardInterrupt:
            pass

        await context.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print(
            "\nStopped by user."
        )

    except Exception as e:

        print(
            "\nFatal error:",
            repr(e)
        )
