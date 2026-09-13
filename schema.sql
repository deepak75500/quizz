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

CREATE INDEX IF NOT EXISTS idx_questions_year
ON questions(year);

CREATE INDEX IF NOT EXISTS idx_questions_category
ON questions(category);

CREATE INDEX IF NOT EXISTS idx_questions_difficulty
ON questions(difficulty);

CREATE INDEX IF NOT EXISTS idx_questions_session
ON questions(session);

CREATE INDEX IF NOT EXISTS idx_questions_hash
ON questions(question_hash);


CREATE TABLE IF NOT EXISTS processed_papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    pdf_url TEXT UNIQUE NOT NULL,

    pdf_name TEXT,

    question_count INTEGER DEFAULT 0,

    processed_at TEXT DEFAULT CURRENT_TIMESTAMP
);
