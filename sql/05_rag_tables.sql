-- Tables for the explanation layer. Run once, after review; nothing here is created automatically.
-- GOLD.KB_DOCS holds only the 10 reusable knowledge documents: text, sources and hashes. Embeddings are
-- computed locally and cached on disk, so the table has no vector column.
-- GOLD.EXPLANATIONS holds generated explanations. It stores ids, hashes and text; there is no
-- column for evaluation outcomes.

USE SCHEMA DRIVE_HEALTH.GOLD;

CREATE TABLE IF NOT EXISTS DRIVE_HEALTH.GOLD.KB_DOCS (
    doc_id               VARCHAR       NOT NULL,
    source_type          VARCHAR       NOT NULL,
    title                VARCHAR       NOT NULL,
    text                 VARCHAR       NOT NULL,
    source_name          VARCHAR       NOT NULL,
    source_reference     VARCHAR       NOT NULL,
    source_version       VARCHAR,
    source_accessed_on   DATE          NOT NULL,
    additional_sources   VARIANT,
    text_sha256          VARCHAR(64)   NOT NULL,
    kb_version           VARCHAR       NOT NULL,
    created_at           TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (doc_id)
);

CREATE TABLE IF NOT EXISTS DRIVE_HEALTH.GOLD.EXPLANATIONS (
    explanation_id            VARCHAR(36)   NOT NULL,
    created_at                TIMESTAMP_NTZ NOT NULL,
    kb_version                VARCHAR       NOT NULL,
    question                  VARCHAR       NOT NULL,
    question_sha256           VARCHAR(64)   NOT NULL,
    request_kind              VARCHAR       NOT NULL,
    run_id                    VARCHAR(36),
    method                    VARCHAR,
    serial_number             VARCHAR,
    prompt_version            VARCHAR       NOT NULL,
    llm_model                 VARCHAR,
    llm_params                VARIANT,
    embedding_model           VARCHAR,
    retrieved_doc_ids         ARRAY,
    retrieval_scores          ARRAY,
    min_similarity_used       FLOAT,
    drive_report_sha256       VARCHAR(64),
    retrieved_context_sha256  VARCHAR(64),
    cited_doc_ids             ARRAY,
    answer_text               VARCHAR       NOT NULL,
    answer_source             VARCHAR       NOT NULL,
    validation_status         VARCHAR       NOT NULL,
    validation_errors         VARIANT,
    llm_attempts              NUMBER(3,0),
    prompt_tokens             NUMBER(10,0),
    completion_tokens         NUMBER(10,0),
    latency_ms                NUMBER(10,0),
    cache_key                 VARCHAR(64)   NOT NULL,
    PRIMARY KEY (explanation_id)
);
