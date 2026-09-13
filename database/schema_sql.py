"""记忆系统的 MySQL DDL 集中维护。

采用"创建不存在 + ALTER 补齐"策略，对历史表缺失列做补齐。
新代码统一使用 `vector_id` / `category` / `relation` / `weight` / `user_id`。
"""

import logging

from utils.request_context import log_exception

logger = logging.getLogger(__name__)

# ---------- DDL：幂等创建 ----------
DDL_STATEMENTS: tuple[str, ...] = (
    # ---------- 人格 ----------
    """
    CREATE TABLE IF NOT EXISTS personas (
        user_id     VARCHAR(128) NOT NULL PRIMARY KEY,
        persona     TEXT NOT NULL,
        updated_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # ---------- 分层记忆（统一管理 L0/L1/L2） ----------
    """
    CREATE TABLE IF NOT EXISTS memories (
        id              BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        user_id         VARCHAR(128) NOT NULL,
        role_id         VARCHAR(128) NOT NULL DEFAULT 'default',
        level           VARCHAR(4) NOT NULL DEFAULT 'L1',
        content         TEXT NOT NULL,
        summary         TEXT NULL,
        emotion_tag     VARCHAR(32) NOT NULL DEFAULT 'neutral',
        emotion_intensity FLOAT NOT NULL DEFAULT 0.0,
        vector_id       VARCHAR(64) NULL,
        parent_id       BIGINT NULL,
        importance      FLOAT NOT NULL DEFAULT 0.5,
        memory_type     VARCHAR(64) NOT NULL DEFAULT 'fact',
        category        VARCHAR(64) NOT NULL DEFAULT 'general',
        embedding_dim   INT NULL,
        created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_mem_user_role_level (user_id, role_id, level, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # ---------- 记忆关系 ----------
    """
    CREATE TABLE IF NOT EXISTS memory_relations (
        id          BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        user_id     VARCHAR(128) NOT NULL,
        role_id     VARCHAR(128) NOT NULL DEFAULT 'default',
        source_id   BIGINT NOT NULL,
        target_id   BIGINT NOT NULL,
        relation    VARCHAR(16) NOT NULL,
        weight      FLOAT NOT NULL DEFAULT 1.0,
        created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_rel_user_source (user_id, source_id),
        INDEX idx_rel_user_target (user_id, target_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # ---------- 抽取日志 ----------
    """
    CREATE TABLE IF NOT EXISTS memory_extract_logs (
        id              BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        user_id         VARCHAR(128) NOT NULL,
        session_id      VARCHAR(64) NOT NULL,
        user_msg        TEXT NOT NULL,
        assistant_msg   TEXT NOT NULL,
        status          VARCHAR(16) NOT NULL DEFAULT 'pending',
        error           TEXT NULL,
        created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        finished_at     DATETIME NULL,
        INDEX idx_log_user_status (user_id, status)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # ---------- 情感分析日志 ----------
    """
    CREATE TABLE IF NOT EXISTS emotion_logs (
        id          BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        user_id     VARCHAR(128) NOT NULL,
        text        TEXT NOT NULL,
        emotion     VARCHAR(32) NOT NULL,
        intensity   FLOAT NOT NULL,
        reason      VARCHAR(255) NULL,
        created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_emo_user (user_id, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # ---------- 对话会话（短期对话记忆，毫秒级轻量沟通） ----------
    """
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id          BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        user_id     VARCHAR(128) NOT NULL,
        role_id     VARCHAR(128) NOT NULL DEFAULT 'default',
        session_id  VARCHAR(64) NOT NULL,
        summary     TEXT NULL,
        last_msg_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        msg_count   INT NOT NULL DEFAULT 0,
        importance  FLOAT NOT NULL DEFAULT 0.3,
        -- 遗忘权重：随时间衰减（decay_factor），按访问频率/重要性累加保留权重
        retain_score FLOAT NOT NULL DEFAULT 1.0,
        archived    TINYINT(1) NOT NULL DEFAULT 0,
        created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uk_session (session_id),
        INDEX idx_chsess_user_retain (user_id, role_id, retain_score, last_msg_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # ---------- 对话消息（多轮上下文缓冲） ----------
    """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id          BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        session_id  VARCHAR(64) NOT NULL,
        user_id     VARCHAR(128) NOT NULL,
        role_id     VARCHAR(128) NOT NULL DEFAULT 'default',
        role        VARCHAR(16) NOT NULL,
        content     TEXT NOT NULL,
        token_count INT NOT NULL DEFAULT 0,
        created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_chmsg_session (session_id, created_at),
        INDEX idx_chmsg_user_role (user_id, role_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # ====== Role Core 8 张新表(PRD 02_PRD_data_model.md, M0 引入)======
    # ---------- 1. 结构化人格(role_persona) ----------
    """
    CREATE TABLE IF NOT EXISTS role_persona (
        user_id          BIGINT NOT NULL,
        role_id          BIGINT NOT NULL,
        identity         TEXT NULL,
        background       TEXT NULL,
        traits_tags      VARCHAR(2048) NULL,
        `values`         TEXT NULL,
        speaking_style   TEXT NULL,
        taboos           TEXT NULL,
        example_dialogs  TEXT NULL,
        version          INT NOT NULL DEFAULT 1,
        created_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (user_id, role_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # ---------- 2. OCEAN 五维(role_traits) ----------
    """
    CREATE TABLE IF NOT EXISTS role_traits (
        user_id            BIGINT NOT NULL,
        role_id            BIGINT NOT NULL,
        openness           DECIMAL(5,3) NOT NULL DEFAULT 0,
        conscientiousness  DECIMAL(5,3) NOT NULL DEFAULT 0,
        extraversion       DECIMAL(5,3) NOT NULL DEFAULT 0,
        agreeableness      DECIMAL(5,3) NOT NULL DEFAULT 0,
        neuroticism        DECIMAL(5,3) NOT NULL DEFAULT 0,
        preset_type        VARCHAR(32) NOT NULL DEFAULT 'companion_default',
        created_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (user_id, role_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # ---------- 3. 三桶心情(role_mood) ----------
    """
    CREATE TABLE IF NOT EXISTS role_mood (
        user_id            BIGINT NOT NULL,
        role_id            BIGINT NOT NULL,
        instant_val        DECIMAL(5,3) NOT NULL DEFAULT 0,
        instant_intensity  DECIMAL(5,3) NOT NULL DEFAULT 0,
        instant_emotion    VARCHAR(32) NOT NULL DEFAULT 'neutral',
        short_val          DECIMAL(5,3) NOT NULL DEFAULT 0,
        short_intensity    DECIMAL(5,3) NOT NULL DEFAULT 0,
        short_emotion      VARCHAR(32) NOT NULL DEFAULT 'neutral',
        baseline_val       DECIMAL(5,3) NOT NULL DEFAULT 0,
        baseline_intensity DECIMAL(5,3) NOT NULL DEFAULT 0,
        baseline_emotion   VARCHAR(32) NOT NULL DEFAULT 'neutral',
        last_event_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        created_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (user_id, role_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # ---------- 4. 信念清单(role_belief) ----------
    """
    CREATE TABLE IF NOT EXISTS role_belief (
        id              BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        user_id         BIGINT NOT NULL,
        role_id         BIGINT NOT NULL,
        topic           VARCHAR(128) NOT NULL,
        stance          TEXT NOT NULL,
        confidence      DECIMAL(4,3) NOT NULL DEFAULT 0,
        evidence_count  INT NOT NULL DEFAULT 0,
        source          VARCHAR(32) NOT NULL DEFAULT 'user_edit',
        user_confirmed  TINYINT(1) NOT NULL DEFAULT 0,
        created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_belief_user_role (user_id, role_id),
        INDEX idx_belief_confirmed (user_confirmed)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # ---------- 5. 三维关系值(role_relationship) ----------
    """
    CREATE TABLE IF NOT EXISTS role_relationship (
        user_id       BIGINT NOT NULL,
        role_id       BIGINT NOT NULL,
        intimacy      DECIMAL(4,3) NOT NULL DEFAULT 0,
        trust         DECIMAL(4,3) NOT NULL DEFAULT 0,
        satisfaction  DECIMAL(4,3) NOT NULL DEFAULT 0,
        co_days       INT NOT NULL DEFAULT 0,
        updated_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (user_id, role_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # ---------- 6. 审计日志(role_core_audit_log) ----------
    """
    CREATE TABLE IF NOT EXISTS role_core_audit_log (
        id            BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        user_id       BIGINT NOT NULL,
        role_id       BIGINT NOT NULL,
        action        VARCHAR(32) NOT NULL,
        target_type   VARCHAR(32) NOT NULL,
        before_json   TEXT NULL,
        after_json    TEXT NULL,
        delta_summary TEXT NULL,
        trigger_event VARCHAR(64) NULL,
        created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_audit_user_role_time (user_id, role_id, created_at),
        INDEX idx_audit_action (action),
        INDEX idx_audit_target (target_type)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # ---------- 7. 用户反馈(role_feedback) ----------
    """
    CREATE TABLE IF NOT EXISTS role_feedback (
        id         BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        user_id    BIGINT NOT NULL,
        role_id    BIGINT NOT NULL,
        message_id VARCHAR(64) NOT NULL,
        rating     INT NOT NULL,
        comment    TEXT NULL,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_feedback_user_role_time (user_id, role_id, created_at),
        INDEX idx_feedback_message (message_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # ---------- 8. 演化建议(role_evolution_suggestion) ----------
    """
    CREATE TABLE IF NOT EXISTS role_evolution_suggestion (
        id              BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        user_id         BIGINT NOT NULL,
        role_id         BIGINT NOT NULL,
        target_type     VARCHAR(32) NOT NULL,
        suggestion_json TEXT NOT NULL,
        status          VARCHAR(16) NOT NULL DEFAULT 'pending',
        source          VARCHAR(32) NOT NULL DEFAULT 'memory_extract',
        confidence      DECIMAL(4,3) NOT NULL DEFAULT 0,
        reason          TEXT NULL,
        created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        resolved_at     DATETIME NULL,
        INDEX idx_sug_user_role_status_time (user_id, role_id, status, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
)


# ---------- ALTER：补齐历史表缺失列 ----------
# 这些语句必须支持重复执行且在列不存在时才创建（MySQL 没有 IF NOT EXISTS ADD COLUMN，
# 需要通过 information_schema 判断后单独执行）。
ALTER_PROBES: tuple[tuple[str, str], ...] = (
    # (table, probe_sql)
    # ---------- memories ----------
    ("memories", "ALTER TABLE memories ADD COLUMN memory_type VARCHAR(64) NOT NULL DEFAULT 'fact'"),
    ("memories", "ALTER TABLE memories ADD COLUMN category VARCHAR(64) NOT NULL DEFAULT 'general'"),
    ("memories", "ALTER TABLE memories ADD COLUMN embedding_dim INT NULL"),
    ("memories", "ALTER TABLE memories ADD COLUMN role_id VARCHAR(128) NOT NULL DEFAULT 'default'"),
    ("memories", "ALTER TABLE memories ADD COLUMN updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"),
    # ---------- memory_relations ----------
    ("memory_relations", "ALTER TABLE memory_relations ADD COLUMN user_id VARCHAR(128) NOT NULL DEFAULT ''"),
    ("memory_relations", "ALTER TABLE memory_relations ADD COLUMN role_id VARCHAR(128) NOT NULL DEFAULT 'default'"),
    ("memory_relations", "ALTER TABLE memory_relations ADD COLUMN relation VARCHAR(16) NOT NULL DEFAULT ''"),
    ("memory_relations", "ALTER TABLE memory_relations ADD COLUMN weight FLOAT NOT NULL DEFAULT 1.0"),
    ("memory_relations", "ALTER TABLE memory_relations ADD INDEX idx_rel_user_source (user_id, source_id)"),
    ("memory_relations", "ALTER TABLE memory_relations ADD INDEX idx_rel_user_target (user_id, target_id)"),
)


async def ensure_schema(cur) -> None:
    """幂等创建全部表结构，并对历史表做列补齐。

    兼容 pymysql / aiomysql 两种游标（同步直接调用 execute，异步需 await）。
    注意：aiomysql 中 execute 和 fetchone 都返回协程（不是 Future）。
    """
    import inspect
    import asyncio

    async def _maybe_await(value):
        if inspect.iscoroutine(value) or asyncio.isfuture(value):
            return await value
        return value

    async def _execute(stmt: str, params: tuple | None = None):
        result = cur.execute(stmt, params or ())
        await _maybe_await(result)

    async def _fetchone():
        result = cur.fetchone()
        return await _maybe_await(result)

    # 1) 创建新表
    # CREATE TABLE IF NOT EXISTS 在表已存在时会生成 NOTE 级 Warning（被 pymysql/aiomysql
    # 通过 warnings 模块打到 stderr），属于预期行为。用 sql_notes=0 抑制。
    await _execute("SET SESSION sql_notes = 0")
    try:
        for stmt in DDL_STATEMENTS:
            await _execute(stmt)
    finally:
        await _execute("SET SESSION sql_notes = 1")

    # 2) ALTER 列补齐（用 information_schema 判断存在性）
    for table, alter_sql in ALTER_PROBES:
        # 提取列名（简单解析：ADD COLUMN xxx）
        col = _extract_column_name(alter_sql)
        if not col:
            continue
        check_sql = (
            "SELECT COUNT(*) AS c FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = %s AND column_name = %s"
        )
        await _execute(check_sql, (table, col))
        row = await _fetchone()
        if asyncio.isfuture(row) or inspect.iscoroutine(row):
            row = await _maybe_await(row)
        exists = bool(row and row[0] and int(row[0]) > 0)
        if not exists:
            try:
                await _execute(alter_sql)
            except Exception as e:
                # 列已存在等冲突直接忽略（Duplicate column 等是预期行为）
                log_exception(
                    logger,
                    "ALTER probe failed",
                    exc=e,
                    level=logging.WARNING,
                    include_traceback=False,
                    stage="schema",
                    event="alter_probe_error",
                    table=table,
                    column=col,
                )


def _extract_column_name(alter_sql: str) -> str | None:
    """从 `ALTER TABLE t ADD COLUMN col ...` 抽取列名。"""
    import re

    m = re.search(r"ADD\s+COLUMN\s+`?(\w+)`?", alter_sql, re.IGNORECASE)
    return m.group(1) if m else None