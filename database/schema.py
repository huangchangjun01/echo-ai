"""MySQL 表结构：记忆分层（L0/L1/L2）、记忆关系、人格、核心记忆、抽取日志。

所有 DDL 在启动时通过 `ensure_schema` 幂等创建，并对历史表缺失列做 ALTER 补齐。
设计原则：单用户、单服务可水平扩展为按 user_id 分表；这里为了简化保持单库多表。

支持两种游标：
- 异步 (aiomysql.Cursor)：`await ensure_schema(cur)`
- 同步 (pymysql.Cursor)：直接调用 `ensure_schema(cur)`
通过 inspect 动态判断，避免双份代码。
"""

from __future__ import annotations

from .schema_sql import ensure_schema as _ensure_schema_impl


async def ensure_schema(cur) -> None:
    """幂等创建全部表结构，并对历史表做列补齐。"""
    await _ensure_schema_impl(cur)