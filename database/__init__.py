"""database 包：MySQL 连接池 + 表结构管理。"""

from .mysql import acquire, close_pool, execute, fetch_all, fetch_one, get_pool, init_schema

__all__ = [
    "acquire",
    "close_pool",
    "execute",
    "fetch_all",
    "fetch_one",
    "get_pool",
    "init_schema",
]