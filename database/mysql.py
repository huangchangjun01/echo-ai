from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.asyncio.engine import AsyncEngine

from config.config import get_settings
from database.models import Base

logger = logging.getLogger(__name__)

_engine_lock = threading.Lock()
_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """返回单例异步引擎。"""
    global _engine, _sessionmaker
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                settings = get_settings().mysql
                dsn = settings.dsn
                logger.info("Creating async engine: %s", dsn)
                _engine = create_async_engine(
                    dsn,
                    pool_size=5,
                    max_overflow=10,
                    pool_recycle=3600,
                    pool_pre_ping=True,
                    echo=False,
                )
                _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """异步上下文管理器，返回 AsyncSession，自动提交/回滚并关闭。"""
    engine = get_engine()
    if _sessionmaker is None:
        raise RuntimeError("sessionmaker not initialized")
    session = _sessionmaker()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def init_db() -> None:
    """创建所有表（幂等）。"""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created (if not exists)")


async def close_db() -> None:
    """关闭引擎，释放连接池。"""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _sessionmaker = None
        logger.info("Database engine disposed")