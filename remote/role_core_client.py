"""echo-ai → echo-core 角色内核(role_core)HTTP 客户端。

M0 阶段提供 persona/traits/mood/relationship/suggestion 的最小远程调用。
完整 M3/M6/M7 接口在后续阶段补全。

设计要点:
- 复用既有 remote 模块的 httpx AsyncClient 模式
- 内部接口走 X-Internal-Token 鉴权
- 失败仅记日志,不抛错(PRD 5 级回退)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from utils.request_context import log_exception
from utils.request_context import current_context

logger = logging.getLogger(__name__)


ECHO_CORE_BASE_URL = os.getenv("ECHO_CORE_BASE_URL", "http://localhost:8080")
ECHO_CORE_INTERNAL_TOKEN = os.getenv("ECHO_CORE_INTERNAL_TOKEN", "")


class RoleCoreClient:
    """调用 echo-core /api/role-core/* 的异步 HTTP 客户端。

    注入上下文从 utils.request_context.current_context() 读取 rid/uid,
    透传到 echo-core 后由其日志链路统一显示。
    """

    def __init__(self, base_url: str | None = None, internal_token: str | None = None) -> None:
        self.base_url = (base_url or ECHO_CORE_BASE_URL).rstrip("/")
        self.internal_token = internal_token or ECHO_CORE_INTERNAL_TOKEN

    def _headers(self) -> dict[str, str]:
        ctx = current_context()
        h = {"Content-Type": "application/json"}
        if ctx.get("request_id"):
            h["X-Request-Id"] = ctx["request_id"]
        if self.internal_token:
            h["X-Internal-Token"] = self.internal_token
        if ctx.get("user_id"):
            h["X-User-Id"] = ctx["user_id"]
        return h

    async def _post(self, path: str, payload: dict) -> dict | None:
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload, headers=self._headers())
            if resp.status_code >= 400:
                log_exception(
                    logger,
                    f"role_core_client POST {path} failed: status={resp.status_code} body={resp.text[:200]}",
                    exc=None,
                    level=logging.WARNING,
                    include_traceback=False,
                    stage="role_core_client",
                    event="http_error",
                    status=resp.status_code,
                    path=path,
                )
                return None
            data = resp.json()
            if data.get("code") != 200:
                log_exception(
                    logger,
                    f"role_core_client POST {path} business error: {data.get('message')}",
                    exc=None,
                    level=logging.WARNING,
                    include_traceback=False,
                    stage="role_core_client",
                    event="biz_error",
                    path=path,
                )
                return None
            return data.get("data")
        except Exception as e:
            log_exception(
                logger,
                f"role_core_client POST {path} exception",
                exc=e,
                level=logging.WARNING,
                include_traceback=False,
                stage="role_core_client",
                event="exception",
                path=path,
            )
            return None

    async def _get(self, path: str, params: dict | None = None) -> dict | None:
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params=params or {}, headers=self._headers())
            if resp.status_code >= 400:
                log_exception(
                    logger,
                    f"role_core_client GET {path} failed: status={resp.status_code}",
                    exc=None,
                    level=logging.WARNING,
                    include_traceback=False,
                    stage="role_core_client",
                    event="http_error",
                    status=resp.status_code,
                    path=path,
                )
                return None
            data = resp.json()
            if data.get("code") != 200:
                log_exception(
                    logger,
                    f"role_core_client GET {path} business error: {data.get('message')}",
                    exc=None,
                    level=logging.WARNING,
                    include_traceback=False,
                    stage="role_core_client",
                    event="biz_error",
                    path=path,
                )
                return None
            return data.get("data")
        except Exception as e:
            log_exception(
                logger,
                f"role_core_client GET {path} exception",
                exc=e,
                level=logging.WARNING,
                include_traceback=False,
                stage="role_core_client",
                event="exception",
                path=path,
            )
            return None

    # ===== Mood(M2) =====

    async def report_mood_event(
        self,
        user_id: str,
        role_id: str,
        event_impact: float,
        event_intensity: float,
        emotion: str,
        trigger_event: str = "dialogue",
    ) -> bool:
        """上报 mood 事件(M2 阶段实现,M0 先打桩)。"""
        # echo-core UpdateMoodRequest.UserID 是 uint,字符串无法直接 unmarshal
        # 这里尝试把 user_id / role_id 转 int(失败则保持原值)
        try:
            uid_int = int(user_id)
        except (TypeError, ValueError):
            uid_int = user_id
        try:
            rid_int = int(role_id)
        except (TypeError, ValueError):
            rid_int = role_id
        payload = {
            "userId": uid_int,
            "roleId": rid_int,
            "eventImpact": event_impact,
            "eventIntensity": event_intensity,
            "emotion": emotion,
            "triggerEvent": trigger_event,
        }
        data = await self._post("/api/role-core/mood/event", payload)
        return data is not None

    # ===== Relationship(M3) =====

    async def report_relationship_event(
        self,
        user_id: str,
        role_id: str,
        event_type: str,
        source: str = "dialogue",
    ) -> bool:
        """上报关系事件(M3 阶段实现,M0 先打桩)。"""
        payload = {
            "roleId": role_id,
            "eventType": event_type,
            "source": source,
        }
        data = await self._post("/api/role-core/relationship/event", payload)
        return data is not None

    # ===== Suggestion(M6) =====

    async def create_suggestion(
        self,
        user_id: str,
        role_id: str,
        target_type: str,
        suggestion_json: dict,
        source: str,
        confidence: float,
        reason: str = "",
    ) -> dict | None:
        """LLM 抽取后回写建议到 echo-core。"""
        try:
            uid_int = int(user_id)
        except (TypeError, ValueError):
            uid_int = user_id
        try:
            rid_int = int(role_id)
        except (TypeError, ValueError):
            rid_int = role_id
        payload = {
            "userId": uid_int,
            "roleId": rid_int,
            "targetType": target_type,
            "suggestionJson": json.dumps(suggestion_json, ensure_ascii=False),
            "source": source,
            "confidence": confidence,
            "reason": reason,
        }
        return await self._post("/api/role-core/suggestions", payload)


# 默认单例
_default_client: RoleCoreClient | None = None


def get_role_core_client() -> RoleCoreClient:
    global _default_client
    if _default_client is None:
        _default_client = RoleCoreClient()
    return _default_client