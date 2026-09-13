"""角色内核(Role Core)子包。

PRD 02~06 设计：每个用户每角色拥有独立的结构化人格、OCEAN 五维性格、三桶心情、
信念清单、三维关系值。本子包把这些能力的"加载/拼装"封装成统一的接口，
biz/chat.py 通过 `build_segments(user_id, role_id)` 拿到 7 段 system prompt
片段，再喂给 `config.prompts.build_system_prompt`。

设计原则（PRD 历史教训 842d7c5）：
1. **新表与旧表分离** — 不修改 personas 表的任何代码路径，只新增读 role_persona 表的接口，
   旧 personas 表的兼容由 load_persona_legacy 保留。
2. **铁律导向** — LLM 抽取只写 suggestions 表（在本子包外的 `memory/role_evolution.py` 实现），
   本子包只读不写主表。
3. **新表不存在时降级** — 若 role_persona / role_traits 表未建（早期部署），
   自动降级到 personas + DEFAULT_PERSONA，保证核心对话链路永不中断（PRD 5级回退）。
4. **轻 ContextVar 注入** — 通过 `bind_request` 注入 user_id / role_id，
   业务函数内部从 `current_context()` 取，避免参数穿透。
"""

from .presets import OCEAN_PRESETS, get_preset_traits, get_preset_name
from .persona import load_persona, render_persona_for_prompt, load_persona_legacy
from .traits import load_traits, render_traits_for_prompt
from .prompts_inject import build_segments, SEGMENT_BUDGET

__all__ = [
    "OCEAN_PRESETS",
    "get_preset_traits",
    "get_preset_name",
    "load_persona",
    "load_persona_legacy",
    "render_persona_for_prompt",
    "load_traits",
    "render_traits_for_prompt",
    "build_segments",
    "SEGMENT_BUDGET",
]