import asyncio
import os

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.core.message.components import Image
import requests
import traceback

from .draw import render_navigation_image
from .navigation import (
    CYNOSURAL_FIELD_BLOCKED_MESSAGE,
    format_navigation_plan,
    format_request_error,
    is_high_security_destination,
    parse_nav_command,
    parse_trinav_command,
    query_navigation_plan,
    query_triglavian_blackops_plan,
    resolve_system,
)

# ── Load HTML template (fallback for remote t2i) ──
_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "data", "t2i_templates", "navigation.html")
with open(_TEMPLATE_PATH, "r", encoding="utf-8") as _f:
    NAV_TEMPLATE = _f.read()

_RENDER_OPTIONS = {
    "full_page": True,
    "type": "png",
    "quality": 100,
}


@register("eve_navigator", "Baizi", "跨星系导航插件，基于EVE宇宙图谱计算最优跳跃路线", "2.3.0")
class EvePlugin(Star):
    def __init__(self, context: Context, config: dict = None):
        super().__init__(context, config or {})

    async def initialize(self):
        logger.info("[EVE] 导航插件已初始化")

    async def shutdown(self):
        logger.info("[EVE] 导航插件已卸载")

    async def _render_result(self, event: AstrMessageEvent, result):
        """Render navigation result: local PIL first, remote t2i fallback, plain text last."""
        # 1. Local PIL rendering
        try:
            image_bytes = await asyncio.to_thread(render_navigation_image, result)
            yield event.chain_result([Image.fromBytes(image_bytes)])
            return
        except Exception as local_exc:
            logger.warning(f"[EVE] 本地渲染失败: {local_exc}")

        # 2. Remote t2i rendering
        try:
            url = await self.html_render(NAV_TEMPLATE, {"data": result}, options=_RENDER_OPTIONS)
            yield event.image_result(url)
            return
        except Exception as remote_exc:
            logger.warning(f"[EVE] 远程渲染失败: {remote_exc}")

        # 3. Plain text fallback
        yield event.plain_result(f"[EVE] {format_navigation_plan(result)}")

    @filter.command("nav")
    async def cmd_nav(self, event: AstrMessageEvent):
        try:
            text = event.message_str
            parsed = parse_nav_command(text)

            if parsed is None:
                return

            if parsed == "usage":
                yield event.plain_result(
                    "[EVE] 用法: /nav <起点星系> <终点星系> [super|capital|none] [maxJumpLy]\n"
                    "示例: /nav 埃克瑟斯 出口 6 super"
                )
                return

            if parsed == "params":
                yield event.plain_result(
                    "[EVE] 参数错误: security 仅支持 super / capital / none，maxJumpLy 必须是正数"
                )
                return

            from_system, to_system, security, max_jump_ly = parsed

            # check start == end (after fuzzy resolution)
            from_rec = resolve_system(from_system, fuzzy=True)
            to_rec = resolve_system(to_system, fuzzy=True)
            from_label = from_rec["zh"] if from_rec else from_system
            to_label = to_rec["zh"] if to_rec else to_system
            if from_label.lower() == to_label.lower():
                yield event.plain_result("[EVE] 起点和终点不能相同")
                return

            if is_high_security_destination(to_system):
                yield event.plain_result(f"[EVE] {CYNOSURAL_FIELD_BLOCKED_MESSAGE}")
                return

            try:
                result = await asyncio.to_thread(query_navigation_plan, from_system, to_system, security, max_jump_ly)
                async for msg in self._render_result(event, result):
                    yield msg
            except requests.RequestException as exc:
                yield event.plain_result(f"[EVE] {format_request_error(exc)}")

        except Exception as e:
            logger.error(f"[EVE] nav 命令异常: {e}")
            traceback.print_exc()
            yield event.plain_result(f"[EVE] 系统错误: {e}")

    @filter.command("trinav")
    async def cmd_trinav(self, event: AstrMessageEvent):
        try:
            text = event.message_str
            parsed = parse_trinav_command(text)

            if parsed is None:
                return

            if parsed == "usage":
                yield event.plain_result(
                    "[EVE] 用法: /trinav <目的地星系>\n"
                    "示例: /trinav 出口"
                )
                return

            to_system = parsed

            if is_high_security_destination(to_system):
                yield event.plain_result(f"[EVE] {CYNOSURAL_FIELD_BLOCKED_MESSAGE}")
                return

            try:
                result = await asyncio.to_thread(query_triglavian_blackops_plan, to_system)
                async for msg in self._render_result(event, result):
                    yield msg
            except requests.RequestException as exc:
                yield event.plain_result(f"[EVE] {format_request_error(exc)}")

        except Exception as e:
            logger.error(f"[EVE] trinav 命令异常: {e}")
            traceback.print_exc()
            yield event.plain_result(f"[EVE] 系统错误: {e}")
