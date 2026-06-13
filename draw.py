"""PIL-based local renderer for EVE navigation results.

Generates a dark-themed navigation image without depending on any remote t2i service.
"""

import io
import os
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from .navigation import (
    TRIGLAVIAN_CONSTELLATION_LABELS,
    format_decimal,
    format_ly,
    format_navigation_plan,
    format_system_endpoint,
    format_travel_mode,
    is_positive_value,
)

# ── Font ──
_FONT_DIR = os.path.dirname(__file__)
_FONT_CANDIDATES = [
    os.path.join(_FONT_DIR, "DouyinSansBold.otf"),
    os.path.join(_FONT_DIR, "data", "font.ttf"),
]


def _find_font() -> str:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return ""  # fallback to PIL default


_FONT_PATH = _find_font()


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    if _FONT_PATH:
        try:
            return ImageFont.truetype(_FONT_PATH, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ── Theme colors (dark EVE style) ──
C_BG = (10, 14, 20)
C_CARD_BG = (20, 26, 36)
C_BORDER = (30, 42, 58)
C_ACCENT = (93, 184, 232)
C_ACCENT_DIM = (56, 128, 176)
C_TEXT = (204, 218, 232)
C_TEXT_DIM = (136, 153, 176)
C_TEXT_BRIGHT = (238, 242, 248)
C_DANGER = (232, 96, 110)
C_WARNING = (232, 188, 80)
C_SUCCESS = (79, 200, 152)
C_SEC_HIGH = (72, 184, 132)
C_SEC_LOW = (229, 184, 76)
C_SEC_NULL = (224, 85, 106)
C_TRIGLAVIAN = (93, 184, 232)

# ── Layout constants ──
IMG_WIDTH = 560
PADDING = 28
CARD_RADIUS = 10
CARD_PAD = 14
LINE_SPACING = 6

# ── Fonts (lazy loaded) ──
_fonts = {}


def _font(name: str) -> ImageFont.FreeTypeFont:
    sizes = {
        "title": 22,
        "subtitle": 14,
        "route": 20,
        "tag": 12,
        "stat_value": 26,
        "stat_label": 11,
        "step_name": 16,
        "step_meta": 12,
        "step_idx": 11,
        "notice": 13,
        "fallback": 14,
    }
    if name not in _fonts:
        _fonts[name] = _load_font(sizes.get(name, 14))
    return _fonts[name]


# ── Drawing helpers ──


def _draw_rounded_rect(draw: ImageDraw.ImageDraw, xy, radius, fill=None, outline=None, width=1):
    x1, y1, x2, y2 = xy
    if x1 >= x2 or y1 >= y2:
        return
    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if fill:
        draw.rectangle((x1 + radius, y1, x2 - radius, y2), fill=fill)
        draw.rectangle((x1, y1 + radius, x2, y2 - radius), fill=fill)
        draw.pieslice((x1, y1, x1 + 2 * radius, y1 + 2 * radius), 180, 270, fill=fill)
        draw.pieslice((x2 - 2 * radius, y1, x2, y1 + 2 * radius), 270, 360, fill=fill)
        draw.pieslice((x1, y2 - 2 * radius, x1 + 2 * radius, y2), 90, 180, fill=fill)
        draw.pieslice((x2 - 2 * radius, y2 - 2 * radius, x2, y2), 0, 90, fill=fill)
    if outline and width > 0:
        draw.arc((x1, y1, x1 + 2 * radius, y1 + 2 * radius), 180, 270, fill=outline, width=width)
        draw.arc((x2 - 2 * radius, y1, x2, y1 + 2 * radius), 270, 360, fill=outline, width=width)
        draw.arc((x1, y2 - 2 * radius, x1 + 2 * radius, y2), 90, 180, fill=outline, width=width)
        draw.arc((x2 - 2 * radius, y2 - 2 * radius, x2, y2), 0, 90, fill=outline, width=width)
        draw.line([(x1 + radius, y1), (x2 - radius, y1)], fill=outline, width=width)
        draw.line([(x1 + radius, y2), (x2 - radius, y2)], fill=outline, width=width)
        draw.line([(x1, y1 + radius), (x1, y2 - radius)], fill=outline, width=width)
        draw.line([(x2, y1 + radius), (x2, y2 - radius)], fill=outline, width=width)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _sec_color(sec) -> Tuple[int, int, int]:
    try:
        v = float(sec)
    except (TypeError, ValueError):
        return C_TEXT_DIM
    if v >= 0.5:
        return C_SEC_HIGH
    if v > 0:
        return C_SEC_LOW
    return C_SEC_NULL


def _draw_tag(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, color: Tuple[int, int, int],
              bg_alpha: float = 0.10) -> int:
    """Draw a pill-shaped tag, return (x_end, y_end)."""
    font = _font("tag")
    tw, th = _text_size(draw, text, font)
    pad_x, pad_y = 10, 4
    w = tw + pad_x * 2
    h = th + pad_y * 2
    r = h // 2
    # bg
    bg = tuple(int(c * bg_alpha + C_BG[i] * (1 - bg_alpha)) for i, c in enumerate(color))
    _draw_rounded_rect(draw, (x, y, x + w, y + h), r, fill=bg, outline=tuple(int(c * 0.3 + C_BG[i] * 0.7) for i, c in enumerate(color)), width=1)
    draw.text((x + pad_x, y + pad_y), text, font=font, fill=color)
    return w


# ── Main renderer ──


def render_navigation_image(data: Any) -> bytes:
    """Render navigation data to PNG bytes using PIL."""
    if not isinstance(data, dict):
        return _render_fallback(data)

    route = data.get("route")
    if isinstance(route, list) and all(isinstance(item, dict) for item in route):
        return _render_structured(data)

    # Try message-style
    for key in ("message", "msg", "detail"):
        if key in data and isinstance(data[key], str):
            return _render_fallback(data[key])

    if "error" in data and isinstance(data["error"], str):
        text = data["error"]
        if "permission" in data:
            text += f": {data['permission']}"
        return _render_fallback(text)

    return _render_fallback(format_navigation_plan(data))


def _render_fallback(text: Any) -> bytes:
    """Render plain text as a simple image."""
    text = str(text)
    font = _font("fallback")
    # measure
    temp = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(temp)
    max_w = IMG_WIDTH - PADDING * 2
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        wrapped = textwrap.wrap(paragraph, width=40)
        lines.extend(wrapped if wrapped else [""])

    line_h = _text_size(draw, "Ay", font)[1] + LINE_SPACING
    total_h = PADDING * 2 + len(lines) * line_h + 40

    img = Image.new("RGB", (IMG_WIDTH, total_h), C_BG)
    draw = ImageDraw.Draw(img)

    # Header
    draw.text((PADDING, PADDING), "EVE 导航", font=_font("title"), fill=C_TEXT_BRIGHT)
    draw.text((PADDING, PADDING + 30), "查询结果", font=_font("subtitle"), fill=C_TEXT_DIM)

    y = PADDING + 60
    for line in lines:
        draw.text((PADDING, y), line, font=font, fill=C_TEXT)
        y += line_h

    # Footer
    draw.text((PADDING, total_h - 30), "EVE Navigator · AstrBot", font=_font("tag"), fill=C_TEXT_DIM)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _render_structured(data: dict) -> bytes:
    is_triglavian = data.get("mode") == "triglavian_black_ops"
    start = data.get("start")
    end = data.get("end")
    route = data.get("route", [])

    start_name = format_system_endpoint(start)
    end_name = format_system_endpoint(end)

    # ── Phase 1: measure all sections to compute total height ──
    temp = Image.new("RGB", (1, 1))
    tdraw = ImageDraw.Draw(temp)

    y = 0

    # Header area
    y += 60  # icon + title + subtitle

    # Route line
    y += 40

    # Triglavian info
    triglavian_tags_h = 0
    if is_triglavian:
        triglavian_tags_h = 30
        y += triglavian_tags_h

    # Tags row
    y += 32

    # Stats grid
    stat_cards = _collect_stats(data, is_triglavian)
    n_stats = len(stat_cards)
    if n_stats > 0:
        stat_rows = (n_stats + 2) // 3  # 3 per row
        y += stat_rows * 60 + 10

    # Safety tags
    safety_tags = _collect_safety_tags(data)
    if safety_tags:
        y += 30

    # Route steps
    y += 24  # section title
    step_heights = []
    for node in route:
        if not isinstance(node, dict):
            h = 30
        else:
            h = 70  # base
            meta_parts = _step_meta_parts(node, is_triglavian)
            if meta_parts:
                h += 18
        step_heights.append(h)
        y += h + 10

    # Notices
    notices = _collect_notices(data)
    y += len(notices) * 40

    # Footer
    y += 40

    total_h = y + PADDING * 2

    # ── Phase 2: draw ──
    img = Image.new("RGB", (IMG_WIDTH, total_h), C_BG)
    draw = ImageDraw.Draw(img)
    y = PADDING

    # Header
    icon = "[T]" if is_triglavian else "[N]"
    header_title = "三神裔黑隐导航" if is_triglavian else "EVE 星系导航"
    draw.text((PADDING, y), icon, font=_font("title"), fill=C_ACCENT)
    draw.text((PADDING + 36, y), header_title, font=_font("title"), fill=C_TEXT_BRIGHT)
    draw.text((PADDING + 36, y + 28), "基于实时宇宙图谱 · 最优跳跃路线", font=_font("subtitle"), fill=C_TEXT_DIM)
    y += 60

    # Route line
    route_text = f"{start_name}  →  {end_name}"
    draw.text((PADDING, y), route_text, font=_font("route"), fill=C_TEXT_BRIGHT)
    y += 40

    # Triglavian info tags
    if is_triglavian:
        x = PADDING
        constellation = _get_constellation_display(data, route)
        if data.get("autoSelectedStart"):
            w = _draw_tag(draw, x, y, "自动起点", C_ACCENT, 0.15)
            x += w + 8
        if constellation:
            w = _draw_tag(draw, x, y, f"三神裔星座 {constellation}", C_ACCENT, 0.15)
            x += w + 8
        if data.get("candidateStartCount") is not None:
            w = _draw_tag(draw, x, y, f"候选起点 {data['candidateStartCount']} 个", C_ACCENT, 0.10)
            x += w + 8
        y += triglavian_tags_h

    # Tags row
    x = PADDING
    if data.get("shipClass"):
        w = _draw_tag(draw, x, y, f"{data['shipClass']}", C_ACCENT)
        x += w + 8
    if data.get("safetyStandardLabel"):
        w = _draw_tag(draw, x, y, f"{data['safetyStandardLabel']}", C_ACCENT)
        x += w + 8
    if data.get("maxJumpLy"):
        w = _draw_tag(draw, x, y, f"最大 {data['maxJumpLy']} ly", C_ACCENT)
        x += w + 8
    if data.get("safetySatisfied") is True:
        w = _draw_tag(draw, x, y, "满足安全标准", C_SUCCESS, 0.12)
        x += w + 8
    elif data.get("safetySatisfied") is False:
        w = _draw_tag(draw, x, y, "未满足安全标准", C_DANGER, 0.12)
        x += w + 8
    if data.get("fallbackApplied"):
        w = _draw_tag(draw, x, y, "回退路线", C_WARNING, 0.12)
        x += w + 8
    y += 32

    # Stats grid
    if stat_cards:
        cols = 3
        card_w = (IMG_WIDTH - PADDING * 2 - 10 * (cols - 1)) // cols
        for i, (value, label) in enumerate(stat_cards):
            row = i // cols
            col = i % cols
            cx = PADDING + col * (card_w + 10)
            cy = y + row * 60
            _draw_rounded_rect(draw, (cx, cy, cx + card_w, cy + 50), CARD_RADIUS, fill=C_CARD_BG, outline=C_BORDER, width=1)
            vt = str(value)
            vw, _ = _text_size(draw, vt, _font("stat_value"))
            draw.text((cx + (card_w - vw) // 2, cy + 6), vt, font=_font("stat_value"), fill=C_TEXT_BRIGHT)
            lw, _ = _text_size(draw, label, _font("stat_label"))
            draw.text((cx + (card_w - lw) // 2, cy + 34), label, font=_font("stat_label"), fill=C_TEXT_DIM)
        stat_rows = (len(stat_cards) + cols - 1) // cols
        y += stat_rows * 60 + 10

    # Safety tags
    if safety_tags:
        x = PADDING
        for text, color in safety_tags:
            w = _draw_tag(draw, x, y, text, color, 0.12)
            x += w + 8
        y += 30

    # Route steps
    draw.text((PADDING, y), "跳 跃 路 线", font=_font("step_idx"), fill=C_TEXT_DIM)
    y += 24

    for idx, node in enumerate(route):
        is_first = idx == 0
        is_last = idx == len(route) - 1
        h = step_heights[idx]

        # Card background
        _draw_rounded_rect(draw, (PADDING, y, IMG_WIDTH - PADDING, y + h), CARD_RADIUS, fill=C_CARD_BG, outline=C_BORDER, width=1)

        # Timeline dot
        dot_x = PADDING - 14
        dot_y = y + 16
        dot_r = 5
        dot_color = C_SUCCESS if is_first else (C_ACCENT if is_last else C_ACCENT_DIM)
        draw.ellipse((dot_x - dot_r, dot_y - dot_r, dot_x + dot_r, dot_y + dot_r), fill=dot_color)

        if not isinstance(node, dict):
            draw.text((PADDING + CARD_PAD, y + CARD_PAD), str(node), font=_font("step_name"), fill=C_TEXT)
            y += h + 10
            continue

        label = format_system_endpoint(node)
        mode = format_travel_mode(node)

        # Step index + mode
        idx_text = f"{idx + 1} · {mode}"
        draw.text((PADDING + CARD_PAD, y + 8), idx_text, font=_font("step_idx"), fill=C_TEXT_DIM)

        # System name
        draw.text((PADDING + CARD_PAD, y + 24), label, font=_font("step_name"), fill=C_TEXT_BRIGHT)

        # Meta line
        meta_parts = _step_meta_parts(node, is_triglavian)
        if meta_parts:
            meta_x = PADDING + CARD_PAD
            meta_y = y + 48
            for part_text, part_color in meta_parts:
                draw.text((meta_x, meta_y), part_text, font=_font("step_meta"), fill=part_color)
                pw, _ = _text_size(draw, part_text, _font("step_meta"))
                meta_x += pw + 14

        y += h + 10

    # Notices
    for notice_text, is_danger in notices:
        color = C_DANGER if is_danger else C_WARNING
        bg = tuple(int(c * 0.06 + C_BG[i] * 0.94) for i, c in enumerate(color))
        _draw_rounded_rect(draw, (PADDING, y, IMG_WIDTH - PADDING, y + 34), CARD_RADIUS, fill=bg)
        # left border
        draw.rectangle((PADDING, y, PADDING + 4, y + 34), fill=color)
        draw.text((PADDING + 14, y + 8), notice_text, font=_font("notice"), fill=color)
        y += 40

    # Footer
    draw.text((PADDING, total_h - 30), "EVE Navigator · AstrBot", font=_font("tag"), fill=C_TEXT_DIM)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ── Data extraction helpers ──


def _collect_stats(data: dict, is_triglavian: bool) -> List[Tuple[str, str]]:
    cards = []
    if is_triglavian and data.get("blackOpsJumps") is not None:
        cards.append((str(data["blackOpsJumps"]), "黑隐跳跃"))
    elif data.get("jumps") is not None:
        cards.append((str(data["jumps"]), "跳跃次数"))
    if data.get("stargateSteps") is not None:
        cards.append((str(data["stargateSteps"]), "星门段数"))
    if data.get("jumpBridgeSteps") is not None:
        cards.append((str(data["jumpBridgeSteps"]), "跳桥段数"))
    if data.get("totalDistanceLy") is not None:
        try:
            cards.append((f"{float(data['totalDistanceLy']):.1f}", "跳跃距离(ly)"))
        except (TypeError, ValueError):
            pass
    if data.get("totalTravelDistanceLy") is not None:
        try:
            cards.append((f"{float(data['totalTravelDistanceLy']):.1f}", "总旅行(ly)"))
        except (TypeError, ValueError):
            pass
    if data.get("directDistanceLy") is not None:
        try:
            cards.append((f"{float(data['directDistanceLy']):.1f}", "直线距离(ly)"))
        except (TypeError, ValueError):
            pass
    return cards


def _collect_safety_tags(data: dict) -> List[Tuple[str, Tuple[int, int, int]]]:
    tags = []
    if data.get("preferredStops") is not None:
        tags.append((f"优选 {data['preferredStops']}", C_SUCCESS))
    if data.get("secondaryStops") is not None:
        tags.append((f"次级 {data['secondaryStops']}", C_WARNING))
    if data.get("unsafeStops") is not None:
        tags.append((f"不安全 {data['unsafeStops']}", C_DANGER))
    return tags


def _collect_notices(data: dict) -> List[Tuple[str, bool]]:
    notices = []
    for key in ("fallbackMessage", "superRouteWarning"):
        text = data.get(key)
        if isinstance(text, str) and text:
            notices.append((text, False))
    esi = data.get("esiWarnings")
    if isinstance(esi, list) and esi:
        notices.append(("；".join(str(i) for i in esi if i), True))
    elif isinstance(esi, str) and esi:
        notices.append((esi, True))
    return notices


def _step_meta_parts(node: dict, is_triglavian: bool) -> List[Tuple[str, Tuple[int, int, int]]]:
    parts = []
    sec = node.get("sec")
    if sec is not None:
        try:
            sec_val = float(sec)
            parts.append((f"安等 {sec_val:.1f}", _sec_color(sec_val)))
        except (TypeError, ValueError):
            pass
    region = node.get("regionName")
    if region:
        parts.append((region, C_TEXT_DIM))
    jump_ly = node.get("jumpLy") or node.get("legDistanceLy")
    if is_positive_value(jump_ly):
        try:
            parts.append((f"{float(jump_ly):.1f} ly", C_TEXT_DIM))
        except (TypeError, ValueError):
            pass
    safety = node.get("safety")
    if isinstance(safety, dict) and safety.get("label"):
        parts.append((f"落点 {safety['label']}", C_ACCENT))
    if node.get("isTriglavianSystem"):
        parts.append(("三神裔星系", C_TRIGLAVIAN))
    if node.get("triglavianConstellationName"):
        parts.append((node["triglavianConstellationName"], C_TRIGLAVIAN))
    bridge = node.get("jumpBridgeStructureName")
    if bridge:
        parts.append((f"跳桥: {bridge}", C_TEXT_DIM))
    return parts


def _get_constellation_display(data: dict, route: list) -> str:
    constellation = data.get("startConstellation")
    if isinstance(constellation, dict):
        name = constellation.get("name", "")
        label = constellation.get("label", "")
        if name:
            resolved_label = label or TRIGLAVIAN_CONSTELLATION_LABELS.get(name)
            if resolved_label and resolved_label != name:
                return f"{name} / {resolved_label}"
            return name
        if label:
            return label
    # Fallback: extract from route
    for node in route:
        if isinstance(node, dict) and node.get("isTriglavianSystem") and node.get("triglavianConstellationName"):
            return node["triglavianConstellationName"]
    return ""
