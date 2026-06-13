import json
import logging
import math
import os
import re

import requests

logger = logging.getLogger("EVE_Navigation")

# ============================================================
# config
# ============================================================


def normalize_nav_api_base_url(value):
    value = str(value or "").strip()
    if value.startswith(("http://", "https://")):
        return value
    return f"https://{value}"


NAV_API_BASE_URL = normalize_nav_api_base_url(os.getenv("NAV_API_BASE_URL", "aio.dusy.run"))
NAV_API_TOKEN = os.getenv(
    "NAV_API_TOKEN",
    "5557ebae3d04bc72145c6b56d44e50051694d2a0645688bda2d1fa417daee873",
)
NAV_DEFAULT_MAX_JUMP_LY = os.getenv("NAV_MAX_JUMP_LY", "6")
NAV_SECURITY_MODES = {"super", "capital", "none"}
NAV_SECURITY_ALIASES = {
    "s": "super", "super": "super", "sup": "super",
    "safe": "super", "safest": "super", "safety": "super",
    "安全": "super", "高安": "super", "超旗": "super",
    "超级": "super", "超级旗舰": "super",
    "c": "capital", "capital": "capital", "cap": "capital",
    "caps": "capital", "cipital": "capital", "captial": "capital",
    "旗舰": "capital", "大航": "capital", "资本": "capital",
    "n": "none", "none": "none", "no": "none",
    "null": "none", "any": "none", "all": "none",
    "unsafe": "none", "无": "none", "不限": "none", "随意": "none",
}
TRIGLAVIAN_CONSTELLATION_LABELS = {
    "Krai Perun": "雷电",
    "Krai Svarog": "熔火",
    "Krai Veles": "暗泽",
}
CYNOSURAL_FIELD_BLOCKED_MESSAGE = "你确定这里能开诱导？"

# ============================================================
# local system resolver (with vector-based fuzzy matching)
# ============================================================

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _normalize_system_query(value):
    return re.sub(r"\s+", "", str(value or "")).lower()


def _ngrams(value):
    text = _normalize_system_query(value)
    if not text:
        return []
    result = []
    for size in (1, 2, 3):
        if len(text) < size:
            continue
        result.extend(text[i:i + size] for i in range(len(text) - size + 1))
    return result


class SystemAliasResolver:
    def __init__(self):
        self.records = {}
        self.alias_to_ids = {}
        self.overrides = {}
        self.idf = {}
        self.vectors = {}
        self._load_aliases()
        self._load_overrides()
        self._load_vectors()

    def _load_aliases(self):
        path = os.path.join(_DATA_DIR, "system_aliases.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.records = {str(item["id"]): item for item in data.get("systems", [])}
            alias_to_ids = data.get("alias_to_ids", {})
            self.alias_to_ids = {
                _normalize_system_query(alias): [str(sid) for sid in ids]
                for alias, ids in alias_to_ids.items()
            }
        else:
            self._load_fallback_records()

    def _load_fallback_records(self):
        path = os.path.join(_DATA_DIR, "universe_map.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            universe_map = json.load(f)
        for node in universe_map.get("nodes", []):
            system_id = str(node.get("id", ""))
            if not system_id.startswith("3"):
                continue
            name = node.get("label", "")
            if not name:
                continue
            self.records[system_id] = {
                "id": system_id,
                "zh": name,
                "en": name,
                "aliases": [name],
                "sec": node.get("sec"),
            }
            self.alias_to_ids.setdefault(_normalize_system_query(name), []).append(system_id)

    def _load_overrides(self):
        path = os.path.join(_DATA_DIR, "system_alias_overrides.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.overrides = {
            _normalize_system_query(alias): str(system_id)
            for alias, system_id in data.get("aliases", {}).items()
        }

    def _load_vectors(self):
        path = os.path.join(_DATA_DIR, "system_alias_vectors.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.idf = {feature: float(weight) for feature, weight in data.get("idf", {}).items()}
        self.vectors = {
            str(system_id): {feature: float(weight) for feature, weight in vector.items()}
            for system_id, vector in data.get("vectors", {}).items()
        }

    def resolve(self, query, fuzzy=False, vector=True):
        normalized = _normalize_system_query(query)
        if not normalized:
            return None

        # 1. override aliases (common misspellings)
        override_id = self.overrides.get(normalized)
        if override_id:
            record = self.records.get(override_id)
            if record:
                return self._format_record(record)

        # 2. exact alias match
        exact_ids = self.alias_to_ids.get(normalized)
        if exact_ids:
            record = self.records.get(exact_ids[0])
            if record:
                return self._format_record(record)

        if fuzzy:
            # 3. prefix match on aliases
            prefix_match = self._alias_scan(normalized, startswith=True)
            if prefix_match:
                return prefix_match

            # 4. contains match on aliases
            contains_match = self._alias_scan(normalized, startswith=False)
            if contains_match:
                return contains_match

            # 5. TF-IDF vector similarity match
            if vector:
                vector_match = self._vector_match(normalized)
                if vector_match:
                    return vector_match

        return None

    def _format_record(self, record):
        return {
            "zh": record.get("zh", ""),
            "en": record.get("en", record.get("zh", "")),
            "sec": record.get("sec", 0),
        }

    def _alias_scan(self, query, startswith):
        for alias, system_ids in self.alias_to_ids.items():
            matched = alias.startswith(query) if startswith else query in alias
            if matched:
                record = self.records.get(system_ids[0])
                if record:
                    return self._format_record(record)
        return None

    def _vector_match(self, query):
        query_vector = self._query_vector(query)
        if not query_vector:
            return None

        best_system_id = None
        best_score = 0
        for system_id, vector in self.vectors.items():
            score = sum(query_vector.get(feature, 0) * weight for feature, weight in vector.items())
            if score > best_score:
                best_system_id = system_id
                best_score = score

        if best_system_id and best_score >= 0.35:
            record = self.records.get(best_system_id)
            if record:
                return self._format_record(record)
        return None

    def _query_vector(self, query):
        counts = {}
        for feature in _ngrams(query):
            if feature in self.idf:
                counts[feature] = counts.get(feature, 0) + 1
        if not counts:
            return {}

        vector = {feature: count * self.idf[feature] for feature, count in counts.items()}
        norm = math.sqrt(sum(weight * weight for weight in vector.values()))
        if norm == 0:
            return {}
        return {feature: weight / norm for feature, weight in vector.items()}


_resolver_instance = None


def _get_resolver():
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = SystemAliasResolver()
    return _resolver_instance


def resolve_system(name, fuzzy=False):
    """Resolve a system name to a record dict with zh/en/sec fields.

    Returns None if not found.
    """
    return _get_resolver().resolve(name, fuzzy=fuzzy)


# ============================================================
# API queries
# ============================================================


def navigation_name_candidates(raw_name, record):
    candidates = []
    if record:
        candidates.extend([record.get("zh"), record.get("en")])
    else:
        candidates.append(raw_name)

    result = []
    for candidate in candidates:
        if not candidate or candidate in result:
            continue
        result.append(candidate)
    return result


def parse_navigation_response(response):
    try:
        return response.json()
    except ValueError:
        return response.text


IMPORT_SUGGESTION_PATTERNS = [
    "你是不是想选：",
    "你是不是想选:",
    "你是不是要找：",
    "你是不是要找:",
    "Did you mean ",
    "did you mean ",
]


def _extract_response_text(data):
    """Extract a unified error/message text from any response type."""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        return f"{data.get('error', '')} {data.get('message', '')}"
    return ""


def is_missing_system_response(data):
    text = _extract_response_text(data)
    return "找不到星系" in text or "not found" in text.lower()


def extract_system_suggestion(data):
    """Extract a suggested system name from a missing-system response.

    Handles patterns like:
      - 找不到星系：0mv。你是不是想选：0MV-4W
      - Did you mean Jita?
    """
    text = _extract_response_text(data)
    for pattern in IMPORT_SUGGESTION_PATTERNS:
        idx = text.find(pattern)
        if idx == -1:
            continue
        after = text[idx + len(pattern):].strip()
        # take until end-of-sentence or whitespace
        suggestion = ""
        for ch in after:
            if ch in ("。", ".", "？", "?", "！", "!", " ", "\n", "\r"):
                break
            suggestion += ch
        suggestion = suggestion.strip().rstrip("。.?？!！")
        if suggestion and suggestion != data if isinstance(data, str) else suggestion:
            return suggestion
    return None


def _resolve_suggestion(suggestion):
    """Try local resolution on an API suggestion; returns candidates list."""
    record = resolve_system(suggestion, fuzzy=True)
    return navigation_name_candidates(suggestion, record)


def _try_navigation_candidates(from_system, to_system, from_record, to_record,
                               security, max_jump_ly, url, headers):
    """Try all name candidate combinations; retry with API suggestions on miss."""
    tried_from = set()
    tried_to = set()
    last_response = None
    last_error = None

    from_candidates = list(navigation_name_candidates(from_system, from_record))
    to_candidates = list(navigation_name_candidates(to_system, to_record))

    while True:
        made_progress = False
        for from_name in from_candidates:
            if from_name in tried_from:
                continue
            for to_name in to_candidates:
                if to_name in tried_to:
                    continue
                if from_name.lower() == to_name.lower():
                    continue
                params = {
                    "from": from_name,
                    "start": from_name,
                    "to": to_name,
                    "end": to_name,
                    "maxJumpLy": max_jump_ly,
                    "security": security,
                }
                try:
                    response = requests.get(url, headers=headers, params=params, timeout=30)
                    last_response = parse_navigation_response(response)
                    response.raise_for_status()
                except requests.HTTPError as exc:
                    last_error = exc
                    if last_response is not None:
                        if not is_missing_system_response(last_response):
                            return last_response, None
                        suggestion = extract_system_suggestion(last_response)
                        if suggestion:
                            resolved = _resolve_suggestion(suggestion)
                            for name in resolved:
                                if name not in tried_from and name not in tried_to:
                                    from_candidates.append(name)
                                    to_candidates.append(name)
                            if resolved:
                                tried_from.add(from_name)
                                tried_to.add(to_name)
                                made_progress = True
                                break
                    tried_from.add(from_name)
                    tried_to.add(to_name)
                    continue

                if not is_missing_system_response(last_response):
                    return last_response, None

                tried_from.add(from_name)
                tried_to.add(to_name)

                suggestion = extract_system_suggestion(last_response)
                if suggestion:
                    resolved = _resolve_suggestion(suggestion)
                    for name in resolved:
                        if name not in tried_from and name not in tried_to:
                            from_candidates.append(name)
                            to_candidates.append(name)
                    if resolved:
                        made_progress = True
                        break

            if made_progress:
                break

        if not made_progress:
            break

    return last_response, last_error


def query_navigation_plan(from_system, to_system, security="super", max_jump_ly=None):
    from_record = resolve_system(from_system, fuzzy=True)
    to_record = resolve_system(to_system, fuzzy=True)
    url = f"{NAV_API_BASE_URL.rstrip('/')}/api/plugins/navigation/api/plan"
    headers = {"Authorization": f"Bearer {NAV_API_TOKEN}"}
    max_jump_ly = max_jump_ly or NAV_DEFAULT_MAX_JUMP_LY

    last_response, last_error = _try_navigation_candidates(
        from_system, to_system, from_record, to_record,
        security, max_jump_ly, url, headers,
    )

    if last_response is not None and not is_missing_system_response(last_response):
        return last_response
    if last_response is not None:
        return last_response
    if last_error is not None:
        raise last_error
    return "导航服务未返回结果"


def query_triglavian_blackops_plan(to_system):
    to_record = resolve_system(to_system, fuzzy=True)
    url = f"{NAV_API_BASE_URL.rstrip('/')}/api/plugins/navigation/api/plan/triglavian-blackops"
    headers = {"Authorization": f"Bearer {NAV_API_TOKEN}"}

    tried_to = set()
    last_response = None
    last_error = None

    to_candidates = list(navigation_name_candidates(to_system, to_record))

    while True:
        made_progress = False
        for to_name in to_candidates:
            if to_name in tried_to:
                continue
            params = {"to": to_name}
            try:
                response = requests.get(url, headers=headers, params=params, timeout=30)
                last_response = parse_navigation_response(response)
                response.raise_for_status()
            except requests.HTTPError as exc:
                last_error = exc
                if last_response is not None:
                    if not is_missing_system_response(last_response):
                        return last_response
                    suggestion = extract_system_suggestion(last_response)
                    if suggestion:
                        resolved = _resolve_suggestion(suggestion)
                        for name in resolved:
                            if name not in tried_to:
                                to_candidates.append(name)
                        if resolved:
                            tried_to.add(to_name)
                            made_progress = True
                            break
                tried_to.add(to_name)
                continue

            if not is_missing_system_response(last_response):
                return last_response

            tried_to.add(to_name)
            suggestion = extract_system_suggestion(last_response)
            if suggestion:
                resolved = _resolve_suggestion(suggestion)
                for name in resolved:
                    if name not in tried_to:
                        to_candidates.append(name)
                if resolved:
                    made_progress = True
                    break

        if not made_progress:
            break

    if last_response is not None and not is_missing_system_response(last_response):
        return last_response
    if last_response is not None:
        return last_response
    if last_error is not None:
        raise last_error
    return "导航服务未返回结果"


def is_high_security_destination(system_name):
    record = resolve_system(system_name, fuzzy=True)
    if not record:
        return False
    try:
        return float(record.get("sec", 0)) > 0.5
    except (TypeError, ValueError):
        return False


# ============================================================
# formatting
# ============================================================


def format_navigation_plan(data):
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        return "\n".join(str(item) for item in data)
    if not isinstance(data, dict):
        return str(data)

    message = format_navigation_message(data)
    if message:
        return message

    route = data.get("route")
    if isinstance(route, list) and all(isinstance(item, dict) for item in route):
        if data.get("mode") == "triglavian_black_ops":
            return format_triglavian_navigation_plan(data)
        return format_standard_navigation_plan(data)

    if data.get("mode") == "triglavian_black_ops":
        return format_triglavian_navigation_plan(data)

    for key in ("route", "systems", "path", "plan"):
        item = data.get(key)
        if isinstance(item, list):
            return " -> ".join(str(value) for value in item)
        if isinstance(item, str):
            return item

    return json.dumps(data, ensure_ascii=False, indent=2)


def format_navigation_message(data):
    for key in ("message", "msg", "detail"):
        if key in data and isinstance(data[key], str):
            return data[key]

    if "error" in data and isinstance(data["error"], str):
        output = data["error"]
        if "permission" in data:
            output += f": {data['permission']}"
        return output

    return ""


def format_standard_navigation_plan(data):
    lines = [
        f"导航路线: {format_system_endpoint(data.get('start'))} -> {format_system_endpoint(data.get('end'))}",
    ]

    params = []
    if data.get("shipClass"):
        params.append(f"模式 {data['shipClass']}")
    if data.get("safetyStandardLabel"):
        params.append(f"标准 {data['safetyStandardLabel']}")
    if data.get("maxJumpLy") is not None:
        params.append(f"最大跳距 {format_ly(data['maxJumpLy'])}")
    if isinstance(data.get("safeNav"), bool):
        params.append(f"安全导航 {'是' if data['safeNav'] else '否'}")
    if params:
        lines.append("参数: " + " / ".join(params))

    overview = format_navigation_overview(data)
    if overview:
        lines.append("概览: " + "，".join(overview))

    safety = format_navigation_safety(data)
    if safety:
        lines.append("安全: " + "，".join(safety))

    route_lines = format_route_lines(data.get("route", []))
    if route_lines:
        lines.append("路线:")
        lines.extend(route_lines)

    lines.extend(format_navigation_notices(data))
    return "\n".join(lines)


def format_triglavian_navigation_plan(data):
    lines = [
        f"三神裔黑隐导航: {format_system_endpoint(data.get('start'))} -> {format_system_endpoint(data.get('end'))}",
    ]

    start = format_system_endpoint(data.get("start"))
    constellation = format_start_triglavian_constellation(data)
    start_label = "自动起点" if data.get("autoSelectedStart") else "起点"
    if constellation:
        lines.append(f"{start_label}: {start}（三神裔星座 {constellation}）")
    else:
        lines.append(f"{start_label}: {start}")

    params = []
    if data.get("maxJumpLy") is not None:
        params.append(f"最大跳距 {format_ly(data['maxJumpLy'])}")
    if data.get("candidateStartCount") is not None:
        params.append(f"候选三神裔起点 {data['candidateStartCount']} 个")
    if params:
        lines.append("参数: " + " / ".join(params))

    overview = format_navigation_overview(data, triglavian=True)
    if overview:
        lines.append("概览: " + "，".join(overview))

    route_lines = format_route_lines(data.get("route", []), show_triglavian=True)
    if route_lines:
        lines.append("路线:")
        lines.extend(route_lines)

    lines.extend(format_navigation_notices(data))
    return "\n".join(lines)


def format_navigation_overview(data, triglavian=False):
    parts = []
    if triglavian and data.get("blackOpsJumps") is not None:
        parts.append(f"黑隐跳跃 {data['blackOpsJumps']} 次")
    elif data.get("jumps") is not None:
        parts.append(f"跳跃 {data['jumps']} 次")

    if data.get("stargateSteps") is not None:
        parts.append(f"星门 {data['stargateSteps']} 段")
    if data.get("jumpBridgeSteps") is not None:
        parts.append(f"跳桥 {data['jumpBridgeSteps']} 段")
    if data.get("totalDistanceLy") is not None:
        parts.append(f"跳跃距离 {format_ly(data['totalDistanceLy'])}")
    if data.get("totalTravelDistanceLy") is not None:
        parts.append(f"总旅行 {format_ly(data['totalTravelDistanceLy'])}")
    if data.get("directDistanceLy") is not None:
        parts.append(f"直线 {format_ly(data['directDistanceLy'])}")
    return parts


def format_navigation_safety(data):
    parts = []
    if isinstance(data.get("safetySatisfied"), bool):
        parts.append("已满足安全标准" if data["safetySatisfied"] else "未满足安全标准")
    if data.get("fallbackApplied"):
        parts.append("已使用回退路线")

    stops = []
    if data.get("preferredStops") is not None:
        stops.append(f"优选落点 {data['preferredStops']}")
    if data.get("secondaryStops") is not None:
        stops.append(f"次级落点 {data['secondaryStops']}")
    if data.get("unsafeStops") is not None:
        stops.append(f"不安全落点 {data['unsafeStops']}")
    if data.get("unqualifiedStops") is not None:
        stops.append(f"未达标落点 {data['unqualifiedStops']}")
    if stops:
        parts.append(" / ".join(stops))
    return parts


def format_navigation_notices(data):
    lines = []
    seen = set()
    for key in ("fallbackMessage", "superRouteWarning"):
        text = data.get(key)
        if isinstance(text, str) and text and text not in seen:
            lines.append(f"提示: {text}")
            seen.add(text)

    warnings = data.get("esiWarnings")
    if isinstance(warnings, list):
        warning_text = "；".join(str(item) for item in warnings if item)
    elif isinstance(warnings, str):
        warning_text = warnings
    else:
        warning_text = ""
    if warning_text:
        lines.append(f"ESI 警告: {warning_text}")
    return lines


def format_route_lines(route, show_triglavian=False):
    if not isinstance(route, list):
        return []

    lines = []
    for index, node in enumerate(route, start=1):
        if not isinstance(node, dict):
            lines.append(f"{index}. {node}")
            continue

        details = [format_travel_mode(node)]
        distance = format_step_distance(node)
        if distance:
            details.append(distance)

        safety = format_step_safety(node)
        if safety:
            details.append(f"落点 {safety}")

        if show_triglavian:
            constellation = format_node_triglavian_constellation(node)
            if constellation:
                details.append(f"三神裔星座 {constellation}")

        bridge_name = node.get("jumpBridgeStructureName")
        if bridge_name:
            details.append(f"跳桥 {bridge_name}")

        line = f"{index}. {format_system_endpoint(node)}{format_system_meta(node)} - {'；'.join(details)}"
        lines.append(line)
    return lines


def format_system_endpoint(value):
    if isinstance(value, dict):
        for key in ("label", "name", "zh", "en", "id"):
            item = value.get(key)
            if item:
                return str(item)
        return "-"
    if value:
        return str(value)
    return "-"


def format_system_meta(node):
    meta = []
    if node.get("regionName"):
        meta.append(node["regionName"])
    if node.get("sec") is not None:
        meta.append(f"安等 {format_decimal(node['sec'], 3)}")
    return f"（{'，'.join(meta)}）" if meta else ""


def format_travel_mode(node):
    return node.get("travelModeLabel") or {
        "start": "起点",
        "jump": "跳跃",
        "stargate": "星门",
        "jump_bridge": "跳桥",
    }.get(node.get("travelMode"), "途经")


def format_step_distance(node):
    value = node.get("jumpLy")
    if not is_positive_value(value):
        value = node.get("legDistanceLy")
    if not is_positive_value(value):
        return ""

    mode = format_travel_mode(node)
    label = "跳距" if "跳跃" in mode else "距离"
    return f"{label} {format_ly(value)}"


def format_step_safety(node):
    safety = node.get("safety")
    if isinstance(safety, dict):
        if safety.get("label"):
            return str(safety["label"])
        if safety.get("preferred"):
            return "优选"
        if safety.get("secondary"):
            return "次级"

    infrastructure = node.get("infrastructure")
    if not isinstance(infrastructure, dict):
        return ""

    labels = []
    if infrastructure.get("fixedCynoBeacon"):
        labels.append("固定诱导")
    if infrastructure.get("keepstar"):
        labels.append("星城")
    if infrastructure.get("fortizar"):
        labels.append("堡垒")
    if infrastructure.get("sotiyo"):
        labels.append("索迪约")
    if infrastructure.get("npcStation"):
        count = infrastructure.get("npcStationCount")
        labels.append(f"NPC站x{count}" if count and count > 1 else "NPC站")
    if infrastructure.get("genericDockable"):
        labels.append("可停靠建筑")
    return "、".join(dedupe(labels))


def format_start_triglavian_constellation(data):
    constellation = data.get("startConstellation")
    if isinstance(constellation, dict):
        output = format_triglavian_constellation(
            constellation.get("name"),
            constellation.get("label"),
        )
        if output:
            return output

    route = data.get("route")
    if isinstance(route, list):
        for node in route:
            if isinstance(node, dict) and node.get("isTriglavianSystem"):
                output = format_node_triglavian_constellation(node)
                if output:
                    return output
    return ""


def format_node_triglavian_constellation(node):
    return format_triglavian_constellation(node.get("triglavianConstellationName"))


def format_triglavian_constellation(name, label=None):
    if name:
        label = label or TRIGLAVIAN_CONSTELLATION_LABELS.get(name)
        if label and label != name:
            return f"{name} / {label}"
        return str(name)
    if label:
        return str(label)
    return ""


def format_ly(value):
    return f"{format_decimal(value, 2)} ly"


def format_decimal(value, digits):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    text = f"{number:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def is_positive_value(value):
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def dedupe(values):
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


# ============================================================
# command parsing
# ============================================================


def is_positive_number(value):
    try:
        return float(value) > 0
    except ValueError:
        return False


def normalize_security_mode(value):
    return NAV_SECURITY_ALIASES.get(str(value).strip().lower())


def parse_nav_command(text):
    parts = text.strip().split()
    if not parts or parts[0] not in ("nav", "导航"):
        return None

    if len(parts) not in (3, 4, 5):
        return "usage"

    security = "super"
    max_jump_ly = NAV_DEFAULT_MAX_JUMP_LY
    for item in parts[3:]:
        security_mode = normalize_security_mode(item)
        if security_mode:
            security = security_mode
        elif is_positive_number(item):
            max_jump_ly = item
        else:
            return "params"

    return parts[1], parts[2], security, max_jump_ly


def parse_trinav_command(text):
    parts = text.strip().split()
    if not parts or parts[0] not in ("trinav", "三角导航"):
        return None
    if len(parts) != 2:
        return "usage"
    return parts[1]


def format_request_error(exc):
    if isinstance(exc, requests.Timeout):
        return "导航服务请求超时，请稍后再试"
    if isinstance(exc, requests.HTTPError):
        status_code = exc.response.status_code if exc.response is not None else "未知"
        body = exc.response.text if exc.response is not None else ""
        output = f"导航服务返回错误: HTTP {status_code}"
        if body:
            output += f"\n{body[:500]}"
        return output
    return f"导航服务请求失败: {exc}"
