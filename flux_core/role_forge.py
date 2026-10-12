"""Task-local role synthesis protocol for FLUX RoleForge."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


ROLE_PLAN_RE = re.compile(
    r"<FLUX_ROLE_PLAN>\s*(\{.*?\})\s*</FLUX_ROLE_PLAN>",
    re.DOTALL | re.IGNORECASE,
)
VALID_STATUSES = {"solved", "needs_team", "needs_meta_review"}


@dataclass(frozen=True)
class RoleBlueprint:
    name: str
    capability: str
    instruction: str
    evidence_goal: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class RoleForgePlan:
    valid: bool
    status: str
    checks: List[str]
    unresolved_risks: List[str]
    missing_capabilities: List[str]
    roles: List[RoleBlueprint]
    reason: str = ""

    @property
    def solo_ready(self) -> bool:
        return bool(
            self.valid
            and self.status == "solved"
            and self.checks
            and not self.unresolved_risks
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["solo_ready"] = self.solo_ready
        return payload


def role_forge_instruction(task_type: str) -> str:
    return (
        "You must emit exactly one <FLUX_ROLE_PLAN> JSON object. Omitting this "
        "envelope is a controller failure even if you know the answer. The JSON "
        "must use this schema: "
        "{\"status\":\"solved|needs_team|needs_meta_review\",\"checks\":[...],"
        "\"unresolved_risks\":[...],\"missing_capabilities\":[...],\"roles\":["
        "{\"name\":\"RoleName\",\"capability\":\"...\",\"instruction\":\"...\","
        "\"evidence_goal\":\"...\"}]}. "
        "with status (solved|needs_team|needs_meta_review), checks, unresolved_risks, "
        "missing_capabilities, and roles. Each role must contain name, capability, "
        "instruction, and evidence_goal. First solve and self-check the task internally. Use solved "
        "only when the required output contract is satisfied, at least one concrete check "
        "was performed, and no unresolved risk remains. Use needs_team when specific "
        "capabilities are missing. Use needs_meta_review when you cannot reliably assess "
        "difficulty or construct the role plan. Role definitions must be task-general "
        "capabilities and must not contain a proposed answer, option letter, hidden label, "
        "or benchmark-specific oracle information. Create no roles when status is solved. "
        f"The task contract category is {task_type}. Close with </FLUX_ROLE_PLAN>, then "
        "provide the normal final response in the required benchmark format."
    )


def role_architect_instruction(task_type: str) -> str:
    coverage = (
        "For reasoning tasks, the pool should normally cover four separable "
        "functions when the task is non-trivial: an independent domain solver, "
        "an evidence/checking role, an adversarial alternative or option "
        "elimination role, and a synthesis/adjudication role. If fewer roles "
        "are enough, explain that through the checks and unresolved risks, but "
        "do not collapse verification, challenge, and synthesis into the same "
        "role unless the task is genuinely simple. "
    )
    return (
        "You are FLUX's task architect, not a solver. Your only job is to design "
        "the task-local role pool that later graph edits may recruit from. Emit "
        "exactly one <FLUX_ROLE_PLAN> JSON object and no final benchmark answer. "
        "Your entire response must have this shape and nothing else: "
        "<FLUX_ROLE_PLAN>{\"status\":\"needs_team\",\"checks\":[\"...\"],"
        "\"unresolved_risks\":[\"...\"],\"missing_capabilities\":[\"...\"],"
        "\"roles\":[{\"name\":\"RoleName\",\"capability\":\"...\","
        "\"instruction\":\"...\",\"evidence_goal\":\"...\"}]}</FLUX_ROLE_PLAN>. "
        "Use status=needs_team and provide at least two valid complementary roles. Each role must "
        "contain name, capability, instruction, and evidence_goal. Roles must be "
        "specific to the capability gaps of this task, non-overlapping, useful for "
        "independent reasoning or verification, and free of answer leakage. "
        + coverage
        + "Do not "
        "include option letters, final answers, code solutions, gold labels, hidden "
        "tests, or benchmark oracle information in role definitions. If you cannot "
        "construct a valid non-empty role pool, use status=needs_meta_review with "
        "empty roles; the controller will mark the task as failed. "
        f"The task contract category is {task_type}."
    )


def _clean_text(value: Any, *, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _clean_name(value: Any) -> str:
    parts = re.findall(r"[A-Za-z][A-Za-z0-9]*", str(value or ""))
    return "".join(part[:1].upper() + part[1:] for part in parts)[:48]


def _strings(value: Any, *, limit: int = 8) -> List[str]:
    raw = value if isinstance(value, list) else ([value] if value else [])
    out: List[str] = []
    for item in raw:
        text = _clean_text(item, limit=240)
        if text and text.lower() not in {"none", "n/a", "na", "null", "no risks"} and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _contains_answer_payload(text: str) -> bool:
    return bool(
        re.search(
            r"(?:\bfinal[_ ]answer\s*[:=]\s*(?:[A-Z]|[-+]?\d+(?:\.\d+)?)\b|\bthe answer is\s+(?:[A-Z]|[-+]?\d+(?:\.\d+)?)\b|\bcorrect option\s*(?:is|=|:)\s*(?:[A-Z]|\d+)\b|\bgold label\b|\bhidden test\b|####\s*[-+]?\d)",
            str(text or ""),
            re.IGNORECASE,
        )
    )


def _escape_invalid_json_backslashes(text: str) -> str:
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", str(text or ""))


def _loads_role_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_escape_invalid_json_backslashes(text))


def _raw_decode_role_json(text: str) -> tuple[Any, int]:
    decoder = json.JSONDecoder()
    try:
        return decoder.raw_decode(text)
    except json.JSONDecodeError:
        return decoder.raw_decode(_escape_invalid_json_backslashes(text))


def parse_role_forge_plan(
    text: str,
    *,
    max_roles: int = 12,
    min_roles: int = 1,
) -> RoleForgePlan:
    match = ROLE_PLAN_RE.search(str(text or ""))
    raw_payload = None
    reason = ""
    if match:
        raw_payload = match.group(1)
    else:
        open_match = re.search(r"<FLUX_ROLE_PLAN>\s*", str(text or ""), re.IGNORECASE)
        if open_match:
            try:
                payload_obj, _ = _raw_decode_role_json(str(text or "")[open_match.end():].lstrip())
                raw_payload = json.dumps(payload_obj)
                reason = "missing closing role-plan tag"
            except json.JSONDecodeError:
                return RoleForgePlan(False, "needs_meta_review", [], [], [], [], "missing role-plan envelope")
        else:
            return RoleForgePlan(False, "needs_meta_review", [], [], [], [], "missing role-plan envelope")
    try:
        payload = _loads_role_json(raw_payload)
    except json.JSONDecodeError:
        return RoleForgePlan(False, "needs_meta_review", [], [], [], [], "invalid role-plan JSON")
    if not isinstance(payload, dict):
        return RoleForgePlan(False, "needs_meta_review", [], [], [], [], "role plan is not an object")
    status = _clean_text(payload.get("status"), limit=32).lower()
    if status not in VALID_STATUSES:
        return RoleForgePlan(False, "needs_meta_review", [], [], [], [], "invalid status")

    roles: List[RoleBlueprint] = []
    seen: set[str] = set()
    raw_roles = payload.get("roles") if isinstance(payload.get("roles"), list) else []
    for raw in raw_roles:
        if not isinstance(raw, dict):
            continue
        name = _clean_name(raw.get("name"))
        capability = _clean_text(raw.get("capability"), limit=120)
        instruction = _clean_text(raw.get("instruction"), limit=420)
        evidence_goal = _clean_text(raw.get("evidence_goal"), limit=220)
        key = capability.lower()
        if (
            not name
            or not capability
            or not instruction
            or not evidence_goal
            or key in seen
            or _contains_answer_payload(" ".join([capability, instruction, evidence_goal]))
        ):
            continue
        seen.add(key)
        roles.append(RoleBlueprint(name, capability, instruction, evidence_goal))
        if len(roles) >= max(1, int(max_roles)):
            break

    checks = _strings(payload.get("checks"))
    risks = _strings(payload.get("unresolved_risks"))
    missing = _strings(payload.get("missing_capabilities"))
    if status == "solved" and roles:
        roles = []
    required_roles = max(1, int(min_roles))
    if status == "needs_team" and len(roles) < required_roles:
        status = "needs_meta_review"
    valid = bool(status == "solved" or (status == "needs_team" and len(roles) >= required_roles))
    return RoleForgePlan(valid, status, checks, risks, missing, roles, reason)


def meta_role_blueprints() -> List[RoleBlueprint]:
    return [
        RoleBlueprint(
            "TaskDiagnostician",
            "task diagnosis",
            "Determine whether the task is independently solvable and identify missing capabilities without proposing an answer by imitation.",
            "A valid FLUX role plan with explicit checks, risks, and missing capabilities.",
        ),
        RoleBlueprint(
            "RoleArchitect",
            "role architecture",
            "Translate diagnosed capability gaps into a minimal non-overlapping role pool with evidence goals.",
            "A bounded role pool whose capabilities cover every unresolved risk.",
        ),
    ]
