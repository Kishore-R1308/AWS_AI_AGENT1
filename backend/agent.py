import json
import re
from typing import TypedDict, Any

from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph

from backend.aws_tools import (
    get_ec2_instances,
    get_s3_buckets,
    get_s3_storage_summary,
    get_rds_instances,
    get_vpcs,
    get_subnets,
    get_internet_gateways,
    get_route_tables,
    get_security_groups,
    get_cost_summary,
    get_cost_by_service,
    get_patch_status,
    get_lambda_functions,
    get_cloudwatch_metrics,
    get_cloudtrail_events,
    get_inspector_findings,
    get_resource_tags,
    get_ec2_tags,
    get_s3_tags,
    get_lambda_tags,
)

from backend.config import GROQ_API_KEY, GROQ_MODEL
from backend.rag import retrieve_context


# ============================================================
# TOOL MAP
# ============================================================

TOOL_MAP = {
    "get_ec2_instances": get_ec2_instances,
    "get_s3_buckets": get_s3_buckets,
    "get_s3_storage_summary": get_s3_storage_summary,
    "get_rds_instances": get_rds_instances,
    "get_vpcs": get_vpcs,
    "get_subnets": get_subnets,
    "get_internet_gateways": get_internet_gateways,
    "get_route_tables": get_route_tables,
    "get_security_groups": get_security_groups,
    "get_cost_summary": get_cost_summary,
    "get_cost_by_service": get_cost_by_service,
    "get_patch_status": get_patch_status,
    "get_lambda_functions": get_lambda_functions,
    "get_cloudwatch_metrics": get_cloudwatch_metrics,
    "get_cloudtrail_events": get_cloudtrail_events,
    "get_inspector_findings": get_inspector_findings,
    "get_resource_tags": get_resource_tags,
    "get_ec2_tags": get_ec2_tags,
    "get_s3_tags": get_s3_tags,
    "get_lambda_tags": get_lambda_tags,
}


# ============================================================
# STATE
# ============================================================

class AgentState(TypedDict, total=False):
    session_id: str
    query: str
    conversation_history: str

    intent: str
    tools: list[str]
    service: str

    resolved_query: str
    resource_refs: list[str]

    context: str
    tool_result: str

    rca: dict[str, Any]
    recommendations: list[str]

    answer: str


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0,
)


# ============================================================
# TOKEN / CONTEXT CONTROL
# ============================================================

def _limit_text(
    text: str,
    max_chars: int,
) -> str:
    """
    Limit text passed to the LLM.

    Character limits are deliberately conservative because
    Groq enforces token-per-minute limits.
    """

    if not text:
        return ""

    text = str(text).strip()

    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars]
        + "\n\n[Additional context truncated for efficiency.]"
    )


def _trim_history(
    history: str,
    max_chars: int = 5000,
) -> str:
    """
    Keep recent conversation context.

    The most recent part is retained because it is generally
    the most useful for follow-up questions.
    """

    if not history:
        return ""

    history = history.strip()

    if len(history) <= max_chars:
        return history

    return (
        "[Earlier conversation omitted for context efficiency.]\n"
        + history[-max_chars:]
    )


# ============================================================
# HELPERS
# ============================================================

def _safe_text(value) -> str:
    if hasattr(value, "content"):
        return str(value.content)

    return str(value)


def _clean_json_response(content: str) -> str:
    content = content.strip()

    if content.startswith("```"):
        lines = content.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        content = "\n".join(lines).strip()

    return content


def _extract_json(content: str) -> dict:
    """
    Robustly extract the first JSON object from an LLM response.
    """

    content = _clean_json_response(content)

    try:
        parsed = json.loads(content)

        return (
            parsed
            if isinstance(parsed, dict)
            else {}
        )

    except Exception:
        pass

    match = re.search(
        r"\{.*\}",
        content,
        re.DOTALL,
    )

    if match:
        try:
            parsed = json.loads(
                match.group(0)
            )

            return (
                parsed
                if isinstance(parsed, dict)
                else {}
            )

        except Exception:
            pass

    return {}


def _normalize_intent(
    intent: str,
) -> str:

    if not intent:
        return "KNOWLEDGE"

    normalized = (
        str(intent)
        .strip()
        .upper()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if normalized in {
        "RCA",
        "ROOT_CAUSE",
        "ROOTCAUSE",
        "ROOT_CAUSE_ANALYSIS",
        "INCIDENT",
        "DIAGNOSIS",
        "TROUBLESHOOTING",
    }:
        return "RCA"

    if normalized in {
        "MONITORING",
        "MONITOR",
        "LIVE_DATA",
        "INVENTORY",
        "OBSERVABILITY",
    }:
        return "MONITORING"

    return "KNOWLEDGE"


def _normalize_tools(
    tools,
) -> list[str]:

    if not isinstance(
        tools,
        list,
    ):
        return []

    result = []

    for tool in tools:

        if not isinstance(
            tool,
            str,
        ):
            continue

        tool = tool.strip()

        if (
            tool in TOOL_MAP
            and tool not in result
        ):
            result.append(tool)

    return result


def _normalize_services(
    services,
) -> str:

    if not isinstance(
        services,
        list,
    ):
        return ""

    cleaned = []

    for service in services:

        value = str(
            service
        ).strip()

        if (
            value
            and value not in cleaned
        ):
            cleaned.append(value)

    return ", ".join(cleaned)


# ============================================================
# RESOURCE EXTRACTION
# ============================================================

def _extract_ec2_ids(
    text: str,
) -> set[str]:

    if not text:
        return set()

    return set(
        re.findall(
            r"\bi-[0-9a-fA-F]{8,17}\b",
            text,
        )
    )


def _extract_arns(
    text: str,
) -> set[str]:

    if not text:
        return set()

    return set(
        re.findall(
            r"\barn:[A-Za-z0-9:/_.+=,@-]+\b",
            text,
        )
    )


def _known_resource_ids(
    history: str,
    tool_result: str,
) -> set[str]:

    known = set()

    known.update(
        _extract_ec2_ids(
            history
        )
    )

    known.update(
        _extract_ec2_ids(
            tool_result
        )
    )

    known.update(
        _extract_arns(
            history
        )
    )

    known.update(
        _extract_arns(
            tool_result
        )
    )

    return known


def _ground_text(
    text: str,
    trusted_text: str,
) -> str:
    """
    Prevent fabricated EC2 IDs from appearing.

    Only resource IDs present in trusted conversation/AWS
    evidence are allowed through.
    """

    if not text:
        return ""

    known_ids = _known_resource_ids(
        trusted_text,
        trusted_text,
    )

    def replace_id(match):

        value = match.group(0)

        if value in known_ids:
            return value

        return "the affected EC2 instance"

    return re.sub(
        r"\bi-[0-9a-fA-F]{8,17}\b",
        replace_id,
        text,
    )


# ============================================================
# FOLLOW-UP DETECTION
# ============================================================

def _is_followup_query(
    query: str,
) -> bool:

    q = query.lower().strip()

    patterns = [
        r"\bit\b",
        r"\bits\b",
        r"\bthis\b",
        r"\bthat\b",
        r"\bthese\b",
        r"\bthose\b",
        r"\bthey\b",
        r"\bthem\b",
        r"\btheir\b",
        r"\bthe same\b",
        r"\bthe other\b",
        r"\bwhich one\b",
        r"\bwhich of them\b",
        r"\bwhat about\b",
        r"\bhow about\b",
        r"\bwhat should i do\b",
        r"\bwhat can i do\b",
        r"\bhow can i fix\b",
        r"\bhow do i fix\b",
        r"\bhow can i prevent\b",
        r"\bhow do i prevent\b",
        r"\bhow can this be prevented\b",
        r"\bwhy is this happening\b",
        r"\bwhy did this happen\b",
        r"\bwhy this\b",
    ]

    return any(
        re.search(
            pattern,
            q,
        )
        for pattern in patterns
    )


def _history_contains_incident(
    history: str,
) -> bool:

    if not history:
        return False

    h = history.lower()

    incident_terms = [
        "root cause",
        "rca",
        "incident",
        "problem",
        "issue",
        "abnormal",
        "unhealthy",
        "failed",
        "failure",
        "stopped",
        "high cpu",
        "high memory",
        "high utilization",
        "error",
        "crash",
        "timeout",
        "unavailable",
        "not working",
        "why is this happening",
        "why did this happen",
        "what should i do",
        "how can i fix",
    ]

    return any(
        term in h
        for term in incident_terms
    )


def _looks_like_prevention_or_remediation(
    query: str,
) -> bool:

    q = query.lower()

    terms = [
        "prevent",
        "avoid",
        "fix",
        "resolve",
        "remediate",
        "mitigate",
        "recover",
        "protect",
        "stop this",
        "happen again",
        "future",
        "recommend",
        "recommendation",
        "what should i do",
        "what can i do",
        "how should i",
    ]

    return any(
        term in q
        for term in terms
    )


def _looks_like_rca_query(
    query: str,
) -> bool:

    q = query.lower().strip()

    rca_patterns = [
        "why is",
        "why are",
        "why was",
        "why were",
        "why did",
        "why does",
        "why do",
        "root cause",
        "what caused",
        "cause of",
        "investigate",
        "troubleshoot",
        "diagnose",
        "what went wrong",
        "what is causing",
        "what might be causing",
        "why this",
        "why that",
    ]

    return any(
        pattern in q
        for pattern in rca_patterns
    )


# ============================================================
# ANSWER DETAIL DETECTION
# ============================================================

def _detail_requested(
    query: str,
) -> bool:
    """
    Detect when the user explicitly asks for a detailed answer.
    """

    q = query.lower().strip()

    detail_patterns = [
        "in detail",
        "detailed",
        "more detail",
        "explain deeply",
        "explain thoroughly",
        "thoroughly",
        "comprehensive",
        "deep dive",
        "step by step",
        "step-by-step",
        "full explanation",
        "complete explanation",
        "elaborate",
        "explain everything",
        "explain in depth",
        "in depth",
    ]

    return any(
        phrase in q
        for phrase in detail_patterns
    )


def _is_simple_definition_query(
    query: str,
) -> bool:
    """
    Identify short definition-style AWS questions.
    """

    q = query.lower().strip()

    definition_patterns = [
        r"^what is [a-z0-9 ._-]+[?]?$",
        r"^what are [a-z0-9 ._-]+[?]?$",
        r"^define [a-z0-9 ._-]+[?]?$",
        r"^explain [a-z0-9 ._-]+[?]?$",
    ]

    return any(
        re.search(
            pattern,
            q,
        )
        for pattern in definition_patterns
    )


# ============================================================
# PLANNER
# ============================================================

def planner_node(
    state: AgentState,
):

    query = state["query"]

    history = _trim_history(
        state.get(
            "conversation_history",
            "",
        ),
        5000,
    )

    prompt = f"""
You are the planning layer of a production AWS AI Assistant.

Determine the user's actual current request.

Use the recent conversation only to resolve references
and understand follow-up questions.

INTENTS:

KNOWLEDGE
- AWS concepts
- AWS documentation
- architecture
- general explanations
- general best practices
- questions that do not require live AWS account data

MONITORING
- current AWS resources
- inventory
- current status
- current CPU/metrics
- current costs
- CloudTrail activity
- Inspector findings
- patch status
- tags

RCA
- why something failed
- why something stopped
- why something is unhealthy
- root cause
- troubleshooting
- incident investigation
- what caused something
- what is causing something
- fixing an existing incident
- preventing recurrence of an existing incident
- recommendations following an incident

IMPORTANT:

If the question asks WHY something is happening,
classify as RCA.

If the question asks HOW MUCH, LIST, SHOW, CHECK,
or otherwise requests current AWS account data,
classify as MONITORING.

If the question is general AWS knowledge,
classify as KNOWLEDGE.

If a follow-up refers to an earlier incident using words such as
"it", "this", "that", "the issue", "the problem", "again",
"how can I fix it", or "how can I prevent it",
use RCA when the previous context contains an incident.

FOLLOW-UP EXAMPLES:

Previous:
"List my EC2 instances."

Current:
"Which ones are running?"

Resolved:
"Which of the user's EC2 instances are running?"

Previous:
"Why is my EC2 instance experiencing high CPU usage?"

Current:
"How can I prevent this?"

Resolved:
"How can the high CPU issue on the affected EC2 instance be prevented?"

AVAILABLE TOOLS:

EC2:
- get_ec2_instances
- get_cloudwatch_metrics
- get_ec2_tags

S3:
- get_s3_buckets
- get_s3_storage_summary
- get_s3_tags

RDS:
- get_rds_instances

VPC:
- get_vpcs
- get_subnets
- get_internet_gateways
- get_route_tables
- get_security_groups

COST:
- get_cost_summary
- get_cost_by_service

PATCH:
- get_patch_status

LAMBDA:
- get_lambda_functions
- get_lambda_tags

SECURITY:
- get_cloudtrail_events
- get_inspector_findings

GENERAL TAGS:
- get_resource_tags

TOOL SELECTION:

EC2 inventory:
get_ec2_instances

EC2 CPU:
get_ec2_instances
get_cloudwatch_metrics

EC2 RCA:
get_ec2_instances
get_cloudwatch_metrics
get_cloudtrail_events
get_ec2_tags

S3 inventory:
get_s3_buckets

S3 storage:
get_s3_storage_summary

RDS:
get_rds_instances

VPC:
get_vpcs

Subnet:
get_subnets

Internet gateway:
get_internet_gateways

Route table:
get_route_tables

Security group:
get_security_groups

AWS cost:
get_cost_summary

Cost by service:
get_cost_by_service

Patch:
get_patch_status

Lambda:
get_lambda_functions

CloudTrail:
get_cloudtrail_events

Inspector:
get_inspector_findings

Tags:
get_resource_tags

RESOURCE GROUNDING:

Never invent AWS resource IDs.

Only return resource references explicitly present
in the conversation or current user query.

Return ONLY JSON.

{{
  "intent": "KNOWLEDGE | MONITORING | RCA",
  "tools": [],
  "services": [],
  "resolved_query": "",
  "resource_refs": []
}}

RECENT CONVERSATION:
{history if history else "No previous conversation."}

CURRENT USER QUERY:
{query}
"""

    response = llm.invoke(
        prompt
    )

    plan = _extract_json(
        _safe_text(response)
    )

    intent = _normalize_intent(
        plan.get("intent")
    )

    tools = _normalize_tools(
        plan.get("tools", [])
    )

    services = _normalize_services(
        plan.get("services", [])
    )

    resolved_query = str(
        plan.get(
            "resolved_query"
        )
        or query
    ).strip()

    resource_refs = plan.get(
        "resource_refs",
        [],
    )

    if not isinstance(
        resource_refs,
        list,
    ):
        resource_refs = []

    resource_refs = [
        str(ref).strip()
        for ref in resource_refs
        if str(ref).strip()
    ]

    # --------------------------------------------------------
    # Deterministic RCA correction
    # --------------------------------------------------------

    incident_followup = (
        _history_contains_incident(
            history
        )
        and (
            _is_followup_query(query)
            or _looks_like_prevention_or_remediation(
                query
            )
        )
    )

    direct_rca = _looks_like_rca_query(
        query
    )

    if (
        incident_followup
        or direct_rca
    ):
        intent = "RCA"

    # --------------------------------------------------------
    # Determine service when planner did not provide it
    # --------------------------------------------------------

    q = resolved_query.lower()

    if not services:

        if "ec2" in q:
            services = "EC2"

        elif "s3" in q:
            services = "S3"

        elif "rds" in q:
            services = "RDS"

        elif "vpc" in q:
            services = "VPC"

        elif (
            "cost" in q
            or "spend" in q
            or "billing" in q
        ):
            services = "COST"

        elif "lambda" in q:
            services = "LAMBDA"

        elif (
            "cloudtrail" in q
            or "api activity" in q
        ):
            services = "CLOUDTRAIL"

        elif (
            "inspector" in q
            or "security finding" in q
            or "security findings" in q
        ):
            services = "SECURITY"

        elif (
            "patch" in q
            or "patched" in q
        ):
            services = "PATCH"

        elif "tag" in q:
            services = "TAGS"

    # --------------------------------------------------------
    # Ensure tools for RCA
    # --------------------------------------------------------

    if intent == "RCA":

        if "ec2" in q or not tools:

            rca_tools = [
                "get_ec2_instances",
                "get_cloudwatch_metrics",
                "get_cloudtrail_events",
                "get_ec2_tags",
            ]

            for tool in rca_tools:

                if tool not in tools:
                    tools.append(tool)

        elif "s3" in q:

            if "get_s3_buckets" not in tools:
                tools.append(
                    "get_s3_buckets"
                )

            if "get_s3_storage_summary" not in tools:
                tools.append(
                    "get_s3_storage_summary"
                )

        elif (
            "cost" in q
            or "spend" in q
            or "billing" in q
        ):

            if "get_cost_summary" not in tools:
                tools.append(
                    "get_cost_summary"
                )

            if "get_cost_by_service" not in tools:
                tools.append(
                    "get_cost_by_service"
                )

        elif (
            "security" in q
            or "inspector" in q
        ):

            if "get_inspector_findings" not in tools:
                tools.append(
                    "get_inspector_findings"
                )

            if "get_cloudtrail_events" not in tools:
                tools.append(
                    "get_cloudtrail_events"
                )

        elif "patch" in q:

            if "get_patch_status" not in tools:
                tools.append(
                    "get_patch_status"
                )

    # --------------------------------------------------------
    # Ensure tools for obvious monitoring
    # --------------------------------------------------------

    if (
        intent == "MONITORING"
        and not tools
    ):

        if "ec2" in q:
            tools.append(
                "get_ec2_instances"
            )

        elif "s3" in q:

            if (
                "storage" in q
                or "size" in q
                or "usage" in q
            ):
                tools.append(
                    "get_s3_storage_summary"
                )

            else:
                tools.append(
                    "get_s3_buckets"
                )

        elif "rds" in q:
            tools.append(
                "get_rds_instances"
            )

        elif "vpc" in q:
            tools.append(
                "get_vpcs"
            )

        elif "subnet" in q:
            tools.append(
                "get_subnets"
            )

        elif "lambda" in q:
            tools.append(
                "get_lambda_functions"
            )

        elif (
            "cost" in q
            or "spend" in q
            or "billing" in q
        ):
            tools.append(
                "get_cost_by_service"
            )

        elif "cloudtrail" in q:
            tools.append(
                "get_cloudtrail_events"
            )

        elif "inspector" in q:
            tools.append(
                "get_inspector_findings"
            )

        elif "patch" in q:
            tools.append(
                "get_patch_status"
            )

    # --------------------------------------------------------
    # Ground resource references
    # --------------------------------------------------------

    trusted = (
        history
        + "\n"
        + query
    )

    valid_refs = []

    for ref in resource_refs:

        if ref in trusted:
            valid_refs.append(ref)

    for instance_id in _extract_ec2_ids(
        trusted
    ):

        if instance_id not in valid_refs:
            valid_refs.append(
                instance_id
            )

    return {
        "intent": intent,
        "tools": tools,
        "service": services,
        "resolved_query": resolved_query,
        "resource_refs": valid_refs,
    }


# ============================================================
# KNOWLEDGE / QDRANT RAG
# ============================================================

def knowledge_node(
    state: AgentState,
):

    query = (
        state.get(
            "resolved_query"
        )
        or state["query"]
    )

    context = retrieve_context(
        query,
        8,
    )

    context = _limit_text(
        context or "",
        9000,
    )

    return {
        "context": context
    }


# ============================================================
# MONITORING
# ============================================================

def monitoring_node(
    state: AgentState,
):

    session_id = state["session_id"]

    tools = state.get(
        "tools",
        [],
    )

    results = {}

    for tool_name in tools:

        tool = TOOL_MAP.get(
            tool_name
        )

        if not tool:

            results[tool_name] = {
                "error": "Unknown tool"
            }

            continue

        try:

            result = tool(
                session_id
            )

            results[tool_name] = result

        except Exception as exc:

            results[tool_name] = {
                "error": str(exc)
            }

    serialized = json.dumps(
        results,
        indent=2,
        default=str,
    )

    serialized = _limit_text(
        serialized,
        9000,
    )

    return {
        "tool_result": serialized
    }


# ============================================================
# RCA
# ============================================================

def rca_node(
    state: AgentState,
):

    query = (
        state.get(
            "resolved_query"
        )
        or state["query"]
    )

    history = _trim_history(
        state.get(
            "conversation_history",
            "",
        ),
        4000,
    )

    tool_result = _limit_text(
        state.get(
            "tool_result",
            "",
        ),
        9000,
    )

    prompt = f"""
You are a senior AWS Site Reliability Engineer.

Perform a grounded Root Cause Analysis.

Use ONLY:

1. Recent conversation context
2. Current resolved request
3. Live AWS evidence

Never invent:

- EC2 instance IDs
- resource names
- metrics
- timestamps
- users
- API calls
- CloudTrail events
- AWS resources
- causes

CURRENT REQUEST:
{query}

RECENT CONVERSATION:
{history if history else "No previous conversation."}

LIVE AWS EVIDENCE:
{tool_result if tool_result else "No AWS evidence available."}

Return ONLY valid JSON:

{{
  "incident_summary": "",
  "confirmed_facts": [],
  "likely_causes": [],
  "assumptions": [],
  "evidence": [],
  "root_cause": "",
  "confidence": "High | Medium | Low",
  "impact": ""
}}

RULES:

- confirmed_facts must contain ONLY facts directly supported by the live AWS evidence or explicitly stated by the user.
- likely_causes must contain ONLY plausible causes supported by the available evidence.
- Do NOT invent causes, user expectations, intentions, requirements, or assumptions.
- The assumptions list MUST be empty unless an assumption is genuinely necessary to interpret the request.
- Never infer what the user expects unless the user explicitly stated it.
- Do NOT treat an AWS resource state as evidence of user intent.
- Do NOT repeat the same fact as multiple causes.
- A condition such as "stopped", "no public IP", or "high CPU" is a fact, not automatically a root cause.
- Clearly distinguish confirmed facts from inferred or possible causes.
- Only state a root cause when the available evidence supports it.
- If the evidence is insufficient, set root_cause to:
  "Root cause cannot be determined from the available evidence."
- Do not invent resource IDs.
- Do not invent CloudTrail events.
- Do not invent CPU values.
- Do not invent metrics, timestamps, resources, or API results.
"""

    response = llm.invoke(
        prompt
    )

    data = _extract_json(
        _safe_text(response)
    )

    if not data:

        data = {
            "incident_summary": (
                "The available AWS evidence "
                "was insufficient to produce "
                "a structured RCA."
            ),
            "confirmed_facts": [],
            "likely_causes": [],
            "assumptions": [],
            "evidence": [],
            "root_cause": (
                "Root cause cannot be determined "
                "from the available evidence."
            ),
            "confidence": "Low",
            "impact": "",
        }

    trusted = (
        history
        + "\n"
        + tool_result
    )

    for key in [
        "incident_summary",
        "root_cause",
        "impact",
    ]:

        if isinstance(
            data.get(key),
            str,
        ):

            data[key] = _ground_text(
                data[key],
                trusted,
            )

    for key in [
        "confirmed_facts",
        "likely_causes",
        "assumptions",
        "evidence",
    ]:

        values = data.get(
            key,
            [],
        )

        if not isinstance(
            values,
            list,
        ):
            values = [
                str(values)
            ]

        data[key] = [
            _ground_text(
                str(value),
                trusted,
            )
            for value in values
            if str(value).strip()
        ]

    return {
        "rca": data
    }


# ============================================================
# RECOMMENDATIONS
# ============================================================

def recommendation_node(
    state: AgentState,
):

    query = (
        state.get(
            "resolved_query"
        )
        or state["query"]
    )

    history = _trim_history(
        state.get(
            "conversation_history",
            "",
        ),
        3000,
    )

    tool_result = _limit_text(
        state.get(
            "tool_result",
            "",
        ),
        7000,
    )

    rca = state.get(
        "rca",
        {},
    )

    rca_text = _limit_text(
        json.dumps(
            rca,
            indent=2,
            default=str,
        ),
        6000,
    )

    prompt = f"""
You are a senior AWS reliability engineer.

Provide practical recommendations based ONLY on:

- user request
- relevant conversation
- live AWS evidence
- RCA

USER REQUEST:
{query}

RECENT CONVERSATION:
{history if history else "No previous conversation."}

AWS EVIDENCE:
{tool_result if tool_result else "No AWS evidence available."}

RCA:
{rca_text}

Return ONLY valid JSON:

{{
  "recommendations": []
}}

Requirements:

- Provide 3 to 5 useful recommendations.
- Prioritize the most important action first.
- Each recommendation must be actionable.
- Do not claim an action has already been performed.
- Do not invent resource IDs.
- Do not invent metrics.
- Do not invent AWS events.
- Do not recommend destructive actions automatically.
- Do not recommend deleting resources unless explicitly requested.
- If evidence is insufficient, recommend investigation.
- For prevention questions, focus on preventing recurrence.
"""

    response = llm.invoke(
        prompt
    )

    data = _extract_json(
        _safe_text(response)
    )

    recommendations = data.get(
        "recommendations",
        [],
    )

    if not isinstance(
        recommendations,
        list,
    ):

        recommendations = [
            str(recommendations)
        ]

    trusted = (
        history
        + "\n"
        + tool_result
    )

    recommendations = [
        _ground_text(
            str(item),
            trusted,
        )
        for item in recommendations
        if str(item).strip()
    ]

    return {
        "recommendations": recommendations[:5]
    }


# ============================================================
# FINAL ANSWER
# ============================================================

def final_node(
    state: AgentState,
):

    query = (
        state.get(
            "resolved_query"
        )
        or state["query"]
    )

    original_query = state["query"]

    intent = state.get(
        "intent",
        "KNOWLEDGE",
    )

    history = _trim_history(
        state.get(
            "conversation_history",
            "",
        ),
        3000,
    )

    detail_requested = _detail_requested(
        original_query
    )

    simple_definition = _is_simple_definition_query(
        original_query
    )

    # ========================================================
    # KNOWLEDGE / RAG
    # ========================================================

    if intent == "KNOWLEDGE":

        context = _limit_text(
            state.get(
                "context",
                "",
            ),
            9000,
        )

        if detail_requested:

            style_instruction = """
The user explicitly requested detail.

Give a comprehensive but focused explanation.

Use useful headings and bullet points where appropriate.

Explain:
- the direct definition or answer
- the important concepts
- how it works
- relevant AWS components
- practical examples
- important considerations or limitations
- a concise practical summary

Use the retrieved knowledge as the source of truth.

Do not add unrelated AWS topics merely to make the answer longer.
"""

        elif simple_definition:

            style_instruction = """
This is a simple AWS definition question.

Give a useful explanation rather than an overly short one.

Start with a clear definition.

Then explain the most important characteristics or components,
followed by a short practical example or use case if useful.

Do not turn a simple definition into a long tutorial.
"""

        else:

            style_instruction = """
Give a moderately detailed, useful AWS explanation.

Answer the question directly first.

Then explain the important concepts needed to understand the answer.

Use bullets, examples, or short headings when they improve clarity.

Do not artificially shorten the answer.

Do not turn every question into a large tutorial.
"""

        prompt = f"""
You are an accurate AWS technical assistant.

Answer the user's question using the retrieved AWS knowledge.

RECENT CONVERSATION:
{history if history else "No previous conversation."}

USER QUESTION:
{original_query}

RESOLVED QUESTION:
{query}

RETRIEVED AWS KNOWLEDGE:
{context if context else "No relevant knowledge was retrieved."}

RESPONSE STYLE:
{style_instruction}

Rules:

- Answer the user's actual question.
- Use the resolved question to understand follow-ups.
- Use retrieved knowledge as the factual source.
- Do not invent AWS account-specific facts.
- Do not invent resource IDs.
- Do not invent metrics.
- If the retrieved knowledge is insufficient, clearly say so.
- Avoid unnecessary repetition.
- Keep the explanation technically accurate.
"""

    # ========================================================
    # MONITORING
    # ========================================================

    elif intent == "MONITORING":

        tool_result = _limit_text(
            state.get(
                "tool_result",
                "",
            ),
            9000,
        )

        q = original_query.lower()

        if any(
            word in q
            for word in [
                "list",
                "show",
                "which",
                "what are",
            ]
        ):

            monitoring_style = """
The user is asking for current AWS resources.

Keep the response concise.

Prefer:
- a count first when available
- then a compact table/list
- only fields relevant to the question

Do not dump the raw AWS API response.
"""

        elif any(
            word in q
            for word in [
                "cpu",
                "metric",
                "utilization",
                "usage",
                "performance",
            ]
        ):

            monitoring_style = """
The user is asking for current AWS metrics.

Give the relevant metric values and time period.

Then give a short interpretation if useful.

Do not add unrelated resource information.
"""

        elif any(
            word in q
            for word in [
                "cost",
                "spend",
                "billing",
                "expensive",
            ]
        ):

            monitoring_style = """
The user is asking about AWS costs.

Lead with the relevant cost/service information.

If service-level data is available, identify the largest relevant
cost and briefly explain what the data shows.

Do not speculate about exact causes of cost unless supported.
"""

        elif any(
            word in q
            for word in [
                "compare",
                "difference",
                "versus",
                "vs",
            ]
        ):

            monitoring_style = """
Give a compact comparison using only the live AWS data.

Highlight the differences that matter to the user's question.
"""

        else:

            monitoring_style = """
Answer concisely but with enough context to make the AWS data useful.

Give the key finding first.

Include only information relevant to the question.

Do not dump raw API output.
"""

        prompt = f"""
You are an AWS monitoring assistant.

Answer using ONLY the live AWS evidence.

RECENT CONVERSATION:
{history if history else "No previous conversation."}

USER QUESTION:
{original_query}

RESOLVED QUESTION:
{query}

LIVE AWS RESULTS:
{tool_result if tool_result else "No live AWS data was returned."}

RESPONSE STYLE:
{monitoring_style}

Rules:

- Never invent resources.
- Never invent metrics.
- Never invent counts.
- Never invent costs.
- Never invent timestamps.
- Never invent EC2 IDs.
- Clearly distinguish unavailable data from actual findings.
- Answer the current question directly.
- Do not expose raw JSON unless the user explicitly asks for it.
"""

    # ========================================================
    # RCA
    # ========================================================

    else:

        tool_result = _limit_text(
            state.get(
                "tool_result",
                "",
            ),
            7000,
        )

        rca = state.get(
            "rca",
            {},
        )

        recommendations = state.get(
            "recommendations",
            [],
        )

        rca_text = _limit_text(
            json.dumps(
                rca,
                indent=2,
                default=str,
            ),
            5000,
        )

        recommendations_text = _limit_text(
            json.dumps(
                recommendations,
                indent=2,
                default=str,
            ),
            3000,
        )

        if detail_requested:

            rca_style = """
The user explicitly requested detail.

Provide a detailed troubleshooting/RCA explanation.

Use this structure:

### Finding

Explain what the AWS evidence actually shows.

### Likely cause

Explain the most plausible causes and clearly distinguish
confirmed evidence from inference.

### Evidence

Mention the important AWS evidence supporting the analysis.

### Recommended actions

Give practical remediation and prevention steps.

If the evidence is insufficient, explicitly say that the
root cause cannot be confirmed.
"""

        else:

            rca_style = """
Keep the RCA focused and useful.

Use exactly this structure:

### Finding

State what the current AWS evidence actually shows.

### Likely cause

State the most plausible explanation, but do not present
an unconfirmed cause as fact.

### Recommended actions

Give practical next steps.

Keep the explanation concise enough to scan, while still
including the important evidence needed to understand the finding.

If the evidence is insufficient, say so clearly.
"""

        prompt = f"""
You are an AWS incident response assistant.

Answer the user's request using ONLY the evidence below.

RECENT CONVERSATION:
{history if history else "No previous conversation."}

USER QUESTION:
{original_query}

RESOLVED QUESTION:
{query}

LIVE AWS FINDINGS:
{tool_result if tool_result else "No AWS evidence available."}

RCA:
{rca_text}

RECOMMENDATIONS:
{recommendations_text}

RESPONSE STYLE:
{rca_style}

Rules:

- Do not invent evidence.
- Do not invent resource IDs.
- Do not invent metrics.
- Do not invent timestamps.
- Separate confirmed facts from likely causes.
- Never call an unproven cause confirmed.
- If evidence is insufficient, say so.
- Do not claim remediation was performed.
- Answer the user's actual question directly.
- If the user asks about prevention, focus on preventing recurrence.
- Do not repeat the entire raw AWS response.
- Do not repeat every recommendation verbatim if they can be summarized clearly.
"""

    response = llm.invoke(
        prompt
    )

    answer = _safe_text(
        response
    )

    trusted = (
        history
        + "\n"
        + state.get(
            "tool_result",
            "",
        )
    )

    answer = _ground_text(
        answer,
        trusted,
    )

    return {
        "answer": answer
    }


# ============================================================
# ROUTING
# ============================================================

def route_after_planner(
    state: AgentState,
):

    intent = state.get(
        "intent",
        "KNOWLEDGE",
    )

    if intent == "KNOWLEDGE":
        return "knowledge"

    return "monitoring"


def route_after_monitoring(
    state: AgentState,
):

    intent = state.get(
        "intent",
        "MONITORING",
    )

    if intent == "RCA":
        return "rca"

    return "final"


# ============================================================
# GRAPH
# ============================================================

graph = StateGraph(
    AgentState
)

graph.add_node(
    "planner",
    planner_node,
)

graph.add_node(
    "knowledge",
    knowledge_node,
)

graph.add_node(
    "monitoring",
    monitoring_node,
)

graph.add_node(
    "rca",
    rca_node,
)

graph.add_node(
    "recommendation",
    recommendation_node,
)

graph.add_node(
    "final",
    final_node,
)

graph.add_edge(
    START,
    "planner",
)

graph.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "knowledge": "knowledge",
        "monitoring": "monitoring",
    },
)

graph.add_edge(
    "knowledge",
    "final",
)

graph.add_conditional_edges(
    "monitoring",
    route_after_monitoring,
    {
        "rca": "rca",
        "final": "final",
    },
)

graph.add_edge(
    "rca",
    "recommendation",
)

graph.add_edge(
    "recommendation",
    "final",
)

graph.add_edge(
    "final",
    END,
)

agent_graph = graph.compile()


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def run_agent(
    session_id: str,
    query: str,
    conversation_history: str = "",
):

    trimmed_history = _trim_history(
        conversation_history,
        5000,
    )

    initial_state: AgentState = {
        "session_id": session_id,
        "query": query.strip(),
        "conversation_history": trimmed_history,
    }

    result = agent_graph.invoke(
        initial_state
    )

    return {
        "answer": result.get(
            "answer",
            "",
        ),
        "intent": result.get(
            "intent",
            "KNOWLEDGE",
        ),
        "service": result.get(
            "service",
            "",
        ),
        "tools": result.get(
            "tools",
            [],
        ),
        "resolved_query": result.get(
            "resolved_query",
            query,
        ),
        "resource_refs": result.get(
            "resource_refs",
            [],
        ),
        "rca": result.get(
            "rca",
            {},
        ),
        "recommendations": result.get(
            "recommendations",
            [],
        ),
    }