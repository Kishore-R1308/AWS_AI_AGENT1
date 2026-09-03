import json
from typing import TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph

from backend.aws_tools import (
    get_ec2_instances,
    get_rds_instances,
    get_s3_buckets,
    get_s3_storage_summary,
    get_vpcs,
    get_subnets,
    get_internet_gateways,
    get_route_tables,
    get_security_groups,
    get_cost_by_service,
    get_cost_summary,
    get_patch_status,
    get_lambda_functions,
)
from backend.schemas import Plan
from backend.config import GROQ_API_KEY, GROQ_MODEL
from backend.rag import retrieve_context


# =====================================================
# TOOL MAP
# =====================================================

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
}


# =====================================================
# STATE
# =====================================================

class AgentState(TypedDict, total=False):
    session_id: str
    query: str
    conversation_history: str
    intent: str
    tools: list[str]
    context: str
    service: str
    tool_result: str
    answer: str


# =====================================================
# LLM
# =====================================================

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0,
    api_key=GROQ_API_KEY,
)

planner_llm = llm.with_structured_output(Plan)


# =====================================================
# PLANNER NODE
# =====================================================

def planner_node(state):
    prompt = f"""
You are an AWS AI planner.

Use the previous conversation to understand the user's
current question.

Previous Conversation:
{state.get("conversation_history", "No previous conversation.")}

If the current question contains words such as:
- it
- this
- that
- they
- those
- them
- how many
- what about it
- what about them
- show me those

use the previous conversation to determine what the user
is referring to.

Your responsibilities:

1. Determine the user's intent.
2. Decide whether AWS account data is required.
3. Select ALL required tools.

Available Tools:

get_ec2_instances
get_s3_buckets
get_s3_storage_summary
get_rds_instances
get_vpcs
get_subnets
get_internet_gateways
get_route_tables
get_security_groups
get_cost_summary
get_cost_by_service
get_patch_status
get_lambda_functions

Intent Definitions:

KNOWLEDGE:
AWS concepts
Explanations
Documentation
Architecture
Configuration
How-to questions
Best practices

MONITORING:
Questions requiring live AWS account data
Inventory
Counts
Resource status
Storage usage
Existing resources

Rules:

- Detect ALL services mentioned or implied.
- A question may require multiple tools.
- Never omit a required tool.
- Use get_s3_storage_summary only when storage usage,
  bucket sizing, or storage comparison is required.
- For follow-up questions, use the previous conversation
  to resolve the user's reference.
- Return JSON only.

Examples:

User:
What is EC2?

Response:
{{
    "intent": "KNOWLEDGE",
    "tools": []
}}

User:
How many EC2 instances do I have?

Response:
{{
    "intent": "MONITORING",
    "tools": [
        "get_ec2_instances"
    ]
}}

User:
List EC2 and S3 resources.

Response:
{{
    "intent": "MONITORING",
    "tools": [
        "get_ec2_instances",
        "get_s3_buckets"
    ]
}}

User:
Show EC2, S3 and RDS inventory.

Response:
{{
    "intent": "MONITORING",
    "tools": [
        "get_ec2_instances",
        "get_s3_buckets",
        "get_rds_instances"
    ]
}}

User:
Which S3 bucket consumes the most storage?

Response:
{{
    "intent": "MONITORING",
    "tools": [
        "get_s3_storage_summary"
    ]
}}

Previous Conversation:
{state.get("conversation_history", "")}

Current User Query:
{state["query"]}

Return only the structured output.
"""

    try:
        response = llm.invoke(prompt)

        content = response.content
        result = json.loads(content)

        intent = result.get("intent", "KNOWLEDGE").upper()
        tools = result.get("tools", [])

        valid_tools = [
            tool for tool in tools
            if tool in TOOL_MAP
        ]

        return {
            "intent": intent,
            "tools": valid_tools,
        }

    except Exception as e:
        print("Planner Error:", str(e))

        return {
            "intent": "KNOWLEDGE",
            "tools": [],
        }


# =====================================================
# KNOWLEDGE NODE
# =====================================================

def knowledge_node(state):
    context = retrieve_context(state["query"])

    return {
        "context": context
    }


# =====================================================
# MONITORING NODE
# =====================================================

def monitoring_node(state):
    session_id = state["session_id"]
    tools = state.get("tools", [])

    print("\nSelected Tools:", tools)

    results = {}

    for tool_name in tools:
        try:
            print(f"Executing {tool_name}")

            tool_function = TOOL_MAP[tool_name]

            results[tool_name] = tool_function(session_id)

        except Exception as e:
            print(
                f"Error executing {tool_name}: {str(e)}"
            )

            results[tool_name] = {
                "error": str(e)
            }

    return {
        "tool_result": json.dumps(
            results,
            indent=2,
            default=str
        )
    }


# =====================================================
# FINAL ANSWER NODE
# =====================================================

def final_answer_node(state):
    history = state.get("conversation_history", "")

    if state["intent"] == "KNOWLEDGE":
        prompt = f"""
You are an AWS technical assistant.

Previous Conversation:
{history}

Current Question:
{state["query"]}

Knowledge Base:
{state.get("context", "")}

Rules:
- Use the previous conversation when needed to understand the current question.
- Answer using the supplied AWS knowledge.
- Do not invent information.
- If the question refers to something from the previous conversation, use that context.
- Give a clear and practical answer.
"""

    else:
        prompt = f"""
You are an AWS monitoring assistant.

Previous Conversation:
{history}

Current Question:
{state["query"]}

Tools Executed:
{state.get("tools")}

AWS Results:
{state.get("tool_result")}

Rules:
1. Use previous conversation to understand references such as "it", "this", "that", "they", etc.
2. Use only the current AWS Results for live AWS information.
3. Never invent resources, counts, or names.
4. If multiple services were returned, summarize each service.
5. Provide totals whenever possible.
6. If no resources exist, explicitly state that.
7. If errors are present, summarize them clearly.
8. Use headings, tables, and bullet points where useful.
9. Give a concise and clear answer.

Generate the final answer.
"""

    response = llm.invoke(prompt)

    return {"answer": response.content}

# =====================================================
# ROUTER
# =====================================================

def route_after_planner(state):

    if state["intent"] == "KNOWLEDGE":
        return "knowledge"

    return "monitoring"


# =====================================================
# GRAPH
# =====================================================

builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("knowledge", knowledge_node)
builder.add_node("monitoring", monitoring_node)
builder.add_node("final", final_answer_node)

builder.add_edge(START, "planner")

builder.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "knowledge": "knowledge",
        "monitoring": "monitoring",
    },
)

builder.add_edge("knowledge", "final")
builder.add_edge("monitoring", "final")

builder.add_edge("final", END)

graph = builder.compile()


# =====================================================
# RUN AGENT
# =====================================================

def run_agent(
    session_id,
    query,
    conversation_history="",
):
    result = graph.invoke(
        {
            "session_id": session_id,
            "query": query,
            "conversation_history": conversation_history,
        }
    )

    return {
        "answer": result["answer"],
        "intent": result["intent"],
        "tools": result.get("tools", []),
    }