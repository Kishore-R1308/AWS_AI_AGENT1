import os

import requests
import streamlit as st


# ============================================================
# BACKEND URL
# ============================================================

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000",
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AWS AI Agent",
    page_icon="☁️",
    layout="wide",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    section[data-testid="stSidebar"] {
        width: 360px !important;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    section[data-testid="stSidebar"] .stButton > button {
        border: none;
        background: transparent;
        text-align: left;
        border-radius: 8px;
        padding: 7px 10px;
        min-height: 38px;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #e8e8e8;
    }

    div[data-testid="stChatInput"] {
        margin-bottom: 15px;
    }

    .rca-section {
        margin-top: 12px;
        padding: 12px 16px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.25);
    }

    .recommendation-card {
        padding: 10px 14px;
        margin: 8px 0;
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.20);
    }

    .recommendation-title {
        font-weight: 600;
        margin-bottom: 4px;
    }

    .recommendation-details {
        opacity: 0.9;
        font-size: 0.92rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "aws_session_id": None,
    "aws_connected": False,
    "account_id": None,
    "conversation_id": None,
    "messages": [],
    "conversations": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# API HELPERS
# ============================================================

def api_get(path, timeout=10):
    return requests.get(
        f"{BACKEND_URL}{path}",
        timeout=timeout,
    )


def api_post(path, payload=None, timeout=30):
    return requests.post(
        f"{BACKEND_URL}{path}",
        json=payload or {},
        timeout=timeout,
    )


def api_delete(path, timeout=10):
    return requests.delete(
        f"{BACKEND_URL}{path}",
        timeout=timeout,
    )


# ============================================================
# CONVERSATIONS
# ============================================================

def load_conversations():
    """Load conversations belonging to the connected AWS account."""

    if not st.session_state.account_id:
        st.session_state.conversations = []
        return

    try:
        response = api_get(
            f"/conversations/{st.session_state.account_id}"
        )

        if response.status_code == 200:
            st.session_state.conversations = response.json()
        else:
            st.session_state.conversations = []

    except Exception:
        st.session_state.conversations = []


def create_conversation():
    """Create a new conversation."""

    if not st.session_state.account_id:
        return None

    try:
        response = api_post(
            "/conversations",
            {
                "account_id": st.session_state.account_id
            },
        )

        if response.status_code != 200:
            st.error("Could not create a new chat.")
            return None

        data = response.json()

        st.session_state.conversation_id = str(
            data["id"]
        )

        st.session_state.messages = []

        load_conversations()

        return st.session_state.conversation_id

    except Exception as exc:
        st.error(
            f"Could not create conversation: {exc}"
        )

        return None


def load_conversation(conversation_id):
    """Load all messages for a conversation."""

    try:
        conversation_id = str(conversation_id)

        response = api_get(
            f"/conversations/{conversation_id}/messages"
        )

        if response.status_code != 200:
            st.error(
                "Could not load this conversation."
            )
            return

        st.session_state.conversation_id = conversation_id

        st.session_state.messages = response.json()

    except Exception as exc:
        st.error(
            f"Could not load conversation: {exc}"
        )


def delete_conversation(conversation_id):
    """
    Delete a conversation.

    If the deleted conversation was active:
    - open the newest remaining conversation if one exists
    - otherwise leave the screen without an active chat
    """

    try:
        conversation_id = str(conversation_id)

        response = api_delete(
            f"/conversations/{conversation_id}"
        )

        if response.status_code != 200:
            st.error(
                "Could not delete the conversation."
            )
            return

        was_active = (
            str(st.session_state.conversation_id)
            == conversation_id
        )

        load_conversations()

        if was_active:
            st.session_state.conversation_id = None
            st.session_state.messages = []

            if st.session_state.conversations:
                latest_id = str(
                    st.session_state.conversations[0]["id"]
                )

                load_conversation(latest_id)

        st.rerun()

    except Exception as exc:
        st.error(
            f"Could not delete conversation: {exc}"
        )


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(value):
    """
    Clean accidental Streamlit/browser anchor text and
    safely convert values to displayable text.
    """

    if value is None:
        return ""

    text = str(value).strip()

    # Remove accidental markdown anchor fragments such as:
    # [svg](http://localhost:8501/#...)
    while "[svg](" in text:
        start = text.find("[svg](")

        if start == -1:
            break

        end = text.find(")", start)

        if end == -1:
            break

        text = (
            text[:start]
            + text[end + 1:]
        )

    return text.strip()


def render_value(value):
    """
    Render strings, lists and dictionaries safely.
    """

    if value is None:
        return

    if isinstance(value, dict):
        for key, item in value.items():
            if item is None:
                continue

            label = str(key).replace(
                "_",
                " ",
            ).title()

            st.markdown(
                f"**{label}:** {clean_text(item)}"
            )

        return

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                render_value(item)
            else:
                text = clean_text(item)

                if text:
                    st.markdown(
                        f"- {text}"
                    )

        return

    text = clean_text(value)

    if text:
        st.write(text)


# ============================================================
# RCA RENDERING
# ============================================================

def render_rca(rca):
    """Render structured Root Cause Analysis cleanly."""

    if not rca:
        return

    if not isinstance(rca, dict):
        text = clean_text(rca)

        if text:
            st.markdown(
                "#### 🔍 Root Cause Analysis"
            )
            st.write(text)

        return

    st.markdown(
        "#### 🔍 Root Cause Analysis"
    )

    confirmed_facts = rca.get(
        "confirmed_facts"
    )

    likely_causes = rca.get(
        "likely_causes"
    )

    assumptions = rca.get(
        "assumptions"
    )

    confidence = rca.get(
        "confidence"
    )

    impact = rca.get(
        "impact"
    )

    analysis = rca.get(
        "analysis"
    )

    if confirmed_facts:
        st.markdown(
            "**Confirmed facts**"
        )

        if isinstance(
            confirmed_facts,
            list,
        ):
            for item in confirmed_facts:
                text = clean_text(item)

                if text:
                    st.markdown(
                        f"- {text}"
                    )
        else:
            st.write(
                clean_text(
                    confirmed_facts
                )
            )

    if likely_causes:
        st.markdown(
            "**Likely causes**"
        )

        if isinstance(
            likely_causes,
            list,
        ):
            for item in likely_causes:
                text = clean_text(item)

                if text:
                    st.markdown(
                        f"- {text}"
                    )
        else:
            st.write(
                clean_text(
                    likely_causes
                )
            )

    if assumptions:
        st.markdown(
            "**Assumptions**"
        )

        if isinstance(
            assumptions,
            list,
        ):
            for item in assumptions:
                text = clean_text(item)

                if text:
                    st.markdown(
                        f"- {text}"
                    )
        else:
            st.write(
                clean_text(
                    assumptions
                )
            )

    if confidence:
        st.markdown(
            f"**Confidence:** "
            f"{clean_text(confidence)}"
        )

    if impact:
        st.markdown(
            f"**Impact:** "
            f"{clean_text(impact)}"
        )

    if analysis:
        st.markdown(
            "**Analysis**"
        )
        st.write(
            clean_text(analysis)
        )


# ============================================================
# RECOMMENDATIONS RENDERING
# ============================================================

def render_recommendations(
    recommendations,
):
    """
    Render recommendations as clean cards.

    Supports:
    - list[str]
    - list[dict]
    - single string
    - single dict
    """

    if not recommendations:
        return

    st.markdown(
        "#### 💡 Recommended Actions"
    )

    if isinstance(
        recommendations,
        dict,
    ):
        recommendations = [
            recommendations
        ]

    elif not isinstance(
        recommendations,
        list,
    ):
        recommendations = [
            recommendations
        ]

    for index, recommendation in enumerate(
        recommendations,
        start=1,
    ):

        # ----------------------------------------------------
        # Structured recommendation
        # ----------------------------------------------------

        if isinstance(
            recommendation,
            dict,
        ):
            priority = recommendation.get(
                "priority"
            )

            action = recommendation.get(
                "action"
            )

            details = recommendation.get(
                "details"
            )

            reason = recommendation.get(
                "reason"
            )

            expected_result = recommendation.get(
                "expected_result"
            )

            if not action:
                action = recommendation.get(
                    "title"
                )

            if not details:
                details = reason

            priority_text = ""

            if priority is not None:
                priority_text = (
                    f"Priority {clean_text(priority)}"
                )

            action_text = clean_text(
                action
            )

            details_text = clean_text(
                details
            )

            expected_text = clean_text(
                expected_result
            )

            st.markdown(
                '<div class="recommendation-card">',
                unsafe_allow_html=True,
            )

            if priority_text:
                st.markdown(
                    f'<div class="recommendation-title">'
                    f'{priority_text}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            if action_text:
                st.markdown(
                    f"**{action_text}**"
                )

            if details_text:
                st.markdown(
                    f'<div class="recommendation-details">'
                    f'{details_text}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            if expected_text:
                st.markdown(
                    f"**Expected result:** "
                    f"{expected_text}"
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

            continue

        # ----------------------------------------------------
        # Plain text recommendation
        # ----------------------------------------------------

        text = clean_text(
            recommendation
        )

        if text:
            st.markdown(
                f"**{index}.** {text}"
            )


# ============================================================
# MESSAGE RENDERING
# ============================================================

def render_message(message):
    """Render one stored conversation message."""

    role = message.get("role")

    if role not in (
        "user",
        "assistant",
    ):
        return

    with st.chat_message(role):

        content = clean_text(
            message.get(
                "content",
                "",
            )
        )

        if content:
            st.markdown(content)

        if role == "assistant":

            intent = message.get(
                "intent"
            )

            if intent:

                metadata = [
                    f"Intent: {intent}"
                ]

                if (
                    intent == "MONITORING"
                    and message.get(
                        "service"
                    )
                ):
                    metadata.append(
                        f"Service: "
                        f"{message['service']}"
                    )

                st.caption(
                    " | ".join(metadata)
                )

            render_rca(
                message.get("rca")
            )

            render_recommendations(
                message.get(
                    "recommendations"
                )
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "☁️ AWS AI Agent"
    )

    st.subheader(
        "AWS Connection"
    )

    st.write(
        "Connect using a cross-account IAM role."
    )

    access_key = st.text_input(
        "AWS Access Key",
        type="password",
    )

    secret_key = st.text_input(
        "AWS Secret Key",
        type="password",
    )

    region = st.text_input(
        "AWS Region",
        value="us-east-1",
    )

    role_arn = st.text_input(
        "Cross Account Role ARN",
        placeholder=(
            "arn:aws:iam::123456789012:"
            "role/AIAgentReadOnlyRole"
        ),
    )

    connect_button = st.button(
        "🔗 Connect to AWS",
        use_container_width=True,
    )


    # ========================================================
    # AWS CONNECTION
    # ========================================================

    if connect_button:

        if (
            not access_key
            or not secret_key
            or not role_arn
        ):

            st.error(
                "Access key, secret key and "
                "role ARN are required."
            )

        else:

            with st.spinner(
                "Authenticating with AWS..."
            ):

                try:

                    response = api_post(
                        "/aws/connect",
                        {
                            "access_key": access_key,
                            "secret_key": secret_key,
                            "region": region,
                            "role_arn": role_arn,
                        },
                        timeout=30,
                    )

                    if response.status_code == 200:

                        data = response.json()

                        st.session_state.aws_session_id = (
                            data.get(
                                "session_id"
                            )
                        )

                        st.session_state.aws_connected = (
                            True
                        )

                        st.session_state.account_id = (
                            data.get(
                                "account_id"
                            )
                        )

                        st.session_state.conversation_id = (
                            None
                        )

                        st.session_state.messages = []

                        # Load existing conversations.
                        load_conversations()

                        # If conversations exist, open
                        # the newest one.
                        if (
                            st.session_state.conversations
                        ):

                            latest_id = str(
                                st.session_state.conversations[
                                    0
                                ]["id"]
                            )

                            load_conversation(
                                latest_id
                            )

                        # Do NOT automatically create
                        # an empty conversation.

                        st.success(
                            "AWS Connected Successfully"
                        )

                        st.rerun()

                    else:

                        try:
                            detail = (
                                response.json().get(
                                    "detail",
                                    response.text,
                                )
                            )

                        except Exception:
                            detail = response.text

                        st.error(
                            f"AWS connection failed "
                            f"(HTTP "
                            f"{response.status_code}): "
                            f"{detail}"
                        )

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Cannot connect to the FastAPI backend. "
                        "Make sure Uvicorn is running on port 8000."
                    )

                except requests.exceptions.Timeout:

                    st.error(
                        "The backend request timed out."
                    )

                except Exception as exc:

                    st.error(
                        f"Backend error: {exc}"
                    )


    # ========================================================
    # CONNECTED STATE
    # ========================================================

    if st.session_state.aws_connected:

        st.success(
            "🟢 AWS Connected"
        )

        st.write(
            f"Account: "
            f"`{st.session_state.account_id}`"
        )

        st.divider()

        st.subheader(
            "💬 Chat History"
        )

        # ----------------------------------------------------
        # NEW CHAT
        # ----------------------------------------------------

        if st.button(
            "➕ New Chat",
            use_container_width=True,
        ):

            if create_conversation():
                st.rerun()


        st.divider()

        load_conversations()

        # ----------------------------------------------------
        # CHAT LIST
        # ----------------------------------------------------

        if not st.session_state.conversations:

            st.caption(
                "No conversations yet."
            )

        else:

            for conversation in (
                st.session_state.conversations
            ):

                conversation_id = str(
                    conversation["id"]
                )

                title = (
                    conversation.get(
                        "title"
                    )
                    or "New Chat"
                )

                is_active = (
                    conversation_id
                    == str(
                        st.session_state.conversation_id
                    )
                )

                col1, col2 = st.columns(
                    [5, 1],
                    gap="small",
                )

                with col1:

                    label = (
                        f"▶ {title}"
                        if is_active
                        else title
                    )

                    if st.button(
                        label,
                        key=f"chat_{conversation_id}",
                        use_container_width=True,
                    ):

                        load_conversation(
                            conversation_id
                        )

                        st.rerun()

                with col2:

                    if st.button(
                        "🗑️",
                        key=f"delete_{conversation_id}",
                        help="Delete chat",
                        use_container_width=True,
                    ):

                        delete_conversation(
                            conversation_id
                        )

    else:

        st.warning(
            "🔴 AWS Not Connected"
        )


# ============================================================
# MAIN CHAT
# ============================================================

if not st.session_state.aws_connected:

    st.title(
        "☁️ AWS AI Agent"
    )

    st.info(
        "Connect your AWS account from the sidebar "
        "to start chatting."
    )

    st.stop()


# ============================================================
# NO ACTIVE CONVERSATION
# ============================================================

if not st.session_state.conversation_id:

    st.title(
        "☁️ AWS AI Agent"
    )

    st.info(
        "Select an existing conversation from the sidebar "
        "or click **➕ New Chat** to start."
    )

    st.stop()


# ============================================================
# DISPLAY HISTORY
# ============================================================

for message in st.session_state.messages:

    render_message(
        message
    )


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Ask about AWS..."
)


if prompt:

    # --------------------------------------------------------
    # User message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(prompt)


    # --------------------------------------------------------
    # Assistant response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Agent is thinking..."
        ):

            try:

                response = api_post(
                    "/chat",
                    {
                        "session_id":
                            st.session_state.aws_session_id,

                        "conversation_id":
                            st.session_state.conversation_id,

                        "message":
                            prompt,
                    },
                    timeout=120,
                )


                # ------------------------------------------------
                # API ERROR
                # ------------------------------------------------

                if response.status_code != 200:

                    try:

                        detail = (
                            response.json().get(
                                "detail",
                                "Agent failed.",
                            )
                        )

                    except Exception:

                        detail = response.text

                    st.error(
                        detail
                    )


                # ------------------------------------------------
                # SUCCESS
                # ------------------------------------------------

                else:

                    data = response.json()


                    # ------------------------------------------------
                    # Main answer
                    # ------------------------------------------------

                    answer = clean_text(
                        data.get(
                            "answer",
                            "",
                        )
                    )

                    if answer:

                        st.markdown(
                            answer
                        )


                    # ------------------------------------------------
                    # Metadata
                    # ------------------------------------------------

                    intent = data.get(
                        "intent"
                    )

                    if intent:

                        caption = (
                            f"Intent: {intent}"
                        )

                        if (
                            intent == "MONITORING"
                            and data.get(
                                "service"
                            )
                        ):

                            caption += (
                                f" | Service: "
                                f"{data['service']}"
                            )

                        st.caption(
                            caption
                        )


                    # ------------------------------------------------
                    # RCA
                    # ------------------------------------------------

                    render_rca(
                        data.get(
                            "rca"
                        )
                    )


                    # ------------------------------------------------
                    # Recommendations
                    # ------------------------------------------------

                    render_recommendations(
                        data.get(
                            "recommendations"
                        )
                    )


                    # ------------------------------------------------
                    # Update local state
                    # ------------------------------------------------

                    st.session_state.messages.extend(
                        [
                            {
                                "role": "user",
                                "content": prompt,
                            },
                            {
                                "role": "assistant",
                                "content": data.get(
                                    "answer",
                                    "",
                                ),
                                "intent": data.get(
                                    "intent"
                                ),
                                "service": data.get(
                                    "service"
                                ),
                                "rca": data.get(
                                    "rca"
                                ),
                                "recommendations": data.get(
                                    "recommendations"
                                ),
                            },
                        ]
                    )


                    load_conversations()


            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to the FastAPI backend. "
                    "Make sure Uvicorn is running on port 8000."
                )


            except requests.exceptions.Timeout:

                st.error(
                    "The chat request timed out."
                )


            except Exception as exc:

                st.error(
                    f"Error: {exc}"
                )