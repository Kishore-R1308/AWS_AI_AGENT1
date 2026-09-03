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


# ============================================================
# CONVERSATIONS
# ============================================================

def load_conversations():

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

        st.session_state.conversation_id = data["id"]
        st.session_state.messages = []

        load_conversations()

        return data["id"]

    except Exception as exc:

        st.error(
            f"Could not create conversation: {exc}"
        )

        return None


def load_conversation(conversation_id):

    try:

        response = api_get(
            f"/conversations/{conversation_id}/messages"
        )

        if response.status_code != 200:

            st.error("Could not load this conversation.")
            return

        st.session_state.conversation_id = conversation_id
        st.session_state.messages = response.json()

    except Exception as exc:

        st.error(
            f"Could not load conversation: {exc}"
        )


def delete_conversation(conversation_id):

    try:

        response = requests.delete(
            f"{BACKEND_URL}/conversations/{conversation_id}",
            timeout=10,
        )

        if response.status_code != 200:

            st.error("Could not delete the conversation.")
            return

        if (
            st.session_state.conversation_id
            == conversation_id
        ):

            st.session_state.conversation_id = None
            st.session_state.messages = []

            create_conversation()

        load_conversations()

        st.rerun()

    except Exception as exc:

        st.error(
            f"Could not delete conversation: {exc}"
        )


# ============================================================
# MESSAGE RENDERING
# ============================================================

def render_message(message):

    role = message.get("role")

    if role not in ("user", "assistant"):
        return

    with st.chat_message(role):

        st.write(
            message.get("content", "")
        )

        if role == "assistant":

            intent = message.get("intent")

            if intent:

                metadata = [
                    f"Intent: {intent}"
                ]

                if (
                    intent == "MONITORING"
                    and message.get("service")
                ):

                    metadata.append(
                        f"Service: {message['service']}"
                    )

                st.caption(
                    " | ".join(metadata)
                )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("☁️ AWS AI Agent")

    st.subheader("AWS Connection")

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

    if connect_button:

        if (
            not access_key
            or not secret_key
            or not role_arn
        ):

            st.error(
                "Access key, secret key and role ARN are required."
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
                            data["session_id"]
                        )

                        st.session_state.aws_connected = True

                        st.session_state.account_id = (
                            data["account_id"]
                        )

                        st.session_state.conversation_id = None
                        st.session_state.messages = []

                        load_conversations()

                        if st.session_state.conversations:

                            load_conversation(
                                st.session_state.conversations[0]["id"]
                            )

                        else:

                            create_conversation()

                        st.success(
                            "AWS Connected Successfully"
                        )

                        st.rerun()

                    else:

                        try:

                            detail = response.json().get(
                                "detail",
                                "AWS connection failed.",
                            )

                        except Exception:

                            detail = response.text

                        st.error(detail)

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

        st.success("🟢 AWS Connected")

        st.write(
            f"Account: `{st.session_state.account_id}`"
        )

        st.divider()

        st.subheader("💬 Chat History")

        if st.button(
            "＋ New Chat",
            use_container_width=True,
        ):

            create_conversation()
            st.rerun()

        st.divider()

        load_conversations()

        for conversation in st.session_state.conversations:

            conversation_id = conversation["id"]

            title = (
                conversation["title"]
                or "New Chat"
            )

            is_active = (
                conversation_id
                == st.session_state.conversation_id
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
                    "🗑",
                    key=f"delete_{conversation_id}",
                    help="Delete chat",
                    use_container_width=True,
                ):

                    delete_conversation(
                        conversation_id
                    )

    else:

        st.warning("🔴 AWS Not Connected")


# ============================================================
# MAIN CHAT
# ============================================================

if not st.session_state.aws_connected:

    st.title("☁️ AWS AI Agent")

    st.info(
        "Connect your AWS account from the sidebar "
        "to start chatting."
    )

    st.stop()


# ============================================================
# ENSURE CONVERSATION
# ============================================================

if not st.session_state.conversation_id:

    create_conversation()


# ============================================================
# DISPLAY HISTORY
# ============================================================

for message in st.session_state.messages:

    render_message(message)


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Ask about AWS..."
)


if prompt:

    if not st.session_state.conversation_id:

        create_conversation()


    # --------------------------------------------------------
    # User message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.write(prompt)


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

                if response.status_code != 200:

                    try:

                        detail = response.json().get(
                            "detail",
                            "Agent failed.",
                        )

                    except Exception:

                        detail = response.text

                    st.error(detail)

                else:

                    data = response.json()

                    st.write(
                        data["answer"]
                    )

                    caption = (
                        f"Intent: {data['intent']}"
                    )

                    if (
                        data["intent"]
                        == "MONITORING"
                        and data.get("service")
                    ):

                        caption += (
                            f" | Service: "
                            f"{data['service']}"
                        )

                    st.caption(caption)

                    # Update local state

                    st.session_state.messages.extend(
                        [
                            {
                                "role": "user",
                                "content": prompt,
                            },
                            {
                                "role": "assistant",
                                "content": data["answer"],
                                "intent": data["intent"],
                                "service": data.get("service"),
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