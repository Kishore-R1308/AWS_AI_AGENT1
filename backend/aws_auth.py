
import boto3

from backend.config import AWS_SESSION_DURATION


# ============================================================
# AWS SESSION STORE
# ============================================================

# Demo-only in-memory session store.
#
# Production recommendation:
# Store temporary AWS session information in a secure,
# server-side credential/session store.
AWS_SESSIONS = {}


# ============================================================
# AWS CONNECTION
# ============================================================

def connect_aws(
    session_id,
    access_key,
    secret_key,
    region,
    role_arn,
):
    """
    Authenticate with the supplied AWS credentials,
    assume the requested IAM role, and create a temporary
    session.

    The long-lived access key and secret key are NOT stored.
    Only the temporary assumed-role credentials are retained
    in memory.
    """

    if not session_id:
        raise ValueError(
            "Session ID is required."
        )

    if not access_key or not secret_key:
        raise ValueError(
            "AWS access key and secret key are required."
        )

    if not region:
        raise ValueError(
            "AWS region is required."
        )

    if not role_arn:
        raise ValueError(
            "IAM role ARN is required."
        )

    # --------------------------------------------------------
    # Authenticate using supplied credentials
    # --------------------------------------------------------

    sts = boto3.client(
        "sts",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )

    # --------------------------------------------------------
    # Assume IAM role
    # --------------------------------------------------------

    params = {
        "RoleArn": role_arn,
        "RoleSessionName": (
            f"aws-ai-agent-{session_id[:8]}"
        ),
        "DurationSeconds": AWS_SESSION_DURATION,
    }

    response = sts.assume_role(
        **params
    )

    credentials = response["Credentials"]

    # --------------------------------------------------------
    # Store temporary credentials
    # --------------------------------------------------------

    AWS_SESSIONS[session_id] = {
        "aws_access_key_id": credentials[
            "AccessKeyId"
        ],
        "aws_secret_access_key": credentials[
            "SecretAccessKey"
        ],
        "aws_session_token": credentials[
            "SessionToken"
        ],
        "region": region,
        "expiration": credentials[
            "Expiration"
        ],
    }

    # --------------------------------------------------------
    # Verify assumed identity
    # --------------------------------------------------------

    assumed_sts = boto3.client(
        "sts",
        aws_access_key_id=credentials[
            "AccessKeyId"
        ],
        aws_secret_access_key=credentials[
            "SecretAccessKey"
        ],
        aws_session_token=credentials[
            "SessionToken"
        ],
        region_name=region,
    )

    identity = assumed_sts.get_caller_identity()

    account_id = identity["Account"]
    arn = identity["Arn"]

    # Store account ID for conversation ownership
    # validation in main.py.
    AWS_SESSIONS[session_id][
        "account_id"
    ] = account_id

    return {
        "connected": True,
        "account_id": account_id,
        "arn": arn,
        "region": region,
        "message": (
            "AWS account connected successfully."
        ),
        "session_id": session_id,
    }


# ============================================================
# SESSION CREDENTIALS
# ============================================================

def get_session_credentials(
    session_id,
):
    """
    Retrieve temporary AWS credentials for a session.
    """

    if not session_id:
        raise ValueError(
            "AWS session ID is required."
        )

    credentials = AWS_SESSIONS.get(
        session_id
    )

    if not credentials:
        raise ValueError(
            "AWS account is not connected. "
            "Please connect first."
        )

    return credentials


# ============================================================
# AWS CLIENT
# ============================================================

def get_aws_client(
    session_id,
    service_name,
):
    """
    Create a boto3 client using the temporary
    assumed-role credentials associated with the session.
    """

    if not service_name:
        raise ValueError(
            "AWS service name is required."
        )

    credentials = get_session_credentials(
        session_id
    )

    return boto3.client(
        service_name,
        aws_access_key_id=credentials[
            "aws_access_key_id"
        ],
        aws_secret_access_key=credentials[
            "aws_secret_access_key"
        ],
        aws_session_token=credentials[
            "aws_session_token"
        ],
        region_name=credentials[
            "region"
        ],
    )
