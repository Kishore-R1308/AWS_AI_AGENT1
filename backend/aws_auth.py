import boto3

from config import AWS_SESSION_DURATION

# Demo-only in-memory session store.
# Use a secure session/credential store in production.
AWS_SESSIONS = {}


def connect_aws(
    session_id,
    access_key,
    secret_key,
    region,
    role_arn,
):
    sts = boto3.client(
        "sts",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )

    params = {
        "RoleArn": role_arn,
        "RoleSessionName": f"aws-ai-agent-{session_id[:8]}",
        "DurationSeconds": AWS_SESSION_DURATION,
    }

    response = sts.assume_role(**params)
    credentials = response["Credentials"]

    AWS_SESSIONS[session_id] = {
        "aws_access_key_id": credentials["AccessKeyId"],
        "aws_secret_access_key": credentials["SecretAccessKey"],
        "aws_session_token": credentials["SessionToken"],
        "region": region,
        "expiration": credentials["Expiration"],
    }

    assumed_sts = boto3.client(
        "sts",
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
        region_name=region,
    )

    identity = assumed_sts.get_caller_identity()
    account_id=identity["Account"]
    arn=identity["Arn"]
    AWS_SESSIONS[session_id]["account_id"] = account_id

    return {
        "connected": True,
        "account_id": identity["Account"],
        "arn": identity["Arn"],
        "region": region,
        "message": "AWS account connected successfully.",
        "session_id": session_id,
    }


def get_session_credentials(session_id):
    credentials = AWS_SESSIONS.get(session_id)

    if not credentials:
        raise ValueError(
            "AWS account is not connected. Please connect first."
        )

    return credentials


def get_aws_client(session_id, service_name):
    credentials = get_session_credentials(session_id)

    return boto3.client(
        service_name,
        aws_access_key_id=credentials["aws_access_key_id"],
        aws_secret_access_key=credentials["aws_secret_access_key"],
        aws_session_token=credentials["aws_session_token"],
        region_name=credentials["region"],
    )