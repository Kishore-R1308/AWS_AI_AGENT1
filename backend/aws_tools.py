from aws_auth import get_aws_client
from datetime import date, timedelta


def get_ec2_instances(session_id):
    ec2 = get_aws_client(session_id, "ec2")
    response = ec2.describe_instances()

    results = []

    for reservation in response.get("Reservations", []):
        for instance in reservation.get("Instances", []):
            name = "Unnamed"

            for tag in instance.get("Tags", []):
                if tag["Key"] == "Name":
                    name = tag["Value"]

            results.append(
                {
                    "instance_id": instance.get("InstanceId"),
                    "name": name,
                    "state": instance.get("State", {}).get("Name"),
                    "instance_type": instance.get("InstanceType"),
                    "private_ip": instance.get("PrivateIpAddress"),
                    "public_ip": instance.get("PublicIpAddress"),
                    "vpc_id": instance.get("VpcId"),
                    "subnet_id": instance.get("SubnetId"),
                }
            )

    return results


def get_s3_buckets(session_id):
    s3 = get_aws_client(session_id, "s3")
    response = s3.list_buckets()

    return [
        {
            "name": bucket["Name"],
            "created": str(bucket.get("CreationDate")),
        }
        for bucket in response.get("Buckets", [])
    ]


def get_rds_instances(session_id):
    rds = get_aws_client(session_id, "rds")
    response = rds.describe_db_instances()

    return [
        {
            "identifier": db.get("DBInstanceIdentifier"),
            "status": db.get("DBInstanceStatus"),
            "engine": db.get("Engine"),
            "engine_version": db.get("EngineVersion"),
            "instance_class": db.get("DBInstanceClass"),
            "storage_gb": db.get("AllocatedStorage"),
        }
        for db in response.get("DBInstances", [])
    ]


def get_s3_storage_summary(session_id):
    s3 = get_aws_client(session_id, "s3")
    response = s3.list_buckets()

    results = []

    for bucket in response.get("Buckets", []):
        bucket_name = bucket["Name"]
        total_bytes = 0
        object_count = 0

        try:
            paginator = s3.get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=bucket_name):
                for obj in page.get("Contents", []):
                    total_bytes += obj.get("Size", 0)
                    object_count += 1

            results.append(
                {
                    "bucket": bucket_name,
                    "object_count": object_count,
                    "size_bytes": total_bytes,
                    "size_mb": round(total_bytes / (1024 * 1024), 2),
                    "size_gb": round(
                        total_bytes / (1024 * 1024 * 1024), 4
                    ),
                }
            )

        except Exception as exc:
            results.append(
                {
                    "bucket": bucket_name,
                    "error": str(exc),
                }
            )

    results.sort(
        key=lambda item: item.get("size_bytes", 0),
        reverse=True,
    )

    return results


def get_vpcs(session_id):
    ec2 = get_aws_client(session_id, "ec2")
    response = ec2.describe_vpcs()

    results = []

    for vpc in response.get("Vpcs", []):
        name = "Unnamed"

        for tag in vpc.get("Tags", []):
            if tag.get("Key") == "Name":
                name = tag.get("Value")

        results.append(
            {
                "vpc_id": vpc.get("VpcId"),
                "name": name,
                "state": vpc.get("State"),
                "cidr_block": vpc.get("CidrBlock"),
                "is_default": vpc.get("IsDefault"),
                "dhcp_options_id": vpc.get("DhcpOptionsId"),
            }
        )

    return results


def get_subnets(session_id):
    ec2 = get_aws_client(session_id, "ec2")
    response = ec2.describe_subnets()

    results = []

    for subnet in response.get("Subnets", []):
        name = "Unnamed"

        for tag in subnet.get("Tags", []):
            if tag.get("Key") == "Name":
                name = tag.get("Value")

        results.append(
            {
                "subnet_id": subnet.get("SubnetId"),
                "name": name,
                "vpc_id": subnet.get("VpcId"),
                "cidr_block": subnet.get("CidrBlock"),
                "availability_zone": subnet.get("AvailabilityZone"),
                "state": subnet.get("State"),
                "available_ip_count": subnet.get(
                    "AvailableIpAddressCount"
                ),
                "default_for_az": subnet.get("DefaultForAz"),
            }
        )

    return results


def get_route_tables(session_id):
    ec2 = get_aws_client(session_id, "ec2")
    response = ec2.describe_route_tables()

    results = []

    for route_table in response.get("RouteTables", []):
        name = "Unnamed"

        for tag in route_table.get("Tags", []):
            if tag.get("Key") == "Name":
                name = tag.get("Value")

        results.append(
            {
                "route_table_id": route_table.get("RouteTableId"),
                "name": name,
                "vpc_id": route_table.get("VpcId"),
                "routes": route_table.get("Routes", []),
                "associations": route_table.get(
                    "Associations", []
                ),
            }
        )

    return results


def get_internet_gateways(session_id):
    ec2 = get_aws_client(session_id, "ec2")
    response = ec2.describe_internet_gateways()

    results = []

    for gateway in response.get("InternetGateways", []):
        name = "Unnamed"

        for tag in gateway.get("Tags", []):
            if tag.get("Key") == "Name":
                name = tag.get("Value")

        vpc_ids = []

        for attachment in gateway.get("Attachments", []):
            if attachment.get("VpcId"):
                vpc_ids.append(attachment.get("VpcId"))

        results.append(
            {
                "internet_gateway_id": gateway.get(
                    "InternetGatewayId"
                ),
                "name": name,
                "vpc_ids": vpc_ids,
                "attachments": gateway.get(
                    "Attachments", []
                ),
            }
        )

    return results


def get_security_groups(session_id):
    ec2 = get_aws_client(session_id, "ec2")
    response = ec2.describe_security_groups()

    results = []

    for group in response.get("SecurityGroups", []):
        results.append(
            {
                "group_id": group.get("GroupId"),
                "group_name": group.get("GroupName"),
                "description": group.get("Description"),
                "vpc_id": group.get("VpcId"),
                "ingress_rules": group.get(
                    "IpPermissions", []
                ),
                "egress_rules": group.get(
                    "IpPermissionsEgress", []
                ),
            }
        )

    return results




def get_cost_summary(session_id):
    ce = get_aws_client(session_id, "ce")

    end_date = date.today()
    start_date = end_date - timedelta(days=30)

    response = ce.get_cost_and_usage(
        TimePeriod={
            "Start": start_date.strftime("%Y-%m-%d"),
            "End": end_date.strftime("%Y-%m-%d"),
        },
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
    )

    return response.get("ResultsByTime", [])


def get_cost_by_service(session_id):
    ce = get_aws_client(session_id, "ce")

    end_date = date.today()
    start_date = end_date - timedelta(days=30)

    response = ce.get_cost_and_usage(
        TimePeriod={
            "Start": start_date.strftime("%Y-%m-%d"),
            "End": end_date.strftime("%Y-%m-%d"),
        },
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
    )

    return response.get("ResultsByTime", [])


def get_patch_status(session_id):
    ssm = get_aws_client(session_id, "ssm")

    managed_instances = ssm.describe_instance_information()
    instance_ids = [i["InstanceId"] for i in managed_instances["InstanceInformationList"]]

    response = ssm.describe_instance_patch_states(InstanceIds=instance_ids)

    return [
        {
            "instance_id": instance.get("InstanceId"),
            "missing_count": instance.get("MissingCount"),
            "installed_count": instance.get("InstalledCount"),
            "failed_count": instance.get("FailedCount"),
            "operation_end_time": str(instance.get("OperationEndTime")),
        }
        for instance in response.get("InstancePatchStates", [])
    ]

import boto3

def get_lambda_functions(session_id):
    lambda_client = get_aws_client(session_id, "lambda")

    try:
        paginator = lambda_client.get_paginator("list_functions")
        functions = []

        for page in paginator.paginate():
            for fn in page.get("Functions", []):
                functions.append({
                    "function_name": fn.get("FunctionName"),
                    "runtime": fn.get("Runtime"),
                    "last_modified": fn.get("LastModified"),
                    "arn": fn.get("FunctionArn"),
                })

        return functions

    except Exception as e:
        return {"error": f"Failed to list Lambda functions: {str(e)}"}

