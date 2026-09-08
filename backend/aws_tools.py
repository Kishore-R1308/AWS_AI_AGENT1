from datetime import datetime, timedelta, timezone, date

from backend.aws_auth import get_aws_client


# ============================================================
# HELPERS
# ============================================================

def _client(session_id: str, service_name: str):
    """
    Return an authenticated AWS boto3 client for the user's
    active AWS session.
    """
    return get_aws_client(session_id, service_name)


def _tag_dict(tags):
    """
    Convert AWS tag list into a simple dictionary.
    """
    if not tags:
        return {}

    return {
        tag.get("Key"): tag.get("Value")
        for tag in tags
        if tag.get("Key")
    }


# ============================================================
# EC2
# ============================================================

def get_ec2_instances(session_id: str):
    """
    Return all EC2 instances with useful inventory details.
    """
    ec2 = _client(session_id, "ec2")

    response = ec2.describe_instances()

    instances = []

    for reservation in response.get("Reservations", []):
        for instance in reservation.get("Instances", []):

            tags = _tag_dict(instance.get("Tags", []))

            instances.append({
                "instance_id": instance.get("InstanceId"),
                "name": tags.get("Name"),
                "state": instance.get("State", {}).get("Name"),
                "instance_type": instance.get("InstanceType"),
                "private_ip": instance.get("PrivateIpAddress"),
                "public_ip": instance.get("PublicIpAddress"),
                "vpc_id": instance.get("VpcId"),
                "subnet_id": instance.get("SubnetId"),
                "availability_zone": (
                    instance.get("Placement", {}).get("AvailabilityZone")
                ),
                "architecture": instance.get("Architecture"),
                "launch_time": str(instance.get("LaunchTime"))
                if instance.get("LaunchTime")
                else None,
                "tags": tags,
            })

    return {
        "count": len(instances),
        "instances": instances,
    }


# ============================================================
# S3
# ============================================================

def get_s3_buckets(session_id: str):
    """
    Return all S3 buckets visible to the authenticated account.
    """
    s3 = _client(session_id, "s3")

    response = s3.list_buckets()

    buckets = []

    for bucket in response.get("Buckets", []):
        buckets.append({
            "name": bucket.get("Name"),
            "creation_date": str(bucket.get("CreationDate"))
            if bucket.get("CreationDate")
            else None,
        })

    return {
        "count": len(buckets),
        "buckets": buckets,
    }


# ============================================================
# S3 STORAGE SUMMARY
# ============================================================

def get_s3_storage_summary(session_id: str):
    """
    Calculate S3 object counts and storage usage for every bucket.

    Uses pagination so large buckets are handled safely.
    """
    s3 = _client(session_id, "s3")

    buckets_response = s3.list_buckets()

    summaries = []

    for bucket in buckets_response.get("Buckets", []):
        bucket_name = bucket.get("Name")

        if not bucket_name:
            continue

        total_bytes = 0
        object_count = 0

        try:
            paginator = s3.get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=bucket_name):
                objects = page.get("Contents", [])

                object_count += len(objects)

                for obj in objects:
                    total_bytes += obj.get("Size", 0) or 0

            total_mb = total_bytes / (1024 ** 2)
            total_gb = total_bytes / (1024 ** 3)

            summaries.append({
                "bucket": bucket_name,
                "object_count": object_count,
                "size_bytes": total_bytes,
                "size_mb": round(total_mb, 2),
                "size_gb": round(total_gb, 4),
            })

        except Exception as exc:
            summaries.append({
                "bucket": bucket_name,
                "object_count": 0,
                "size_bytes": 0,
                "size_mb": 0,
                "size_gb": 0,
                "error": str(exc),
            })

    summaries.sort(
        key=lambda item: item.get("size_bytes", 0),
        reverse=True,
    )

    total_bytes = sum(
        item.get("size_bytes", 0)
        for item in summaries
    )

    return {
        "bucket_count": len(summaries),
        "total_bytes": total_bytes,
        "total_gb": round(total_bytes / (1024 ** 3), 4),
        "buckets": summaries,
    }


# ============================================================
# RDS
# ============================================================

def get_rds_instances(session_id: str):
    """
    Return RDS instance inventory.
    """
    rds = _client(session_id, "rds")

    paginator = rds.get_paginator("describe_db_instances")

    instances = []

    for page in paginator.paginate():
        for db in page.get("DBInstances", []):

            instances.append({
                "identifier": db.get("DBInstanceIdentifier"),
                "engine": db.get("Engine"),
                "engine_version": db.get("EngineVersion"),
                "status": db.get("DBInstanceStatus"),
                "instance_class": db.get("DBInstanceClass"),
                "allocated_storage_gb": db.get("AllocatedStorage"),
                "availability_zone": db.get("AvailabilityZone"),
                "multi_az": db.get("MultiAZ"),
                "endpoint": (
                    db.get("Endpoint", {}).get("Address")
                    if db.get("Endpoint")
                    else None
                ),
                "port": (
                    db.get("Endpoint", {}).get("Port")
                    if db.get("Endpoint")
                    else None
                ),
                "vpc_id": db.get("DBSubnetGroup", {}).get("VpcId")
                if db.get("DBSubnetGroup")
                else None,
            })

    return {
        "count": len(instances),
        "instances": instances,
    }


# ============================================================
# VPC
# ============================================================

def get_vpcs(session_id: str):
    """
    Return VPC inventory.
    """
    ec2 = _client(session_id, "ec2")

    response = ec2.describe_vpcs()

    vpcs = []

    for vpc in response.get("Vpcs", []):

        tags = _tag_dict(vpc.get("Tags", []))

        vpcs.append({
            "vpc_id": vpc.get("VpcId"),
            "cidr_block": vpc.get("CidrBlock"),
            "state": vpc.get("State"),
            "is_default": vpc.get("IsDefault"),
            "dhcp_options_id": vpc.get("DhcpOptionsId"),
            "instance_tenancy": vpc.get("InstanceTenancy"),
            "name": tags.get("Name"),
            "tags": tags,
        })

    return {
        "count": len(vpcs),
        "vpcs": vpcs,
    }


# ============================================================
# SUBNETS
# ============================================================

def get_subnets(session_id: str):
    """
    Return subnet inventory.
    """
    ec2 = _client(session_id, "ec2")

    response = ec2.describe_subnets()

    subnets = []

    for subnet in response.get("Subnets", []):

        tags = _tag_dict(subnet.get("Tags", []))

        subnets.append({
            "subnet_id": subnet.get("SubnetId"),
            "vpc_id": subnet.get("VpcId"),
            "cidr_block": subnet.get("CidrBlock"),
            "availability_zone": subnet.get("AvailabilityZone"),
            "availability_zone_id": subnet.get("AvailabilityZoneId"),
            "state": subnet.get("State"),
            "available_ip_count": subnet.get("AvailableIpAddressCount"),
            "default_for_az": subnet.get("DefaultForAz"),
            "map_public_ip_on_launch": subnet.get(
                "MapPublicIpOnLaunch"
            ),
            "name": tags.get("Name"),
            "tags": tags,
        })

    return {
        "count": len(subnets),
        "subnets": subnets,
    }


# ============================================================
# INTERNET GATEWAYS
# ============================================================

def get_internet_gateways(session_id: str):
    """
    Return Internet Gateway inventory and VPC attachments.
    """
    ec2 = _client(session_id, "ec2")

    response = ec2.describe_internet_gateways()

    gateways = []

    for gateway in response.get("InternetGateways", []):

        tags = _tag_dict(gateway.get("Tags", []))

        vpc_ids = [
            attachment.get("VpcId")
            for attachment in gateway.get("Attachments", [])
            if attachment.get("VpcId")
        ]

        gateways.append({
            "internet_gateway_id": gateway.get(
                "InternetGatewayId"
            ),
            "state": (
                gateway.get("Attachments", [{}])[0].get("State")
                if gateway.get("Attachments")
                else None
            ),
            "vpc_ids": vpc_ids,
            "attachments": gateway.get("Attachments", []),
            "name": tags.get("Name"),
            "tags": tags,
        })

    return {
        "count": len(gateways),
        "internet_gateways": gateways,
    }


# ============================================================
# ROUTE TABLES
# ============================================================

def get_route_tables(session_id: str):
    """
    Return route tables, routes and subnet associations.
    """
    ec2 = _client(session_id, "ec2")

    response = ec2.describe_route_tables()

    route_tables = []

    for table in response.get("RouteTables", []):

        tags = _tag_dict(table.get("Tags", []))

        associations = []

        for association in table.get("Associations", []):
            associations.append({
                "association_id": association.get(
                    "RouteTableAssociationId"
                ),
                "subnet_id": association.get("SubnetId"),
                "main": association.get("Main"),
                "association_state": association.get(
                    "AssociationState", {}
                ),
            })

        route_tables.append({
            "route_table_id": table.get("RouteTableId"),
            "vpc_id": table.get("VpcId"),
            "routes": table.get("Routes", []),
            "associations": associations,
            "name": tags.get("Name"),
            "tags": tags,
        })

    return {
        "count": len(route_tables),
        "route_tables": route_tables,
    }


# ============================================================
# SECURITY GROUPS
# ============================================================

def get_security_groups(session_id: str):
    """
    Return detailed security group information.
    """
    ec2 = _client(session_id, "ec2")

    response = ec2.describe_security_groups()

    groups = []

    for group in response.get("SecurityGroups", []):

        groups.append({
            "group_id": group.get("GroupId"),
            "group_name": group.get("GroupName"),
            "description": group.get("Description"),
            "vpc_id": group.get("VpcId"),
            "inbound_rules": group.get("IpPermissions", []),
            "outbound_rules": group.get(
                "IpPermissionsEgress",
                [],
            ),
            "tags": _tag_dict(group.get("Tags", [])),
        })

    return {
        "count": len(groups),
        "security_groups": groups,
    }


# ============================================================
# COST SUMMARY
# ============================================================

def get_cost_summary(session_id: str):
    """
    Return AWS cost for the current month to date.
    """
    ce = _client(session_id, "ce")

    today = date.today()
    month_start = today.replace(day=1)

    response = ce.get_cost_and_usage(
        TimePeriod={
            "Start": month_start.isoformat(),
            "End": today.isoformat(),
        },
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
    )

    results = []

    for period in response.get("ResultsByTime", []):
        amount = (
            period.get("Total", {})
            .get("UnblendedCost", {})
            .get("Amount", "0")
        )

        results.append({
            "start": period.get("TimePeriod", {}).get("Start"),
            "end": period.get("TimePeriod", {}).get("End"),
            "amount": float(amount),
            "unit": (
                period.get("Total", {})
                .get("UnblendedCost", {})
                .get("Unit")
            ),
        })

    return {
        "currency": "USD",
        "period_start": month_start.isoformat(),
        "period_end": today.isoformat(),
        "results": results,
    }


# ============================================================
# COST BY SERVICE
# ============================================================

def get_cost_by_service(session_id: str):
    """
    Return current-month AWS cost grouped by service.
    """
    ce = _client(session_id, "ce")

    today = date.today()
    month_start = today.replace(day=1)

    response = ce.get_cost_and_usage(
        TimePeriod={
            "Start": month_start.isoformat(),
            "End": today.isoformat(),
        },
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
        GroupBy=[
            {
                "Type": "DIMENSION",
                "Key": "SERVICE",
            }
        ],
    )

    services = []

    for period in response.get("ResultsByTime", []):

        for group in period.get("Groups", []):

            keys = group.get("Keys", [])

            service_name = (
                keys[0]
                if keys
                else "Unknown"
            )

            amount = (
                group.get("Metrics", {})
                .get("UnblendedCost", {})
                .get("Amount", "0")
            )

            services.append({
                "service": service_name,
                "amount": float(amount),
            })

    services.sort(
        key=lambda item: item["amount"],
        reverse=True,
    )

    return {
        "period_start": month_start.isoformat(),
        "period_end": today.isoformat(),
        "services": services,
    }


# ============================================================
# PATCH STATUS
# ============================================================

def get_patch_status(session_id: str):
    """
    Return Systems Manager patch compliance information.
    """
    ssm = _client(session_id, "ssm")

    response = ssm.describe_instance_information()

    instances = []

    for item in response.get("InstanceInformationList", []):

        instance_id = item.get("InstanceId")

        if not instance_id:
            continue

        try:
            patch_response = ssm.describe_instance_patches(
                InstanceId=instance_id
            )

            patches = patch_response.get(
                "Patches",
                [],
            )

            missing = sum(
                1
                for patch in patches
                if patch.get("State") == "Missing"
            )

            installed = sum(
                1
                for patch in patches
                if patch.get("State") == "Installed"
            )

            failed = sum(
                1
                for patch in patches
                if patch.get("State") == "Failed"
            )

            instances.append({
                "instance_id": instance_id,
                "ping_status": item.get("PingStatus"),
                "platform": item.get("PlatformName"),
                "platform_version": item.get(
                    "PlatformVersion"
                ),
                "installed": installed,
                "missing": missing,
                "failed": failed,
            })

        except Exception as exc:
            instances.append({
                "instance_id": instance_id,
                "error": str(exc),
            })

    return {
        "count": len(instances),
        "instances": instances,
    }


# ============================================================
# LAMBDA
# ============================================================

def get_lambda_functions(session_id: str):
    """
    Return Lambda function inventory.
    """
    lambda_client = _client(session_id, "lambda")

    paginator = lambda_client.get_paginator(
        "list_functions"
    )

    functions = []

    for page in paginator.paginate():

        for function in page.get("Functions", []):

            functions.append({
                "name": function.get("FunctionName"),
                "arn": function.get("FunctionArn"),
                "runtime": function.get("Runtime"),
                "handler": function.get("Handler"),
                "memory_size": function.get("MemorySize"),
                "timeout": function.get("Timeout"),
                "code_size": function.get("CodeSize"),
                "last_modified": function.get("LastModified"),
                "description": function.get("Description"),
                "state": function.get("State"),
                "last_update_status": function.get(
                    "LastUpdateStatus"
                ),
            })

    return {
        "count": len(functions),
        "functions": functions,
    }


# ============================================================
# CLOUDWATCH METRICS
# ============================================================

def get_cloudwatch_metrics(
    session_id: str,
    instance_id: str | None = None,
):
    """
    Retrieve EC2 CPU utilization for the previous 24 hours.

    If instance_id is provided, retrieve metrics for that instance.
    Otherwise retrieve metrics for all EC2 instances.
    """
    cloudwatch = _client(session_id, "cloudwatch")
    ec2 = _client(session_id, "ec2")

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=24)

    if instance_id:
        instance_ids = [instance_id]
    else:
        ec2_response = ec2.describe_instances()

        instance_ids = []

        for reservation in ec2_response.get(
            "Reservations",
            [],
        ):
            for instance in reservation.get(
                "Instances",
                [],
            ):
                current_id = instance.get("InstanceId")

                if current_id:
                    instance_ids.append(current_id)

    queries = []

    for current_id in instance_ids:

        queries.append({
            "Id": (
                "cpu_"
                + current_id.replace("-", "_")
            ),
            "MetricStat": {
                "Metric": {
                    "Namespace": "AWS/EC2",
                    "MetricName": "CPUUtilization",
                    "Dimensions": [
                        {
                            "Name": "InstanceId",
                            "Value": current_id,
                        }
                    ],
                },
                "Period": 300,
                "Stat": "Average",
            },
            "ReturnData": True,
        })

    if not queries:
        return {
            "period_hours": 24,
            "metrics": [],
        }

    results = []

    for start in range(0, len(queries), 500):

        batch = queries[start:start + 500]

        response = cloudwatch.get_metric_data(
            MetricDataQueries=batch,
            StartTime=start_time,
            EndTime=end_time,
            ScanBy="TimestampDescending",
        )

        results.extend(
            response.get(
                "MetricDataResults",
                [],
            )
        )

    metrics = []

    for result in results:

        metrics.append({
            "id": result.get("Id"),
            "label": result.get("Label"),
            "timestamps": [
                str(timestamp)
                for timestamp in result.get(
                    "Timestamps",
                    [],
                )
            ],
            "values": result.get(
                "Values",
                [],
            ),
        })

    return {
        "period_hours": 24,
        "start_time": str(start_time),
        "end_time": str(end_time),
        "metrics": metrics,
    }


# ============================================================
# CLOUDTRAIL EVENTS
# ============================================================

def get_cloudtrail_events(session_id: str):
    """
    Return recent CloudTrail management events from the
    previous hour.
    """
    cloudtrail = _client(session_id, "cloudtrail")

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=1)

    events = []

    paginator = cloudtrail.get_paginator(
        "lookup_events"
    )

    try:
        for page in paginator.paginate(
            LookupAttributes=[],
            StartTime=start_time,
            EndTime=end_time,
            PaginationConfig={
                "MaxItems": 50,
            },
        ):
            for event in page.get(
                "Events",
                [],
            ):

                events.append({
                    "event_id": event.get("EventId"),
                    "event_name": event.get("EventName"),
                    "event_time": str(
                        event.get("EventTime")
                    )
                    if event.get("EventTime")
                    else None,
                    "username": event.get("Username"),
                    "resources": event.get(
                        "Resources",
                        [],
                    ),
                    "cloudtrail_event": event.get(
                        "CloudTrailEvent"
                    ),
                })

                if len(events) >= 50:
                    break

            if len(events) >= 50:
                break

    except Exception as exc:
        return {
            "period_hours": 1,
            "events": [],
            "error": str(exc),
        }

    return {
        "period_hours": 1,
        "count": len(events),
        "events": events,
    }


# ============================================================
# INSPECTOR FINDINGS
# ============================================================

def get_inspector_findings(session_id: str):
    """
    Return critical and high severity Amazon Inspector findings.
    """
    inspector = _client(session_id, "inspector2")

    findings = []

    try:
        paginator = inspector.get_paginator(
            "list_findings"
        )

        for page in paginator.paginate(
            filterCriteria={
                "severity": [
                    {
                        "comparison": "EQUALS",
                        "value": "CRITICAL",
                    },
                    {
                        "comparison": "EQUALS",
                        "value": "HIGH",
                    },
                ]
            }
        ):
            finding_ids = page.get(
                "findingIds",
                [],
            )

            if finding_ids:
                details = inspector.batch_get_findings(
                    findingArns=finding_ids
                )

                findings.extend(
                    details.get(
                        "findings",
                        [],
                    )
                )

    except Exception as exc:
        return {
            "count": 0,
            "findings": [],
            "error": str(exc),
        }

    return {
        "count": len(findings),
        "findings": findings,
    }


# ============================================================
# RESOURCE TAGS
# ============================================================

def get_resource_tags(session_id: str):
    """
    Retrieve resources and their tags using the AWS Resource
    Groups Tagging API.
    """
    tagging = _client(
        session_id,
        "resourcegroupstaggingapi",
    )

    resources = []

    paginator = tagging.get_paginator(
        "get_resources"
    )

    for page in paginator.paginate():

        for resource in page.get(
            "ResourceTagMappingList",
            [],
        ):

            resources.append({
                "resource_arn": resource.get(
                    "ResourceARN"
                ),
                "tags": _tag_dict(
                    resource.get(
                        "Tags",
                        [],
                    )
                ),
            })

    return {
        "count": len(resources),
        "resources": resources,
    }


# ============================================================
# EC2 TAGS
# ============================================================

def get_ec2_tags(session_id: str):
    """
    Retrieve EC2 instances and their tags.
    """
    ec2 = _client(session_id, "ec2")

    response = ec2.describe_instances()

    instances = []

    for reservation in response.get(
        "Reservations",
        [],
    ):
        for instance in reservation.get(
            "Instances",
            [],
        ):

            instances.append({
                "instance_id": instance.get(
                    "InstanceId"
                ),
                "tags": _tag_dict(
                    instance.get(
                        "Tags",
                        [],
                    )
                ),
            })

    return {
        "count": len(instances),
        "instances": instances,
    }


# ============================================================
# S3 TAGS
# ============================================================

def get_s3_tags(session_id: str):
    """
    Retrieve tags for all S3 buckets.
    """
    s3 = _client(session_id, "s3")

    response = s3.list_buckets()

    buckets = []

    for bucket in response.get(
        "Buckets",
        [],
    ):

        bucket_name = bucket.get("Name")

        if not bucket_name:
            continue

        try:
            tag_response = s3.get_bucket_tagging(
                Bucket=bucket_name
            )

            tags = _tag_dict(
                tag_response.get(
                    "TagSet",
                    [],
                )
            )

        except Exception:
            tags = {}

        buckets.append({
            "bucket": bucket_name,
            "tags": tags,
        })

    return {
        "count": len(buckets),
        "buckets": buckets,
    }


# ============================================================
# LAMBDA TAGS
# ============================================================

def get_lambda_tags(session_id: str):
    """
    Retrieve tags for all Lambda functions.
    """
    lambda_client = _client(
        session_id,
        "lambda",
    )

    paginator = lambda_client.get_paginator(
        "list_functions"
    )

    functions = []

    for page in paginator.paginate():

        for function in page.get(
            "Functions",
            [],
        ):

            function_arn = function.get(
                "FunctionArn"
            )

            if not function_arn:
                continue

            try:
                response = lambda_client.list_tags(
                    Resource=function_arn
                )

                tags = response.get(
                    "Tags",
                    {},
                )

            except Exception:
                tags = {}

            functions.append({
                "function_name": function.get(
                    "FunctionName"
                ),
                "function_arn": function_arn,
                "tags": tags,
            })

    return {
        "count": len(functions),
        "functions": functions,
    }

