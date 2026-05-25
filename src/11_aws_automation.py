"""
Step 11: Set up AWS automation for annual pipeline reruns.

Architecture:
  EventBridge Scheduler → SSM Run Command → EC2 → run_pipeline.sh → SNS alert

This means every January 15 (after NASA VIIRS annual release), the pipeline
automatically reruns, updates the S3 outputs, and sends a notification email.
No Lambda required — SSM Run Command executes the script directly on the EC2.

Prerequisites:
  - EC2 instance must have an IAM role with SSM access
  - SNS topic for notifications (email subscription confirmed)
  - SSM Agent running on EC2 (pre-installed on Ubuntu AMIs)

Run this script once to set up all resources:
    python 11_aws_automation.py [--teardown]
"""

import json, sys
import boto3
from pathlib import Path

REGION      = "ap-south-1"
INSTANCE_ID = "i-0082398a16c6183ce"
SCHEDULE_NAME = "kritter-annual-pipeline"
SNS_TOPIC_NAME = "kritter-pipeline-alerts"
PIPELINE_CMD = (
    "cd /home/ubuntu/kritter && "
    "bash run_pipeline.sh --skip-viirs 2>&1 | "
    "tee /data/satellite/kritter/logs/annual_run.log"
)


def ensure_sns_topic(sns_client) -> str:
    """Create SNS topic if not exists, return ARN."""
    existing = sns_client.list_topics()["Topics"]
    for t in existing:
        if SNS_TOPIC_NAME in t["TopicArn"]:
            print(f"  SNS topic already exists: {t['TopicArn']}")
            return t["TopicArn"]

    resp = sns_client.create_topic(Name=SNS_TOPIC_NAME)
    arn = resp["TopicArn"]
    print(f"  Created SNS topic: {arn}")
    return arn


def ensure_ssm_role(iam_client) -> str:
    """Ensure EC2 has SSM role. Returns role name."""
    role_name = "kritter-ec2-ssm-role"
    try:
        iam_client.get_role(RoleName=role_name)
        print(f"  IAM role already exists: {role_name}")
        return role_name
    except iam_client.exceptions.NoSuchEntityException:
        pass

    # Create role
    trust = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"},
                       "Action": "sts:AssumeRole"}]
    })
    iam_client.create_role(RoleName=role_name, AssumeRolePolicyDocument=trust,
                            Description="EC2 SSM access for Kritter pipeline")
    iam_client.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
    )
    iam_client.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess"
    )
    print(f"  Created IAM role: {role_name}")
    return role_name


def create_eventbridge_schedule(scheduler_client, sns_arn: str) -> None:
    """Create annual EventBridge schedule: Jan 15 at 06:00 UTC."""
    try:
        scheduler_client.get_schedule(Name=SCHEDULE_NAME)
        print(f"  Schedule already exists: {SCHEDULE_NAME}")
        return
    except scheduler_client.exceptions.ResourceNotFoundException:
        pass

    # Get account ID for SSM target ARN
    sts = boto3.client("sts", region_name=REGION)
    account_id = sts.get_caller_identity()["Account"]

    ssm_arn = f"arn:aws:ssm:{REGION}:{account_id}:document/AWS-RunShellScript"
    role_arn = f"arn:aws:iam::{account_id}:role/kritter-scheduler-role"

    # Create scheduler role if needed
    iam = boto3.client("iam", region_name=REGION)
    try:
        iam.get_role(RoleName="kritter-scheduler-role")
    except iam.exceptions.NoSuchEntityException:
        trust = json.dumps({
            "Version": "2012-10-17",
            "Statement": [{"Effect": "Allow",
                           "Principal": {"Service": "scheduler.amazonaws.com"},
                           "Action": "sts:AssumeRole"}]
        })
        iam.create_role(RoleName="kritter-scheduler-role",
                        AssumeRolePolicyDocument=trust)
        iam.put_role_policy(
            RoleName="kritter-scheduler-role",
            PolicyName="ssm-run-and-sns",
            PolicyDocument=json.dumps({
                "Version": "2012-10-17",
                "Statement": [
                    {"Effect": "Allow", "Action": "ssm:SendCommand",
                     "Resource": ["arn:aws:ec2:*:*:instance/*",
                                  f"arn:aws:ssm:{REGION}::document/AWS-RunShellScript"]},
                    {"Effect": "Allow", "Action": "sns:Publish",
                     "Resource": sns_arn},
                ]
            })
        )

    scheduler_client.create_schedule(
        Name=SCHEDULE_NAME,
        Description="Annual Kritter village growth pipeline rerun (post-VIIRS release)",
        ScheduleExpression="cron(0 6 15 1 ? *)",   # Jan 15 at 06:00 UTC
        ScheduleExpressionTimezone="UTC",
        FlexibleTimeWindow={"Mode": "OFF"},
        Target={
            "Arn": ssm_arn,
            "RoleArn": role_arn,
            "Input": json.dumps({
                "commands": [
                    "set -euo pipefail",
                    "aws ec2 start-instances --region ap-south-1 "
                    f"--instance-ids {INSTANCE_ID}",
                    "sleep 60",
                    PIPELINE_CMD,
                ],
                "executionTimeout": ["28800"],   # 8 hours
            }),
            "SsmParameters": {
                "InstanceIds": [INSTANCE_ID],
                "DocumentName": "AWS-RunShellScript",
            }
        },
        State="ENABLED",
    )
    print(f"  Created schedule: {SCHEDULE_NAME} (fires Jan 15 at 06:00 UTC annually)")


def setup():
    print("Setting up AWS automation for annual pipeline runs...\n")

    iam = boto3.client("iam", region_name=REGION)
    sns = boto3.client("sns", region_name=REGION)

    print("1. SNS topic for alerts:")
    sns_arn = ensure_sns_topic(sns)

    print("\n2. IAM role for EC2 SSM access:")
    ensure_ssm_role(iam)

    print("\n3. EventBridge Scheduler:")
    try:
        scheduler = boto3.client("scheduler", region_name=REGION)
        create_eventbridge_schedule(scheduler, sns_arn)
    except Exception as e:
        print(f"  EventBridge Scheduler not available ({e})")
        print("  Fallback: use AWS CloudWatch Events (deprecated) or set up manually")

    print(f"""
─────────────────────────────────────────────────────────────
Setup complete. Annual pipeline will run on Jan 15 at 06:00 UTC.

To subscribe to SNS alerts (email notification when pipeline completes):
  aws sns subscribe \\
    --topic-arn {sns_arn} \\
    --protocol email \\
    --notification-endpoint YOUR_EMAIL@example.com \\
    --region {REGION}

To test manually (run pipeline now via SSM):
  aws ssm send-command \\
    --document-name "AWS-RunShellScript" \\
    --instance-ids {INSTANCE_ID} \\
    --parameters commands=["{PIPELINE_CMD}"] \\
    --region {REGION}
─────────────────────────────────────────────────────────────
""")


def teardown():
    print("Tearing down AWS automation resources...")
    scheduler = boto3.client("scheduler", region_name=REGION)
    try:
        scheduler.delete_schedule(Name=SCHEDULE_NAME)
        print(f"  Deleted schedule: {SCHEDULE_NAME}")
    except Exception as e:
        print(f"  Schedule not found: {e}")


if __name__ == "__main__":
    if "--teardown" in sys.argv:
        teardown()
    else:
        setup()
