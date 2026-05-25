import boto3
from datetime import datetime, timezone, timedelta

ec2 = boto3.client('ec2', region_name='ap-south-1')
cw = boto3.client('cloudwatch', region_name='ap-south-1')

INSTANCE = 'i-0082398a16c6183ce'
end = datetime.now(timezone.utc)
start = end - timedelta(minutes=20)

# Instance status checks
print("=== Instance Status Checks ===")
status = ec2.describe_instance_status(InstanceIds=[INSTANCE])
if status['InstanceStatuses']:
    ist = status['InstanceStatuses'][0]
    print("  System check:", ist['SystemStatus']['Status'])
    print("  Instance check:", ist['InstanceStatus']['Status'])
else:
    print("  No status data (may be warming up)")

# Memory - EC2 doesn't report memory by default without CloudWatch agent
# Check CPU and NetworkOut
for metric in ['CPUUtilization', 'NetworkIn', 'NetworkOut']:
    m = cw.get_metric_statistics(
        Namespace='AWS/EC2', MetricName=metric,
        Dimensions=[{'Name': 'InstanceId', 'Value': INSTANCE}],
        StartTime=start, EndTime=end, Period=300, Statistics=['Average']
    )
    pts = sorted(m['Datapoints'], key=lambda x: x['Timestamp'])
    if pts:
        latest = pts[-1]
        ts = latest['Timestamp'].strftime('%H:%M UTC')
        val = latest['Average']
        if metric == 'CPUUtilization':
            print(f"  CPU at {ts}: {val:.1f}%")
        else:
            print(f"  {metric} at {ts}: {val/1e6:.2f} MB/s avg")
    else:
        print(f"  {metric}: no data")

# Check all recent metrics
print("\n=== NetworkIn last 20 min (MB/s) ===")
n = cw.get_metric_statistics(
    Namespace='AWS/EC2', MetricName='NetworkIn',
    Dimensions=[{'Name': 'InstanceId', 'Value': INSTANCE}],
    StartTime=start, EndTime=end, Period=60, Statistics=['Average']
)
for pt in sorted(n['Datapoints'], key=lambda x: x['Timestamp']):
    ts = pt['Timestamp'].strftime('%H:%M')
    mb = pt['Average'] / 1e6
    bar = '#' * int(mb * 2)
    print(f"  {ts}: {mb:.2f} MB/s  {bar}")
