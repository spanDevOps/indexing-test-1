#!/usr/bin/env python3
import boto3
import time
import os
from datetime import datetime, timedelta
from tabulate import tabulate
from botocore.config import Config

def get_job_metrics(cloudwatch, queue_name):
    """Get real-time metrics for the job queue"""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=5)
    
    metrics = cloudwatch.get_metric_data(
        MetricDataQueries=[
            {
                'Id': 'cpu',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'AWS/Batch',
                        'MetricName': 'CPUUtilization',
                        'Dimensions': [{'Name': 'JobQueue', 'Value': queue_name}]
                    },
                    'Period': 60,
                    'Stat': 'Average'
                }
            },
            {
                'Id': 'memory',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'AWS/Batch',
                        'MetricName': 'MemoryUtilization',
                        'Dimensions': [{'Name': 'JobQueue', 'Value': queue_name}]
                    },
                    'Period': 60,
                    'Stat': 'Average'
                }
            }
        ],
        StartTime=start_time,
        EndTime=end_time
    )
    
    # Get latest values
    cpu = metrics['MetricDataResults'][0]['Values'][-1] if metrics['MetricDataResults'][0]['Values'] else 0
    memory = metrics['MetricDataResults'][1]['Values'][-1] if metrics['MetricDataResults'][1]['Values'] else 0
    
    return cpu, memory

def monitor_jobs():
    # Initialize clients
    batch = boto3.client('batch')
    cloudwatch = boto3.client('cloudwatch')
    
    # Get queue names from environment
    primary_queue = os.getenv('PRIMARY_QUEUE_NAME')
    secondary_queue = os.getenv('SECONDARY_QUEUE_NAME')
    
    while True:
        os.system('clear')
        print(f"Face Recognition Job Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)
        
        for queue in [primary_queue, secondary_queue]:
            print(f"\nQueue: {queue}")
            print("-" * 40)
            
            # Get job metrics
            cpu, memory = get_job_metrics(cloudwatch, queue)
            
            # Get job list
            jobs = batch.list_jobs(
                jobQueue=queue,
                filters=[{'name': 'RUNNING'}, {'name': 'SUBMITTED'}, {'name': 'PENDING'}]
            )['jobSummaryList']
            
            # Print metrics
            print(f"CPU Utilization: {cpu:.1f}%")
            print(f"Memory Utilization: {memory:.1f}%")
            print(f"Active Jobs: {len(jobs)}")
            
            if jobs:
                # Prepare job table
                job_data = []
                for job in jobs:
                    job_data.append([
                        job['jobId'][-8:],  # Short ID
                        job['status'],
                        job.get('container', {}).get('vcpus', '-'),
                        job.get('container', {}).get('memory', '-'),
                        job.get('startedAt', '-')
                    ])
                
                print("\nActive Jobs:")
                print(tabulate(
                    job_data,
                    headers=['Job ID', 'Status', 'vCPUs', 'Memory', 'Started At'],
                    tablefmt='grid'
                ))
        
        print("\nPress Ctrl+C to exit")
        time.sleep(10)  # Update every 10 seconds

if __name__ == '__main__':
    try:
        monitor_jobs()
    except KeyboardInterrupt:
        print("\nMonitoring stopped")
