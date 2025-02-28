# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "batch_logs" {
  name              = "/aws/batch/face-recognition"
  retention_in_days = 14
}

# CloudWatch Dashboard for Face Recognition Service
resource "aws_cloudwatch_dashboard" "face_recognition" {
  dashboard_name = "face-recognition-monitoring"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          view    = "timeSeries"
          metrics = [
            ["AWS/Batch", "CPUUtilization", "JobQueue", aws_batch_job_queue.face_recognition_primary.name],
            [".", ".", ".", aws_batch_job_queue.face_recognition_secondary.name]
          ]
          region = data.aws_region.current.name
          title  = "CPU Utilization by Queue"
          period = 60
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          view    = "timeSeries"
          metrics = [
            ["AWS/Batch", "MemoryUtilization", "JobQueue", aws_batch_job_queue.face_recognition_primary.name],
            [".", ".", ".", aws_batch_job_queue.face_recognition_secondary.name]
          ]
          region = data.aws_region.current.name
          title  = "Memory Utilization by Queue"
          period = 60
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 8
        height = 6
        properties = {
          view    = "timeSeries"
          metrics = [
            ["AWS/Batch", "RunningJobs", "JobQueue", aws_batch_job_queue.face_recognition_primary.name],
            [".", ".", ".", aws_batch_job_queue.face_recognition_secondary.name]
          ]
          region = data.aws_region.current.name
          title  = "Running Jobs"
          period = 60
        }
      },
      {
        type   = "metric"
        x      = 8
        y      = 6
        width  = 8
        height = 6
        properties = {
          view    = "timeSeries"
          metrics = [
            ["AWS/Batch", "PendingJobs", "JobQueue", aws_batch_job_queue.face_recognition_primary.name],
            [".", ".", ".", aws_batch_job_queue.face_recognition_secondary.name]
          ]
          region = data.aws_region.current.name
          title  = "Pending Jobs"
          period = 60
        }
      },
      {
        type   = "metric"
        x      = 16
        y      = 6
        width  = 8
        height = 6
        properties = {
          view    = "timeSeries"
          metrics = [
            ["AWS/SQS", "ApproximateNumberOfMessagesVisible", "QueueName", aws_sqs_queue.indexing.name],
            [".", "ApproximateAgeOfOldestMessage", ".", "."]
          ]
          region = data.aws_region.current.name
          title  = "SQS Queue Metrics"
          period = 60
        }
      }
    ]
  })
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  alarm_name          = "face-recognition-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name        = "CPUUtilization"
  namespace          = "AWS/Batch"
  period             = 300
  statistic          = "Average"
  threshold          = 80
  alarm_description  = "CPU utilization is too high"
  alarm_actions      = []  # Add SNS topic ARN here if needed

  dimensions = {
    JobQueue = aws_batch_job_queue.face_recognition_primary.name
  }
}

resource "aws_cloudwatch_metric_alarm" "job_failures" {
  alarm_name          = "face-recognition-job-failures"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name        = "FailedJobs"
  namespace          = "AWS/Batch"
  period             = 300
  statistic          = "Sum"
  threshold          = 2
  alarm_description  = "Multiple job failures detected"
  alarm_actions      = []  # Add SNS topic ARN here if needed

  dimensions = {
    JobQueue = aws_batch_job_queue.face_recognition_primary.name
  }
}
