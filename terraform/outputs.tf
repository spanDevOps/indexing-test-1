output "primary_queue_arn" {
  description = "ARN of the primary batch job queue"
  value       = aws_batch_job_queue.face_recognition_primary.arn
}

output "secondary_queue_arn" {
  description = "ARN of the secondary batch job queue"
  value       = aws_batch_job_queue.face_recognition_secondary.arn
}

output "s3_bucket_processed" {
  description = "Name of the S3 bucket for processed images"
  value       = aws_s3_bucket.processed.id
}

output "sqs_indexing_queue_url" {
  description = "URL of the SQS queue for indexing"
  value       = aws_sqs_queue.indexing.url
}

output "cloudwatch_dashboard_url" {
  description = "URL of the CloudWatch dashboard"
  value       = "https://${data.aws_region.current.name}.console.aws.amazon.com/cloudwatch/home?region=${data.aws_region.current.name}#dashboards:name=${aws_cloudwatch_dashboard.face_recognition.dashboard_name}"
}
