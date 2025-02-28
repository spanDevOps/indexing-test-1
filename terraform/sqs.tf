# SQS Queues for face processing
resource "aws_sqs_queue" "indexing" {
  name                      = "dev-indexing"
  delay_seconds             = 0
  max_message_size         = 262144
  message_retention_seconds = 345600  # 4 days
  visibility_timeout_seconds = 1800    # 30 minutes
  receive_wait_time_seconds = 20      # Enable long polling
}

resource "aws_sqs_queue" "group_face" {
  name                      = "dev-group-face"
  delay_seconds             = 0
  max_message_size         = 262144
  message_retention_seconds = 345600
  visibility_timeout_seconds = 1800
  receive_wait_time_seconds = 20
}

# SQS Queue Policy
resource "aws_sqs_queue_policy" "indexing" {
  queue_url = aws_sqs_queue.indexing.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          AWS = [
            "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/face-recognition-batch-job",
            data.aws_caller_identity.current.arn
          ]
        }
        Action = [
          "sqs:SendMessage",
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes"
        ]
        Resource = aws_sqs_queue.indexing.arn
      }
    ]
  })
}

resource "aws_sqs_queue_policy" "group_face" {
  queue_url = aws_sqs_queue.group_face.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          AWS = [
            "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/face-recognition-batch-job",
            data.aws_caller_identity.current.arn
          ]
        }
        Action = [
          "sqs:SendMessage",
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes"
        ]
        Resource = aws_sqs_queue.group_face.arn
      }
    ]
  })
}
