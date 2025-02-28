class MessageProcessingError(Exception):
    """Base exception for message processing errors"""
    pass

class RetryableError(MessageProcessingError):
    """Error that should trigger a retry"""
    pass

class NonRetryableError(MessageProcessingError):
    """Error that should not be retried"""
    pass

class S3DownloadError(RetryableError):
    """S3 download failed - should retry"""
    pass

class FileValidationError(NonRetryableError):
    """File validation failed - should not retry"""
    pass

class FaceProcessingError(RetryableError):
    """Face processing failed - should retry"""
    pass

class DatabaseError(RetryableError):
    """Database operation failed - should retry"""
    pass 