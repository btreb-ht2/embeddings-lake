from aws_cdk import (
    # Duration,
    Stack,
    aws_s3 as s3,
)
from constructs import Construct

class EmbeddingsLakeStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        s3.Bucket(
            self, "EmbeddingsBucket",
        )
        # The code that defines your stack goes here

        # example resource
        # queue = sqs.Queue(
        #     self, "EmbeddingsLakeQueue",
        #     visibility_timeout=Duration.seconds(300),
        # )
