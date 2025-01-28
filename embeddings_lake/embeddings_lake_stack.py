from aws_cdk import (
    # Duration,
    Stack,
    aws_s3 as s3,
    aws_lambda as lambda_
)
from constructs import Construct

class EmbeddingsLakeStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        bucket = s3.Bucket(
            self, "EmbeddingsBucket",
        )
        # The code that defines your stack goes here

        # example resource
        # queue = sqs.Queue(
        #     self, "EmbeddingsLakeQueue",
        #     visibility_timeout=Duration.seconds(300),
        # )

        function = lambda_.Function(
            self,
            "EmbeddingsFunction",
            runtime=lambda_.Runtime.PYTHON_3_10,
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("embeddings_lake/assets/lambda/embedder")
        )
