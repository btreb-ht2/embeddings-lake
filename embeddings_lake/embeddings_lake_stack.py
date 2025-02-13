from aws_cdk import (
    # Duration,
    Stack,
    aws_s3 as s3,
    aws_lambda as lambda_,
    BundlingOptions
)
from constructs import Construct

class EmbeddingsLakeStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        bucket = s3.Bucket(
            self, "EmbeddingsBucket",
        )

        # layer_bundling_command = (
        #     "pip install -r requirements.txt "
        #     "-t /asset-output/python && "
        #     "find /asset-output/python -type d -name '__pycache__' -exec rm -rf {} + && "
        #     "cp -au . /asset-output/python"
        # )

        # lambda_layer_pydantic = lambda_.LayerVersion(
        #     self,
        #     "numpyLambdaLayer",
        #     compatible_runtimes=[lambda_.Runtime.PYTHON_3_10],
        #     code=lambda_.Code.from_asset(
        #         "embeddings_lake/assets/lambda/layers/pydantic",
        #         bundling=BundlingOptions(
        #             image=lambda_.Runtime.PYTHON_3_10.bundling_image,
        #             command=[
        #                 "bash",
        #                 "-c",
        #                 layer_bundling_command,
        #             ]
        #         )
        #     )
        # )

        # https://www.youtube.com/watch?v=jyuZDkiHe2Q
        lambda_layer_pydantic = lambda_.LayerVersion(
            self,
            "pydanticLambdaLayer",
            code=lambda_.Code.from_asset("embeddings_lake/assets/lambda/layers/pydantic/lambda-layer-pydantic.zip"),
        )

        lambda_layer_pandas = lambda_.LayerVersion.from_layer_version_arn(
            self,
            "pandasLambdaLayer",
            layer_version_arn="arn:aws:lambda:us-east-1:336392948345:layer:AWSSDKPandas-Python310:23"
        )

        function = lambda_.Function(
            self,
            "EmbeddingsFunction",
            runtime=lambda_.Runtime.PYTHON_3_10,
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("embeddings_lake/assets/lambda/embedder"),
            layers=[lambda_layer_pandas, lambda_layer_pydantic]
        )

        lambda_lake_instantiate = lambda_.Function(
            self,
            "FunctionInstantiateLake",
            runtime=lambda_.Runtime.PYTHON_3_10,
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("embeddings_lake/assets/lambda/laker"),
            environment={"BUCKET_NAME": bucket.bucket_name },
            layers=[lambda_layer_pandas, lambda_layer_pydantic]
        )

        lambda_embedding_hash = lambda_.Function(
            self,
            "FunctionHashVector",
            runtime=lambda_.Runtime.PYTHON_3_10,
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("embeddings_lake/assets/lambda/hasher"),
            environment={"BUCKET_NAME": bucket.bucket_name },
            layers=[lambda_layer_pandas, lambda_layer_pydantic]
        )

        lambda_embedding_add = lambda_.Function(
            self,
            "FunctionEmbeddingAdd",
            runtime=lambda_.Runtime.PYTHON_3_10,
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("embeddings_lake/assets/lambda/adder"),
            environment={"BUCKET_NAME": bucket.bucket_name },
            layers=[lambda_layer_pandas, lambda_layer_pydantic]
        )

        lambda_embedding_query = lambda_.Function(
            self,
            "FunctionEmbeddingQuery",
            runtime=lambda_.Runtime.PYTHON_3_10,
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("embeddings_lake/assets/lambda/query"),
            environment={"BUCKET_NAME": bucket.bucket_name },
            layers=[lambda_layer_pandas, lambda_layer_pydantic]
        )