from aws_cdk import (
    # Duration,
    Stack,
    aws_s3 as s3,
    aws_lambda as lambda_,
    aws_stepfunctions_tasks as tasks,
    aws_stepfunctions as sfn,
    aws_apigateway as apigateway,
    aws_iam as iam,
    BundlingOptions
)
from constructs import Construct

class EmbeddingsLakeStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)


        lambda_endpoint = "lambda.amazonaws.com"

        bucket_segments = s3.Bucket(
            self, 
            "BucketSegments",
            encryption=s3.BucketEncryption(s3.BucketEncryption.S3_MANAGED),
            object_ownership=s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
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

        policy_lake_instantiate = iam.ManagedPolicy(
            self,
            "PolicyLambdaLakeInstantiation",
            managed_policy_name="EmbeddingsLake_LambdaLakeInstation",
            document=iam.PolicyDocument(
                statements=[
                    iam.PolicyStatement(
                        effect=iam.Effect.ALLOW,
                        actions=[
                            "s3:PutObject"
                        ],
                        resources=[
                            bucket_segments.bucket_arn,
                            bucket_segments.arn_for_objects("*"),
                        ],
                    )
                ]
            )
        )

        policy_embedding_add = iam.ManagedPolicy(
            self,
            "PolicyLambdaEmbeddingAdd",
            managed_policy_name="EmbeddingsLake_LambdaEmbeddingAdd",
            document=iam.PolicyDocument(
                statements=[
                    iam.PolicyStatement(
                        effect=iam.Effect.ALLOW,
                        actions=[
                            "s3:PutObject"
                        ],
                        resources=[
                            bucket_segments.bucket_arn,
                            bucket_segments.arn_for_objects("*"),
                        ],
                    )
                ]
            )
        )

        policy_embedding_hash = iam.ManagedPolicy(
            self,
            "PolicyLambdaEmbeddingHash",
            managed_policy_name="EmbeddingsLake_LambdaEmbeddingHash",
            document=iam.PolicyDocument(
                statements=[
                    iam.PolicyStatement(
                        effect=iam.Effect.ALLOW,
                        actions=[
                            "s3:GetObject",
                            "s3:ListBucket"
                        ],
                        resources=[
                            bucket_segments.bucket_arn,
                            bucket_segments.arn_for_objects("*"),
                        ],
                    )
                ]
            )
        )


        role_lambda_lake_instantiate = iam.Role(
            self,
            "RoleLambdaLakeInstantiation",
            assumed_by=iam.ServicePrincipal(lambda_endpoint),
            role_name="EmbeddingsLake_Role_lambda_Lake_Instantiation",
            managed_policies=[
                policy_lake_instantiate,
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
                ]
            
        )

        role_lambda_embedding_hash = iam.Role(
            self,
            "RoleLambdaEmbeddingHash",
            assumed_by=iam.ServicePrincipal(lambda_endpoint),
            role_name="EmbeddingsLake_Role_lambda_Embedding_Hash",
            managed_policies=[
                policy_embedding_hash,
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
                ]
            
        )

        role_lambda_embedding_add = iam.Role(
            self,
            "RoleLambdaEmbeddingAdd",
            assumed_by=iam.ServicePrincipal(lambda_endpoint),
            role_name="EmbeddingsLake_Role_lambda_Embedding_Add",
            managed_policies=[
                policy_embedding_add,
                ]
            
        )

        lambda_lake_instantiate = lambda_.Function(
            self,
            "FunctionInstantiateLake",
            runtime=lambda_.Runtime.PYTHON_3_10,
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("embeddings_lake/assets/lambda/laker"),
            environment={"BUCKET_NAME": bucket_segments.bucket_name },
            layers=[lambda_layer_pandas, lambda_layer_pydantic],
            role=role_lambda_lake_instantiate
        )

        lambda_embedding_hash = lambda_.Function(
            self,
            "FunctionEmbeddingHash",
            runtime=lambda_.Runtime.PYTHON_3_10,
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("embeddings_lake/assets/lambda/hasher"),
            environment={"BUCKET_NAME": bucket_segments.bucket_name },
            layers=[lambda_layer_pandas, lambda_layer_pydantic],
            role=role_lambda_embedding_hash
        )

        lambda_embedding_add = lambda_.Function(
            self,
            "FunctionEmbeddingAdd",
            runtime=lambda_.Runtime.PYTHON_3_10,
            memory_size=256,
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("embeddings_lake/assets/lambda/adder"),
            environment={"BUCKET_NAME": bucket_segments.bucket_name },
            layers=[lambda_layer_pandas, lambda_layer_pydantic],
            role=role_lambda_embedding_add
        )

        lambda_embedding_adjacent = lambda_.Function(
            self,
            "FunctionEmbeddingAdjacent",
            runtime=lambda_.Runtime.PYTHON_3_10,
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("embeddings_lake/assets/lambda/adjacent"),
            environment={"BUCKET_NAME": bucket_segments.bucket_name },
            layers=[lambda_layer_pandas, lambda_layer_pydantic]
        )

        lambda_embedding_query = lambda_.Function(
            self,
            "FunctionEmbeddingQuery",
            runtime=lambda_.Runtime.PYTHON_3_10,
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("embeddings_lake/assets/lambda/query"),
            environment={"BUCKET_NAME": bucket_segments.bucket_name },
            layers=[lambda_layer_pandas, lambda_layer_pydantic]
        )

        task_embedding_hash = tasks.LambdaInvoke(
            self,
            "Hash Embedding",
            lambda_function=lambda_embedding_hash
        )

        task_embedding_add = tasks.LambdaInvoke(
            self,
            "Add Embedding",
            lambda_function=lambda_embedding_add,
        )

        task_embedding_adjacents = tasks.LambdaInvoke(
            self,
            "Get Adjacent Segments", 
            lambda_function=lambda_embedding_adjacent
        )

        task_embedding_query = tasks.LambdaInvoke(
            self,
            "Search Segment",
            lambda_function=lambda_embedding_query
        )

        choice_embedding = sfn.Choice(
            self,
            "Embedding Choice"
        )

        choice_embedding.when(
            condition=sfn.Condition.boolean_equals(variable="$.Payload.add", value=True),
            next=task_embedding_add
        )

        choice_embedding.otherwise(
            task_embedding_adjacents
        )      

        task_embedding_hash.next(choice_embedding)

        map_search_segments = sfn.Map(self, "Query Segments",
            max_concurrency=10,
            #items_path=sfn.JsonPath.string_at("$$.Payload.segment_indices"),
            #items_path=sfn.JsonPath.array
            items_path="$.Payload.segmentIndices",
            #result_path="$.mapOutput",
        )

        map_search_segments.item_processor(task_embedding_query)

        task_embedding_adjacents.next(map_search_segments)

        state_machine_embedding = sfn.StateMachine(
            self,
            "Embeddings Lake State Machine",
            definition=task_embedding_hash
        )

        api = apigateway.RestApi(
            self,
            "API Gateway",
            deploy_options=apigateway.StageOptions(
                stage_name="prod"
            )
        )

        api_integration_response_list = [
            apigateway.IntegrationResponse(status_code="200"),
            apigateway.IntegrationResponse(status_code="400"),
            apigateway.IntegrationResponse(status_code="500"),
        ]

        api_method_response_list = [
            apigateway.MethodResponse(status_code="200"),
            apigateway.MethodResponse(status_code="400"),
            apigateway.MethodResponse(status_code="500"),            
        ]


        api_resource_lake = api.root.add_resource("lake")

        api_resource_lake_embedding = api_resource_lake.add_resource("embedding")

        api_resource_lake.add_method(
            http_method="PUT",
            integration = apigateway.LambdaIntegration(
                    handler = lambda_lake_instantiate,
                    integration_responses=api_integration_response_list,
                    proxy=False
                    ),
            method_responses=api_method_response_list
        )

        api_resource_lake_embedding.add_method(
            http_method="PUT",
            integration = apigateway.LambdaIntegration(
                    handler = lambda_embedding_hash,
                    integration_responses=api_integration_response_list,
                    proxy=False
                    ),
            method_responses=api_method_response_list
        )