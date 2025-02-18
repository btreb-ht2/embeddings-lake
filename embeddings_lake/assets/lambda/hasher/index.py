from numpy import array as np_array, dot as np_dot, random as np_random
from math import log as math_log
from boto3 import resource as boto3_resource
import os
import json


s3_resource = boto3_resource("s3")


BUCKET_NAME = os.environ['BUCKET_NAME']


def lsh(vector, hyperplanes):
    """Hashes a vector using all hyperplanes.

    Returns a string of 0s and 1s.
    """
    return int(
        "".join(
            [
                "1" if np_dot(hyperplane, vector) > 0 else "0"
                for hyperplane in hyperplanes
            ]
        ),
        base=2,
    )    

def vector_router(vector: np_array, hyperplanes) -> int:
    if isinstance(vector, list):
        vector = np_array(vector)
    closest_index = lsh(vector, hyperplanes)
    return closest_index

def append_to_bucket(shard_index, embedding, metadata, document):
    
    uid = 123
    
    return uid

def lambda_handler(event, context):

    print(event)
    lake_name = event['lake_name']


    object = s3_resource.Object(BUCKET_NAME, f"{lake_name}/lake_config.json")
    object_contents = object.get()["Body"].read().decode("utf-8")
    lake_config = json.loads(object_contents)

    print(lake_config)

    dimension = lake_config["lake_dimensions"]
    hyperplanes = lake_config["lake_hyperplanes"]
    num_shards = lake_config["lake_shards"]

    embedding = event['embedding']

    metadata = {"id": "1"}
    document = "TODO placeholder for document"

    shard_index = vector_router(np_array(embedding), hyperplanes)


    # uid = append_to_bucket(
    #     shard_index,
    #     embedding, 
    #     metadata=metadata, 
    #     document=document
    #     )

    print(f"shard index: {shard_index}")
    return { 
        'statusCode': 200, 
        'body': 'Success',
        'embedding': embedding,
        'embedding_hash_index': shard_index,
        'num_shards': num_shards,
        'add': False
    }
