from numpy import array as np_array, dot as np_dot, random as np_random
from math import log as math_log



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
    #TODO: get from s3 json file for lake_name
    dimension = 5
    #TODO: get from s3 json file for lake_name
    approx_shards = 100
    #TODO: get from s3 json file for lake_name
    num_hashes = 7
    #TODO: get from s3 json file for lake_name
    bucket_size = 1000
    #TODO: get from s3 json file for lake_name
    hyperplanes = 13

    num_shards = int(math_log(approx_shards, 2) + 0.5)

    embedding = event['embedding']
    metadata = {"id": "1"}
    document = "TODO placeholder for document"

    shard_index = vector_router(embedding, hyperplanes)


    uid = append_to_bucket(
        shard_index,
        embedding, 
        metadata=metadata, 
        document=document
        )




    print("hello world!")