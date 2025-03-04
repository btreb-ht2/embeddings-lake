import logging
import os
from boto3 import client

logger = logging.getLogger()
logger.setLevel(level=logging.INFO)

BUCKET_NAME = os.environ['BUCKET_NAME']

s3_client = client("s3")


def get_adjacent_segments(lake_name, base_hash, num_shards, radius=1):
        # Added radius to include closest bucket as well
        segment_indices = []
        for delta in range(-radius, radius+1):
            candidate_segment = (base_hash + delta) % num_shards
            candidate_key = f"{lake_name}/segment-{candidate_segment}.parquet"
            try:
                response = s3_client.head_object(Bucket=BUCKET_NAME, Key=candidate_key)
                segment_indices.append(candidate_segment)
            except Exception:
                logger.info(f"Fragment {candidate_key} does not exist in S3")
        return segment_indices


def lambda_handler(event, context):

    logger.info(event)

    n_results: int = 4,
    radius: int = 1
    lake_name = event['Payload']['lake_name']
    embedding = event['Payload']['embedding']
    embedding_hash_index = event['Payload']['segment_index']
    num_shards = event["Payload"]['num_shards']


    segment_indices_to_search = get_adjacent_segments(lake_name, embedding_hash_index, num_shards, radius)

    segments_as_strings = list(map(str, segment_indices_to_search))

    results = { 
        'segmentIndices': segments_as_strings,
        'embedding': embedding,
        'lakeName': lake_name,
    }

    logger.info(results)

    return results