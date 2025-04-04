import logging
import os
from boto3 import client
import re

logger = logging.getLogger()
logger.setLevel(level=logging.ERROR)

BUCKET_NAME = os.environ['BUCKET_NAME']

s3_client = client("s3")


def get_segments(lake_name):
    segments_in_bucket = []
    prefix = lake_name + "/"
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    # TODO: get second batch of second 1,000 objects
    logger.info("response")
    logger.info(response)
    for obj in response['Contents']:
        obj_string = obj['Key']
        #logger.info("obj_string")
        #logger.info(obj_string)
        match = re.search(r"-(\d+)\.", obj_string)
        if match:
            number = int(match.group(1))
            #logger.info("number")
            #logger.info(number)            
            segments_in_bucket.append(number)
    logger.info("segments_in_bucket presort")
    logger.info(segments_in_bucket)       
    segments_in_bucket.sort()
    logger.info(segments_in_bucket)
    return segments_in_bucket
        

def get_adjacent_segments(lake_name, segment_value, num_shards, radius, segments_in_bucket):
    hash_index = segments_in_bucket.index(segment_value)
    segment_indices = []
    for delta in range(-radius, radius+1):
        logger.info(f"delta: {delta}")
        candidate_index = delta+hash_index
        #logger.info(f"candidate_index: {candidate_index}")
        try:
            candidate_segment = segments_in_bucket[candidate_index]
            candidate_key = f"{lake_name}/segment-{candidate_segment}.parquet"
            logger.info(candidate_key)
        except IndexError:
            logger.info("No adjacent segment available.")          
        try:
            response = s3_client.head_object(Bucket=BUCKET_NAME, Key=candidate_key)
            segment_indices.append(candidate_segment)
        except Exception:
            logger.info(f"Fragment {candidate_key} does not exist in S3.")
    return list(set(segment_indices))


def lambda_handler(event, context):

    logger.debug(event)

    #n_results: int = 4,
    radius = event['Payload']['radius']
    lake_name = event['Payload']['lake_name']
    embedding = event['Payload']['embedding']
    segment_index = event['Payload']['segment_index']
    num_shards = event["Payload"]['num_shards']

    segments_in_bucket = get_segments(lake_name=lake_name)

    segment_indices_to_search = get_adjacent_segments(lake_name, segment_index, num_shards, radius, segments_in_bucket)

    segments_as_strings = list(map(str, segment_indices_to_search))

    results = { 
        'segmentIndices': segments_as_strings,
        'embedding': embedding,
        'lakeName': lake_name,
    }

    logger.info(results)

    return results