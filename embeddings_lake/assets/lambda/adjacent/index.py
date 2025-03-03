import logging


logger = logging.getLogger()
logger.setLevel(level=logging.INFO)



def get_adjacent_segments(base_hash, num_shards, radius=1):
        # Added radius to include closest bucket as well
        segment_indices = []
        for delta in range(-radius, radius+1):
            segment_indices.append((base_hash + delta) % num_shards)
        return segment_indices



def lambda_handler(event, context):

    logger.info(event)

    n_results: int = 4,
    radius: int = 0
    lake_name = event['Payload']['lake_name']
    embedding = event['Payload']['embedding']
    embedding = event['Payload']['embedding']
    embedding_hash_index = event['Payload']['segment_index']
    num_shards = event["Payload"]['num_shards']


    segment_indices_to_search = get_adjacent_segments(embedding_hash_index, num_shards, radius)

    segments_as_strings = list(map(str, segment_indices_to_search))

    results = { 
        'segmentIndices': segments_as_strings,
        'embedding': embedding,
        'lakeName': lake_name,
    }

    logger.info(results)

    return results