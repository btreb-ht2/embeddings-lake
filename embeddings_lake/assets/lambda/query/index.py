def get_adjacent_segments(base_hash, num_shards, radius=1):
        # Added radius to include closest bucket as well
        segment_indices = []
        for delta in range(-radius, radius+1):
            segment_indices.append((base_hash + delta) % num_shards)
        return segment_indices



def lambda_handler(event, context):

    print(event)

    n_results: int = 4,
    radius: int = 5
    embedding = event["Payload"]["embedding"]
    embedding_hash_index = event["Payload"]["embedding_hash_index"]
    num_shards = event["Payload"]["num_shards"]


    segment_indices_to_search = get_adjacent_segments(embedding_hash_index, num_shards, radius)


    return { 
        'statusCode': 200, 
        'body': 'Complete',
        'segment_indices': segment_indices_to_search,
        'embedding': embedding,
    }