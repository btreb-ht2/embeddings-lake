import os
import logging
from boto3 import client


logger = logging.getLogger()
#logging.basicConfig(level=logging.DEBUG)
logger.setLevel(level=logging.INFO)

TABLE_NAME = os.environ['TABLE_NAME']

dynamodb_client = client("dynamodb")

def update_entry_point(lake_name, segment_index, entry_point):

    item = {
        'lakeName': {'S': lake_name},
        'shardIndex': {'N': str(segment_index)},
        'entryPoint': {'N': str(entry_point)}
    }

    result = dynamodb_client.put_item(
        TableName=TABLE_NAME,
        Item=item
    )

    return result

def get_entry_point(lake_name, segment_index,):
    
    item = {
        'lakeName': {'S': lake_name},
        'shardIndex': {'N': str(segment_index)},
    }

    result = dynamodb_client.get_item(
        TableName=TABLE_NAME,
        Key=item
    )

    logger.info(result)

    return result


def lambda_handler(event, context):

    logger.info(event)

    lake_name = event['Payload']['lakeName']
    segment_index = event['Payload']['segmentIndex']

    entry_point = 5

    result = get_entry_point(lake_name, segment_index)

    logger.info(result)
