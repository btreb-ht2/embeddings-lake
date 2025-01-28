import logging
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    print("hello from print statement in lambda!")
    logger.info(json.dumps(event))