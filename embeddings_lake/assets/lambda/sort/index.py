import os
import logging


logger = logging.getLogger()
#logging.basicConfig(level=logging.DEBUG)
logger.setLevel(level=logging.INFO)

BUCKET_NAME = os.environ['BUCKET_NAME']





def lambda_handler(event, context):

    logger.info(event)
    logger.info(len(event))
    # Event Handler

    all_results = []
    
    for sub_event in event:
        logger.debug(sub_event)
        logger.debug(sub_event['Payload'])
        logger.debug("here01")
        for sub_sub_event in sub_event['Payload']:
            logger.debug("here02")
            logger.debug(sub_sub_event)
            all_results.append(sub_sub_event)

    logger.debug(all_results)

    # sort in ascending order
    sorted_data_desc = sorted(all_results, key=lambda x: x['distance'])

    logger.info(sorted_data_desc)

    logger.debug(sorted_data_desc[0])

    return sorted_data_desc[0]