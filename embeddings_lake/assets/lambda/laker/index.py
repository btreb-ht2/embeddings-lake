import os
import tempfile
from json import dump
from boto3 import client as boto3_client


s3_client = boto3_client("s3")

BUCKET_NAME = os.environ['BUCKET_NAME']



def lambda_handler(event, context):



    file_name = 'lake_config.json' 

    print(event)
    print(event['lake_name'])
    print(event['lake_dimensions'])
    print(event['lake_shards'])
    print("hello world!")


    data = {"lake_name": event['lake_name'],
            "lake_dimensions": event['lake_dimensions'],
            "lake_shards": event['lake_shards']
            }

    with tempfile.NamedTemporaryFile(mode='w') as temporary_file:
        dump(data, temporary_file, indent=4)
        temporary_file.flush()

        upload_file_response = s3_client.upload_file(
            Filename=temporary_file.name, 
            Bucket=BUCKET_NAME, 
            Key=f"{event['lake_name']}/{file_name}"
            ) 
        print(upload_file_response)
    
    return { 
        'statusCode': 200, 
        'body': 'File uploaded successfully.' 
    }