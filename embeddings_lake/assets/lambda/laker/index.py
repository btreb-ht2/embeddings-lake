import os
from json import dump
from boto3 import client as boto3_client


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

    with open(f"/tmp/{file_name}", 'w') as file:
        dump(data, file, indent=4)

    s3 = boto3_client('s3') 
    s3.upload_file(Filename=f"/tmp/{file_name}", Bucket=BUCKET_NAME, Key=f"{event['lake_name']}/{file_name}") 
    return { 
        'statusCode': 200, 
        'body': 'File uploaded successfully.' 
    }