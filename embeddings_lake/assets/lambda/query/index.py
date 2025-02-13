import os
from json import dump
from boto3 import client as boto3_client


BUCKET_NAME = os.environ['BUCKET_NAME']



def lambda_handler(event, context):

    print("hello world!")