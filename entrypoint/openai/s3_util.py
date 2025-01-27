import logging
import boto3
from botocore.exceptions import ClientError

from entrypoint.openai.log import setup_logging

logger = setup_logging()

def get_s3_client(config):
    s3_client = boto3.client('s3', aws_access_key_id = config['aws_access_key_id'], aws_secret_access_key = config['aws_secret_access_key'])
    return s3_client

def upload_fileobj(s3_client, file, bucket, object_name):
    try:
        response = s3_client.upload_fileobj(file, bucket, object_name)
        logging.info(response)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def upload_file_and_get_presigned_url(s3_client, bucket, object_name, file):
    try:
        # Upload the file
        upload_fileobj(s3_client, file, bucket, object_name)
        logger.info(f"File {object_name} uploaded to {bucket}/{object_name}")
        response_url = create_presigned_url(s3_client, bucket, object_name)
        if response_url is not None:
            logger.info(f"Presigned URL: {response_url}")
            return response_url
        else:
            logger.info(f"Presigned URL failed")
            return None
    except Exception as e:
        logger.error(f"Error uploading file {object_name}: {e}")
        return None   
 # Generate a presigned URL for the S3 object
def create_presigned_url(s3_client, bucket_name, object_name, expiration=3600):
    try:
        response_url = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except ClientError as e:
        logging.error(e)
        return None
    # The response contains the presigned URL
    return response_url
