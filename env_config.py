import os


def bool_evaluator(val):
    return True if val.lower() == 'true' else False


TEMP_FILES_PATH = '/tmp/temp_files'

DEVICE = os.environ.get('DEVICE', 'cuda')

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_RESOURCE_BUCKET_NAME = os.getenv("S3_RESOURCE_BUCKET_NAME")

SQS_PAGE_QUEUE_URL = os.environ.get('SQS_PAGE_QUEUE_URL', '')
SQS_GPU_QUEUE_URL = os.environ.get('SQS_GPU_QUEUE_URL', '')

CB_MODEL_VERSION = os.environ.get('CB_MODEL_VERSION', '')
BOX_MODEL_VERSION = os.environ.get('BOX_MODEL_VERSION', '')

RETRIEVE_AWS_CONFIGS = bool_evaluator(os.environ.get('RETRIEVE_AWS_CONFIGS',
                                                     'true'))
