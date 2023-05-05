import os

import boto3
import numpy as np
from PIL import Image

import env_config


def download_page_s3(file_path, file_name, file_id, temp_folder):
    s3 = boto3.resource('s3').Bucket(env_config.S3_BUCKET_NAME)
    with open(os.path.join(temp_folder, file_name), 'wb') as file_buffer:
        s3.download_fileobj(os.path.join(file_id, file_path), file_buffer)

    page = np.array(Image.open(os.path.join(temp_folder, file_name)))
    return page


def check_model_exists(model_type, file_version):
    model_files = os.listdir('model_weights')

    if model_files:
        if model_type == 'cb':
            model_file = 'cb_detection_weight'
        elif model_type == 'box':
            model_file = 'box_weight'
        else:
            return False
        current_file = [m for m in model_files if model_file in m]

        if not current_file:
            return False

        # Check if current version is same as required version
        current_file = current_file[0]
        current_file_name, _ = os.path.splitext(current_file)
        _, current_version = current_file_name.split('__')

        if current_version == file_version:
            return True

    return False


def download_model(model_type):
    """
    This function downloads model weights from
    S3 bucket.
    """
    # Create a directory to download models
    if not os.path.exists('model_weights'):
        os.mkdir('model_weights')

    s3 = boto3.resource('s3').Bucket(env_config.S3_RESOURCE_BUCKET_NAME)
    # Get file version to download.
    # If no version is given take the latest version
    if model_type == 'cb':
        file_version = env_config.CB_MODEL_VERSION
        file_name = 'cb_detection_weight.pth'
    elif model_type == 'box':
        file_version = env_config.BOX_MODEL_VERSION
        file_name = 'box_weight.pth'
    else:
        return None

    s3_file_object = s3.Object(key=file_name)
    file_version = file_version if file_version else s3_file_object.version_id

    # Check if file aready exists
    if not check_model_exists(model_type, file_version):
        store_file_name = '{}__{}.pth'.format(os.path.splitext(file_name)[0],
                                              file_version)
        with open(os.path.join('model_weights', store_file_name), 'wb') as op:
            s3.download_fileobj(file_name, op,
                                ExtraArgs={'VersionId': file_version})


def fill_common_configs_from_aws():
    """
    This function will dynamically fill
    the configs from AWS to env_config
    variable.
    """
    ssm = boto3.client('ssm')
    # Get list of env variables defined in env_config
    envs = [i for i in dir(env_config) if i.isupper()]

    # Fetch available config values from AWS
    # 10 values can be queried at once
    for env_batch in [envs[i: i+10] for i in range(0, len(envs), 10)]:
        response = ssm.get_parameters(Names=env_batch)
        # Iterate over each parameter and assign to env_config
        for parameter in response['Parameters']:
            key, value = parameter['Name'], parameter['Value']
            exec("env_config.{} = '{}'".format(key, value))
