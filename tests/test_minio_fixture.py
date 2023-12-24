import boto3
from moto import mock_s3
import pytest

# using this article https://www.sanjaysiddhanti.com/2020/04/08/s3testing/

from recipe import Recipe, S3_BUCKET


@pytest.fixture
def s3():
    """Pytest fixture that creates the recipes bucket in 
    the fake moto AWS account
    
    Yields a fake boto3 s3 client
    """
    with mock_s3():
        s3_client = boto3.client('s3',
                         endpoint_url='http://127.0.0.1:9000',
                         aws_access_key_id='nVobiQ2pBendB0nGOoxi',
                         aws_secret_access_key='tn7duToNNSKZuarizESJZWcaLwnsxf75K8Wp8J9Z')
        s3_client.create_bucket(Bucket=S3_BUCKET)
        yield s3_client

def test_create_and_get(s3):
    Recipe(name="nachos", instructions="Melt cheese on chips").save()

    recipe = Recipe.get_by_name("nachos")
    assert recipe.name == "nachos"
    assert recipe.instructions == "Melt cheese on chips"
    
