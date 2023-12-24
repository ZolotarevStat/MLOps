import boto3
from moto import mock_s3
import pytest

# using this article https://www.sanjaysiddhanti.com/2020/04/08/s3testing/

# from recipe import Recipe, S3_BUCKET


@pytest.fixture
def s3():
    """Pytest fixture that creates the recipes bucket in 
    the fake moto AWS account
    
    Yields a fake boto3 s3 client
    """
    with mock_s3():
        s3_client = boto3.client('s3')
        s3_client.create_bucket(Bucket="test")
        yield s3_client


def test_create_and_get(s3):
    # Recipe(name="nachos", instructions="Melt cheese on chips").save()
    s3.s3_client.put_object(Body='testing_minio', Bucket='test', Key="test.txt")
    # recipe = Recipe.get_by_name("nachos")
    # assert recipe.name == "nachos"
    # assert recipe.instructions == "Melt cheese on chips"
    response = s3.s3_client.get_object(Bucket='zol-hw-mlops', Key="modelsDict.pkl")
    test_txt = response["Body"].read()
    assert test_txt == 'testing_minio'
