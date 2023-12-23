echo 'Starting Minio server...'
minio server minio/
mc alias set hatefuls3 http://127.0.0.1:9000 vtvVX2Lk2hDo6rs0jiBf 2MbKHxqYS39aypu0RNW5xpRkhbg99xspagqrFHYv

echo 'Creating bucket...'
mc mb hatefuls3/zol_hw_mlops

echo "Initializing dvc..."
dvc init -f
dvc remote remove hatefuls3
dvc remote add -d hatefuls3 s3://zol-hw-mlops -f
dvc remote modify --local hatefuls3 endpointurl http://127.0.0.1:9000
dvc remote modify --local hatefuls3 access_key_id my_id
dvc remote modify --local hatefuls3 secret_access_key my_key

echo "Running api..."
python src/api.py