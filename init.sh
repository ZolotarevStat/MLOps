echo 'Starting Minio server...'
minio server minio/
mc alias set hatefuls3 http://127.0.0.1:9000 nVobiQ2pBendB0nGOoxi tn7duToNNSKZuarizESJZWcaLwnsxf75K8Wp8J9Z

echo 'Creating bucket...'
mc mb hatefuls3/zol-hw-mlops

echo "Initializing dvc..."
dvc init -f
# dvc remote remove hatefuls3
dvc remote add -d hatefuls3 s3://zol-hw-mlops -f
dvc remote modify hatefuls3 endpointurl http://127.0.0.1:9000
dvc remote modify hatefuls3 access_key_id nVobiQ2pBendB0nGOoxi
dvc remote modify hatefuls3 secret_access_key tn7duToNNSKZuarizESJZWcaLwnsxf75K8Wp8J9Z

echo "Running api..."
python src/api.py