# Image Initialization
FROM python:3.9.6

RUN wget https://dl.min.io/server/minio/release/linux-amd64/minio && \
    chmod +x minio && \
    mv minio /usr/local/bin/ && \
    wget https://dl.min.io/client/mc/release/linux-amd64/mc && \
    chmod +x mc && \
    mv mc /usr/local/bin/

# Setting workdir inside container
WORKDIR /app

# Setting dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copying api into container
COPY src/ .
COPY init.sh .

# RUN init.sh
CMD ["sh", "init.sh"] && python api.py