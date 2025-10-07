FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt ./    
RUN pip3 install --upgrade pip

RUN apt update && apt install -y --no-install-recommends gcc \
 && pip install pandas \
 && apt remove -y gcc \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y libhdf5-dev
RUN pip install h5py

RUN pip3 install --no-cache-dir -r requirements.txt --index-url https://download.pytorch.org/whl/cpu

COPY . .
CMD ["python", "src/inference.py"]