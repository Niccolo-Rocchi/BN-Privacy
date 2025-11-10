FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    swig \ 
    libglpk-dev \
    python3-dev \ 
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

RUN pip install --upgrade pip
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

CMD /bin/sh
