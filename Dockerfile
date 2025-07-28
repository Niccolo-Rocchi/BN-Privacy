FROM python:bookworm

WORKDIR /workspace
COPY . .

RUN pip install --upgrade pip
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

CMD /bin/sh
