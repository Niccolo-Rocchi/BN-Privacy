FROM python:bookworm

WORKDIR /workspace
COPY . .

RUN pip install --upgrade pip
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
RUN python -m scripts.generate_data

CMD /bin/sh
