FROM python

WORKDIR /workspace
COPY . .

RUN python -m pip install --upgrade pip
RUN if [ -f requirements.txt ]; then python -m pip install -r requirements.txt; fi
RUN python src/generation.py

CMD /bin/sh
