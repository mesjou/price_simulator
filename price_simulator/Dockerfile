FROM python:3.7

ADD src/algorithm/main.py /

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

COPY . /price_simulator/

CMD [ "python", "./main.py" ]
