FROM python:3.12-slim
WORKDIR /app
COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY ./src /app/src
ENV PYTHONPATH=/app

EXPOSE 3555
ENTRYPOINT ["python", "src/main.py"]