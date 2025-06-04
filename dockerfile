FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src
COPY ./data ./data
COPY ./models ./models

EXPOSE 7860

CMD ["python", "src/app.py"]
