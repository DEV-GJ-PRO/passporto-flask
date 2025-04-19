FROM python:3.9-slim
   WORKDIR /app
   COPY . .
   RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev && rm -rf /var/lib/apt/lists/*
   RUN pip install --no-cache-dir -r requirements.txt
   EXPOSE $PORT
   CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:$PORT", "app:app"]