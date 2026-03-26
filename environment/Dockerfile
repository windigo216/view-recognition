FROM python:3.12-slim
RUN apt-get update && apt-get install -y tmux chafa && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy pillow
COPY selected_questions/ /questions/
WORKDIR /home/user