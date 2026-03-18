# 1. Use an official, lightweight Python 3.10 image
FROM python:3.10-slim

# 2. Set the directory inside the container
WORKDIR /app

# 3. Install essential system build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the requirements file first (this speeds up future builds)
COPY requirements.txt .

# 5. Install all Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your application code
COPY . .

# 7. Tell Railway/Render which port to expose
EXPOSE 7860

# 8. Start the Gunicorn server with high timeouts for AI processing
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1", "--threads", "2"]
