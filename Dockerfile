FROM python:3.10-slim

WORKDIR /code

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=1200 -r requirements.txt

# Copy the application code
COPY ./app ./app

# Create data directory
RUN mkdir -p /code/data
ENV MIMIC_DATA_DIR=/code/data

# Crucial: Add the current directory to PYTHONPATH so "from app.x import y" works
ENV PYTHONPATH=/code

# Run the app using the module path
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]