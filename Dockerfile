# Base image
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Copy requirements
COPY pyproject.toml .

# Install dependencies
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run server (example)
CMD ["uvicorn", "evo_logic:app", "--host", "0.0.0.0", "--port", "8000"]
