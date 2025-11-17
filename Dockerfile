# Use official Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy files from your PC into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run your script (change filename if needed)
CMD ["python", "code_try.py"]
