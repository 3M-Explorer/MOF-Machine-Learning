# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose a default port if your app serves a web app (adjust if needed)
EXPOSE 5000

# Define the command to run your app (if applicable)
# CMD ["python", "app.py"]
