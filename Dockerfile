# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the server code to the container
COPY . /app

# Copy the credentials JSON file into the container
COPY credentials/coen6313proj-442020-2cc291d18994.json /app/credentials/coen6313proj-442020-2cc291d18994.json

# Install system dependencies (if any)
RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your app runs on
# EXPOSE 5000
EXPOSE 8080

# Set the command to run the Flask app
# CMD ["python", "server_clean.py"]
# CMD ["python", "server_clean.py", "--debug"]
CMD ["python", "server.py", "--debug"]

