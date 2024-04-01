# Specify the Dockerfile syntax version
# syntax=docker/dockerfile:1

# Use the Python 3.10 Alpine base image
FROM python:3.10-alpine

# Set the working directory to /code
WORKDIR /code

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Install necessary dependencies
RUN apk add --no-cache gcc musl-dev linux-headers

# Copy the requirements.txt file and install the dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Copy the entire project directory to the working directory
COPY . .

# Set the command to run the Flask application
CMD ["flask", "--app", "app.py", "run"]
