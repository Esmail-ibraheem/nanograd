# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code to the working directory
COPY . .

# Make port 5000 available to the host (adjust as needed)
EXPOSE 5000

# Define environment variable
ENV NAME World

# Run the application
CMD ["python", "main.py"]
