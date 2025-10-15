# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# These are needed for some Python packages (numpy, pandas, sklearn, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir: Disables the cache, which reduces the image size.
# --trusted-host pypi.python.org: Can help avoid SSL issues in some network environments.
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# Copy the rest of the application code into the container at /app
# This includes app.py, pages/, models, and data files
COPY . .

# Make port 8501 available to the world outside this container
# Streamlit uses port 8501 by default
EXPOSE 8501

# Define environment variable
ENV NAME="DS-Experiments"

# Create a healthcheck to verify the app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run app.py when the container launches
# streamlit run app.py tells Streamlit to run the main app file
# --server.port=8501 specifies the port to run on (default)
# --server.address=0.0.0.0 makes the app accessible from outside the container
# --server.headless=true runs without opening a browser
# --server.fileWatcherType=none disables file watching for better performance
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none"]
