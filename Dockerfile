# Use a Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt to the container
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary Python scripts and the CSV file into the container
COPY file1.py file2.py file3.py image_search.py /app/
COPY fashion.csv /app/

# Expose the port that Streamlit will run on
EXPOSE 8501

# Command to run the Python scripts in order and start Streamlit
CMD ["sh", "-c", "python3 file1.py && python3 file2.py && streamlit run file3.py --server.port 8501 --server.address 0.0.0.0"]
