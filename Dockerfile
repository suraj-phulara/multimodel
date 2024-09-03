# Use a Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy Python scripts and the CSV file to the container
COPY web.py image_search.py ingest_data.py /app/
COPY fashion.csv /app/

# Install any Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit will run on
EXPOSE 8501

# Command to run Python scripts in order and start Streamlit
CMD ["sh", "-c", "python3 ingest_data.py && python3 file2.py && streamlit run web.py --server.port 8501 --server.address 0.0.0.0"]
