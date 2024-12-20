# Use official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy source code to working directory
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for the app
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]
