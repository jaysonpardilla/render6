# Use official Python image
FROM python:3.10-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required by mediapipe
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Collect static files
#RUN python manage.py collectstatic --no-input

# Run migrations
#RUN python manage.py migrate

# Expose port
EXPOSE 8000

# Start the app with gunicorn

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

