# Build stage
FROM python:3.11-slim-bullseye as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11-slim-bullseye

# Install only required runtime dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    libmagic1 \
    imagemagick \
    libmagickwand-dev \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /etc/ImageMagick-6 \
    && echo '<policymap> \n\
    <policy domain="delegate" rights="read|write" pattern="HEIC" /> \n\
    <policy domain="coder" rights="read|write" pattern="HEIC" /> \n\
    </policymap>' > /etc/ImageMagick-6/policy.xml

# Create a non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser -s /sbin/nologin -M appuser && \
    mkdir -p /app /photos /secrets && \
    chown -R appuser:appuser /app /photos && \
    chmod -R 755 /app /photos

# Copy Python packages from builder
COPY --from=builder /root/.local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Set working directory and copy application
WORKDIR /app
COPY photo_uploader.py .
RUN chown appuser:appuser /app/photo_uploader.py && \
    chmod 644 /app/photo_uploader.py

# Switch to non-root user
USER appuser

# Configure environment variables
ENV PHOTO_UPLOADER_CONCURRENT=4 \
    GOOGLE_APPLICATION_CREDENTIALS=/secrets/credentials.json \
    PHOTO_UPLOADER_RENAME_FILES=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Define volumes
VOLUME ["/photos", "/secrets"]

# Set entrypoint
ENTRYPOINT ["python", "photo_uploader.py"]