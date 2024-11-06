# GCS Photo Uploader

A Python utility for efficiently uploading photos to Google Cloud Storage with automatic organization, HEIC conversion, and duplicate detection.

## Features

- 📁 Automatic folder organization by year/month
- 🔄 HEIC/HEIF to JPEG conversion
- 🔍 Advanced duplicate detection
- ⚡ Concurrent uploads
- 🏷️ Metadata preservation
- 🔀 Optional file renaming
- 📝 Detailed upload history
- 🐳 Docker support

## Quick Start

### 1. Prerequisites

- Google Cloud Platform account
- Google Cloud CLI installed
- Docker installed (optional, for containerized usage)

### 2. Google Cloud Setup

1. Install Google Cloud CLI:

```bash
# For macOS
brew install google-cloud-sdk

# For Ubuntu/Debian
sudo apt-get install google-cloud-sdk
```

2. Initialize Google Cloud CLI and create credentials:

```bash
# Login to Google Cloud
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Create application default credentials
gcloud auth application-default login
```

This will create credentials at:

- Linux/macOS: `$HOME/.config/gcloud/application_default_credentials.json`
- Windows: `%APPDATA%/gcloud/application_default_credentials.json`

3. Create a Google Cloud Storage bucket:

```bash
gsutil mb -l us-central1 gs://YOUR_BUCKET_NAME
```

### 3. Prepare Photos

1. Create a photos directory:

```bash
mkdir photos
```

2. Copy your photos to this directory:

```bash
# Example: Copy all photos from another location
cp -r /path/to/your/photos/* photos/

# Or manually copy your photos to the 'photos' directory
```

The directory can contain subdirectories, and all supported image files will be processed recursively.

Supported formats:

- JPEG/JPG
- PNG
- HEIC/HEIF (automatically converted to JPEG)

### 4. Configuration

Create a `.env` file:

```.env
# Google Cloud Storage settings
PHOTO_UPLOADER_BUCKET=your-bucket-name
PHOTO_UPLOADER_DIRECTORY=/photos
PHOTO_UPLOADER_CONCURRENT=4
PHOTO_UPLOADER_RENAME_FILES=true
GOOGLE_CLOUD_PROJECT=your-project-id

# Google Cloud credentials
GOOGLE_APPLICATION_CREDENTIALS=/secrets/application_default_credentials.json
```

### 5. Build and Run

```bash
# Build Docker image
docker build -t photo-uploader .

# Run container
docker run \
    -v "$(pwd)/photos:/photos:ro" \
    -v "$HOME/.config/gcloud:/secrets:ro" \
    --env-file .env \
    --user $(id -u):$(id -g) \
    photo-uploader
```

## Usage Details

### Directory Structure

Photos are organized in GCS as follows:

```
bucket_name/
├── 2023/
│   ├── 01/
│   │   ├── 20230101_120000_1234.jpg
│   │   └── 20230115_083000_5678.jpg
│   └── 02/
└── 2024/
```

### File Naming

When `PHOTO_UPLOADER_RENAME_FILES=true`, files are renamed using:

```
YYYYMMDD_HHMMSS_XXXX.ext
```

- `YYYYMMDD_HHMMSS`: Original photo date
- `XXXX`: Random 4-digit number
- `ext`: Original extension (lowercase)

### Duplicate Detection

Files are considered duplicates if any of the following match:

- Identical SHA256 hash
- Same creation time (±2 seconds) and file size
- Same dimensions and similar metadata

### Upload History

A detailed upload history is generated showing:

- Original file paths
- Destination paths in GCS
- Duplicate detections
- Conversion results

## Environment Variables

| Variable                       | Description          | Default | Required |
| ------------------------------ | -------------------- | ------- | -------- |
| PHOTO_UPLOADER_BUCKET          | GCS bucket name      | -       | Yes      |
| PHOTO_UPLOADER_DIRECTORY       | Source directory     | /photos | Yes      |
| PHOTO_UPLOADER_CONCURRENT      | Concurrent uploads   | 4       | No       |
| PHOTO_UPLOADER_RENAME_FILES    | Enable file renaming | false   | No       |
| PHOTO_UPLOADER_DRY_RUN         | Enable dry run       | false   | No       |
| GOOGLE_CLOUD_PROJECT           | Project ID           | -       | Yes      |
| GOOGLE_APPLICATION_CREDENTIALS | Credentials path     | -       | Yes      |

## Project Structure

```
gcs-photo-uploader/
├── photo_uploader.py
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

## License

MIT License
