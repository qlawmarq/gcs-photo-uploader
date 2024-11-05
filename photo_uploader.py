"""
Google Cloud Storage Photo Uploader with improved duplicate detection.
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
import hashlib
from typing import List, Set, Optional, Tuple, Dict
import magic
import piexif
from PIL import Image
from wand.image import Image as WandImage
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor
import logging
import tempfile
import shutil
import random
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class PhotoMetadata:
    """Photo metadata container for duplicate detection."""
    file_hash: str
    file_size: int
    creation_date: datetime
    width: Optional[int] = None
    height: Optional[int] = None
    camera_model: Optional[str] = None
    original_path: Optional[str] = None
    cloud_path: Optional[str] = None

class PhotoUploader:
    def __init__(self, bucket_name: str, concurrent_uploads: int = 4, 
                 rename_files: bool = False, project_id: Optional[str] = None):
        """Initialize PhotoUploader with specified configuration."""
        self.bucket_name = bucket_name
        self.client = storage.Client(project=project_id) if project_id else storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.concurrent_uploads = concurrent_uploads
        self.rename_files = rename_files
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
        self.processed_hashes: Set[str] = set()
        self.existing_files: Dict[str, PhotoMetadata] = {}
        self.upload_history: Dict[str, str] = {}  # original_path -> cloud_path
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Configure and return logger instance."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _load_existing_files(self) -> None:
        """Load metadata of existing files in the bucket."""
        self.logger.info("Loading existing files from bucket...")
        blobs = self.bucket.list_blobs()
        for blob in blobs:
            try:
                metadata = blob.metadata or {}
                if not metadata:
                    continue

                self.existing_files[blob.name] = PhotoMetadata(
                    file_hash=metadata.get('file_hash', ''),
                    file_size=blob.size,
                    creation_date=datetime.fromisoformat(metadata.get('photo_date', '')),
                    cloud_path=blob.name
                )
                self.processed_hashes.add(metadata.get('file_hash', ''))
            except Exception as e:
                self.logger.warning(f"Failed to load metadata for {blob.name}: {e}")

    def _extract_photo_metadata(self, filepath: Path) -> PhotoMetadata:
        """Extract comprehensive metadata from photo file."""
        file_hash = hashlib.sha256(filepath.read_bytes()).hexdigest()
        file_size = filepath.stat().st_size
        creation_date = datetime.fromtimestamp(os.path.getmtime(filepath))
        width = height = None
        camera_model = None

        try:
            with Image.open(filepath) as img:
                width, height = img.size
                exif_dict = piexif.load(img.info.get('exif', b''))
                if exif_dict.get('Exif'):
                    for field in [36867, 36868, 306]:  # Date fields
                        if field in exif_dict['Exif']:
                            date_str = exif_dict['Exif'][field].decode('utf-8')
                            creation_date = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
                            break
                    
                    if exif_dict.get('0th') and 272 in exif_dict['0th']:  # Camera model
                        camera_model = exif_dict['0th'][272].decode('utf-8').strip()
        except Exception as e:
            self.logger.debug(f"Failed to extract EXIF from {filepath}: {e}")

        return PhotoMetadata(
            file_hash=file_hash,
            file_size=file_size,
            creation_date=creation_date,
            width=width,
            height=height,
            camera_model=camera_model,
            original_path=str(filepath)
        )

    def _is_duplicate(self, metadata: PhotoMetadata) -> Optional[str]:
        """
        Check if photo is duplicate using multiple criteria.
        Returns cloud path of duplicate if found, None otherwise.
        """
        # Check by hash
        if metadata.file_hash in self.processed_hashes:
            return next(
                (f.cloud_path for f in self.existing_files.values() 
                 if f.file_hash == metadata.file_hash),
                None
            )

        # Check by size and date
        similar_files = [
            f for f in self.existing_files.values()
            if abs((f.creation_date - metadata.creation_date).total_seconds()) < 2 and
            f.file_size == metadata.file_size
        ]

        if similar_files:
            # Additional checks for similar files
            for existing in similar_files:
                if (metadata.width and metadata.height and 
                    existing.width == metadata.width and 
                    existing.height == metadata.height):
                    return existing.cloud_path

        return None

    def _upload_file(self, filepath: Path) -> bool:
        """Upload single file to Google Cloud Storage with duplicate detection."""
        try:
            metadata = self._extract_photo_metadata(filepath)
            
            # Check for duplicates
            duplicate_path = self._is_duplicate(metadata)
            if duplicate_path:
                self.logger.info(f"Skipping duplicate: {filepath.name} (already exists as {duplicate_path})")
                self.upload_history[str(filepath)] = duplicate_path
                return False

            # Generate destination path
            date_str = metadata.creation_date.strftime('%Y%m%d_%H%M%S')
            filename = (f"{date_str}_{random.randint(1000, 9999)}{filepath.suffix.lower()}"
                       if self.rename_files else filepath.name)
            destination_path = f"{metadata.creation_date.year}/{metadata.creation_date.month:02d}/{filename}"

            # Upload file
            blob = self.bucket.blob(destination_path)
            blob.upload_from_filename(str(filepath))
            
            # Set metadata
            blob.metadata = {
                'upload_date': datetime.now().isoformat(),
                'photo_date': metadata.creation_date.isoformat(),
                'file_hash': metadata.file_hash,
                'original_name': filepath.name,
                'camera_model': metadata.camera_model or 'unknown'
            }
            blob.patch()

            self.processed_hashes.add(metadata.file_hash)
            self.existing_files[destination_path] = metadata
            metadata.cloud_path = destination_path
            self.upload_history[str(filepath)] = destination_path
            
            self.logger.info(f"Uploaded: {filepath.name} -> {destination_path}")
            return True

        except Exception as e:
            self.logger.error(f"Upload failed for {filepath.name}: {str(e)}")
            return False

    def _save_upload_history(self, work_dir: Path) -> None:
        """Save upload history to a log file."""
        try:
            history_path = work_dir / 'upload_history.log'
            with open(history_path, 'w') as f:
                f.write("Original File -> Cloud Storage Path\n")
                f.write("-" * 50 + "\n")
                for orig, cloud in sorted(self.upload_history.items()):
                    f.write(f"{orig} -> {cloud}\n")
            self.logger.info(f"Upload history saved to {history_path}")
        except Exception as e:
            self.logger.error(f"Failed to save upload history: {e}")

    def process_directory(self, directory: Path) -> None:
        """Process and upload all valid photos in directory."""
        try:
            work_dir = Path(tempfile.mkdtemp())
            self.logger.info("Initializing upload process...")
            
            # Load existing files first
            self._load_existing_files()
            
            # Process files
            files_to_process = []
            for filepath in directory.rglob('*'):
                if not filepath.is_file() or filepath.suffix.lower() not in self.supported_formats:
                    continue
                    
                try:
                    if not magic.Magic(mime=True).from_file(str(filepath)).startswith('image/'):
                        continue
                except Exception:
                    continue

                # Copy to work directory
                target_path = work_dir / filepath.relative_to(directory)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(filepath, target_path)

                # Convert HEIC if necessary
                if target_path.suffix.lower() in {'.heic', '.heif'}:
                    converted_path = self._process_heic(target_path)
                    if converted_path:
                        files_to_process.append(converted_path)
                else:
                    files_to_process.append(target_path)

            if not files_to_process:
                self.logger.warning("No valid image files found")
                return

            self.logger.info(f"Processing {len(files_to_process)} files...")
            
            # Perform uploads
            with ThreadPoolExecutor(max_workers=self.concurrent_uploads) as executor:
                results = list(executor.map(self._upload_file, files_to_process))

            # Report results
            total = len(files_to_process)
            successful = sum(1 for r in results if r)
            skipped = sum(1 for r in results if not r)
            
            self.logger.info(f"Upload summary:")
            self.logger.info(f"- Total processed: {total}")
            self.logger.info(f"- Successfully uploaded: {successful}")
            self.logger.info(f"- Skipped (duplicates): {skipped}")

            # Save upload history
            self._save_upload_history(work_dir)

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _process_heic(self, filepath: Path) -> Optional[Path]:
        """Convert HEIC/HEIF to JPEG if necessary."""
        try:
            jpeg_path = filepath.with_suffix('.jpg')
            with WandImage(filename=str(filepath)) as img:
                original_exif = img.profiles.get('exif', None)
                img.compression_quality = 95
                img.save(filename=str(jpeg_path))

                if original_exif:
                    with WandImage(filename=str(jpeg_path)) as jpeg_img:
                        jpeg_img.profiles['exif'] = original_exif
                        jpeg_img.save(filename=str(jpeg_path))

            self.logger.info(f"Converted {filepath.name} to JPEG")
            return jpeg_path
        except Exception as e:
            self.logger.error(f"HEIC conversion failed for {filepath}: {str(e)}")
            return None

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Upload photos to Google Cloud Storage')
    parser.add_argument('--bucket', help='Destination bucket name (override PHOTO_UPLOADER_BUCKET)')
    parser.add_argument('--directory', help='Source directory path (override PHOTO_UPLOADER_DIRECTORY)')
    parser.add_argument('--concurrent', type=int, help='Number of concurrent uploads (override PHOTO_UPLOADER_CONCURRENT)')
    parser.add_argument('--project-id', help='Google Cloud project ID (override GOOGLE_CLOUD_PROJECT)')
    
    args = parser.parse_args()
    
    config = {
        'bucket_name': args.bucket or os.environ.get('PHOTO_UPLOADER_BUCKET'),
        'directory_path': args.directory or os.environ.get('PHOTO_UPLOADER_DIRECTORY'),
        'concurrent_uploads': args.concurrent or int(os.environ.get('PHOTO_UPLOADER_CONCURRENT', '4')),
        'project_id': args.project_id or os.environ.get('GOOGLE_CLOUD_PROJECT'),
        'rename_files': os.environ.get('PHOTO_UPLOADER_RENAME_FILES', '').lower() == 'true'
    }

    if not all([config['bucket_name'], config['directory_path']]):
        print("Error: Bucket name and directory path are required")
        return

    directory_path = Path(config['directory_path'])
    if not directory_path.exists():
        print(f"Error: Directory '{directory_path}' does not exist")
        return

    print("Configuration:", "\n".join(f"- {k}: {v}" for k, v in config.items()))
    
    uploader = PhotoUploader(
        config['bucket_name'],
        config['concurrent_uploads'],
        config['rename_files'],
        config['project_id']
    )
    uploader.process_directory(directory_path)

if __name__ == '__main__':
    main()