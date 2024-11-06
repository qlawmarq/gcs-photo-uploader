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

class DuplicateDetector:
    """Handles duplicate detection for both local and cloud storage."""
    def __init__(self, logger: logging.Logger):
        self.cloud_files: Dict[str, PhotoMetadata] = {}
        self.local_files: Dict[str, PhotoMetadata] = {}
        self.processed_hashes: Set[str] = set()
        self.logger = logger

    def add_cloud_file(self, path: str, metadata: PhotoMetadata) -> None:
        """Add cloud file metadata to the detector."""
        self.cloud_files[path] = metadata
        self.processed_hashes.add(metadata.file_hash)

    def add_local_file(self, path: str, metadata: PhotoMetadata) -> None:
        """Add local file metadata to the detector."""
        self.local_files[path] = metadata
        self.processed_hashes.add(metadata.file_hash)

    def find_duplicate(self, metadata: PhotoMetadata) -> Optional[Tuple[str, str]]:
        """
        Check if photo is duplicate in either local or cloud storage.
        Returns tuple of (location, path) if duplicate found, None otherwise.
        Location can be either 'local' or 'cloud'.
        """
        # Check by hash first (most reliable method)
        if metadata.file_hash in self.processed_hashes:
            # Check in cloud files
            for path, existing in self.cloud_files.items():
                if existing.file_hash == metadata.file_hash:
                    return ('cloud', path)
            # Check in local files
            for path, existing in self.local_files.items():
                if existing.file_hash == metadata.file_hash:
                    return ('local', path)

        # Check by size and date
        similar_files = self._find_similar_files(metadata)
        if similar_files:
            # Additional checks for similar files
            for location, path, existing in similar_files:
                if self._is_likely_duplicate(metadata, existing):
                    return (location, path)

        return None

    def _find_similar_files(self, metadata: PhotoMetadata) -> List[Tuple[str, str, PhotoMetadata]]:
        """Find files with similar properties (size and creation date)."""
        similar_files = []
        
        # Check both cloud and local files
        for location, files in [('cloud', self.cloud_files), ('local', self.local_files)]:
            for path, existing in files.items():
                if (abs((existing.creation_date - metadata.creation_date).total_seconds()) < 2 and
                    existing.file_size == metadata.file_size):
                    similar_files.append((location, path, existing))
        
        return similar_files

    def _is_likely_duplicate(self, metadata1: PhotoMetadata, metadata2: PhotoMetadata) -> bool:
        """Compare two files' metadata to determine if they are likely duplicates."""
        # If dimensions are available, compare them
        if (metadata1.width and metadata1.height and 
            metadata2.width and metadata2.height):
            if (metadata1.width == metadata2.width and 
                metadata1.height == metadata2.height):
                return True

        # If camera models are available, compare them
        if (metadata1.camera_model and metadata2.camera_model and
            metadata1.camera_model == metadata2.camera_model):
            return True

        return False

class PhotoUploader:
    def __init__(
            self, 
            bucket_name: str, 
            concurrent_uploads: int = 4, 
            rename_files: bool = False, 
            project_id: Optional[str] = None,
            dry_run: bool = False
        ):
        """Initialize PhotoUploader with specified configuration."""
        self.bucket_name = bucket_name
        self.client = storage.Client(project=project_id) if project_id else storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.concurrent_uploads = concurrent_uploads
        self.rename_files = rename_files
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
        self.upload_history: Dict[str, str] = {}
        self.logger = self._setup_logger()
        self.duplicate_detector = DuplicateDetector(self.logger)
        self.dry_run = dry_run
        
        if self.dry_run:
            self.logger.info("=== DRY RUN MODE: No actual uploads will be performed ===")

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

                photo_metadata = PhotoMetadata(
                    file_hash=metadata.get('file_hash', ''),
                    file_size=blob.size,
                    creation_date=datetime.fromisoformat(metadata.get('photo_date', '')),
                    cloud_path=blob.name
                )
                self.duplicate_detector.add_cloud_file(blob.name, photo_metadata)
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

    def _save_upload_history(self, work_dir: Path) -> None:
        """Save upload history to a log file."""
        try:
            history_path = work_dir / 'upload_history.log'
            with open(history_path, 'w', encoding='utf-8') as f:
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
            
            self._load_existing_files()
            
            self.logger.info("Collecting local file metadata...")
            local_files = self._collect_local_files(directory)
            
            files_to_process = []
            duplicates_found = []
            
            # ファイル処理（両モードで同じフロー）
            for filepath in local_files:
                try:
                    metadata = self._extract_photo_metadata(filepath)
                    
                    # 重複チェック（両モードで実行）
                    duplicate_result = self.duplicate_detector.find_duplicate(metadata)
                    if duplicate_result:
                        location, duplicate_path = duplicate_result
                        duplicates_found.append((str(filepath), location, duplicate_path))
                        self.logger.info(
                            f"{'[DRY RUN] ' if self.dry_run else ''}Skipping duplicate: {filepath.name} "
                            f"(already exists in {location} as {duplicate_path})"
                        )
                        continue
                    
                    self.duplicate_detector.add_local_file(str(filepath), metadata)
                    
                    target_path = self._prepare_file_for_upload(filepath, work_dir)
                    if target_path:
                        files_to_process.append(target_path)
                
                except Exception as e:
                    self.logger.error(f"{'[DRY RUN] ' if self.dry_run else ''}Failed to process {filepath}: {str(e)}")

            if not files_to_process:
                self.logger.warning(f"{'[DRY RUN] ' if self.dry_run else ''}No valid files to upload")
                return

            self.logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Processing {len(files_to_process)} files...")
            
            with ThreadPoolExecutor(max_workers=self.concurrent_uploads) as executor:
                results = list(executor.map(self._upload_file, files_to_process))

            total = len(files_to_process)
            successful = sum(1 for r in results if r)
            skipped = len(duplicates_found)
            
            self.logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Upload summary:")
            self.logger.info(f"- Total files: {total + skipped}")
            self.logger.info(f"- Successfully processed: {successful}")
            self.logger.info(f"- Skipped (duplicates): {skipped}")

            for original, location, duplicate_path in duplicates_found:
                self.upload_history[original] = f"{'[DRY RUN] ' if self.dry_run else ''}Duplicate found in {location}: {duplicate_path}"

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

    def _collect_local_files(self, directory: Path) -> List[Path]:
        """Collect all valid photo files from the local directory."""
        valid_files = []
        for filepath in directory.rglob('*'):
            if (filepath.is_file() and 
                filepath.suffix.lower() in self.supported_formats and
                magic.Magic(mime=True).from_file(str(filepath)).startswith('image/')):
                valid_files.append(filepath)
        
        self.logger.info(f"Found {len(valid_files)} valid image files")
        return valid_files

    def _prepare_file_for_upload(self, filepath: Path, work_dir: Path) -> Optional[Path]:
        """Prepare file for upload by copying to work directory and converting if necessary."""
        try:
            base_dir = Path(os.environ.get('PHOTO_UPLOADER_DIRECTORY', '')).resolve()
            if not base_dir.exists():
                base_dir = filepath.parent.resolve()

            try:
                relative_path = filepath.resolve().relative_to(base_dir)
            except ValueError:
                relative_path = filepath.name

            target_path = work_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(filepath, target_path)

            if target_path.suffix.lower() in {'.heic', '.heif'}:
                return self._process_heic(target_path)
            return target_path
        except Exception as e:
            self.logger.error(f"Failed to prepare {filepath} for upload: {str(e)}")
            return None

    def _upload_file(self, filepath: Path) -> bool:
        """Upload single file to Google Cloud Storage."""
        try:
            metadata = self._extract_photo_metadata(filepath)
            
            # Generate destination path
            date_str = metadata.creation_date.strftime('%Y%m%d_%H%M%S')
            filename = (f"{date_str}_{random.randint(1000, 9999)}{filepath.suffix.lower()}"
                       if self.rename_files else filepath.name)
            destination_path = f"{metadata.creation_date.year}/{metadata.creation_date.month:02d}/{filename}"

            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would upload: {filepath.name} -> {destination_path}")
                self.logger.debug(f"[DRY RUN] Metadata: size={metadata.file_size}, "
                                f"date={metadata.creation_date.isoformat()}, "
                                f"camera={metadata.camera_model or 'unknown'}")
            else:
                blob = self.bucket.blob(destination_path)
                blob.upload_from_filename(str(filepath))
                
                blob.metadata = {
                    'upload_date': datetime.now().isoformat(),
                    'photo_date': metadata.creation_date.isoformat(),
                    'file_hash': metadata.file_hash,
                    'original_name': filepath.name,
                    'camera_model': metadata.camera_model or 'unknown'
                }
                blob.patch()
                self.logger.info(f"Uploaded: {filepath.name} -> {destination_path}")

            metadata.cloud_path = destination_path
            self.duplicate_detector.add_cloud_file(destination_path, metadata)
            self.upload_history[str(filepath)] = f"{'[DRY RUN] ' if self.dry_run else ''}Uploaded to {destination_path}"
            
            return True

        except Exception as e:
            self.logger.error(f"{'[DRY RUN] ' if self.dry_run else ''}Failed to process {filepath.name}: {str(e)}")
            return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Upload photos to Google Cloud Storage')
    parser.add_argument('--bucket', help='Destination bucket name (override PHOTO_UPLOADER_BUCKET)')
    parser.add_argument('--directory', help='Source directory path (override PHOTO_UPLOADER_DIRECTORY)')
    parser.add_argument('--concurrent', type=int, help='Number of concurrent uploads (override PHOTO_UPLOADER_CONCURRENT)')
    parser.add_argument('--project-id', help='Google Cloud project ID (override GOOGLE_CLOUD_PROJECT)')
    parser.add_argument('--dry-run', action='store_true', help='Perform a dry run without actual uploads')
    
    args = parser.parse_args()
    
    config = {
        'bucket_name': args.bucket or os.environ.get('PHOTO_UPLOADER_BUCKET'),
        'directory_path': args.directory or os.environ.get('PHOTO_UPLOADER_DIRECTORY'),
        'concurrent_uploads': args.concurrent or int(os.environ.get('PHOTO_UPLOADER_CONCURRENT', '4')),
        'project_id': args.project_id or os.environ.get('GOOGLE_CLOUD_PROJECT'),
        'rename_files': os.environ.get('PHOTO_UPLOADER_RENAME_FILES', '').lower() == 'true',
        'dry_run': args.dry_run or os.environ.get('PHOTO_UPLOADER_DRY_RUN', '').lower() == 'true'
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
        config['project_id'],
        config['dry_run']
    )
    uploader.process_directory(directory_path)

if __name__ == '__main__':
    main()