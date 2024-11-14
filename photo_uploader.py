"""
Google Cloud Storage Photo Uploader with improved duplicate detection.
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
import hashlib
import string
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
from google.api_core import retry
import time

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
        """
        # Check in cloud files
        for path, existing in self.cloud_files.items():
            if self._is_likely_duplicate(metadata, existing):
                return ('cloud', path)
        
        # Check in local files
        for path, existing in self.local_files.items():
            if self._is_likely_duplicate(metadata, existing):
                return ('local', path)

        return None

    def _is_likely_duplicate(self, metadata1: PhotoMetadata, metadata2: PhotoMetadata) -> bool:
        """
        Compare metadata to determine if they are likely duplicates.
        Uses creation date and camera model for comparison.
        """
        # Check if creation dates are close (within 1 seconds)
        if abs((metadata1.creation_date - metadata2.creation_date).total_seconds()) > 1:
            return False
        
        # If camera models are available and match
        if (metadata1.camera_model and metadata2.camera_model and
            metadata1.camera_model == metadata2.camera_model):
            return True
        
        # If at least one file has no camera model, but dates match exactly
        if metadata1.creation_date == metadata2.creation_date:
            return True

        return False


class PhotoUploader:
    def __init__(
            self, 
            bucket_name: str, 
            concurrent_uploads: int = 4, 
            rename_files: bool = False, 
            project_id: Optional[str] = None,
            dry_run: bool = False,
            directory_path: str = '/photos'
        ):
        """Initialize PhotoUploader with specified configuration."""
        self.bucket_name = bucket_name
        self.client = storage.Client(project=project_id) if project_id else storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.concurrent_uploads = concurrent_uploads
        self.rename_files = rename_files
        self.directory_path = directory_path
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
        """
        Extract comprehensive metadata from photo file.
        Maintains all metadata extraction capabilities while keeping code simple.
        """
        file_hash = hashlib.sha256(filepath.read_bytes()).hexdigest()
        file_size = filepath.stat().st_size
        creation_date = datetime.fromtimestamp(os.path.getmtime(filepath))
        width = height = None
        camera_model = None

        try:
            with Image.open(filepath) as img:
                width, height = img.size
                
                # Extract EXIF data - try both direct exif and _getexif methods
                exif_dict = None
                try:
                    exif_data = img.info.get('exif', b'')
                    exif_dict = piexif.load(exif_data)
                except Exception:
                    if hasattr(img, '_getexif'):
                        exif_dict = img._getexif()
                        if exif_dict:
                            exif_dict = {'Exif': exif_dict}

                if exif_dict:
                    # Extract date information
                    for ifd in ('Exif', '0th'):
                        if ifd not in exif_dict:
                            continue
                        
                        # Check date fields in order of preference
                        date_fields = ((36867, 'Exif'), (36868, 'Exif'), (306, '0th'))  # DateTimeOriginal, DateTimeDigitized, DateTime
                        for field, dict_name in date_fields:
                            if field in exif_dict[ifd]:
                                try:
                                    date_str = exif_dict[ifd][field].decode('utf-8') if isinstance(exif_dict[ifd][field], bytes) else str(exif_dict[ifd][field])
                                    creation_date = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
                                    break
                                except Exception:
                                    continue
                                
                        # Extract camera model
                        if 272 in exif_dict[ifd]:  # Model field
                            try:
                                camera_model = exif_dict[ifd][272].decode('utf-8') if isinstance(exif_dict[ifd][272], bytes) else str(exif_dict[ifd][272])
                                camera_model = camera_model.strip()
                            except Exception:
                                pass

        except Exception as e:
            self.logger.debug(f"Metadata extraction partial failure for {filepath}: {e}")

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
            photos_dir = Path(self.directory_path)
            log_dir = photos_dir / 'logs'
            log_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            history_path = log_dir / f'upload_history_{timestamp}.log'
            
            with open(history_path, 'w', encoding='utf-8') as f:
                f.write("Upload History Report\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write("-" * 50 + "\n\n")
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
            
            for filepath in local_files:
                try:
                    metadata = self._extract_photo_metadata(filepath)
                    
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
        """
        Convert HEIC/HEIF to JPEG with complete metadata preservation.
        Handles IFDRational and other EXIF data types properly.
        """
        try:
            jpeg_path = filepath.with_suffix('.jpg')

            # Extract original metadata using PIL
            original_metadata = None
            original_icc = None
            try:
                with Image.open(filepath) as img:
                    exif_dict = None
                    if hasattr(img, '_getexif'):
                        exif_dict = img._getexif()
                    if exif_dict:
                        # Convert to piexif format
                        zeroth_ifd = {}
                        exif_ifd = {}
                        gps_ifd = {}
                        
                        for tag_id, value in exif_dict.items():
                            try:
                                # Handle different types of EXIF data
                                if isinstance(value, str):
                                    value = value.encode('utf-8')
                                elif hasattr(value, 'numerator') and hasattr(value, 'denominator'):
                                    # Handle IFDRational values
                                    value = (value.numerator, value.denominator)
                                elif isinstance(value, tuple) and len(value) == 2:
                                    # Already in rational format
                                    value = (int(value[0]), int(value[1]))
                                elif isinstance(value, bytes):
                                    # Already in bytes format
                                    pass
                                elif isinstance(value, int):
                                    # Integer values
                                    pass
                                else:
                                    # Convert other types to string then bytes
                                    value = str(value).encode('utf-8')

                                # Categorize based on tag
                                if tag_id in piexif.ImageIFD.__dict__.values():
                                    zeroth_ifd[tag_id] = value
                                elif tag_id in piexif.ExifIFD.__dict__.values():
                                    exif_ifd[tag_id] = value
                                elif tag_id in piexif.GPSIFD.__dict__.values():
                                    gps_ifd[tag_id] = value
                            except Exception as e:
                                self.logger.debug(f"Skipping EXIF tag {tag_id}: {e}")
                                continue
                        
                        original_metadata = {
                            "0th": zeroth_ifd,
                            "Exif": exif_ifd,
                            "GPS": gps_ifd,
                            "1st": {},
                            "Interop": {},
                        }
                    original_icc = img.info.get('icc_profile')
            except Exception as e:
                self.logger.warning(f"Failed to extract original metadata from {filepath}: {e}")

            # Convert HEIC to JPEG using Wand
            with WandImage(filename=str(filepath)) as img:
                # Preserve color depth and space
                img.depth = 16
                img.type = 'truecolor'
                
                # Get ICC profile from Wand if not already extracted
                if not original_icc and 'icc' in img.profiles:
                    original_icc = img.profiles['icc']

                # Convert to highest quality JPEG
                img.format = 'jpeg'
                img.compression_quality = 100
                img.options['jpeg:optimize'] = 'true'
                img.options['jpeg:sampling-factor'] = '1x1'
                
                # Save initial conversion
                img.save(filename=str(jpeg_path))

            # Apply metadata to converted file
            if original_metadata or original_icc:
                with Image.open(jpeg_path) as img:
                    # Prepare save arguments
                    save_kwargs = {
                        'format': 'JPEG',
                        'quality': 100,
                        'optimize': True,
                        'subsampling': 0
                    }

                    # Add ICC profile if available
                    if original_icc:
                        save_kwargs['icc_profile'] = original_icc

                    # If we have EXIF data, add it
                    if original_metadata:
                        try:
                            exif_bytes = piexif.dump(original_metadata)
                            save_kwargs['exif'] = exif_bytes
                        except Exception as e:
                            self.logger.debug(f"Could not dump full EXIF data for {filepath}: {e}")
                            # Attempt to save with partial metadata
                            try:
                                # Create minimal EXIF with essential data
                                minimal_metadata = {
                                    "0th": {},
                                    "Exif": {},
                                    "GPS": {},
                                    "1st": {},
                                    "Interop": {},
                                }
                                # Copy only essential tags that we know will work
                                essential_tags = [
                                    piexif.ImageIFD.Make,
                                    piexif.ImageIFD.Model,
                                    piexif.ExifIFD.DateTimeOriginal,
                                    piexif.ExifIFD.DateTimeDigitized,
                                    piexif.ImageIFD.DateTime
                                ]
                                for tag in essential_tags:
                                    for ifd in ['0th', 'Exif']:
                                        if tag in original_metadata[ifd]:
                                            minimal_metadata[ifd][tag] = original_metadata[ifd][tag]
                                
                                exif_bytes = piexif.dump(minimal_metadata)
                                save_kwargs['exif'] = exif_bytes
                            except Exception as e:
                                self.logger.debug(f"Could not dump minimal EXIF data for {filepath}: {e}")

                    # Save with metadata
                    img.save(jpeg_path, **save_kwargs)

            self.logger.info(f"Converted {filepath.name} to JPEG with metadata preservation")
            return jpeg_path

        except Exception as e:
            self.logger.error(f"HEIC conversion failed for {filepath}: {str(e)}")
            if jpeg_path.exists():
                jpeg_path.unlink()
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
            base_dir = Path(self.directory_path).resolve()
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


    def _preserve_metadata(self, source_path: Path, temp_path: Path) -> None:
        """
        Preserve all image metadata during processing.
        Handles various image formats and ensures complete metadata preservation.
        """
        try:
            with Image.open(source_path) as src_img:
                # Try to get existing EXIF data
                exif_bytes = src_img.info.get('exif')
                icc_profile = src_img.info.get('icc_profile')
                
                # Extract all possible metadata if EXIF is not directly available
                if not exif_bytes and hasattr(src_img, '_getexif'):
                    exif_dict = src_img._getexif() or {}
                    
                    # Prepare EXIF dictionary structure
                    new_exif_dict = {
                        "0th": {},      # Basic image metadata
                        "Exif": {},     # EXIF specific data
                        "GPS": {},      # GPS data if available
                        "1st": {},      # Thumbnail metadata
                        "Interop": {},  # Interoperability data
                    }

                    # Map all available EXIF data to appropriate IFDs
                    for tag_id, value in exif_dict.items():
                        try:
                            # Convert string values to bytes if needed
                            if isinstance(value, str):
                                value = value.encode('utf-8')
                            
                            # Categorize EXIF tags into appropriate IFDs
                            if tag_id in piexif.ImageIFD.__dict__.values():
                                new_exif_dict["0th"][tag_id] = value
                            elif tag_id in piexif.ExifIFD.__dict__.values():
                                new_exif_dict["Exif"][tag_id] = value
                            elif tag_id in piexif.GPSIFD.__dict__.values():
                                new_exif_dict["GPS"][tag_id] = value
                        except Exception as e:
                            self.logger.debug(f"Skipping problematic EXIF tag {tag_id}: {e}")

                    try:
                        exif_bytes = piexif.dump(new_exif_dict)
                    except Exception as e:
                        self.logger.error(f"Failed to dump EXIF data: {e}")
                        exif_bytes = None

                # Preserve the original image format and mode
                target_format = src_img.format
                target_mode = src_img.mode

                # Prepare save parameters based on format
                save_kwargs = {
                    'format': target_format,
                    'quality': 100 if target_format == 'JPEG' else None,
                    'exif': exif_bytes,
                    'icc_profile': icc_profile,
                }

                # Additional format-specific settings
                if target_format == 'JPEG':
                    save_kwargs.update({
                        'optimize': True,
                        'subsampling': 0  # Highest quality subsampling
                    })
                elif target_format == 'PNG':
                    save_kwargs.update({
                        'optimize': True
                    })
                elif target_format == 'TIFF':
                    save_kwargs.update({
                        'compression': 'tiff_lzw'  # Lossless compression
                    })

                # Remove None values from save_kwargs
                save_kwargs = {k: v for k, v in save_kwargs.items() if v is not None}

                # Save with preserved metadata and format
                with Image.open(temp_path) as img:
                    # Ensure the mode matches the source
                    if img.mode != target_mode and target_mode in {'RGB', 'RGBA', 'L'}:
                        img = img.convert(target_mode)
                    
                    temp_output = temp_path.with_suffix('.tmp')
                    img.save(temp_output, **save_kwargs)

                # Move temporary file to target location
                shutil.move(temp_output, temp_path)

        except Exception as e:
            self.logger.error(f"Error preserving metadata: {e}")
            # If metadata preservation fails, ensure the basic image is still saved
            if os.path.exists(temp_path.with_suffix('.tmp')):
                shutil.move(temp_path.with_suffix('.tmp'), temp_path)
            raise

        finally:
            # Clean up temporary file if it exists
            if os.path.exists(temp_path.with_suffix('.tmp')):
                os.remove(temp_path.with_suffix('.tmp'))

    def _upload_file(self, filepath: Path, max_retries: int = 3, initial_delay: float = 1.0) -> bool:
        """
        Upload single file to Google Cloud Storage with retry logic.
        Only preserves metadata for converted HEIC/HEIF files.
        """
        attempt = 0
        delay = initial_delay

        while attempt <= max_retries:
            try:
                metadata = self._extract_photo_metadata(filepath)
                
                # Generate destination path
                date_str = metadata.creation_date.strftime('%Y%m%d_%H%M%S')
                random_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))
                filename = (f"{date_str}_{random_str}{filepath.suffix.lower()}"
                        if self.rename_files else filepath.name)
                destination_path = f"{metadata.creation_date.year}/{metadata.creation_date.month:02d}/{filename}"

                if self.dry_run:
                    self.logger.info(f"[DRY RUN] Would upload: {filepath.name} -> {destination_path}")
                    self.logger.debug(f"[DRY RUN] Metadata: size={metadata.file_size}, "
                                    f"date={metadata.creation_date.isoformat()}, "
                                    f"camera={metadata.camera_model or 'unknown'}")
                    return True

                # Determine if file needs conversion (HEIC/HEIF)
                needs_conversion = filepath.suffix.lower() in {'.heic', '.heif'}

                with tempfile.NamedTemporaryFile(suffix=filepath.suffix) as temp_file:
                    temp_path = Path(temp_file.name)
                    
                    if needs_conversion:
                        # For HEIC/HEIF files, convert and preserve metadata
                        converted_path = self._process_heic(filepath)
                        if not converted_path:
                            raise Exception("HEIC/HEIF conversion failed")
                        temp_path = converted_path
                    else:
                        # For other formats, just copy the file
                        shutil.copy2(filepath, temp_path)
                    
                    blob = self.bucket.blob(destination_path)
                    
                    # Configure retry on specific exceptions
                    retry_config = retry.Retry(
                        predicate=retry.if_exception_type(
                            ConnectionError,
                            ConnectionAbortedError,
                            ConnectionResetError,
                            TimeoutError
                        ),
                        initial=delay,
                        maximum=delay * 4,
                        multiplier=2,
                        deadline=300
                    )
                    
                    # Upload with retry configuration
                    blob.upload_from_filename(
                        str(temp_path),
                        retry=retry_config,
                        timeout=60
                    )
                    
                    blob.metadata = {
                        'upload_date': datetime.now().isoformat(),
                        'photo_date': metadata.creation_date.isoformat(),
                        'file_hash': metadata.file_hash,
                        'original_name': filepath.name,
                        'camera_model': metadata.camera_model or 'unknown'
                    }
                    blob.patch()
                    
                    # Clean up converted file if it exists
                    if needs_conversion and os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                self.logger.info(f"Uploaded: {filepath.name} -> {destination_path}")
                metadata.cloud_path = destination_path
                self.duplicate_detector.add_cloud_file(destination_path, metadata)
                self.upload_history[str(filepath)] = f"Uploaded to {destination_path}"
                
                return True

            except Exception as e:
                attempt += 1
                if attempt <= max_retries:
                    self.logger.warning(
                        f"Upload attempt {attempt}/{max_retries} failed for {filepath.name}: {str(e)}. "
                        f"Retrying in {delay:.1f} seconds..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    self.logger.error(
                        f"Failed to process {filepath.name} after {max_retries} attempts: {str(e)}"
                    )
                    return False

        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Upload photos to Google Cloud Storage')
    parser.add_argument('--bucket', help='Destination bucket name (override PHOTO_UPLOADER_BUCKET)')
    parser.add_argument('--concurrent', type=int, help='Number of concurrent uploads (override PHOTO_UPLOADER_CONCURRENT)')
    parser.add_argument('--project-id', help='Google Cloud project ID (override GOOGLE_CLOUD_PROJECT)')
    parser.add_argument('--dry-run', action='store_true', help='Perform a dry run without actual uploads')
    
    args = parser.parse_args()
    
    config = {
        'bucket_name': args.bucket or os.environ.get('PHOTO_UPLOADER_BUCKET'),
        'directory_path': '/photos',
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
        config['dry_run'],
        config['directory_path']
    )
    uploader.process_directory(directory_path)

if __name__ == '__main__':
    main()