"""
FaaSLoRA S3 Storage Client

Provides interface for storing and retrieving LoRA artifacts from S3-compatible storage.
"""

import os
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    from botocore.config import Config
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    # Mock classes for development
    class ClientError(Exception): pass
    class NoCredentialsError(Exception): pass
    class Config: pass

from ..utils.config import Config as FaaSConfig
from ..utils.logger import get_logger


@dataclass
class S3Object:
    """S3 object metadata"""
    key: str
    size: int
    last_modified: float
    etag: str
    storage_class: str = "STANDARD"
    metadata: Dict[str, str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class UploadProgress:
    """Upload progress information"""
    bytes_transferred: int
    total_bytes: int
    percentage: float
    speed_bps: float = 0.0
    eta_seconds: float = 0.0


class S3Client:
    """
    S3 storage client for LoRA artifacts
    
    Provides high-level interface for storing and retrieving LoRA artifacts
    from S3-compatible storage services.
    """
    
    def __init__(self, config: FaaSConfig):
        """
        Initialize S3 client
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # S3 configuration
        storage_config = config.get('storage', {})
        remote_config = (
            storage_config.get('remote', {})
            if isinstance(storage_config.get('remote', {}), dict)
            else {}
        )
        s3_config = (
            storage_config.get('s3', {})
            if isinstance(storage_config.get('s3', {}), dict)
            else {}
        )
        remote_provider = str(
            remote_config.get(
                'provider',
                storage_config.get('backend', storage_config.get('provider', 's3')),
            )
        ).lower()
        merged_s3_config = {}
        if remote_provider in {'s3', 'oss', 'gcs'}:
            merged_s3_config.update(remote_config)
        merged_s3_config.update(s3_config)
        
        self.bucket_name = merged_s3_config.get(
            'bucket_name',
            merged_s3_config.get('bucket', 'faaslora-artifacts'),
        )
        self.region = merged_s3_config.get('region', 'us-east-1')
        self.endpoint_url = merged_s3_config.get('endpoint_url')  # For S3-compatible services
        self.access_key = (
            merged_s3_config.get('access_key_id')
            or merged_s3_config.get('access_key')
            or os.getenv('AWS_ACCESS_KEY_ID')
        )
        self.secret_key = (
            merged_s3_config.get('secret_access_key')
            or merged_s3_config.get('secret_key')
            or os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        
        # Performance settings
        self.multipart_threshold = merged_s3_config.get('multipart_threshold', 64 * 1024 * 1024)  # 64MB
        self.multipart_chunksize = merged_s3_config.get('multipart_chunksize', 16 * 1024 * 1024)  # 16MB
        self.max_concurrency = merged_s3_config.get('max_concurrency', 10)
        self.max_bandwidth = merged_s3_config.get('max_bandwidth')  # bytes per second
        
        # Retry settings
        self.max_retries = merged_s3_config.get('max_retries', 3)
        self.retry_delay = merged_s3_config.get('retry_delay', 1.0)
        
        # Client state
        self.s3_client = None
        self.s3_resource = None
        self.is_initialized = False
        
        if not BOTO3_AVAILABLE:
            self.logger.warning("boto3 not available, S3 client will not function")
    
    async def initialize(self):
        """Initialize S3 client and verify connection"""
        if not BOTO3_AVAILABLE:
            raise RuntimeError("Cannot initialize S3 client: boto3 not available")
        
        if self.is_initialized:
            return
        
        try:
            self.logger.info("Initializing S3 client...")
            
            # Configure boto3
            boto_config = Config(
                region_name=self.region,
                retries={'max_attempts': self.max_retries},
                max_pool_connections=self.max_concurrency
            )
            
            # Create S3 client
            client_kwargs = {
                'config': boto_config,
                'aws_access_key_id': self.access_key,
                'aws_secret_access_key': self.secret_key
            }
            
            if self.endpoint_url:
                client_kwargs['endpoint_url'] = self.endpoint_url
            
            self.s3_client = boto3.client('s3', **client_kwargs)
            self.s3_resource = boto3.resource('s3', **client_kwargs)
            
            # Verify connection and bucket access
            await self._verify_bucket_access()
            
            self.is_initialized = True
            self.logger.info("S3 client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize S3 client: {e}")
            raise
    
    async def _verify_bucket_access(self):
        """Verify bucket exists and is accessible"""
        try:
            # Check if bucket exists
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            self.logger.info(f"Verified access to bucket: {self.bucket_name}")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # Try to create bucket
                self.logger.info(f"Bucket {self.bucket_name} not found, attempting to create...")
                await self._create_bucket()
            else:
                self.logger.error(f"Error accessing bucket {self.bucket_name}: {e}")
                raise
    
    async def _create_bucket(self):
        """Create S3 bucket"""
        try:
            if self.region == 'us-east-1':
                # us-east-1 doesn't need LocationConstraint
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            
            self.logger.info(f"Created bucket: {self.bucket_name}")
            
        except ClientError as e:
            self.logger.error(f"Failed to create bucket {self.bucket_name}: {e}")
            raise
    
    async def upload_artifact(self, 
                            artifact_id: str, 
                            file_path: str,
                            metadata: Optional[Dict[str, str]] = None,
                            progress_callback: Optional[callable] = None) -> S3Object:
        """
        Upload LoRA artifact to S3
        
        Args:
            artifact_id: Unique artifact identifier
            file_path: Local file path to upload
            metadata: Additional metadata to store
            progress_callback: Optional progress callback function
            
        Returns:
            S3Object with upload information
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Generate S3 key
            s3_key = f"artifacts/{artifact_id}"
            
            # Get file info
            file_size = os.path.getsize(file_path)
            
            # Prepare metadata
            upload_metadata = {
                'artifact_id': artifact_id,
                'upload_time': str(time.time()),
                'file_size': str(file_size)
            }
            if metadata:
                upload_metadata.update(metadata)
            
            self.logger.info(f"Uploading artifact {artifact_id} to S3 (size: {file_size} bytes)")
            
            # Choose upload method based on file size
            if file_size > self.multipart_threshold:
                s3_object = await self._multipart_upload(
                    s3_key, file_path, upload_metadata, progress_callback
                )
            else:
                s3_object = await self._simple_upload(
                    s3_key, file_path, upload_metadata, progress_callback
                )
            
            self.logger.info(f"Successfully uploaded artifact {artifact_id}")
            return s3_object
            
        except Exception as e:
            self.logger.error(f"Failed to upload artifact {artifact_id}: {e}")
            raise
    
    async def _simple_upload(self, 
                           s3_key: str, 
                           file_path: str, 
                           metadata: Dict[str, str],
                           progress_callback: Optional[callable] = None) -> S3Object:
        """Simple upload for small files"""
        start_time = time.time()
        file_size = os.path.getsize(file_path)
        
        def progress_hook(bytes_transferred):
            if progress_callback:
                elapsed = time.time() - start_time
                speed = bytes_transferred / elapsed if elapsed > 0 else 0
                progress = UploadProgress(
                    bytes_transferred=bytes_transferred,
                    total_bytes=file_size,
                    percentage=(bytes_transferred / file_size) * 100,
                    speed_bps=speed,
                    eta_seconds=(file_size - bytes_transferred) / speed if speed > 0 else 0
                )
                progress_callback(progress)
        
        # Upload file
        with open(file_path, 'rb') as f:
            self.s3_client.upload_fileobj(
                f, self.bucket_name, s3_key,
                ExtraArgs={'Metadata': metadata},
                Callback=progress_hook
            )
        
        # Get object info
        response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
        
        return S3Object(
            key=s3_key,
            size=response['ContentLength'],
            last_modified=response['LastModified'].timestamp(),
            etag=response['ETag'].strip('"'),
            metadata=response.get('Metadata', {})
        )
    
    async def _multipart_upload(self, 
                              s3_key: str, 
                              file_path: str, 
                              metadata: Dict[str, str],
                              progress_callback: Optional[callable] = None) -> S3Object:
        """Multipart upload for large files"""
        start_time = time.time()
        file_size = os.path.getsize(file_path)
        bytes_transferred = 0
        
        # Initiate multipart upload
        response = self.s3_client.create_multipart_upload(
            Bucket=self.bucket_name,
            Key=s3_key,
            Metadata=metadata
        )
        upload_id = response['UploadId']
        
        try:
            parts = []
            part_number = 1
            
            with open(file_path, 'rb') as f:
                while True:
                    # Read chunk
                    chunk = f.read(self.multipart_chunksize)
                    if not chunk:
                        break
                    
                    # Upload part
                    part_response = self.s3_client.upload_part(
                        Bucket=self.bucket_name,
                        Key=s3_key,
                        PartNumber=part_number,
                        UploadId=upload_id,
                        Body=chunk
                    )
                    
                    parts.append({
                        'ETag': part_response['ETag'],
                        'PartNumber': part_number
                    })
                    
                    # Update progress
                    bytes_transferred += len(chunk)
                    if progress_callback:
                        elapsed = time.time() - start_time
                        speed = bytes_transferred / elapsed if elapsed > 0 else 0
                        progress = UploadProgress(
                            bytes_transferred=bytes_transferred,
                            total_bytes=file_size,
                            percentage=(bytes_transferred / file_size) * 100,
                            speed_bps=speed,
                            eta_seconds=(file_size - bytes_transferred) / speed if speed > 0 else 0
                        )
                        progress_callback(progress)
                    
                    part_number += 1
            
            # Complete multipart upload
            self.s3_client.complete_multipart_upload(
                Bucket=self.bucket_name,
                Key=s3_key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )
            
            # Get object info
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            
            return S3Object(
                key=s3_key,
                size=response['ContentLength'],
                last_modified=response['LastModified'].timestamp(),
                etag=response['ETag'].strip('"'),
                metadata=response.get('Metadata', {})
            )
            
        except Exception as e:
            # Abort multipart upload on error
            try:
                self.s3_client.abort_multipart_upload(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    UploadId=upload_id
                )
            except:
                pass
            raise e
    
    async def download_artifact(self, 
                              artifact_id: str, 
                              local_path: str,
                              progress_callback: Optional[callable] = None) -> bool:
        """
        Download LoRA artifact from S3
        
        Args:
            artifact_id: Artifact identifier
            local_path: Local path to save file
            progress_callback: Optional progress callback function
            
        Returns:
            True if download successful
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            s3_key = f"artifacts/{artifact_id}"
            
            # Check if object exists
            try:
                response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
                file_size = response['ContentLength']
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    self.logger.error(f"Artifact {artifact_id} not found in S3")
                    return False
                raise
            
            self.logger.info(f"Downloading artifact {artifact_id} from S3 (size: {file_size} bytes)")
            
            # Create directory if needed
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download with progress tracking
            start_time = time.time()
            bytes_transferred = 0
            
            def progress_hook(chunk_size):
                nonlocal bytes_transferred
                bytes_transferred += chunk_size
                if progress_callback:
                    elapsed = time.time() - start_time
                    speed = bytes_transferred / elapsed if elapsed > 0 else 0
                    progress = UploadProgress(
                        bytes_transferred=bytes_transferred,
                        total_bytes=file_size,
                        percentage=(bytes_transferred / file_size) * 100,
                        speed_bps=speed,
                        eta_seconds=(file_size - bytes_transferred) / speed if speed > 0 else 0
                    )
                    progress_callback(progress)
            
            # Download file
            with open(local_path, 'wb') as f:
                self.s3_client.download_fileobj(
                    self.bucket_name, s3_key, f,
                    Callback=progress_hook
                )
            
            self.logger.info(f"Successfully downloaded artifact {artifact_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download artifact {artifact_id}: {e}")
            return False
    
    async def delete_artifact(self, artifact_id: str) -> bool:
        """
        Delete LoRA artifact from S3
        
        Args:
            artifact_id: Artifact identifier
            
        Returns:
            True if deletion successful
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            s3_key = f"artifacts/{artifact_id}"
            
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            self.logger.info(f"Deleted artifact {artifact_id} from S3")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete artifact {artifact_id}: {e}")
            return False
    
    async def list_artifacts(self, prefix: str = "", limit: int = 1000) -> List[S3Object]:
        """
        List LoRA artifacts in S3
        
        Args:
            prefix: Key prefix to filter by
            limit: Maximum number of objects to return
            
        Returns:
            List of S3Object instances
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            s3_prefix = f"artifacts/{prefix}" if prefix else "artifacts/"
            
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=s3_prefix,
                PaginationConfig={'MaxItems': limit}
            )
            
            objects = []
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        objects.append(S3Object(
                            key=obj['Key'],
                            size=obj['Size'],
                            last_modified=obj['LastModified'].timestamp(),
                            etag=obj['ETag'].strip('"'),
                            storage_class=obj.get('StorageClass', 'STANDARD')
                        ))
            
            return objects
            
        except Exception as e:
            self.logger.error(f"Failed to list artifacts: {e}")
            return []
    
    async def get_artifact_info(self, artifact_id: str) -> Optional[S3Object]:
        """
        Get artifact information from S3
        
        Args:
            artifact_id: Artifact identifier
            
        Returns:
            S3Object if found, None otherwise
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            s3_key = f"artifacts/{artifact_id}"
            
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            
            return S3Object(
                key=s3_key,
                size=response['ContentLength'],
                last_modified=response['LastModified'].timestamp(),
                etag=response['ETag'].strip('"'),
                metadata=response.get('Metadata', {})
            )
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return None
            raise
        except Exception as e:
            self.logger.error(f"Failed to get artifact info for {artifact_id}: {e}")
            return None
    
    async def cleanup(self):
        """Cleanup S3 client resources"""
        try:
            if self.s3_client:
                # Close any open connections
                self.s3_client.close()
            
            self.is_initialized = False
            self.logger.info("S3 client cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during S3 client cleanup: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get S3 client statistics"""
        return {
            "initialized": self.is_initialized,
            "bucket_name": self.bucket_name,
            "region": self.region,
            "endpoint_url": self.endpoint_url,
            "multipart_threshold": self.multipart_threshold,
            "multipart_chunksize": self.multipart_chunksize,
            "max_concurrency": self.max_concurrency
        }
