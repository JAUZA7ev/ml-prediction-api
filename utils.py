import os
import tensorflow as tf
from google.cloud import storage
from pathlib import Path

def load_model_from_gcs(bucket_name, model_path):
    """Load a TensorFlow model from Google Cloud Storage, with local caching."""
    local_model_dir = Path(model_path)
    if not local_model_dir.exists():
        os.makedirs(local_model_dir, exist_ok=True)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=model_path)

        # Download model files
        for blob in blobs:
            local_file_path = local_model_dir / blob.name.split('/')[-1]
            blob.download_to_filename(str(local_file_path))

    # Load the model
    return tf.keras.models.load_model(local_model_dir)
