## hash of name, id and facedb embeddings
import hashlib
import os
from dotenv import load_dotenv

load_dotenv()

HASH_ALGORITHM = os.getenv('HASH_ALGORITHM', 'sha256')

def hash_name(name):
    """
    Generate hash of name for unique identifier using algorithm from .env.
    """
    h = hashlib.new(HASH_ALGORITHM)
    h.update(name.encode())
    return h.hexdigest()

def hash_embedding(embedding):
    """
    Generate hash of face embedding for verification using algorithm from .env.
    """
    h = hashlib.new(HASH_ALGORITHM)
    h.update(embedding.tobytes())
    return h.hexdigest()

def name_exists(known_embeddings, name):
    """
    Check if name already exists in database by comparing original names.
    """
    for user_data in known_embeddings.values():
        if isinstance(user_data, dict) and user_data.get('name') == name:
            return True
    return False