"""Content-addressed preprocessing identity; replacing a file invalidates its cache."""
from utils.research_io import file_sha256, object_sha256


def image_cache_key(item, preprocessing_identity):
    return object_sha256(dict(source_sha256=file_sha256(item['image']),
                              preprocessing=preprocessing_identity,format_version=2)).encode('ascii')
