import re
from urllib.parse import urlparse


def _parse_s3_uri(uri: str):
    # supports s3://bucket/key or https://bucket.s3.amazonaws.com/key
    if uri.startswith("s3://"):
        m = re.match(r"^s3://([^/]+)/(.+)$", uri)
        if not m: raise ValueError("Bad s3 uri")
        return m.group(1), m.group(2)
    if "amazonaws.com" in uri:
        p = urlparse(uri)
        # virtual-hosted–style: https://bucket.s3.region.amazonaws.com/key
        host_parts = p.netloc.split(".")
        bucket = host_parts[0]
        key = p.path.lstrip("/")
        return bucket, key
    raise ValueError("Unsupported CSV URI; must be s3:// or S3 https URL")

