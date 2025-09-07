from hmac import new as hnew
from hashlib import sha256
import time


def make_token(job_id: int, secret: str, ttl: int):
    exp = int(time.time()) + ttl
    sig = hnew(secret.encode(), f"{job_id}.{exp}".encode(), sha256).hexdigest()
    return f"{exp}.{sig}"