import hmac, hashlib, time


def make_callback_token(job_id: int, secret: str, ttl_s: int) -> str:
    exp = int(time.time()) + ttl_s
    msg = f"{job_id}.{exp}".encode()
    sig = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()
    return f"{exp}.{sig}"

def verify_callback_token(token: str, job_id: int, secret: str) -> bool:
    try:
        exp_str, sig = token.split(".", 1)
        exp = int(exp_str)
    except Exception:
        return False
    if exp < int(time.time()):
        return False
    msg = f"{job_id}.{exp}".encode()
    expected = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()
    # timing-safe compare
    return hmac.compare_digest(expected, sig)
