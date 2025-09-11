FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# (Optional) have bash, curl for debugging
# RUN apt-get update && apt-get install -y --no-install-recommends bash curl && \
#     rm -rf /var/lib/apt/lists/*

# Copy only the ml/ subtree
COPY ./ml /app

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
