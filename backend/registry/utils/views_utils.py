def _job_artifacts_list(job):
    """
    Extract the artifacts list from your JSONField.
    Expected shape: list[{"key": str, "s3_uri": str}, ...]
    Adjust the attribute name if yours isn't 'artifacts'.
    """
    data = getattr(job, "artifacts", None)  # <-- change to your actual JSONField name if needed
    return getattr(data, "s3_uri", [])