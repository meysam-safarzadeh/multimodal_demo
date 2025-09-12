# backend/registry/permissions.py
from rest_framework.permissions import BasePermission, SAFE_METHODS
from django.conf import settings
from registry.utils.callback import verify_callback_token


class HasValidCallbackForWrite(BasePermission):
    """
    - GET is allowed (keep your current policy).
    - POST/PATCH/PUT/DELETE require a valid token in header or query.
    """
    def has_permission(self, request, view):
        if request.method in SAFE_METHODS:
            return True
        token = request.headers.get("X-Callback-Token") or request.query_params.get("token")
        job_id = view.kwargs.get(view.lookup_url_kwarg or 'pk')
        if not token or not job_id:
            return False
        try:
            job_id = int(job_id)
        except ValueError:
            return False
        return verify_callback_token(
            token,
            job_id,
            getattr(settings, "CALLBACK_SECRET", "")
        )
