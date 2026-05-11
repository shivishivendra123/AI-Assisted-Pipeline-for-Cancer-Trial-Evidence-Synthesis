from __future__ import annotations
import google.auth
import google.auth.transport.requests as gar

def google_access_token() -> str:
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    req = gar.Request()
    creds.refresh(req)
    return creds.token
