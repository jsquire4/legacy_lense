"""Shared SSE (Server-Sent Events) formatting utility."""

import json


def sse_event(event: str, data: dict) -> str:
    """Format a single SSE event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
