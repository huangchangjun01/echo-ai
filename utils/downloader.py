import requests


def download_file(url: str) -> bytes:
    """Download the content at `url` and return raw bytes.

    Automatically adds http:// scheme if missing.
    Raises requests.HTTPError on non-200 responses.
    """
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content
