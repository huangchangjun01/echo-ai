"""Tests for the SSRF guard in utils.downloader."""
from __future__ import annotations

import pytest

from utils.downloader import _is_safe_url


def test_blocks_loopback_url():
    with pytest.raises(Exception):
        _is_safe_url("http://127.0.0.1/admin", allowed_subdomains=[])


def test_blocks_private_ip_literal():
    with pytest.raises(Exception):
        _is_safe_url("http://10.0.0.5/secret", allowed_subdomains=[])


def test_blocks_non_http_scheme():
    with pytest.raises(Exception):
        _is_safe_url("file:///etc/passwd", allowed_subdomains=[])


def test_allows_subdomain_match(monkeypatch):
    # Avoid real DNS by short-circuiting resolver.
    from utils import downloader

    monkeypatch.setattr(
        downloader,
        "_resolve_addresses",
        lambda host: {"93.184.216.34"},  # example.com public IP
    )
    # Should not raise.
    downloader._is_safe_url("http://tfpdkiq9g.hn-bkt.clouddn.com/x.png", allowed_subdomains=["hn-bkt.clouddn.com"])


def test_blocks_subdomain_mismatch(monkeypatch):
    from utils import downloader

    monkeypatch.setattr(
        downloader,
        "_resolve_addresses",
        lambda host: {"8.8.8.8"},
    )
    with pytest.raises(Exception):
        downloader._is_safe_url("http://attacker.com/x", allowed_subdomains=["hn-bkt.clouddn.com"])