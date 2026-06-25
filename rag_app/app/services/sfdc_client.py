"""
Salesforce REST API Client for RAG App.

Supports two authentication modes:
1. Session ID (Bearer token from browser cookie - used by sf_case_exporter)
2. OAuth2 password flow (client_id + client_secret + username + password)

Usage:
    from app.services.sfdc_client import get_sfdc_client
    sf = get_sfdc_client()
    results = sf.query("SELECT Id, Title FROM Knowledge__kav WHERE PublishStatus='Online' LIMIT 5")
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


class SalesforceError(Exception):
    """Exception for Salesforce API errors."""

    def __init__(self, status: int, content: str, url: str = ""):
        self.status = status
        self.content = content
        self.url = url
        super().__init__(f"Salesforce API error {status}: {content}")


class SalesforceClient:
    """
    Minimal Salesforce REST API client.
    Supports session_id auth and OAuth2 password flow.
    """

    DEFAULT_API_VERSION = "59.0"

    def __init__(
        self,
        instance_url: str,
        session_id: Optional[str] = None,
        access_token: Optional[str] = None,
        version: str = DEFAULT_API_VERSION,
    ):
        self.sf_version = version
        token = session_id or access_token
        if not token:
            raise ValueError("Either session_id or access_token must be provided")

        parsed = urlparse(instance_url)
        self.sf_instance = parsed.hostname
        if parsed.port and parsed.port != 443:
            self.sf_instance = f"{self.sf_instance}:{parsed.port}"

        self.base_url = f"https://{self.sf_instance}/services/data/v{self.sf_version}/"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "X-PrettyPrint": "1",
        }
        self.session = requests.Session()

    def _call_salesforce(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        headers = self.headers.copy()
        additional_headers = kwargs.pop("headers", {})
        headers.update(additional_headers or {})
        result = self.session.request(method, url, headers=headers, timeout=30, **kwargs)
        if result.status_code >= 300:
            try:
                error_content = result.json()
            except Exception:
                error_content = result.text
            raise SalesforceError(result.status_code, str(error_content), url)
        return result

    def query(self, soql: str, include_deleted: bool = False) -> Dict[str, Any]:
        endpoint = "queryAll/" if include_deleted else "query/"
        url = self.base_url + endpoint
        result = self._call_salesforce("GET", url, params={"q": soql})
        return result.json()

    def query_more(self, next_records_url: str) -> Dict[str, Any]:
        url = f"https://{self.sf_instance}{next_records_url}"
        result = self._call_salesforce("GET", url)
        return result.json()

    def query_all(self, soql: str, include_deleted: bool = False) -> Dict[str, Any]:
        all_records: list = []
        result = self.query(soql, include_deleted=include_deleted)
        while True:
            all_records.extend(result["records"])
            if result["done"]:
                break
            result = self.query_more(result["nextRecordsUrl"])
        return {"records": all_records, "totalSize": len(all_records), "done": True}

    def search(self, sosl: str) -> List[Dict[str, Any]]:
        """Execute a SOSL search query."""
        url = self.base_url + "search/"
        result = self._call_salesforce("GET", url, params={"q": sosl})
        return result.json().get("searchRecords", [])

    def get_sobject(self, sobject: str, record_id: str) -> Dict[str, Any]:
        """Get a single SObject record by ID."""
        url = f"{self.base_url}sobjects/{sobject}/{record_id}"
        result = self._call_salesforce("GET", url)
        return result.json()

    def test_connection(self) -> bool:
        """Test if the connection is valid by querying a KA record."""
        try:
            url = self.base_url + "query/"
            self._call_salesforce("GET", url, params={
                "q": "SELECT Id FROM Knowledge__kav WHERE PublishStatus='Online' LIMIT 1"
            })
            return True
        except Exception as e:
            logger.error("SFDC connection test failed: %s", e)
            return False

    def describe_sobject(self, sobject: str) -> Dict[str, Any]:
        """Describe an SObject to list its fields and metadata."""
        url = f"{self.base_url}sobjects/{sobject}/describe/"
        result = self._call_salesforce("GET", url)
        return result.json()


def _read_sid_file(path: str) -> str:
    """Read session ID from a file, stripping whitespace. Works on Windows and Linux."""
    sid_path = Path(path).expanduser().resolve()
    return sid_path.read_text(encoding="utf-8").strip()


def _find_sid_file() -> Optional[str]:
    """
    Search for sid.txt in platform-agnostic locations.
    Returns the first existing path, or None.
    """
    candidates = [
        os.getenv("SF_SID_FILE", ""),
        # K8s secret mount
        "/etc/sfdc/sid.txt",
        # Linux home
        str(Path.home() / "sid.txt"),
        # Windows common locations
        str(Path.home() / "sid.txt"),
    ]
    for p in candidates:
        if p and Path(p).is_file():
            return p
    return None


def authenticate_oauth2(
    login_url: str,
    client_id: str,
    client_secret: str,
    username: str,
    password: str,
    security_token: str = "",
) -> tuple:
    """
    Authenticate via OAuth2 password flow.
    Returns (access_token, instance_url).
    """
    payload = {
        "grant_type": "password",
        "client_id": client_id,
        "client_secret": client_secret,
        "username": username,
        "password": password + security_token,
    }
    token_url = f"{login_url}/services/oauth2/token"
    response = requests.post(token_url, data=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data["access_token"], data["instance_url"]


def get_sfdc_client() -> Optional[SalesforceClient]:
    """
    Build a SalesforceClient from environment variables.

    Auth priority:
    1. SF_SID / SF_SID_FILE  → session ID auth
    2. SFDC_CLIENT_ID + SFDC_CLIENT_SECRET + SFDC_USERNAME + SFDC_PASSWORD → OAuth2

    Returns None if no credentials are configured.
    """
    sf_url = os.getenv("SF_URL", "https://hp.my.salesforce.com")

    # --- Mode 1: Session ID ---
    sid = os.getenv("SF_SID", "")
    sid_file = os.getenv("SF_SID_FILE", "")

    # Try explicit SID_FILE first
    if sid_file and Path(sid_file).is_file():
        sid = _read_sid_file(sid_file)
    # If SF_SID itself is a file path, read it
    elif sid and Path(sid).is_file():
        sid = _read_sid_file(sid)
    # Auto-discover sid.txt in known locations
    elif not sid:
        found = _find_sid_file()
        if found:
            logger.info("[SFDC] Auto-discovered SID file at: %s", found)
            sid = _read_sid_file(found)

    if sid:
        logger.info("[SFDC] Authenticating with session ID")
        return SalesforceClient(instance_url=sf_url, session_id=sid)

    # --- Mode 2: OAuth2 ---
    client_id = os.getenv("SFDC_CLIENT_ID", "")
    client_secret = os.getenv("SFDC_CLIENT_SECRET", "")
    username = os.getenv("SFDC_USERNAME", "")
    password = os.getenv("SFDC_PASSWORD", "")
    security_token = os.getenv("SFDC_SECURITY_TOKEN", "")
    login_url = os.getenv("SFDC_LOGIN_URL", "https://login.salesforce.com")

    if client_id and client_secret and username and password:
        logger.info("[SFDC] Authenticating with OAuth2 password flow")
        try:
            access_token, instance_url = authenticate_oauth2(
                login_url, client_id, client_secret, username, password, security_token
            )
            return SalesforceClient(instance_url=instance_url, access_token=access_token)
        except Exception as e:
            logger.error("[SFDC] OAuth2 authentication failed: %s", e)
            return None

    logger.warning("[SFDC] No credentials configured. SFDC features disabled.")
    return None
