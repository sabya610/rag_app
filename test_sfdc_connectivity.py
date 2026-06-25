#!/usr/bin/env python3
"""
SFDC Connectivity Test - run inside the K8s pod to diagnose connection issues.
Usage: python test_sfdc_connectivity.py [--sid-file /path/to/sid.txt]
"""
import os, sys, socket, ssl, json, time

SF_URL = os.getenv("SF_URL", "https://hp.my.salesforce.com")
SF_SID = os.getenv("SF_SID", "")
SF_SID_FILE = os.getenv("SF_SID_FILE", "")

# Try to load SID from file
if not SF_SID and SF_SID_FILE and os.path.isfile(SF_SID_FILE):
    SF_SID = open(SF_SID_FILE).read().strip()
if not SF_SID and os.path.isfile("/etc/sfdc/sid.txt"):
    SF_SID = open("/etc/sfdc/sid.txt").read().strip()

# Also accept --sid-file argument
for i, a in enumerate(sys.argv):
    if a == "--sid-file" and i + 1 < len(sys.argv):
        SF_SID = open(sys.argv[i + 1]).read().strip()

from urllib.parse import urlparse
parsed = urlparse(SF_URL)
host = parsed.hostname
port = parsed.port or 443

print(f"{'='*60}")
print(f"SFDC Connectivity Test")
print(f"{'='*60}")
print(f"SF_URL     : {SF_URL}")
print(f"Host       : {host}:{port}")
print(f"SF_SID     : {'SET (' + str(len(SF_SID)) + ' chars)' if SF_SID else 'NOT SET'}")
print(f"HTTP_PROXY : {os.getenv('HTTP_PROXY', os.getenv('http_proxy', 'NOT SET'))}")
print(f"HTTPS_PROXY: {os.getenv('HTTPS_PROXY', os.getenv('https_proxy', 'NOT SET'))}")
print(f"NO_PROXY   : {os.getenv('NO_PROXY', os.getenv('no_proxy', 'NOT SET'))[:80]}...")
print()

# Test 1: DNS resolution
print("[Test 1] DNS Resolution...")
try:
    ips = socket.getaddrinfo(host, port)
    ip = ips[0][4][0]
    print(f"  OK - {host} -> {ip}")
except Exception as e:
    print(f"  FAIL - {e}")

# Test 2: TCP connection
print("[Test 2] TCP Connection...")
try:
    s = socket.create_connection((host, port), timeout=10)
    print(f"  OK - TCP connection to {host}:{port}")
    s.close()
except Exception as e:
    print(f"  FAIL - {e}")

# Test 3: SSL/TLS handshake
print("[Test 3] SSL/TLS Handshake...")
try:
    ctx = ssl.create_default_context()
    with socket.create_connection((host, port), timeout=10) as sock:
        with ctx.wrap_socket(sock, server_hostname=host) as ssock:
            cert = ssock.getpeercert()
            print(f"  OK - TLS {ssock.version()}, cert subject: {cert.get('subject', '?')}")
except Exception as e:
    print(f"  FAIL - {e}")
    print(f"  (Corporate proxy may be intercepting SSL)")

# Test 4: HTTP GET via requests (respects proxy env vars)
print("[Test 4] HTTPS GET via requests (with proxy if set)...")
try:
    import requests
    r = requests.get(f"{SF_URL}/services/data/", timeout=15)
    print(f"  OK - HTTP {r.status_code}, {len(r.content)} bytes")
    if r.status_code == 200:
        print(f"  Response: {r.text[:200]}")
except Exception as e:
    print(f"  FAIL - {e}")

# Test 5: HTTP GET bypassing proxy
print("[Test 5] HTTPS GET bypassing proxy...")
try:
    import requests
    r = requests.get(f"{SF_URL}/services/data/", timeout=15,
                     proxies={"http": None, "https": None})
    print(f"  OK - HTTP {r.status_code}, {len(r.content)} bytes")
except Exception as e:
    print(f"  FAIL - {e}")

# Test 6: API auth with SID
if SF_SID:
    print("[Test 6] SFDC API Auth (session ID)...")
    try:
        import requests
        headers = {
            "Authorization": f"Bearer {SF_SID}",
            "Content-Type": "application/json"
        }
        # Try with proxy
        r = requests.get(f"{SF_URL}/services/data/v59.0/limits",
                        headers=headers, timeout=15)
        print(f"  OK - HTTP {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"  Authenticated! API calls remaining: {data.get('DailyApiRequests', {}).get('Remaining', '?')}")
        elif r.status_code == 401:
            print(f"  FAIL - 401 Unauthorized (SID expired or invalid)")
            print(f"  Response: {r.text[:300]}")
        else:
            print(f"  Response: {r.text[:300]}")
    except Exception as e:
        print(f"  FAIL - {e}")

    print("[Test 7] SFDC API Auth (no proxy)...")
    try:
        import requests
        r = requests.get(f"{SF_URL}/services/data/v59.0/limits",
                        headers=headers, timeout=15,
                        proxies={"http": None, "https": None})
        print(f"  OK - HTTP {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"  Authenticated! API calls remaining: {data.get('DailyApiRequests', {}).get('Remaining', '?')}")
        elif r.status_code == 401:
            print(f"  FAIL - SID expired/invalid")
        else:
            print(f"  Response: {r.text[:300]}")
    except Exception as e:
        print(f"  FAIL - {e}")
else:
    print("[Test 6] SKIPPED - No SID available")

print(f"\n{'='*60}")
print("Done.")
