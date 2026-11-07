"""Small per-trial relay: a fixed model route and public HTTP(S) egress.

The task container has only an internal Docker network. This relay is its only
peer with egress. Public destinations are resolved once, checked, then dialled by
IP to avoid DNS rebinding; private/control-plane destinations are rejected.
Only ports 80/443 are supported for task downloads. This is not a general VPN.
"""
from __future__ import annotations

import http.client
import http.server
import ipaddress
import os
import select
import socket
import threading
import time
import urllib.parse

MAX_BODY = 512_000
MAX_TRANSFER = 512 * 1024 * 1024
HOP_HEADERS = {"connection", "proxy-connection", "proxy-authorization", "keep-alive", "te", "trailer", "transfer-encoding", "upgrade"}


def public_addresses(host: str, port: int) -> list[str]:
    if port not in {80, 443}:
        raise ValueError("Only public HTTP(S) ports are allowed")
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not addresses:
        raise ValueError("Destination has no address")
    ips = list(dict.fromkeys(item[4][0] for item in addresses))
    if any(not ipaddress.ip_address(ip).is_global for ip in ips):
        raise ValueError("Private, local, reserved and metadata addresses are blocked")
    return ips


def public_address(host: str, port: int) -> str:
    return public_addresses(host, port)[0]


def connect_public(host: str, port: int):
    """Try the validated DNS snapshot, within one bounded connection budget.

    A failed first CDN address must not become a permanent policy denial. All
    addresses are checked before any connection, including unused fallbacks.
    Numeric destinations avoid a second hostname lookup at the dial boundary.
    """
    ips = public_addresses(host, port)
    deadline = time.monotonic() + 15
    last_error = None
    for ip in ips[:16]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            connection = socket.create_connection((ip, port), timeout=min(5, remaining))
            connection.settimeout(15)
            return connection
        except OSError as error:
            last_error = error
    raise OSError("No validated public address was reachable") from last_error


def public_http_connection(host: str, port: int):
    # Assign the already connected socket: HTTPConnection must never resolve
    # the original hostname again after our destination validation.
    connection = http.client.HTTPConnection(host, port, timeout=100)
    connection.sock = connect_public(host, port)
    connection.sock.settimeout(100)
    return connection


class Relay(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    gateway = ""
    expires_at = float("inf")

    def log_message(self, *_):
        # Tokens, request bodies and user task text must not enter relay logs.
        pass

    def _error(self, code, message):
        self.close_connection = True
        self.send_error(code, message)

    def _body(self):
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("Chunked request bodies are unsupported")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) > 1:
            raise ValueError("Duplicate Content-Length")
        size = int(lengths[0]) if lengths else 0
        if size < 0 or size > MAX_BODY:
            raise ValueError("Request body too large")
        self.connection.settimeout(100)
        data = self.rfile.read(size)
        if len(data) != size:
            raise ValueError("Truncated body")
        return data

    def _request(self):
        if time.time() >= self.expires_at:
            return self._error(410, "Trial expired")
        if self.path == "/health" and self.command == "GET":
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")
            return
        connection = None
        response_started = False
        try:
            body = self._body()
            model_request = self.path == "/v1/chat/completions" and self.command == "POST"
            if model_request:
                target = urllib.parse.urlsplit(self.gateway.rstrip("/") + "/chat/completions")
                # This is the sole deliberately private destination, fixed by the
                # trusted worker; the task cannot choose host, port or path.
                if target.scheme != "http":
                    raise ValueError("Local model gateway must use HTTP")
                connection = http.client.HTTPConnection(target.hostname, target.port, timeout=100)
                headers = {"Content-Type": "application/json", "Authorization": self.headers.get("Authorization", "")}
                path = target.path
            else:
                target = urllib.parse.urlsplit(self.path)
                if target.scheme != "http" or not target.hostname or target.username or target.password:
                    raise ValueError("Use absolute public HTTP URLs or CONNECT for HTTPS")
                port = target.port or 80
                connection = public_http_connection(target.hostname, port)
                headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS | {"host", "content-length", "authorization"}}
                headers["Host"] = target.netloc
                path = urllib.parse.urlunsplit(("", "", target.path or "/", target.query, ""))
            headers["Content-Length"] = str(len(body))
            headers["Connection"] = "close"
            connection.request(self.command, path, body=body, headers=headers)
            response = connection.getresponse()
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in HOP_HEADERS:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            response_started = True
            total = 0
            while chunk := response.read(65536):
                total += len(chunk)
                if total > MAX_TRANSFER or time.time() >= self.expires_at:
                    break
                self.wfile.write(chunk)
            self.close_connection = True
        except ValueError:
            if not response_started:
                self._error(403, "Destination or request is not permitted")
        except (OSError, http.client.HTTPException):
            if not response_started:
                self._error(502, "Upstream unavailable")
        finally:
            self.close_connection = True
            if connection:
                connection.close()

    do_GET = _request
    do_POST = _request
    do_HEAD = _request
    do_PUT = _request
    do_DELETE = _request

    def do_CONNECT(self):
        upstream = None
        established = False
        try:
            if time.time() >= self.expires_at:
                raise ValueError("Expired")
            target = urllib.parse.urlsplit("//" + self.path)
            if not target.hostname or target.port != 443 or target.username or target.password:
                raise ValueError("CONNECT requires public TLS port 443")
            upstream = connect_public(target.hostname, 443)
            self.send_response(200, "Connection Established")
            self.end_headers()
            established = True
            peers = [self.connection, upstream]
            total = 0
            deadline = min(time.time() + 180, self.expires_at)
            while time.time() < deadline and total < MAX_TRANSFER:
                ready, _, _ = select.select(peers, [], [], min(5, max(0, deadline - time.time())))
                for source in ready:
                    chunk = source.recv(65536)
                    if not chunk:
                        return
                    total += len(chunk)
                    (upstream if source is self.connection else self.connection).sendall(chunk)
        except ValueError:
            if not established:
                self._error(403, "Destination or request is not permitted")
        except OSError:
            if not established:
                self._error(502, "Upstream unavailable")
        finally:
            if upstream:
                upstream.close()
            self.close_connection = True


class BoundedServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *args, **kwargs):
        self.slots = threading.BoundedSemaphore(32)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


if __name__ == "__main__":
    Relay.gateway = os.environ["AOS_GATEWAY_URL"]
    Relay.expires_at = float(os.environ["AOS_EXPIRES_AT"])
    server = BoundedServer(("0.0.0.0", 8080), Relay)
    timer = threading.Timer(max(0.1, Relay.expires_at - time.time()), server.shutdown)
    timer.daemon = True
    timer.start()
    server.serve_forever(poll_interval=0.2)
