"""Public-egress failures must not weaken address validation or corrupt TLS."""
import socket
from unittest.mock import Mock

import pytest

from service.sandbox import relay


def dns(*ips):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443)) for ip in ips]


def test_public_failover_uses_one_validated_dns_snapshot(monkeypatch):
    lookup = Mock(return_value=dns("1.1.1.1", "8.8.8.8", "8.8.8.8"))
    connected = Mock()
    dial = Mock(side_effect=[OSError("first address unavailable"), connected])
    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    monkeypatch.setattr(socket, "create_connection", dial)
    assert relay.connect_public("cdn.example", 443) is connected
    assert lookup.call_count == 1
    assert [c.args[0] for c in dial.call_args_list] == [("1.1.1.1", 443), ("8.8.8.8", 443)]
    connected.settimeout.assert_called_once_with(15)


def test_private_fallback_blocks_entire_lookup_before_dial(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", Mock(return_value=dns("1.1.1.1", "169.254.169.254")))
    dial = Mock()
    monkeypatch.setattr(socket, "create_connection", dial)
    with pytest.raises(ValueError, match="blocked"):
        relay.connect_public("mixed.example", 443)
    dial.assert_not_called()


def test_fallbacks_share_one_time_budget(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", Mock(return_value=dns("1.1.1.1", "8.8.8.8", "9.9.9.9", "1.0.0.1")))
    elapsed = [0]
    timeouts = []
    monkeypatch.setattr(relay.time, "monotonic", lambda: elapsed[0])

    def unavailable(address, *, timeout):
        timeouts.append(timeout)
        elapsed[0] += timeout
        raise TimeoutError()

    monkeypatch.setattr(socket, "create_connection", unavailable)
    with pytest.raises(OSError, match="reachable"):
        relay.connect_public("cdn.example", 443)
    assert timeouts == [5, 5, 5]
    assert elapsed[0] == 15


def handler():
    instance = relay.Relay.__new__(relay.Relay)
    instance.path = "cdn.example:443"
    instance.expires_at = float("inf")
    instance.connection = object()
    instance.send_response = Mock()
    instance.end_headers = Mock()
    instance._error = Mock()
    return instance


@pytest.mark.parametrize("error,status", [(ValueError("blocked"), 403), (OSError("unreachable"), 502)])
def test_connect_distinguishes_policy_from_network_failure(monkeypatch, error, status):
    instance = handler()
    monkeypatch.setattr(relay, "connect_public", Mock(side_effect=error))
    instance.do_CONNECT()
    assert instance._error.call_args.args[0] == status
    assert instance.close_connection is True


def test_tunnel_failure_does_not_inject_http_into_established_tls(monkeypatch):
    instance = handler()
    upstream = Mock()
    monkeypatch.setattr(relay, "connect_public", Mock(return_value=upstream))
    monkeypatch.setattr(relay.select, "select", Mock(side_effect=OSError("lost connection")))
    instance.do_CONNECT()
    instance.send_response.assert_called_once_with(200, "Connection Established")
    instance._error.assert_not_called()
    upstream.close.assert_called_once()


def test_http_uses_prevalidated_socket_without_hostname_dial(monkeypatch):
    connected = Mock()
    monkeypatch.setattr(relay, "connect_public", Mock(return_value=connected))
    connection = relay.public_http_connection("cdn.example", 80)
    assert connection.sock is connected
    connected.settimeout.assert_called_once_with(100)
    connection.close()
