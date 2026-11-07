"""API-key authentication and the Principal every repository call is scoped by."""
from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from uuid import UUID

from fastapi import Depends, Header

from .db import get_pool
from .errors import ApiError

KEY_PREFIX = "aos_"


def generate_api_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class Principal:
    org_id: UUID
    org_name: str
    user_id: UUID
    email: str
    role: str  # admin | member

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


async def get_principal(authorization: str | None = Header(default=None)) -> Principal:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise ApiError(401, "Unauthorized", "Provide 'Authorization: Bearer <api key>'.")
    raw = authorization.split(" ", 1)[1].strip()
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        SELECT k.org_id, o.name AS org_name, u.id AS user_id, u.email, u.role
        FROM api_keys k JOIN users u ON u.id = k.user_id JOIN organizations o ON o.id = k.org_id
        WHERE k.key_hash = $1 AND k.revoked_at IS NULL
        """,
        hash_api_key(raw),
    )
    if row is None:
        raise ApiError(401, "Unauthorized", "Unknown or revoked API key.")
    return Principal(row["org_id"], row["org_name"], row["user_id"], row["email"], row["role"])


async def require_admin(p: Principal = Depends(get_principal)) -> Principal:
    if not p.is_admin:
        raise ApiError(403, "Forbidden", "This action requires the admin role.")
    return p
