"""Auth dependency: resolve the ``X-API-Key`` header to a principal.

M1–M4: any valid key is accepted; the seeded dev key works out of the box. M5
layers role/ownership enforcement on top of the same principal.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from harness_service.config import get_settings
from harness_service.constants import Role
from harness_service.db import get_session
from harness_service.db.models import User


@dataclass(frozen=True)
class Principal:
    user_id: UUID
    org_id: UUID
    role: Role


async def get_principal(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    session: AsyncSession = Depends(get_session),
) -> Principal:
    # Dev convenience: fall back to the seeded key when the header is absent.
    key = x_api_key or get_settings().dev_api_key
    user = (await session.execute(select(User).where(User.api_key == key))).scalars().first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing API key"
        )
    return Principal(user_id=user.id, org_id=user.org_id, role=Role(user.role))
