"""Seed a default org + admin user for local/dev use (M1–M4).

M5 replaces the single dev principal with real org/user provisioning + role
enforcement. Until then, one seeded admin key makes the API easy to exercise.
"""

from __future__ import annotations

import logging

from sqlalchemy import select

from harness_service.config import Settings
from harness_service.constants import Role
from harness_service.db import session_scope
from harness_service.db.models import Organization, User

logger = logging.getLogger("harness.seed")

DEV_ORG_NAME = "dev-org"
DEV_USER_EMAIL = "dev@local"


async def ensure_dev_principal(settings: Settings) -> None:
    async with session_scope() as s:
        existing = (
            await s.execute(select(User).where(User.api_key == settings.dev_api_key))
        ).scalars().first()
        if existing is not None:
            return
        org = (
            await s.execute(select(Organization).where(Organization.name == DEV_ORG_NAME))
        ).scalars().first()
        if org is None:
            org = Organization(name=DEV_ORG_NAME)
            s.add(org)
            await s.flush()
        s.add(
            User(
                org_id=org.id,
                email=DEV_USER_EMAIL,
                role=Role.ADMIN,
                api_key=settings.dev_api_key,
            )
        )
        logger.info("seeded dev principal (org=%s, key=%s)", DEV_ORG_NAME, settings.dev_api_key)
