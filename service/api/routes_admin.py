"""Organisation / membership management. Org creation needs the bootstrap token; member creation needs an org admin."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Header

from ..auth import Principal, get_principal, require_admin
from ..config import settings
from ..db import get_pool
from ..errors import ApiError
from ..models import CredentialOut, MeOut, OrgCreate, UserCreate
from .. import repo

router = APIRouter(prefix="/v1", tags=["admin"])


@router.post("/orgs", response_model=CredentialOut, status_code=201, summary="Create an organisation (bootstrap token)")
async def create_org(body: OrgCreate, x_bootstrap_token: str | None = Header(default=None)):
    if not settings.bootstrap_admin_token or x_bootstrap_token != settings.bootstrap_admin_token:
        raise ApiError(403, "Forbidden", "Valid X-Bootstrap-Token required to create organisations.")
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            cred = await repo.create_org(conn, body.name, body.admin_email, body.max_concurrent_jobs)
        except Exception as e:  # unique violation etc.
            if "unique" in str(e).lower():
                raise ApiError(409, "Conflict", f"Organisation '{body.name}' already exists.")
            raise
    return CredentialOut(**cred)


@router.post("/orgs/{org_id}/users", response_model=CredentialOut, status_code=201, summary="Add a member (org admin only)")
async def create_user(org_id: UUID, body: UserCreate, p: Principal = Depends(require_admin)):
    if org_id != p.org_id:
        raise ApiError(404, "Not found", "Organisation not found.")  # never reveal other orgs
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            cred = await repo.create_user(conn, p.org_id, p.org_name, body.email, body.role)
        except Exception as e:
            if "unique" in str(e).lower():
                raise ApiError(409, "Conflict", f"User '{body.email}' already exists in this organisation.")
            raise
    return CredentialOut(**cred)


@router.get("/me", response_model=MeOut, summary="Who am I")
async def me(p: Principal = Depends(get_principal)):
    return MeOut(org_id=p.org_id, org_name=p.org_name, user_id=p.user_id, email=p.email, role=p.role)
