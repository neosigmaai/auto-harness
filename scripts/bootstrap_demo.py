"""Create two demo organisations (Acme: admin + member, Beta: member) and print their API keys.
Usage: python -m scripts.bootstrap_demo   (needs DATABASE_URL / .env)"""
from __future__ import annotations

import asyncio
import json

from service import repo
from service.db import get_pool, migrate


async def main():
    await migrate()
    pool = await get_pool()
    out = {}
    async with pool.acquire() as conn:
        acme = await repo.create_org(conn, "acme", "priya@acme.example", 2)
        sam = await repo.create_user(conn, acme["org_id"], "acme", "sam@acme.example", "member")
        beta = await repo.create_org(conn, "beta", "lee@beta.example", 2)
        out = {"acme_admin_priya": acme["api_key"], "acme_member_sam": sam["api_key"], "beta_admin_lee": beta["api_key"]}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
