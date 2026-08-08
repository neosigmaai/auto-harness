"""Create an org, its first admin, and optionally more members. No signup endpoint.

    python seed.py acme                          # org + admin@acme.test
    python seed.py acme alice bob:admin          # ... plus a member and a second admin

Extra arguments are `name` or `name:role`, role defaulting to member. Every key is printed
once here and never again. Only its SHA-256 hash is stored.
"""

from __future__ import annotations

import sys

import config
import db

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "acme"
    org_id, admin_key = db.create_org(name, f"admin@{name}.test")
    print(f"org {name}  {org_id}")
    print(f"  admin   admin@{name}.test   {admin_key}")

    for arg in sys.argv[2:]:
        who, _, role = arg.partition(":")
        role = role or "member"
        email = who if "@" in who else f"{who}@{name}.test"
        print(f"  {role:<7} {email:<20} {db.create_member(org_id, email, role)}")

    print(f"\nShown once, stored only as hashes. Save them now."
          f"\nWritten to {config.db_label()} — the API must be on the same one "
          f"(check GET /health).")
