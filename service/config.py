
from __future__ import annotations

import logging
import os
import re

logging.basicConfig(
    format="%(levelname)-5s | %(asctime)s | %(name)-9s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.getenv("LOG_LEVEL", "INFO"),
)
for _noisy in ("pgserver", "httpx", "httpcore", "openai", "e2b"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)   

PLATFORM_MAX_JOBS = int(os.getenv("PLATFORM_MAX_JOBS", "2"))    
MAX_ITERATIONS_CAP = int(os.getenv("MAX_ITERATIONS_CAP", "5")) 
N_CONCURRENT = 8             
PER_TASK_TIMEOUT = int(os.getenv("PER_TASK_TIMEOUT", "900"))    

# Lease and sandbox timeout are deadline + this, never this alone. 
LEASE_SLACK_SECONDS = 600

CLAIM_LOCK = 0x414F5F4A4F4253 # Held for the claim txn. "AO_JOBS" in ASCII for pg_locks.

PLATFORM_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
E2B_API_KEY = os.getenv("E2B_API_KEY", "")
OPTIMIZER_MODEL = os.getenv("OPTIMIZER_MODEL", "gpt-5.4")   
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-5.4")           


TASK_SUBSET = [
    "fix-git", "prove-plus-comm", "cobol-modernization", "overfull-hbox",
    "crack-7z-hash", "raman-fitting", "kv-store-grpc", "pytorch-model-recovery",
    "nginx-request-logging", "polyglot-c-py", "openssl-selfsigned-cert",
    "hf-model-inference", "multi-source-data-merger", "extract-elf",
    "git-leak-recovery", "sanitize-git-repo", "chess-best-move", "regex-log",
    "db-wal-recovery", "largest-eigenval", "configure-git-webserver",
]
HOLDOUT_FRACTION = 0.3        

_resolved_db: str | None = None


def db_label() -> str:
    url = database_url()
    if "host=" in url:
        return url.split("host=", 1)[1]         # local pgserver: socket dir is the identity
    return re.sub(r"//[^@/]*@", "//", url)      # strip credentials


def database_url() -> str:
    """DATABASE_URL, else boot a local Postgres so dev needs no setup."""
    global _resolved_db
    if _resolved_db is None:
        url = os.getenv("DATABASE_URL")
        if not url:
            import pgserver

            url = pgserver.get_server(os.getenv("PGDATA_DIR", "/tmp/aos-pgdata")).get_uri()
        _resolved_db = url
        logging.getLogger("db").info("using %s", db_label())
    return _resolved_db
