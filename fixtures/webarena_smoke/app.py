"""
Minimal Postmill-shaped fixture for WebArena smoke testing.

Serves a handful of canned Reddit/Postmill-style pages on localhost:9999.
The HTML is deliberately simple but uses proper semantic elements (h1, nav,
article, a[href]) so WebArena's accessibility-tree parser produces clean
observations for the agent.

Hardcoded data — no DB, no auth. Matches the answers referenced in
webarena_data/smoke_tasks/*.json. If you change any of the strings here,
update those task files too.

Run with:
    python fixtures/webarena_smoke/app.py
    # → serving on http://localhost:9999
"""

from __future__ import annotations

from flask import Flask, abort, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Fixture data. Keep in sync with webarena_data/smoke_tasks/*.json answers.
# ---------------------------------------------------------------------------

POSTS: list[dict] = [
    {
        "id": "1",
        "slug": "hello-world",
        "title": "Hello World",
        "author": "alice",
        "forum": "general",
        "body": "First post on this forum. Welcome everyone.",
        "score": 42,
        "comments": [
            {"author": "bob", "body": "Nice to meet you."},
            {"author": "carol", "body": "Welcome!"},
            {"author": "dave", "body": "Looking forward to more posts."},
        ],
    },
    {
        "id": "2",
        "slug": "distributed-systems",
        "title": "Distributed Systems",
        "author": "bob",
        "forum": "programming",
        "body": "A brief introduction to the CAP theorem and why partitions matter.",
        "score": 31,
        "comments": [
            {"author": "alice", "body": "Great overview."},
            {"author": "eve", "body": "What about PACELC?"},
        ],
    },
    {
        "id": "3",
        "slug": "consensus-algorithms",
        "title": "Consensus Algorithms",
        "author": "carol",
        "forum": "programming",
        "body": "Raft vs Paxos — which one should you reach for?",
        "score": 28,
        "comments": [
            {"author": "bob", "body": "Raft is easier to teach."},
        ],
    },
    {
        "id": "4",
        "slug": "best-sci-fi-of-the-decade",
        "title": "Best Sci-Fi of the Decade",
        "author": "dave",
        "forum": "books",
        "body": "Looking for recommendations. Hard sci-fi preferred.",
        "score": 17,
        "comments": [
            {"author": "carol", "body": "Try Children of Time."},
            {"author": "frank", "body": "Project Hail Mary is excellent."},
        ],
    },
    {
        "id": "5",
        "slug": "a-quiet-week",
        "title": "A Quiet Week",
        "author": "eve",
        "forum": "general",
        "body": "Not much going on. Hope you are all doing well.",
        "score": 9,
        "comments": [],
    },
    # Posts 6-8 are deliberately placed deep on the front page so their
    # titles/link IDs fall below WebArena's default observation-truncation
    # cutoff (max_obs_length ≈ 1920 tokens). Tasks 10007 and 10008 target
    # these posts, producing a reliable "link not in the accessibility tree"
    # failure for the baseline agent. Move them up only if you also retune
    # the failing-task set.
    {
        "id": "6",
        "slug": "python-type-hints",
        "title": "Python Type Hints in 2026",
        "author": "frank",
        "forum": "programming",
        "body": "Type hints are now essential for large codebases.",
        "score": 22,
        "comments": [
            {"author": "alice", "body": "mypy strict mode is worth the pain."},
        ],
    },
    {
        "id": "7",
        "slug": "ocean-currents-slowing",
        "title": "Ocean Currents Slowing",
        "author": "grace",
        "forum": "general",
        "body": "The Gulf Stream is weakening faster than expected.",
        "score": 14,
        "comments": [
            {"author": "henry", "body": "The AMOC collapse risk is underrated."},
        ],
    },
    {
        "id": "8",
        "slug": "tolkien-legacy",
        "title": "Tolkien's Legacy",
        "author": "henry",
        "forum": "books",
        "body": "Tolkien invented modern high fantasy as we know it.",
        "score": 11,
        "comments": [
            {"author": "grace", "body": "The Silmarillion is the secret masterpiece."},
            {"author": "frank", "body": "Agreed, underrated work."},
        ],
    },
]

FORUMS = ["general", "programming", "books"]


def _find_post(post_id: str) -> dict | None:
    for p in POSTS:
        if p["id"] == post_id:
            return p
    return None


def _layout(title: str, body_html: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title} - Postmill Smoke</title>
</head>
<body>
  <header>
    <h1><a href="/">Postmill Smoke</a></h1>
    <nav aria-label="Forums">
      <ul>
        {''.join(f'<li><a href="/f/{f}">f/{f}</a></li>' for f in FORUMS)}
      </ul>
    </nav>
    <form action="/search" method="get" role="search">
      <label for="q">Search</label>
      <input id="q" name="q" type="text">
      <button type="submit">Search</button>
    </form>
  </header>
  <main>
    {body_html}
  </main>
</body>
</html>"""


def _post_summary_html(p: dict) -> str:
    return f"""
    <article>
      <h2><a href="/f/{p['forum']}/{p['id']}/{p['slug']}">{p['title']}</a></h2>
      <p>
        submitted by <a href="/user/{p['author']}">{p['author']}</a>
        to <a href="/f/{p['forum']}">f/{p['forum']}</a>
      </p>
      <p>Score: {p['score']} points | Comments: {len(p['comments'])}</p>
    </article>
    """


@app.route("/")
def index() -> str:
    body = "<h2>Front page</h2>\n" + "\n".join(
        _post_summary_html(p) for p in POSTS
    )
    return _layout("Front page", body)


@app.route("/f/<forum>")
def forum(forum: str) -> str:
    if forum not in FORUMS:
        abort(404)
    forum_posts = [p for p in POSTS if p["forum"] == forum]
    body = f"<h2>f/{forum}</h2>\n" + (
        "\n".join(_post_summary_html(p) for p in forum_posts)
        or "<p>No posts yet.</p>"
    )
    return _layout(f"f/{forum}", body)


@app.route("/f/<forum>/<post_id>/<slug>")
def post_detail(forum: str, post_id: str, slug: str) -> str:
    p = _find_post(post_id)
    if not p or p["forum"] != forum:
        abort(404)
    comments_html = "".join(
        f"<li><strong>{c['author']}</strong>: {c['body']}</li>"
        for c in p["comments"]
    ) or "<li>No comments yet.</li>"
    body = f"""
    <article>
      <h2>{p['title']}</h2>
      <p>
        submitted by <a href="/user/{p['author']}">{p['author']}</a>
        to <a href="/f/{p['forum']}">f/{p['forum']}</a>
      </p>
      <p>{p['body']}</p>
      <p>Score: {p['score']} points</p>
    </article>
    <section aria-label="Comments">
      <h3>Comments ({len(p['comments'])})</h3>
      <ul>{comments_html}</ul>
    </section>
    """
    return _layout(p["title"], body)


@app.route("/user/<username>")
def user_profile(username: str) -> str:
    user_posts = [p for p in POSTS if p["author"] == username]
    if not user_posts:
        body = f"<h2>User: {username}</h2><p>This user has not posted anything.</p>"
    else:
        body = f"<h2>User: {username}</h2>\n" + "\n".join(
            _post_summary_html(p) for p in user_posts
        )
    return _layout(f"user/{username}", body)


@app.route("/search")
def search() -> str:
    q = (request.args.get("q") or "").strip().lower()
    if not q:
        return _layout("Search", "<h2>Search</h2><p>Enter a query above.</p>")
    hits = [
        p for p in POSTS
        if q in p["title"].lower() or q in p["body"].lower()
    ]
    body = f"<h2>Search results for '{q}'</h2>\n" + (
        "\n".join(_post_summary_html(p) for p in hits)
        or "<p>No results.</p>"
    )
    return _layout(f"search: {q}", body)


if __name__ == "__main__":
    # Host 0.0.0.0 so the harness can reach it from any docker/network context.
    # Port 9999 matches WebArena's default REDDIT URL convention.
    app.run(host="0.0.0.0", port=9999, debug=False, use_reloader=False)
