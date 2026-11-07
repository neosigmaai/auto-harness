"""Generate editable Excalidraw and a matching PNG preview (Pillow optional).

Run with the bundled artifact Python, or install Pillow to regenerate the preview.
Only native rectangles, arrows and text are used: no embedded bitmap diagram.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W, H = 1800, 2210
elements = []
preview = []
N = 0
INK = "#142a3b"
BLUE = "#2563eb"
GREEN = "#17806a"


def base(kind, ident, x, y, w, h, color=INK, fill="transparent"):
    return dict(id=ident, type=kind, x=x, y=y, width=w, height=h, angle=0,
                strokeColor=color, backgroundColor=fill, fillStyle="solid", strokeWidth=2,
                strokeStyle="solid", roughness=0, opacity=100, groupIds=[], frameId=None,
                roundness={"type": 3} if kind == "rectangle" else None,
                seed=int(hashlib.sha256(ident.encode()).hexdigest()[:7], 16), version=1,
                versionNonce=1, isDeleted=False, boundElements=None, updated=1, link=None, locked=False)


def text(ident, x, y, label, size=22, color=INK, width=1600, container=None):
    lines = label.splitlines()
    e = base("text", ident, x, y, width, len(lines) * size * 1.25, color)
    e.update(text=label, originalText=label, fontSize=size, fontFamily=2, textAlign="left",
             verticalAlign="top", containerId=container, autoResize=False, lineHeight=1.25)
    elements.append(e)
    preview.append(("text", x, y, label, size, color))


def rect(ident, x, y, w, h, label=None, fill="#ffffff", stroke="#b6c4d1", size=22):
    e = base("rectangle", ident, x, y, w, h, stroke, fill)
    if label:
        e["boundElements"] = [{"type": "text", "id": ident + "_text"}]
    elements.append(e)
    preview.append(("rect", x, y, w, h, fill, stroke))
    if label:
        text(ident + "_text", x + 22, y + 20, label, size, width=w-44, container=ident)
    return e


def arrow(ident, points, color=BLUE, label=None, labelpos=None):
    x, y = points[0]
    ps = [[a-x, b-y] for a,b in points]
    e = base("arrow", ident, x, y, max(p[0] for p in ps)-min(p[0] for p in ps),
             max(p[1] for p in ps)-min(p[1] for p in ps), color)
    e.update(points=ps, startBinding=None, endBinding=None, startArrowhead=None, endArrowhead="arrow",
             elbowed=False, lastCommittedPoint=None)
    elements.append(e)
    preview.append(("arrow", points, color))
    if label:
        text(ident+"_label", *labelpos, label, 18, color, width=max(470, len(label) * 10))


def bind_arrows():
    """Use the native orbit/fixedPoint schema in the user's saved diagrams.

    Reciprocal bindings keep connectors attached when a rectangle is moved.
    Fixed points use normalized coordinates at the existing drawn endpoint.
    """
    connections = {
        "api_db": ("api", "db"), "db_worker": ("db", "worker"),
        "worker_optimizer": ("worker", "optimizer"), "worker_harbor": ("worker", "harbor"),
        "optimizer_gateway": ("optimizer", "gateway"), "gateway_provider": ("gateway", "provider"),
        "harbor_candidate": ("harbor", "candidate"), "candidate_relay": ("candidate", "relay"),
        "relay_gateway": ("relay", "gateway"), "loop_propose": ("parent", "propose"),
        "loop_validate": ("propose", "validate"), "loop_screen": ("validate", "screen"),
        "loop_evaluate": ("screen", "evaluate"), "loop_confirm": ("evaluate", "confirm"),
        "loop_decide": ("confirm", "decide"), "continue": ("decide", "parent"),
    }
    by_id = {element["id"]: element for element in elements}
    for ident, peers in connections.items():
        connector = by_id[ident]
        for field, peer_id, point in zip(("startBinding", "endBinding"), peers,
                                        (connector["points"][0], connector["points"][-1])):
            peer = by_id[peer_id]
            fixed = [(connector["x"] + point[0] - peer["x"]) / peer["width"],
                     (connector["y"] + point[1] - peer["y"]) / peer["height"]]
            # Existing Excalidraw files nudge exact midpoints off the degenerate
            # center; retain their convention while preserving edge anchors.
            fixed = [0.5001 if abs(value - 0.5) < 1e-9 else max(0.0, min(1.0, value)) for value in fixed]
            connector[field] = {"mode": "orbit", "fixedPoint": fixed, "elementId": peer_id}
            if peer["boundElements"] is None:
                peer["boundElements"] = []
            peer["boundElements"].append({"type": "arrow", "id": ident})


def build():
    elements.clear()
    preview.clear()
    text("title", 50, 35, "Agent Optimization Service", 40)
    text("subtitle", 50, 98, "Python + PostgreSQL + sandboxed agents | quality, runtime and cost are measured separately", 23, "#526a7f")
    rect("trusted_zone", 40, 170, 1720, 560, fill="#f0f6ff", stroke="#94b7ee")
    text("trusted_label", 60, 185, "TRUSTED CONTROL PLANE  /  API and worker are separate processes", 20, BLUE)
    rect("api", 70, 240, 430, 200,
         "FastAPI + test_client.py\nAuthenticate organisation + member\nSubmit -> commit -> 202; poll/cancel\nOwner/admin access on every resource")
    rect("db", 620, 240, 500, 200,
         "PostgreSQL: durable authority\nQueue, leases and admission slots\nVersions, attempts + frozen image IDs\nDecisions, history and spend ledger")
    rect("worker", 1240, 240, 470, 200,
         "Async Python workers\nClaim a job; renew ownership\nResume durable stages after failure\nNo candidate import or execution")
    arrow("api_db", [(500,335),(620,335)])
    arrow("db_worker", [(1120,335),(1240,335)])
    rect("gateway", 70, 520, 430, 170,
         "Budgeted inference gateway\nNEOSIGMA_OPEN_API_KEY only\nReserve -> call -> settle usage\nScoped, revocable trial capability", size=22)
    rect("optimizer", 620, 520, 500, 170,
         "Optimizer (trusted LLM request)\nFeedback evidence + rejection history\nOne quality or efficiency hypothesis\nGenerated Python is stored as data")
    rect("harbor", 1240, 520, 470, 170,
         "Fixed Harbor / Docker adapter\nBounded task slots + deadlines\nFrozen tool image or cold setup\nCheck readiness; fresh verifier\nCancel, label, expire and clean up")
    arrow("worker_optimizer", [(1320,440),(1320,475),(870,475),(870,520)])
    arrow("worker_harbor", [(1530,440),(1530,520)])
    arrow("optimizer_gateway", [(620,605),(500,605)])
    rect("provider", 70, 850, 430, 180,
         "External model provider\napi.openai.com (approved endpoint)\nFixed model within each comparison\nConfigured token prices = estimates",
         fill="#fff8e7", stroke="#d6af5d")
    arrow("gateway_provider", [(170,690),(170,850)], label="Provider authentication only", labelpos=(190,760))
    rect("sandbox_zone", 550, 820, 1210, 390, fill="#edfaf5", stroke="#77bda7")
    text("sandbox_label", 575, 840, "ISOLATED EXECUTION  /  fresh task environment for every attempt", 20, GREEN)
    rect("relay", 590, 910, 440, 190,
         "Fixed per-trial relay\nTask-facing internal network\nModel endpoint -> scoped gateway\nPublic HTTP(S) downloads allowed\nPrivate destinations blocked",
         fill="#ffffff", stroke="#77bda7", size=21)
    rect("candidate", 1110, 910, 600, 190,
         "Untrusted task container\nFull candidate Python + model/tool loop\nTask files and shell commands stay here\nNo host mounts, Docker socket or real keys\nPinned verifier runs after agent completion",
         fill="#ffffff", stroke="#77bda7", size=21)
    arrow("harbor_candidate", [(1470,690),(1470,910)], GREEN, "Provision / supervise / collect", (1210,765))
    arrow("candidate_relay", [(1110,1010),(1030,1010)], GREEN)
    arrow("relay_gateway", [(810,910),(810,770),(420,770),(420,690)], GREEN)
    text("sandbox_note", 590, 1125,
         "Build clean tool images once; pin IDs in each job. Every trial gets a fresh writable layer.\n"
         "No candidate or credentials enter preparation images. Verification still runs for every trial.\n"
         "Shared-state verification has residual tampering risk; Docker is not a microVM.",
         18, "#3e6d5e", width=1120)

    text("loop_title", 50, 1270, "A durable experiment loop, with a fixed evaluation plan", 30)
    labels = [
        ("parent", "Baseline / parent\nImmutable source\nFeedback evidence"),
        ("propose", "Propose\nOne hypothesis\nHistory included"),
        ("validate", "Validate\nSyntax / interface\nReject duplicates"),
        ("screen", "Quality: screen\nEfficiency: bypass\nNo inferred passes"),
        ("evaluate", "Evaluate\nComplete fixed\nselection set"),
        ("confirm", "Confirm\nMatched fresh\nparent + candidate"),
        ("decide", "Decide\nQuality + latency\n+ model cost"),
    ]
    for i, (ident, label) in enumerate(labels):
        x = 50 + 245*i
        rect(ident, x, 1350, 215, 135, label, fill="#f5f7fa", size=20)
        if i:
            arrow("loop_"+ident, [(x-30,1415),(x,1415)])
    arrow("continue", [(1625,1485),(1625,1540),(155,1540),(155,1485)], label="Accept -> new parent; reject -> preserve parent. Stop on persisted time/spend/patience limits.", labelpos=(240,1560))
    rect("final_eval", 50, 1630, 1690, 120,
         "After selection freezes: evaluate baseline and selected version on held-back tasks; record prior exposure.\n"
         "Report all attempts, uncertainty, operational cost and total search spend. Incomplete evidence stays incomplete.",
         fill="#fff8e7", stroke="#d6af5d", size=23)

    text("contracts_title", 50, 1805, "Non-negotiable contracts and explicit trade-offs", 30)
    rect("correctness", 50, 1870, 540, 245,
         "CONSISTENCY + RECOVERY\nShort job-row locks fence mutations\nAtomic admission and budget reserves\nImmutable attempts; no parent-score fill\nCancellation blocks later promotion\nAt-least-once execution can repeat spend\nPostgreSQL is required for new dispatch",
         fill="#f0f6ff", stroke="#94b7ee", size=21)
    rect("measurement", 630, 1870, 540, 245,
         "MEASUREMENT\nFeedback / selection / holdout separation\nSame model, revision and limits per trial\nExecuted tests + reward + token estimates\nAttempt wall time; Docker cost unpriced\nSmall holdout limits generalization\nNo promised speedup or success gain",
         fill="#edfaf5", stroke="#77bda7", size=21)
    rect("scope", 1210, 1870, 530, 245,
         "SCOPE + TRADE-OFFS\nDocker is the implemented provider\nE2B is an optional future adapter\nSingle Postgres is not high availability\nLocal resource caps limit concurrency\nShared verifier isolation is imperfect\nOperational evidence: docs/VALIDATION.md",
         fill="#fff8e7", stroke="#d6af5d", size=21)
    text("footer", 50, 2160, "Editable native Excalidraw elements | Architecture contract and measured status are separate | November 2026", 18, "#526a7f")
    bind_arrows()
    doc = {"type": "excalidraw", "version": 2, "source": "https://excalidraw.com",
           "elements": elements, "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None}, "files": {}}
    (ROOT / "final_architecture.excalidraw").write_text(json.dumps(doc, indent=2))


def render():
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (W,H), "white")
    d = ImageDraw.Draw(im)
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    for p in preview:
        if p[0] == "rect":
            _, x,y,w,h,fill,stroke = p
            d.rounded_rectangle((x,y,x+w,y+h), radius=12, fill=fill, outline=stroke, width=2)
        elif p[0] == "text":
            _, x,y,label,size,color = p
            f = ImageFont.truetype(font_path,size)
            for j,line in enumerate(label.splitlines()):
                d.text((x,y+j*size*1.25),line,font=f,fill=color)
        else:
            _, points,color = p
            d.line(points,fill=color,width=3)
            a,b = points[-2],points[-1]
            angle = math.atan2(b[1]-a[1],b[0]-a[0])
            wings=[(b[0]-13*math.cos(angle+s),b[1]-13*math.sin(angle+s)) for s in [-.45,.45]]
            d.line([wings[0],b,wings[1]],fill=color,width=3)
    im.save(ROOT / "docs" / "architecture.png")


if __name__ == "__main__":
    build()
    try:
        render()
    except ImportError:
        print("Excalidraw saved; install Pillow for PNG preview")
    print(f"Saved {len(elements)} editable elements")
