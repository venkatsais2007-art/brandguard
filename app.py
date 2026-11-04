"""
BrandGuard - Round 2 MVP (Team STRAWHATS)
Backend: Flask + SQLite + Gemini.  Run:  python app.py
"""
import os
import re
import json
import sqlite3
import difflib
from datetime import datetime, timezone

from flask import (Flask, request, redirect, url_for, session, flash, g,
                   render_template_string, get_flashed_messages, abort)
from markupsafe import Markup, escape
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "brandguard-demo-secret")
DB_PATH = os.environ.get("DB_PATH", "brandguard.db")
API_KEY = os.environ.get("GEMINI_API_KEY", "")

# The app tries these models one by one until one works.
# You can put your own model name in .env as GEMINI_MODEL=...
MODELS = [m for m in [os.environ.get("GEMINI_MODEL"),
                      "gemini-3.1-flash-lite",
                      "gemini-3.5-flash-lite",
                      "gemini-3-flash-preview"] if m]

ROLES = {"creator": "Creator (writer)", "reviewer": "Reviewer", "manager": "Brand Manager"}

SAMPLE_RULES = """FreshKart Brand Guidelines (sample)

Tone: Our voice is friendly, simple and honest. Write short sentences. Do not shout or use hype.
Claims: Never use words like "best", "#1", "fastest" or "guaranteed" unless we have approved proof for the claim.
Approved delivery claim: The only approved delivery claim is "Delivery within 24 hours in select cities".
Approved fact: FreshKart operates in 12 cities as of 2026.
Terminology: Always call the people who buy from us "customers". Never call them "users".
Offers: Do not promise free delivery without naming the minimum order value.
Calls to action: Use clear action words like "Order now". Avoid "Click here".
"""
SAMPLE_WORDS = "users => customers\nclick here => order now"
SAMPLE_DRAFT = ("FreshKart is India's #1 fastest delivery service in every city. "
                "We guarantee our users will love it! Get free delivery always. Click here to buy now.")

STATUS_LABEL = {"ai_reviewed": "AI checked", "in_review": "In review",
                "changes_requested": "Changes requested", "approved": "Approved",
                "published": "Published", "draft": "Draft"}

# (from_status, to_status): roles allowed to do it
TRANSITIONS = {
    ("ai_reviewed", "in_review"): ("creator", "manager"),
    ("changes_requested", "in_review"): ("creator", "manager"),
    ("in_review", "approved"): ("reviewer", "manager"),
    ("in_review", "changes_requested"): ("reviewer", "manager"),
    ("approved", "published"): ("manager",),
}
BUTTON_LABEL = {"in_review": "Send for review", "approved": "Approve",
                "changes_requested": "Request changes", "published": "Mark as published"}
WHO_NEXT = {"ai_reviewed": "Creator: review the suggestions, then send for review.",
            "changes_requested": "Creator: fix the issues and send for review again.",
            "in_review": "Reviewer or Brand Manager: approve or request changes.",
            "approved": "Brand Manager: mark as published.",
            "published": "Done. This content is published."}


# ----------------------------------------------------------------- database
def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.executescript("""
    CREATE TABLE IF NOT EXISTS rules (id INTEGER PRIMARY KEY, text TEXT, words TEXT, updated TEXT);
    CREATE TABLE IF NOT EXISTS content (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, text TEXT,
        status TEXT, version INTEGER DEFAULT 1, ai_note TEXT, created TEXT);
    CREATE TABLE IF NOT EXISTS versions (id INTEGER PRIMARY KEY AUTOINCREMENT, content_id INTEGER,
        n INTEGER, text TEXT, note TEXT, created TEXT);
    CREATE TABLE IF NOT EXISTS findings (id INTEGER PRIMARY KEY AUTOINCREMENT, content_id INTEGER,
        category TEXT, severity TEXT, original TEXT, why TEXT, rule_quote TEXT, suggestion TEXT,
        confidence REAL, source TEXT, decision TEXT DEFAULT 'pending', final_text TEXT, applied INTEGER DEFAULT 0);
    CREATE TABLE IF NOT EXISTS comments (id INTEGER PRIMARY KEY AUTOINCREMENT, content_id INTEGER,
        role TEXT, text TEXT, created TEXT);
    CREATE TABLE IF NOT EXISTS activity (id INTEGER PRIMARY KEY AUTOINCREMENT, content_id INTEGER,
        role TEXT, action TEXT, created TEXT);
    """)
    db.commit()
    db.close()


init_db()


def log(content_id, action):
    get_db().execute("INSERT INTO activity (content_id, role, action, created) VALUES (?,?,?,?)",
                     (content_id, current_role(), action, now()))


def current_role():
    r = session.get("role", "creator")
    return r if r in ROLES else "creator"


def get_rules():
    row = get_db().execute("SELECT * FROM rules WHERE id = 1").fetchone()
    return row


def parse_word_pairs(words_text):
    pairs = []
    for line in (words_text or "").splitlines():
        if "=>" in line:
            old, new = [p.strip() for p in line.split("=>", 1)]
            if old and new:
                pairs.append((old, new))
    return pairs


# ------------------------------------------------------------ review engine
def norm(s):
    s = (s or "").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def find_exact(needle, hay):
    """Find the needle in hay ignoring extra spaces/case. Returns the real text from hay."""
    tokens = (needle or "").split()
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(t) for t in tokens)
    m = re.search(pattern, hay, re.I)
    return m.group(0) if m else None


def find_rule_sentence(quote, rules_text):
    """Check that the rule the AI quoted really exists in the saved rules."""
    q = norm(quote)
    if len(q) < 8:
        return None
    sentences = [s.strip() for s in re.split(r"[\n\r]+|(?<=[.!?])\s+", rules_text) if s.strip()]
    for s in sentences:
        if q in norm(s) or norm(s) in q:
            return s
    close = difflib.get_close_matches(q, [norm(s) for s in sentences], n=1, cutoff=0.85)
    if close:
        for s in sentences:
            if norm(s) == close[0]:
                return s
    return None


def word_list_checks(draft, pairs):
    out = []
    for old, new in pairs:
        matches = list(re.finditer(r"\b" + re.escape(old) + r"\b", draft, re.I))
        if not matches:
            continue
        orig = matches[0].group(0)
        sug = new[:1].upper() + new[1:] if orig[:1].isupper() else new
        extra = f" (appears {len(matches)} times)" if len(matches) > 1 else ""
        out.append(dict(category="terminology", severity="medium", original=orig,
                        why=f'The brand word list says to use "{new}" instead of "{old}"{extra}.',
                        rule_quote=f"Word list: {old} => {new}", suggestion=sug,
                        confidence=1.0, source="word list"))
    return out


def ask_gemini(draft, rules_text):
    """Ask Gemini to find problems, using ONLY the saved brand rules."""
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")
    from google import genai
    from google.genai import types

    prompt = f"""You are a brand compliance reviewer. Use ONLY the BRAND RULES below. Do not use outside knowledge.
Find problems in the DRAFT (tone, terminology, facts, unsupported promotional claims, accessibility).

For every problem return an object with these keys:
- category: one of "tone", "terminology", "fact", "claim", "accessibility", "other"
- severity: "low", "medium" or "high"
- original: the exact words from the DRAFT that have the problem (copy them exactly)
- why: one short sentence in very simple English
- rule_quote: the exact sentence copied from the BRAND RULES that this breaks
- suggestion: replacement words for "original". Change as few words as possible, keep the writer's meaning and style.
- confidence: a number from 0 to 1

If a problem has no matching rule in the BRAND RULES, do not report it.
If the draft is fine, return an empty list.
Return ONLY JSON like: {{"findings": [ ... ]}}

BRAND RULES:
\"\"\"
{rules_text[:20000]}
\"\"\"

DRAFT:
\"\"\"
{draft[:6000]}
\"\"\"
"""
    client = genai.Client(api_key=API_KEY)
    last_error = None
    for model in MODELS:
        try:
            resp = client.models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2))
            text = (resp.text or "").strip()
            text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
            data = json.loads(text)
            items = data.get("findings", []) if isinstance(data, dict) else data
            return items if isinstance(items, list) else [], model
        except Exception as e:  # try the next model
            last_error = e
    raise RuntimeError(f"Gemini did not answer ({last_error}). Check your key and model name.")


def run_review(draft, rules_text, words_text):
    """Returns (findings, note). Every AI finding must point to a real rule and real text."""
    findings = word_list_checks(draft, parse_word_pairs(words_text))
    note = ""
    dropped = 0
    try:
        items, model = ask_gemini(draft, rules_text)
        for it in items:
            if not isinstance(it, dict):
                continue
            real_orig = find_exact(str(it.get("original", "")), draft)
            real_rule = find_rule_sentence(str(it.get("rule_quote", "")), rules_text)
            if not real_orig or not real_rule:
                dropped += 1          # no real text or no real rule -> thrown away
                continue
            if any(norm(real_orig) in norm(f["original"]) or norm(f["original"]) in norm(real_orig)
                   for f in findings):
                continue              # already found by the word list
            try:
                conf = max(0.0, min(1.0, float(it.get("confidence", 0.7))))
            except (TypeError, ValueError):
                conf = 0.5
            sev = it.get("severity") if it.get("severity") in ("low", "medium", "high") else "medium"
            cat = it.get("category") if it.get("category") in (
                "tone", "terminology", "fact", "claim", "accessibility", "other") else "other"
            findings.append(dict(category=cat, severity=sev, original=real_orig,
                                 why=str(it.get("why", ""))[:300], rule_quote=real_rule,
                                 suggestion=str(it.get("suggestion", ""))[:500],
                                 confidence=conf, source="AI (Gemini)"))
        note = f"AI model used: {model}."
        if dropped:
            note += f" {dropped} AI result(s) were removed because they did not point to a real rule or real text."
    except Exception as e:
        note = f"AI check was not available ({e}). Only the word-list check was run."
    findings.sort(key=lambda f: draft.lower().find(f["original"].lower()))
    return findings, note


def apply_text(text, f):
    """Apply one accepted fix to the text."""
    final = f["final_text"] if f["final_text"] is not None else f["suggestion"]
    if f["source"] == "word list":
        return re.sub(r"\b" + re.escape(f["original"]) + r"\b", lambda m: final, text, flags=re.I)
    return text.replace(f["original"], final, 1)


def highlight(text, findings):
    spans = []
    for f in findings:
        if f["applied"]:
            continue
        i = text.find(f["original"])
        if i >= 0:
            spans.append((i, i + len(f["original"]), f["id"], f["decision"]))
    spans.sort()
    out, pos = [], 0
    for s, e, fid, dec in spans:
        if s < pos:
            continue
        out.append(escape(text[pos:s]))
        out.append(Markup(f'<mark class="m-{dec}" title="Issue #{fid}">') + escape(text[s:e]) + Markup("</mark>"))
        pos = e
    out.append(escape(text[pos:]))
    return Markup("").join(out)


# ------------------------------------------------------------------ layout
BASE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BrandGuard - Brand Governance Platform</title>
<style>
:root{--bg:#070b17;--panel:#0e1530;--panel2:#131c3d;--line:#24305c;--text:#e2e8f0;--muted:#94a3b8;
--cyan:#22d3ee;--violet:#a78bfa;--green:#34d399;--amber:#fbbf24;--red:#f87171}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,Segoe UI,Arial,sans-serif;
background-image:linear-gradient(#0d1633 1px,transparent 1px),linear-gradient(90deg,#0d1633 1px,transparent 1px);background-size:28px 28px}
.wrap{max-width:980px;margin:0 auto;padding:0 16px}
header{background:#050813ee;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}
.nav{display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:10px 16px}
.logo{font-weight:800;color:var(--cyan);text-decoration:none;font-size:18px;letter-spacing:.5px}
nav a{color:var(--text);text-decoration:none;margin-right:12px;padding:4px 2px;border-bottom:2px solid transparent}
nav a:hover{border-color:var(--cyan)}
.role{margin-left:auto;display:flex;gap:6px;align-items:center;font-size:13px;color:var(--muted)}
select,input[type=text],textarea{background:#080e22;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:8px;font:inherit;width:100%}
.role select{width:auto}
textarea{min-height:140px;resize:vertical}
h1{font-size:26px;margin:22px 0 6px} h2{font-size:19px;margin:0 0 10px;color:var(--cyan)} h3{margin:0 0 6px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin:14px 0}
.muted{color:var(--muted)} .small{font-size:13px}
.btn{display:inline-block;background:var(--cyan);color:#041018;border:0;border-radius:8px;padding:8px 14px;font-weight:700;cursor:pointer;text-decoration:none;font-size:14px}
.btn.alt{background:transparent;color:var(--text);border:1px solid var(--line)}
.btn.green{background:var(--green)} .btn.red{background:var(--red)} .btn.amber{background:var(--amber)}
.btn[disabled]{opacity:.5;cursor:not-allowed}
.flash{background:#0b2a36;border:1px solid var(--cyan);padding:10px 14px;border-radius:8px;margin:12px 0}
.flash.err{background:#33121a;border-color:var(--red)}
.badge{display:inline-block;font-size:12px;padding:2px 8px;border-radius:99px;border:1px solid var(--line);margin-right:6px;color:var(--muted)}
.b-high{border-color:var(--red);color:var(--red)} .b-medium{border-color:var(--amber);color:var(--amber)} .b-low{border-color:var(--green);color:var(--green)}
.b-status{border-color:var(--violet);color:var(--violet)} .b-check{border-color:var(--amber);color:var(--amber)}
.textbox{background:#080e22;border:1px solid var(--line);border-radius:8px;padding:12px;white-space:pre-wrap}
mark{padding:1px 3px;border-radius:4px;color:#fff}
.m-pending{background:#7a5a0a} .m-accepted{background:#0f6b4d} .m-ignored{background:#3b4260;text-decoration:line-through}
.rule{border-left:3px solid var(--violet);padding:4px 10px;color:#cbd5e1;margin:6px 0;font-style:italic}
.orig{color:var(--red);text-decoration:line-through}
table{width:100%;border-collapse:collapse} td,th{padding:8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--cyan);font-size:13px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px} @media(max-width:720px){.grid{grid-template-columns:1fr}}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
footer{margin:30px 0;padding:14px 0;border-top:1px solid var(--line);color:var(--muted);font-size:12.5px}
details{margin:6px 0} summary{cursor:pointer;color:var(--cyan)}
ol.steps li{margin:6px 0}
</style></head><body>
<header><div class="wrap nav">
<a class="logo" href="/">&#9632; BrandGuard</a>
<nav><a href="/">Home</a><a href="/rules">Brand Rules</a><a href="/check">New Check</a><a href="/content">All Content</a><a href="/report">Report</a></nav>
<form method="post" action="/role" class="role"><label for="r">You are:</label>
<select id="r" name="role" onchange="this.form.submit()">
{% for k,v in roles.items() %}<option value="{{k}}" {% if k==role %}selected{% endif %}>{{v}}</option>{% endfor %}
</select></form></div></header>
<main class="wrap">
{% for cat,msg in flashes %}<div class="flash {{ 'err' if cat=='error' else '' }}">{{ msg }}</div>{% endfor %}
{{ body }}
</main>
<div class="wrap"><footer>
<b>AI disclosure:</b> findings and suggestions are generated with Google Gemini. The text you submit and the saved brand rules are sent to Gemini to do the check.
Suggestions are never applied automatically: a person always makes the final decision. This is a hackathon demo (Team STRAWHATS).
</footer></div>
</body></html>"""


def render(body_tpl, **ctx):
    inner = render_template_string(body_tpl, role=current_role(), roles=ROLES, **ctx)
    return render_template_string(BASE, body=Markup(inner), role=current_role(), roles=ROLES,
                                  flashes=get_flashed_messages(with_categories=True))


# ------------------------------------------------------------------ routes
HOME = r"""
<h1>Check content before it goes public</h1>
<p class="muted">BrandGuard compares a draft with your company's approved brand rules, explains each problem, suggests a fix,
and lets people review and approve. Nothing is changed unless a person accepts it.</p>
<div class="card">
<h2>Try it in 2 minutes</h2>
<ol class="steps">
<li>Click <b>Load sample data</b>. It saves sample brand rules and opens the check page with a bad draft.</li>
<li>Click <b>Check this content</b>. Wait about 10 to 20 seconds for the AI.</li>
<li>For each problem: read the reason and the rule it breaks, then <b>Accept</b> (you can edit the fix first) or <b>Ignore</b>.</li>
<li>Click <b>Create new version</b>, then <b>Send for review</b>.</li>
<li>Use the <b>You are</b> menu (top right) to switch to <b>Reviewer</b>, then <b>Approve</b> or <b>Request changes</b>.</li>
<li>Open <b>Report</b> to see which mistakes repeat.</li>
</ol>
<form method="post" action="/sample"><button class="btn">Load sample data</button></form>
</div>
<div class="grid">
<div class="card"><h3>Who can do what</h3>
<p class="small muted"><b>Creator:</b> submit content, accept or ignore fixes, send for review.<br>
<b>Reviewer:</b> approve or request changes.<br>
<b>Brand Manager:</b> edit the brand rules, plus everything above.</p></div>
<div class="card"><h3>How the check works</h3>
<p class="small muted">1) Word list check (exact). 2) Gemini reads the draft using only your saved rules.
3) Our code removes any AI result that does not point to a real rule and real text.</p></div>
</div>
"""


@app.route("/")
def home():
    return render(HOME)


@app.route("/role", methods=["POST"])
def set_role():
    r = request.form.get("role", "creator")
    session["role"] = r if r in ROLES else "creator"
    return redirect(request.referrer or url_for("home"))


@app.route("/sample", methods=["POST"])
def sample():
    db = get_db()
    db.execute("INSERT OR REPLACE INTO rules (id, text, words, updated) VALUES (1,?,?,?)",
               (SAMPLE_RULES, SAMPLE_WORDS, now()))
    db.commit()
    flash("Sample brand rules saved. Now check the sample draft.")
    return redirect(url_for("check", sample=1))


RULES = r"""
<h1>Brand Rules</h1>
<p class="muted">The approved source material. The AI may only use what is written here.</p>
{% if not can_edit %}<div class="flash">You are a {{ roles[role] }}. Switch to <b>Brand Manager</b> (top right) to edit the rules.</div>{% endif %}
<form method="post" class="card">
<h2>Guidelines, approved facts and claims</h2>
<textarea name="text" style="min-height:260px" {% if not can_edit %}readonly{% endif %}
 placeholder="Paste your brand guidelines here. One rule per line works best.">{{ text }}</textarea>
<h2 style="margin-top:14px">Word list (exact replacements)</h2>
<p class="small muted">One per line, written as: wrong word =&gt; preferred word</p>
<textarea name="words" style="min-height:90px" {% if not can_edit %}readonly{% endif %}>{{ words }}</textarea>
<p class="small muted">{% if updated %}Last saved: {{ updated }} (UTC){% else %}Nothing saved yet.{% endif %}</p>
<div class="row"><button class="btn" {% if not can_edit %}disabled{% endif %}>Save rules</button>
<button class="btn alt" formaction="/sample" formmethod="post">Load sample rules</button></div>
</form>
"""


@app.route("/rules", methods=["GET", "POST"])
def rules():
    can_edit = current_role() == "manager"
    db = get_db()
    if request.method == "POST":
        if not can_edit:
            flash("Only a Brand Manager can edit the rules.", "error")
            return redirect(url_for("rules"))
        text = request.form.get("text", "").strip()
        words = request.form.get("words", "").strip()
        if not text:
            flash("Please enter some brand rules.", "error")
            return redirect(url_for("rules"))
        db.execute("INSERT OR REPLACE INTO rules (id, text, words, updated) VALUES (1,?,?,?)", (text, words, now()))
        db.commit()
        flash("Brand rules saved.")
        return redirect(url_for("rules"))
    row = get_rules()
    return render(RULES, can_edit=can_edit, text=row["text"] if row else "",
                  words=row["words"] if row else "", updated=row["updated"] if row else "")


CHECK = r"""
<h1>New Check</h1>
<p class="muted">Paste a draft. It will be checked against the saved brand rules before it is published.</p>
{% if not has_rules %}<div class="flash err">There are no brand rules yet. Use <b>Load sample data</b> on the Home page, or ask a Brand Manager to add rules.</div>{% endif %}
<form method="post" class="card" onsubmit="var b=this.querySelector('button.btn');b.disabled=true;b.textContent='Checking... please wait (10-20 seconds)';">
<label class="small muted">Title</label>
<input type="text" name="title" value="{{ 'Summer offer post' if sample else '' }}" placeholder="e.g. Diwali offer post" maxlength="120">
<label class="small muted" style="display:block;margin-top:10px">Draft text</label>
<textarea name="text" required placeholder="Paste your draft here...">{{ draft if sample else '' }}</textarea>
<div style="margin-top:10px"><button class="btn">Check this content</button></div>
</form>
"""


@app.route("/check", methods=["GET", "POST"])
def check():
    row = get_rules()
    if request.method == "POST":
        if current_role() == "reviewer":
            flash("Reviewers review content. Switch to Creator (top right) to submit a draft.", "error")
            return redirect(url_for("check"))
        if not row:
            flash("Please add brand rules first.", "error")
            return redirect(url_for("check"))
        text = request.form.get("text", "").strip()
        title = request.form.get("title", "").strip() or "Untitled draft"
        if not text:
            flash("Please paste some text to check.", "error")
            return redirect(url_for("check"))
        findings, note = run_review(text, row["text"], row["words"])
        db = get_db()
        cur = db.execute("INSERT INTO content (title, text, status, version, ai_note, created) VALUES (?,?,?,?,?,?)",
                         (title, text, "ai_reviewed", 1, note, now()))
        cid = cur.lastrowid
        db.execute("INSERT INTO versions (content_id, n, text, note, created) VALUES (?,?,?,?,?)",
                   (cid, 1, text, "Original draft", now()))
        for f in findings:
            db.execute("""INSERT INTO findings (content_id, category, severity, original, why, rule_quote,
                          suggestion, confidence, source) VALUES (?,?,?,?,?,?,?,?,?)""",
                       (cid, f["category"], f["severity"], f["original"], f["why"], f["rule_quote"],
                        f["suggestion"], f["confidence"], f["source"]))
        log(cid, f"Submitted draft. {len(findings)} issue(s) found.")
        db.commit()
        return redirect(url_for("view_content", cid=cid))
    return render(CHECK, has_rules=bool(row), sample=bool(request.args.get("sample")), draft=SAMPLE_DRAFT)


LIST = r"""
<h1>All Content</h1>
<div class="card">
{% if items %}
<table><tr><th>Title</th><th>Status</th><th>Version</th><th>Issues</th><th>Created</th></tr>
{% for c in items %}<tr><td><a style="color:var(--cyan)" href="/content/{{c.id}}">{{ c.title }}</a></td>
<td><span class="badge b-status">{{ labels[c.status] }}</span></td><td>v{{ c.version }}</td><td>{{ c.n }}</td><td class="small muted">{{ c.created }}</td></tr>{% endfor %}
</table>
{% else %}<p class="muted">Nothing here yet. Go to <a style="color:var(--cyan)" href="/check">New Check</a>.</p>{% endif %}
</div>
"""


@app.route("/content")
def list_content():
    items = get_db().execute("""SELECT c.*, (SELECT COUNT(*) FROM findings f WHERE f.content_id=c.id) AS n
                                FROM content c ORDER BY c.id DESC""").fetchall()
    return render(LIST, items=items, labels=STATUS_LABEL)


DETAIL = r"""
<h1>{{ c.title }}</h1>
<div class="row" style="margin-bottom:6px">
<span class="badge b-status">{{ labels[c.status] }}</span><span class="badge">Version {{ c.version }}</span>
<span class="small muted">Created {{ c.created }} UTC</span></div>
<p class="small muted">{{ ai_note }}</p>

<div class="card"><h2>Current text</h2>
<div class="textbox">{{ highlighted }}</div>
<p class="small muted">Highlighted words have an issue: <mark class="m-pending">waiting</mark> <mark class="m-accepted">accepted</mark> <mark class="m-ignored">ignored</mark></p>
</div>

<div class="card"><h2>Issues found ({{ findings|length }})</h2>
{% if not findings %}<p class="muted">No issues found. This content follows the saved rules.</p>{% endif %}
{% for f in findings %}
<div class="card" style="background:var(--panel2)">
<div class="row"><b>#{{ f.id }}</b>
<span class="badge">{{ f.category }}</span><span class="badge b-{{ f.severity }}">{{ f.severity }}</span>
<span class="badge">{{ f.source }}</span>
{% if f.confidence < 0.6 %}<span class="badge b-check">please check by hand</span>{% endif %}
<span class="badge b-status">{{ f.decision }}{% if f.applied %} (applied){% endif %}</span></div>
<p style="margin:8px 0 2px"><span class="orig">{{ f.original }}</span></p>
<p style="margin:2px 0"><b>Why:</b> {{ f.why }}</p>
<div class="rule">Rule: {{ f.rule_quote }}</div>
{% if can_decide and not f.applied %}
<form method="post" action="/content/{{ c.id }}/finding/{{ f.id }}" class="row">
<input type="text" name="final_text" value="{{ f.final_text if f.final_text is not none else f.suggestion }}" style="flex:1;min-width:200px" aria-label="Suggested fix (you can edit it)">
<button class="btn green" name="decision" value="accepted">Accept</button>
<button class="btn alt" name="decision" value="ignored">Ignore</button></form>
{% else %}<p class="small"><b>Suggested fix:</b> {{ f.final_text if f.final_text is not none else f.suggestion }}</p>{% endif %}
</div>{% endfor %}
{% if can_decide and to_apply %}
<form method="post" action="/content/{{ c.id }}/apply"><button class="btn amber">Create new version with {{ to_apply }} accepted fix(es)</button></form>
{% elif not can_decide and findings %}<p class="small muted">Switch to Creator or Brand Manager to accept or ignore fixes.</p>{% endif %}
</div>

<div class="card"><h2>Review and approval</h2>
<p class="small muted">{{ who_next }}</p>
<form method="post" action="/content/{{ c.id }}/transition">
<textarea name="comment" style="min-height:70px" placeholder="Add a comment (optional)"></textarea>
<div class="row" style="margin-top:8px">
{% for to,label in actions %}<button class="btn {{ 'green' if to=='approved' else ('red' if to=='changes_requested' else '') }}" name="to" value="{{ to }}">{{ label }}</button>{% endfor %}
{% if not actions %}<span class="small muted">No action available for the {{ roles[role] }} right now.</span>{% endif %}
</div></form>
</div>

<div class="grid">
<div class="card"><h2>Comments</h2>
{% for m in comments %}<p class="small"><b>{{ roles[m.role] }}</b> <span class="muted">{{ m.created }}</span><br>{{ m.text }}</p>{% else %}<p class="small muted">No comments yet.</p>{% endfor %}
<form method="post" action="/content/{{ c.id }}/comment"><input type="text" name="comment" placeholder="Write a comment" required>
<button class="btn alt" style="margin-top:6px">Add comment</button></form></div>
<div class="card"><h2>Version history</h2>
{% for v in versions %}<details {% if loop.first %}open{% endif %}><summary>Version {{ v.n }} - {{ v.note }} <span class="muted small">({{ v.created }})</span></summary>
<div class="textbox small">{{ v.text }}</div></details>{% endfor %}</div>
</div>

<div class="card"><h2>Activity log</h2>
{% for a in activity %}<p class="small" style="margin:3px 0"><span class="muted">{{ a.created }}</span> - <b>{{ roles[a.role] }}</b>: {{ a.action }}</p>{% endfor %}</div>
"""


def load_content(cid):
    c = get_db().execute("SELECT * FROM content WHERE id=?", (cid,)).fetchone()
    if not c:
        abort(404)
    return c


@app.route("/content/<int:cid>")
def view_content(cid):
    db = get_db()
    c = load_content(cid)
    findings = db.execute("SELECT * FROM findings WHERE content_id=? ORDER BY id", (cid,)).fetchall()
    role = current_role()
    actions = [(to, BUTTON_LABEL[to]) for (frm, to), roles_ok in TRANSITIONS.items()
               if frm == c["status"] and role in roles_ok]
    to_apply = sum(1 for f in findings if f["decision"] == "accepted" and not f["applied"])
    return render(DETAIL, c=c, findings=findings, labels=STATUS_LABEL, ai_note=c["ai_note"],
                  highlighted=highlight(c["text"], findings),
                  can_decide=role in ("creator", "manager") and c["status"] in ("ai_reviewed", "changes_requested"),
                  to_apply=to_apply, actions=actions, who_next=WHO_NEXT.get(c["status"], ""),
                  comments=db.execute("SELECT * FROM comments WHERE content_id=? ORDER BY id", (cid,)).fetchall(),
                  versions=db.execute("SELECT * FROM versions WHERE content_id=? ORDER BY n DESC", (cid,)).fetchall(),
                  activity=db.execute("SELECT * FROM activity WHERE content_id=? ORDER BY id DESC", (cid,)).fetchall())


@app.route("/content/<int:cid>/finding/<int:fid>", methods=["POST"])
def decide(cid, fid):
    c = load_content(cid)
    if current_role() not in ("creator", "manager") or c["status"] not in ("ai_reviewed", "changes_requested"):
        flash("You cannot change fixes right now.", "error")
        return redirect(url_for("view_content", cid=cid))
    decision = request.form.get("decision")
    if decision not in ("accepted", "ignored"):
        abort(400)
    final = request.form.get("final_text", "").strip()
    db = get_db()
    db.execute("UPDATE findings SET decision=?, final_text=? WHERE id=? AND content_id=?", (decision, final, fid, cid))
    log(cid, f"{decision.capitalize()} issue #{fid}.")
    db.commit()
    return redirect(url_for("view_content", cid=cid))


@app.route("/content/<int:cid>/apply", methods=["POST"])
def apply_fixes(cid):
    c = load_content(cid)
    if current_role() not in ("creator", "manager") or c["status"] not in ("ai_reviewed", "changes_requested"):
        flash("You cannot create a new version right now.", "error")
        return redirect(url_for("view_content", cid=cid))
    db = get_db()
    rows = db.execute("SELECT * FROM findings WHERE content_id=? AND decision='accepted' AND applied=0", (cid,)).fetchall()
    if not rows:
        flash("Accept at least one fix first.", "error")
        return redirect(url_for("view_content", cid=cid))
    text = c["text"]
    for f in rows:
        text = apply_text(text, f)
    n = c["version"] + 1
    db.execute("UPDATE content SET text=?, version=? WHERE id=?", (text, n, cid))
    db.execute("INSERT INTO versions (content_id, n, text, note, created) VALUES (?,?,?,?,?)",
               (cid, n, text, f"Applied {len(rows)} accepted fix(es)", now()))
    db.execute("UPDATE findings SET applied=1 WHERE content_id=? AND decision='accepted' AND applied=0", (cid,))
    log(cid, f"Created version {n} with {len(rows)} accepted fix(es).")
    db.commit()
    flash(f"Version {n} created.")
    return redirect(url_for("view_content", cid=cid))


@app.route("/content/<int:cid>/comment", methods=["POST"])
def add_comment(cid):
    load_content(cid)
    text = request.form.get("comment", "").strip()
    if text:
        db = get_db()
        db.execute("INSERT INTO comments (content_id, role, text, created) VALUES (?,?,?,?)",
                   (cid, current_role(), text[:1000], now()))
        log(cid, "Added a comment.")
        db.commit()
    return redirect(url_for("view_content", cid=cid))


@app.route("/content/<int:cid>/transition", methods=["POST"])
def transition(cid):
    c = load_content(cid)
    to = request.form.get("to")
    allowed = TRANSITIONS.get((c["status"], to))
    if not allowed or current_role() not in allowed:
        flash("That step is not allowed for your role right now.", "error")
        return redirect(url_for("view_content", cid=cid))
    db = get_db()
    if to == "in_review":
        waiting = db.execute("SELECT COUNT(*) FROM findings WHERE content_id=? AND decision='accepted' AND applied=0",
                             (cid,)).fetchone()[0]
        if waiting:
            flash("You accepted some fixes but have not applied them. Click 'Create new version' first.", "error")
            return redirect(url_for("view_content", cid=cid))
    comment = request.form.get("comment", "").strip()
    if comment:
        db.execute("INSERT INTO comments (content_id, role, text, created) VALUES (?,?,?,?)",
                   (cid, current_role(), comment[:1000], now()))
    db.execute("UPDATE content SET status=? WHERE id=?", (to, cid))
    log(cid, f"Changed status to '{STATUS_LABEL[to]}'.")
    db.commit()
    flash(f"Status is now: {STATUS_LABEL[to]}.")
    return redirect(url_for("view_content", cid=cid))


REPORT = r"""
<h1>Repeated Mistakes Report</h1>
<p class="muted">Shows which problems keep coming back, and which rules people often ignore (these may be unclear).</p>
<div class="grid">
<div class="card"><h2>Issues by type</h2>
{% if cats %}<table><tr><th>Type</th><th>Count</th></tr>{% for r in cats %}<tr><td>{{ r.category }}</td><td>{{ r.n }}</td></tr>{% endfor %}</table>
{% else %}<p class="muted small">No data yet.</p>{% endif %}</div>
<div class="card"><h2>Content by status</h2>
{% if stats %}<table><tr><th>Status</th><th>Count</th></tr>{% for r in stats %}<tr><td>{{ labels[r.status] }}</td><td>{{ r.n }}</td></tr>{% endfor %}</table>
{% else %}<p class="muted small">No data yet.</p>{% endif %}</div>
</div>
<div class="card"><h2>Most broken rules</h2>
{% if rules %}<table><tr><th>Rule</th><th>Times flagged</th><th>Ignored</th><th>Hint</th></tr>
{% for r in rules %}<tr><td class="small">{{ r.rule_quote }}</td><td>{{ r.n }}</td><td>{{ r.ign }}</td>
<td class="small">{% if r.n >= 2 and r.ign * 2 >= r.n %}<span class="badge b-check">Often ignored: this rule may be unclear</span>{% elif r.n >= 3 %}<span class="badge b-medium">Repeated mistake</span>{% endif %}</td></tr>{% endfor %}</table>
{% else %}<p class="muted small">No data yet. Run a few checks first.</p>{% endif %}</div>
"""


@app.route("/report")
def report():
    db = get_db()
    return render(REPORT, labels=STATUS_LABEL,
                  cats=db.execute("SELECT category, COUNT(*) n FROM findings GROUP BY category ORDER BY n DESC").fetchall(),
                  stats=db.execute("SELECT status, COUNT(*) n FROM content GROUP BY status ORDER BY n DESC").fetchall(),
                  rules=db.execute("""SELECT rule_quote, COUNT(*) n, SUM(decision='ignored') ign FROM findings
                                      GROUP BY rule_quote ORDER BY n DESC LIMIT 10""").fetchall())


@app.route("/health")
def health():
    return "ok"


if __name__ == "__main__":
    app.run(debug=True, port=5000)
