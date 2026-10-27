"""The bard: a literary pass that retells a people's or a polity's "What they remember"
records as their myth (analysis only; nothing here feeds back into the simulation).

    from analysis import bard
    inp = bard.polity_input(model, pid)        # or bard.lineage_input(model, g)
    out = bard.tell(inp)                       # {"text", "label", "backend", "model", "notice", ...}

Input (``polity_input``, ``lineage_input``): the narrator's structured JSON for the records
(``narrator.record_json``: action, each actor's role and name, place, year span or "the
first age", the narrator's sentence, what memory changed) plus the culture traits of the
people telling it, as tone. Nothing else: no event log, no provenance, no sim state. A
polity tells its ``max_records`` most vivid records; a lineage its origins (the narrator's
"Origins": records descending from years 0 to ``reports.origins_until_year``). Both are told in
time order.

Backends (``settings.backend`` in ``configs/bard.yaml``):

- ``formulaic`` (default; deterministic, offline): an oral-formulaic grammar. An opening ("In
  the first age...", "When a monument was raised at Mebu..."), one stock line per record by
  its verb and roles, a refrain every few records, a closing formula. Every choice is a hash
  of the telling and the record; culture traits pick the tone of the formulas (ceremony for
  high ritual investment, battle for high militarism), never what is told. Records no line
  fits, and loose ones, are said as the narrator says them.
- ``llm``: any OpenAI-compatible chat endpoint (base URL, model and API key from the settings
  or the environment), given the instructions, the narrator JSON and the culture traits only;
  at most ``requests_per_day`` calls. A missing or failing endpoint falls back to formulaic,
  with a notice.

Epithets (``epithets``): every group of people named in a telling gets one to three fixed
epithets, from how often it holds each role in the telling's records and, for the people
telling it, their culture traits; each lists what it comes from. Pure functions of the input.

Every telling is cached by (record ids, record versions, backend, model) and the culture traits,
so the same records never cost a second call. ``export_input`` is exactly what an LLM receives,
with the instructions, for pasting into a chat; ``myths_markdown`` writes "Myths of the peoples".

    python analysis/bard.py --checkpoint runs/<run>/checkpoints/year_03000.pkl --out runs/<run>/bard
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
import zlib
from functools import lru_cache
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BARD_FILE = REPO / "configs" / "bard.yaml"
FIELD = re.compile(r"\{(\w+)\}")


class BardError(RuntimeError):
    """The llm backend could not tell it (no endpoint, a failed call, the daily limit)."""


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load(path: str | Path = BARD_FILE) -> dict:
    return _load(str(path))


# --- the input -------------------------------------------------------------------------------------


def _culture(model, values) -> dict:
    return {t: round(float(v), 3) for t, v in zip(model.config.founders.culture_traits, values)}


def _subject(model, kind: str, i: int) -> dict:
    from deluge.data.events import entity
    from deluge.naming import Speaker

    eid = entity(kind, i)
    g = i if kind == "lineage" else int(model.polities.lineage[i])
    return {"kind": kind, "id": int(i), "entity": eid, "name": Speaker(model).entity(eid),
            "group": model.lineages.names[g]}


def _records(model, recs, hist=None, extra=None) -> list[dict]:
    """The records as the narrator gives them, in time order; ``extra``: fields to add per record."""
    from analysis.narrator import History, record_json
    from deluge.naming import Speaker

    sp, hist = Speaker(model), hist or History(model)
    out = [{**record_json(model, sp, r, hist), **(extra[k] if extra else {})} for k, r in enumerate(recs)]
    return sorted(out, key=lambda r: (r["year"], r["record"]))


def polity_input(model, pid: int, settings: dict | None = None) -> dict:
    """A polity's telling: its ``max_records`` most vivid records, its culture traits."""
    s = (settings or load())["settings"]
    st = model.memory
    recs = sorted((st.records[m] for m in st.stores.get(int(pid), ())), key=lambda r: (-r.vividness, r.mem_id))
    return {"subject": _subject(model, "polity", int(pid)),
            "culture": _culture(model, model.polities.culture[int(pid)]),
            "records": _records(model, recs[: s["max_records"]])}


def lineage_input(model, g: int, settings: dict | None = None) -> dict:
    """A founding people's telling, from its whole memory: the records most widely held across
    its living polities (``max_records``, ranked by vividness times the share of its polities
    holding them), after them, in a section of their own, the records of the first years
    (provenance in years 0 to ``reports.origins_until_year``; ``first_years_records``, ranked
    alike) when any survive; and its living polities' culture traits (weighted by their people).
    Each record carries its ``section`` ("first_years" or "held") and ``held_by`` (that share)."""
    import numpy as np

    from analysis.provenance import held_records

    s = (settings or load())["settings"]
    P = model.polities
    pids = [p for p in P.alive_ids().tolist() if P.lineage[p] == g]
    w = np.array([max(float(P.population[p]), 1e-9) for p in pids])
    culture = (w @ P.culture[pids]) / w.sum() if pids else np.full(P.culture.shape[1], 0.5)
    held, _ = held_records(model, int(g))
    ranked = sorted(held, key=lambda e: (-e["vividness"] * e["share"], e["record"].mem_id))
    first = [e for e in ranked if e["early"]][: s["first_years_records"]]
    rest = [e for e in ranked if not e["early"]][: s["max_records"]]
    chosen = [(e, "first_years") for e in first] + [(e, "held") for e in rest]
    recs = _records(model, [e["record"] for e, _ in chosen],
                    extra=[{"section": sec, "held_by": round(e["share"], 3)} for e, sec in chosen])
    order = {"first_years": 0, "held": 1}
    recs.sort(key=lambda r: order[r["section"]])  # stable: each section in time order
    return {"subject": _subject(model, "lineage", int(g)), "culture": _culture(model, culture), "records": recs}


def has_records(model, kind: str, i: int) -> bool:
    """Whether a polity (its store) or a founding lineage (any living polity's store) holds a record."""
    st, P = model.memory, model.polities
    if kind == "polity":
        return bool(st.stores.get(int(i)))
    return any(st.stores.get(p) for p in P.alive_ids().tolist() if P.lineage[p] == i)


def llm_content(inp: dict) -> dict:
    """What the llm backend sends as the user message: the narrator JSON and the culture traits."""
    return {"subject": {k: inp["subject"][k] for k in ("kind", "name", "group")}, "culture": inp["culture"],
            "records": inp["records"]}


def export_input(inp: dict, settings: dict | None = None) -> dict:
    """``bard_input.json``: exactly what an LLM would receive, and the instructions."""
    s = (settings or load())["settings"]
    return {"instructions": s["llm"]["instructions"], "input": llm_content(inp)}


# --- epithets ---------------------------------------------------------------------------------------


def epithets(inp: dict, data: dict | None = None) -> dict[int, list[dict]]:
    """Per group of people named in the telling (entity id): one to three epithets, each with
    what it comes from (``epithets`` in ``configs/bard.yaml``)."""
    data = data or load()
    e = data["epithets"]
    counts: dict[int, dict[tuple, list[int]]] = {}
    names: dict[int, str] = {}
    for r in inp["records"]:
        for a in r["actors"]:
            if not a["people"]:
                continue
            names[a["entity"]] = a["name"]
            if a["role"] is not None and r["form"] == "sentence":
                counts.setdefault(a["entity"], {}).setdefault((r["action"], a["role"]), []).append(r["record"])
    subject = inp["subject"]["entity"]
    names.setdefault(subject, inp["subject"]["name"])
    out: dict[int, list[dict]] = {}
    for eid in sorted(names):
        mine = counts.get(eid, {})
        kept, seen = [], set()
        for k, (verb, role, least, word) in enumerate(e["records"]):
            recs = mine.get((verb, role), [])
            # One epithet per verb and role: the first rule listed for it that holds.
            if len(recs) >= least and (verb, role) not in seen and word not in {f["epithet"] for f in kept}:
                seen.add((verb, role))
                kept.append({"epithet": word, "rank": (-len(recs), k),
                             "from": f"{role} of '{verb}' in {len(recs)} of the records told "
                                     f"(#{', #'.join(str(x) for x in sorted(recs))})"})
        kept.sort(key=lambda f: f["rank"])
        if eid == subject:
            for trait, side, bound, word in e["culture"]:
                v = inp["culture"].get(trait)
                if v is not None and ((side == "high" and v >= bound) or (side == "low" and v <= bound)):
                    kept.append({"epithet": word, "rank": (0, len(e["records"])),
                                 "from": f"culture trait {trait} {v:.2f} ({'at least' if side == 'high' else 'at most'} "
                                         f"{bound})"})
        if kept:  # no rule holds: no epithet
            out[eid] = [{"epithet": f["epithet"], "from": f["from"]} for f in kept[: e["max"]]]
    return out


# --- the formulaic backend -------------------------------------------------------------------------


def tone(culture: dict, settings: dict) -> str:
    thr = settings["tone_threshold"]
    best, value = "plain", thr
    for name, trait in settings["tone_traits"].items():
        v = culture.get(trait, 0.0)
        if v >= value:
            best, value = name, v
    return best


def _pick(options: dict | list, mood: str, *key) -> str:
    if isinstance(options, dict):
        options = options.get(mood) or options["plain"]
    return options[zlib.crc32("|".join(str(k) for k in key).encode("utf-8")) % len(options)]


def _counted(name: str, n: int, eid: int) -> str:
    """``n`` identical things, as the narrator counts them ("two monuments"; ``plural_forms`` in
    configs/narrator.yaml); a form that needs more than the name falls back to its default."""
    from deluge.data.events import decode
    from deluge.naming import load_data

    d = load_data(str(REPO / "configs" / "narrator.yaml"))
    forms = d["plural_forms"]
    form = forms.get(decode(eid)[0], forms["default"])
    if "{phrase}" in form:
        form = forms["default"]
    words = d["numbers"]
    return form.format(count=words[n] if n < len(words) else f"{n:,}", name=name)


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


class _Teller:
    def __init__(self, inp: dict, data: dict) -> None:
        self.inp, self.g, self.s = inp, data["grammar"], data["settings"]
        self.mood = tone(inp["culture"], self.s)
        self.ep = {k: v[0]["epithet"] for k, v in epithets(inp, data).items()}
        self.named: set[int] = set()
        self.key = (inp["subject"]["kind"], inp["subject"]["id"])

    def pick(self, options, *key) -> str:
        return _pick(options, self.mood, *self.key, *key)

    def roles(self, rec: dict, mention: bool = True) -> dict[str, str]:
        by: dict[str, list[str]] = {}
        things: dict[str, list] = {}  # identical things in one role: counted as the narrator counts them
        for a in rec["actors"]:
            if a["role"] is None:
                continue
            name = a["name"]
            if not a["people"]:
                seen = things.setdefault(a["role"], [])
                if any(n == name for n, _, _ in seen):
                    seen[[n for n, _, _ in seen].index(name)][1] += 1
                    continue
                seen.append([name, 1, a["entity"]])
            if mention and a["people"] and a["entity"] not in self.named:
                self.named.add(a["entity"])
                name = f"{name}, {self.ep[a['entity']]}," if a["entity"] in self.ep else name
            by.setdefault(a["role"], []).append(name)
        for role, seen in things.items():
            for name, n, eid in seen:
                if n > 1:
                    by[role][by[role].index(name)] = _counted(name, n, eid)
        return {r: " and ".join(v) for r, v in by.items()}

    def present(self, rec: dict) -> frozenset:
        return frozenset(a["role"] for a in rec["actors"] if a["role"] is not None)

    def fill(self, shape: str, values: dict, rec: dict) -> str:
        place = rec["place"]["name"]
        text = shape.format(**values, at=f" {place}" if place else "")
        return re.sub(r",([.;:])", r"\1", text).replace(",,", ",")  # an epithet's closing comma before a stop

    def when(self, rec: dict) -> str | None:
        if rec["form"] != "sentence":
            return None
        roles = self.present(rec)
        for shape in self.g["when"].get(rec["action"], ()):
            if set(FIELD.findall(shape)) - {"at"} <= roles:
                return self.fill(shape, self.roles(rec, mention=False), rec)
        return None

    def line(self, rec: dict, k: int) -> tuple[str, bool]:
        """(the record's line, whether it is the narrator's sentence, with its own time; a stock
        line is not capitalized yet)."""
        if rec["form"] != "sentence":
            return self.pick(self.g["loose"], rec["record"], "loose").format(text=rec["text"]), True
        roles = self.present(rec)
        sets = [name for name, rs in self.g["line_sets"].get(rec["action"], []) if frozenset(rs) == roles]
        sets = sets or [rec["action"]]
        for name in sets:
            options = self.g["lines"].get(name)
            if not options:
                continue
            for mood in (self.mood, "plain"):
                fits = [o for o in options.get(mood, []) if frozenset(FIELD.findall(o)) - {"at"} == roles]
                if fits:
                    shape = _pick(fits, mood, *self.key, rec["record"], k)
                    return self.fill(shape, self.roles(rec), rec), False
        return self.pick(self.g["said"], rec["record"], "said").format(text=rec["text"]), True

    def tell(self) -> str:
        g, inp = self.g, self.inp
        subject = inp["subject"]["name"]
        recs = inp["records"]
        if not recs:
            return _cap(self.pick(g["nothing"], "nothing").format(Subject=_cap(subject), subject=subject))
        first = recs[0]
        first_age = first["year"] == 0 and first["year_span"] > 1
        if first_age:
            lines = [self.pick(g["openings"]["first_age"], "open")]
        else:
            deed = self.when(first)
            lines = [self.pick(g["openings"]["when"], "open").format(deed=deed)] if deed else \
                [self.pick(g["openings"]["none"], "open").format(subject=subject)]
        prev_time = first["time"] if first_age else None
        every = g["refrain"]["every"]
        sections = {r.get("section") for r in recs}
        section = None
        for k, rec in enumerate(recs):
            # A lineage's telling: the first years under a heading of their own, then the rest.
            if "first_years" in sections and rec.get("section") != section:
                section = rec.get("section")
                lines.append(self.pick(g["sections"][section], "section", section))
            text, timed = self.line(rec, k)
            if not timed and rec["time"] != prev_time:
                lead = self.pick(g["time_lead"], rec["record"], "time").format(time=rec["time"], Time=_cap(rec["time"]))
                first = text.split(" ", 1)[0]
                if first in self.g["lower_after_lead"]:  # a formula's own first word, not a name
                    text = text[:1].lower() + text[1:]
                text = f"{lead} {text}"
            prev_time = rec["time"]
            lines.append(_cap(text))
            if (k + 1) % every == 0 and k + 1 < len(recs):
                lines.append(self.pick({m: v for m, v in g["refrain"].items() if m != "every"}, "refrain", k))
        lines.append(self.pick(g["closing"], "close").format(subject=subject, Subject=_cap(subject)))
        return "\n".join(lines)


def formulaic(inp: dict, data: dict | None = None) -> str:
    """The formulaic backend's telling (deterministic)."""
    return _Teller(inp, data or load()).tell()


# --- the llm backend --------------------------------------------------------------------------------


def llm_settings(settings: dict) -> dict:
    """The llm settings, empty values filled from the environment."""
    c = dict(settings["llm"])
    for k in ("base_url", "model", "api_key"):
        if not c.get(k):
            c[k] = os.environ.get(c[f"{k}_env"], "")
    return c


def request_body(inp: dict, c: dict) -> dict:
    """The chat request: the instructions, then the narrator JSON and the culture traits."""
    return {"model": c["model"], "temperature": c["temperature"], "max_tokens": c["max_tokens"],
            "messages": [{"role": "system", "content": c["instructions"]},
                         {"role": "user", "content": json.dumps(llm_content(inp), ensure_ascii=False)}]}


def _count_request(cache: Path, limit: int) -> None:
    """Count one call today against ``requests_per_day``; BardError past it."""
    path = cache / "requests.json"
    today = datetime.date.today().isoformat()
    try:
        seen = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        seen = {}
    if seen.get(today, 0) >= limit:
        raise BardError(f"the daily limit of {limit} requests is reached")
    seen = {today: seen.get(today, 0) + 1}
    cache.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seen), encoding="utf-8")


def llm(inp: dict, settings: dict, cache: Path) -> str:
    c = llm_settings(settings)
    if not c["base_url"] or not c["model"]:
        raise BardError("no endpoint is configured (base_url and model)")
    _count_request(cache, int(c["requests_per_day"]))
    body = json.dumps(request_body(inp, c)).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if c["api_key"]:
        headers["Authorization"] = f"Bearer {c['api_key']}"
    req = urllib.request.Request(c["base_url"].rstrip("/") + "/chat/completions", data=body, headers=headers,
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=float(c["timeout_s"])) as resp:
            reply = json.loads(resp.read().decode("utf-8"))
        text = reply["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise BardError(f"the endpoint failed: {exc}") from exc
    if not text:
        raise BardError("the endpoint answered with no text")
    return text


# --- telling, with the cache -----------------------------------------------------------------------


def cache_key(inp: dict, backend: str, model: str) -> str:
    recs = inp["records"]
    blob = json.dumps([[r["record"] for r in recs], [r["version"] for r in recs], backend, model, inp["culture"],
                       inp["subject"]["kind"], inp["subject"]["id"]], sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cache_dir(settings: dict) -> Path:
    p = Path(settings["cache_dir"])
    return p if p.is_absolute() else REPO / p


def tell(inp: dict, backend: str | None = None, data: dict | None = None, cache: str | Path | None = None) -> dict:
    """The bard's telling of ``inp`` with the configured backend (or ``backend``), from the cache
    when these records were told before: {"text", "label", "backend", "model", "notice", "cached"}.
    The llm backend falls back to formulaic when it cannot tell it; ``notice`` says why."""
    data = data or load()
    s = data["settings"]
    backend = backend or s["backend"]
    where = Path(cache) if cache is not None else cache_dir(s)
    notice = ""
    if backend == "llm":
        model = llm_settings(s)["model"]
        hit = _cached(where, cache_key(inp, "llm", model))
        if hit:
            return {**hit, "cached": True}
        try:
            text = llm(inp, s, where)
            out = {"text": text, "label": f"the bard's telling (llm: {model})", "backend": "llm", "model": model,
                   "notice": ""}
            _store(where, cache_key(inp, "llm", model), out)
            return {**out, "cached": False}
        except BardError as exc:
            notice = f"The llm backend could not tell it ({exc}); this is the formulaic telling."
    key = cache_key(inp, "formulaic", "")
    hit = _cached(where, key)
    if hit:
        return {**hit, "notice": notice, "cached": True}
    out = {"text": formulaic(inp, data), "label": "the bard's telling (formulaic)", "backend": "formulaic",
           "model": "", "notice": ""}
    _store(where, key, out)
    return {**out, "notice": notice, "cached": False}


def _cached(where: Path, key: str) -> dict | None:
    try:
        return json.loads((where / f"{key}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _store(where: Path, key: str, out: dict) -> None:
    where.mkdir(parents=True, exist_ok=True)
    (where / f"{key}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def cached_telling(inp: dict, data: dict | None = None, cache: str | Path | None = None) -> dict | None:
    """The telling already made for ``inp`` with the configured backend, if any (no call)."""
    data = data or load()
    s = data["settings"]
    where = Path(cache) if cache is not None else cache_dir(s)
    if s["backend"] == "llm":
        hit = _cached(where, cache_key(inp, "llm", llm_settings(s)["model"]))
        if hit:
            return hit
    return _cached(where, cache_key(inp, "formulaic", ""))


# --- exports ----------------------------------------------------------------------------------------


def myths_markdown(model, inputs: list[dict], data: dict | None = None, cache: str | Path | None = None) -> str:
    """"Myths of the peoples": per telling, the narrator's plain text and, where one was made,
    the bard's telling, with the epithets and what each comes from."""
    data = data or load()
    m = model
    out = ["# Myths of the peoples", "",
           f"World seed {m.config.run.world_seed}, history seed {m.config.run.history_seed}, year {m.year}. "
           "Per people or polity: what they remember, as the narrator tells it, and beside it the bard's telling "
           "where one was made. The bard adds nothing to the records.", ""]
    for inp in inputs:
        subj = inp["subject"]
        out += [f"## {_cap(subj['name'])} (group {subj['group']}, {subj['kind']} {subj['id']})", "",
                "Culture traits: " + ", ".join(f"{k.replace('_', ' ')} {v:.2f}" for k, v in inp["culture"].items()),
                "", "### What they remember (the narrator)", ""]
        heads = {"first_years": "Of the first years", "held": "Most widely held across its polities"}
        section = None
        for r in inp["records"]:
            if r.get("section") and r["section"] != section:
                section = r["section"]
                out += ["", f"*{heads[section]}:*", ""] if out[-1] else [f"*{heads[section]}:*", ""]
            share = f" (held by {100 * r['held_by']:.0f}% of its polities)" if "held_by" in r else ""
            out.append(f"- {r['text']}{share}")
        if not inp["records"]:
            out.append("- Nothing.")
        out.append("")
        told = cached_telling(inp, data, cache)
        if told:
            out += [f"### {_cap(told['label'])}", ""] + told["text"].splitlines() + [""]
        names = {a["entity"]: a["name"] for r in inp["records"] for a in r["actors"]}
        names.setdefault(subj["entity"], subj["name"])
        out += ["### Epithets", ""]
        for eid, eps in epithets(inp, data).items():
            out.append(f"- {names[eid]}: " + "; ".join(f"\"{e['epithet']}\" ({e['from']})" for e in eps))
        out.append("")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    import pickle

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="a checkpoint (scripts/run.py) or a pickled model")
    p.add_argument("--out", required=True)
    p.add_argument("--polities", type=int, default=5, help="tell the largest N living polities (by people)")
    p.add_argument("--backend", choices=("formulaic", "llm"))
    args = p.parse_args(argv)
    with open(args.checkpoint, "rb") as f:
        state = pickle.load(f)
    model = state["model"] if isinstance(state, dict) else state
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    P = model.polities
    pids = sorted(P.alive_ids().tolist(), key=lambda q: (-float(P.population[q]), q))[: args.polities]
    inputs = [polity_input(model, q) for q in pids]
    inputs += [lineage_input(model, g) for g in range(len(model.lineages.founders))]
    for inp in inputs:
        s = inp["subject"]
        (out / f"bard_input_{s['kind']}_{s['id']}.json").write_text(
            json.dumps(export_input(inp), ensure_ascii=False, indent=1), encoding="utf-8")
        if inp["records"]:
            told = tell(inp, args.backend)
            print(f"{s['name']}: {told['label']}{' (cached)' if told['cached'] else ''} {told['notice']}")
    (out / "myths_of_the_peoples.md").write_text(myths_markdown(model, inputs), encoding="utf-8")
    print(f"wrote {len(inputs)} inputs and the myths to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
