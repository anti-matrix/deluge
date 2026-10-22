"""Round 13 step 3: the bard (``analysis/bard.py``).

- The formulaic backend is deterministic; its grammar and epithet tables use no myth or motif
  vocabulary.
- Epithets are stable and traceable to the records and culture traits they come from.
- The llm backend's request holds only the narrator JSON, the culture traits and the
  instructions (a mock endpoint inspects it); the cache prevents a second call; a missing
  endpoint falls back to formulaic with a notice, and the UI works without one.
"""

import copy
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import yaml

from conftest import small_model
from test_memory import VOCABULARY

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from analysis import bard  # noqa: E402


@pytest.fixture(scope="module")
def ran():
    m = small_model(years=400)
    m.run()
    return m


def told_polity(m):
    """The living polity with the most records."""
    st = m.memory
    return max(m.polities.alive_ids().tolist(), key=lambda p: (len(st.stores.get(p, ())), -p))


def _strings(x, skip):
    if isinstance(x, str):
        if x not in skip:
            yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _strings(v, skip)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v, skip)


def test_grammar_and_epithets_use_no_myth_or_motif_vocabulary(ran):
    data = yaml.safe_load((REPO / "configs" / "bard.yaml").read_text(encoding="utf-8"))
    traits = set(ran.config.founders.culture_traits)
    words = list(_strings({"grammar": data["grammar"], "epithets": data["epithets"]}, traits))
    hits = sorted({h.group(0).lower() for w in words for h in VOCABULARY.finditer(w)})
    assert words and hits == []


def test_formulaic_is_deterministic(ran, tmp_path):
    pid = told_polity(ran)
    inp = bard.polity_input(ran, pid)
    assert inp["records"]
    a, b = bard.formulaic(inp), bard.formulaic(copy.deepcopy(inp))
    assert a == b and a
    other = small_model(years=400)
    other.run()
    assert bard.formulaic(bard.polity_input(other, pid)) == a
    out = bard.tell(inp, "formulaic", cache=tmp_path)
    assert out["text"] == a and out["label"] == "the bard's telling (formulaic)" and not out["cached"]
    # Only what the records hold: every name in the telling is a name the records give.
    names = {x["name"] for r in inp["records"] for x in r["actors"]} | {inp["subject"]["name"]}
    for r in inp["records"]:
        for x in r["actors"]:
            if x["people"] and x["role"] is not None and r["form"] == "sentence":
                assert x["name"] in a
    assert names


def test_epithets_are_stable_and_traceable(ran):
    pid = told_polity(ran)
    inp = bard.polity_input(ran, pid)
    eps = bard.epithets(inp)
    assert eps == bard.epithets(copy.deepcopy(inp)) == bard.epithets(bard.polity_input(ran, pid))
    data = bard.load()
    rules = {r[3]: r for r in data["epithets"]["records"]}
    culture = {r[3]: r for r in data["epithets"]["culture"]}
    by_id = {r["record"]: r for r in inp["records"]}
    for eid, items in eps.items():
        assert 1 <= len(items) <= data["epithets"]["max"]
        for e in items:
            assert e["from"]
            if e["epithet"] in rules:
                verb, role, least, _ = rules[e["epithet"]]
                ids = [int(x) for x in e["from"].split("(#")[1].rstrip(")").split(", #")]
                assert len(ids) >= least
                for i in ids:  # each record named holds the entity in that role
                    r = by_id[i]
                    assert r["action"] == verb and any(a["entity"] == eid and a["role"] == role for a in r["actors"])
            elif e["epithet"] in culture:
                trait = culture[e["epithet"]][0]
                assert eid == inp["subject"]["entity"] and f"{trait} {inp['culture'][trait]:.2f}" in e["from"]
            else:
                assert e["epithet"] == data["epithets"]["fallback"] and e.get("fallback")


class _Mock(BaseHTTPRequestHandler):
    calls: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        type(self).calls.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
        reply = json.dumps({"choices": [{"message": {"content": "A telling from the mock endpoint."}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):
        pass


@pytest.fixture
def endpoint():
    _Mock.calls = []
    server = HTTPServer(("127.0.0.1", 0), _Mock)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1", _Mock.calls
    server.shutdown()


def llm_data(base_url="", key="", per_day=10):
    data = copy.deepcopy(bard.load())
    data["settings"]["backend"] = "llm"
    data["settings"]["llm"].update(base_url=base_url, model="mock-model" if base_url else "", api_key=key,
                                   requests_per_day=per_day, base_url_env="DELUGE_TEST_UNSET_URL",
                                   model_env="DELUGE_TEST_UNSET_MODEL", api_key_env="DELUGE_TEST_UNSET_KEY")
    return data


def _keys(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from _keys(v)
    elif isinstance(x, list):
        for v in x:
            yield from _keys(v)


def test_llm_request_holds_only_narrator_json_culture_and_instructions(ran, endpoint, tmp_path):
    url, calls = endpoint
    inp = bard.polity_input(ran, told_polity(ran))
    data = llm_data(url, key="secret")
    out = bard.tell(inp, data=data, cache=tmp_path)
    assert out["text"] == "A telling from the mock endpoint." and out["label"] == "the bard's telling (llm: mock-model)"
    assert len(calls) == 1 and calls[0]["path"] == "/v1/chat/completions" and calls[0]["auth"] == "Bearer secret"
    body = calls[0]["body"]
    assert body["model"] == "mock-model" and [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == data["settings"]["llm"]["instructions"]
    sent = json.loads(body["messages"][1]["content"])
    assert set(sent) == {"subject", "culture", "records"}
    assert sent["records"] == json.loads(json.dumps(inp["records"])) and sent["culture"] == inp["culture"]
    keys = set(_keys(sent))
    assert not keys & {"provenance", "events", "event", "magnitude", "vividness", "symbols", "holders", "signatures"}
    # The export is exactly what was sent, with the instructions.
    exported = bard.export_input(inp, data)
    assert exported == {"instructions": data["settings"]["llm"]["instructions"], "input": sent}
    # The cache: the same records never cost a second call.
    again = bard.tell(inp, data=data, cache=tmp_path)
    assert again["cached"] and again["text"] == out["text"] and len(calls) == 1


def test_llm_falls_back_without_an_endpoint_or_past_the_limit(ran, endpoint, tmp_path):
    inp = bard.polity_input(ran, told_polity(ran))
    out = bard.tell(inp, data=llm_data(), cache=tmp_path)
    assert out["backend"] == "formulaic" and "no endpoint" in out["notice"] and out["text"] == bard.formulaic(inp)
    url, calls = endpoint
    lim = llm_data(url, per_day=0)
    out = bard.tell(inp, data=lim, cache=tmp_path / "limit")
    assert out["backend"] == "formulaic" and "daily limit" in out["notice"] and calls == []


def test_ui_tells_on_demand_with_no_endpoint(ran, monkeypatch, tmp_path):
    import solara

    from deluge.viz import app

    data = llm_data()
    data["settings"]["cache_dir"] = str(tmp_path)
    monkeypatch.setattr(bard, "load", lambda path=None: data)
    session = app.Session(ran)
    pid = told_polity(ran)
    inp = app.bard_input(session, "polity", pid)
    out = app.bard_tell(inp)
    assert out["backend"] == "formulaic" and out["notice"] and out["text"]
    # The components render (nothing is told until the button is pressed).
    before = len(list(tmp_path.glob("*.json")))
    box, rc = solara.render(app.Bard(session, "polity", pid, 0), handle_error=False)
    rc.close()
    box, rc = solara.render(app.Myths(session, 0), handle_error=False)
    rc.close()
    assert len(list(tmp_path.glob("*.json"))) == before
    md = app.myths_export(session)
    assert md.startswith("# Myths of the peoples") and "What they remember (the narrator)" in md
