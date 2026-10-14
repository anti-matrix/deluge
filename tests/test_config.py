import dataclasses
import shutil

import pytest
import yaml

from deluge.config import DEFAULT_CONFIG_PATH, Config, load_config


def test_default_config_loads():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.run.years == 12900
    assert isinstance(cfg.rng.streams, tuple)
    assert "mesa" in cfg.rng.streams


def test_config_is_frozen():
    cfg = load_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.run.history_seed = 1  # type: ignore[misc]


def test_overrides_return_new_config():
    cfg = load_config()
    other = cfg.with_overrides({"run": {"history_seed": 7}})
    assert other.run.history_seed == 7
    assert cfg.run.history_seed == 1
    assert other.run.world_seed == cfg.run.world_seed
    assert other.run.years == cfg.run.years
    assert other.hash() != cfg.hash()


def test_hash_is_stable():
    assert load_config().hash() == load_config().hash()


def _write(tmp_path, data):
    # The config names the knowledge graph file next to it (Round 4); copy it along.
    shutil.copy(DEFAULT_CONFIG_PATH.parent / data["knowledge"]["graph_file"], tmp_path)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def test_unknown_key_rejected(tmp_path):
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["run"]["seeed"] = 1
    with pytest.raises(KeyError, match="seeed"):
        load_config(_write(tmp_path, data))


def test_missing_key_rejected(tmp_path):
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    del data["run"]["years"]
    with pytest.raises(KeyError, match="years"):
        load_config(_write(tmp_path, data))


def test_wrong_type_rejected(tmp_path):
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["run"]["years"] = "many"
    with pytest.raises(TypeError):
        load_config(_write(tmp_path, data))
