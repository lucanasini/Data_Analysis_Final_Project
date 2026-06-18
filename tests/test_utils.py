"""
test_utils.py
"""

from src.final_project.utils import load_config_json


def test_load_config(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text('{"a": 1}')

    config = load_config_json(str(path))
    assert config["a"] == 1