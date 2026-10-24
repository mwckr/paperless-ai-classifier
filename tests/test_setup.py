"""Guided setup: fresh install asks the essentials, update appends new keys, checks report connections."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.mocks import TOKEN, MockServers  # noqa: E402


def _fresh_env(monkeypatch_env):
    tmp = tempfile.mkdtemp(prefix="pcls-setup-")
    env_file = Path(tmp, ".env")
    os.environ["PCLS_ENV"] = str(env_file)
    for mod in [m for m in sys.modules if m.startswith("app")]:
        del sys.modules[mod]
    import app.setup as setup
    import app.config as config
    assert config.ENV_FILE == env_file
    return setup, config, env_file


def test_fresh_interactive_then_update_then_check():
    with MockServers() as m:
        setup, config, env_file = _fresh_env(None)
        answers = iter([m.paperless_url, m.ollama_url, "gemma4:e4b", "2", "false", "0", "18999"])
        setup.input = lambda prompt="": next(answers)          # noqa: E731  (PAPERLESS_TOKEN comes from getpass)
        setup.getpass.getpass = lambda prompt="": TOKEN
        assert setup.run(interactive=True, check_only=False) == 0
        text = env_file.read_text()
        assert f"PAPERLESS_URL={m.paperless_url}" in text and f"PAPERLESS_TOKEN={TOKEN}" in text
        assert "MAX_PAGES=2" in text and "AUTO_COMMIT=false" in text and "API_PORT=18999" in text
        assert "OCR_TEXT_MODE=off" in text and "# ---" in text, "all other keys written with defaults and comments"
        s = config.Settings.load()
        assert s.MAX_PAGES == 2 and s.AUTO_COMMIT is False and s.API_PORT == 18999

        # simulate an older .env that lacks keys introduced later
        lines = [ln for ln in text.splitlines() if not ln.startswith(("OLLAMA_NUM_CTX=", "SKIP_CLASSIFIED=", "MAX_PAGES="))]
        env_file.write_text("\n".join(lines) + "\n")
        answers = iter(["1"])                                   # MAX_PAGES is essential -> asked
        assert setup.run(interactive=True, check_only=True) == 0
        text = env_file.read_text()
        assert "Added by update" in text and "OLLAMA_NUM_CTX=4096" in text and "SKIP_CLASSIFIED=true" in text and "MAX_PAGES=1" in text

        # complete file: check is silent and succeeds, also non-interactively
        assert setup.run(interactive=False, check_only=True) == 0


def test_non_interactive_requires_token():
    with MockServers():
        setup, config, env_file = _fresh_env(None)
        assert setup.run(interactive=False, check_only=False) == 1   # no token -> fail
        assert env_file.exists() and "PAPERLESS_TOKEN=" in env_file.read_text()
        env_file.write_text(env_file.read_text().replace("PAPERLESS_TOKEN=", f"PAPERLESS_TOKEN={TOKEN}"))
        assert setup.run(interactive=False, check_only=False) == 0


def test_connection_checks():
    with MockServers() as m:
        setup, _, _ = _fresh_env(None)
        assert setup.check_paperless(m.paperless_url, TOKEN) == "ok"
        assert setup.check_paperless(m.paperless_url, "wrong") == "token rejected"
        assert setup.check_paperless("http://127.0.0.1:1", TOKEN).startswith("unreachable")
        assert setup.check_ollama(m.ollama_url, "gemma4:e4b") == "ok"
        assert setup.check_ollama(m.ollama_url, "nope:1b").startswith("model not found")
