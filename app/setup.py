"""
Guided setup, used by install.sh (fresh install) and update.sh (every update).

  python -m app.setup            interactive: ask for missing essentials, add new keys, test connections
  python -m app.setup --check    same, but prints one line and exits 0 when nothing is missing
  python -m app.setup --non-interactive   never prompts; fills defaults, exits 1 when essentials are missing

Rules
- A complete .env with working connections: nothing is asked.
- Keys that were added in a newer version: essential ones are asked, the rest are
  appended with their default and a comment.
- Connection tests failing (wrong token, Ollama down, model not pulled): the
  related settings are offered for re-entry.
"""
import argparse
import getpass
import sys
from datetime import date
from typing import Dict, List, Optional

import requests
from dotenv import dotenv_values, set_key

from .config import ENV_FILE, FIELDS, GROUPS, Field, validate

_TRUE = ("1", "true", "yes", "on")


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _ask(field: Field, current: Optional[str]) -> str:
    """Prompt for one value until it validates. Enter keeps the current/default value."""
    default = current if current not in (None, "") else _fmt(field.default)
    print(f"\n{field.label}" + (f"  ({field.help})" if field.help else ""))
    if field.options:
        print(f"  options: {', '.join(field.options)}")
    while True:
        if field.type == "secret":
            shown = ("*" * 8 + default[-4:]) if default else "none"
            raw = getpass.getpass(f"  {field.key} [{shown}]: ")
        else:
            raw = input(f"  {field.key} [{default}]: ")
        raw = raw.strip() or default
        try:
            value = validate(field.key, raw)
        except ValueError as e:
            print(f"  {e}")
            continue
        if field.type == "secret" and not value:
            print("  A value is required.")
            continue
        return value


def _write_fresh(values: Dict[str, str]) -> None:
    lines = ["# Paperless AI Classifier configuration",
             f"# Created by the guided setup on {date.today().isoformat()}.",
             "# Every key can also be changed in the dashboard (Config tab).", ""]
    for group in GROUPS:
        lines.append(f"# --- {group} ---")
        for f in FIELDS:
            if f.group != group:
                continue
            if f.help:
                lines.append(f"# {f.help}")
            lines.append(f"{f.key}={values.get(f.key, _fmt(f.default))}")
        lines.append("")
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    ENV_FILE.write_text("\n".join(lines), encoding="utf-8")
    try:
        ENV_FILE.chmod(0o600)
    except OSError:
        pass


def _append(fields: List[Field], values: Dict[str, str]) -> None:
    with ENV_FILE.open("a", encoding="utf-8") as fh:
        fh.write(f"\n# --- Added by update on {date.today().isoformat()} ---\n")
        for f in fields:
            if f.help:
                fh.write(f"# {f.help}\n")
            fh.write(f"{f.key}={values.get(f.key, _fmt(f.default))}\n")


def check_paperless(url: str, token: str) -> str:
    try:
        r = requests.get(f"{url.rstrip('/')}/api/documents/", params={"page_size": 1},
                         headers={"Authorization": f"Token {token}"}, timeout=8)
        if r.status_code == 200:
            return "ok"
        if r.status_code in (401, 403):
            return "token rejected"
        return f"HTTP {r.status_code}"
    except requests.RequestException as e:
        return f"unreachable ({type(e).__name__})"


def check_ollama(url: str, model: str) -> str:
    try:
        r = requests.get(f"{url.rstrip('/')}/api/tags", timeout=8)
        if r.status_code != 200:
            return f"HTTP {r.status_code}"
        names = [m.get("name", "") for m in r.json().get("models", [])]
        if model in names or f"{model}:latest" in names or any(n.split(":")[0] == model for n in names):
            return "ok"
        return f"model not found (installed: {', '.join(names) or 'none'})"
    except requests.RequestException as e:
        return f"unreachable ({type(e).__name__})"


def run(interactive: bool, check_only: bool) -> int:
    existing = dict(dotenv_values(ENV_FILE)) if ENV_FILE.exists() else None
    values: Dict[str, str] = {k: v for k, v in (existing or {}).items() if v is not None}
    changed = False

    if existing is None:
        print(f"No configuration found, creating {ENV_FILE}")
        if interactive:
            print("Press Enter to accept the value in brackets.")
            for f in FIELDS:
                if f.ask:
                    values[f.key] = _ask(f, None)
        _write_fresh(values)
        values = {k: v for k, v in dotenv_values(ENV_FILE).items() if v is not None}
        changed = True
    else:
        missing = [f for f in FIELDS if f.key not in existing]
        if missing:
            print(f"{len(missing)} new setting(s) since this configuration was written.")
            to_ask = [f for f in missing if f.ask and interactive]
            for f in to_ask:
                values[f.key] = _ask(f, None)
            _append(missing, values)
            for f in missing:
                values.setdefault(f.key, _fmt(f.default))
            print("  added: " + ", ".join(f.key for f in missing))
            changed = True

    # essentials present?
    problems: List[str] = []
    for f in FIELDS:
        if f.ask and f.type == "secret" and not values.get(f.key):
            problems.append(f"{f.key} is empty")
    if problems and not interactive:
        for p in problems:
            print(f"ERROR: {p}")
        return 1

    # connection tests, with re-entry loop in interactive mode
    for attempt in range(3):
        pl = check_paperless(values.get("PAPERLESS_URL", ""), values.get("PAPERLESS_TOKEN", ""))
        ol = check_ollama(values.get("OLLAMA_URL", ""), values.get("OLLAMA_MODEL", ""))
        if pl == "ok" and ol == "ok":
            break
        print(f"Paperless: {pl}\nOllama:    {ol}")
        if not interactive:
            break
        retry: List[str] = []
        if pl != "ok":
            retry += ["PAPERLESS_URL", "PAPERLESS_TOKEN"]
        if ol != "ok":
            retry += ["OLLAMA_URL"] + (["OLLAMA_MODEL"] if "model not found" in ol else [])
        answer = input("Re-enter these settings now? [Y/n] ").strip().lower()
        if answer in ("n", "no"):
            if "model not found" in ol:
                print(f"Pull it on the Ollama host with:  ollama pull {values.get('OLLAMA_MODEL')}")
            break
        for key in retry:
            f = next(x for x in FIELDS if x.key == key)
            values[key] = _ask(f, values.get(key))
            set_key(str(ENV_FILE), key, values[key], quote_mode="never")
            changed = True
    else:
        pl = check_paperless(values.get("PAPERLESS_URL", ""), values.get("PAPERLESS_TOKEN", ""))
        ol = check_ollama(values.get("OLLAMA_URL", ""), values.get("OLLAMA_MODEL", ""))

    status = f"Paperless {pl}, Ollama {ol}"
    if check_only and not changed:
        print(f"Settings complete ({status})")
    else:
        print(f"Configuration written to {ENV_FILE} ({status})")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Guided setup for the Paperless AI Classifier")
    ap.add_argument("--check", action="store_true", help="quiet when nothing is missing (used by update.sh)")
    ap.add_argument("--non-interactive", action="store_true", help="never prompt; fill defaults")
    args = ap.parse_args()
    interactive = not args.non_interactive and sys.stdin.isatty()
    sys.exit(run(interactive=interactive, check_only=args.check))


if __name__ == "__main__":
    main()
