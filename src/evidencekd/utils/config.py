"""YAML config with attribute access and a PyYAML-free fallback.

Same shape as ``vindr_cxr_vqa.config``: relative values under ``paths:`` are
resolved against the repository root (the parent of ``configs/``), so a
config works from any working directory and the Kaggle notebook can rewrite
only the paths it moves.
"""
from __future__ import annotations

import json
import os
from typing import Any


class Config(dict):
    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError:
            raise AttributeError(f"config has no key {k!r}; keys: {sorted(self)}") from None
        return Config(v) if isinstance(v, dict) and not isinstance(v, Config) else v

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self))


def _coerce(s: str) -> Any:
    s = s.strip()
    if s == "" or s == "null" or s == "~":
        return None
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        return [_coerce(x) for x in inner.split(",")] if inner else []
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _strip_comment(line: str) -> str:
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out).rstrip()


def _mini_yaml(text: str) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw in text.splitlines():
        line = _strip_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, rest = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if rest.strip() == "":
            child: dict = {}
            parent[key.strip()] = child
            stack.append((indent, child))
        else:
            parent[key.strip()] = _coerce(rest)
    return root


def load_config(path: str) -> Config:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text) or {}
    except ImportError:
        data = _mini_yaml(text)
    cfg = Config(data)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    if "paths" in cfg and isinstance(cfg["paths"], dict):
        for k, v in list(cfg["paths"].items()):
            if isinstance(v, str) and v and not os.path.isabs(v):
                cfg["paths"][k] = os.path.normpath(os.path.join(repo_root, v)).replace("\\", "/")
    return cfg


def dump_config(cfg: Config, path: str) -> None:
    """Write as YAML when PyYAML is present, else as JSON that the mini parser
    cannot read — so the fallback writes a flat YAML by hand."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        import yaml  # type: ignore
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
        return
    except ImportError:
        pass

    def emit(d: dict, indent: int, out: list[str]) -> None:
        for k, v in d.items():
            pad = " " * indent
            if isinstance(v, dict):
                out.append(f"{pad}{k}:")
                emit(v, indent + 2, out)
            elif isinstance(v, list):
                out.append(f"{pad}{k}: [{', '.join(json.dumps(x) for x in v)}]")
            elif v is None:
                out.append(f"{pad}{k}: null")
            else:
                out.append(f"{pad}{k}: {json.dumps(v)}")

    lines: list[str] = []
    emit(cfg.to_dict(), 0, lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
