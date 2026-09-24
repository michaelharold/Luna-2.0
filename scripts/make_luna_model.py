#!/usr/bin/env python
# scripts/make_luna_model.py
"""
make_luna_model.py — build the custom Ollama model Luna talks to.

WHY a custom model at all? brain.py already sends persona + knowledge as the
system message on every request, so the robot works from the bare base model
too (llm_ollama falls back to OLLAMA_BASE_MODEL with a hint). The "luna" build
exists so that

  • `ollama run luna` on the Pi (or the Mac) behaves EXACTLY like the robot —
    same persona, same knowledge, same sampling — for quick manual checks and
    for the benchmark (scripts/bench_llm.py) to measure what actually ships;
  • the generation parameters (temperature, num_ctx, num_predict) travel with
    the model, so a bare `ollama run luna` doesn't silently use different ones;
  • rebuilding after a knowledge.txt edit is one command, and the Modelfile
    is kept under models/llm/ as a record of what was baked.

The text is persona.system_prompt(...) — the SAME function brain.py uses — so
the two can never drift. With OLLAMA_BAKE_KNOWLEDGE the whole knowledge base
goes into the SYSTEM (Ollama prefix-caches it, so it's prefilled once per
load, not per question); --no-bake gives the persona alone.

Usage (from the project root, venv active):
    python scripts/make_luna_model.py                  # luna from the config base
    python scripts/make_luna_model.py --base gemma2:2b --name luna-gemma
    python scripts/make_luna_model.py --dry-run        # print the Modelfile only
"""

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import config   # noqa: E402
import persona  # noqa: E402

MODELFILE_DIR = os.path.join(ROOT, "models", "llm")
PROMPT_WARN_CHARS = 20000   # past this the baked prefix starts hurting prefill


def escape_triple_quotes(text):
    """A SYSTEM \"\"\"...\"\"\" block ends at the first triple quote inside
    it; break any such run so the prompt can never terminate itself early."""
    return text.replace('"""', '\\"\\"\\"')


def build_system(bake=True, knowledge_path=None):
    """The system prompt text: persona alone, or persona + knowledge base."""
    if not bake:
        return persona.system_prompt(None)
    path = knowledge_path or os.path.join(ROOT, config.KNOWLEDGE_PATH)
    try:
        with open(path, "r") as f:
            knowledge = f.read()
    except OSError as e:
        print(f"[make_luna_model] Could not read {path} ({e}) — baking the "
              f"persona only")
        return persona.system_prompt(None)
    return persona.system_prompt(knowledge)


def render_modelfile(base, system_text,
                     temperature=config.LLM_TEMPERATURE,
                     num_ctx=config.OLLAMA_NUM_CTX,
                     num_predict=config.LLM_MAX_TOKENS):
    """The Modelfile as a string (pure — this is what the tests check)."""
    return (f"FROM {base}\n"
            f'SYSTEM """{escape_triple_quotes(system_text)}"""\n'
            f"PARAMETER temperature {temperature}\n"
            f"PARAMETER num_ctx {num_ctx}\n"
            f"PARAMETER num_predict {num_predict}\n")


def modelfile_path(name):
    """models/llm/Modelfile.<name> — tags and namespaces made file-safe."""
    safe = name.replace(":", "_").replace("/", "_")
    return os.path.join(MODELFILE_DIR, f"Modelfile.{safe}")


def create_model(name, path):
    """Run `ollama create`; return the process exit code (0 = built)."""
    exe = shutil.which("ollama")
    if not exe:
        print("[make_luna_model] `ollama` is not on PATH. Install Ollama "
              "(https://ollama.com) and make sure the service is running.")
        return 2
    cmd = [exe, "create", name, "-f", path]
    print(f"[make_luna_model] $ {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=600)
    except subprocess.TimeoutExpired:
        print("[make_luna_model] ollama create timed out after 600 s")
        return 1
    except OSError as e:
        print(f"[make_luna_model] could not run ollama: {e}")
        return 1
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if out:
        print(out)
    if proc.returncode != 0:
        print(f"[make_luna_model] ollama create failed (exit "
              f"{proc.returncode}): {err or 'no error text'}")
        if "pull" in err or "not found" in err:
            print(f"[make_luna_model] hint: the base model must be pulled "
                  f"first — `ollama pull <base>`")
        return proc.returncode
    print(f"[make_luna_model] built '{name}' — try: ollama run {name}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default=config.OLLAMA_BASE_MODEL,
                    help="base model tag (default: config.OLLAMA_BASE_MODEL)")
    ap.add_argument("--name", default=config.OLLAMA_MODEL,
                    help="name of the model to create (default: "
                         "config.OLLAMA_MODEL)")
    ap.add_argument("--no-bake", action="store_true",
                    help="persona only; do not bake data/knowledge.txt in")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the Modelfile and exit without creating")
    args = ap.parse_args(argv)

    bake = config.OLLAMA_BAKE_KNOWLEDGE and not args.no_bake
    system_text = build_system(bake=bake)
    modelfile = render_modelfile(args.base, system_text)

    n = len(system_text)
    print(f"[make_luna_model] system prompt: {n} characters "
          f"({'persona + knowledge' if bake else 'persona only'})")
    if n > PROMPT_WARN_CHARS:
        print(f"[make_luna_model] WARNING: prompt exceeds {PROMPT_WARN_CHARS} "
              f"characters — prefill on the Pi will be slow. Consider "
              f"OLLAMA_BAKE_KNOWLEDGE = False (RAG-lite still supplies the "
              f"facts per turn) or trimming knowledge.txt.")

    if args.dry_run:
        print(modelfile)
        return 0

    path = modelfile_path(args.name)
    try:
        os.makedirs(MODELFILE_DIR, exist_ok=True)
        with open(path, "w") as f:
            f.write(modelfile)
    except OSError as e:
        print(f"[make_luna_model] could not write {path}: {e}")
        return 1
    print(f"[make_luna_model] wrote {path}")
    return create_model(args.name, path)


if __name__ == "__main__":
    sys.exit(main())
