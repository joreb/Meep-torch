"""verify_comment_only_edit.py

RESULT: compares each changed file's AST with docstrings stripped against a
pre-edit snapshot. Comments never reach the AST, so any executable change shows.
Used to verify a 13-file stale-comment sweep touched no code. Needs a snapshot
directory; see the SNAP path inside.
"""
import ast, sys, pathlib, subprocess, os
ROOT = pathlib.Path("/home/papac/Documents/GitHub/misc/Meep-torch")
BASE = pathlib.Path(os.environ["CLAUDE_JOB_DIR"])/"tmp/base"

changed = subprocess.run(["git","diff","--name-only","--","meep_torch","symbolic_test","tests"],
                         cwd=ROOT, capture_output=True, text=True).stdout.split()

def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
               and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree

bad = []
for rel in changed:
    cur = ROOT/rel
    old = BASE/rel
    if not old.exists():
        bad.append(f"{rel}: NO BASELINE COPY"); continue
    a = strip_docstrings(ast.parse(old.read_text()))
    b = strip_docstrings(ast.parse(cur.read_text()))
    if ast.dump(a) != ast.dump(b):
        bad.append(f"{rel}: AST DIFFERS (executable change)")
    else:
        print(f"OK  {rel}")
print()
if bad:
    print("PROBLEMS:"); [print(" ", x) for x in bad]; sys.exit(1)
print("ALL CHANGED FILES: identical AST modulo docstrings -> comment/docstring only")
