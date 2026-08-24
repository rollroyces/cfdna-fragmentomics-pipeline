"""Smoke test: lr_regularization_sweep's print statements reference
variables in scope. Catches the n_nonzero bug that crashed L1
sweeps on commit 7bbdef5.

Static check (not running the actual L1 sweep — which takes
minutes with saga solver on 60k features).
"""
import os
import ast


def test_lr_sweep_no_bare_n_nonzero_in_fstring():
    """The L1 print statement previously crashed with NameError
    because it referenced the bare variable `n_nonzero`, which was
    out of scope. The fix renamed the in-scope counter to
    `n_nonzero_seed` and used `r.get('n_nonzero', '?')` in the print.

    This test parses the source and checks that no f-string in
    lr_regularization_sweep.py contains a bare `{n_nonzero}` token
    (which would NameError at runtime).
    """
    repo = os.path.join(os.path.dirname(__file__), "..")
    src_path = os.path.join(repo, "scripts", "lr_regularization_sweep.py")
    with open(src_path) as f:
        source = f.read()

    # Static guard: source must not contain '{n_nonzero}' (without
    # the _seed suffix). That token is the regression.
    bare_count = source.count("{n_nonzero}")
    assert bare_count == 0, (
        f"Bare `{{n_nonzero}}` reference found in {src_path} "
        f"({bare_count} occurrences). This is the NameError bug. "
        f"Use `{{n_nonzero_seed}}` or `{{r.get('n_nonzero', '?')}}` instead.")

    # The internal counter `n_nonzero_seed` must still exist (it's the
    # in-scope replacement for the old bare `n_nonzero`).
    assert "n_nonzero_seed" in source, (
        "Internal counter `n_nonzero_seed` missing — regression")

    # And the script must parse without syntax errors.
    ast.parse(source)
