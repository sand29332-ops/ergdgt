# Repository import

On 2026-09-14, the owner requested a complete copy of
[`gamerapex82-cloud/refree`](https://github.com/gamerapex82-cloud/refree) into the
previously empty connected repository, `sand29332-ops/ergdgt`.

- Source branch: `hoplite/kranioi-5b44d8a8`.
- Source commit: `8cb22ec0e1cef56c42cf38f610304df2cb881ef8`.
- The initial import preserves all 98 tracked files and all 69 reachable commits,
  including their original authorship and commit IDs.
- The connected `origin` remains `sand29332-ops/ergdgt`; the source repository was
  not modified. This is a source/history copy, not a GitHub fork relationship.
- No market tapes, local credentials, or ignored build artifacts were imported.
  The existing proprietary project declaration is unchanged.

References to the original repository's `main`, pull-request numbers, Windows
checkout, and historical test runs in the handoff documents describe that earlier
project. New work and verification in this copy are recorded separately.

## Reproduce locally

The Python runtime requires Python 3.10+; CI uses Python 3.12. A C++20 compiler is
needed for the engine/parity tests, but the seeded dashboard also works without
the compiled module.

On Linux, `bash .hoplite/setup.sh` installs the declared Python/build dependencies
into `.venv` and builds the portable CPU engine. It also installs missing compiler
or Python development packages when running as root on an apt-based sandbox;
other machines must provide those prerequisites. Re-running it is safe. Then:

```bash
.venv/bin/python -m pytest python_quant/tests bindings/tests
.venv/bin/ctest --test-dir build --output-on-failure
.venv/bin/ruff check python_quant/nexus_quant/ bindings/
bash .hoplite/run.sh
```

The managed preview uses port 3000 by default (or `$PORT`). It is a local,
seeded simulation with no authentication, database, market connectivity, or
production credentials. CUDA compilation and hardware performance measurement
remain separate Person A work; this setup makes no GPU or latency claim.

Import baseline verified on Linux, 2026-09-14: **159 pytest passes, one optional
real-tape skip; CTest 5/5; lint clean**. The compiled Python extension and all
eight binding tests were exercised. These are fresh results in this copy, not
the historical upstream counts. Windows verification is delegated to CI.
