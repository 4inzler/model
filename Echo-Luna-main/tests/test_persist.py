import json
import shutil
from pathlib import Path

from echolab import persist
from echolab.state import IterationResult, SentienceReport


def test_persist_writes_confined_to_echo_lab():
    lab_root = Path(".echo_lab")
    shutil.rmtree(lab_root, ignore_errors=True)

    ctx = persist.initialize(run_id="test", enable_real_write=True)
    report = SentienceReport()
    result = IterationResult(
        iteration=1,
        report=report,
        bench_scores={"cognitive_synthesis": 0.8},
        benchmarks_passed=True,
        snippet="print('hello')",
        run_logs=[],
    )

    artifact_path = persist.write_artifact(ctx, result.snippet, result.iteration)
    trace_path = persist.append_trace(ctx, result)
    persist.finalize(ctx)

    assert artifact_path is not None
    assert trace_path is not None
    assert artifact_path.resolve().is_relative_to(lab_root.resolve())
    assert trace_path.resolve().is_relative_to(lab_root.resolve())

    manifest_data = json.loads((lab_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_data, "manifest should list at least one entry"
    for entry in manifest_data:
        assert entry.startswith(str(lab_root)), "manifest entry outside .echo_lab"
