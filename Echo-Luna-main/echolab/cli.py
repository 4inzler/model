
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from typing import List

from . import bench, metrics, persist, reflect, runner, stop
from .persist import PersistContext
from .regen import regenerate_snippet
from .state import Config, IterationResult, SentienceReport

BANNER = """================================================================
EchoLab sandbox | proxies of sentience-like behavior are observational proxies only
================================================================"""


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="echolab")
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="Run the EchoLab iterative loop")
    run_cmd.add_argument("--max-iterations", type=int, default=100)
    run_cmd.add_argument("--max-seconds", type=float, default=60.0)
    run_cmd.add_argument("--runner-timeout", type=float, default=10.0)
    run_cmd.add_argument(
        "--enable-real-write",
        action="store_true",
        help="Persist artifacts into .echo_lab/ (default is DRY-RUN)",
    )
    run_cmd.add_argument(
        "--no-benchmarks",
        action="store_true",
        help="Skip benchmark gating (for diagnostics)",
    )
    return parser.parse_args(argv)


def _format_proxy_table(report: SentienceReport) -> str:
    headers = ["Proxy", "Value"]
    rows = [[name, f"{getattr(report, name):.3f}"] for name in report.as_dict()]
    width1 = max(len(headers[0]), max(len(r[0]) for r in rows))
    width2 = max(len(headers[1]), max(len(r[1]) for r in rows))
    lines = [
        f"| {headers[0]:<{width1}} | {headers[1]:>{width2}} |",
        f"| {'-' * width1} | {'-' * width2} |",
    ]
    for name, value in rows:
        lines.append(f"| {name:<{width1}} | {value:>{width2}} |")
    lines.append(f"| mean_proxy{' ' * (width1 - 10)} | {report.mean():>{width2}.3f} |")
    return "\n".join(lines)


def _format_bench_table(scores: dict[str, float]) -> str:
    if not scores:
        return "No benchmarks executed."
    width = max(len(name) for name in scores)
    return "\n".join(f"- {name:<{width}} : {value:.3f}" for name, value in sorted(scores.items()))


def _log_to_dict(log: runner.RunLog) -> dict[str, object]:
    return {
        "command": log.command,
        "returncode": log.returncode,
        "stdout": log.stdout,
        "stderr": log.stderr,
        "duration": log.duration,
        "timeout": log.timeout,
    }


def _run_iteration(
    iteration: int,
    report: SentienceReport,
    config: Config,
    ctx: PersistContext,
    history: List[IterationResult],
    start_time: float,
) -> IterationResult:
    updated_report = metrics.update_proxies(report)
    snippet = regenerate_snippet(iteration, updated_report)

    evaluation_code = snippet + "\nprint(describe_iteration())"
    snippet_log = runner.run_python(evaluation_code, timeout=config.runner_timeout)

    scores, passed = bench.run_benchmarks(updated_report)
    if not config.require_benchmarks:
        passed = True

    stop_reason, goal = stop.should_stop(
        config=config,
        report=updated_report,
        passed=passed,
        iteration=iteration,
        start_time=start_time,
    )

    result = IterationResult(
        iteration=iteration,
        report=updated_report,
        bench_scores=scores,
        benchmarks_passed=passed,
        snippet=snippet,
        run_logs=[_log_to_dict(snippet_log)],
        stop_reason=stop_reason,
        goal_achieved=goal,
    )

    history.append(result)

    if config.enable_real_write:
        artifact_path = persist.write_artifact(ctx, snippet, iteration)
        if artifact_path is not None:
            result.run_logs.append({"artifact": str(artifact_path)})
        trace_path = persist.append_trace(ctx, result)
        if trace_path is not None:
            result.run_logs.append({"trace": str(trace_path)})

    return result


def run_loop(args: argparse.Namespace) -> int:
    print(BANNER)
    config = Config(
        max_iterations=args.max_iterations,
        max_seconds=args.max_seconds,
        require_benchmarks=not args.no_benchmarks,
        enable_real_write=args.enable_real_write,
        runner_timeout=args.runner_timeout,
    )

    random.seed(2025)
    report = SentienceReport()
    history: List[IterationResult] = []
    start_time = time.time()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ctx = persist.initialize(run_id=run_id, enable_real_write=config.enable_real_write)

    reflection = "Initial plan: stay within sandbox, gather baseline benchmarks."
    print(reflection)

    for iteration in range(1, config.max_iterations + 1):
        result = _run_iteration(iteration, report, config, ctx, history, start_time)
        report = result.report

        print(f"\n--- Iteration {iteration} ---")
        print(_format_proxy_table(report))
        print("Benchmarks:")
        print(_format_bench_table(result.bench_scores))
        print("Regenerated snippet:")
        print(result.snippet)
        summary_json = json.dumps(result.summary())
        print(summary_json)

        reflection = reflect.reflect(history, config.reflection_window)
        print(reflection)

        if result.stop_reason:
            reason = result.stop_reason
            print(f"Stop reason: {reason}")
            status = (
                "Goal condition placeholder reached."
                if result.goal_achieved
                else "Goal condition pending."
            )
            print(status)
            break
    else:
        print("Reached iteration cap without trigger; halting for safety.")

    persist.finalize(ctx)
    return 0


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(argv)
    if args.command == "run":
        return run_loop(args)
    raise SystemExit(1)


if __name__ == "__main__":
    sys.exit(main())
