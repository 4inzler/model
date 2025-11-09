
from __future__ import annotations

import random
import time
from hashlib import sha1

from autogen_sim.config import SimulationConfig
from autogen_sim.display import (
    STOP_REASON_DESCRIPTIONS,
    format_proxy_table,
    iteration_summary_json,
)
from autogen_sim.engine import (
    achieve_sentience_placeholder,
    check_stop_conditions,
    generate_proxy_deltas,
    simulate_benchmarks,
)
from autogen_sim.regeneration import regenerate_code_snippet, rewrite_all_files
from autogen_sim.report import SentienceProxyReport

BANNER = "=" * 64


def default_esc_signal(iteration: int, elapsed: float) -> bool:
    _ = (iteration, elapsed)
    return False


def digest_snippet(snippet: str) -> str:
    return sha1(snippet.encode("utf-8")).hexdigest()[:12]


def run() -> None:
    print(BANNER)
    print("Safe sandbox simulation | DRY-RUN active by default | proxies of sentience-like behavior ≠ proof of sentience")
    print(BANNER)

    random.seed(2025)

    config = SimulationConfig(esc_signal_fn=default_esc_signal)
    report = SentienceProxyReport.initial()
    start_time = time.time()

    iteration = 0

    while True:
        iteration += 1
        deltas = generate_proxy_deltas()
        report = report.update(deltas)

        benchmarks_passed = (
            simulate_benchmarks(report) if config.require_benchmarks else True
        )

        snippet = regenerate_code_snippet(iteration, report)
        rewrite_message = rewrite_all_files(snippet, config.enable_real_writes)

        stop_reason = check_stop_conditions(
            config=config,
            report=report,
            benchmarks_passed=benchmarks_passed,
            iteration=iteration,
            start_time=start_time,
        )
        goal_achieved = achieve_sentience_placeholder(report, benchmarks_passed)

        print(f"\n--- Iteration {iteration} ---")
        print(format_proxy_table(report))
        print(f"Benchmarks passed: {benchmarks_passed}")
        print("Regenerated snippet:")
        print(snippet)
        print(rewrite_message)
        summary_json = iteration_summary_json(
            iteration=iteration,
            report=report,
            benchmarks_passed=benchmarks_passed,
            stop_reason=stop_reason,
            snippet_digest=digest_snippet(snippet),
        )
        print(summary_json)

        if stop_reason:
            description = STOP_REASON_DESCRIPTIONS.get(stop_reason, stop_reason)
            print(description)
            final_line = (
                "Goal condition placeholder reached." if goal_achieved else "Goal condition pending."
            )
            print(final_line)
            break

        if iteration >= config.max_iterations:
            print("Safety: reached configured iteration limit without stop signal.")
            break

        time.sleep(0.25)


if __name__ == "__main__":
    run()
