from echolab.runner import run_python


def test_runner_times_out():
    log = run_python("import time; time.sleep(0.5)", timeout=0.1)
    assert log.timeout is True
    assert log.returncode == -1
    assert "<timeout>" in log.stderr
