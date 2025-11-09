from echolab.metrics import update_proxies
from echolab.state import SentienceReport


def test_update_proxies_within_bounds():
    report = SentienceReport()
    for _ in range(5):
        report = update_proxies(report)
        for value in report.as_dict().values():
            assert 0.0 <= value <= 1.0
