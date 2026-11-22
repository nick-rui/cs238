"""Check source means, feedback-free replay, and numerical schedule equivalence."""

import argparse
import json
import numpy as np
import run as r


class ForbiddenPolicy:
    def __call__(self, *_):
        raise AssertionError("Open-loop replay must never call the policy")


def check(out):
    r.torch.set_num_threads(1)
    test = r.e.MNIST(str(r.e.DATA_DIR), train=False, download=False)
    testset = (test.data.reshape(-1, 784).float() / 255, test.targets)
    checks = []
    for family in r.FAMILIES:
        traces = r.readcsv(r.trace_path(family))
        original = r.readcsv(r.SOURCE / f"{family}_evaluation.csv")
        for ps in r.POLICY_SEEDS:
            source = [t for t in traces if t["variant"] == "gradient" and int(t["policy_seed"]) == ps]
            schedule = r.mean_schedule(family, ps, traces)
            for row in schedule:
                group = [t for t in source if int(t["step"]) == row["step"]]
                for key in r.columns(family):
                    assert np.isclose(row[key], np.mean([float(t[key]) for t in group]), rtol=1e-12)
            for broken in [source[:-1], source + source[:1]]:
                try:
                    r.mean_schedule(family, ps, broken)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Missing/duplicated source trace was accepted")

            # Replaying an individual episode's exact historical settings must
            # reproduce its closed-loop metrics, including time-varying moments.
            seed, order = r.b.TESTSEEDS[0], "single_middle"
            exact = sorted([t for t in source if int(t["seed"]) == seed and t["order"] == order],
                           key=lambda t: int(t["step"]))
            exact = [{key: float(value) for key, value in t.items() if key in r.columns(family) + ["step"]}
                     for t in exact]
            rr, tr, *_ = r.replay(family, exact, [seed], [order], testset, pol=ForbiddenPolicy())
            ref = next(t for t in original if t["variant"] == "gradient" and int(t["policy_seed"]) == ps
                       and int(t["seed"]) == seed and t["order"] == order)
            errors = {key: abs(float(rr[0][key]) - float(ref[key])) for key in r.METRICS}
            assert errors["accuracy"] < 1e-6 and max(errors.values()) < 3e-5, (family, ps, errors)
            assert str(rr[0]["failed"]) == ref["failed"]
            r.verify_replay(family, exact, tr)

            # Different initializations/orderings/control losses still receive
            # identical settings, and cannot access even a callable policy.
            _, tr, logs, *_ = r.replay(family, schedule, [1900000, 1900001],
                                     ["mixed", "single_late"], pol=ForbiddenPolicy())
            assert logs is None
            r.verify_replay(family, schedule, tr)

            folder = out / f"{family}_gradient_{ps}"
            stored = r.readcsv(folder / "schedule.csv")
            assert len(stored) == 40
            for actual, expected in zip(stored, schedule):
                assert all(float(actual[key]) == expected[key] for key in expected)
            replay_traces = r.readcsv(folder / "traces.csv.gz")
            keys = [(int(t["seed"]), t["order"], int(t["step"])) for t in replay_traces]
            assert len(keys) == 8000 and len(set(keys)) == 8000
            assert set(keys) == {(s, o, t) for s in r.b.TESTSEEDS for o in r.b.ORDERS for t in r.STEPS}
            r.verify_replay(family, schedule, replay_traces)
            provenance = json.loads((folder / "provenance.json").read_text())
            assert all(r.sha256(r.ROOT / path) == digest
                       for path, digest in provenance["source_sha256"].items())
            checks.append(dict(family=family, policy_seed=ps, schedule_source_episodes=200,
                               verified_replay_trace_rows=len(replay_traces),
                               exact_historical_replay_errors=errors, no_policy_calls=True))

    paired = r.readcsv(out / "paired_differences.csv")
    assert len(paired) == 1600
    result = dict(checks=checks, total_episodes=1600, total_verified_trace_rows=64000)
    (out / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=r.Path, default=r.DEFAULT_OUT)
    check(parser.parse_args().output)
