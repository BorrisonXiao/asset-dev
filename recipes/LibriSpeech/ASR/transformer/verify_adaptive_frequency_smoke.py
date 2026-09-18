"""Fail the canary job unless real updates, paired validation and resume worked."""

import json
from pathlib import Path
import sys
import yaml

root = Path(sys.argv[1])
events = [json.loads(line) for line in (root / "frequency_events.jsonl").read_text().splitlines()]
assert [event["step"] for event in events] == [0, 2, 4], events
for event in events:
    assert event["lr"] == [5e-6, 2e-5]
    assert event["training_batch_step"] == event["step"]
    assert event["source_optimizer_step"] == 24000
    assert set(event["splits"]) == {"dev-clean", "dev-other"}
    assert all(stats["utterances"] == 16 for stats in event["splits"].values())
    checkpoint = root / "save" / f"CKPT+continuation-step-{event['step']:06d}"
    brain = yaml.safe_load((checkpoint / "brain.ckpt").read_text())
    assert brain["step"] == event["step"], brain
    assert brain["optimizer_step"] == event["step"], brain
    assert (checkpoint / "random_state.ckpt").stat().st_size > 0
    state = json.loads((checkpoint / "frequency.ckpt").read_text())["state"]
    assert state["last_step"] == event["step"]
selection = json.loads((root / "selected_checkpoint.json").read_text())
assert selection["completed"] and selection["completed_new_steps"] == 4
assert not selection["test_sets_evaluated"]
assert Path(selection["checkpoint"]).is_dir()
print("ADAPTIVE_FREQUENCY_SMOKE_PASS: matched state, four real updates, paired validation, checkpoint/resume, max-step stop")
