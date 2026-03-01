from pathlib import Path

from marderlab_tools.stimulus.stimulusgen import StimulusSpec, generate_burst_train, write_stimulus_file


def test_generate_burst_train_and_write(tmp_path: Path) -> None:
    spec = StimulusSpec(duration_s=5.0, sample_rate_hz=1000.0, burst_count=3, inter_burst_s=1.0)
    payload = generate_burst_train(spec)
    assert payload["time_s"].size == 5000
    assert len(payload["burst_starts_s"]) == 3

    out = write_stimulus_file(tmp_path / "stimulus.csv", payload)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "time_s,stimulus_v" in text
