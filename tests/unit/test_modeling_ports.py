from marderlab_tools.modeling.hiksim import HiKSimParams, run_hiksim
from marderlab_tools.modeling.modelfiber import FiberParams, run_modelfiber
from marderlab_tools.modeling.musclemodelrealistic_vm import MuscleVMParams, run_musclemodelrealistic_vm
from marderlab_tools.modeling.untitled_model import UntitledParams, run_untitled_model


def test_hiksim_runs() -> None:
    out = run_hiksim(HiKSimParams(duration_s=5.0, dt_s=0.01))
    assert out["time_s"].size > 10
    assert "summary" in out


def test_modelfiber_runs() -> None:
    out = run_modelfiber(FiberParams(duration_s=2.0, dt_s=0.001))
    assert out["ejp"].size > 10
    assert out["summary"]["n_spikes"] >= 1


def test_musclemodel_runs() -> None:
    out = run_musclemodelrealistic_vm(MuscleVMParams(duration_s=2.0, dt_s=0.001, temperatures_c=(12.0,)))
    assert out["time_s"].size > 10
    assert "12.0C" in out["traces"]


def test_untitled_model_runs() -> None:
    out = run_untitled_model(UntitledParams(duration_s=1.0, dt_s=0.001))
    assert out["voltage_v"].size > 10
    assert "q10" in out["summary"]
