from experiments.cn_privacy import exp, generate
import sys


def test_generation():

    # Generate models and data
    generate.main()


def test_def_ran_atk_mle(monkeypatch):

    monkeypatch.setattr(
        sys, "argv", ["def_mec=def_ran", "delta=0.3", "atk_mec=atk_mle", "n_bns=5"]
    )

    # Run experiment
    exp.main()


def test_def_idm_atk_mle(monkeypatch):

    monkeypatch.setattr(
        sys, "argv", ["def_mec=def_idm", "ess=1", "atk_mec=atk_mle", "n_bns=5"]
    )

    # Run experiment
    exp.main()
