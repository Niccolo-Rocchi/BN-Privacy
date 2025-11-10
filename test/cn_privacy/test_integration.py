from experiments.cn_privacy import data, exp


def test_integration():

    # Generate BNs and data
    data.main()

    # Run experiment
    exp.main()
