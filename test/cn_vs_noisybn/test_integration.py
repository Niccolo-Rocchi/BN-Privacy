from experiments.cn_vs_noisybn import generate, exp


def test_integration():

    # Generate models and data
    generate.main()

    # Run experiment
    exp.main()
