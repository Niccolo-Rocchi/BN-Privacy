# Towards Privacy-Aware Bayesian Networks: A Credal Approach

## Set up Python environment

Create and activate a Python virtual environment with: 

```bash
python3 -m venv venv
source venv/bin/activate[.fish]  # use `.fish` suffix if using fish shell
```

Install all dependencies with: 

```bash
pip install -r requirements.txt
```

Upgrade all Python packages with: 

```bash
pip install --upgrade $(pip freeze | cut -d '=' -f 1)
pip freeze > requirements.txt
```

This updates the requirements file with the upgraded packages.

## Experiments

`<name>` is the name of experiment to run. It can be one of the following.

1. `cn_privacy`: run membership inference attack against a Bayesian network (BN), its related credal network (CN), and computes the theoretical privacy estimate of BN. The pipeline and results are described in the paper.

2. `cn_vs_noisybn`: additional experiment, not reported in the paper. It compares two privacy techniques, namely the CN and a noisy version of BN. All models are naive Bayes with target variable T. First, the CN and noisy BN hyperparameters are fine-tuned so that they achieve the same privacy level; then, their accuracy is computed in terms of most probable explanation (MPE) on variable T.

## Run code

### With Docker (recommended)

1. Build the Docker image:

```bash
docker build . -t bnp:2025
```

2. Run the experiment:

```bash
docker run [-d] [--rm] -v bnp:/workspace bnp:2025 python -m experiments.<name>.main
```

3. Results available at: 

`/var/lib/docker/volumes/bnp/_data/experiments/<name>/output/`.

### Without Docker

1. Run the experiment: 

```bash
python -m experiments.<name>.main
```

2. Results available at: 

`experiments/<name>/output/`.

## Test code

Run tests with:

```bash
pytest
```

Test results are available at: 

`test/<name>/output/`.

## Formatting

Format code by running:

```bash
black .
isort .
```

## Plot results

Use the `Plot_results.ipynb` notebook available for each experiment. Plots will be saved at: `experiments/<name>/output/plots`.
