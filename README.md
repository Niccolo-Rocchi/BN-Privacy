# Towards Privacy-Aware Bayesian Networks: A Credal Approach

Code for paper ["Towards Privacy-Aware Bayesian Networks: A Credal Approach"](https://doi.org/10.3233/FAIA251419) presented at [ECAI 2025](https://ecai2025.org/).

## Setting up Python environment

Create and activate a Python virtual environment with: 

```bash
python3 -m venv venv
source venv/bin/activate[.fish]  # use `.fish` suffix if using fish shell
```

Install dependencies with: 

```bash
pip install -r requirements.txt
```

Upgrade dependencies with: 

```bash
pip install --upgrade $(pip freeze | cut -d '=' -f 1)
pip freeze > requirements.txt
```

## Experiments

`<name>` is the name of the experiment to run. It can be one of the following.

1. `cn_privacy`: run membership inference attack against a Bayesian network (BN), its related credal network (CN), and compute the theoretical privacy estimate of BN. The pipeline and results are described in the paper.

2. `cn_vs_noisybn`: additional experiment, not reported in the paper. It compares two privacy techniques, namely the CN and a noisy version of BN. All models are naive Bayes with target variable T. First, the CN and noisy BN hyperparameters are fine-tuned so that they achieve the same privacy level; then, their accuracy is computed in terms of most probable explanation (MPE) on variable T.

### Running code

1. Run the experiment: 

```bash
python -m experiments.<name>.exp
```

*Notice:* this will delete any existing ground-truth model, data, and result.

2. Results available at: 

`experiments/<name>/output/`.

### Using Docker (recommended)

1. Build the Docker image:

```bash
docker build . -t bnp:2025
```

2. Run the Docker container:

```bash
docker run [-d] [--rm] -v bnp:/workspace bnp:2025 python -m experiments.<name>.exp
```

3. Results available at: 

`/var/lib/docker/volumes/bnp/_data/experiments/<name>/output/`.

## Testing code

Run integration tests with:

```bash
pytest [--cov=src] [--cov-report=term-missing] [--capture=no]
```

Results are available at: 

`test/<name>/output/`.

## Formatting and linting

Format code by running:

```bash
black .
isort .
```

Lint code by running:

```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=venv
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --exclude=venv
```

Analyze code by running:

```bash
pylint $(git ls-files '*.py')
```

## Plotting results

Use the `Plot_results.ipynb` notebook available for each experiment. Plots will be saved at: `experiments/<name>/output/plots`.
