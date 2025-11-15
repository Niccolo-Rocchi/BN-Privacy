# Towards Privacy-Aware Bayesian Networks: A Credal Approach

Code for paper ["Towards Privacy-Aware Bayesian Networks: A Credal Approach"](https://doi.org/10.3233/FAIA251419) presented at [ECAI 2025](https://ecai2025.org/).

## Setting up Python environment

Create and activate a Python virtual environment: 

```bash
python3 -m venv venv
source venv/bin/activate[.fish]  # use `.fish` suffix if using fish shell
```

Install dependencies: 

```bash
pip install -r requirements.txt
```

Upgrade dependencies: 

```bash
pip install --upgrade $(pip freeze | cut -d '=' -f 1)
pip freeze > requirements.txt
```
## Preliminaries

### Experiments

`<name>` is the name of the experiment to run. Each `<name>` has its own directory, which is named the same way. Each of these contains the experiment logic, configuration file (`config.yaml`), output (which path specified in configurations), and a `Plot_results.ipynb` notebook to plot results.

`<name>` can be one of the following:

1. `cn_privacy`: run membership inference attack against a Bayesian network (BN), its related credal network (CN), and compute the theoretical privacy estimate of BN. The pipeline and results are described in the paper.

2. `cn_vs_noisybn`: additional experiment, not reported in the paper. It compares two privacy techniques, namely the CN and a noisy version of BN. All models are naive Bayes with target variable T. First, the CN and noisy BN hyperparameters are fine-tuned so that they achieve the same privacy level; then, their accuracy is computed in terms of most probable explanation (MPE) on variable T.

### Attacks and defenses

Each experiment requires the user to specify one defense and one attack mechanisms, plus additional related hyperparameters. Below, the mechanisms and hyperparameters names are reported. Further details are provided in the paper.

Implemented defenses:
- `def_idm`. Requires: `ess`
- `def_ran`. Requires: `delta`

Implemented attacks:
- `atk_mle`. Requires: `n_bns`

## Running code

### Local computation

*Notice:* each of the following command will overwrite any related output. 

1) Generate models and data:

```bash
python -m experiments.<name>.generate
```

2) Run an experiment: 

```bash
python -m experiments.<name>.exp def_mec=<def_mec_name> [def_params] atk_mec=<atk_mec_name> [atk_params]
```

### Using Docker (recommended)

1. Build the Docker image:

```bash
docker build . -t bnp:2025
```

2. Run the Docker container:

```bash
docker run [-d] [--rm] -v bnp:/workspace bnp:2025 <command>
```

where `<command>` follows the same syntax as in the local computation.

3. Results available at: 

`/var/lib/docker/volumes/bnp/_data/`.

## Testing code

Run integration tests:

```bash
pytest [--cov=src] [--cov-report=term-missing] [--capture=no]
```

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
