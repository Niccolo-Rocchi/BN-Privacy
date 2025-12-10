# Towards Privacy-Aware Bayesian Networks: A Credal Approach

Code for paper ["Towards Privacy-Aware Bayesian Networks: A Credal Approach"](https://doi.org/10.3233/FAIA251419) presented at [ECAI 2025](https://ecai2025.org/).

## Preliminaries

### Experiments

`<name>` is the name of the experiment to run. Each `<name>` has its own directory, which is named the same way. Each of these contains the experiment logic, configuration file (`config.yaml`), eventually generated models and data, output directory, and a `Plot_results.ipynb` notebook to plot results.

`<name>` can be one of the following:

1. `cn_privacy`: run membership inference attack against a Bayesian network (BN), its related credal network (CN), and compute the theoretical privacy estimate of BN.

2. `cn_vs_noisybn`: compare two privacy techniques, namely the CN and a noisy version of BN. All models are naive Bayes with target variable T. First, the CN and noisy BN hyperparameters are fine-tuned so that they achieve the same privacy level; then, their accuracy is computed in terms of most probable explanation (MPE) on variable T.

For additional details, we refer to the paper.

### Attacks and defenses

Each experiment requires the user to specify one defense and one attack mechanisms, plus additional related hyperparameters. Below, the mechanisms and hyperparameters names are reported.

Implemented defenses:
- `def_idm`. Requires: `ess`.
- `def_ran`. Requires: `delta`.

Implemented attacks:
- `atk_mle`.
- `atk_cen`.
- `atk_ran`.
- `atk_ent`.

## Running code

### Using Docker (recommended)

The `compose.yaml` file contains a set of pre-set experiments. Additional ones can also be specified. The `generate_compose.py` file helps in generating them automatically.

Generate models and data for all experiments (controlled by `config.yaml`):

```sh
python -m experiments.cn_privacy.generate
python -m experiments.cn_vs_noisybn.generate
```

Run one or more experiments with:

```sh
docker compose up [service name]
```

Results will be available under `experiments/<name>/output_*`.

To check the status, run one or more of the following:

```sh
docker compose ps
docker compose logs [service name]
docker stats
```

### Local computation

Create and activate a Python virtual environment: 

```sh
python3 -m venv venv
source venv/bin/activate[.fish]  # use `.fish` suffix if using fish shell
```

Install dependencies: 

```sh
pip install -r requirements.txt
```

*Notice*: if some package is missing locally, see the `Dockerfile` for additional packages to be installed (names refer to Ubuntu/Debian).

Upgrade dependencies: 

```sh
pip install --upgrade $(pip freeze | cut -d '=' -f 1)
pip freeze > requirements.txt
```

*Notice:* each of the following command will overwrite any related output. 

Generate models and data (controlled by `config.yaml`):

```sh
python -m experiments.<name>.generate
```

Run an experiment: 

```sh
python -m experiments.<name>.exp def_mec=<def_name> [param=value] atk_mec=<def_name> [param=value]
```

Results will be available under `experiments/<name>/output`.


## Testing code

Run integration tests:

```sh
pytest [--cov=src] [--cov-report=term-missing] [--capture=no]
```

## Formatting and linting

Format code by running:

```sh
black .
isort .
```

Lint code by running:

```sh
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=venv
flake8 . --count --exit-zero --max-complexity=10 --ignore=E203 --max-line-length=140 --statistics --exclude=venv
```

Analyze code by running:

```sh
pylint $(git ls-files '*.py')
```

## Running actions locally

Install `act`:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
```

Run `act` with: 

```sh
sudo ./bin/act [-W <path_to_file>]
```
