# Towards Privacy-Aware Bayesian Networks: A Credal Approach

## Experiments

`<name>` is the name of experiment to run. It can be one of the following.

1. `cn_privacy`: run membership inference attack against a Bayesian network (BN), its related credal network (CN), and computes the theoretical privacy estimate of BN. The pipeline and results are described in the paper.
2. `cn_vs_noisybn`: additional experiment, not reported in the paper. It compares two privacy techniques, namely the CN and a noisy version of BN. All models are naive Bayes with target variable T. First, the CN and noisy BN hyperparameters are fine-tuned so that they achieve the same privacy level; then, their accuracy is computed in terms of most probable explaination (MPE) on variable T.

## How to 

### Run code

#### With Docker (recommended)
1. Run: `docker build . -t bnp:2025` to build the Docker image,
2. Run: `docker run -d --rm -v bnp:/workspace bnp:2025 python -m experiments.<name>.main`,
3. Results available at: `/var/lib/docker/volumes/bnp/_data/experiments/<name>/output/`.

#### Without Docker
Create and activate a Python virtual environment. Then, install the required packages: `pip install -r requirements.txt`. 

1. Run: `python -m experiments.<name>.main`,
2. Results available at: `experiments/<name>/output/`.

### Test
Tests can be run with: `pytest`. Results available at: `test/<name>/output/`.

### Plot results
Use the `Plot_results.ipynb` notebook, available for each experiment. Plots will be available at: `experiments/<name>/output/plots`.

### Upgrade all Python packages
Run: `pip install --upgrade $(pip freeze | cut -d '=' -f 1)` to upgrade the packages, then `pip freeze > requirements.txt` to update the requirements file.
