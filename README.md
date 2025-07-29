# How to use

## Run code

`<name>` is the name of experiment to run. Two alternatives are: `cn_privacy` and `cn_vs_noisybn`.

### With Docker (recommended)
1. Run: `docker build . -t bnp:2025` to build the Docker image,
2. Run: `docker run -d --rm -v bnp:/workspace bnp:2025 python -m experiments.<name>.main`,
3. Results available at: `/var/lib/docker/volumes/bnp/_data/experiments/<name>/output/`.

### Without Docker
Create and activate a Python virtual environment. Then, install the required packages: `pip install -r requirements.txt`. 

1. Run: `python -m experiments.<name>.main`,
2. Results available at: `experiments/<name>/output/`.

## Test
Tests can be run with: `pytest`. Results available at: `test/<name>/output/`.

## Plot results
Use the `Plot_results.ipynb` notebook.

## Upgrade all Python packages
Run: `pip install --upgrade $(pip freeze | cut -d '=' -f 1)` to upgrade the packages, then `pip freeze > requirements.txt` to update the requirements file.
