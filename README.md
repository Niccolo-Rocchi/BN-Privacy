# How to use (TODO: add info on branches)

## Run code

### With Docker (recommended)
1. Run: `docker build . -t bnp:2025` to build the Docker image,
2. Run: `docker run -d --rm -v bnp:/workspace/results bnp:2025 python -m scripts.run_experiment`,
3. Results available in: `/var/lib/docker/volumes/bnp/_data/`.

### Without Docker
Create and activate a Python virtual environment. Then, install the required packages: `pip install -r requirements.txt`. 

1. Run: `python -m scripts.generate_data`,
2. Run: `python -m scripts.run_experiment`,
3. Results available in: `results/`.

## Plot results
Use the `Plot_results.ipynb` notebook.

## How to upgrade all Python packages
Run: `pip install --upgrade $(pip freeze | cut -d '=' -f 1)` to upgrade the packages, then `pip freeze > requirements.txt` to update the requirements file.
