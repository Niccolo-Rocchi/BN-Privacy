# How to use (TODO: add info on branches)

## Run code

### With Docker (recommended for reproducibility)
1. Run: `docker build . -t ecai2025` to build the Docker image,
2. Run: `docker run -d --rm -v ecai:/workspace/results ecai2025 python src/main.py`,
3. Results available in: `/var/lib/docker/volumes/ecai/_data/results/`.
4. Copy the results folder into the current directory to be able to plot the results.

### Without Docker
Create and activate a Python virtual environment. Then, install the required packages: `pip install -r requirements.txt`. 

1. Run: `python src/generation.py`,
2. Run: `python src/main.py`,
3. Results available in: `results/`.

## Plot results
Use the `Plot_results.ipynb` notebook.

## How to update all Python packages
Run: `pip install --upgrade $(pip freeze | cut -d '=' -f 1)` to upgrade the packages, then `pip freeze > requirements.txt` to update the requirements file.
