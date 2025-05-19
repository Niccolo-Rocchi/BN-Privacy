# How to use (to be updated for this branch)

## Run code

### With Docker (recommended for reproducibility)
1. Run: `docker build . -t ecai2025` to build the Docker image,
2. Run: `docker run -d --rm -v ecai:/workspace/results ecai2025 python ./src/main.py`,
3. Results available at: `/var/lib/docker/volumes/ecai/_data/results/`.
4. Copy the `./results/` folder into the current directory to be able to plot the results.

### Without Docker
Create and activate a Python virtual environment, then install the packages listed in `./requirements.txt`. 

1. Run: `python ./src/generation.py`,
2. Run: `python ./src/main.py`,
3. Results available at: `./results/`.

## Plot results
Run the `Plot_results.ipynb` notebook. You should be able to choose which experiments to plot.
