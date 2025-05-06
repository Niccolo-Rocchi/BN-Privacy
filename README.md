# How to use

## With Docker
1. Build image: `docker build . -t ecai2025`
2. Run with: `docker run -d --rm -v ecai:/workspace/results ecai2025 python main.py`
3. Results available at `/var/lib/docker/volumes/ecai/_data/results/`

## Without Docker
Create and activate a Python virtual environment, then install the packages listed in `requirements.txt`. 
1. Run `python generate.py`
2. Run with `python main.py`
3. Results available at `./results/`
