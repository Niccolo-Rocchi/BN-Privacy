# How to use

## With Docker
1. Build image: `docker build . -t ecai-exps`
2. Run container: `docker run -d --rm -v ecai:/workspace/results ecai-exps`
3. Results available at `/var/lib/docker/volumes/ecai/_data/`
