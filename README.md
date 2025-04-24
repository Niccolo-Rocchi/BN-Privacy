# How to use

## With Docker
1. Build image: `docker build . -t ecai2025`
2. Choose the script `<script_name>.py` to execute
2. Run container: `docker run [-d] --rm -v ecai:/workspace/results ecai2025 python <script_name>.py`
3. Results available at `/var/lib/docker/volumes/ecai/_data/*`
