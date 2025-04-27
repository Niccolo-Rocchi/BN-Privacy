# How to use

## With Docker
1. Build image: `docker build . -t ecai2025`
2. Run with: `docker run -d --rm -v ecai:/workspace/results ecai2025 python main.py`
3. Results available at `/var/lib/docker/volumes/ecai/_data/*`

## Without Docker
1. Run `python config.py`
2. Run with `python main.py`
3. Results available at `./results/*`

## How to read the `config.yaml` file

```
Last update: %Y-%m-%d %H:%M

invar:                              # Invariant configs
  n_ds: 3                               # Number of data sample for each BN
  n_modmax: 2                           # Max number of modalities per node
  gpop_ss: 10000                        # Sample size (general population)
  rpop_ss: 5000                         # Sample size (reference population)
  pool_ss: 1000                         # Sample size (pool)
  error: np.arange(0, 1, 0.05)          # Error (alpha) range
  n_bns: 5                              # Number of BNs to extract from credal set  

var:                                # Variable configs
  idm:                                # Local IDM experiments
    - n_nodes: 5                        # Number of nodes (note: at 100 nodes, 0 power for gpop_ss = [1e4, 3e4].)
      edge_ratio: 1.2                   # Edges/nodes ratio
      ess: 1                            # Equivalent sample size (ESS) for local IDM
      meta: exp_1                       # Number of experiment

  cont:                               # Contamination experiments
    - n_nodes: 5
      edge_ratio: 1.2
      eps: 0.001
      meta: exp_1
```
