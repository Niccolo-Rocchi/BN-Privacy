from itertools import product
from datetime import datetime

def generate_config(file):

    # Define invar. hyperparameters
    invar = '''invar:
  n_ds: 10
  n_modmax: 2
  gpop_ss: 10000
  rpop_ss: 5000
  pool_ss: 1000
  error: np.arange(0, 1, 0.05)
  n_bns: 20
    '''

    t = "  "

    # Define var. hyperparamters
    n_nodes = [5, 10, 20, 50, 70, 100]
    edge_ratios = [1, 1.2, 1.5, 2]
    ess = [1, 2]
    eps = [1e-3, 1e-2, 5e-2, 1e-1]

    # Print on file
    print("# Last update: " + datetime.now().strftime("%Y-%m-%d %H:%M"), file=file)
    print(invar, file=file)

    print("var:", file=file)

    print(f"{t}idm:", file=file)
    idx = 1
    for n, r in product(n_nodes, edge_ratios):
        for s in ess:
            print(f"{t*2}-", end=" ", file=file)
            print(f"n_nodes: {n}\n{t*3}edge_ratio: {r}\n{t*3}ess: {s}\n{t*3}meta: exp_{idx}\n", file=file)
            idx += 1

    print("  cont:", file=file)
    idx = 1
    for n, r in product(n_nodes, edge_ratios):
        for e in eps:
            print(f"{t*2}-", end=" ", file=file)
            print(f"n_nodes: {n}\n{t*3}edge_ratio: {r}\n{t*3}eps: {e}\n{t*3}meta: exp_{idx}\n", file=file)
            idx += 1

if __name__ == "__main__":

    with open("config.yaml", "w") as f:
        generate_config(f)