import inspect

import numpy as np
import pandas as pd
import pyagrum as gum

from src.config import get_cur_dir, safe_assert, set_seed
from src.utils import add_counts_to_bn, check_consistency


# Apply defense mechanism to a BN, namely, derive a CN from a BN
def defense_mechanism(exp, config, def_mec, def_args) -> None:

    # Get current directory
    cur_dir = get_cur_dir(config)

    # Read data
    gpop = pd.read_csv(f'{cur_dir / config["data_path"]}/{exp}.csv')

    # Set seed
    set_seed()

    # For each data sample ...
    for sample in range(config["samples"]):

        # ... read the related BN
        bn = gum.loadBN(
            f"{cur_dir}/{config['bns_path']}/pool/bn_{exp}_sample{sample}.bif"
        )

        # ... retrieve pool, ...
        pool = gpop[gpop[f"in-pool-{sample}"]].iloc[:, : len(bn.nodes())]

        # ... and derive the CN
        def_mec_fn = globals()[def_mec]  # Get the related function
        sig = inspect.signature(def_mec_fn)  # Get its signature
        args = {
            k: v
            for k, v in {
                "bn": bn,
                "ess": def_args.get("ess", None),
                "delta": def_args.get("delta", None),
                "data": pool,
            }.items()
            if k in sig.parameters
        }
        cn = def_mec_fn(**args)  # Keep only `def_mec` args

        # Save results
        base_path = cur_dir / config["cns_path"]
        cn.saveBNsMinMax(
            f"{base_path}/bn_min_{exp}_sample{sample}.bif",
            f"{base_path}/bn_max_{exp}_sample{sample}.bif",
        )

    return


# Estimate a CN from data by local IDM
def def_idm(bn, ess, data):
    bn_counts = gum.BayesNet(bn)
    add_counts_to_bn(bn_counts, data)
    cn = gum.CredalNet(bn_counts)
    cn.idmLearning(ess)

    return cn


# Build a CN by bloating each BN parameter with a fixed-size random interval
def def_ran(bn, delta):

    # Initialize the extreme BNs
    bn_min = gum.BayesNet(bn)
    bn_max = gum.BayesNet(bn)

    # For each node ...
    for n in bn.nodes():

        # ... get the CPT, ...
        cpt = bn.cpt(n).toarray()

        # ... get a matrix of eta's, ...
        eta = np.random.uniform(0, delta, cpt.size).reshape(cpt.shape)

        # ... perturb the CPT, ...
        cpt_min = np.minimum(1 - delta, np.maximum(0, cpt - eta))
        cpt_max = np.minimum(1, np.maximum(delta, cpt - eta + delta))

        # ... and store it into the extreme BNs
        bn_min.cpt(n).fillWith(cpt_min.flatten())
        bn_max.cpt(n).fillWith(cpt_max.flatten())

        # Debug
        safe_assert(np.all(cpt_min <= cpt))
        safe_assert(np.all(cpt_max >= cpt))
        safe_assert(np.all(np.abs(cpt_max - cpt_min - delta) < 1e-6))

    # Build the CN from the extreme BNs
    cn = gum.CredalNet(bn_min, bn_max)
    cn.intervalToCredal()

    # Debug
    safe_assert(check_consistency(bn, bn_min, bn_max))

    return cn


# Create noisy BN by adding Laplacian noise (Zhang et al., 2017)
def noisy_bn(bn, scale: float):

    bn_ie = gum.LazyPropagation(bn)
    bn_ie.makeInference()

    bn_noisy = gum.BayesNet(bn)

    # For each node X ...
    for node in bn.names():

        # Get the joint P(X, Pa(X))
        joint = bn_ie.jointPosterior(bn.family(node))

        # Add noise to P(X, Pa(X)) and normalize
        noise = np.random.laplace(scale=scale, size=np.prod(joint.shape))
        noisy_joint = np.clip(
            joint.toarray().flatten() + noise, a_min=10e-9, a_max=None
        )
        noisy_joint = noisy_joint / np.sum(noisy_joint)
        joint.fillWith(noisy_joint)

        # Compute the conditional P(X | Pa(X))
        cond = joint / joint.sumOut(node)

        # Fill noisy BN
        bn_noisy.cpt(node).fillWith(cond)

    # Check noisy bn
    bn_noisy.check()  # OK if = ().

    return bn_noisy
