import inspect

import numpy as np
import pandas as pd
import pyagrum as gum

from src.config import get_cur_dir, set_seed
from src.mia import get_ll
from src.utils import centroid_cn, sample_from_cn


# Apply attack mechanism to a BN, namely, derive a BN from a CN
def attack_mechanism(exp, config, atk_mec, atk_args) -> None:

    # Get current directory
    cur_dir = get_cur_dir(config)

    # Read data
    gpop = pd.read_csv(f'{cur_dir / config["data_path"]}/{exp}.csv')
    base_path = cur_dir / config["cns_path"]

    # Set seed
    set_seed()

    # For each data sample ...
    for sample in range(config["samples"]):

        # ... read the related CN
        bn_min = gum.loadBN(f"{base_path}/bn_min_{exp}_sample{sample}.bif")
        bn_max = gum.loadBN(f"{base_path}/bn_max_{exp}_sample{sample}.bif")

        # ... retrieve rpop, ...
        rpop = gpop[gpop[f"in-rpop-{sample}"]].iloc[:, : len(bn_min.nodes())]

        # ... and derive the BN
        atk_mec_fn = globals()[atk_mec]  # Get the related function
        sig = inspect.signature(atk_mec_fn)  # Get its signature
        args = {
            k: v
            for k, v in {
                "bn_min": bn_min,
                "bn_max": bn_max,
                "data": rpop,
                "n_bns": atk_args.get("n_bns", None),
            }.items()
            if k in sig.parameters
        }
        bn = atk_mec_fn(**args)
        gum.saveBN(
            bn, f'{cur_dir / config["atk_path"]}/{f"bn_{exp}_sample{sample}"}.bif'
        )

    return


# Get a random BN inside a CN
def atk_ran(bn_min, bn_max):

    bn = sample_from_cn(bn_min, bn_max, 1)

    return bn[0]


# Get the centroid of a CN
def atk_cen(bn_min, bn_max):

    bn = centroid_cn(bn_min, bn_max)

    return bn


# Get the maximum likelihood BN inside a CN
def atk_mle(bn_min, bn_max, data, n_bns: int):

    # Sample from the CN ...
    bns_sample = sample_from_cn(bn_min, bn_max, n_bns)

    # ... and take the MLE one
    bn = mle_bn(bns_sample, data)

    return bn


# Get the maximum likelihood BN within a set
def mle_bn(bns_sample, data):
    """
    Given a list `bns_sample` of BNs,
    find argmax_{BN in bns_sample} ll(BN | data),
    where ll is the log-likelihood function.
    """

    mle_bn = None
    mle = -np.inf

    for bn in bns_sample:

        # Estimate the likelihood of data
        bn_ie = gum.LazyPropagation(bn)
        llr_im = data.apply(lambda x: get_ll(x.to_dict(), bn_ie), axis=1).dropna()
        llr = np.sum(llr_im)

        if llr > mle:
            mle_bn = bn
            mle = llr

    return mle_bn
