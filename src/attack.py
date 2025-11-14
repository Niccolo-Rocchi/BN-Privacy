import inspect

import numpy as np
import pandas as pd
import pyagrum as gum

from src.config import get_out_path
from src.mia import get_ll
from src.utils import sample_from_cn


# Apply attack mechanism to a BN, namely, derive a BN from a CN
def attack_mechanism(atk_mec, exp, config) -> None:

    # Get output path
    out_path = get_out_path(config)

    # Read data
    gpop = pd.read_csv(f'{out_path / config["data_path"]}/{exp}.csv')
    base_path = out_path / config["cns_path"]

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
                "n_bns": config["n_bns"],
            }.items()
            if k in sig.parameters
        }
        bn = atk_mec_fn(**args)
        gum.saveBN(
            bn, f'{out_path / config["atk_path"]}/{f"bn_{exp}_sample{sample}"}.bif'
        )

    return


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
