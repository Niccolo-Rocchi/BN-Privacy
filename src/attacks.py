import numpy as np
import pyagrum as gum

from src.utils import get_ll, sample_from_cn


# Get the maximum likelihood BN inside a CN
def atk_mle(cn, data, exp, config):

    # Sample from the CN ...
    n_bns = config["n_bns"]
    bns_sample = sample_from_cn(cn, exp, n_bns)

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
