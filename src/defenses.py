import numpy as np
import pyagrum as gum

from src.utils import add_counts_to_bn, check_consistency, safe_assert


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
        cpt_min = np.minimum(1-delta, np.maximum(0, cpt-eta))
        cpt_max = np.minimum(1, np.maximum(delta, cpt - eta + delta))

        # ... and store it into the extreme BNs
        bn_min.cpt(n).fillWith(cpt_min.flatten())
        bn_max.cpt(n).fillWith(cpt_max.flatten())

        # Debug
        safe_assert(np.all(cpt_min <= cpt))
        safe_assert(np.all(cpt_max >= cpt))
        safe_assert(np.all(np.abs(cpt_max-cpt_min-delta)<1e-6))

    # Build the CN from the extreme BNs
    cn = gum.CredalNet(bn_min, bn_max)
    cn.intervalToCredal()
    
    # Debug
    safe_assert(check_consistency(bn, bn_min, bn_max))

    return cn
