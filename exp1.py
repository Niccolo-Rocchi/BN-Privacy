## Libraries
import numpy as np
import pandas as pd
import math
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm
import pyagrum as gum
from pathlib import Path

## Log-likelihood ratio function
def LLR(x: dict, bn_theta_ie, bn_theta_hat_ie):

    # Erase all evidences and apply addEvidence(key,value) for every pairs in x
    bn_theta_ie.setEvidence(x)
    assert(bn_theta_ie.nbrHardEvidence() == len(bn.nodes()))
    bn_theta_hat_ie.setEvidence(x)
    assert(bn_theta_hat_ie.nbrHardEvidence() == len(bn.nodes()))

    # Compute P(x | BN)
    L_theta = bn_theta_ie.evidenceProbability()
    L_theta_hat = bn_theta_hat_ie.evidenceProbability()

    # Check for denominator
    if L_theta_hat < 1e-16:
        return None
    
    return math.log(L_theta / L_theta_hat)

## LLR test
def reject_H0(x: dict, bn_theta_ie, bn_theta_hat_ie, t: float) -> bool:

    # Compute the value of LLR(x)
    llr = LLR(x, bn_theta_ie, bn_theta_hat_ie)

    # If denominator is 0, then don't reject H0
    if llr is None:
        return False
    
    return llr < t


### MAIN

if __name__ == "__main__":

    ## Create results directory
    results_path = Path("./results")
    results_path.mkdir(parents=True, exist_ok=True)

    ## Generate ground-truth BN
    n_nodes = 3     # (must be n_nodes > 2)
    bn_gen = gum.BNGenerator()
    bn = bn_gen.generate(n_nodes=n_nodes, n_arcs=3, n_modmax=3) # 

    ## Generate data
    gpop_ss = 500  #
    ratio = 6   #
    pool_ss = gpop_ss // ratio
    rpop_ss = gpop_ss - pool_ss

    assert(gpop_ss == pool_ss + rpop_ss)

    data_gen = gum.BNDatabaseGenerator(bn)
    data_gen.drawSamples(gpop_ss)
    data_gen.setDiscretizedLabelModeRandom()

    gpop = data_gen.to_pandas()
    pool_idx = np.random.choice(gpop_ss, replace=False, size=pool_ss)
    pool = gpop.iloc[pool_idx]
    rpop = gpop.iloc[~ gpop.index.isin(pool_idx)]

    assert(gpop.shape[0]==gpop_ss)
    assert(pool.shape[0]==pool_ss)
    assert(rpop.shape[0]==rpop_ss)

    ## Estimate BN(theta) from rpop and BN(theta_hat) from pool
    theta_learner=gum.BNLearner(rpop)
    theta_learner.useSmoothingPrior(1e-5)
    theta_hat_learner=gum.BNLearner(pool)
    theta_hat_learner.useSmoothingPrior(1e-5)

    bn_theta = theta_learner.learnParameters(bn.dag())
    bn_theta_hat = theta_hat_learner.learnParameters(bn.dag())

    ## Estimate BN(theta_min) and BN(theta_max) from a CN by local IDM
    # Add counts of events to BN
    pop = pool
    for node in bn.names():
        var = bn.variable(node)
        parents = bn.parents(node)
        parent_names = [bn.variable(p).name() for p in parents]

        shape = [bn.variable(p).domainSize() for p in parents] + [var.domainSize()]
        counts_array = np.zeros(shape, dtype=float)  # float, not int!

        for _, row in pop.iterrows():
            try:
                key = tuple([int(row[p]) for p in parent_names] + [int(row[node])])
                counts_array[key] += 1.0
            except KeyError:
                continue

        bn.cpt(node).fillWith(counts_array.flatten().tolist())
    
    # Learn the CN through local IDM
    ess = 1   #
    cn = gum.CredalNet(bn)
    cn.idmLearning(ess)

    # Extract min-max BNs
    cn.saveBNsMinMax("./bn_min.bif", "./bn_max.bif")
    bn_min = gum.loadBN("./bn_min.bif")
    bn_max = gum.loadBN("./bn_max.bif")

    ## Compute the LLR ECDFs under H_0, i.e. x not in pool
    # Create objects for inference
    bn_theta_ie = gum.LazyPropagation(bn_theta)
    bn_theta_hat_ie = gum.LazyPropagation(bn_theta_hat)
    bn_min_ie = gum.LazyPropagation(bn_min)
    bn_max_ie = gum.LazyPropagation(bn_max)

    # Estimate the distribution of LLR(x) from rpop
    L_im = rpop.apply(lambda x: LLR(x.to_dict(), bn_theta_ie, bn_theta_hat_ie), axis=1)
    L_im_min = rpop.apply(lambda x: LLR(x.to_dict(), bn_theta_ie, bn_max_ie), axis=1)
    L_im_max = rpop.apply(lambda x: LLR(x.to_dict(), bn_theta_ie, bn_min_ie), axis=1)

    # Estimate the ECDFs
    ecdf = ECDF(L_im)
    ecdf_min = ECDF(L_im_min)
    ecdf_max = ECDF(L_im_max)

    ## Computation
    # Set ground truth membership
    gpop["in-pool"] = False
    gpop.loc[pool_idx, "in-pool"] = True
    
    assert(sum(gpop["in-pool"] == True) == pool_ss)

    # Set threshold range
    t_range = np.arange(-1e1, 1e1, 0.05)    #

    # Init the error vector 
    error = []

    # Init the power vector (BN)
    power_bn = []

    # For each threshold ...
    for t in t_range:

        # Store the related error
        error = error + [ecdf(t)]

        # Perform LR test on whole population
        y_pred = gpop[[*bn.names()]].apply(lambda x: reject_H0(x.to_dict(), bn_theta_ie, bn_theta_hat_ie, t), axis=1)

        # Compute and store power (tpr)
        tpr = sum(gpop["in-pool"] & y_pred) / gpop_ss
        power_bn = power_bn + [tpr]

    # Init the power vector (CN)
    power_cn = []

    # For each threshold ...
    for t in t_range:

        # Perform min-max LR test on whole population
        y_pred_min = gpop[[*bn.names()]].apply(lambda x: reject_H0(x.to_dict(), bn_theta_ie, bn_max_ie, t), axis=1)
        y_pred_max = gpop[[*bn.names()]].apply(lambda x: reject_H0(x.to_dict(), bn_theta_ie, bn_min_ie, t), axis=1)
        cons = y_pred_min == y_pred_max

        # Compute and store power (tpr)
        tpr = sum(gpop[cons]["in-pool"] & y_pred_min[cons]) / gpop_ss
        power_cn = power_cn + [tpr]

    # Compute theoretical bound
    compl = bn.dim()
    bound = math.sqrt(compl/pool_ss)

    # Find power (beta) for any error (alpha) given theoretical bound
    z_alpha = [norm.ppf(1 - i).item() for i in error]
    z_one_minus_beta = [bound - i for i in z_alpha]
    beta = [norm.cdf(i).item() for i in z_one_minus_beta]

    ## Save results
    results = pd.DataFrame(
        {"error": error,
        "power_bound": beta,
        "power_BN": power_bn,
        "power_CN": power_cn}
    )
    results.to_csv(f"./results/nnodes{n_nodes}-compl{compl}-s{ess}.csv")
