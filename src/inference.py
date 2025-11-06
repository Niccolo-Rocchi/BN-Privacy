import math

import numpy as np
import pandas as pd
import pyagrum as gum
from more_itertools import random_product

from src.config import get_base_path, set_global_seed
from src.utils import (add_counts_to_bn, get_min_max_bns, noisy_bn, safe_assert)


def run_inferences(exp, ess, eps, config):

    base_path = get_base_path(config)
    target = config["target_var"]

    # Set seed
    set_global_seed(config["seed"])

    # Set list of evidence
    evid_vec = [
        random_product(*((0, 1) for _ in range(config["n_nodes"] - 1)))
        for _ in range(config["n_infer"])
    ]

    # Store ground-truth BN
    gt = gum.loadBN(f'{base_path / config["bns_path"]}/{exp}.bif')
    gpop = pd.read_csv(f'{base_path / config["data_path"]}/{exp}.csv')

    # Learn BN from gpop
    bn_learner = gum.BNLearner(gpop)
    bn_learner.useSmoothingPrior(1e-5)
    bn = bn_learner.learnParameters(gt.dag())

    # Learn CN from gpop
    bn_copy = gum.BayesNet(bn)
    add_counts_to_bn(bn_copy, gpop)
    cn = gum.CredalNet(bn_copy)
    cn.idmLearning(ess)

    # Learn noisy BN from gpop
    scale = (2 * bn.size()) / (len(gpop) * eps)
    bn_noisy = noisy_bn(bn, scale)

    # Run inferences
    gt_mpes, _ = run_inference_bn(gt, target, evid_vec)
    bn_mpes, bn_probs = run_inference_bn(bn, target, evid_vec)
    bn_noisy_mpes, bn_noisy_probs = run_inference_bn(bn_noisy, target, evid_vec)
    cn_mpes, cn_probs, cn_probs_alt = run_inference_cn(cn, target, evid_vec, exp)

    # Save results
    results = pd.DataFrame(
        {
            "gt_mpes": gt_mpes,
            "bn_mpes": bn_mpes,
            "bn_probs": bn_probs,
            "bn_noisy_mpes": bn_noisy_mpes,
            "bn_noisy_probs": bn_noisy_probs,
            "cn_mpes": cn_mpes,
            "cn_probs": cn_probs,
            "cn_probs_alt": cn_probs_alt,
        }
    )

    res_path = (
        base_path
        / config["results_path"]
        / f'results_nodes{config["n_nodes"]}_ess{ess}'
    )
    results.to_csv(f"{res_path}/{exp}.csv", index=False)


# MPE function for BN
def mpe_bn(bn_ie: gum.LazyPropagation, target: str, evid: dict) -> tuple:

    # Set evidence
    bn_ie.setEvidence(evid)

    # Compute MPE and log(prob)
    out = bn_ie.mpeLog2Posterior()
    mpe = out[0].todict().get(target)
    prob = np.exp2(out[1])

    return mpe, prob


# MPE function for CN
def mpe_cn(
    bn_min: gum.BayesNet, bn_max: gum.BayesNet, target: str, children: dict
) -> tuple:
    """
    Get the MPE of a CN as: argmax_t log P_lower(target=t | children).
    bn_min and bn_max derive from a CN.
    The DAG is a naive Bayes with `target` a binary target variable.
    Returns the MPE, its probability, and the lower probability of the alternative class.
    """

    lp1 = nb_log_lower_posterior(bn_min, bn_max, target, 1, children)
    lp0 = nb_log_lower_posterior(bn_min, bn_max, target, 0, children)

    if lp1 > lp0:
        return (1, math.exp(lp1), math.exp(lp0))

    return (0, math.exp(lp0), math.exp(lp1))


# Get a value from a BN's CPT
def cpt_value(
    bn: gum.BayesNet, x_var: str, x_value: float, parents: dict = None
) -> float:
    """
    Get P(X=x | parents) from the BN's CPT of X.
    `x_var` is the X name, while `x_value` is x.
    """

    cpt = bn.cpt(x_var)
    inst = gum.Instantiation(cpt)
    inst[x_var] = x_value

    if parents:
        for var in parents.keys():
            inst[var] = parents[var]
        safe_assert(bn.parents(x_var) == set(bn.ids(parents.keys())))
    else:
        safe_assert(len(bn.parents(x_var)) == 0)

    return max(cpt.get(inst), 1e-10)  # Smoothing


# Get a naive Bayes log-joint
def nb_log_joint(
    bn: gum.BayesNet, target: str, t: float, children: dict
) -> float:
    """
    Get log[P(target=t, children)] by exploiting the BN factorization.
    The DAG is a naive Bayes with `target` a binary target variable.
    """

    sum_log = math.log(cpt_value(bn, target, t))
    for var, val in children.items():
        sum_log += math.log(cpt_value(bn, var, val, {target: t}))

    return sum_log


# Get the lower posterior from a CN
def nb_log_lower_posterior(
    bn_min: gum.BayesNet, bn_max: gum.BayesNet, target: str, t: float, children: dict
) -> float:
    """
    Get log P_lower(target=t | children).
    bn_min and bn_max derive from a CN.
    The DAG is a naive Bayes with `target` a binary target variable.
    """

    l_lower = nb_log_joint(bn_min, target, t, children)
    l_upper = nb_log_joint(bn_max, target, 1 - t, children)

    return l_lower - l_upper - math.log1p(math.exp(l_lower - l_upper))


# Run inferences on a BN
def run_inference_bn(bn, target: str, evid_vec):
    """
    The BN is assumed to be a naive Bayes model with `target` the target variable.
    """

    # Store information
    cov = sorted(list(bn.names()))
    cov.remove(target)

    # Debug
    safe_assert(len(cov) == bn.size() - 1)

    # Create object for inference
    bn_ie = gum.LazyPropagation(bn)

    # Compute all combinations of evidence
    mpes = []
    probs = []
    for e in evid_vec:
        evid = dict(zip(cov, e))
        mpe, prob = mpe_bn(bn_ie, target, evid)
        mpes.append(mpe)
        probs.append(prob)

    # Debug
    safe_assert(len(mpes) == len(evid_vec))
    safe_assert(len(probs) == len(evid_vec))

    return mpes, probs


# Run inferences on a CN
def run_inference_cn(cn, target: str, evid_vec, exp: str):
    """
    The CN is assumed to be a naive Bayes model with `target` the target variable.
    """

    # Store information
    bn_min, bn_max = get_min_max_bns(cn, exp)
    cov = sorted(list(bn_min.names()))
    cov.remove(target)

    # Debug
    safe_assert(len(cov) == bn_min.size() - 1)

    # Compute all combinations of evidence
    mpes = []
    probs = []
    probs_alt = []
    for e in evid_vec:
        evid = dict(zip(cov, e))
        mpe, prob, prob_alt = mpe_cn(bn_min, bn_max, target, evid)
        mpes.append(mpe)
        probs.append(prob)
        probs_alt.append(prob_alt)

    # Debug
    safe_assert(len(mpes) == len(evid_vec))
    safe_assert(len(probs) == len(evid_vec))
    safe_assert(len(probs_alt) == len(evid_vec))

    return mpes, probs, probs_alt
