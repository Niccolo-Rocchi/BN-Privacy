import math
import traceback

import numpy as np
import pandas as pd
import pyagrum as gum
from scipy.stats import norm
from sklearn import metrics

from src.config import get_base_path, set_global_seed
from src.utils import (add_counts_to_bn, get_ll, get_llr, get_noisy_bn,
                       sample_from_cn)


# Get the attack power related to a fixed error
def get_power(llr_ref, llr_gen, ground_truth, error) -> float:

    # Compute the threshold
    t = np.quantile(llr_ref, 1 - error).item()

    # Test: L(x) > t => reject H_0 => assign `x` to target_pop
    y_pred = llr_gen > t

    # Compute power (i.e., true positive rate)
    power = sum(ground_truth & y_pred) / sum(ground_truth)

    return power


# MIA: membership inference attack
def run_mia(model, baseline, rpop, gpop, ground_truth, error_vec):

    # Compute llr(x) on reference and general populations
    llr_ref = (
        rpop.apply(lambda x: get_llr(x.to_dict(), baseline, model), axis=1)
        .dropna()
        .sort_values()
    )
    llr_gen = gpop[[*rpop.columns]].apply(
        lambda x: get_llr(x.to_dict(), baseline, model), axis=1
    )

    power_vec = []

    # Get the power for each error
    for error in error_vec:
        power = get_power(llr_ref, llr_gen, ground_truth, error)
        power_vec.append(power)

    # Compute and store AUC
    auc = metrics.auc(error_vec, power_vec)

    return power_vec, auc


# Get the maximum likelihood BN
def get_maxll_bn(bns_sample, rpop):
    """
    Given a list `bns_sample` of BNs,
    find argmax_{BN in bns_sample} ll(BN | rpop),
    where ll is the log-likelihood function.
    """

    maxll_bn = None
    maxll = -np.inf

    for bn in bns_sample:

        # Estimate the likelihood of rpop
        bn_ie = gum.LazyPropagation(bn)
        llr_im = rpop.apply(lambda x: get_ll(x.to_dict(), bn_ie), axis=1).dropna()
        llr = np.sum(llr_im)

        if llr > maxll:
            maxll_bn = bn
            maxll = llr

    return maxll_bn


# Find eps s.t. |AUC(eps) - AUC(CN)| < tol
def get_eps(exp, ess, config):

    # Get base path
    base_path = get_base_path(config)

    # Set seed
    set_global_seed(config["seed"])

    # Init hyperp.
    eps_vec = eval(config["ess_dict"][ess])
    results_path = base_path / config["results_path"]
    n_samples = config["n_samples"]
    n_bns = config["n_bns"]
    error = eval(config["error"])
    tol = config["tol"]

    # Read data
    gpop = pd.read_csv(f'{base_path / config["data_path"]}/{exp}.csv')
    bn = gum.loadBN(f'{base_path / config["bns_path"]}/{exp}.bif')
    n_nodes = config["n_nodes"]
    gpop_ss = config["gpop_ss"]
    rpop_ss = int(gpop_ss * config["rpop_prop"])
    pool_ss = int(gpop_ss * config["pool_prop"])

    # Debug
    assert gpop_ss == gpop.shape[0]
    assert n_nodes == gpop.shape[1]

    bn_theta_vec = []
    bn_theta_hat_vec = []
    cn_vec = []

    # For any data sample ...
    for sample in range(n_samples):

        # ... sample pool and rpop, ...
        pool_idx = np.random.choice(range(gpop_ss), size=pool_ss, replace=False)
        gpop[f"in-pool-{sample}"] = gpop.index.isin(pool_idx)
        pool = gpop[gpop[f"in-pool-{sample}"]].iloc[:, :n_nodes]
        rpop = gpop[~gpop[f"in-pool-{sample}"]].iloc[:, :n_nodes].sample(rpop_ss)

        # ... estimate BN from rpop, ...
        learner = gum.BNLearner(rpop)
        learner.useSmoothingPrior(1e-5)
        bn_theta_vec.append(learner.learnParameters(bn.dag()))

        # ... estimate BN from pool, ...
        learner = gum.BNLearner(pool)
        learner.useSmoothingPrior(1e-5)
        bn_theta_hat_vec.append(learner.learnParameters(bn.dag()))

        # ... and estimate CN from pool (by local IDM)
        bn_counts = gum.BayesNet(bn)
        add_counts_to_bn(bn_counts, pool)
        cn = gum.CredalNet(bn_counts)
        cn.idmLearning(ess)
        cn_vec.append(cn)

        # Debug
        assert len(pool) == sum(gpop[f"in-pool-{sample}"])
        assert len(pool) == pool_ss
        assert len(rpop) == rpop_ss

    # Debug
    assert len(bn_theta_vec) == n_samples
    assert len(bn_theta_hat_vec) == n_samples
    assert len(cn_vec) == n_samples

    # Run MIA against CN
    auc_cn_vec = []
    for sample in range(n_samples):

        # Retrieve sample-related info
        y_true = gpop[f"in-pool-{sample}"]
        cn = cn_vec[sample]
        bn_theta_ie = gum.LazyPropagation(bn_theta_vec[sample])

        # Extract random subset within simplex
        bns_sample = sample_from_cn(cn, n_bns, "inside")

        # Get the maximum likelihood BN
        best_bn = get_maxll_bn(bns_sample, rpop)
        bn_ie = gum.LazyPropagation(best_bn)

        # MIA
        try:
            _, auc = run_mia(bn_ie, bn_theta_ie, rpop, gpop, y_true, error)
            auc_cn_vec.append(auc)

        except Exception:

            # Debug
            with open(f"{results_path}/log.txt", "a") as log:
                log.write(f"{exp}: error with sample {sample} (CN).\n")
                log.write(traceback.format_exc())

    # Compute Avg(AUC(CN)) across data samples
    auc_cn = sum(auc_cn_vec) / len(auc_cn_vec)

    # Find eps
    eps_best = eps_vec[-1]

    # For each eps ...
    for eps in eps_vec:

        auc_bn_noisy_vec = []

        # ... run MIA against noisy BN ...
        for sample in range(n_samples):

            # Retrieve sample-related info
            y_true = gpop[f"in-pool-{sample}"]
            bn_theta_hat = bn_theta_hat_vec[sample]
            bn_theta_ie = gum.LazyPropagation(bn_theta_vec[sample])

            # Get noisy BN
            scale = (2 * bn_theta_hat.size()) / (len(pool) * eps)
            bn_noisy = get_noisy_bn(bn_theta_hat, scale)
            bn_noisy_ie = gum.LazyPropagation(bn_noisy)

            try:

                # MIA
                _, auc = run_mia(bn_noisy_ie, bn_theta_ie, rpop, gpop, y_true, error)
                auc_bn_noisy_vec.append(auc)

            except Exception:

                # Debug
                with open(f"{results_path}/log.txt", "a") as log:
                    log.write(
                        f"{exp}: error with sample {sample} (BN noisy, eps: {eps}).\n"
                    )
                    log.write(traceback.format_exc())

        # ... and compute Avg(AUC(eps)) across data samples
        auc_bn = sum(auc_bn_noisy_vec) / n_samples

        # Condition on |AUC(eps) - AUC(CN)|
        if abs(auc_cn - auc_bn) <= tol:
            eps_best = eps
            break

    # Store found eps
    meta_file_path = (
        results_path
        / f'results_nodes{config["n_nodes"]}_ess{ess}'
        / config["meta_file"]
    )
    with open(meta_file_path, "a") as m:
        m.write(f"- {exp}. Nodes: {n_nodes} Eps: {eps_best}\n")

    return exp, ess, eps_best


# Membership attack against CN and BN
def attack_cn_bn(exp, ess, config):

    # Get base path
    base_path = get_base_path(config)

    # Set seed
    set_global_seed(config["seed"])

    # Init hyperp.
    results_path = base_path / config["results_path"]
    n_samples = config["n_samples"]
    n_bns = config["n_bns"]
    error = eval(config["error"])

    # Read data
    gpop = pd.read_csv(f'{base_path / config["data_path"]}/{exp}.csv')
    bn = gum.loadBN(f'{base_path / config["bns_path"]}/{exp}.bif')
    n_nodes = len(bn.nodes())
    gpop_ss = config["gpop_ss"]
    rpop_ss = int(gpop_ss * config["rpop_prop"])
    pool_ss = int(gpop_ss * config["pool_prop"])

    # Debug
    assert gpop_ss == gpop.shape[0]
    assert n_nodes == gpop.shape[1]

    bn_theta_vec = []
    bn_theta_hat_vec = []
    cn_vec = []

    # For any data sample ...
    for sample in range(n_samples):

        # ... sample pool and rpop, ...
        pool_idx = np.random.choice(range(gpop_ss), size=pool_ss, replace=False)
        gpop[f"in-pool-{sample}"] = gpop.index.isin(pool_idx)
        pool = gpop[gpop[f"in-pool-{sample}"]].iloc[:, :n_nodes]
        rpop = gpop[~gpop[f"in-pool-{sample}"]].iloc[:, :n_nodes].sample(rpop_ss)

        # ... estimate BN from rpop, ...
        learner = gum.BNLearner(rpop)
        learner.useSmoothingPrior(1e-5)
        bn_theta_vec.append(learner.learnParameters(bn.dag()))

        # ... estimate BN from pool, ...
        learner = gum.BNLearner(pool)
        learner.useSmoothingPrior(1e-5)
        bn_theta_hat_vec.append(learner.learnParameters(bn.dag()))

        # ... and estimate CN from pool (by local IDM)
        bn_counts = gum.BayesNet(bn)
        add_counts_to_bn(bn_counts, pool)
        cn = gum.CredalNet(bn_counts)
        cn.idmLearning(ess)
        cn_vec.append(cn)

        # Debug
        assert len(pool) == sum(gpop[f"in-pool-{sample}"])
        assert len(pool) == pool_ss
        assert len(rpop) == rpop_ss

    # Debug
    assert len(bn_theta_vec) == n_samples
    assert len(bn_theta_hat_vec) == n_samples
    assert len(cn_vec) == n_samples

    # Compute theoretical bound
    compl = bn.dim()
    bound = math.sqrt(compl / pool_ss)

    # Find power (beta) for any error (alpha) given theoretical bound
    z_alpha = [norm.ppf(1 - i).item() for i in error]
    z_one_minus_beta = [bound - i for i in z_alpha]
    beta = [norm.cdf(i).item() for i in z_one_minus_beta]

    # Init results
    results = pd.DataFrame({"error": error, "power_bound": beta})

    # Run MIA against CN
    for sample in range(n_samples):

        # Retrieve sample-related info
        y_true = gpop[f"in-pool-{sample}"]
        cn = cn_vec[sample]
        bn_theta_ie = gum.LazyPropagation(bn_theta_vec[sample])

        # Extract random subset within simplex
        bns_sample = sample_from_cn(cn, n_bns, "inside")

        # Get the maximum likelihood BN
        best_bn = get_maxll_bn(bns_sample, rpop)
        bn_ie = gum.LazyPropagation(best_bn)

        # MIA
        try:
            power_vec, _ = run_mia(bn_ie, bn_theta_ie, rpop, gpop, y_true, error)
            results[f"power_CN_sample{sample}"] = power_vec

        except Exception:

            # Debug
            with open(f"{results_path}/log.txt", "a") as log:
                log.write(f"{exp}: error with sample {sample} (CN).\n")
                log.write(traceback.format_exc())

    # Run MIA against BN
    for sample in range(n_samples):

        # Retrieve sample-related info
        y_true = gpop[f"in-pool-{sample}"]
        bn_theta_hat_ie = gum.LazyPropagation(bn_theta_hat_vec[sample])
        bn_theta_ie = gum.LazyPropagation(bn_theta_vec[sample])

        try:

            # MIA
            power_vec, _ = run_mia(
                bn_theta_hat_ie, bn_theta_ie, rpop, gpop, y_true, error
            )
            results[f"power_BN_sample{sample}"] = power_vec

        except Exception:

            # Debug
            with open(f"{results_path}/log.txt", "a") as log:
                log.write(f"{exp}: error with sample {sample} (BN).\n")
                log.write(traceback.format_exc())

    # Save results
    results.to_csv(f"{results_path}/{exp}-ess{ess}.csv", index=False)
