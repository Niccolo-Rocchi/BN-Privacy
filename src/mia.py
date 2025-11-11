import math
import traceback

import numpy as np
import pandas as pd
import pyagrum as gum
from scipy.stats import norm
from sklearn import metrics

import src.attacks
from src.config import get_out_path, set_global_seed
import src.defenses
from src.utils import check_consistency, get_llr, get_min_max_bns, noisy_bn, safe_assert, save_bn


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


# Find eps s.t. |AUC(eps) - AUC(CN)| < tol
def get_eps(exp, ess, config):

    # Get output path
    out_path = get_out_path(config)

    # Set seed
    set_global_seed(config["seed"])

    # Init hyperp.
    eps_vec = eval(config["ess_dict"][ess])
    results_path = out_path / config["results_path"]
    n_samples = config["n_samples"]
    error = eval(eval(config["error"]))
    tol = config["tol"]
    def_mec = eval(config["def_mec"])
    atk_mec = eval(config["atk_mec"])

    # Read data
    gpop = pd.read_csv(f'{out_path / config["data_path"]}/{exp}.csv')
    bn = gum.loadBN(f'{out_path / config["bns_path"]}/{exp}.bif')
    n_nodes = config["n_nodes"]
    gpop_ss = config["gpop_ss"]
    rpop_ss = int(gpop_ss * config["rpop_prop"])
    pool_ss = int(gpop_ss * config["pool_prop"])

    # Debug
    safe_assert(gpop_ss == gpop.shape[0])
    safe_assert(n_nodes == gpop.shape[1])

    bn_theta_vec = []
    bn_theta_hat_vec = []
    cn_vec = []

    # For any data sample ...
    for sample in range(n_samples):

        # ... retrieve pool and rpop, ...
        pool = gpop[gpop[f"in-pool-{sample}"]].iloc[:, :n_nodes]
        rpop = gpop[gpop[f"in-rpop-{sample}"]].iloc[:, :n_nodes]

        # ... estimate BN from rpop, ...
        learner = gum.BNLearner(rpop)
        learner.useSmoothingPrior(1e-5)
        bn_theta_vec.append(learner.learnParameters(bn.dag()))

        # ... estimate BN from pool, ...
        learner = gum.BNLearner(pool)
        learner.useSmoothingPrior(1e-5)
        bn_theta_hat_vec.append(learner.learnParameters(bn.dag()))

        # ... and run Defense mechanism: estimate the CN
        cn = def_mec(bn, ess, pool)
        cn_vec.append(cn)

        # Debug
        safe_assert(len(pool) == sum(gpop[f"in-pool-{sample}"]))
        safe_assert(len(pool) == pool_ss)
        safe_assert(len(rpop) == rpop_ss)

    # Debug
    safe_assert(len(bn_theta_vec) == n_samples)
    safe_assert(len(bn_theta_hat_vec) == n_samples)
    safe_assert(len(cn_vec) == n_samples)

    # Run MIA against CN
    auc_cn_vec = []
    for sample in range(n_samples):

        # Retrieve sample-related info
        y_true = gpop[f"in-pool-{sample}"]
        cn = cn_vec[sample]
        bn_theta_ie = gum.LazyPropagation(bn_theta_vec[sample])

        # Attack mechanism: extract a BN from the CN
        ext_bn = atk_mec(cn, rpop, exp, config)
        bn_ie = gum.LazyPropagation(ext_bn)

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
            bn_noisy = noisy_bn(bn_theta_hat, scale)
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

# Learn BN parameters from a given DAG and data
def learn_bn_params(dag, data):

    learner = gum.BNLearner(data)
    learner.useSmoothingPrior(1e-5)
    bn = learner.learnParameters(dag)

    return bn

# Estimate BNs from rpop and pool
def phase_estimation(exp, config) -> None:

    # Get output path
    out_path = get_out_path(config)

    # Read data
    gpop = pd.read_csv(f'{out_path / config["data_path"]}/{exp}.csv')
    bn = gum.loadBN(f'{out_path / config["bns_path"]}/{exp}.bif')
    n_nodes = len(bn.nodes())
    gpop_ss = config["gpop_ss"]
    rpop_ss = int(gpop_ss * config["rpop_prop"])
    pool_ss = int(gpop_ss * config["pool_prop"])

    # Debug
    safe_assert(gpop_ss == gpop.shape[0])
    safe_assert(n_nodes == gpop.shape[1])

    # For each data sample ...
    for sample in range(config["n_samples"]):

        # ... retrieve pool and rpop, ...
        pool = gpop[gpop[f"in-pool-{sample}"]].iloc[:, :n_nodes]
        rpop = gpop[gpop[f"in-rpop-{sample}"]].iloc[:, :n_nodes]

        # ... estimate BN from rpop, ...
        bn_learnt = learn_bn_params(bn.dag(), rpop)
        save_bn(bn_learnt, f"bn_{exp}_sample{sample}", out_path / config["rpop_path"])

        # ... estimate BN from pool, ...
        bn_learnt = learn_bn_params(bn.dag(), pool)
        save_bn(bn_learnt, f"bn_{exp}_sample{sample}", out_path / config["pool_path"])

        # Debug
        safe_assert(len(pool) == sum(gpop[f"in-pool-{sample}"]))
        safe_assert(len(pool) == pool_ss)
        safe_assert(len(rpop) == rpop_ss)
    
    return


# Apply defense mechanism to a BN, namely, derive a CN from a BN
def phase_defense_mechanism(def_mec, exp, ess, config) -> None:

    # Get output path
    out_path = get_out_path(config)
    
    # Read data
    gpop = pd.read_csv(f'{out_path / config["data_path"]}/{exp}.csv')

    # For each data sample ...
    for sample in range(config["n_samples"]):

        # ... read the related BN
        bn = gum.loadBN(f"{out_path}/{config['pool_path']}/bn_{exp}_sample{sample}.bif")

        # ... retrieve pool, ...
        pool = gpop[gpop[f"in-pool-{sample}"]].iloc[:, :len(bn.nodes())]

        # ... and derive the CN
        def_mec_fn = getattr(src.defenses, def_mec)
        cn = def_mec_fn(bn, ess, pool)
        # bn_min, bn_max = get_min_max_bns(cn, exp)
        # save_bn(bn_min, f"bn_min_{exp}_sample{sample}", out_path / config["cns_path"] / f"ESS: {ess}")
        # save_bn(bn_max, f"bn_max_{exp}_sample{sample}", out_path / config["cns_path"] / f"ESS: {ess}")
        base_path = out_path / config["cns_path"] / f"ESS: {ess}"
        cn.saveBNsMinMax(f"{base_path}/bn_min_{exp}_sample{sample}.bif", f"{base_path}/bn_max_{exp}_sample{sample}.bif")

    
    return

# Apply attack mechanism to a BN, namely, derive a BN from a CN
def phase_attack_mechanism(atk_mec, exp, ess, config) -> None:

    # Get output path
    out_path = get_out_path(config)
    
    # Read data
    gpop = pd.read_csv(f'{out_path / config["data_path"]}/{exp}.csv')

    # For each data sample ...
    for sample in range(config["n_samples"]):

        # ... read the related CN
        bn_min = gum.loadBN(f"{out_path}/{config['cns_path']}/ESS: {ess}/bn_min_{exp}_sample{sample}.bif")
        bn_max = gum.loadBN(f"{out_path}/{config['cns_path']}/ESS: {ess}/bn_max_{exp}_sample{sample}.bif")

        # ... retrieve rpop, ...
        rpop = gpop[gpop[f"in-rpop-{sample}"]].iloc[:, :len(bn_min.nodes())]

        # ... and derive the BN
        atk_mec_fn = getattr(src.attacks, atk_mec)
        bn = atk_mec_fn(bn_min, bn_max, rpop, exp, config)
        save_bn(bn, f"bn_{exp}_sample{sample}", out_path / config['atk_path'] / f"ESS: {ess}")

    return

# MIA attack vs a BN
def phase_mia_vs_bn(exp, config) -> None:

    # Get output path
    out_path = get_out_path(config)

    # Init results
    results = pd.DataFrame({"error": eval(config["error"])})
    
    # Read data
    gpop = pd.read_csv(f'{out_path / config["data_path"]}/{exp}.csv')

    # For each data sample ...
    for sample in range(config["n_samples"]):

        # ... read the BNs as estimated from rpop and pool, ...
        bn_theta = gum.loadBN(f"{out_path}/{config['rpop_path']}/bn_{exp}_sample{sample}.bif")
        bn_theta_hat = gum.loadBN(f"{out_path}/{config['pool_path']}/bn_{exp}_sample{sample}.bif")

        bn_theta_hat_ie = gum.LazyPropagation(bn_theta_hat)
        bn_theta_ie = gum.LazyPropagation(bn_theta)

        # ... retrieve rpop, ...
        rpop = gpop[gpop[f"in-rpop-{sample}"]].iloc[:, :len(bn_theta.nodes())]

        # try:

        # ... and perform membership inference on gpop
        power_vec, _ = run_mia(
            bn_theta_hat_ie, bn_theta_ie, rpop, gpop, gpop[f"in-pool-{sample}"], eval(config["error"])
        )
        results[f"power_BN_sample{sample}"] = power_vec

        # except Exception:

        #     # Debug
        #     with open(f"{results_path}/log.txt", "a") as log:
        #         log.write(f"{exp}: error with sample {sample} (BN).\n")
        #         log.write(traceback.format_exc())

    # Save results
    results.to_csv(f'{out_path}/{config["results_path"]}/bn_{exp}.csv', index=False)

# MIA attack vs a CN
def phase_mia_vs_cn(exp, ess, config) -> None:

    # Get output path
    out_path = get_out_path(config)

    # Init results
    results = pd.DataFrame({"error": eval(config["error"])})
    
    # Read data
    gpop = pd.read_csv(f'{out_path / config["data_path"]}/{exp}.csv')

    # For each data sample ...
    for sample in range(config["n_samples"]):

        # ... read the BN as inferred from the CN
        bn_theta_hat = gum.loadBN(f'{out_path}/{config["atk_path"]}/ESS: {ess}/bn_{exp}_sample{sample}.bif')

        # ... read the BN as estimated from rpop, ...
        bn_theta = gum.loadBN(f"{out_path}/{config['rpop_path']}/bn_{exp}_sample{sample}.bif")

        bn_theta_hat_ie = gum.LazyPropagation(bn_theta_hat)
        bn_theta_ie = gum.LazyPropagation(bn_theta)

        # ... retrieve rpop, ...
        rpop = gpop[gpop[f"in-rpop-{sample}"]].iloc[:, :len(bn_theta.nodes())]

        # try:

        # ... and perform membership inference on gpop
        power_vec, _ = run_mia(
            bn_theta_hat_ie, bn_theta_ie, rpop, gpop, gpop[f"in-pool-{sample}"], eval(config["error"])
        )
        results[f"power_CN_sample{sample}"] = power_vec

        # except Exception:

        #     # Debug
        #     with open(f"{results_path}/log.txt", "a") as log:
        #         log.write(f"{exp}: error with sample {sample} (BN).\n")
        #         log.write(traceback.format_exc())

    # Save results
    results.to_csv(f'{out_path}/{config["results_path"]}/cn_{exp}-ess{ess}.csv', index=False)

# Get theoretical power
def phase_theoretical_power(exp, config):

    # Read BN
    bn = gum.loadBN(f'{get_out_path(config) / config["bns_path"]}/{exp}.bif')

    # Compute bound
    bound = math.sqrt(bn.dim() / int(config["gpop_ss"] * config["pool_prop"]))

    # Find power (beta) for any error (alpha) given theoretical bound
    z_alpha = [norm.ppf(1 - i).item() for i in eval(config["error"])]
    z_one_minus_beta = [bound - i for i in z_alpha]
    beta = [norm.cdf(i).item() for i in z_one_minus_beta]

    return beta
    
    
