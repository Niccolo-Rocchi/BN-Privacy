import math

import numpy as np
import pandas as pd
import pyagrum as gum
from scipy.stats import norm
from sklearn import metrics

import src.attacks
import src.defenses
from src.config import get_out_path
from src.utils import get_llr, noisy_bn, safe_assert, safe_open_dir, save_bn


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
def phase_find_eps(exp, ess, config) -> dict:

    # Get output path
    out_path = get_out_path(config)

    # Read data
    gpop = pd.read_csv(f'{out_path / config["data_path"]}/{exp}.csv')
    gpop_ss = config["gpop_ss"]
    pool_ss = int(gpop_ss * config["pool_prop"])
    auc_meta = pd.read_csv(f'{out_path}/{config["auc_meta"]}')
    auc_cn = auc_meta.loc[
        (auc_meta["exp"] == exp) & (auc_meta["ess"] == ess), "auc_cn"
    ].values[0]
    eps_vec = eval(config["ess_dict"][ess])

    eps_best = eps_vec[-1]

    # For each eps ...
    for eps in eps_vec:

        # Init results
        results = pd.DataFrame({"error": eval(config["error"])})

        auc_noisy_bns = []

        # ... and for each data sample ...
        for sample in range(config["n_samples"]):

            # ... read the BNs as estimated from rpop and pool, ...
            bn_theta = gum.loadBN(
                f"{out_path}/{config['bns_path']}/rpop/bn_{exp}_sample{sample}.bif"
            )
            bn_theta_hat = gum.loadBN(
                f"{out_path}/{config['bns_path']}/pool/bn_{exp}_sample{sample}.bif"
            )

            # Get noisy BN
            scale = (2 * bn_theta_hat.size()) / (pool_ss * eps)
            bn_noisy = noisy_bn(bn_theta_hat, scale)

            bn_noisy_ie = gum.LazyPropagation(bn_noisy)
            bn_theta_ie = gum.LazyPropagation(bn_theta)

            # ... retrieve rpop, ...
            rpop = gpop[gpop[f"in-rpop-{sample}"]].iloc[:, : len(bn_theta.nodes())]

            # try:

            # ... and perform membership inference on gpop
            power_vec, auc = run_mia(
                bn_noisy_ie,
                bn_theta_ie,
                rpop,
                gpop,
                gpop[f"in-pool-{sample}"],
                eval(config["error"]),
            )
            results[f"power_noisyBN_sample{sample}"] = power_vec
            auc_noisy_bns.append(auc)

            # except Exception:

            #     # Debug
            #     with open(f"{results_path}/log.txt", "a") as log:
            #         log.write(f"{exp}: error with sample {sample} (BN).\n")
            #         log.write(traceback.format_exc())

        # Compute Avg(AUC(eps)) across data samples
        auc_noisy_bn = sum(auc_noisy_bns) / config["n_samples"]

        # Condition on |AUC(eps) - AUC(CN)|
        if abs(auc_cn - auc_noisy_bn) <= config["tol"]:
            eps_best = eps
            break

    # Save noisy BNs
    for sample in range(config["n_samples"]):
        bn_theta_hat = gum.loadBN(
            f"{out_path}/{config['bns_path']}/pool/bn_{exp}_sample{sample}.bif"
        )
        scale = (2 * bn_theta_hat.size()) / (pool_ss * eps_best)
        bn_noisy = noisy_bn(bn_theta_hat, scale)
        save_bn(bn_noisy, f"bn_{exp}_sample{sample}", out_path / config["noisy_path"])

    return {
        "exp": exp,
        "ess": ess,
        "auc_cn": auc_cn,
        "auc_noisy_bn": auc_noisy_bn,
        "eps": eps_best,
    }


# Learn BN parameters from a given BN and data
def learn_bn_params(bn, data):

    bn_copy = gum.BayesNet(bn)

    learner = gum.BNLearner(data, bn_copy)
    learner.useSmoothingPrior(1e-5)
    bn_learnt = learner.learnParameters(bn_copy)

    return bn_learnt


# Estimate BNs from rpop and pool
def phase_estimation(exp, config) -> None:

    # Get output path
    out_path = get_out_path(config)

    # Read data
    gpop = pd.read_csv(f'{out_path / config["data_path"]}/{exp}.csv')
    bn = gum.loadBN(f'{out_path / config["bns_path"]}/gt/{exp}.bif')
    n_nodes = len(bn.nodes())
    gpop_ss = config["gpop_ss"]
    rpop_ss = int(gpop_ss * config["rpop_prop"])
    pool_ss = int(gpop_ss * config["pool_prop"])

    # Debug
    safe_assert(gpop_ss == gpop.shape[0])
    safe_assert(n_nodes == gpop.loc[:, ~gpop.columns.str.contains("in-")].shape[1])

    # For each data sample ...
    for sample in range(config["n_samples"]):

        # ... retrieve pool and rpop, ...
        pool = gpop[gpop[f"in-pool-{sample}"]].iloc[:, :n_nodes]
        rpop = gpop[gpop[f"in-rpop-{sample}"]].iloc[:, :n_nodes]

        # ... estimate BN from rpop, ...
        bn_learnt = learn_bn_params(bn, rpop)
        save_bn(bn_learnt, f"bn_{exp}_sample{sample}", out_path / config['bns_path'] / "rpop")

        # ... estimate BN from pool, ...
        bn_learnt = learn_bn_params(bn, pool)
        save_bn(bn_learnt, f"bn_{exp}_sample{sample}", out_path / config['bns_path'] / "pool")

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
        bn = gum.loadBN(f"{out_path}/{config['bns_path']}/pool/bn_{exp}_sample{sample}.bif")

        # ... retrieve pool, ...
        pool = gpop[gpop[f"in-pool-{sample}"]].iloc[:, : len(bn.nodes())]

        # ... and derive the CN
        def_mec_fn = getattr(src.defenses, def_mec)
        cn = def_mec_fn(bn, ess, pool)
        base_path = out_path / config["cns_path"] / f"ESS: {ess}"
        safe_open_dir(base_path)
        cn.saveBNsMinMax(
            f"{base_path}/bn_min_{exp}_sample{sample}.bif",
            f"{base_path}/bn_max_{exp}_sample{sample}.bif",
        )

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
        bn_min = gum.loadBN(
            f"{out_path}/{config['cns_path']}/ESS: {ess}/bn_min_{exp}_sample{sample}.bif"
        )
        bn_max = gum.loadBN(
            f"{out_path}/{config['cns_path']}/ESS: {ess}/bn_max_{exp}_sample{sample}.bif"
        )

        # ... retrieve rpop, ...
        rpop = gpop[gpop[f"in-rpop-{sample}"]].iloc[:, : len(bn_min.nodes())]

        # ... and derive the BN
        atk_mec_fn = getattr(src.attacks, atk_mec)
        bn = atk_mec_fn(bn_min, bn_max, rpop, exp, config)
        base_path = out_path / config["atk_path"] / f"ESS: {ess}"
        safe_open_dir(base_path)
        save_bn(
            bn,
            f"bn_{exp}_sample{sample}",
            base_path
        )

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
        bn_theta = gum.loadBN(
            f"{out_path}/{config['bns_path']}/rpop/bn_{exp}_sample{sample}.bif"
        )
        bn_theta_hat = gum.loadBN(
            f"{out_path}/{config['bns_path']}/pool/bn_{exp}_sample{sample}.bif"
        )

        bn_theta_hat_ie = gum.LazyPropagation(bn_theta_hat)
        bn_theta_ie = gum.LazyPropagation(bn_theta)

        # ... retrieve rpop, ...
        rpop = gpop[gpop[f"in-rpop-{sample}"]].iloc[:, : len(bn_theta.nodes())]

        # try:

        # ... and perform membership inference on gpop
        power_vec, _ = run_mia(
            bn_theta_hat_ie,
            bn_theta_ie,
            rpop,
            gpop,
            gpop[f"in-pool-{sample}"],
            eval(config["error"]),
        )
        results[f"power_BN_sample{sample}"] = power_vec

        # except Exception:

        #     # Debug
        #     with open(f"{results_path}/log.txt", "a") as log:
        #         log.write(f"{exp}: error with sample {sample} (BN).\n")
        #         log.write(traceback.format_exc())

    # Save results
    results.to_csv(f'{out_path}/{config["results_path"]}/bns/bn_{exp}.csv', index=False)


# MIA attack vs a CN
def phase_mia_vs_cn(exp, ess, config, save_res=True) -> dict:

    # Get output path
    out_path = get_out_path(config)

    # Init results
    results = pd.DataFrame({"error": eval(config["error"])})

    # Read data
    gpop = pd.read_csv(f'{out_path / config["data_path"]}/{exp}.csv')

    # For each data sample ...
    auc_cns = []
    for sample in range(config["n_samples"]):

        # ... read the BN as inferred from the CN
        bn_theta_hat = gum.loadBN(
            f'{out_path}/{config["atk_path"]}/ESS: {ess}/bn_{exp}_sample{sample}.bif'
        )

        # ... read the BN as estimated from rpop, ...
        bn_theta = gum.loadBN(
            f"{out_path}/{config['bns_path']}/rpop/bn_{exp}_sample{sample}.bif"
        )

        bn_theta_hat_ie = gum.LazyPropagation(bn_theta_hat)
        bn_theta_ie = gum.LazyPropagation(bn_theta)

        # ... retrieve rpop, ...
        rpop = gpop[gpop[f"in-rpop-{sample}"]].iloc[:, : len(bn_theta.nodes())]

        # try:

        # ... and perform membership inference on gpop
        power_vec, auc = run_mia(
            bn_theta_hat_ie,
            bn_theta_ie,
            rpop,
            gpop,
            gpop[f"in-pool-{sample}"],
            eval(config["error"]),
        )
        results[f"power_CN_sample{sample}"] = power_vec
        auc_cns.append(auc)

        # except Exception:

        #     # Debug
        #     with open(f"{results_path}/log.txt", "a") as log:
        #         log.write(f"{exp}: error with sample {sample} (BN).\n")
        #         log.write(traceback.format_exc())

    # Compute Avg(AUC(CN)) across data samples
    auc_cn = sum(auc_cns) / len(auc_cns)

    # Save results
    if save_res:
        results.to_csv(
            f'{out_path}/{config["results_path"]}/cns/cn_{exp}-ess{ess}.csv', index=False
        )

    return {"exp": exp, "ess": ess, "auc_cn": auc_cn}


# Get theoretical power
def phase_theoretical_power(exp, config) -> None:

    # Get output path
    out_path = get_out_path(config)

    # Read data
    bn = gum.loadBN(f'{get_out_path(config) / config["bns_path"]}/gt/{exp}.bif')
    results = pd.read_csv(f'{out_path}/{config["results_path"]}/bns/bn_{exp}.csv')

    # Compute bound
    bound = math.sqrt(bn.dim() / int(config["gpop_ss"] * config["pool_prop"]))

    # Find power (beta) for any error (alpha) given theoretical bound
    z_alpha = [norm.ppf(1 - i).item() for i in eval(config["error"])]
    z_one_minus_beta = [bound - i for i in z_alpha]
    beta = [norm.cdf(i).item() for i in z_one_minus_beta]

    # Save results
    results["power_bound"] = beta
    results.to_csv(f'{out_path}/{config["results_path"]}/bns/bn_{exp}.csv', index=False)

    return
