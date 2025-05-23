import math
import warnings
import traceback
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn import metrics
import pyagrum as gum
from tqdm import tqdm

from utils import *

warnings.filterwarnings('ignore')

# Run local IDM experiment
def run_idm(conf):

    # Init global hyperp.
    n_ds = 10
    n_bns = 50
    error = np.logspace(-4, 0, 20, endpoint=False)
    tol = 0.01
    eps_list = conf[2]
    
    # Init local hyperp.
    exp = conf[0]
    gpop = pd.read_csv(f"./data/{exp}.csv")
    bn = gum.loadBN(f"./bns/{exp}.bif")
    n_nodes = len(bn.nodes())
    ess = conf[1].get("ess")
    gpop_ss = len(gpop)
    rpop_ss = gpop_ss // 2
    pool_ss = gpop_ss // 4

    # Debug
    assert(gpop_ss == gpop.shape[0])
    assert(n_nodes == gpop.shape[1])

    # Store information
    bn_theta_dss = []
    bn_theta_hat_dss = []
    cn_dss = []
    for ds in range(n_ds):

        # Sample pool and rpop
        pool_idx = np.random.choice(range(gpop_ss), size=pool_ss, replace=False)
        gpop[f"in-pool-{ds}"] = gpop.index.isin(pool_idx)
        pool = gpop[gpop[f"in-pool-{ds}"]].iloc[:, :n_nodes]
        rpop = gpop[~gpop[f"in-pool-{ds}"]].iloc[:, :n_nodes].sample(rpop_ss)

        # Estimate BN(theta) from rpop
        theta_learner=gum.BNLearner(rpop)
        theta_learner.useSmoothingPrior(1e-5)
        bn_theta = theta_learner.learnParameters(bn.dag())

        # Estimate BN(theta_hat) from pool
        theta_hat_learner=gum.BNLearner(pool)
        theta_hat_learner.useSmoothingPrior(1e-5)
        bn_theta_hat = theta_hat_learner.learnParameters(bn.dag())

        # Estimate CN by local IDM from BN(theta_hat)
        bn_copy = gum.BayesNet(bn)
        add_counts(bn_copy, pool)
        cn = gum.CredalNet(bn_copy)
        cn.idmLearning(ess)

        # Save nets
        bn_theta_dss.append(bn_theta)
        bn_theta_hat_dss.append(bn_theta_hat)
        cn_dss.append(cn)

        # Debug
        assert(len(pool) == sum(gpop[f"in-pool-{ds}"]))
        assert(len(pool) == pool_ss)
        assert(len(rpop) == rpop_ss)

    # Debug
    assert(len(bn_theta_dss) == n_ds)
    assert(len(bn_theta_hat_dss) == n_ds)
    assert(len(cn_dss) == n_ds)

    # MIA (CN)
    auc_cn_dss = []
    for ds in range(n_ds):  # tqdm(range(n_ds), desc="Data samples (CN)", unit="item")

        # Retrieve ds-related info
        y_true = gpop[f"in-pool-{ds}"]
        pos = len(pool)
        cn = cn_dss[ds]
        bn_theta_ie = gum.LazyPropagation(bn_theta_dss[ds])

        # Extract random subset within simplex
        bns_sample = get_simplex_inner(cn, n_bns)

        try:

            # MIA 
            best_sum = -np.inf
            best_bn_idx = 0
            for i, bn_s in enumerate(bns_sample):

                # Estimate the likelihood of rpop
                bn_s_ie = gum.LazyPropagation(bn_s)
                L_im_s = rpop.apply(lambda x: LL(x.to_dict(), bn_s_ie), axis=1).dropna()

                L_sum = np.sum(L_im_s)
                if L_sum > best_sum: 
                    best_sum = L_sum
                    best_bn_idx = i

            # Get best BN inside simplex
            bn_ie = gum.LazyPropagation(bns_sample[best_bn_idx])
            L_im = rpop.apply(lambda x: LLR(x.to_dict(), bn_theta_ie, bn_ie), axis=1).dropna()

            # Compute LLR(x) on general population
            llr_gpop = gpop[[*bn.names()]].apply(lambda x: LLR(x.to_dict(), bn_theta_ie, bn_ie), axis=1)

            # Init the power vector (CN)
            power_cn = []

            # For each error ...
            for e in error:

                # Compute threshold
                t = np.quantile(L_im, 1-e).item()

                # LLR test. Reject H_0? True => x in pool; False => x in rpop.
                y_pred_s = llr_gpop > t
                
                # Compute and store power (tpr)
                tpr = sum(y_true & y_pred_s) / pos
                power_cn = power_cn + [tpr]

            # Compute and store AUC
            auc = metrics.auc(error, power_cn)
            auc_cn_dss.append(auc)

        except:
            
            with open("./results/log.txt", "a") as log: 
                log.write(f"{exp}: error with sample {ds} (CN).\n")
                # log.write(traceback.format_exc())

            
    # Compute Avg(AUC)
    auc_cn = sum(auc_cn_dss) / len(auc_cn_dss)

    # MIA (Noisy BN)
    e_best = eps_list[-1]
    for eps in eps_list:    # tqdm(eps_list, unit="item", desc="Eps", dynamic_ncols=True)

        auc_bn_noisy_dss = []

        for ds in range(n_ds):  # tqdm(range(n_ds), unit="item", desc="Data samples (BN eps)", dynamic_ncols=True, leave=False)

            # Retrieve ds-related info
            y_true = gpop[f"in-pool-{ds}"]
            pos = len(pool)
            bn_theta_hat = bn_theta_hat_dss[ds]
            bn_theta_ie = gum.LazyPropagation(bn_theta_dss[ds])

            # Get noisy BN
            scale = (2 * bn_theta_hat.size()) / (len(pool) * eps)
            bn_noisy = get_noisy_bn(bn_theta_hat, scale)

            try:                

                # MIA
                bn_noisy_ie = gum.LazyPropagation(bn_noisy)
                L_im = rpop.apply(lambda x: LLR(x.to_dict(), bn_theta_ie, bn_noisy_ie), axis=1).dropna().sort_values() 

                # Compute LLR(x) on general population
                llr_gpop = gpop[[*bn.names()]].apply(lambda x: LLR(x.to_dict(), bn_theta_ie, bn_noisy_ie), axis=1)

                # Init the power vector (BN)
                power_bn_noisy = []

                # For each error ...
                for e in error:

                    # Compute threshold
                    t = np.quantile(L_im, 1-e).item()

                    # LLR test. Reject H_0? True => x in pool; False => x in rpop.
                    y_pred = llr_gpop > t

                    # Compute and store power (tpr)
                    tpr = sum(y_true & y_pred) / pos
                    power_bn_noisy = power_bn_noisy + [tpr]

                # Compute and store AUC
                auc = metrics.auc(error, power_bn_noisy)
                auc_bn_noisy_dss.append(auc)

            except:

                with open("./results/log.txt", "a") as log: 
                    log.write(f"{exp}: error with sample {ds} (BN noisy, eps: {eps}).\n")
                    # log.write(traceback.format_exc())

            
        # Compute Avg(AUC)
        auc_bn = sum(auc_bn_noisy_dss) / n_ds

        # Check
        if abs(auc_cn - auc_bn) <= tol:
            e_best = eps
            break

    with open("./results/exp_meta.txt", "a") as m: 
        m.write(f"- {exp}. Nodes: {n_nodes} Eps: {eps}\n")

    return exp, e_best

def run_inference_bn(bn, evid_list):
    '''
    Notice: `bn` is assumed to be a Naive Bayes model with target variable `T`.
    '''

    # Store information
    cov = sorted([i for i in bn.names()])
    cov.remove("T")

    # Debug
    assert(len(cov) == bn.size() - 1)

    # Create object for inference
    bn_ie = gum.LazyPropagation(bn)

    # Compute all combinations of evidence
    mpes = []
    for e in evid_list:
        evid = dict(zip(cov, e))
        mpe = MPE_bn(bn_ie, "T", evid)
        mpes.append(mpe)
        
    # Debug
    assert(len(mpes) == len(evid_list))

    return mpes

def run_inference_cn(cn, evid_list):
    '''
    Notice: `cn` is assumed to be a Naive Bayes model with target variable `T`.
    '''

    # Store information
    bn = cn.current_bn()
    cov = sorted([i for i in bn.names()])
    cov.remove("T")

    # Debug
    assert(len(cov) == bn.size() - 1)

    # Create object for inference
    cn.computeBinaryCPTMinMax()

    # Compute all combinations of evidence
    mpes = []
    probs = []
    for e in evid_list:
        evid = dict(zip(cov, e))
        cn_ie = gum.CNLoopyPropagation(cn)
        mpe, prob = MPE_cn(cn_ie, "T", evid)
        mpes.append(mpe)
        probs.append(prob)
        
    # Debug
    assert(len(mpes) == len(evid_list))
    assert(len(probs) == len(evid_list))

    return mpes, probs


def run_inferences(exp, eps, ess, evid_list):

    # Store GT BN
    gt = gum.loadBN(f"./bns/{exp}.bif")
    gpop = pd.read_csv(f"./data/{exp}.csv")

    # Learn BN
    bn_learner=gum.BNLearner(gpop)
    bn_learner.useSmoothingPrior(1e-5)
    bn = bn_learner.learnParameters(gt.dag())

    # Learn CN
    bn_copy = gum.BayesNet(bn)
    add_counts(bn_copy, gpop)
    cn = gum.CredalNet(bn_copy)
    cn.idmLearning(ess)

    # Learn noisy BN 
    scale = (2 * bn.size()) / (len(gpop) * eps)
    bn_noisy = get_noisy_bn(bn, scale)

    # Run inferences
    gt_mpes = run_inference_bn(gt, evid_list)
    bn_mpes = run_inference_bn(bn, evid_list)
    bn_noisy_mpes = run_inference_bn(bn_noisy, evid_list)
    cn_mpes, cn_probs = run_inference_cn(cn, evid_list)

    # Save results
    results = pd.DataFrame(
        {
            "gt_mpes": gt_mpes, 
            "bn_mpes": bn_mpes,
            "bn_noisy_mpes": bn_noisy_mpes,
            "cn_mpes": cn_mpes, 
            "cn_probs": cn_probs
        }
    )

    results.to_csv(f"results/{exp}-eps{eps}.csv", index = False)



