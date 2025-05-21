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
    gpop_ss = 2000
    rpop_ss = 1000
    pool_ss = 100
    n_ds = 10
    n_bns = 50
    error = np.logspace(-4, 0, 20, endpoint=False)
    eps_list = np.arange(1, 50, 2)
    tol = 0.01
    
    # Init local hyperp.
    exp = conf[0]
    gpop = pd.read_csv(f"./data/{exp}.csv")
    bn = gum.loadBN(f"./bns/{exp}.bif")
    n_nodes = len(bn.nodes())
    ess = conf[1].get("ess")

    # Debug
    assert(gpop_ss == gpop.shape[0])
    assert(n_nodes == gpop.shape[1])

    # Compute theoretical bound
    compl = bn.dim()
    bound = math.sqrt(compl/pool_ss)

    # Find power (beta) for any error (alpha) given theoretical bound
    z_alpha = [norm.ppf(1 - i).item() for i in error]
    z_one_minus_beta = [bound - i for i in z_alpha]
    beta = [norm.cdf(i).item() for i in z_one_minus_beta]

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
    for ds in tqdm(range(n_ds), desc="Data samples (CN)", unit="item"):

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

    # Debug
    assert(len(auc_cn_dss) == n_ds)

    # MIA (Noisy BN)
    e_best = eps_list[-1]
    for eps in tqdm(eps_list, unit="item", desc="Eps", dynamic_ncols=True):

        auc_bn_noisy_dss = []

        for ds in tqdm(range(n_ds), unit="item", desc="Data samples (BN eps)", dynamic_ncols=True, leave=False):

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
                    log.write(traceback.format_exc())

            
        # Compute Avg(AUC)
        auc_bn = sum(auc_bn_noisy_dss) / n_ds

        # Debug
        assert(len(auc_bn_noisy_dss) == n_ds)

        # Check
        if abs(auc_cn - auc_bn) <= tol:
            e_best = eps
            break

    print(f"Best eps: {e_best}, AUCs: {auc_cn:.3f} (CN), {auc_bn:.3f} (BN noisy), Diff. AUC: {abs(auc_cn - auc_bn):.3f}")   ##

    return eps

def run_inference_bn(bn):
    '''
    Notice: `bn` is assumed to be a Naive Bayes model with target variable `T`.
    '''

    # Store information
    n_nodes = bn.size()
    cov = sorted([i for i in bn.names()])
    cov.remove("T")

    # Debug
    assert(len(cov) == bn.size() - 1)

    # Create object for inference
    bn_ie = gum.LazyPropagation(bn)

    # Compute all combinations of evidence
    evid_gen = product(*((0,1) for _ in range(n_nodes - 1)))
    mpes = []
    for e in evid_gen:
        evid = dict(zip(cov, e))
        mpe = MPE_bn(bn_ie, "T", evid)
        mpes.append(mpe)
        
    # Debug
    assert(len(mpes) == 2**(len(cov)))

    return mpes

def run_inference_cn(cn):
    '''
    Notice: `cn` is assumed to be a Naive Bayes model with target variable `T`.
    '''

    # Store information
    bn = cn.current_bn()
    n_nodes = bn.size()
    cov = sorted([i for i in bn.names()])
    cov.remove("T")

    # Debug
    assert(len(cov) == bn.size() - 1)

    # Create object for inference
    cn.computeBinaryCPTMinMax()

    # Compute all combinations of evidence
    evid_gen = product(*((0,1) for _ in range(n_nodes - 1)))
    mpes = []
    probs = []
    for e in evid_gen:
        evid = dict(zip(cov, e))
        cn_ie = gum.CNLoopyPropagation(cn)
        mpe, prob = MPE_cn(cn_ie, "T", evid)
        mpes.append(mpe)
        probs.append(prob)
        
    # Debug
    assert(len(mpes) == 2**(len(cov)))
    assert(len(probs) == 2**(len(cov)))

    return mpes, probs

