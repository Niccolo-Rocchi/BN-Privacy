import pandas as pd
from scipy.stats import norm
import pyagrum as gum
from numpy import random
from tempfile import TemporaryDirectory

from src.utils import *
from src.config import *


def run_inferences(exp, eps, ess, config):

    base_path = get_base_path(config)

    # Set seed
    set_global_seed(config["seed"])

    # Set list of evidence
    evid_vec = [random_product(*((0,1) for _ in range(config["n_nodes"] - 1))) for _ in range(config["n_infer"])]

    # Store ground-truth BN
    gt = gum.loadBN(f'{base_path / config["bns_path"]}/{exp}.bif')
    gpop = pd.read_csv(f'{base_path / config["data_path"]}/{exp}.csv')

    # Learn BN from gpop
    bn_learner=gum.BNLearner(gpop)
    bn_learner.useSmoothingPrior(1e-5)
    bn = bn_learner.learnParameters(gt.dag())

    # Learn CN from gpop
    bn_copy = gum.BayesNet(bn)
    add_counts_to_bn(bn_copy, gpop)
    cn = gum.CredalNet(bn_copy)
    cn.idmLearning(ess)

    # Learn noisy BN from gpop
    scale = (2 * bn.size()) / (len(gpop) * eps)
    bn_noisy = get_noisy_bn(bn, scale)

    # Run inferences
    gt_mpes, _ = run_inference_bn(gt, evid_vec)
    bn_mpes, bn_probs = run_inference_bn(bn, evid_vec)
    bn_noisy_mpes, bn_noisy_probs = run_inference_bn(bn_noisy, evid_vec)
    cn_mpes, cn_probs, cn_probs_alt = run_inference_cn(cn, evid_vec, exp)

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
            "cn_probs_alt": cn_probs_alt
        }
    )

    dir = base_path / config["results_path"] / f'results_nodes{config["n_nodes"]}_ess{ess}'
    results.to_csv(f"{dir}/{exp}.csv", index = False)

# MPE function for BN
def MPE_bn(bn_ie: gum.LazyPropagation, target: str, evid: dict) -> tuple:

    # Set evidence
    bn_ie.setEvidence(evid)

    # Compute MPE and log(prob)
    out = bn_ie.mpeLog2Posterior()
    mpe = out[0].todict().get(target)
    prob = np.exp2(out[1])

    return mpe, prob

# MPE function for CN
def MPE_cn(bn_min:gum.BayesNet, bn_max:gum.BayesNet, T: str, children: dict) -> tuple:

    '''
    Get the MPE of a CN as: argmax_t log P_lower(T=t | children), together with its lower probability. 
    bn_min and bn_max derive from a binary CN.
    The DAG is assumed to be a Naive Bayes model with T the target variable.
    Return the MPE, its probability, and the lower probability of the alternative class.
    '''    

    lp1 = get_lower_posterior(bn_min, bn_max, T, 1, children)
    lp0 = get_lower_posterior(bn_min, bn_max, T, 0, children)

    if lp1 > lp0:
        return (1, math.exp(lp1), math.exp(lp0))
    else:
        return (0, math.exp(lp0), math.exp(lp1))
    
# Get a value from a BN's CPT
def get_cond(bn: gum.BayesNet, X: str, x: float, parents: dict = None) -> float:

    '''
    Get P(X=x | parents) from the BN's CPT of X.
    '''

    cpt = bn.cpt(X)
    inst = gum.Instantiation(cpt)
    inst[X] = x 
    if not parents:
        assert(len(bn.parents(X)) == 0)
        pass
    else:
        assert(bn.parents(X) == set(bn.ids(parents.keys())))
        for var in parents.keys():
            inst[var] = parents[var]
            
    return max(cpt.get(inst), 1e-10)    # Smoothing
    
# Get a Naive Bayes log-joint
def get_NaiveBayes_log_joint(bn: gum.BayesNet, T: str, t: float, children: dict) -> float:
    
    '''
    Get log[P(T=t, children)] from the BN's CPT of T.
    The BN is assumed to be a Naive Bayes model with T the target variable.
    '''

    sum_log = 0
    for var, val in children.items():
        sum_log += math.log(get_cond(bn, var, val, {T:t}))
    sum_log += math.log(get_cond(bn, T, t))

    return sum_log

# Get the lower posterior from a CN
def get_lower_posterior(bn_min:gum.BayesNet, bn_max:gum.BayesNet, T: str, t: float, children: dict) -> float:
    
    '''
    Get log P_lower(T=t | children). 
    bn_min and bn_max derive from a binary CN.
    The DAG is assumed to be a Naive Bayes model with T the target variable.
    '''

    lp_lower = get_NaiveBayes_log_joint(bn_min, T, t, children)
    lp_upper = get_NaiveBayes_log_joint(bn_max, T, 1-t, children)

    return lp_lower - lp_upper - math.log1p(math.exp(lp_lower - lp_upper))

# Run inferences on a BN
def run_inference_bn(bn, evid_vec):
    '''
    The BN is assumed to be a Naive Bayes model with T the target variable.
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
    probs = []
    for e in evid_vec:
        evid = dict(zip(cov, e))
        mpe, prob = MPE_bn(bn_ie, "T", evid)
        mpes.append(mpe)
        probs.append(prob)
        
    # Debug
    assert(len(mpes) == len(evid_vec))
    assert(len(probs) == len(evid_vec))

    return mpes, probs

# Run inferences on a CN
def run_inference_cn(cn, evid_vec, exp: str):

    '''
    The CN is assumed to be a Naive Bayes model with T the target variable.
    '''

    # Store information
    with TemporaryDirectory() as tmp_path:
        cn.saveBNsMinMax(f"{tmp_path}/bn_min_{exp}.bif", f"{tmp_path}/bn_max_{exp}.bif")
        bn_min = gum.loadBN(f"{tmp_path}/bn_min_{exp}.bif")
        bn_max = gum.loadBN(f"{tmp_path}/bn_max_{exp}.bif")
    cov = sorted([i for i in bn_min.names()])
    cov.remove("T")

    # Debug
    assert(len(cov) == bn_min.size() - 1)

    # Compute all combinations of evidence
    mpes = []
    probs = []
    probs_alt = []
    for e in evid_vec:
        evid = dict(zip(cov, e))
        mpe, prob, prob_alt = MPE_cn(bn_min, bn_max, "T", evid)
        mpes.append(mpe)
        probs.append(prob)
        probs_alt.append(prob_alt)
        
    # Debug
    assert(len(mpes) == len(evid_vec))
    assert(len(probs) == len(evid_vec))
    assert(len(probs_alt) == len(evid_vec))

    return mpes, probs, probs_alt