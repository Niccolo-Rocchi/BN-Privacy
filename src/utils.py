import numpy as np
import math
import pyagrum as gum
from collections import defaultdict
import numpy as np
from numpy.random import random_sample
import io
import re
from contextlib import redirect_stdout
from itertools import product
from more_itertools import random_product

# Log-likelihood function
def LL(x:dict, theta):

    # Erase all evidences and apply addEvidence(key,value) for every pairs in x
    theta.setEvidence(x)

    # Compute P(x | theta)
    ll = theta.evidenceProbability()

    return np.log(ll)

# Log-likelihood ratio (LLR) function
def LLR(x: dict, theta, theta_hat):

    # Compute log-likelihoods
    ll_theta = LL(x, theta)
    ll_theta_hat = LL(x, theta_hat)

    return ll_theta_hat - ll_theta

# Parse the credal network
def parse_credal_net(cn_str: str):
    credal_dict = defaultdict(lambda: defaultdict(list))
    current_var = None

    lines = cn_str.strip().split('\n')

    for line in lines:
        line = line.strip()

        # Variable identification
        var_match = re.match(r'^([A-Za-z0-9_]+):', line)
        if var_match:
            current_var = var_match.group(1)
            continue

        if current_var is None or not line:
            continue

        # CPT identification
        cpt_match = re.match(r'^<([^>]*)>\s*:\s*(.*)', line)
        if cpt_match:
            condition = f"<{cpt_match.group(1).strip()}>"
            raw_cpt = cpt_match.group(2)

            # Extraction of inner lists: [[x,x,x], [x,x,x], ...]
            vectors = re.findall(r'\[\s*([^\[\]]+?)\s*\]', raw_cpt)
            for vec in vectors:
                prob_list = [float(x.strip()) for x in vec.split(',')]
                credal_dict[current_var][condition].append(prob_list)

    return credal_dict

# Compute the vertices of CN simplex or a random subset of them
def get_simplex(cn, n: int = None) -> list:
    
    '''
    Returns a full (or random) list of BNs laying on the vertices 
    of the simplex, i.e. the credal set.
    '''

    # Store the CN in form of string
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print(cn)
    cn_text = buffer.getvalue()

    # Parse CN
    parsed = parse_credal_net(cn_text)

    # Get baseline DAG and init simplex
    dag = gum.BayesNet(cn.current_bn())
    bns = []

    # Compute slots and store variables indexes
    slots = []
    for var in parsed:
        for cond, vectors in parsed[var].items():
            slots.append((var, cond, vectors))
    var_idx = {var:[idx for idx, elem in enumerate(slots) if elem[0] == var] for var in dag.names()}

    # If 'n' is provided...
    if bool(n):
        # Get 'n' random combinations of CPTs
        combinations = [random_product(*[vecs for _, _, vecs in slots]) for _ in range(n)]
        n_combs = len(combinations)
        assert(n_combs == n)
    else:
        # Get all combinations of CPTs
        combinations = list(product(*[vecs for _, _, vecs in slots]))
        n_combs = len(combinations)

    # For each combination...
    for combo in combinations:

        # Init BN and ...
        bn_tmp = gum.BayesNet(dag)

        # Fill its CPTs
        for var in dag.names():
            array = np.array([(combo[idx]) for idx in var_idx.get(var)]).flatten()
            bn_tmp.cpt(var).fillWith(array)

        bns.append(bn_tmp)
    
    # Debug
    # assert(n_combs == len(bns))

    return bns

# Compute a random subset of BNs withing the simplex
def get_simplex_inner(cn, n: int) -> list:
    
    '''
    Returns a random list of BNs laying within 
    the simplex, i.e. the credal set.
    '''

    # Store the CN in form of string
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print(cn)
    cn_text = buffer.getvalue()

    # Parse CN
    parsed = parse_credal_net(cn_text)

    # Get baseline DAG and init simplex
    dag = gum.BayesNet(cn.current_bn())
    bns = []

    # Compute slots and store variables indexes
    slots = []
    for var in parsed:
        for cond, vectors in parsed[var].items():
            slots.append((var, cond, vectors))
    var_idx = {var:[idx for idx, elem in enumerate(slots) if elem[0] == var] for var in dag.names()}

    # Define BN generator
    def gen_bn(slots):
        try:
            p_1 = [(vecs[0][0] - vecs[1][0]) * random_sample() + vecs[1][0] for _, _, vecs in slots]
        except IndexError:
            p_1 = [vecs[0][0] for _, _, vecs in slots]

        p = [[x, 1-x] for x in p_1]

        # Debug
        assert(np.sum(np.array(p), axis=1).all() == 1.)

        yield p

    # Draw n random BNs
    k = 0
    while k < n:

        # Draw a random BN within the simplex
        combo = next(gen_bn(slots))

        # Init BN and ...
        bn_tmp = gum.BayesNet(dag)

        # Fill its CPTs
        for var in dag.names():
            array = np.array([(combo[idx]) for idx in var_idx.get(var)]).flatten()
            bn_tmp.cpt(var).fillWith(array)

        bns.append(bn_tmp)
        k += 1
    
    # Debug
    # assert(n_combs == len(bns))

    return bns

# Check simplex computation
def are_all_bn_different(bn_list):

    def serialize_bn(bn):
        cpt_data = []
        for var in bn.names():
            cpt = bn.cpt(var)
            flat = [f"{v:.8f}" for v in cpt.toarray().flatten()]
            cpt_data.append(f"{var}:" + ",".join(flat))
        return "|".join(cpt_data)

    signatures = set()
    for bn in bn_list:
        sig = serialize_bn(bn)
        signatures.add(sig)
        
    print(f"({len(signatures)}/{len(bn_list)} different BNs.)")

    return

# Add counts of events to a BN
def add_counts(bn, data):

    for node in bn.names():
        var = bn.variable(node)
        parents = bn.parents(node)
        parent_names = [bn.variable(p).name() for p in parents]

        shape = [bn.variable(p).domainSize() for p in parents] + [var.domainSize()]
        counts_array = np.zeros(shape, dtype=float)  # float, not int

        for _, row in data.iterrows():
            try:
                key = tuple([int(row[p]) for p in parent_names] + [int(row[node])])
                counts_array[key] += 1.0
            except KeyError:
                continue

        bn.cpt(node).fillWith(counts_array.flatten().tolist())
        
    return

# Create noisy bn (Zhang et al., 2017)
def get_noisy_bn(bn, scale: float):

    bn_ie = gum.LazyPropagation(bn)
    bn_ie.makeInference()

    bn_noisy = gum.BayesNet(bn)

    # For each node ...
    for node in bn.names():

        # Get the joint P(X, Pa(X))
        joint = bn_ie.jointPosterior(bn.family(node))

        # Add noise to P(X, Pa(X)) and normalize
        noise = np.random.laplace(scale=scale, size=np.prod(joint.shape))
        noisy_joint = np.clip(joint.toarray().flatten() + noise, a_min=10e-10, a_max=None)
        noisy_joint = noisy_joint / np.sum(noisy_joint)
        joint.fillWith(noisy_joint)

        # Compute the conditional P(X | Pa(X))
        cond = joint / joint.sumOut(node)

        # Fill noisy BN
        bn_noisy.cpt(node).fillWith(cond)

    # Check noisy bn
    bn_noisy.check()    # OK if = ().

    return bn_noisy

# MPE function for BN
def MPE_bn(bn_ie: gum.LazyPropagation, target: str, evid: dict) -> tuple:

    # Set evidence
    bn_ie.setEvidence(evid)

    # Compute MPE and log(prob)
    out = bn_ie.mpeLog2Posterior()
    mpe = out[0].todict().get(target)
    prob = np.exp2(out[1])

    return mpe, prob

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
            
    return cpt.get(inst)

# Get a Naive Bayes log-joint
def get_NB_log_joint(bn: gum.BayesNet, T: str, t: float, children: dict) -> float:
    
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

    lp_lower = get_NB_log_joint(bn_min, T, t, children)
    lp_upper = get_NB_log_joint(bn_max, T, 1-t, children)

    return lp_lower - lp_upper - math.log1p(math.exp(lp_lower - lp_upper))

# MPE function for CN (without using pyagrum)
def MPE_cn(bn_min:gum.BayesNet, bn_max:gum.BayesNet, T: str, children: dict) -> tuple:

    '''
    Get the MPE of a CN as: argmax_t log P_lower(T=t | children), together with its lower probability. 
    bn_min and bn_max derive from a binary CN.
    The DAG is assumed to be a Naive Bayes model with T the target variable.
    '''

    lp1 = get_lower_posterior(bn_min, bn_max, T, 1, children)
    lp0 = get_lower_posterior(bn_min, bn_max, T, 0, children)

    if lp1 > lp0:
        return (1, lp1)
    else:
        return (0, lp0)
    
# MPE function for CN (using pyagrum)
def MPE_cn_pyagrum(cn_ie: gum.CNLoopyPropagation, target: str, evid: dict):

    # Set evidence
    cn_ie.setEvidence(evid)
    
    # Compute MPE
    cn_ie.makeInference()
    marg = cn_ie.marginalMin(target).argmax()
    mpe = marg[0][0].get(target)
    prob = marg[1]

    return mpe, prob

# Run inferences on a BN (by using pyagrum)
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
    probs = []
    for e in evid_list:
        evid = dict(zip(cov, e))
        mpe, prob = MPE_bn(bn_ie, "T", evid)
        mpes.append(mpe)
        probs.append(prob)
        
    # Debug
    assert(len(mpes) == len(evid_list))
    assert(len(probs) == len(evid_list))

    return mpes, probs

# Run inferences on a CN (without using pyagrum)
def run_inference_cn(cn, evid_list):

    '''
    Notice: `cn` is assumed to be a binary Naive Bayes model with target variable `T`.
    '''

    # Store information
    cn.saveBNsMinMax("bn_min.bif", "bn_max.bif")
    bn_min = gum.loadBN("bn_min.bif")
    bn_max = gum.loadBN("bn_max.bif")
    cov = sorted([i for i in bn_min.names()])
    cov.remove("T")

    # Debug
    assert(len(cov) == bn_min.size() - 1)

    # Compute all combinations of evidence
    mpes = []
    probs = []
    for e in evid_list:
        evid = dict(zip(cov, e))
        mpe, prob = MPE_cn(bn_min, bn_max, "T", evid)
        mpes.append(mpe)
        probs.append(prob)
        
    # Debug
    assert(len(mpes) == len(evid_list))
    assert(len(probs) == len(evid_list))

    return mpes, probs

# Run inferences on a CN (by using pyagrum)
def run_inference_cn_pyagrum(cn, evid_list):
    
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
        mpe, prob = MPE_cn_pyagrum(cn_ie, "T", evid)
        mpes.append(mpe)
        probs.append(prob)
        
    # Debug
    assert(len(mpes) == len(evid_list))
    assert(len(probs) == len(evid_list))

    return mpes, probs
            
