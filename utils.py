# Libraries
import numpy as np
import pandas as pd
import math
from scipy.stats import norm
import pyagrum as gum
from pathlib import Path
from pandarallel import pandarallel
from collections import defaultdict
import numpy as np
from numpy.random import random_sample
import io
import re
from contextlib import redirect_stdout
from itertools import product
from more_itertools import random_product
from tqdm import tqdm
import warnings

# Log-likelihood ratio (LLR) function
def LLR(x: dict, theta, theta_hat):

    # Erase all evidences and apply addEvidence(key,value) for every pairs in x
    theta.setEvidence(x)
    theta_hat.setEvidence(x)

    # Compute P(x | BN)
    L_theta = theta.evidenceProbability()
    L_theta_hat = theta_hat.evidenceProbability()

    # Check for denominator
    if L_theta_hat < 1e-16:
        return np.inf
    
    # Debug
    # assert(theta.nbrHardEvidence() == len(bn.nodes()))
    # assert(theta_hat.nbrHardEvidence() == len(bn.nodes()))

    return math.log(L_theta / L_theta_hat)

# Parse the credal network (TODO: improve)
def parse_credal_net(cn_str: str):
    credal_dict = defaultdict(lambda: defaultdict(list))
    current_var = None

    lines = cn_str.strip().split('\n')

    for line in lines:
        line = line.strip()

        # Identificazione della variabile
        var_match = re.match(r'^([A-Za-z0-9_]+):Range\(\[.*\]\)', line)
        if var_match:
            current_var = var_match.group(1)
            continue

        if current_var is None or not line:
            continue

        # Identificazione di una CPT con intestazione <condizioni>
        cpt_match = re.match(r'^<([^>]*)>\s*:\s*(.*)', line)
        if cpt_match:
            condition = f"<{cpt_match.group(1).strip()}>"
            raw_cpt = cpt_match.group(2)

            # Estrarre tutte le liste interne: [[x,x,x], [x,x,x], ...]
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
        p_1 = [(vecs[0][0] - vecs[1][0]) * random_sample() + vecs[1][0] for _, _, vecs in slots]
        p = [[x, 1-x] for x in p_1]

        # Debug
        # assert(np.sum(np.array(p), axis=1).all() == 1.)

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