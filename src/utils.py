import io
import re
from collections import defaultdict
from contextlib import redirect_stdout

import numpy as np
import pyagrum as gum
from more_itertools import random_product
from numpy import random
from numpy.random import random_sample


# Log-likelihood function
def get_ll(x: dict, theta):

    # Erase all evidences and apply addEvidence(key,value) for every pairs in x
    theta.setEvidence(x)

    # Compute P(x | theta)
    ll = theta.evidenceProbability()

    return np.log(ll)


# Log-likelihood ratio (llr) function
def get_llr(x: dict, theta, theta_hat):

    # Compute log-likelihoods
    ll_theta = get_ll(x, theta)
    ll_theta_hat = get_ll(x, theta_hat)

    return ll_theta_hat - ll_theta


# Parse the credal network
def parse_cn(cn) -> tuple:

    # Get the DAG
    dag = gum.BayesNet(cn.current_bn())

    # Cast CN to string
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print(cn)
    cn_str = buffer.getvalue()

    credal_dict = defaultdict(lambda: defaultdict(list))
    current_var = None

    lines = cn_str.strip().split("\n")

    for line in lines:
        line = line.strip()

        # Variable identification
        var_match = re.match(r"^([A-Za-z0-9_]+):", line)
        if var_match:
            current_var = var_match.group(1)
            continue

        if current_var is None or not line:
            continue

        # CPT identification
        cpt_match = re.match(r"^<([^>]*)>\s*:\s*(.*)", line)
        if cpt_match:
            condition = f"<{cpt_match.group(1).strip()}>"
            raw_cpt = cpt_match.group(2)

            # Extraction of inner lists: [[x,x,x], [x,x,x], ...]
            vectors = re.findall(r"\[\s*([^\[\]]+?)\s*\]", raw_cpt)
            for vec in vectors:
                prob_vec = [float(x.strip()) for x in vec.split(",")]
                credal_dict[current_var][condition].append(prob_vec)

    params = []
    for var in credal_dict:
        for cond, vectors in credal_dict[var].items():
            params.append((var, cond, vectors))

    return dag, params


# Compute a random subset of BNs from the CN
def sample_from_cn(cn, n: int, where: str) -> list:
    """
    Sample random BNs from the CN.

    Parameters:
    - `cn`: the given CN.
    - `n`: number of BNs to extract from the CN.
    - `where`: can be `inside` or `outside`.
        "inside": the BNs are taken from within the credal set;
        "outside": the BNs are vertices of the credal set.
    """

    random.seed(42)

    # Parse CN
    dag, params = parse_cn(cn)

    # Store variables indexes
    var_idx = {
        var: [idx for idx, elem in enumerate(params) if elem[0] == var]
        for var in dag.names()
    }

    # Cases
    if where == "inside":
        sample = sample_inside
    elif where == "outside":
        sample = sample_outside
    else:
        msg = "'where' can be either 'inside' or 'outside'"
        print(msg)
        raise ValueError(msg)

    # Draw n random BNs
    k = 0
    bns = []
    while k < n:

        # Init an empty BN
        bn = gum.BayesNet(dag)

        # Sample from CN
        next_sample = next(sample(params))

        # Fill the BN's CPTs
        for var in dag.names():
            array = np.array([(next_sample[idx]) for idx in var_idx.get(var)]).flatten()
            bn.cpt(var).fillWith(array)

        bns.append(bn)
        k += 1

    # Debug
    # assert(n == len(bns))

    return bns


# Given a parsed CN called `params`, sample a BN inside the credal set
def sample_inside(params):

    p_1 = [
        (vecs[0][0] - vecs[1][0]) * random_sample() + vecs[1][0]
        for _, _, vecs in params
    ]
    p = [[x, 1 - x] for x in p_1]

    # Debug
    # assert(np.sum(np.array(p), axis=1).all() == 1.)

    yield p


# Given a parsed CN called `params`, sample a vertex of the credal set
def sample_outside(params):

    yield random_product(*[vecs for _, _, vecs in params])


# Check BNs sampled from a CN
def are_all_bns_different(bn_vec) -> None:

    signatures = set()
    for bn in bn_vec:
        cpt_data = []
        for var in bn.names():
            cpt = bn.cpt(var)
            flat = [f"{v:.8f}" for v in cpt.toarray().flatten()]
            cpt_data.append(f"{var}:" + ",".join(flat))
        sig = "|".join(cpt_data)
        signatures.add(sig)

    print(f"({len(signatures)}/{len(bn_vec)} different BNs.)")


# Add counts of events to a BN
def add_counts_to_bn(bn, data):

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


# Compact a dictionary to be printable
def compact_dict(d):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            new_dict[k] = (
                f"np.ndarray: [{v[0]:.2g}, {v[1]:.2g}, ..., {v[-1]:.2g}], length={len(v)}"
            )
        else:
            new_dict[k] = v
    return new_dict


# Create noisy BN by adding Laplacian noise (Zhang et al., 2017)
def get_noisy_bn(bn, scale: float):

    bn_ie = gum.LazyPropagation(bn)
    bn_ie.makeInference()

    bn_noisy = gum.BayesNet(bn)

    # For each node X ...
    for node in bn.names():

        # Get the joint P(X, Pa(X))
        joint = bn_ie.jointPosterior(bn.family(node))

        # Add noise to P(X, Pa(X)) and normalize
        noise = np.random.laplace(scale=scale, size=np.prod(joint.shape))
        noisy_joint = np.clip(
            joint.toarray().flatten() + noise, a_min=10e-10, a_max=None
        )
        noisy_joint = noisy_joint / np.sum(noisy_joint)
        joint.fillWith(noisy_joint)

        # Compute the conditional P(X | Pa(X))
        cond = joint / joint.sumOut(node)

        # Fill noisy BN
        bn_noisy.cpt(node).fillWith(cond)

    # Check noisy bn
    bn_noisy.check()  # OK if = ().

    return bn_noisy
