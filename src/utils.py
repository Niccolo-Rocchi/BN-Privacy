import sys
from math import prod
from tempfile import TemporaryDirectory

import hopsy
import numpy as np
import pyagrum as gum

IN_PYTEST = "pytest" in sys.modules


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
def noisy_bn(bn, scale: float):

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


# Only perform an `assert` if code is running in `pytest`
def safe_assert(condition):
    if IN_PYTEST:
        assert condition


# Extract BN min and BN max from a CN
def get_min_max_bns(cn, exp: str):

    with TemporaryDirectory() as tmp_path:
        cn.saveBNsMinMax(f"{tmp_path}/bn_min_{exp}.bif", f"{tmp_path}/bn_max_{exp}.bif")
        bn_min = gum.loadBN(f"{tmp_path}/bn_min_{exp}.bif")
        bn_max = gum.loadBN(f"{tmp_path}/bn_max_{exp}.bif")

    return bn_min, bn_max


# Sample from a credal set K(x | pi_x), i.e., a constrained polytope.
def sample_from_cset(vec_min, vec_max):
    """
    A credal set is a polytope in a space of #X parameters, defined by a:
     - Multi-dimensional rectangle, i.e., inequality constraint Ax <= b, and
     - Hyperplane (provided all the variables sum up to 1), i.e., equality constraint A_eq x = b_eq.
    """

    # Define the rectangle
    n_par = len(vec_min)
    A = np.concat((np.eye(n_par), -np.eye(n_par)), axis=0)
    b = np.array(np.concatenate((vec_max, -vec_min)))
    rectangle = hopsy.Problem(A=A, b=b)

    # Define the hyperplane
    A_eq = np.array([np.ones(n_par)])
    b_eq = np.array([1.0])

    # Define the polytope as a constrained rectangle
    constrained_rectangle = hopsy.add_equality_constraints(
        rectangle, A_eq=A_eq, b_eq=b_eq
    )

    # Sample from the polytope
    mc = hopsy.MarkovChain(constrained_rectangle)
    rng = hopsy.RandomNumberGenerator(42)
    _, constrained_samples = hopsy.sample(mc, rng, n_samples=1, thinning=10)
    constrained_samples = constrained_samples.flatten()

    # Debug
    safe_assert(np.all(vec_min <= vec_max))
    safe_assert(n_par == len(vec_max))
    safe_assert(n_par == A.shape[1])
    safe_assert(n_par == A_eq.shape[1])
    safe_assert(n_par == len(constrained_samples))

    return constrained_samples


# Sample from two esxtreme CPTs
def sample_from_cpts(cpt_min, cpt_max) -> np.array:

    # Transform CPTs into pandas dataframes
    cpt_min = np.atleast_2d(cpt_min.topandas())
    cpt_max = np.atleast_2d(cpt_max.topandas())

    # Sample conditional distributions
    cpt_sample = []
    for row in range(cpt_min.shape[0]):

        vec_min = cpt_min[row, :]
        vec_max = cpt_max[row, :]

        # Sample from polytope
        vec_sample = sample_from_cset(vec_min, vec_max)
        cpt_sample.append(vec_sample)

    cpt_sample = np.array(cpt_sample).flatten()

    # Debug
    safe_assert(cpt_min.shape == cpt_max.shape)
    safe_assert(cpt_min.shape == cpt_max.shape)
    safe_assert(prod(cpt_min.shape) == prod(cpt_max.shape))
    safe_assert(len(cpt_sample) == prod(cpt_min.shape))

    return cpt_sample


# BNs sampler from a CN
def sample_from_cn(cn, exp: str, n: int):

    # Get the DAG and extreme BNs
    dag = gum.BayesNet(cn.current_bn())
    bn_min, bn_max = get_min_max_bns(cn, exp)

    # Draw n random BNs
    bns = []
    for _ in range(n):

        # Init an empty BN
        bn = gum.BayesNet(dag)

        # For each variable ...
        for var in dag.names():

            # ... sample from the CN CPT, ...
            cpt_sample = sample_from_cpts(bn_min.cpt(var), bn_max.cpt(var))

            # ... and fill the BN's CPT
            bn.cpt(var).fillWith(cpt_sample)

        bns.append(bn)

        # Debug
        safe_assert(check_consistency(bn, bn_min, bn_max))

    # Debug
    safe_assert(n == len(bns))

    return bns


# Check the consistency of a BN as sampled from a CN
def check_consistency(bn, bn_min, bn_max) -> bool:

    for var in bn.names():
        bn_cpt = np.atleast_2d(bn.cpt(var).topandas())
        bn_min_cpt = np.array(bn_min.cpt(var).topandas())
        bn_max_cpt = np.array(bn_max.cpt(var).topandas())

        # Check if probabilities sum to 1
        sum_vec = np.sum(bn_cpt, axis=1)
        probability_integrity = np.all(np.abs(sum_vec - 1) < 1e-5)

        # Check if the BN CPT is >= min CPT
        min_integrity = np.all(bn_cpt >= bn_min_cpt)

        # Check if the BN CPT is <= max CPT
        max_integrity = np.all(bn_cpt <= bn_max_cpt)

        integrity = probability_integrity and min_integrity and max_integrity

        if integrity:
            continue
        else:
            print("probability_integrity: ", probability_integrity)
            print("min_integrity: ", min_integrity)
            print("max_integrity: ", max_integrity)
            return False

    return True
