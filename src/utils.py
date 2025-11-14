from tempfile import TemporaryDirectory

import hopsy
import numpy as np
import pyagrum as gum

from src.config import safe_assert


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


# BNs sampler from a CN
def sample_from_cn(bn_min, bn_max, n_bns: int) -> list:

    # Get the DAG and extreme BNs
    dag = gum.BayesNet(bn_min)

    # For each variable ...
    cpts_dict = {}
    for var in dag.names():

        # ... sample `n_bns` CPTs from the CN
        cpts_dict[var] = sample_from_cpts(bn_min.cpt(var), bn_max.cpt(var), n_bns)

    # For each sample ...
    bns = []
    for i in range(n_bns):

        # ... init an empty BN ...
        bn = gum.BayesNet(dag)

        # ... and fill its CPTs
        for var in dag.names():
            bn.cpt(var).fillWith(cpts_dict[var][i])

        bns.append(bn)

        # Debug
        safe_assert(check_consistency(bn, bn_min, bn_max))

    # Debug
    safe_assert(len(cpts_dict) == len(dag.names()))
    safe_assert(len(bns) == n_bns)

    return bns


# Sample from two extreme CPTs
def sample_from_cpts(cpt_min, cpt_max, n_bns) -> list:

    # Transform CPTs into pandas dataframes
    cpt_min = np.atleast_2d(cpt_min.topandas())
    cpt_max = np.atleast_2d(cpt_max.topandas())

    # For each row in the CPT ...
    credal_dict = {}
    for row in range(cpt_min.shape[0]):

        # ... sample `n_bns` points from the credal set
        credal_dict[row] = sample_from_cset(cpt_min[row, :], cpt_max[row, :], n_bns)

    # For each sample ...
    cpt_samples = []
    for i in range(n_bns):

        # ... build the CPT
        cpt = []
        for row in range(cpt_min.shape[0]):
            cpt.append(credal_dict[row][i])

        cpt = np.array(cpt).flatten()
        cpt_samples.append(cpt)

    # Debug
    safe_assert(cpt_min.shape == cpt_max.shape)
    safe_assert(len(credal_dict) == cpt_min.shape[0])
    safe_assert(len(cpt_samples) == n_bns)

    return cpt_samples


# Sample from a credal set K(x | pi_x), i.e., a constrained polytope.
def sample_from_cset(vec_min, vec_max, n_bns) -> list:
    """
    We assume a credal set is a polytope in a space of #X parameters, defined by a:
     - Multi-dimensional rectangle, i.e., inequality constraint Ax <= b, and
     - Hyperplane (provided all the variables sum up to 1), i.e., equality constraint A_eq x = b_eq.
    In the case of local IDM, this is ensured.
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
    _, constrained_samples = hopsy.sample(mc, rng, n_bns, thinning=10)
    constrained_samples = constrained_samples[0]

    # Debug
    safe_assert(np.all(vec_min <= vec_max))
    safe_assert(n_par == len(vec_max))
    safe_assert(n_par == A.shape[1])
    safe_assert(n_par == A_eq.shape[1])
    safe_assert(len(constrained_samples) == n_bns)
    for i in constrained_samples:
        safe_assert(len(i) == n_par)

    return constrained_samples


# Check the consistency of a BN as sampled from a CN
def check_consistency(bn, bn_min, bn_max) -> bool:

    for var in bn.names():
        bn_cpt = np.atleast_2d(bn.cpt(var).topandas())
        bn_min_cpt = np.atleast_2d(bn_min.cpt(var).topandas())
        bn_max_cpt = np.atleast_2d(bn_max.cpt(var).topandas())

        # Check if probabilities sum to 1
        sum_vec = np.sum(bn_cpt, axis=1)
        probability_consistency = np.all(np.abs(sum_vec - 1) < 1e-5)

        # Check if the BN CPT is >= min CPT
        min_consistency = np.all(bn_cpt >= bn_min_cpt)

        # Check if the BN CPT is <= max CPT
        max_consistency = np.all(bn_cpt <= bn_max_cpt)

        consistency = probability_consistency and min_consistency and max_consistency

        if consistency:
            continue
        else:
            print("probability_consistency: ", probability_consistency)
            print("min_consistency: ", min_consistency)
            print("max_consistency: ", max_consistency)
            return False

    return True


# Check BNs sampled from a CN
def are_all_bns_different(bn_vec) -> bool:

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

    return len(signatures) == len(bn_vec)


# Extract BN min and BN max from a CN
def get_min_max_bns(cn, exp: str):

    with TemporaryDirectory() as tmp_path:
        cn.saveBNsMinMax(f"{tmp_path}/bn_min_{exp}.bif", f"{tmp_path}/bn_max_{exp}.bif")
        bn_min = gum.loadBN(f"{tmp_path}/bn_min_{exp}.bif")
        bn_max = gum.loadBN(f"{tmp_path}/bn_max_{exp}.bif")

    return bn_min, bn_max
