## Libraries
import numpy as np
import pandas as pd
import math
from scipy.stats import norm
import pyagrum as gum
from pathlib import Path
from pandarallel import pandarallel
from collections import defaultdict
import numpy as np
import io
import re
from contextlib import redirect_stdout
from itertools import product
from more_itertools import random_product

## Log-likelihood ratio function
def LLR(x: dict, bn_theta_ie, bn_theta_hat_ie):

    # Erase all evidences and apply addEvidence(key,value) for every pairs in x
    bn_theta_ie.setEvidence(x)
    # assert(bn_theta_ie.nbrHardEvidence() == len(bn.nodes()))
    bn_theta_hat_ie.setEvidence(x)
    # assert(bn_theta_hat_ie.nbrHardEvidence() == len(bn.nodes()))

    # Compute P(x | BN)
    L_theta = bn_theta_ie.evidenceProbability()
    L_theta_hat = bn_theta_hat_ie.evidenceProbability()

    # Check for denominator
    if L_theta_hat < 1e-16:
        return np.nan
    
    return math.log(L_theta / L_theta_hat)

## LLR test
def reject_H0(x: dict, bn_theta_ie, bn_theta_hat_ie, t: float) -> bool:

    # Compute the value of LLR(x)
    llr = LLR(x, bn_theta_ie, bn_theta_hat_ie)

    # If denominator is 0, then don't reject H0
    if llr == np.nan:
        return False
    
    return llr < t

## Parse the credal network
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

## Get CN simplex, or a random subset of it
def get_simplex(cn, n: int = None) -> list:

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
    var_idx = {var:[idx for idx, elem in enumerate(slots) if elem[0] == var] for var in bn.names()}

    # If 'n' is provided...
    if bool(n):
        # Get 'n' random combinations of CPTs
        combinations = [random_product(*[vecs for _, _, vecs in slots]) for _ in range(n)]
        n_combs = len(combinations)
        assert(n_combs == n); print(f"Generating {n_combs} random BNs...")
    else:
        # Get all combinations of CPTs
        combinations = list(product(*[vecs for _, _, vecs in slots]))
        n_combs = len(combinations); print(f"Generating {n_combs} BNs...")

    # For each combination...
    for combo in combinations:

        # Init BN and ...
        bn_tmp = gum.BayesNet(dag)

        # Fill its CPTs
        for var in bn.names():
            array = np.array([(combo[idx]) for idx in var_idx.get(var)]).flatten()
            bn_tmp.cpt(var).fillWith(array)

        bns.append(bn_tmp)
    
    # Check
    assert(n_combs == len(bns))

    return bns

## Check function for simplex extraction
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

    if len(signatures) == len(bn_list):
        print(f"✅ All {len(signatures)} BNs are different.")
        return
    else:
        unique_sigs = set()
        unique_bns = []

        for bn in bn_list:
            sig = serialize_bn(bn)
            if sig not in unique_sigs:
                unique_sigs.add(sig)
                unique_bns.append(bn)
        print(f"⚠️ Obtained {len(unique_bns)}/{len(bn_list)} different BNs.")
        return


### MAIN

if __name__ == "__main__":

    # Initialize parallelization
    pandarallel.initialize(progress_bar=False)

    ## Create results directory
    results_path = Path("./results")
    results_path.mkdir(parents=True, exist_ok=True)

    ## Generate ground-truth BN
    n_nodes = 20     # (must be n_nodes > 2)
    bn_gen = gum.BNGenerator()
    bn = bn_gen.generate(n_nodes=n_nodes, n_arcs=30, n_modmax=2) # 

    ## Generate data
    gpop_ss = 10000  #
    ratio = 6   #
    pool_ss = gpop_ss // ratio
    rpop_ss = gpop_ss - pool_ss

    # assert(gpop_ss == pool_ss + rpop_ss)

    data_gen = gum.BNDatabaseGenerator(bn)
    data_gen.drawSamples(gpop_ss)
    data_gen.setDiscretizedLabelModeRandom()

    gpop = data_gen.to_pandas()
    pool_idx = np.random.choice(gpop_ss, replace=False, size=pool_ss)
    pool = gpop.iloc[pool_idx]
    rpop = gpop.iloc[~ gpop.index.isin(pool_idx)]

    # assert(gpop.shape[0]==gpop_ss)
    # assert(pool.shape[0]==pool_ss)
    # assert(rpop.shape[0]==rpop_ss)

    ## Estimate BN(theta) from rpop and BN(theta_hat) from pool
    theta_learner=gum.BNLearner(rpop)
    theta_learner.useSmoothingPrior(1e-5)
    theta_hat_learner=gum.BNLearner(pool)
    theta_hat_learner.useSmoothingPrior(1e-5)

    bn_theta = theta_learner.learnParameters(bn.dag())
    bn_theta_hat = theta_hat_learner.learnParameters(bn.dag())

    ## Estimate BN(theta_min) and BN(theta_max) from a CN by local IDM
    # Add counts of events to BN
    pop = pool
    for node in bn.names():
        var = bn.variable(node)
        parents = bn.parents(node)
        parent_names = [bn.variable(p).name() for p in parents]

        shape = [bn.variable(p).domainSize() for p in parents] + [var.domainSize()]
        counts_array = np.zeros(shape, dtype=float)  # float, not int!

        for _, row in pop.iterrows():
            try:
                key = tuple([int(row[p]) for p in parent_names] + [int(row[node])])
                counts_array[key] += 1.0
            except KeyError:
                continue

        bn.cpt(node).fillWith(counts_array.flatten().tolist())
    
    # Learn the CN through local IDM
    ess = 1   #
    cn = gum.CredalNet(bn)
    cn.idmLearning(ess)

    # Extract random subset of simplex
    n_bns = 10
    bns_simplex = get_simplex(cn, n_bns)
    are_all_bn_different(bns_simplex)

    ### MIA on ground-truth BN
    ## Compute the LLR ECDFs under H_0, i.e. x not in pool
    # Create objects for inference
    bn_theta_ie = gum.LazyPropagation(bn_theta)
    bn_theta_hat_ie = gum.LazyPropagation(bn_theta_hat)

    # Estimate the distribution of LLR(x) from rpop
    L_im = rpop.parallel_apply(lambda x: LLR(x.to_dict(), bn_theta_ie, bn_theta_hat_ie), axis=1).dropna().sort_values()

    ## Membership inference
    # Set ground truth membership
    gpop["in-pool"] = False
    gpop.loc[pool_idx, "in-pool"] = True
    tp = len(pool_idx)
    
    # Set error (alpha) range
    error = np.arange(0, 1, 0.05)    #

    # Compute theoretical bound
    compl = bn.dim()
    bound = math.sqrt(compl/pool_ss)

    # Find power (beta) for any error (alpha) given theoretical bound
    z_alpha = [norm.ppf(1 - i).item() for i in error]
    z_one_minus_beta = [bound - i for i in z_alpha]
    beta = [norm.cdf(i).item() for i in z_one_minus_beta]

    # Init results
    results = pd.DataFrame(
        {"error": error,
        "power_bound": beta}
    )

    # Init the power vector (BN)
    power_bn = []

    # For each error ...
    for e in error:

        # Compute threshold
        t = np.quantile(L_im, e).item()

        # Perform LLR test on whole population
        y_pred = gpop[[*bn.names()]].parallel_apply(lambda x: reject_H0(x.to_dict(), bn_theta_ie, bn_theta_hat_ie, t), axis=1)

        # Compute and store power (tpr)
        tpr = sum(gpop["in-pool"] & y_pred) / tp
        power_bn = power_bn + [tpr]
    
    # Store results
    results["power_BN"] = power_bn

    ### MIA on simplex BNs
    for i, bn_vertex in enumerate(bns_simplex):

        ## Compute the LLR ECDFs under H_0, i.e. x not in pool
        # Create objects for inference
        bn_vertex_ie = gum.LazyPropagation(bn_vertex)

        # Estimate the distribution of LLR(x) from rpop
        L_im_vertex = rpop.parallel_apply(lambda x: LLR(x.to_dict(), bn_theta_ie, bn_vertex_ie), axis=1).dropna().sort_values()

        ## Membership inference
        # Init the power vector (BN vertex)
        power_bn_vertex = []

        # For each error ...
        for e in error:

            # Compute threshold
            t = np.quantile(L_im_vertex, e).item()

            # Perform LLR test on whole population
            y_pred_vertex = gpop[[*bn.names()]].parallel_apply(lambda x: reject_H0(x.to_dict(), bn_theta_ie, bn_vertex_ie, t), axis=1)

            # Compute and store power (tpr)
            tpr = sum(gpop["in-pool"] & y_pred_vertex) / tp
            power_bn_vertex = power_bn_vertex + [tpr]
        
        # Store results
        results[f"power_BN_v_{i}"] = power_bn_vertex

    ## Save results
    results.to_csv(f"./results/compl{compl}-s{ess}-simplex{n_bns}.csv")
