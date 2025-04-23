import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from scipy.stats import norm
from tqdm import tqdm
import pyagrum as gum
import yaml

from utils import *

warnings.filterwarnings('once')

if __name__ == "__main__":

    # Create results directory if not exists
    results_path = Path("./results/idm")
    results_path.mkdir(parents=True, exist_ok=True)

    # Import and init hyperparameters
    with open('config.yaml') as f:
        config = yaml.safe_load(f.read())

    n_modmax = config["invar"]["n_modmax"]
    gpop_ss = config["invar"]["gpop_ss"]
    ratio = config["invar"]["ratio"]
    error = eval(config["invar"]["error"])
    n_bns = config["invar"]["n_bns"]

    for conf in config["var"]:
        print(f"\n----- Running: {conf['meta']} -----")

        n_nodes = conf["n_nodes"]
        n_arcs = conf["n_arcs"]
        ess = conf["ess"]
        
        # Init parallelization
        pandarallel.initialize(progress_bar=False, verbose=1)   # Show only warnings

        # Generate ground-truth BN
        bn_gen = gum.BNGenerator()
        bn = bn_gen.generate(n_nodes=n_nodes, n_arcs=n_arcs, n_modmax=n_modmax)

        # Sample data   
        data_gen = gum.BNDatabaseGenerator(bn)
        data_gen.drawSamples(gpop_ss)
        data_gen.setDiscretizedLabelModeRandom()
        gpop = data_gen.to_pandas()

        pool_ss = gpop_ss // (ratio + 1)
        pool_idx = np.random.choice(gpop_ss, replace=False, size=pool_ss)
        pool = gpop.iloc[pool_idx]
        rpop = gpop.iloc[~ gpop.index.isin(pool_idx)]

        # Debug
        # assert(gpop_ss == gpop.shape[0])
        # assert(pool.shape[0] + rpop.shape[0] == gpop_ss)

        # Estimate BN(theta) from rpop and BN(theta_hat) from pool
        theta_learner=gum.BNLearner(rpop)
        theta_learner.useSmoothingPrior(1e-5)
        bn_theta = theta_learner.learnParameters(bn.dag())

        theta_hat_learner=gum.BNLearner(pool)
        theta_hat_learner.useSmoothingPrior(1e-5)
        bn_theta_hat = theta_hat_learner.learnParameters(bn.dag())

        ## Estimate CN by local IDM 
        # Add counts of events to BN (from pool)
        for node in bn.names():
            var = bn.variable(node)
            parents = bn.parents(node)
            parent_names = [bn.variable(p).name() for p in parents]

            shape = [bn.variable(p).domainSize() for p in parents] + [var.domainSize()]
            counts_array = np.zeros(shape, dtype=float)  # float, not int!

            for _, row in pool.iterrows():
                try:
                    key = tuple([int(row[p]) for p in parent_names] + [int(row[node])])
                    counts_array[key] += 1.0
                except KeyError:
                    continue

            bn.cpt(node).fillWith(counts_array.flatten().tolist())
        
        # Learn the CN
        cn = gum.CredalNet(bn)
        cn.idmLearning(ess)

        # Extract random subset within simplex
        bns_sample = get_simplex_inner(cn, n_bns)

        # Debug
        # are_all_bn_different(bns_sample)

        ## MIA (theoretical)
        # Set ground truth membership
        gpop["in-pool"] = False
        gpop.loc[pool_idx, "in-pool"] = True
        tp = len(pool_idx)

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

        ## MIA (BN)
        # Estimate the distribution of LLR(x) from rpop (i.e. under H_0)
        bn_theta_ie = gum.LazyPropagation(bn_theta)
        bn_theta_hat_ie = gum.LazyPropagation(bn_theta_hat)
        L_im = rpop.parallel_apply(lambda x: LLR(x.to_dict(), bn_theta_ie, bn_theta_hat_ie), axis=1).dropna().sort_values() 

        # Compute LLR(x) on general population
        llr_gpop = gpop[[*bn.names()]].parallel_apply(lambda x: LLR(x.to_dict(), bn_theta_ie, bn_theta_hat_ie), axis=1)

        # Init the power vector (BN)
        power_bn = []

        # For each error ...
        for e in error:

            # Compute threshold
            t = np.quantile(L_im, e).item()

            # LLR test. Reject H_0? True => x in pool; False => x in rpop.
            y_pred = llr_gpop < t

            # Compute and store power (tpr)
            tpr = sum(gpop["in-pool"] & y_pred) / tp
            power_bn = power_bn + [tpr]

        # Store results
        results["power_BN"] = power_bn

        ## MIA (CN)
        print(f"Compute CN power on {n_bns} inner BNs...")
        for i, bn_vertex in enumerate(tqdm(bns_sample, unit="item", dynamic_ncols=True)):

            # Estimate the distribution of LLR(x) from rpop (i..e under H_0)
            bn_vertex_ie = gum.LazyPropagation(bn_vertex)
            L_im_vertex = rpop.parallel_apply(lambda x: LLR(x.to_dict(), bn_theta_ie, bn_vertex_ie), axis=1).dropna().sort_values()

            # Compute LLR(x) on general population
            llr_gpop_vertex = gpop[[*bn.names()]].parallel_apply(lambda x: LLR(x.to_dict(), bn_theta_ie, bn_vertex_ie), axis=1)

            # Init the power vector (BN vertex)
            power_bn_vertex = []

            # For each error ...
            for e in error:

                # Compute threshold
                t = np.quantile(L_im_vertex, e).item()

                # LLR test. Reject H_0? True => x in pool; False => x in rpop.
                y_pred_vertex = llr_gpop_vertex < t
                
                # Compute and store power (tpr)
                tpr = sum(gpop["in-pool"] & y_pred_vertex) / tp
                power_bn_vertex = power_bn_vertex + [tpr]
            
            # Store results
            results[f"power_BN_v_{i}"] = power_bn_vertex

        # Save results
        results.to_csv(f"./results/idm/{conf['meta']}-compl{compl}.csv")