from run import *
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from itertools import product
from numpy import random
import os
import shutil

num_cores = multiprocessing.cpu_count() - 1

exp_names = [os.path.splitext(f)[0] for f in os.listdir("./data/")]

if __name__ == "__main__":

    # Set seeds
    random.seed(42)
    gum.initRandom(seed=42)

    # For any exp...
    for exp in exp_names: 

        # Find eps s.t. AUC(eps)~AUC(CN)
        conf = [exp, {"ess":1}]
        eps = run_idm(conf)
        print(f"Eps: {eps}")    ##

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
        cn.idmLearning(1)

        # Learn noisy BN 
        scale = (2 * bn.size()) / (len(gpop) * eps)
        bn_noisy = get_noisy_bn(bn, scale)

        # Run inferences
        gt_mpes = run_inference_bn(gt)
        bn_mpes = run_inference_bn(bn)
        bn_noisy_mpes = run_inference_bn(bn_noisy)
        cn_mpes, cn_probs = run_inference_cn(cn)

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

        results.to_csv(f"results/{exp}.csv", index = False)
        

    # Experiments (in parallel)
    # Parallel(n_jobs=num_cores)(delayed(run_idm)(conf) for conf in confs)

    

    