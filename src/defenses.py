import pyagrum as gum

from src.utils import add_counts_to_bn


# Estimate a CN from data by local IDM
def def_idm(bn, ess, data):
    bn_counts = gum.BayesNet(bn)
    add_counts_to_bn(bn_counts, data)
    cn = gum.CredalNet(bn_counts)
    cn.idmLearning(ess)

    return cn
