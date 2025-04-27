import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import pandas as pd

num_cores = multiprocessing.cpu_count()
inputs = tqdm([x for x in range(1000)])

def my_function(x):
    time.sleep(0.1)
    # return x+2
    return pd.DataFrame()

def multiple_of_six(i):
    # return (i*6)
    return pd.DataFrame()

if __name__ == "__main__":
    processed_list = Parallel(n_jobs=num_cores)(delayed(my_function)(i) for i in inputs)

    # processed_list = []
    # for i in inputs:
    #     processed_list.append(my_function(i))

    print(processed_list)