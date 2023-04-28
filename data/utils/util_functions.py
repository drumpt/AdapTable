# import conf
from data.conf import *

def load_opt(dataset: str): # load according opt to dataset name
    opt = None

    if 'abalone' in dataset:
        opt = AbaloneOpt

    if opt is None:
        raise ValueError(f'No matching opt for dataset {dataset}')

    return opt