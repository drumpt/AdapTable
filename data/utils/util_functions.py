# import conf
from data.conf import *

def load_opt(dataset: str): # load according opt to dataset name
    opt = None

    if 'abalone' in dataset:
        opt = AbaloneOpt
    elif 'cholestrol' in dataset:
        opt = CholestrolOpt
    elif 'sarcos' in dataset:
        opt = SarcosOpt
    elif 'boston' in dataset:
        opt = BostonOpt
    elif 'news' in dataset:
        opt = NewsOpt
    elif "diamonds" in dataset:
        opt = DiamondsOpt
    elif "seattlecrime6" in dataset:
        opt = SeattlecrimeOpt
    elif "Brazilian_houses" in dataset:
        opt = BrazilianhousesOpt
    elif "topo_2_1" in dataset:
        opt = TopoOpt
    elif "house_sales" in dataset:
        opt = HouseOpt
    elif "particulate-matter-ukair-2017" in dataset:
        opt = UkairOpt
    elif "analcatdata_supreme" in dataset:
        opt = AnalcatOpt
    elif "delays_zurich_transport" in dataset:
        opt = DelayOpt
    elif "Bike_Sharing_Demand" in dataset:
        opt = BikeOpt
    elif "nyc-taxi-green-dec-2016" in dataset:
        opt = TaxiOpt
    elif "visualizing_soil" in dataset:
        opt = SoilOpt
    elif "SGEMM_GPU_kernel_performance" in dataset:
        opt = GpuOpt
    if opt is None:
        raise ValueError(f'No matching opt for dataset {dataset}')

    return opt