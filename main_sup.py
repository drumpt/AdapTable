import os
import warnings
warnings.filterwarnings("ignore")
import hydra
from omegaconf import OmegaConf

from data.dataset import *
from utils.utils import *


def main_sup_baseline(args):
    global logger
    if hasattr(args, 'seed'):
        set_seed(args.seed)
        print(f"set seed as {args.seed}")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = get_logger(args)
    logger.info(OmegaConf.to_yaml(args))
    disable_logger(args)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    dataset = Dataset(args, logger)
    regression = True if dataset.out_dim == 1 else False

    if args.model == 'lr':
        if regression:
            from sklearn.linear_model import LinearRegression
            source_model = LinearRegression()
        else:
            from sklearn.linear_model import LogisticRegression
            source_model = LogisticRegression()
        source_model = source_model.fit(dataset.dataset.train_x, dataset.dataset.train_y.argmax(1))
    elif args.model == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        source_model = KNeighborsClassifier(n_neighbors=3)
        source_model = source_model.fit(dataset.dataset.train_x, dataset.dataset.train_y.argmax(1))
    elif args.model == 'rf':
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        if regression:
            source_model = RandomForestRegressor(n_estimators=args.num_estimators, max_depth=args.max_depth, random_state=args.seed)
            source_model = source_model.fit(dataset.dataset.train_x, dataset.dataset.train_y)
        else:
            source_model = RandomForestClassifier(n_estimators=args.num_estimators, max_depth=args.max_depth, random_state=args.seed)
            source_model = source_model.fit(dataset.dataset.train_x, dataset.dataset.train_y.argmax(1))
    elif args.model == 'xgboost':
        if regression:
            objective = "reg:linear"
        elif dataset.dataset.train_y.argmax(1).max() == 1:
            objective = "binary:logistic"
        else:
            objective = "multi:softprob"

        from xgboost import XGBRegressor, XGBClassifier
        if regression:
            source_model = XGBRegressor(objective=objective, random_state=args.seed)
            source_model = source_model.fit(dataset.dataset.train_x, dataset.dataset.train_y)
        else:
            source_model = XGBClassifier(n_estimators=args.num_estimators, learning_rate=args.test_lr, max_depth=args.max_depth, random_state=args.seed)
            source_model = source_model.fit(dataset.dataset.train_x, dataset.dataset.train_y.argmax(1))
    else:
        raise NotImplementedError

    test_acc, test_len = 0, 0
    for test_x, test_mask_x, test_y in dataset.test_loader:
        estimated_y = source_model.predict(test_x)
        test_acc += (np.array(estimated_y) == np.argmax(np.array(test_y), axis=-1)).sum()
        test_len += test_x.shape[0]

    logger.info(f"using {args.model} | test acc {test_acc / test_len:.4f}")
    return test_acc



@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(args):
    main_sup_baseline(args)



if __name__ == "__main__":
    main()