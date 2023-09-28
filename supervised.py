import os
import warnings
warnings.filterwarnings("ignore")
import hydra
from omegaconf import OmegaConf
from sklearn.model_selection import RandomizedSearchCV


from data.dataset import *
from utils.utils import *


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
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

    param_grid = get_param_grid(args)

    if args.model == 'lr':
        if regression:
            from sklearn.linear_model import LinearRegression
            source_model = LinearRegression()
        else:
            from sklearn.linear_model import LogisticRegression
            source_model = LogisticRegression()
        source_model = source_model.fit(dataset.train_x, dataset.train_y.argmax(1))
    elif args.model == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        source_model = KNeighborsClassifier()
        rs = RandomizedSearchCV(
            estimator=source_model,
            param_distributions=param_grid,
            n_iter=100,
            cv=5,
            verbose=1,
            n_jobs=-1
        )
        rs.fit(dataset.train_x, dataset.train_y.argmax(1))

        best_params = rs.best_params_
        print(f'best params are: {rs.best_params_}')

        source_model = KNeighborsClassifier(**best_params)
        source_model.fit(dataset.train_x, dataset.train_y.argmax(1))
    elif args.model == 'rf':
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        if regression:
            source_model = RandomForestRegressor(n_estimators=args.num_estimators, max_depth=args.max_depth, random_state=args.seed)
            source_model = source_model.fit(dataset.train_x, dataset.train_y)
        else:
            source_model = RandomForestClassifier(random_state=args.seed)
            rs = RandomizedSearchCV(
                estimator=source_model,
                param_distributions=param_grid,
                n_iter=100,
                cv=5,
                verbose=1,
                n_jobs=-1
            )
            rs.fit(dataset.train_x, dataset.train_y.argmax(1))

            best_params = rs.best_params_
            print(f'best params are: {rs.best_params_}')

            source_model = RandomForestClassifier(**best_params, random_state=args.seed)
            source_model.fit(dataset.train_x, dataset.train_y.argmax(1))
    elif args.model == 'xgboost':
        if regression:
            objective = "reg:linear"
        elif dataset.train_y.argmax(1).max() == 1:
            objective = "binary:logistic"
        else:
            objective = "multi:softprob"

        from xgboost import XGBRegressor, XGBClassifier
        if regression:
            source_model = XGBRegressor(objective=objective, random_state=args.seed)
            rs = RandomizedSearchCV(source_model, param_grid, n_iter=100, cv=5, verbose=1, n_jobs=-1)
            rs.fit(dataset.train_x, dataset.train_y)
            print(f'best params are: {rs.best_params_}')
            # source_model = source_model.fit(dataset.train_x, dataset.train_y)
        else:
            source_model = XGBClassifier(random_state=args.seed)
            rs = RandomizedSearchCV(
                estimator=source_model,
                param_distributions=param_grid,
                n_iter=100,
                cv=5,
                verbose=1,
                n_jobs=-1
            )
            rs.fit(dataset.train_x, dataset.train_y.argmax(1))

            best_params = rs.best_params_
            print(f'best params are: {rs.best_params_}')

            source_model = XGBClassifier(**best_params, random_state=args.seed)
            source_model.fit(dataset.train_x, dataset.train_y.argmax(1))
            # source_model = source_model.fit(dataset.train_x, dataset.train_y.argmax(1))
    elif args.model == 'catboost':
        from catboost import CatBoostClassifier, CatBoostRegressor
        train_x = dataset.train_x
        if regression:
            catboost_model = CatBoostRegressor()
            train_y = dataset.train_y
            rs = RandomizedSearchCV(
                estimator=catboost_model,
                param_distributions=param_grid,
                n_iter=100,
                cv=5,
                verbose=1,
                n_jobs=-1
            )
            rs.fit(train_x, train_y)
            print(f'best params are: {rs.best_params_}')
            source_model = CatBoostRegressor(**rs.best_params_, random_state=args.seed)
            source_model.fit(train_x, train_y)
        else:
            catboost_model = CatBoostClassifier()
            train_y = dataset.train_y.argmax(1)
            rs = RandomizedSearchCV(
                estimator=catboost_model,
                param_distributions=param_grid,
                n_iter=100,
                cv=5,
                verbose=1,
                n_jobs=-1
            )
            rs.fit(train_x, train_y)
            print(f'best params are: {rs.best_params_}')
            source_model = CatBoostClassifier(**rs.best_params_, random_state=args.seed)
            source_model.fit(train_x, train_y)
    else:
        raise NotImplementedError

    test_acc, test_len = 0, 0
    for test_x, test_mask_x, test_y in dataset.test_loader:
        estimated_y = source_model.predict(test_x)
        test_acc += (estimated_y == np.argmax(np.array(test_y), axis=-1)).sum()
        test_len += test_x.shape[0]

    logger.info(f"using {args.model} | test acc {test_acc / test_len:.4f}")
    return test_acc


def get_param_grid(args):
    param_grid = {}
    if args.model == 'xgboost':
        param_grid = {
            'n_estimators': np.arange(50, 200, 5),
            'learning_rate': np.linspace(0.01, 1, 20),
            'max_depth': np.arange(2, 12, 1),
            # 'subsample': np.linspace(0.5, 1, 10),
            # 'colsample_bytree': np.linspace(0.5, 1, 10),
            'gamma': np.linspace(0, 0.5, 11)
        }
    elif args.model == 'rf':
        param_grid = {
            'n_estimators': np.arange(50, 200, 5),
            'max_depth': np.arange(2, 12, 1),
        }
    elif args.model == 'knn':
        param_grid = {
            'n_neighbors': np.arange(2, 12, 1),
        }
    elif args.model == 'lr':
        pass
    elif args.model == 'catboost':
        param_grid = {
            'iterations': np.arange(50, 2000, 50),
            'learning_rate': np.linspace(0.01, 1, 20),
            'depth': np.arange(5, 40, 5),
        }
    else:
        raise NotImplementedError

    return param_grid




if __name__ == "__main__":
    main_sup_baseline()