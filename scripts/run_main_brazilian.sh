python main.py \
    meta_dataset=openml-regression \
    dataset=Brazilian_houses \
    shift_type=null \
    retrain=true \
    seed=0 \
    out_dir=exps/Brazilian_houses \
    --config-name=config_cholestrol.yaml \

python main.py \
    meta_dataset=openml-regression \
    dataset=Brazilian_houses \
    shift_type=null \
    retrain=false \
    seed=1 \
    out_dir=exps/Brazilian_houses \
    --config-name=config_cholestrol.yaml \

python main.py \
    meta_dataset=openml-regression \
    dataset=Brazilian_houses \
    shift_type=null \
    retrain=false \
    seed=2 \
    out_dir=exps/Brazilian_houses \
    --config-name=config_cholestrol.yaml \



python main.py \
    meta_dataset=openml-regression \
    dataset=Brazilian_houses \
    shift_type=Gaussian \
    shift_severity=1 \
    seed=0 \
    out_dir=exps/Brazilian_houses \
    --config-name=config_cholestrol.yaml \

python main.py \
    meta_dataset=openml-regression \
    dataset=Brazilian_houses \
    shift_type=Gaussian \
    shift_severity=1 \
    seed=1 \
    out_dir=exps/Brazilian_houses \
    --config-name=config_cholestrol.yaml \

python main.py \
    meta_dataset=openml-regression \
    dataset=Brazilian_houses \
    shift_type=Gaussian \
    shift_severity=1 \
    seed=2 \
    out_dir=exps/Brazilian_houses \
    --config-name=config_cholestrol.yaml \



python main.py \
    meta_dataset=openml-regression \
    dataset=Brazilian_houses \
    shift_type=random_drop \
    shift_severity=0.8 \
    seed=0 \
    out_dir=exps/Brazilian_houses \
    --config-name=config_cholestrol.yaml \

python main.py \
    meta_dataset=openml-regression \
    dataset=Brazilian_houses \
    shift_type=random_drop \
    shift_severity=0.8 \
    seed=1 \
    out_dir=exps/Brazilian_houses \
    --config-name=config_cholestrol.yaml \

python main.py \
    meta_dataset=openml-regression \
    dataset=Brazilian_houses \
    shift_type=random_drop \
    shift_severity=0.8 \
    seed=2 \
    out_dir=exps/Brazilian_houses \
    --config-name=config_cholestrol.yaml \



python main.py \
    meta_dataset=openml-regression \
    dataset=Brazilian_houses \
    shift_type=column_drop \
    shift_severity=0.8 \
    seed=0 \
    out_dir=exps/Brazilian_houses \
    --config-name=config_cholestrol.yaml \

python main.py \
    meta_dataset=openml-regression \
    dataset=Brazilian_houses \
    shift_type=column_drop \
    shift_severity=0.8 \
    seed=1 \
    out_dir=exps/Brazilian_houses \
    --config-name=config_cholestrol.yaml \

python main.py \
    meta_dataset=openml-regression \
    dataset=Brazilian_houses \
    shift_type=column_drop \
    shift_severity=0.8 \
    seed=2 \
    out_dir=exps/Brazilian_houses \
    --config-name=config_cholestrol.yaml \



python main.py \
    meta_dataset=openml-regression \
    dataset=Brazilian_houses \
    shift_type=mean_shift \
    shift_severity=1 \
    seed=0 \
    out_dir=exps/Brazilian_houses \
    --config-name=config_cholestrol.yaml \

python main.py \
    meta_dataset=openml-regression \
    dataset=Brazilian_houses \
    shift_type=mean_shift \
    shift_severity=1 \
    seed=1 \
    out_dir=exps/Brazilian_houses \
    --config-name=config_cholestrol.yaml \

python main.py \
    meta_dataset=openml-regression \
    dataset=Brazilian_houses \
    shift_type=mean_shift \
    shift_severity=1 \
    seed=2 \
    out_dir=exps/Brazilian_houses \
    --config-name=config_cholestrol.yaml \



python main.py \
    meta_dataset=openml-regression \
    dataset=Brazilian_houses \
    shift_type=std_shift \
    shift_severity=1 \
    seed=0 \
    out_dir=exps/Brazilian_houses \
    --config-name=config_cholestrol.yaml \

python main.py \
    meta_dataset=openml-regression \
    dataset=Brazilian_houses \
    shift_type=std_shift \
    shift_severity=1 \
    seed=1 \
    out_dir=exps/Brazilian_houses \
    --config-name=config_cholestrol.yaml \

python main.py \
    meta_dataset=openml-regression \
    dataset=Brazilian_houses \
    shift_type=std_shift \
    shift_severity=1 \
    seed=2 \
    out_dir=exps/Brazilian_houses \
    --config-name=config_cholestrol.yaml \



python main.py \
    meta_dataset=openml-regression \
    dataset=Brazilian_houses \
    shift_type=mean_std_shift \
    shift_severity=1 \
    seed=0\
    out_dir=exps/Brazilian_houses \
    --config-name=config_cholestrol.yaml \

python main.py \
    meta_dataset=openml-regression \
    dataset=Brazilian_houses \
    shift_type=mean_std_shift \
    shift_severity=1 \
    seed=1 \
    out_dir=exps/Brazilian_houses \
    --config-name=config_cholestrol.yaml \

python main.py \
    meta_dataset=openml-regression \
    dataset=Brazilian_houses \
    shift_type=mean_std_shift \
    shift_severity=1 \
    seed=2 \
    out_dir=exps/Brazilian_houses \
    --config-name=config_cholestrol.yaml \



# python main.py \
#     meta_dataset=tableshift \
#     dataset=heloc \
#     retrain=true \
#     seed=0 \
#     out_dir=exps/heloc_main \

# python main.py \
#     meta_dataset=tableshift \
#     dataset=heloc \
#     retrain=false \
#     seed=1 \
#     out_dir=exps/heloc_main \

# python main.py \
#     meta_dataset=tableshift \
#     dataset=heloc \
#     retrain=false \
#     seed=2 \
#     out_dir=exps/heloc_main \


# python main.py \
#     meta_dataset=tableshift \
#     dataset=diabetes_readmission \
#     retrain=true \
#     seed=0 \
#     out_dir=exps/diabetes_main \

# python main.py \
#     meta_dataset=tableshift \
#     dataset=diabetes_readmission \
#     retrain=false \
#     seed=1 \
#     out_dir=exps/diabetes_main \

# python main.py \
#     meta_dataset=tableshift \
#     dataset=diabetes_readmission \
#     retrain=false \
#     seed=2 \
#     out_dir=exps/diabetes_main \