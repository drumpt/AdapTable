python main.py \
    meta_dataset=openml-cc18 \
    dataset=dna \
    shift_type=null \
    retrain=true \
    seed=0 \
    out_dir=exps/dna \

python main.py \
    meta_dataset=openml-cc18 \
    dataset=dna \
    shift_type=null \
    retrain=true \
    seed=1 \
    out_dir=exps/dna \

python main.py \
    meta_dataset=openml-cc18 \
    dataset=dna \
    shift_type=null \
    retrain=true \
    seed=2 \
    out_dir=exps/dna \



python main.py \
    meta_dataset=openml-cc18 \
    dataset=dna \
    shift_type=Gaussian \
    shift_severity=1 \
    seed=0 \
    out_dir=exps/dna \

python main.py \
    meta_dataset=openml-cc18 \
    dataset=dna \
    shift_type=Gaussian \
    shift_severity=1 \
    seed=1 \
    out_dir=exps/dna \

python main.py \
    meta_dataset=openml-cc18 \
    dataset=dna \
    shift_type=Gaussian \
    shift_severity=1 \
    seed=2 \
    out_dir=exps/dna \



python main.py \
    meta_dataset=openml-cc18 \
    dataset=dna \
    shift_type=random_drop \
    shift_severity=0.6 \
    seed=0 \
    out_dir=exps/dna \

python main.py \
    meta_dataset=openml-cc18 \
    dataset=dna \
    shift_type=random_drop \
    shift_severity=0.6 \
    seed=1 \
    out_dir=exps/dna \

python main.py \
    meta_dataset=openml-cc18 \
    dataset=dna \
    shift_type=random_drop \
    shift_severity=0.6 \
    seed=2 \
    out_dir=exps/dna \



python main.py \
    meta_dataset=openml-cc18 \
    dataset=dna \
    shift_type=column_drop \
    shift_severity=0.6 \
    seed=0 \
    out_dir=exps/dna \

python main.py \
    meta_dataset=openml-cc18 \
    dataset=dna \
    shift_type=column_drop \
    shift_severity=0.6 \
    seed=1 \
    out_dir=exps/dna \

python main.py \
    meta_dataset=openml-cc18 \
    dataset=dna \
    shift_type=column_drop \
    shift_severity=0.6 \
    seed=2 \
    out_dir=exps/dna \



python main.py \
    meta_dataset=openml-cc18 \
    dataset=dna \
    shift_type=mean_shift \
    shift_severity=1 \
    seed=0 \
    out_dir=exps/dna \

python main.py \
    meta_dataset=openml-cc18 \
    dataset=dna \
    shift_type=mean_shift \
    shift_severity=1 \
    seed=1 \
    out_dir=exps/dna \

python main.py \
    meta_dataset=openml-cc18 \
    dataset=dna \
    shift_type=mean_shift \
    shift_severity=1 \
    seed=2 \
    out_dir=exps/dna \



python main.py \
    meta_dataset=openml-cc18 \
    dataset=dna \
    shift_type=std_shift \
    shift_severity=1 \
    seed=0 \
    out_dir=exps/dna \

python main.py \
    meta_dataset=openml-cc18 \
    dataset=dna \
    shift_type=std_shift \
    shift_severity=1 \
    seed=1 \
    out_dir=exps/dna \

python main.py \
    meta_dataset=openml-cc18 \
    dataset=dna \
    shift_type=std_shift \
    shift_severity=1 \
    seed=2 \
    out_dir=exps/dna \



python main.py \
    meta_dataset=openml-cc18 \
    dataset=dna \
    shift_type=mean_std_shift \
    shift_severity=1 \
    seed=0\
    out_dir=exps/dna \

python main.py \
    meta_dataset=openml-cc18 \
    dataset=dna \
    shift_type=mean_std_shift \
    shift_severity=1 \
    seed=1 \
    out_dir=exps/dna \

python main.py \
    meta_dataset=openml-cc18 \
    dataset=dna \
    shift_type=mean_std_shift \
    shift_severity=1 \
    seed=2 \
    out_dir=exps/dna \



python main.py \
    meta_dataset=tableshift \
    dataset=heloc \
    retrain=true \
    seed=0 \
    out_dir=exps/heloc_main \

python main.py \
    meta_dataset=tableshift \
    dataset=heloc \
    retrain=true \
    seed=1 \
    out_dir=exps/heloc_main \

python main.py \
    meta_dataset=tableshift \
    dataset=heloc \
    retrain=true \
    seed=2 \
    out_dir=exps/heloc_main \


python main.py \
    meta_dataset=tableshift \
    dataset=diabetes_readmission \
    retrain=true \
    seed=0 \
    out_dir=exps/diabetes_main \

python main.py \
    meta_dataset=tableshift \
    dataset=diabetes_readmission \
    retrain=true \
    seed=1 \
    out_dir=exps/diabetes_main \

python main.py \
    meta_dataset=tableshift \
    dataset=diabetes_readmission \
    retrain=true \
    seed=2 \
    out_dir=exps/diabetes_main \