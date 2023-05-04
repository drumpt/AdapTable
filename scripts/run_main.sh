python main.py \
    meta_dataset=openml-cc18 \
    dataset=semeion \
    shift_type=null \
    retrain=true \
    out_dir=exp/semeion \

python main.py \
    meta_dataset=openml-cc18 \
    dataset=semeion \
    shift_type=Gaussian \
    shift_severity=1 \
    out_dir=exp/semeion \

python main.py \
    meta_dataset=openml-cc18 \
    dataset=semeion \
    shift_type=random_drop \
    shift_severity=0.6 \
    out_dir=exp/semeion \

python main.py \
    meta_dataset=openml-cc18 \
    dataset=semeion \
    shift_type=column_drop \
    shift_severity=0.6 \
    out_dir=exp/semeion \

python main.py \
    meta_dataset=openml-cc18 \
    dataset=semeion \
    shift_type=mean_shift \
    shift_severity=1 \
    out_dir=exp/semeion \

python main.py \
    meta_dataset=openml-cc18 \
    dataset=semeion \
    shift_type=std_shift \
    shift_severity=1 \
    out_dir=exp/semeion \

python main.py \
    meta_dataset=openml-cc18 \
    dataset=semeion \
    shift_type=mean_std_shift \
    shift_severity=1 \
    out_dir=exp/semeion \

python main.py \
    meta_dataset=tableshift \
    dataset=heloc \
    retrain=true \
    out_dir=exp/heloc_main \

python main.py \
    meta_dataset=tableshift \
    dataset=diabetes_readmission \
    retrain=true \
    out_dir=exp/diabetes_main \