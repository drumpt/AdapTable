############# run in single GPU ##############
GPUS=(0 1 2)
NUM_GPUS=3
##############################################
i=0

wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=60 #12
  local num_max_jobs=60
  echo $num_max_jobs
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}

# for dataset in abalone Bike_Sharing_Demand house_sales analcatdata_supreme
for dataset in Bike_Sharing_Demand
do
    for seed in 0 1 2 3 4
    do
        python main.py \
            --config-name config.yaml \
            meta_dataset=openml-regression \
            dataset=${dataset} \
            shift_type=null \
            retrain=true \
            extra_config=conf/config_dem.yaml \
            seed=${seed} \
            out_dir=regression_dem/${dataset}/seed_${seed}/null \
            log_dir=log/regression_dem/${dataset}/seed_${seed}/null \
            device=cuda:${GPUS[i % ${NUM_GPUS}]} \
            2>&1 &
        wait_n
        i=$((i + 1))
        python main.py \
            --config-name config.yaml \
            meta_dataset=openml-regression \
            dataset=${dataset} \
            shift_type=random_drop \
            shift_severity=0.6 \
            retrain=true \
            extra_config=conf/config_dem.yaml \
            seed=${seed} \
            out_dir=regression_dem/${dataset}/seed_${seed}/random_drop \
            log_dir=log/regression_dem/${dataset}/seed_${seed}/random_drop \
            device=cuda:${GPUS[i % ${NUM_GPUS}]} \
            2>&1 &
        wait_n
        i=$((i + 1))
        python main.py \
            --config-name config.yaml \
            meta_dataset=openml-regression \
            dataset=${dataset} \
            shift_type=column_drop \
            shift_severity=0.6 \
            retrain=true \
            extra_config=conf/config_dem.yaml \
            seed=${seed} \
            out_dir=regression_dem/${dataset}/seed_${seed}/column_drop \
            log_dir=log/regression_dem/${dataset}/seed_${seed}/column_drop \
            device=cuda:${GPUS[i % ${NUM_GPUS}]} \
            2>&1 &
        wait_n
        i=$((i + 1))
        python main.py \
            --config-name config.yaml \
            meta_dataset=openml-regression \
            dataset=${dataset} \
            shift_type=Gaussian \
            shift_severity=1 \
            retrain=true \
            extra_config=conf/config_dem.yaml \
            seed=${seed} \
            out_dir=regression_dem/${dataset}/seed_${seed}/Gaussian \
            log_dir=log/regression_dem/${dataset}/seed_${seed}/Gaussian \
            device=cuda:${GPUS[i % ${NUM_GPUS}]} \
            2>&1 &
        wait_n
        i=$((i + 1))
        python main.py \
            --config-name config.yaml \
            meta_dataset=openml-regression \
            dataset=${dataset} \
            shift_type=mean_std_shift \
            shift_severity=1 \
            retrain=true \
            extra_config=conf/config_dem.yaml \
            seed=${seed} \
            out_dir=regression_dem/${dataset}/seed_${seed}/mean_std_shift \
            log_dir=log/regression_dem/${dataset}/seed_${seed}/mean_std_shift \
            device=cuda:${GPUS[i % ${NUM_GPUS}]} \
            2>&1 &
        wait_n
        i=$((i + 1))
        python main.py \
            --config-name config.yaml \
            meta_dataset=openml-regression \
            dataset=${dataset} \
            shift_type=std_shift \
            shift_severity=1 \
            retrain=true \
            extra_config=conf/config_dem.yaml \
            seed=${seed} \
            out_dir=regression_dem/${dataset}/seed_${seed}/std_shift \
            log_dir=log/regression_dem/${dataset}/seed_${seed}/std_shift \
            device=cuda:${GPUS[i % ${NUM_GPUS}]} \
            2>&1 &
        wait_n
        i=$((i + 1))
        python main.py \
            --config-name config.yaml \
            meta_dataset=openml-regression \
            dataset=${dataset} \
            shift_type=mean_shift \
            shift_severity=1 \
            retrain=true \
            extra_config=conf/config_dem.yaml \
            seed=${seed} \
            out_dir=regression_dem/${dataset}/seed_${seed}/mean_shift \
            log_dir=log/regression_dem/${dataset}/seed_${seed}/mean_shift \
            device=cuda:${GPUS[i % ${NUM_GPUS}]} \
            2>&1 &
        wait_n
        i=$((i + 1))
    done
done