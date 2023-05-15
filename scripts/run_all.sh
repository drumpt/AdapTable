
############# run in single GPU ##############
GPUS=(0 1 2 3)
NUM_GPUS=4
##############################################
i=0

LOG_POSTFIX="main_table"

wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=20 #12
  local num_max_jobs=20
  echo $num_max_jobs
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}

openml_mlpbase(){
  SEED="0 1 2 3 4"
#  DATASET="cmc"

  DATASET="cmc semeion mfeat-karhunen optdigits diabetes mfeat-pixel dna"
  METHOD="mae sar em memo mae_random_mask"

  if [ $method = "mae" ] || [ $method = "mae_random_mask" ] || [ $method = "memo" ]; then
    episodic="true"
  else
    episodic="false"
  fi

  for dataset in $DATASET; do
    for method in $METHOD; do
      for seed in $SEED; do
        if [ $method = "sar" ]; then
          python main.py \
            method=$method \
            episodic=$episodic \
            meta_dataset=openml-cc18 \
            dataset="${dataset}" \
            shift_type=null \
            retrain=true \
            seed=${seed} \
            log_prefix=${LOG_POSTFIX} \
            device=cuda:${GPUS[i % ${NUM_GPUS}]} \
            --config-name config_sar.yaml \
            2>&1 &
          wait_n
          i=$((i + 1))
          python main.py \
              method=$method \
              episodic=$episodic \
              meta_dataset=openml-cc18 \
              dataset="${dataset}" \
              shift_type=random_drop \
              shift_severity=0.6 \
              retrain=true \
              seed=${seed} \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              --config-name config_sar.yaml \
              2>&1 &
          wait_n
          i=$((i + 1))
          python main.py \
              method=$method \
              episodic=$episodic \
              meta_dataset=openml-cc18 \
              dataset="${dataset}" \
              shift_type=column_drop \
              shift_severity=0.6 \
              retrain=true \
              seed=${seed} \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              --config-name config_sar.yaml \
              2>&1 &
          wait_n
          i=$((i + 1))
          python main.py \
              method=$method \
              episodic=$episodic \
              meta_dataset=openml-cc18 \
              dataset="${dataset}" \
              shift_type=Gaussian \
              shift_severity=1 \
              retrain=true \
              seed=${seed} \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              --config-name config_sar.yaml \
              2>&1 &
          wait_n
          i=$((i + 1))
          python main.py \
              method=$method \
              episodic=$episodic \
              meta_dataset=openml-cc18 \
              dataset="${dataset}" \
              shift_type=mean_std_shift \
              shift_severity=1 \
              retrain=true \
              seed=${seed} \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              --config-name config_sar.yaml \
              2>&1 &
          wait_n
          i=$((i + 1))
          python main.py \
              method=$method \
              episodic=$episodic \
              meta_dataset=openml-cc18 \
              dataset="${dataset}" \
              shift_type=std_shift \
              shift_severity=1 \
              retrain=true \
              seed=${seed} \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              --config-name config_sar.yaml \
              2>&1 &
          wait_n
          i=$((i + 1))
          python main.py \
              method=$method \
              episodic=$episodic \
              meta_dataset=openml-cc18 \
              dataset="${dataset}" \
              shift_type=mean_shift \
              shift_severity=1 \
              retrain=true \
              seed=${seed} \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              --config-name config_sar.yaml \
              2>&1 &
          wait_n
          i=$((i + 1))
        else
          python main.py \
            method=$method \
            episodic=$episodic \
            meta_dataset=openml-cc18 \
            dataset="${dataset}" \
            shift_type=null \
            retrain=true \
            seed=${seed} \
            log_prefix=${LOG_POSTFIX} \
            device=cuda:${GPUS[i % ${NUM_GPUS}]} \
            2>&1 &
          wait_n
          i=$((i + 1))
          python main.py \
              method=$method \
              episodic=$episodic \
              meta_dataset=openml-cc18 \
              dataset="${dataset}" \
              shift_type=random_drop \
              shift_severity=0.6 \
              retrain=true \
              seed=${seed} \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              2>&1 &
          wait_n
          i=$((i + 1))
          python main.py \
              method=$method \
              episodic=$episodic \
              meta_dataset=openml-cc18 \
              dataset="${dataset}" \
              shift_type=column_drop \
              shift_severity=0.6 \
              retrain=true \
              seed=${seed} \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              2>&1 &
          wait_n
          i=$((i + 1))
          python main.py \
              method=$method \
              episodic=$episodic \
              meta_dataset=openml-cc18 \
              dataset="${dataset}" \
              shift_type=Gaussian \
              shift_severity=1 \
              retrain=true \
              seed=${seed} \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              2>&1 &
          wait_n
          i=$((i + 1))
          python main.py \
              method=$method \
              episodic=$episodic \
              meta_dataset=openml-cc18 \
              dataset="${dataset}" \
              shift_type=mean_std_shift \
              shift_severity=1 \
              retrain=true \
              seed=${seed} \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              2>&1 &
          wait_n
          i=$((i + 1))
          python main.py \
              method=$method \
              episodic=$episodic \
              meta_dataset=openml-cc18 \
              dataset="${dataset}" \
              shift_type=std_shift \
              shift_severity=1 \
              retrain=true \
              seed=${seed} \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              2>&1 &
          wait_n
          i=$((i + 1))
          python main.py \
              method=$method \
              episodic=$episodic \
              meta_dataset=openml-cc18 \
              dataset="${dataset}" \
              shift_type=mean_shift \
              shift_severity=1 \
              retrain=true \
              seed=${seed} \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              2>&1 &
          wait_n
          i=$((i + 1))
        fi

      done
    done
  done
}

openml_conventional(){
  SEED="0 1 2 3 4"
  DATASET="cmc semeion mfeat-karhunen optdigits diabetes mfeat-pixel dna"
  MODEL="lr knn xgboost rf"
  METHOD="no_adapt"
  for dataset in $DATASET; do
    for method in $METHOD; do
      for model in $MODEL; do
        for seed in $SEED; do
          python main.py \
              method=$method \
              model=$model \
              meta_dataset=openml-cc18 \
              dataset="${dataset}" \
              shift_type=null \
              retrain=true \
              seed=${seed} \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              2>&1 &
          wait_n
          i=$((i + 1))
          python main.py \
              method=$method \
              model=$model \
              meta_dataset=openml-cc18 \
              dataset="${dataset}" \
              shift_type=random_drop \
              shift_severity=0.6 \
              retrain=true \
              seed=${seed} \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              2>&1 &
          wait_n
          i=$((i + 1))
          python main.py \
              method=$method \
              model=$model \
              meta_dataset=openml-cc18 \
              dataset="${dataset}" \
              shift_type=column_drop \
              shift_severity=0.6 \
              retrain=true \
              seed=${seed} \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              2>&1 &
          wait_n
          i=$((i + 1))
          python main.py \
              method=$method \
              model=$model \
              meta_dataset=openml-cc18 \
              dataset="${dataset}" \
              shift_type=Gaussian \
              shift_severity=1 \
              retrain=true \
              seed=${seed} \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              2>&1 &
          wait_n
          i=$((i + 1))
          python main.py \
              method=$method \
              model=$model \
              meta_dataset=openml-cc18 \
              dataset="${dataset}" \
              shift_type=mean_std_shift \
              shift_severity=1 \
              retrain=true \
              seed=${seed} \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              2>&1 &
          wait_n
          i=$((i + 1))
          python main.py \
              method=$method \
              model=$model \
              meta_dataset=openml-cc18 \
              dataset="${dataset}" \
              shift_type=std_shift \
              shift_severity=1 \
              retrain=true \
              seed=${seed} \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              2>&1 &
          wait_n
          i=$((i + 1))
          python main.py \
              method=$method \
              model=$model \
              meta_dataset=openml-cc18 \
              dataset="${dataset}" \
              shift_type=mean_shift \
              shift_severity=1 \
              retrain=true \
              seed=${seed} \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              2>&1 &
          wait_n
          i=$((i + 1))
        done
      done
    done
  done
}

tableshift_mlpbase(){
  SEED="0 1 2 3 4"
  DATASET="heloc mooc anes"
  METHOD="mae sar em memo mae_random_mask"

  if [ $method = "mae" ] || [ $method = "mae_random_mask" ] || [ $method = "memo" ]; then
    episodic="true"
  else
    episodic="false"
  fi

  for dataset in $DATASET; do
    for method in $METHOD; do
      for seed in $SEED; do
        if [ $method = "sar" ]; then
          python main.py \
            method=$method \
            episodic=$episodic \
            meta_dataset=tableshift \
            dataset="${dataset}" \
            retrain=true \
            seed=${seed} \
            log_prefix=${LOG_POSTFIX} \
            device=cuda:${GPUS[i % ${NUM_GPUS}]} \
            --config-name config_sar.yaml \
            2>&1 &
          wait_n
          i=$((i + 1))
        else
          python main.py \
            method=$method \
            episodic=$episodic \
            meta_dataset=tableshift \
            dataset="${dataset}" \
            retrain=true \
            seed=${seed} \
            log_prefix=${LOG_POSTFIX} \
            device=cuda:${GPUS[i % ${NUM_GPUS}]} \
            2>&1 &
          wait_n
          i=$((i + 1))
        fi

      done
    done
  done
}

tableshift_conventional(){
  SEED="0 1 2 3 4"
  DATASET="heloc mooc anes"
  MODEL="lr knn xgboost rf"
  METHOD="no_adapt"
  for dataset in $DATASET; do
    for method in $METHOD; do
      for model in $MODEL; do
        for seed in $SEED; do
          python main.py \
              method=$method \
              model=$model \
              meta_dataset=tableshift \
              dataset="${dataset}" \
              retrain=true \
              seed=${seed} \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              2>&1 &
          wait_n
          i=$((i + 1))
        done
      done
    done
  done
}

folktables_mlpbase(){
  SEED="0 1 2 3 4"
  DATASET="time state time_state"
  METHOD="mae sar em memo mae_random_mask"

  if [ $method = "mae" ] || [ $method = "mae_random_mask" ] || [ $method = "memo" ]; then
    episodic="true"
  else
    episodic="false"
  fi


  for dataset in $DATASET; do
    for method in $METHOD; do
      for seed in $SEED; do
        if [ $method = "sar" ]; then
          python main.py \
            meta_dataset=folktables \
            episodic=$episodic \
            method=$method \
            dataset="${dataset}" \
            retrain=true \
            seed=${seed} \
            log_prefix=${LOG_POSTFIX} \
            device=cuda:${GPUS[i % ${NUM_GPUS}]} \
            --config-name config_sar.yaml \
            2>&1 &
          wait_n
          i=$((i + 1))
        else
          python main.py \
            meta_dataset=folktables \
            episodic=$episodic \
            method=$method \
            dataset="${dataset}" \
            retrain=true \
            seed=${seed} \
            log_prefix=${LOG_POSTFIX} \
            device=cuda:${GPUS[i % ${NUM_GPUS}]} \
            2>&1 &
          wait_n
          i=$((i + 1))
        fi

      done
    done
  done
}

folktables_conventional(){
  SEED="0 1 2 3 4"
  DATASET="time state time_state"
  MODEL="lr knn xgboost rf"
  METHOD="no_adapt"
  for dataset in $DATASET; do
    for method in $METHOD; do
      for model in $MODEL; do
        for seed in $SEED; do
          python main.py \
              method=$method \
              model=$model \
              meta_dataset=folktables \
              dataset="${dataset}" \
              retrain=true \
              seed=${seed} \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              2>&1 &
          wait_n
          i=$((i + 1))
        done
      done
    done
  done
}
# mlp base
openml_mlpbase
wait
tableshift_mlpbase
wait
folktables_mlpbase

# conventional algorithms
#openml_conventional
#tableshift_conventional
#folktables_conventional

#run_semeion
#run_heloc
wait
python send_email.py
