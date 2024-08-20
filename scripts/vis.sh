############# run in single GPU ##############
GPUS=(1 2 3)
NUM_GPUS=3
##############################################
i=0

LOG_DIR="log"
LOG_POSTFIX="231007_vis"
CONF_DIR="conf/baseline_config"

wait_n() {
	#limit the max number of jobs as NUM_MAX_JOB and wait
	background=($(jobs -p))
	local default_num_jobs=8
	local num_max_jobs=8
	echo $num_max_jobs
	if ((${#background[@]} >= num_max_jobs)); then
		wait -n
	fi
}

openml_vis() {
	SEEDS="0"
	MODELS="mlp tabnet fttransformer"
	# MODELS="tabnet"
	METHODS="ours"
	DATASETS="cmc mfeat-pixel dna"
	# DATASETS="dna"
	benchmark="openml-cc18"

	for seed in $SEEDS; do
		for dataset in $DATASETS; do
			for model in $MODELS; do
				for method in $METHODS; do
					python main.py \
						seed=${seed} \
						log_dir=$LOG_DIR \
						log_prefix=${LOG_POSTFIX} \
						device=cuda:${GPUS[i % ${NUM_GPUS}]} \
						out_dir=${LOG_POSTFIX} \
						benchmark=$benchmark \
						dataset="${dataset}" \
						shift_type=null \
						shift_severity=0 \
						vis=true \
						retrain=false \
						--config-name ours_${model}.yaml \
						2>&1 &
					wait_n
					i=$((i + 1))
					#   python main.py \
					#       seed=${seed} \
					#       log_dir=$LOG_DIR \
					#       log_prefix=${LOG_POSTFIX} \
					#       device=cuda:${GPUS[i % ${NUM_GPUS}]} \
					#       out_dir=${LOG_POSTFIX} \
					#       benchmark=$benchmark \
					#       dataset="${dataset}" \
					#       shift_type=categorical \
					#       shift_severity=0.5 \
					#       --config-name ours_${model}.yaml \
					#       2>&1 &
					#   wait_n
					#   i=$((i + 1))
					#   python main.py \
					#       seed=${seed} \
					#       log_dir=$LOG_DIR \
					#       log_prefix=${LOG_POSTFIX} \
					#       device=cuda:${GPUS[i % ${NUM_GPUS}]} \
					#       out_dir=${LOG_POSTFIX} \
					#       benchmark=$benchmark \
					#       dataset="${dataset}" \
					#       shift_type=Gaussian \
					#       shift_severity=0.1 \
					#       --config-name ours_${model}.yaml \
					#       2>&1 &
					#   wait_n
					#   i=$((i + 1))
					#   python main.py \
					#       seed=${seed} \
					#       log_dir=$LOG_DIR \
					#       log_prefix=${LOG_POSTFIX} \
					#       device=cuda:${GPUS[i % ${NUM_GPUS}]} \
					#       out_dir=${LOG_POSTFIX} \
					#       benchmark=$benchmark \
					#       dataset="${dataset}" \
					#       shift_type=uniform \
					#       shift_severity=0.1 \
					#       --config-name ours_${model}.yaml \
					#       2>&1 &
					#   wait_n
					#   i=$((i + 1))
					#   python main.py \
					#       seed=${seed} \
					#       log_dir=$LOG_DIR \
					#       log_prefix=${LOG_POSTFIX} \
					#       device=cuda:${GPUS[i % ${NUM_GPUS}]} \
					#       out_dir=${LOG_POSTFIX} \
					#       benchmark=$benchmark \
					#       dataset="${dataset}" \
					#       shift_type=random_drop \
					#       shift_severity=0.2 \
					#       --config-name ours_${model}.yaml \
					#       2>&1 &
					#   wait_n
					#   i=$((i + 1))
					#   python main.py \
					#       seed=${seed} \
					#       log_dir=$LOG_DIR \
					#       log_prefix=${LOG_POSTFIX} \
					#       device=cuda:${GPUS[i % ${NUM_GPUS}]} \
					#       out_dir=${LOG_POSTFIX} \
					#       benchmark=$benchmark \
					#       dataset="${dataset}" \
					#       shift_type=column_drop \
					#       shift_severity=0.2 \
					#       --config-name ours_${model}.yaml \
					#       2>&1 &
					#   wait_n
					#   i=$((i + 1))
				done
			done
		done
	done
}

tableshift_vis() {
	SEEDS="0"
	MODELS="mlp tabnet fttransformer"
	# MODELS="tabnet"
	METHODS="ours"
	DATASETS="heloc diabetes_readmission anes"
	# DATASETS="diabetes_readmission"
	benchmark="tableshift"

	for seed in $SEEDS; do
		for dataset in $DATASETS; do
			for model in $MODELS; do
				for method in $METHODS; do
					python main.py \
						seed=${seed} \
						log_dir=$LOG_DIR \
						log_prefix=${LOG_POSTFIX} \
						device=cuda:${GPUS[i % ${NUM_GPUS}]} \
						out_dir=${LOSG_LOG_POSTFIX}_${model}_${dataset}_${seed} \
						benchmark=$benchmark \
						dataset="${dataset}" \
						shift_type=None \
						retrain=true \
						shift_severity=0 \
						vis=true \
						--config-name ours_${model}.yaml \
						2>&1 &
					wait_n
					i=$((i + 1))
				done
			done
		done
	done
}

openml_vis
# tableshift_vis

#python send_email.py
