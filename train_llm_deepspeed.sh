export HYDRA_FULL_ERROR=1
nohup accelerate launch --config_file "config/accelerate/accelerate_config.yaml" code/classifier/train_llm.py > llm.log 2>&1 &