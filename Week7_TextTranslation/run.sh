DATASET=wmt2016
BS=32
LR=5e-5
EP=10
DEVICE=cuda
clear

MODEL=helsinki_hand
# python main.py --task=translate --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=translate --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
python main.py --task=translate --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

MODEL=helsinki
# python main.py --task=translate --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=translate --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
python main.py --task=translate --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

