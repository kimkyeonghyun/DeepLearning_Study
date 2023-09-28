DATASET=imdb
BS=32
LR=5e-5
EP=10
DEVICE=cuda
clear

# MODEL=lstm
# python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
# python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --device=${DEVICE}
# python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --device=${DEVICE}

# MODEL=transformer_encoder
# python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
# python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --device=${DEVICE}
# python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --device=${DEVICE}

MODEL=bert
# python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --device=${DEVICE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --device=${DEVICE}
