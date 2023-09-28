# text classification
# arguments

# 라이브러리
import os
import argparse
from utils.utils import parse_bool

# argument 설정
class ArgParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.proj_name = 'Text_classification'

        #Task arguments
        task_list = ['classification']
        self.parser.add_argument('--task', type = str, choices = task_list, default = 'classification',
                                 help = 'Task to do; Must be given.')
        job_list = ['preprocessing', 'training', 'resume_training', 'testing']
        self.parser.add_argument('--job', type = str, choices = job_list, default = 'training',
                                 help = 'Job to do: Must be given.')
        dataset_list = ['imdb']
        self.parser.add_argument('--task_dataset', type = str, choices = dataset_list, default = 'imdb',
                                 help = 'Dataset for the tassk; Must be given.')
        self.parser.add_argument('--description', type = str, default = 'default',
                                 help = 'Description of the experiment; Default is "default')
        
        # Path arguments - Modify these paths to fit your environment
        self.parser.add_argument('--data_path', type = str, default = f'./dataset/',
                                 help = 'Path to the raw dataset before preprocessing')
        self.parser.add_argument('--preprocess_path', type = str, default = f'./preprocessed/{self.proj_name}',
                                 help = 'Path to the preprocessed dataset.')
        self.parser.add_argument('--model_path', type = str, default = f'./model_final/{self.proj_name}',
                                 help = 'Path to the model after training.')
        self.parser.add_argument('--checkpoint_path', type = str, default = f'./model_checkpoint/{self.proj_name}')
        self.parser.add_argument('--result_path', type = str, default = f'./results/{self.proj_name}',
                                 help = 'Path to the result after testing.')
        self.parser.add_argument('--log_path', type = str, default = f'./tensorbord_log/{self.proj_name}',
                                 help = 'Path to the tensorboard log file.')
        
        # Model - Basic arguments
        self.parser.add_argument('--proj_name', type = str, default = 'Text_ImageClassification',
                                 help = 'Name of the project.')
        model_type_list = ['lstm', 'transformer_encoder', 'bert']
        self.parser.add_argument('--model_type', type = str, choices = model_type_list, default = 'rnn',
                                 help = 'Type of the classification model to use.')
        self.parser.add_argument('--model_ispretrained', type = parse_bool, default = True,
                                 help = 'Whether to use pretrained model; Default is True')
        self.parser.add_argument('--dropout_rate', type = float, default = 0.2,
                                 help = 'Dropout rate of the model; Default is 0.2')
        
        self.parser.add_argument('--embedding_dims', type = int, default = 768,
                                 help = 'Embedding_dimentions; Default is 768')
        self.parser.add_argument('--hidden_size', type = int, default = 768,
                                 help = 'Hidden size; Default is 100')
        self.parser.add_argument('--num_layers', type = int, default = 3,
                                 help = 'Num Layers; Default is 3')
        self.parser.add_argument('--max_seq_len', type=int, default=100,
                                 help='Maximum sequence length of the input; Default is 100')
        self.parser.add_argument('--num_transformer_heads', type=int, default=8,
                                 help='Num transformer heads; Default is 8')
        self.parser.add_argument('--num_transformer_layers', type=int, default=6,
                                 help='Num transformer layers; Default is 4')
        
        # Model - Size arguments
        # Model - Optimizer & Scheduler arguments
        optim_list = ['SGD', 'AdaDelta', 'Adam', 'AdamW']
        scheduler_list = ['None', 'StepLR', 'LambdaLR', 'CosineAnnealingLR', 'ConsineAnnealingWarmRestars', 'ReduceLROnPlateau']
        self.parser.add_argument('--optimizer', type = str, choices = optim_list, default = 'Adam',
                                 help = 'Optimzier to use; Default is Adam')
        self.parser.add_argument('--scheduler', type= str, choices = scheduler_list, default = 'None',
                                 help = 'Scheduler to use for classification; If None, no scheduler is used; Default is None')
        
        # Training arguments 1
        self.parser.add_argument('--num_epochs', type = int, default = 50,
                                 help = 'Training epochs; Default is 50')
        self.parser.add_argument('--learning_rate', type = float, default = 5e-5,
                                 help = 'Learning rate of optimizer; Default is 5e-5')
        
        # Training argument 2
        self.parser.add_argument('--num_workers', type = int, default = 2,
                                 help = 'Num CPU Wokers; Default is 2')
        self.parser.add_argument('--batch_size', type = int, default = 32,
                                 help = 'Batch size; Default is 32')
        self.parser.add_argument('--weight_decay', type = float, default = 0,
                                 help = 'Weight decay; Default is 5e-4; If 0, np weight decay')
        self.parser.add_argument('--clip_grad_norm', type = int, default = 5,
                                 help = 'Gradient clipping norm; Default is 5')
        self.parser.add_argument('--label_smoothing_eps', type = float, default = 0.05,
                                 help = 'Label smoothing epsilon; Default is 0.05')
        self.parser.add_argument('--early_stopping_patience', type = int, default = 5,
                                 help = 'Early stopping parience; No early stopping if None; Default is 5')
        self.parser.add_argument('--train_valid_split', type = float, default = 0.1,
                                 help = 'Train/Valud split ratio; Default is 0.1')
        objective_list = ['loss', 'accuracy', 'f1']
        self.parser.add_argument('--optimize_objective', type = str, choices = objective_list, default = 'accuracy',
                                 help = 'objective to optimzie; Default is accuracy')

        # Testing/Inference arguments
        self.parser.add_argument('--test_batch_size', default = 16, type = int,
                                 help = 'Batch size for test; Default is 16')
        
        # Other arguments - Device, Seed, Logging, etc.
        self.parser.add_argument('--device', type = str, default = 'cuda:2',
                                 help = 'Device to use for training; Default is cuda')
        self.parser.add_argument('--seed', type = int, default = 2023,
                                 help = 'Random sedd; Default is 2023')
        self.parser.add_argument('--use_tensorboard', type = parse_bool, default = True,
                                 help = 'Using wandb; Default is True')
        self.parser.add_argument('--use_wandb', type=parse_bool, default=True,
                                 help='Using wandb; Default is True')
        self.parser.add_argument('--log_freq', default = 500, type = int,
                                 help = 'Logging frequency; Default is 500')

    def get_args(self):
        return self.parser.parse_args()

        