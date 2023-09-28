# 3주차 text classification
# IMDB 데이터셋을 Custom Dataset을 활용하여 Load

# 라이브러리 
import time
import argparse

from utils.arguments import ArgParser
from utils.utils import check_path, set_random_seed

def main(args: argparse.Namespace) -> None:
    # Set random see
    if args.seed is not None:
        set_random_seed(args.seed)

    start_time = time.time()

    # 경로 존재 확인
    for path in []:
        check_path(path)

    # 할 job 얻기
    if args.job == None:
        raise ValueError('Please specify the job to do.')
    else:
        if args.task == 'classification':
            if args.job == 'preprocessing':
                from task.preprocessing import preprocessing as job
            elif args.job in ['training', 'resume_training']:
                from task.train import training as job
            elif args.job == 'testing':
                from task.test import testing as job
            else:
                raise ValueError(f'Invalid job: {args.job}')
        else:
            raise ValueError(f'Invalid task: {args.task}')
        
    # job 하기
    job(args)

    elapsed_time = time.time() - start_time
    print(f'Completed {args.job}; Time elapsed: {elapsed_time / 60:.2f} minutes')

if __name__ == '__main__':
    parser = ArgParser()
    args = parser.get_args()

    main(args)