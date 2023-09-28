# text translation
# train

# 라이브러리
import os
import sys
import shutil
import logging
import argparse
from tqdm.auto import tqdm
from nltk.translate.bleu_score import sentence_bleu
import torch
torch.set_num_threads(2) # tokenizer가  모든 cpu에 할당하는걸 방지
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.translation.model import TranslationModel, postprocess
from model.translation.dataset import CustomDataset
from model.optimizer.optimizer import get_optimizer
from model.optimizer.scheduler import get_scheduler
from utils.utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_wandb_exp_name, get_torch_device, check_path
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate import bleu_score


def training(args: argparse.Namespace) -> None:
    device = get_torch_device(args.device)
    # logger 정의
    logger = logging.getLogger(__name__)
    if len(logger.handlers) > 0:
        logger.handlers.clear() # logger 핸들러 삭제
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    # dataset을 load하고 dataloader 정의
    write_log(logger, "Loading data")
    dataset_dict, dataloader_dict = {}, {}
    dataset_dict['train'] = CustomDataset(os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type, f'train_processed.pkl'))
    dataset_dict['valid'] = CustomDataset(os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type, f'valid_processed.pkl'))

    dataloader_dict['train'] = DataLoader(dataset_dict['train'], batch_size = args.batch_size, num_workers = args.num_workers,
                                          shuffle = True, pin_memory = True, drop_last = True)
    # pin_menory: 데이터를 옮길 떄 GPU를 사용하기 위한 옵션
    dataloader_dict['valid'] = DataLoader(dataset_dict['valid'], batch_size = args.batch_size, num_workers = args.num_workers,
                                          shuffle = False, pin_memory = True, drop_last = True)
    args.src_vocab_size = dataset_dict['train'].src_vocab_size
    args.tgt_vocab_size = dataset_dict['train'].tgt_vocab_size

    write_log(logger, 'Loaded data successfully')
    write_log(logger, f'Train dataset size / iterations: {len(dataset_dict["train"])} / {len(dataloader_dict["train"])}')
    write_log(logger, f'Valid dataset size / iterations: {len(dataset_dict["valid"])} / {len(dataloader_dict["valid"])}')

    # 모델 instance
    write_log(logger, 'Building model')
    model = TranslationModel(args).to(device)

    # Optimizer와 scheduler 정의
    write_log(logger, "Building optimizer and scheduler")
    optimizer = get_optimizer(model, learning_rate=args.learning_rate, weight_decay=args.weight_decay, optim_type=args.optimizer)
    scheduler = get_scheduler(optimizer, len(dataloader_dict['train']), num_epochs=args.num_epochs,
                              early_stopping_patience=args.early_stopping_patience, learning_rate=args.learning_rate,
                              scheduler_type=args.scheduler)
    write_log(logger, f"Optimizer: {optimizer}")
    write_log(logger, f"Scheduler: {scheduler}")

    # loss function 정의
    cls_loss = nn.CrossEntropyLoss(label_smoothing = args.label_smoothing_eps)
    write_log(logger, f'Loss function: {cls_loss}')

    # resume_training인경우 checkpoint 로드
    # resume_training: 중단된 train지점에서 다시 train
    start_epoch = 0
    if args.job == 'resume_training':
        write_log(logger, 'Resuming training model')
        load_checkpoint_name = os.path.join(args.checkpoint_path, args.task, args.task_dataset, args.model_type,
                                            f'checkpoint.pt')
        model = model.to('cpu')
        checkpoint = torch.load(load_checkpoint_name, map_location = 'cpu')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        model = model.to(device)
        write_log(logger, f'Loaded checkpoint from {load_checkpoint_name}')

        if args.use_wandb:
            import wandb # Only import wandb when it is used
            from wandb import AlertLevel
            wandb.init(project=args.proj_name,
                       name=get_wandb_exp_name(args),
                       config=args,
                       notes=args.description,
                       tags=["TRAIN",
                             f"Dataset: {args.task_dataset}",
                             f"Model: {args.model_type}"],
                       resume=True,
                       id=checkpoint['wandb_id'])
            wandb.watch(models=model, criterion=cls_loss, log='all', log_freq=10)
        del checkpoint
        
    # tensorboard writer 초기화
    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(args.log_path, get_tb_exp_name(args)))
        writer.add_text('args', str(args))

    # wandb 초기화
    if args.use_wandb and args.job == 'training':
        import wandb
        from wandb import AlertLevel
        wandb.init(project=args.proj_name,
                       name=get_wandb_exp_name(args),
                       config=args,
                       notes=args.description,
                       tags=["TRAIN",
                             f"Dataset: {args.task_dataset}",
                             f"Model: {args.model_type}"])
        wandb.watch(models=model, criterion=cls_loss, log='all', log_freq=10)

    # Train/Valid - trainig 시작
    best_epoch_idx = 0
    best_valid_objective_value = None
    early_stopping_counter = 0

    write_log(logger, f'Start training from epoch {start_epoch}')
    for epoch_idx in range(start_epoch, args.num_epochs):
        # Train - train 모드로 model 정함
        model = model.train()
        train_loss_cls = 0
        train_bleu_cls = 0
        train_total_bleu = 0
        # Train - 한 epoch 반복
        for iter_idx, data_dicts in enumerate(tqdm(dataloader_dict['train'], total=len(dataloader_dict['train']), desc=f'Training - Epoch [{epoch_idx}/{args.num_epochs}]', position=0, leave=True)):
            src = data_dicts['EN_text_ids'].to(device) # torch.Size([32, 100])
            tgt = data_dicts['DE_text_ids'].to(device) # torch.Size([32, 100])
            src_attention_mask = data_dicts['src_attention_mask'].to(device)
            tgt_attention_mask = data_dicts['tgt_attention_mask'].to(device)

            logits = model(src, src_attention_mask, tgt, tgt_attention_mask)
            batch_loss_cls = cls_loss(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
            
            # generated_tokens = model.generator(src, args.max_seq_len)

            # decoded_preds, decoded_labels = postprocess(args, generated_tokens, tgt)
            # for label, pred in zip(decoded_labels, decoded_preds):
            #     score = bleu_score.sentence_bleu(label, pred)
            #     train_total_bleu += score
            # batch_bleu_cls = train_total_bleu / len(decoded_preds)
            # batch_bleu_cls = compute_metrics(args, logits, tgt)
            # Train - Backward pass
            optimizer.zero_grad()
            batch_loss_cls.backward()
            if args.clip_grad_norm > 0:
                # gradient exploding 방지하여 학습의 안정화
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            if args.scheduler in ['StepLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']:
                scheduler.step() # training 반복후 step()
            
            # Train - logging
            train_loss_cls += batch_loss_cls.item()
            # train_bleu_cls += batch_bleu_cls
            # train_BLEU_cls += batch_bleu_cls
            # train_f1_cls += batch_f1_cls
            
            if iter_idx % args.log_freq == 0 or iter_idx == len(dataloader_dict['train']) - 1:
                write_log(logger, f'TRAIN - Epoch [{epoch_idx} / {args.num_epochs}] - Iter[{iter_idx} / {len(dataloader_dict["train"])}] - Loss: {batch_loss_cls.item():.4f}')
                # write_log(logger, f'TRAIN - Epoch [{epoch_idx} / {args.num_epochs}] - Iter[{iter_idx} / {len(dataloader_dict["train"])}] - BLEU: {batch_bleu_cls:.4f}')
            if args.use_tensorboard:
                # 스칼라 기록(tag, scalar_value, global_step) - tensorboard
                writer.add_scalar('TRAIN / Learning_Rate', optimizer.param_groups[0]['lr'], epoch_idx * len(dataloader_dict['train']) + iter_idx)
        
        # Train - epoch logging 종료
        if args.use_tensorboard:
            writer.add_scalar('TRAIN/Loss', train_loss_cls / len(dataloader_dict['train']), epoch_idx)
            # writer.add_scalar('TRAIN/BLEU', train_bleu_cls / len(dataloader_dict['train']), epoch_idx)
            # writer.add_scalar('TRAIN/F1', train_f1_cls / len(dataloader_dict['train']), epoch_idx)

        # Valid - model을 평가모드로 설정
        model = model.eval()
        valid_loss_cls = 0
        # valid_BLEU_cls = 0
        # valid_total_bleu = 0
        # Valid - 한 epoch 반복
        for iter_idx, data_dicts in enumerate(tqdm(dataloader_dict['valid'], total = len(dataloader_dict['valid']), desc = f'Validating - Epoch [{epoch_idx} / {args.num_epochs}]', position = 0, leave = True)):
            # Valid - input data 얻기
            src = data_dicts['EN_text_ids'].to(device)
            tgt = data_dicts['DE_text_ids'].to(device)
            # print(data_dicts['attention_mask'])
            src_attention_mask = data_dicts['src_attention_mask'].to(device)
            tgt_attention_mask = data_dicts['tgt_attention_mask'].to(device)

            # Valid - 전파
            with torch.no_grad():

                logits = model(src, src_attention_mask, tgt, tgt_attention_mask)
                # generated_tokens = model.generator(src, args.max_seq_len)

                # decoded_preds, decoded_labels = postprocess(args, generated_tokens, tgt)
                # for label, pred in zip(decoded_labels, decoded_preds):
                #     score = bleu_score.sentence_bleu(label, pred)
                #     valid_total_bleu += score
                # batch_bleu_cls = valid_total_bleu / len(decoded_preds)
            # calculate_bleu(tgt, output_text)
            # Valid - loss, BLEUuracy, f1 score 계산
            batch_loss_cls = cls_loss(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
            # batch_bleu_cls = score
            
            # Valid - logging
            valid_loss_cls += batch_loss_cls.item()
            # valid_BLEU_cls += batch_bleu_cls
            # valid_bleu_cls += batch_bleu_score_cls
            if iter_idx % args.log_freq == 0 or iter_idx == len(dataloader_dict['valid']) - 1:
                write_log(logger, f"VALID - Epoch [{epoch_idx} / {args.num_epochs}] - Iter [{iter_idx} / {len(dataloader_dict['valid'])}] - Loss: {batch_loss_cls.item():.4f}")
                # write_log(logger, f"VALID - Epoch [{epoch_idx} / {args.num_epochs}] - Iter [{iter_idx} / {len(dataloader_dict['valid'])}] - BLEU: {batch_bleu_cls:.4f}")

        # Valid - scheduler 불러오기
        if args.scheduler == 'LambdaLR':
            scheduler.step()
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(valid_loss_cls)
        
        # Valid - loss 확인 및 model 저장 - 누적 평균 valid 손실, BLEUuracy, f1 score
        valid_loss_cls /= len(dataloader_dict['valid'])
        # valid_BLEU_cls /= len(dataloader_dict['valid'])
        # valid_f1_cls /= len(dataloader_dict['valid'])

        if args.optimize_objective == 'loss':
            valid_objective_value = valid_loss_cls
            valid_objective_value = -1 * valid_objective_value # loss는 최소화, ojective value는 최대화
        # elif args.optimize_objective == 'bleu':
        #     valid_objective_value = valid_BLEU_cls
        else:
            raise NotImplementedError

        # valid_object_value 업데이트 및 log 출력 및 checkpoint 저장
        if best_valid_objective_value is None or valid_objective_value > best_valid_objective_value:
            best_valid_objective_value = valid_objective_value
            best_epoch_idx = epoch_idx
            write_log(logger, f"VALID - Saving checkpoint for best valid {args.optimize_objective}...")
            early_stopping_counter = 0 # early stopping counter 초기화

            checkpoint_save_path = os.path.join(args.checkpoint_path, args.task, args.task_dataset, args.model_type)
            check_path(checkpoint_save_path)

            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None
            }, os.path.join(checkpoint_save_path, f'checkpoint.pt'))
            write_log(logger, f"VALID - Best valid at epoch {best_epoch_idx} - {args.optimize_objective}: {abs(best_valid_objective_value):.4f}")
            write_log(logger, f"VALID - Saved checkpoint to {checkpoint_save_path}")
        else:
            early_stopping_counter += 1
            write_log(logger, f'VALID - Early stopping counter: {early_stopping_counter} / {args.early_stopping_patience}')
        
        # Valid - epoch logging 마침
        if args.use_tensorboard:
            writer.add_scalar('VALID / Loss', valid_loss_cls, epoch_idx)
            # writer.add_scalar('VALID / BLEU', valid_BLEU_cls, epoch_idx)
            # writer.add_scalar('VALID / F1', valid_f1_cls, epoch_idx)
        if args.use_wandb:
            wandb.log({'TRAIN / Epoch_Loss': train_loss_cls / len(dataloader_dict['train']),
                       'TRAIN / Epoch_BLEU': train_bleu_cls / len(dataloader_dict['train']),
                    #    'TRAIN / Epoch_F1': train_f1_cls / len(dataloader_dict['train']),
                       'VALID / Epoch_Loss': valid_loss_cls,
                    #    'VALID / Epoch_BLEU': valid_BLEU_cls,
                    #    'VALID / Epoch_F1': valid_f1_cls,
                       'Epoch_Index': epoch_idx})
            wandb.alert(
                title = 'Epoch End',
                text = f'VALID - Epoch {epoch_idx} - Loss: {valid_loss_cls:.4f}',
                level = AlertLevel.INFO,
                wait_duration = 300
            )

        # Valid - early stopping
        if early_stopping_counter >= args.early_stopping_patience:
            write_log(logger, f'VALID - Early stopping at epoch {epoch_idx}...')
            break

    # Final - training 마침
    write_log(logger, f'Done! Best valid at epoch {best_epoch_idx} - {args.optimize_objective}: {abs(best_valid_objective_value):.4f}')
    if args.use_tensorboard:
        writer.add_text("VALID / BEST", f'Best valid at epoch {best_epoch_idx} - {args.optimize_objective}: {abs(best_valid_objective_value):.4f}')
        writer.close()
            
    # Final - 가장 좋은 checkpoint를 결과 모델로 저장
    final_model_save_path = os.path.join(args.model_path, args.task, args.task_dataset, args.model_type)
    check_path(final_model_save_path)
    # 파일과 폴더를 복사하는 모델: shutil
    shutil.copyfile(os.path.join(checkpoint_save_path, 'checkpoint.pt'), os.path.join(final_model_save_path, 'final_model.pt')) # Copy best checkpoint as final model
    write_log(logger, f"FINAL - Saved final model to {final_model_save_path}")

    if args.use_wandb:
        wandb.finish()