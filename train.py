"""Text style transfer

This is a simplified implementation of:

Toward Controlled Generation of Text, ICML2017
Zhiting Hu, Zichao Yang, Xiaodan Liang, Ruslan Salakhutdinov, Eric Xing

Download the data with the cmd:

$ python prepare_data.py --dataset <dataset>

Train the model with the cmd:

$ python train_torch.py --config config --dataset <dataset>
"""

import os
import argparse
import importlib
import torch
import math
import texar.torch as tx
from tqdm import tqdm
from ctrl_gen_model import CtrlGenModel

try:
    import wandb
except ImportError:
    wandb = None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running PyTorch using {device}")


# Eval and compute BLEU
def eval(model, data_iterator):
    avg_meters = tx.utils.AverageRecorder()

    for batch in data_iterator:
        transferred = model.infer(inputs=batch)
        transferred = model.vocab.map_ids_to_tokens_py(transferred.cpu())
        original = model.vocab.map_ids_to_tokens_py(batch['text_ids'].cpu())

        bleu = tx.evals.corpus_bleu(original, transferred)
        avg_meters.add(bleu)

        data_iterator.set_description(f'BLEU: {avg_meters.to_str(precision=4)}')
        if wandb is not None:
            wandb.log({'BLEU': avg_meters.avg()})

    return avg_meters.avg()


def main():
    parser = argparse.ArgumentParser(description='Model for text style transfer')
    parser.add_argument('--config',
                        default='config',
                        help='The config to use.')
    parser.add_argument('--dataset',
                        help='The name of the dataset to use.',
                        default='yelp')
    parser.add_argument('--base-path',
                        help='base path for the dataset dir',
                        default='./data')
    parser.add_argument('--offline',
                        help='If true will run wandb offline',
                        action='store_true')
    parser.add_argument('--save-checkpoints',
                        help='If true will store checkpoints every 10% of the training process',
                        action='store_true')
    parser.add_argument('--load-checkpoint',
                        help='Whether to start again from the last checkpoint',
                        action='store_true')
    args = parser.parse_args()
    config = importlib.import_module(args.config)

    if wandb is not None:
        mode = 'offline' if args.offline else 'online'
        wandb.init(project="a-project",
                   entity="a-name",
                   mode=mode,
                   config=config.model,
                   settings=wandb.Settings(start_method='fork'))
        config = wandb.config
    else:
        config = tx.HParams(config.model, None)
    checkpoint_path = os.path.join(config.checkpoint_path, 'ckpt.pth')

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Data
    def dataset_config(name):
        return {
            'batch_size': config.batch_size,
            'seed': config.seed,
            'datasets': [
                {
                    'files': f'{args.base_path}/{args.dataset}/sentiment.{name}.text',
                    'vocab_file': f'{args.base_path}/{args.dataset}/vocab',
                    'data_name': ''
                },
                {
                    'files': f'{args.base_path}/{args.dataset}/sentiment.{name}.labels',
                    'data_name': 'labels',
                    'data_type': 'int'
                }
            ],
            'name': 'train'
        }
    train_data = tx.data.MultiAlignedData(dataset_config('train'),
                                          device=device)
    val_data = tx.data.MultiAlignedData(dataset_config('dev'),
                                        device=device)
    test_data = tx.data.MultiAlignedData(dataset_config('test'),
                                         device=device)
    vocab = train_data.vocab(0)

    # Each training batch is used twice: once for updating the generator and
    # once for updating the discriminator. Feedable data iterator is used for
    # such case.
    iterator = tx.data.DataIterator(
        {'train_g': train_data, 'train_d': train_data,
         'val': val_data, 'test': test_data})
    input_len = iterator.get_iterator('train_d').__next__()['text_ids'].size(1)-1

    # Model
    gamma_decay = config.gamma_decay

    model = CtrlGenModel(input_len, vocab,
                         config, device).to(device)

    optim_g = tx.core.get_optimizer(model.g_params(),
                                    hparams=model._hparams.opt)
    optim_d = tx.core.get_optimizer(model.d_params(),
                                    hparams=model._hparams.opt)

    gamma = config.gamma
    lambda_g = 0

    initial_epoch = 1
    if args.load_checkpoint:
        print(f'Restoring checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim_g.load_state_dict(checkpoint['optim_g'])
        optim_d.load_state_dict(checkpoint['optim_d'])
        initial_epoch = checkpoint['epoch']+1
        if initial_epoch > config.pretrain_nepochs+1:
            lambda_g = config.lambda_g

    train_g = tx.core.get_train_op(optimizer=optim_g)
    train_d = tx.core.get_train_op(optimizer=optim_d)

    print(f'Starting training from epoch {initial_epoch}')

    # Train
    for epoch in range(initial_epoch, config.max_nepochs + 1):
        if epoch == config.pretrain_nepochs+1:
            lambda_g = config.lambda_g
            optim_g = tx.core.get_optimizer(model.g_params(),
                                            hparams=model._hparams.opt)
            train_g = tx.core.get_train_op(optimizer=optim_g)

        if epoch > config.pretrain_nepochs:
            # Anneals the gumbel-softmax temperature
            gamma = max(0.001, config.gamma * (gamma_decay ** (epoch-config.pretrain_nepochs)))
        print(f'gamma: {gamma}, lambda_g: {lambda_g}')
        if wandb is not None:
            wandb.log({'Epoch': epoch, 'gamma': gamma, 'lambda_g': lambda_g})

        avg_meters_d = tx.utils.AverageRecorder(size=10)
        avg_meters_g = tx.utils.AverageRecorder(size=10)
        data_iterator = zip(iterator.get_iterator('train_d'),
                            iterator.get_iterator('train_g'))
        if wandb is None or args.offline:
            data_iterator = tqdm(data_iterator,
                                 total=int(len(train_data)/train_data.batch_size))

        for batch_d, batch_g in data_iterator:
            loss_d, accu_d = model.forward(batch_d, step='d')
            loss_d.backward()
            train_d()
            avg_meters_d.add(accu_d)

            loss_g, accu_g = model.forward(batch_g, step='g', gamma=gamma, lambda_g=lambda_g)
            loss_g.backward()
            train_g()
            avg_meters_g.add(accu_g)
            if wandb is None or args.offline:
                data_iterator.set_description(f'Accu_d: {avg_meters_d.to_str(precision=4)}, '
                                              + f'Accu_g: {avg_meters_g.to_str(precision=4)}')
            if wandb is not None:
                accu_g = avg_meters_g.avg()
                accu_g, accu_g_gdy = accu_g[0].item(), accu_g[1].item()
                wandb.log({'Accuracy D': avg_meters_d.avg().item(),
                           'Accuracy G': accu_g,
                           'Accuracy G GDY': accu_g_gdy})
        # checkpoint for evaluation
        model_state = {'model_state_dict': model.state_dict(),
                       'input_len': input_len,
                       'dataset': args.dataset}
        # checkpoint for resuming training
        torch.save({'model_state_dict': model.state_dict(),
                    'optim_d': optim_d.state_dict(),
                    'optim_g': optim_g.state_dict(),
                    'epoch': epoch},
                   checkpoint_path)
        if epoch % math.ceil(config.max_nepochs / 10) == 0:
            if args.save_checkpoints:
                torch.save(model_state,
                           os.path.join(config.checkpoint_path, f'ckpt_epoch_{epoch}.pth'))
    torch.save(model_state,
               os.path.join(config.checkpoint_path, 'final_model.pth'))

    # Eval
    val_iterator = tqdm(iterator.get_iterator('val'),
                        total=int(len(val_data)/val_data.batch_size))
    print("Eval BLEU score: ", eval(model, val_iterator))

    # Test
    test_iterator = tqdm(iterator.get_iterator('test'),
                         total=int(len(test_data)/test_data.batch_size))
    print("Eval BLEU score: ", eval(model, test_iterator))


if __name__ == '__main__':
    main()
