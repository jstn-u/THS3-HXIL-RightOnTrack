import os
import json
import time
import math
import utils
import models
import logger
import inspect
import datetime
import argparse
import data_loader
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
import numpy as np

parser = argparse.ArgumentParser()

# basic args
parser.add_argument('--task', type=str)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=50)

# evaluation args
parser.add_argument('--weight_file', type=str)
parser.add_argument('--result_file', type=str)

# cnn args
parser.add_argument('--kernel_size', type=int)

# rnn args
parser.add_argument('--pooling_method', type=str)

# multi-task args
parser.add_argument('--alpha', type=float)

# data ratio
parser.add_argument('--data_ratio', type=float, default=1.0)

# log file name
parser.add_argument('--log_file', type=str)

args = parser.parse_args()

config = json.load(open('./config.json', 'r'))


def compute_metrics(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))

    nonzero_mask = np.abs(y_true) > eps
    if np.any(nonzero_mask):
        mape = np.mean(
            np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])
        ) * 100.0
    else:
        mape = float('nan')

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > eps else float('nan')

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }


def train(model, elogger, train_set, eval_set):
    elogger.log(str(model))
    elogger.log(str(args._get_kwargs()))

    model.train()

    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs("./saved_weights", exist_ok=True)

    best_rmse = float("inf")
    best_model_path = os.path.join("./saved_weights", "best_model.pt")

    for epoch in range(args.epochs):
        print('Training on epoch {}'.format(epoch))

        model.train()

        for input_file in train_set:
            print('Train on file {}'.format(input_file))

            data_iter = data_loader.get_loader(
                input_file,
                args.batch_size,
                args.data_ratio,
                args.kernel_size
            )

            running_loss = 0.0

            for idx, (attr, traj) in enumerate(data_iter):
                attr, traj = utils.to_var(attr), utils.to_var(traj)

                _, loss = model.eval_on_batch(attr, traj, config)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                print(
                    '\r Progress {:.2f}%, average loss {}'.format(
                        (idx + 1) * 100.0 / len(data_iter),
                        running_loss / (idx + 1.0)
                    ),
                    end=''
                )

            print()
            elogger.log(
                'Training Epoch {}, File {}, Loss {}'.format(
                    epoch, input_file, running_loss / (idx + 1.0)
                )
            )

        # save checkpoint
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        weight_name = f"{args.log_file}_epoch{epoch}_{timestamp}.pt"
        save_path = os.path.join("./saved_weights", weight_name)

        elogger.log('Save weight file {}'.format(save_path))
        torch.save(model.state_dict(), save_path)

        latest_path = os.path.join("./saved_weights", "latest.pt")
        torch.save(model.state_dict(), latest_path)

        print(f"Saved checkpoint to: {save_path}")
        print(f"Saved latest checkpoint to: {latest_path}")

        metrics = evaluate(
            model,
            elogger,
            eval_set,
            save_result=False,
            split_name='validation',
            verbose=False
        )

        val_rmse = metrics['RMSE']
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), best_model_path)
            elogger.log(
                'New best model saved at epoch {} with validation RMSE {:.6f}'.format(
                    epoch, best_rmse
                )
            )

    elogger.log('Best model path: {}'.format(best_model_path))
    elogger.log('Best validation RMSE: {:.6f}'.format(best_rmse))

        # load best model before final evaluation
    model.load_state_dict(torch.load(best_model_path))

    if torch.cuda.is_available():
        model.cuda()

    # final evaluation
    print("\n" + "="*50)
    print('Final metrics after {} epochs:'.format(args.epochs))
    print("="*50)

    evaluate(model, elogger, train_set, save_result=True, split_name='train', verbose=True)
    evaluate(model, elogger, eval_set, save_result=True, split_name='validation', verbose=True)

    print("="*50)

def write_result(fs, pred_dict, attr):
    pred = pred_dict['pred'].data.cpu().numpy()
    label = pred_dict['label'].data.cpu().numpy()

    for i in range(pred_dict['pred'].size()[0]):
        fs.write('%.6f %.6f\n' % (label[i][0], pred[i][0]))


def evaluate(model, elogger, eval_set, save_result=False, split_name='validation', verbose=True):
    model.eval()

    csv_file = None
    writer = None

    if save_result:
        csv_file = open(f'{split_name}_predictions.csv', 'w', newline='')
        writer = csv.writer(csv_file)

        writer.writerow([
            "actual_time_min",
            "predicted_time_min",
            "error_min",
            "error_sec",
            "error_percent"
        ])

    all_true = []
    all_pred = []

    for input_file in eval_set:
        data_iter = data_loader.get_loader(
            input_file,
            args.batch_size,
            args.data_ratio,
            args.kernel_size
        )

        for idx, (attr, traj) in enumerate(data_iter):
            attr, traj = utils.to_var(attr), utils.to_var(traj)

            pred_dict, loss = model.eval_on_batch(attr, traj, config)

            y_true = pred_dict['label'].data.cpu().numpy().reshape(-1)
            y_pred = pred_dict['pred'].data.cpu().numpy().reshape(-1)

            for t, p in zip(y_true, y_pred):
                error_min = abs(t - p)
                error_sec = error_min * 60
                error_pct = (error_min / t) * 100 if t != 0 else 0

                if save_result:
                    writer.writerow([
                        float(t),
                        float(p),
                        float(error_min),
                        float(error_sec),
                        float(error_pct)
                    ])

                all_true.append(t)
                all_pred.append(p)

    if csv_file is not None:
        csv_file.close()

    metrics = compute_metrics(all_true, all_pred)

    os.makedirs("outputs", exist_ok=True)

    metrics_file = os.path.join("outputs", "metrics.txt")

    with open(metrics_file, "a") as f:
        f.write(f"\n=== {split_name.upper()} ===\n")
        f.write(f"Pooling: {args.pooling_method}\n")
        f.write(f"Kernel Size: {args.kernel_size}\n")
        f.write(f"Alpha: {args.alpha}\n")
        f.write(f"Data Ratio: {args.data_ratio}\n")
        f.write(f"MAE  : {metrics['MAE']}\n")
        f.write(f"RMSE : {metrics['RMSE']}\n")
        f.write(f"MAPE : {metrics['MAPE']}\n")
        f.write(f"R2   : {metrics['R2']}\n")
        f.write("-"*40 + "\n")

    if verbose:
        print("=" * 50)
        print(f"{split_name.upper()} METRICS")
        print("MAE  :", metrics['MAE'])
        print("RMSE :", metrics['RMSE'])
        print("MAPE :", metrics['MAPE'])
        print("R2   :", metrics['R2'])
        print("=" * 50)

    elogger.log(str(metrics))

    return metrics


def get_kwargs(model_class):
    model_args = inspect.getfullargspec(model_class.__init__).args
    shell_args = dict(args._get_kwargs())

    kwargs = {}
    for arg in model_args:
        if arg == 'self':
            continue
        if arg in shell_args and shell_args[arg] is not None:
            kwargs[arg] = shell_args[arg]

    return kwargs


def run():
    kwargs = get_kwargs(models.DeepTTE)
    model = models.DeepTTE(**kwargs)

    elogger = logger.Logger(args.log_file)

    if args.task == 'train':
        train(model, elogger, train_set=config['train_set'], eval_set=config['eval_set'])

    elif args.task == 'test':
        model.load_state_dict(torch.load(args.weight_file))
        if torch.cuda.is_available():
            model.cuda()
        evaluate(model, elogger, config['test_set'], save_result=True, split_name='test')


if __name__ == '__main__':
    run()
