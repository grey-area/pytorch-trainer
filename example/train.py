import torch
from configargparse import ArgumentParser
from model import Mobilenet
from pytorch_trainer import PytorchTrainer
from dataloaders import get_dataloaders


def minibatch_fn(iteration, minibatch, models, optimizers,
                 grad_clip_thresh, train):
    results = {}

    x, y = minibatch
    model = models['model']
    y_pred = model(x)

    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(y_pred, y)
    results['loss'] = loss.item()

    if train:
        optimizer = optimizers['model']
        optimizer.zero_grad()
        loss.backward()
        results['grad_norm'] = torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip_thresh
        ).item()
        optimizer.step()

    return results


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', is_config_file=True)
    parser.add_argument('--output-name', type=str, default='default')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--iterations-per-validation', type=int, default=200)
    parser.add_argument('--iterations-per-checkpoint', type=int, default=1000)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    train_dataloader, valid_dataloader = get_dataloaders(args.batch_size)
    model = Mobilenet()

    trainer = PytorchTrainer(
        model_names=['model'],
        model_list=[model],
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        minibatch_fn=minibatch_fn,
        total_iterations=args.iterations,
        iterations_per_validation=args.iterations_per_validation,
        iterations_per_checkpoint=args.iterations_per_checkpoint,
        output_name=args.output_name,
        config=args
    )

    trainer.run()
