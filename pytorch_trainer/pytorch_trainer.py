import torch
import torch.optim as optim

from apex import amp
import tensorboardX

import types
import math
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict



def zero_backward_step(opt, loss, grad_clip_thresh=None):
    opt.zero_grad()

    with amp.scale_loss(loss, opt) as scaled_loss:
            scaled_loss.backward()

    if grad_clip_thresh is not None:
        opt.recorded_grad_norm = torch.nn.utils.clip_grad_norm_(
            amp.master_params(opt), grad_clip_thresh
        )

    opt.step()


class PytorchTrainer:
    def __init__(self, model_names, model_list,
                 train_dataloader,
                 valid_dataloader,
                 minibatch_fn,
                 checkpoint_load_path='',
                 output_path='outputs/default',
                 learning_rate=1e-3, weight_decay=1e-4,
                 grad_clip_thresh=1.0,
                 num_learning_rate_updates=100,
                 total_learning_rate_decay=0.01,
                 total_iterations=100000,
                 iterations_per_validation=1000,
                 iterations_per_checkpoint=10000,
                 seed=1234):
        # Set seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Checkpoint saving path
        output_path = Path(output_path)
        self.checkpoint_save_path = output_path / 'checkpoints'
        self.checkpoint_save_path.mkdir(parents=True, exist_ok=True)

        # Logger
        log_path = output_path / 'logs'
        log_path.mkdir(parents=True, exist_ok=True)
        self.log_writer = tensorboardX.SummaryWriter(log_path)

        torch.backends.cudnn.benchmark = True

        # Move models to GPU
        for model in model_list:
            model.cuda()

        # Set up optimizers
        optimizers = []
        self.learning_rate = learning_rate
        self.grad_clip_thresh = grad_clip_thresh
        for model in model_list:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay
            )
            optimizer.zero_backward_step = types.MethodType(
                zero_backward_step, optimizer
            )
            optimizers.append(optimizer)

        model_list, optimizers = amp.initialize(
            model_list,
            optimizers,
            opt_level='O1',
            loss_scale='dynamic'
        )

        self.models = {}
        self.optimizers = {}
        for name, model, optimizer in zip(model_names, model_list, optimizers):
            self.models[name] = model
            self.optimizers[name] = optimizer

        # Dataloaders
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.iteration = 0
        self.total_iterations = total_iterations
        self.iterations_per_validation = iterations_per_validation
        self.iterations_per_checkpoint = iterations_per_checkpoint

        # Learning rate scheduling
        self.initial_lr = learning_rate
        self.final_lr = learning_rate * total_learning_rate_decay
        self.iterations_per_lr_update = 1 + math.ceil(total_iterations / (num_learning_rate_updates + 1))
        if self.iterations_per_lr_update < 100:
            self.iterations_per_lr_update = 100

        # Registered minibatch function
        self.minibatch_fn = minibatch_fn

        # Checkpoint loading
        if checkpoint_load_path != '':
            self.load_checkpoint(checkpoint_load_path)


    def load_checkpoint(self, checkpoint_path):
        checkpoint_dict = torch.load(checkpoint_path)
        self.iteration = checkpoint_dict['iteration']

        for model_name, model in self.models.items():
            model.load_state_dict(checkpoint_dict[model_name])
        for model_name, optimizer in self.optimizers.items():
            optimizer_key = f'{model_name}.optimizer'
            optimizer.load_state_dict(checkpoint_dict[optimizer_key])


    def save_checkpoint(self):
        checkpoint_fname = self.checkpoint_save_path / f'checkpoint_{self.iteration}.pt'
        checkpoint_dict = {}

        checkpoint_dict['iteration'] = self.iteration

        for model_name, model in self.models.items():
            checkpoint_dict[model_name] = model.state_dict()
        for model_name, optimizer in self.optimizers.items():
            optimizer_key = f'{model_name}.optimizer'
            checkpoint_dict[optimizer_key] = optimizer.state_dict()

        torch.save(checkpoint_dict, checkpoint_fname)


    def update_log(self, data, split='train'):
        if split == 'train':
            # Record learning rate
            self.log_writer.add_scalar(
                f'learning_rate',
                self.learning_rate,
                self.iteration
            )

            # Record seperate optimizer gradient norms
            for model_name, opt in self.optimizers.items():
                if hasattr(opt, 'recorded_grad_norm'):
                    self.log_writer.add_scalar(
                        f'{model_name}.grad_norm',
                        opt.recorded_grad_norm,
                        self.iteration
                    )

        # Record other data
        for key, value in data.items():
            self.log_writer.add_scalar(f'{split}.{key}', value, self.iteration)


    @staticmethod
    def to_cuda(minibatch):
        if isinstance(minibatch, (list, tuple)):
            minibatch = [x.cuda() for x in minibatch]
        else:
            minibatch = minibatch.cuda()
        return minibatch


    def do_validation(self):
        for model in self.models.values():
            model.eval()

        valid_results = defaultdict(float)
        for batch_i, minibatch in enumerate(tqdm(self.valid_dataloader)):
            minibatch = self.to_cuda(minibatch)

            minibatch_results = self.minibatch_fn(
                iteration=self.iteration,
                minibatch=minibatch,
                models=self.models,
                optimizers={},
                grad_clip_thresh=None,
                train=False
            )
            for key, value in minibatch_results.items():
                valid_results[key] += value
        for key, value in  valid_results.items():
            valid_results[key] /= (batch_i + 1)

        self.update_log(valid_results, split='valid')

        for model in self.models.values():
            model.train()


    @staticmethod
    def update_pbar(pbar, data):
        pbar_str = ' '.join([f'{key}: {value:.2f}' for key, value in data.items()])
        pbar.set_description(pbar_str)
        pbar.update(1)


    def update_learning_rate(self):
        alpha = self.iteration / self.total_iterations
        self.learning_rate = self.final_lr**alpha * self.initial_lr**(1 - alpha)

        for optimizer in self.optimizers.values():
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate


    def run(self):
        for model in self.models.values():
            model.train()

        with tqdm(total=self.total_iterations) as pbar:
            pbar.update(self.iteration)
            self.update_learning_rate()

            result_history = defaultdict(list)
            while self.iteration < self.total_iterations:
                for minibatch in self.train_dataloader:
                    minibatch = self.to_cuda(minibatch)

                    # TODO, pass in other stuff?
                    minibatch_results = self.minibatch_fn(
                        iteration=self.iteration,
                        minibatch=minibatch,
                        models=self.models,
                        optimizers=self.optimizers,
                        grad_clip_thresh=self.grad_clip_thresh,
                        train=True
                    )

                    for key, value in minibatch_results.items():
                        result_history[key].append(value)

                    self.update_pbar(pbar, minibatch_results)
                    self.iteration += 1

                    if self.iteration % 10 == 0:
                        smoothed_results = {}
                        for key in result_history:
                            result_history[key] = result_history[key][-10:]
                            smoothed_results[key] = np.mean(result_history[key])

                        self.update_log(smoothed_results, split='train')

                    if self.iteration % self.iterations_per_lr_update == 0:
                        self.update_learning_rate()

                    if self.iteration % self.iterations_per_validation == 0:
                        self.do_validation()

                    due_a_checkpoint = self.iteration % self.iterations_per_checkpoint == 0
                    finishing = self.iteration == self.total_iterations
                    if due_a_checkpoint or finishing:
                        self.save_checkpoint()

                    if self.iteration > self.total_iterations:
                        break
