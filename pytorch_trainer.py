import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import git
from pip._internal.operations import freeze as pip_freeze
import platform
import types
import math
from datetime import datetime
import time
import random
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict


def zero_backward_clip_step(scaler, model, optimizer, loss, grad_clip_thresh=None):
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    if grad_clip_thresh is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip_thresh
        ).item()
    scaler.step(optimizer)
    return grad_norm


def parse_arg_value(value):
    if isinstance(value, (tuple, list)):
        return f'[{", ".join(map(str, value))}]'
    else:
        return value


class PytorchTrainer:
    def __init__(self, model_names, model_list,
                 train_dataloader,
                 valid_dataloader,
                 minibatch_fn,
                 config,
                 checkpoint_load_path='',
                 checkpoint_load_ignore_layers={},
                 checkpoint_load_model_only=False,
                 output_name='default',
                 learning_rate=1e-3, weight_decay=1e-4,
                 optimizer_betas=(0.9, 0.999),
                 optimizer_eps=1e-8,
                 grad_clip_thresh=1.0,
                 num_learning_rate_updates=100,
                 total_learning_rate_decay=0.01,
                 learning_rate_warm_up_iterations=100,
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
        date_str = datetime.now().strftime('%Y-%m-%d')
        time_str = datetime.now().strftime('%H-%M-%S')
        output_path = Path('outputs') / output_name / f'{date_str}_{time_str}'
        self.checkpoint_save_path = output_path / 'checkpoints'
        self.checkpoint_save_path.mkdir(parents=True, exist_ok=True)

        # Save configs
        config_dict = {key.replace("_", "-"): parse_arg_value(value) for key, value in vars(config).items() if key != 'config' and value is not None}
        with (output_path / 'config.conf').open('w') as f:
            f.write('\n'.join(f'{key}: {value}' for key, value in config_dict.items()))

        # For storing metadata with checkpoint
        self.metadata = {'config': config_dict}

        # Save git info
        try:
            repo = git.Repo(search_parent_directories=True)
            head_obj = repo.head.object
            git_info = {
                'commit': head_obj.hexsha,
                'branch': repo.active_branch.name,
                'author': {
                    'name': head_obj.author.name,
                    'email': head_obj.author.email
                }
            }
            try:
                git_info['remotes'] = list(repo.remote().urls),
            except ValueError:
                pass
            with (output_path / 'git_info.json').open('w') as f:
                json.dump(git_info, f, indent=2)
            self.metadata['git'] = git_info
        except (git.exc.InvalidGitRepositoryError, ValueError):
            pass

        # Save run info
        run_info = {
            'start_date': date_str,
            'directory': str(Path('.').resolve()),
            'hostname': platform.node()
        }
        self.run_info_path = output_path / 'run_info.json'
        self.run_start_time = time.time()
        self.metadata['run'] = run_info
        with self.run_info_path.open('w') as f:
            json.dump(self.metadata['run'], f, indent=2)

        # Save pip installed info
        pip_info = list(pip_freeze.freeze())
        self.metadata['pip'] = pip_info
        with (output_path / 'pip_freeze.txt').open('w') as f:
            f.write('\n'.join(pip_info))

        # Logger
        log_path = output_path / 'logs'
        log_path.mkdir(parents=True, exist_ok=True)
        self.log_writer = SummaryWriter(log_path)

        # Move models to GPU
        for model in model_list:
            model.cuda()

        # Set up optimizers
        optimizers = []
        self.learning_rate = 1e-6
        self.grad_clip_thresh = grad_clip_thresh
        for model in model_list:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay,
                betas=optimizer_betas,
                eps=optimizer_eps
            )
            optimizers.append(optimizer)

        self.models = {}
        self.optimizers = {}
        for name, model, optimizer in zip(model_names, model_list, optimizers):
            self.models[name] = model
            self.optimizers[name] = optimizer
        self.scaler = torch.cuda.amp.GradScaler()
        self.scaler.zero_backward_clip_step = types.MethodType(
            zero_backward_clip_step, self.scaler
        )

        # Dataloaders
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.starting_iteration = 0
        self.iteration = 0
        self.total_iterations = total_iterations
        self.iterations_per_validation = iterations_per_validation
        self.iterations_per_checkpoint = iterations_per_checkpoint

        # Learning rate scheduling
        self.learning_rate_warm_up_iterations = learning_rate_warm_up_iterations
        self.initial_lr = learning_rate
        self.final_lr = learning_rate * total_learning_rate_decay
        self.iterations_per_lr_update = 1 + math.ceil(total_iterations / (num_learning_rate_updates + 1))
        if self.iterations_per_lr_update < 100:
            self.iterations_per_lr_update = 100

        # Registered minibatch function
        self.minibatch_fn = minibatch_fn

        # Checkpoint loading
        if checkpoint_load_path != '':
            self.load_checkpoint(
                checkpoint_load_path,
                checkpoint_load_ignore_layers,
                checkpoint_load_model_only
            )


    def update_run_metadata(self):
        self.metadata['run']['end_date'] = datetime.now().strftime('%Y-%m-%d')
        run_duration = (time.time() - self.run_start_time) / 3600
        self.metadata['run']['duration_hours'] = f'{run_duration:.02f}'


    def load_checkpoint(self, checkpoint_path, ignore_layers, model_only):
        checkpoint_dict = torch.load(checkpoint_path)

        for model_name, model in self.models.items():
            model_dict = checkpoint_dict[model_name]
            # If there are layers to ignore
            if model_name in ignore_layers and len(ignore_layers[model_name]) > 0:
                for ignore_layer in ignore_layers[model_name]:
                    if ignore_layer not in model_dict.keys():
                        raise ValueError(f'Layer {ignore_layer} not found in model.')

                model_dict = {k: v for k, v in model_dict.items() if k not in ignore_layers[model_name]}
                dummy_dict = model.state_dict()
                dummy_dict.update(model_dict)
                model_dict = dummy_dict
            model.load_state_dict(model_dict)

        if not model_only:
            self.starting_iteration = checkpoint_dict['iteration']
            for model_name, optimizer in self.optimizers.items():
                optimizer_key = f'{model_name}.optimizer'
                optimizer.load_state_dict(checkpoint_dict[optimizer_key])


    def save_checkpoint(self):
        checkpoint_fname = self.checkpoint_save_path / f'checkpoint_{self.iteration}.pt'
        checkpoint_dict = {}

        checkpoint_dict['iteration'] = self.starting_iteration + self.iteration

        # Save metadata with checkpoint
        self.update_run_metadata()
        checkpoint_dict['metadata'] = self.metadata

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
                self.starting_iteration + self.iteration
            )

            # Record seperate optimizer gradient norms
            for model_name, opt in self.optimizers.items():
                if hasattr(opt, 'recorded_grad_norm'):
                    self.log_writer.add_scalar(
                        f'{model_name}.grad_norm',
                        opt.recorded_grad_norm,
                        self.starting_iteration + self.iteration
                    )

        # Record other data
        for key, value in data.items():
            self.log_writer.add_scalar(f'{split}.{key}', value, self.starting_iteration + self.iteration)


    @classmethod
    def to_cuda(cls, minibatch):
        if isinstance(minibatch, (list, tuple)):
            minibatch = [cls.to_cuda(x) for x in minibatch]
        else:
            minibatch = minibatch.cuda()
        return minibatch


    def do_validation(self):
        for model in self.models.values():
            model.eval()

        valid_results = defaultdict(float)
        for batch_i, minibatch in enumerate(tqdm(self.valid_dataloader)):
            minibatch = self.to_cuda(minibatch)

            with torch.no_grad():
                minibatch_results = self.minibatch_fn(
                    iteration=self.iteration,
                    minibatch=minibatch,
                    models=self.models,
                    optimizers={},
                    scaler=None,
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
        if self.iteration <= self.learning_rate_warm_up_iterations:
            alpha = self.iteration / self.learning_rate_warm_up_iterations
            self.learning_rate = alpha * self.initial_lr
        else:
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

                    minibatch_results = self.minibatch_fn(
                        iteration=self.iteration,
                        minibatch=minibatch,
                        models=self.models,
                        optimizers=self.optimizers,
                        scaler=self.scaler,
                        grad_clip_thresh=self.grad_clip_thresh,
                        train=True
                    )

                    for key, value in minibatch_results.items():
                        result_history[key].append(value)

                    self.update_pbar(pbar, minibatch_results)
                    self.iteration += 1

                    if self.iteration % 100 == 0:
                        smoothed_results = {}
                        for key in result_history:
                            result_history[key] = result_history[key][-10:]
                            smoothed_results[key] = np.mean(result_history[key])

                        self.update_log(smoothed_results, split='train')

                    if self.iteration <= self.learning_rate_warm_up_iterations or self.iteration % self.iterations_per_lr_update == 0:
                        self.update_learning_rate()

                    if self.iteration % self.iterations_per_validation == 0:
                        self.do_validation()

                    due_a_checkpoint = self.iteration % self.iterations_per_checkpoint == 0
                    finishing = self.iteration == self.total_iterations
                    if due_a_checkpoint or finishing:
                        self.save_checkpoint()

                    if self.iteration > self.total_iterations:
                        break

        self.update_run_metadata()
        with self.run_info_path.open('w') as f:
            json.dump(self.metadata['run'], f, indent=2)