# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

import torch

from egg.core.interaction import LoggingStrategy
from egg.zoo.referential_language.data import get_gaussian_dataloader


def log_stats(interaction, mode):
    dump = dict((k, v.mean().item()) for k, v in interaction.aux.items())
    dump.update(dict(mode=mode))
    print(json.dumps(dump), flush=True)


def run_gaussian_test(trainer, opts, data_kwargs):
    if opts.distributed_context.is_distributed:
        game = trainer.game.module.game
    else:
        game = trainer.game.game
    game.test_logging_strategy = LoggingStrategy.minimal()

    gaussian_data_loader = get_gaussian_dataloader(**data_kwargs)
    _, gaussian_interaction = trainer.eval(gaussian_data_loader)
    log_stats(gaussian_interaction, "GAUSSIAN SET")


def run_test(trainer, opts, data_loader):
    logging_test_args = [False, True, True, True, True, True, False]
    test_logging_strategy = LoggingStrategy(*logging_test_args)
    if opts.distributed_context.is_distributed:
        game = trainer.game.module.game
    else:
        game = trainer.game.game
    game.test_logging_strategy = test_logging_strategy

    _, interaction = trainer.eval(data_loader)
    log_stats(interaction, "TEST SET")

    if opts.distributed_context.is_leader and opts.checkpoint_dir:
        output_path = Path(opts.checkpoint_dir) / "interactions"
        output_path.mkdir(exist_ok=True, parents=True)
        interaction_name = f"interaction_{opts.job_id}_{opts.task_id}"

        interaction.aux_input["args"] = opts
        torch.save(interaction, output_path / interaction_name)
