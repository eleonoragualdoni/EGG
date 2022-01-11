# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from egg.core.gs_wrappers import (
    GumbelSoftmaxWrapper,
    SymbolGameGS,
    SymbolReceiverWrapper,
)
from egg.core.interaction import LoggingStrategy
from egg.zoo.referential_language.archs import (
    Receiver,
    Sender,
    VisionWrapper,
    get_cnn,
)


def loss(
    _sender_input,
    _message,
    _receiver_input,
    receiver_output,
    _labels,
    _aux_input,
):
    labels = torch.arange(receiver_output.shape[0], device=receiver_output.device)
    acc = (receiver_output.argmax(dim=-1) == labels).detach().float()
    loss = F.cross_entropy(receiver_output, labels, reduction="none")
    n_objs = receiver_output.shape[0]
    return loss, {"acc": acc, "baseline": torch.Tensor([1 / n_objs])}


def build_gs_game(opts):
    cnn_sender, input_dim = get_cnn(opts.vision_model_name, opts.pretrain_vision)
    cnn_receiver = None
    if not opts.shared_vision:
        cnn_receiver, _ = get_cnn(opts.vision_model_name, opts.pretrain_vision)

    sender = Sender(
        input_dim=input_dim,
        output_dim=opts.vocab_size,
        num_heads=opts.num_heads,
        attention_type=opts.attention_type,
        context_integration=opts.context_integration,
    )
    receiver = Receiver(
        input_dim=input_dim,
        hidden_dim=opts.recv_hidden_dim,
        output_dim=opts.recv_output_dim,
        temperature=opts.recv_temperature,
        use_cosine_sim=opts.use_cosine_similarity,
    )
    sender = GumbelSoftmaxWrapper(
        agent=sender,
        temperature=opts.gs_temperature,
    )
    receiver = SymbolReceiverWrapper(
        receiver,
        opts.vocab_size,
        opts.recv_output_dim,
    )
    logging_test_args = [False, True, True, True, True, True, False]
    test_logging_strategy = LoggingStrategy(*logging_test_args)
    game = SymbolGameGS(
        sender=sender,
        receiver=receiver,
        loss=loss,
        train_logging_strategy=LoggingStrategy.minimal(),
        test_logging_strategy=test_logging_strategy,
    )
    return VisionWrapper(game, cnn_sender, cnn_receiver)


def build_game(opts):
    return build_gs_game(opts)
