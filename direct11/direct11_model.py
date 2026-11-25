"""Construct the six- or eleven-frame implementation with the same spatial backbone."""
import numpy as np
import torch
from architecture import Model

MODEL_KWARGS = dict(
    downnum=4, covnum=2, layer_time=1, baselayer=16,
    cov_kernelsize=3, cov_strides=1, pool_method="maxpooling",
    pool_kernel_size=2, pool_strides=2, ifnormalization="no",
    normalization_method=None, trans_num=1, num_heads=1,
    dim_feedforward=128, trans_dropout_rate=0.0, activate="tanh",
    nlp_activate="relu", da_attention_order="space_first",
    if_bottom_attention_only="yes", if_resnet="yes",
)


def build_model(history_steps=11, input_channels=3, device="cpu"):
    """Build a fresh FP32 Stage 1 network. Caller controls the initialization seed."""
    if history_steps not in (6, 11) or input_channels != 3:
        raise ValueError("Supported Stage 1 input is 6 or 11 frames of 3 surface channels")
    # The original constructor only reads the shapes of its trainx/trainy arguments.
    x_shape = np.empty((1, history_steps, 3, 49, 73), dtype=np.float32)
    y_shape = np.empty((1, 9, 49, 73), dtype=np.float32)
    model = Model(x_shape, y_shape, timestep=history_steps, **MODEL_KWARGS)
    return model.to(device=device, dtype=torch.float32)


def release_forward_cache(model):
    """Drop references retained by the legacy forward after a microbatch finishes."""
    for name in tuple(model.__dict__):
        if (name.startswith(("model_", "covup", "covdown"))
                or name in ("x", "res_layer")):
            del model.__dict__[name]

