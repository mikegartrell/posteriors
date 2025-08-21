from typing import Any
from functools import partial
import torch
from torch.func import grad_and_value
from optree import tree_map, tree_transpose
from tensordict import TensorClass

from posteriors.types import TensorTree, Transform, LogProbFn, Schedule
from posteriors.tree_utils import flexi_tree_map, tree_insert_
from posteriors.utils import is_scalar, CatchAuxError


def build(
    log_posterior: LogProbFn,
    lr: float | Schedule,
    alpha: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    beta: float = 0.0,  # Add beta parameter
    sigma: float = 1.0,
    temperature: float | Schedule = 1.0,
    momenta: TensorTree | float | None = None,
) -> Transform:
    """Builds Adam-SGHMC transform.

    Combines Adam's adaptive gradient scaling with SGHMC sampling.
    
    The algorithm uses Adam's first and second moment estimates to precondition
    the gradients, then applies SGHMC updates with the preconditioned gradients:

    \\begin{align}
    m_t &= β_1 m_{t-1} + (1 - β_1) \\nabla \\log p(θ_t, \\text{batch}) \\\\
    v_t &= β_2 v_{t-1} + (1 - β_2) (\\nabla \\log p(θ_t, \\text{batch}))^2 \\\\
    \\hat{m}_t &= \\frac{m_t}{1 - β_1^t} \\\\
    \\hat{v}_t &= \\frac{v_t}{1 - β_2^t} \\\\
    \\tilde{g}_t &= \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + ε} \\\\
    p_{t+1} &= p_t + ε σ^{-2} r_t \\\\
    r_{t+1} &= r_t + ε \\tilde{g}_t - ε σ^{-2} α r_t
    + N(0, ε T (2 α - ε β T) P_t^{-1})
    \\end{align}
    
    where $P_t^{-1} = \\text{diag}(\\frac{1}{\\sqrt{\\hat{v}_t} + ε})$ is the preconditioning matrix
    from Adam, and $T$ is the temperature.

    Args:
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior value (which can be unnormalised)
            as well as auxiliary information, e.g. from the model call.
        lr: Learning rate,
            scalar or schedule (callable taking step index, returning scalar).
        alpha: Friction coefficient for momentum decay.
        beta1: Exponential decay rate for first moment estimates (Adam parameter).
        beta2: Exponential decay rate for second moment estimates (Adam parameter).
        epsilon: Small constant for numerical stability in Adam.
        beta: Gradient noise coefficient (estimated variance).
        sigma: Standard deviation of momenta target distribution.
        temperature: Temperature of the joint parameter + momenta distribution.
            Scalar or schedule (callable taking step index, returning scalar).
        momenta: Initial momenta. Can be tree like params or scalar.
            Defaults to random iid samples from N(0, 1).

    Returns:
        Adam-SGHMC transform instance.
    """
    init_fn = partial(init, momenta=momenta)
    update_fn = partial(
        update,
        log_posterior=log_posterior,
        lr=lr,
        alpha=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        beta=beta,  # Add beta parameter
        sigma=sigma,
        temperature=temperature,
    )
    return Transform(init_fn, update_fn)


class AdamSGHMCState(TensorClass["frozen"]):
    """State encoding params, momenta, and Adam statistics for Adam-SGHMC.

    Attributes:
        params: Parameters.
        momenta: Momenta for each parameter.
        m: First moment estimates (Adam).
        v: Second moment estimates (Adam).
        log_posterior: Log posterior evaluation.
        step: Current step count.
    """

    params: TensorTree
    momenta: TensorTree
    m: TensorTree
    v: TensorTree
    log_posterior: torch.Tensor = torch.tensor(torch.nan)
    step: torch.Tensor = torch.tensor(0)


def init(
    params: TensorTree, 
    momenta: TensorTree | float | None = None
) -> AdamSGHMCState:
    """Initialise momenta and Adam statistics for Adam-SGHMC.

    Args:
        params: Parameters for which to initialise.
        momenta: Initial momenta. Can be tree like params or scalar.
            Defaults to random iid samples from N(0, 1).

    Returns:
        Initial AdamSGHMCState containing momenta and Adam statistics.
    """
    if momenta is None:
        momenta = tree_map(
            lambda x: torch.randn_like(x, requires_grad=x.requires_grad),
            params,
        )
    elif is_scalar(momenta):
        momenta = tree_map(
            lambda x: torch.full_like(x, momenta, requires_grad=x.requires_grad),
            params,
        )

    # Initialize Adam moment estimates to zero
    m = tree_map(
        lambda x: torch.zeros_like(x, requires_grad=x.requires_grad),
        params,
    )
    v = tree_map(
        lambda x: torch.zeros_like(x, requires_grad=x.requires_grad),
        params,
    )

    return AdamSGHMCState(params, momenta, m, v)


def update(
    state: AdamSGHMCState,
    batch: Any,
    log_posterior: LogProbFn,
    lr: float | Schedule,
    alpha: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    beta: float = 0.0,  # Add beta parameter for gradient noise
    sigma: float = 1.0,
    temperature: float | Schedule = 1.0,
    inplace: bool = False,
) -> tuple[AdamSGHMCState, TensorTree]:
    """Updates parameters, momenta, and Adam statistics for Adam-SGHMC."""
    
    with torch.no_grad(), CatchAuxError():
        grads, (log_post, aux) = grad_and_value(log_posterior, has_aux=True)(
            state.params, batch
        )

    lr = lr(state.step) if callable(lr) else lr
    temperature = temperature(state.step) if callable(temperature) else temperature
    prec = sigma**-2
    step = state.step + 1
    step_scalar = step.item()

    # Bias correction terms
    bias_correction1 = 1 - beta1 ** step_scalar
    bias_correction2 = 1 - beta2 ** step_scalar

    # Update Adam moments
    m_new = tree_map(
        lambda m, g: beta1 * m + (1 - beta1) * g,
        state.m,
        grads,
    )
    
    v_new = tree_map(
        lambda v, g: beta2 * v + (1 - beta2) * g.square(),
        state.v,
        grads,
    )
    
    # Compute bias-corrected moments and preconditioned gradients
    precond_grads = tree_map(
        lambda m, v: (m / bias_correction1) / (torch.sqrt(v / bias_correction2) + epsilon),
        m_new,
        v_new,
    )

    def update_momenta(r, precond_grad, v):
        """Update momenta with proper noise scaling."""
        with torch.no_grad():
            # Compute preconditioning factor (inverse of what we want)
            precond_factor = torch.sqrt(v / bias_correction2) + epsilon
            
            # Base noise variance from SGHMC
            base_noise_variance = temperature * lr * (2 * alpha - temperature * lr * beta)
            
            # Ensure positive variance
            base_noise_variance = torch.clamp(torch.tensor(base_noise_variance), min=1e-8)
            
            # Scale noise by preconditioning factor (not inverse)
            # This maintains proper scaling with the preconditioned gradients
            noise = torch.sqrt(base_noise_variance * precond_factor) * torch.randn_like(r)

            r_new = (
                r
                + lr * precond_grad
                - lr * prec * alpha * r
                + noise
            )

            return r_new

    # Update momenta with corrected noise scaling
    momenta_new = tree_map(
        update_momenta,
        state.momenta,
        precond_grads,
        v_new,
    )

    # Update parameters
    params_new = tree_map(
        lambda p, r: p + lr * prec * r,
        state.params,
        momenta_new,
    )

    if inplace:
        tree_insert_(state.params, params_new)
        tree_insert_(state.momenta, momenta_new)
        tree_insert_(state.m, m_new)
        tree_insert_(state.v, v_new)
        tree_insert_(state.log_posterior, log_post.detach())
        tree_insert_(state.step, step)
        return state, aux
    else:
        return (
            AdamSGHMCState(
                params_new, momenta_new, m_new, v_new, log_post.detach(), step
            ),
            aux,
        )
    