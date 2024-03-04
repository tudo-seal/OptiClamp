from typing import Optional

from botorch.acquisition.multi_objective import monte_carlo
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler


def _get_sobol_qmc_normal_sampler(num_samples: int) -> SobolQMCNormalSampler:
    return SobolQMCNormalSampler(torch.Size((num_samples,)))


import torch
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood

model: SingleTaskGP = None
train_x: torch.Tensor = None
train_y: torch.Tensor = None
bounds: torch.Tensor = None


def singletask_qnehvi_candidates_func(
    train_in: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds_p: "torch.Tensor",
    pending_x: Optional["torch.Tensor"],
) -> "torch.Tensor":
    global model, train_x, train_y, bounds
    n_objectives = train_obj.size(-1)
    train_x = train_in
    bounds = bounds_p
    if train_con is not None:
        train_y = torch.cat([train_obj, train_con], dim=-1)

        n_constraints = train_con.size(1)
        additional_qnehvi_kwargs = {
            "objective": IdentityMCMultiOutputObjective(
                outcomes=list(range(n_objectives))
            ),
            "constraints": [
                (lambda Z, i=i: Z[..., -n_constraints + i])
                for i in range(n_constraints)
            ],
        }
    else:
        train_y = train_obj

        additional_qnehvi_kwargs = {}

    train_x = normalize(train_x, bounds=bounds)
    if pending_x is not None:
        pending_x = normalize(pending_x, bounds=bounds)

    model = SingleTaskGP(
        train_x, train_y, outcome_transform=Standardize(m=train_y.shape[-1])
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Approximate box decomposition similar to Ax when the number of objectives is large.
    # https://github.com/pytorch/botorch/blob/36d09a4297c2a0ff385077b7fcdd5a9d308e40cc/botorch/acquisition/multi_objective/utils.py#L46-L63
    if n_objectives > 4:
        alpha = 10 ** (-8 + n_objectives)
    else:
        alpha = 0.0

    ref_point = train_obj.min(dim=0).values - 1e-8

    ref_point_list = ref_point.tolist()

    # prune_baseline=True is generally recommended by the documentation of BoTorch.
    # cf. https://botorch.org/api/acquisition.html (accessed on 2022/11/18)
    acqf = monte_carlo.qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point_list,
        X_baseline=train_x,
        alpha=alpha,
        prune_baseline=True,
        sampler=_get_sobol_qmc_normal_sampler(256),
        X_pending=pending_x,
        **additional_qnehvi_kwargs,
    )

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=standard_bounds,
        q=1,
        num_restarts=20,
        raw_samples=1024,
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
        sequential=True,
    )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates


def singletask_qehvi_candidates_func(
    train_in: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds_p: "torch.Tensor",
    pending_x: Optional["torch.Tensor"],
) -> "torch.Tensor":
    global model, train_x, train_y, bounds
    train_x = train_in
    bounds = bounds_p
    n_objectives = train_obj.size(-1)
    if train_con is not None:
        train_y = torch.cat([train_obj, train_con], dim=-1)

        is_feas = (train_con <= 0).all(dim=-1)
        train_obj_feas = train_obj[is_feas]

        n_constraints = train_con.size(1)
        additional_qehvi_kwargs = {
            "objective": IdentityMCMultiOutputObjective(
                outcomes=list(range(n_objectives))
            ),
            "constraints": [
                (lambda Z, i=i: Z[..., -n_constraints + i])
                for i in range(n_constraints)
            ],
        }
    else:
        train_y = train_obj

        train_obj_feas = train_obj

        additional_qehvi_kwargs = {}

    train_x = normalize(train_x, bounds=bounds)
    if pending_x is not None:
        pending_x = normalize(pending_x, bounds=bounds)

    model = SingleTaskGP(
        train_x, train_y, outcome_transform=Standardize(m=train_y.shape[-1])
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Approximate box decomposition similar to Ax when the number of objectives is large.
    # https://github.com/pytorch/botorch/blob/36d09a4297c2a0ff385077b7fcdd5a9d308e40cc/botorch/acquisition/multi_objective/utils.py#L46-L63
    if n_objectives > 4:
        alpha = 10 ** (-8 + n_objectives)
    else:
        alpha = 0.0

    ref_point = train_obj.min(dim=0).values - 1e-8

    partitioning = NondominatedPartitioning(
        ref_point=ref_point, Y=train_obj_feas, alpha=alpha
    )

    ref_point_list = ref_point.tolist()

    acqf = monte_carlo.qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point_list,
        partitioning=partitioning,
        sampler=_get_sobol_qmc_normal_sampler(256),
        X_pending=pending_x,
        **additional_qehvi_kwargs,
    )
    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=standard_bounds,
        q=1,
        num_restarts=20,
        raw_samples=1024,
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
        sequential=True,
    )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates
