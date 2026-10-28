# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flow-coordinate adaptations of Consistency Policy and OneDP-S.

Time 1 is Gaussian noise and time 0 is action, as in OpenPI. These are
adaptations, not the papers' EDM/DDPM parameterizations. See the method note.
"""

import torch
import torch.nn.functional as F


def expand(t: torch.Tensor) -> torch.Tensor:
    """Broadcast a batch of scalar times across a complete action block."""
    return t[:, None, None]


def flow_map(velocity, x, t, s, context):
    """Predict the flow endpoint, with an exact identity boundary at s=t."""
    return x - expand(t - s) * velocity(x, t, context, s)


def action_coordinates(value: torch.Tensor, action_dim: int | None) -> torch.Tensor:
    """Select valid coordinates before averaging; None preserves legacy losses."""
    if action_dim is None:
        return value
    if not 0 < action_dim <= value.shape[-1]:
        raise ValueError("Invalid physical action dimension")
    return value[..., :action_dim]


def cp_distance(
    pred: torch.Tensor, target: torch.Tensor, *, action_dim: int | None = None
) -> torch.Tensor:
    """Match official Huber_Loss(delta=-1), using a stable equivalent formula.

    The reference applies sqrt(mse**2 + delta**2) - delta, where mse is
    elementwise squared error. Preserve that exact choice, rather than
    substituting PyTorch smooth_l1_loss or a differently defined Huber norm.
    """
    pred, target = (
        action_coordinates(pred, action_dim),
        action_coordinates(target, action_dim),
    )
    delta = (pred.shape[1] * pred.shape[2]) ** 0.5 * 0.00054
    fourth = (pred.float() - target.float()).square().square()
    return (fourth / ((fourth + delta**2).sqrt() + delta)).mean()


def cp_loss(student, teacher, ema, actions, context, *, bins=80, action_dim=None):
    """Teacher one-step bridge, EMA composition consistency, and native FM anchor."""
    b, device = len(actions), actions.device
    index = torch.randint(1, bins + 1, (b,), device=device)
    t = index.float() / bins
    u = (index - 1).float() / bins
    s = torch.floor(torch.rand(b, device=device) * index) / bins
    zero = torch.zeros_like(t)
    noise = torch.randn_like(actions)
    x = (1 - expand(t)) * actions + expand(t) * noise
    with torch.no_grad():
        xu = x - expand(t - u) * teacher(x, t, context)
        target = flow_map(ema, flow_map(ema, xu, u, s, context), s, zero, context)
    middle = flow_map(student, x, t, s, context)
    # EMA parameters are frozen, but this input derivative MUST reach student.
    pred = flow_map(ema, middle, s, zero, context)
    consistency = cp_distance(pred, target, action_dim=action_dim)
    # Native pi05 flow-matching time distribution and velocity target.
    tm = torch.distributions.Beta(1.5, 1.0).sample((b,)).to(device) * 0.999 + 0.001
    epsilon = torch.randn_like(actions)
    xm = (1 - expand(tm)) * actions + expand(tm) * epsilon
    velocity = student(xm, tm, context, zero)
    fm = F.mse_loss(
        action_coordinates(velocity, action_dim),
        action_coordinates(epsilon - actions, action_dim),
    )
    parts = {"consistency": consistency, "fm": fm}
    if action_dim is not None:
        with torch.no_grad():
            parts["consistency_physical_mse"] = (
                (pred[..., :action_dim] - target[..., :action_dim])
                .float()
                .square()
                .mean()
            )
            parts["cp_distance_delta"] = pred.new_tensor(
                (pred.shape[1] * action_dim) ** 0.5 * 0.00054
            )
            if action_dim < pred.shape[-1]:
                parts["consistency_padding_mse"] = (
                    (pred[..., action_dim:] - target[..., action_dim:])
                    .float()
                    .square()
                    .mean()
                )
                parts["fm_padding_mse"] = (
                    (velocity[..., action_dim:] - (epsilon - actions)[..., action_dim:])
                    .float()
                    .square()
                    .mean()
                )
    return consistency + fm, parts


def fake_loss(
    fake,
    generated,
    context,
    *,
    time_min=0.02,
    time_max=0.95,
    return_parts=False,
    loss_action_dim=None,
):
    """Fit fake epsilon to the student's distribution; lambda(t)=1 as OneDP-S."""
    t = (
        torch.rand(len(generated), device=generated.device) * (time_max - time_min)
        + time_min
    )
    noise = torch.randn_like(generated)
    x = (1 - expand(t)) * generated.detach() + expand(t) * noise
    eps_pred = x + (1 - expand(t)) * fake(x, t, context)
    loss = F.mse_loss(
        action_coordinates(eps_pred, loss_action_dim),
        action_coordinates(noise, loss_action_dim),
    )
    if not return_parts:
        return loss
    with torch.no_grad():
        error = (eps_pred.float() - noise.float()).square()
        parts = {"fake_physical_mse": error[..., :14].mean()}
        if error.shape[-1] > 14:
            parts["fake_padding_mse"] = error[..., 14:].mean()
    return loss, parts


def onedp_loss(
    generated, teacher, fake, context, *, time_min=0.02, time_max=0.95, action_dim=None
):
    """Reverse-KL score gradient with OneDP w=sigma^2 in flow coordinates.

    epsilon_hat=x+(1-t)*v, score=-epsilon_hat/t. The detached coefficient
    t*(epsilon_teacher-epsilon_fake) multiplies x_t, retaining its (1-t)
    derivative w.r.t. the generated action. No teacher/fake Jacobian is used.
    """
    t = (
        torch.rand(len(generated), device=generated.device) * (time_max - time_min)
        + time_min
    )
    x = (1 - expand(t)) * generated + expand(t) * torch.randn_like(generated)
    with torch.no_grad():
        direction = (
            expand(t)
            * (1 - expand(t))
            * (teacher(x.detach(), t, context) - fake(x.detach(), t, context))
        )
    if action_dim is not None:
        if not 0 < action_dim <= generated.shape[-1]:
            raise ValueError("Invalid physical action dimension")
        direction, x = direction[..., :action_dim], x[..., :action_dim]
    return (direction * x).flatten(1).mean(1).mean(), direction.square().mean().sqrt()


def ema_decay(step: int) -> float:
    """Consistency Policy's inv_gamma=1, power=.75, max=.9999 EMA schedule."""
    return min(0.9999, 1 - (1 + max(step - 1, 0)) ** -0.75)


def endpoint_loss(student, teacher, noise, context, *, teacher_steps=10, action_dim=14):
    """Pair native multistep teacher and N1 student endpoints using the SAME noise.

    Compare normalized physical action coordinates over the whole H50 block.
    The teacher integrates all internal coordinates, including padding, but
    padding is not a regression target. No expert-action FM or EMA is added.
    """
    if teacher_steps not in (5, 10) or action_dim != 14 or noise.shape[-1] < action_dim:
        raise ValueError(
            "Endpoint baseline requires native N5/N10 and 14D Aloha actions"
        )
    with torch.no_grad():
        target = teacher.endpoint(noise, context, steps=teacher_steps)
    one = torch.ones(len(noise), device=noise.device)
    prediction = noise - student(noise, one, context)
    error = (
        prediction[..., :action_dim].float() - target[..., :action_dim].float()
    ).square()
    loss = error.mean()
    return loss, {
        "endpoint_mse": loss,
        "joint_mse": error[..., [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].mean(),
        "gripper_mse": error[..., [6, 13]].mean(),
    }
