# Proposal: Finite-Difference Consistency for LAFAN1 EqM

## Problem

`root_vel_fd_mse` and `joint_vel_fd_mse` measure whether the velocity channels match finite differences of the corresponding position channels. In the current EqM setup, the model predicts an update field, not the final trajectory, so the FD constraint must be applied to that predicted update.

## Key Observation

The EqM target is:

```text
target = (x1 - x0) * g(t)
```

Finite difference over time is a linear operator. If both endpoint states are FD-consistent, then `x1 - x0` is also FD-consistent, and multiplying by `g(t)` preserves that structure. This means an FD loss can be defined directly on the predicted update field:

- `root_fd_loss = MSE(pred_root_vel, FD(pred_root_pos))`
- `joint_fd_loss = MSE(pred_jvel, FD(pred_jpos))`

Training loss becomes:

```text
loss = mse(pred, target) + lambda_fd * (root_fd_loss + joint_fd_loss)
```

## Required Change First

This only makes sense if `x0` is also FD-consistent. Right now `x0 = torch.randn_like(x1)`, so the supervision target can violate the FD constraint by construction.

Instead, sample structured noise:

1. Sample random `root_pos` and `jpos`.
2. Compute `root_vel` and `jvel` from finite differences using dataset FPS.
3. Sample root rotation noise separately.
4. Concatenate into the same state layout as the dataset.

That makes `x0`, `x1 - x0`, and the EqM target compatible with the FD penalty.

## Recommended Implementation Order

1. Add a helper to build FD-consistent `x0` in LAFAN1 EqM training.
2. Add an auxiliary FD loss on the network prediction `pred`.
3. Expose a config weight such as `fd_loss_weight`.
4. Optionally project rollout velocities after each sampling step by recomputing them from positions.

## Notes

- The training-time FD loss is the main fix.
- The rollout-time projection is optional and mainly helps sampling stability.
- The same idea can also be applied to Flow Matching, but EqM requires structured noise first because its target depends on `x0`.
