# RFdiffusion Score Function Access and Distillation

This directory contains tools to access the score function of RFdiffusion for distillation or fine-tuning into a single-shot model. The code allows you to:

1. Access the RFdiffusion score function
2. Perform single steps of diffusion
3. Extract gradients of trajectories
4. Set up a teacher-student distillation framework

## Overview

The provided code demonstrates how to access the internal components of RFdiffusion without modifying the original codebase. This is done by leveraging the existing classes and interfaces provided by RFdiffusion.

## What is a score function?

In diffusion models, the score function is the gradient of the log probability density of the data distribution. It represents the direction in which the model believes the noisy data should move to become less noisy. In RFdiffusion, the model predicts X0 (clean structure) directly from Xt (noisy structure), which implicitly contains the score information.

## Key Files

- `distill_model.py`: Contains the `RFDiffusionDistiller` class that handles model distillation
- `example_distillation.py`: Example script showing how to use the distiller

## Using the Distiller

The `RFDiffusionDistiller` class provides the following functionality:

### Initialization

```python
distiller = RFDiffusionDistiller(
    teacher_ckpt_path="models/Base_ckpt.pt",  # Path to teacher model checkpoint
    student_ckpt_path=None       # Optional: Path to student model checkpoint
)
```

### Getting a Single Diffusion Step

```python
batch = distiller.get_single_diffusion_step(
    protein_length=25,  # Length of the protein
    timestep=10         # Timestep to generate
)
```

This returns a dictionary containing:
- `x_t`: Coordinates at timestep t
- `x_t_prev`: Coordinates at timestep t-1 (for supervision)
- Various model inputs needed for prediction

### Accessing the Teacher's Score Function

```python
teacher_pred = distiller.compute_teacher_prediction(batch)
```

This computes the teacher model's prediction of x0 given the noised input x_t. This prediction contains the implicit score function.

### Training the Student Model

```python
# Initialize optimizer
optimizer = torch.optim.Adam(distiller.student_model.parameters(), lr=1e-4)

# Single training step
loss = distiller.train_step(optimizer, protein_length=25, timestep=10)

# Full training loop
losses = distiller.train(
    num_steps=1000,
    lr=1e-4,
    save_path="student_model.pt",
    protein_length=25
)
```

### Generating Full Trajectories

```python
trajectory = distiller.encode_trajectory(
    protein_length=25,
    num_steps=5,
    include_x0=True
)
```

This generates a full diffusion trajectory for encoding or analyzing.

### Saving the Student Model

```python
distiller.save_student_model("student_model.pt")
```

## Implementation Details

Here's what our implementation does under the hood:

1. **Loads the model checkpoint directly**: Rather than using the complex inference pipeline, we load the checkpoint and extract its configuration
2. **Filters incompatible parameters**: Removes any parameters from the config that aren't compatible with model initialization
3. **Creates synthetic diffusion steps**: Generates noised protein structures for training without relying on the integrated diffusion pipeline
4. **Ensures device consistency**: Keeps all tensor operations on the same device
5. **Provides model interfaces**: Offers clean interfaces to access the score function through the model's forward pass

## Understanding the Score Function

In RFdiffusion, the score function is accessed through the model's forward pass:

```python
msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = model(
    msa_masked, msa_full, seq, xyz_prev, idx_pdb,
    t1d=t1d, t2d=t2d, xyz_t=xyz_t, alpha_t=alpha_t,
    t=torch.tensor(t).to(device),
    return_infer=True,
    motif_mask=diffusion_mask
)
```

Here, `px0` is the model's prediction of the clean structure (x0) given the noisy structure at time `t`. This prediction implicitly contains the information needed to denoise the structure, which is equivalent to the score.

## Advanced Use Cases

### Accessing Gradients

To access gradients of the diffusion process, you can modify the batch computation to include `requires_grad=True`:

```python
batch = distiller.get_single_diffusion_step(protein_length=25, timestep=10)
batch['x_t'].requires_grad = True

# Compute prediction
pred = distiller.compute_student_prediction(batch)

# Compute loss
loss = some_loss_function(pred, target)

# Compute gradients
loss.backward()

# Access gradients
gradients = batch['x_t'].grad
```

### Custom Loss Functions

You can implement custom loss functions to guide the training process:

```python
def custom_loss(student_pred, teacher_pred, batch):
    # Standard MSE loss
    mse_loss = F.mse_loss(student_pred, teacher_pred)
    
    # Additional regularization terms
    # ...
    
    return mse_loss + additional_terms
```

## Requirements

- RFdiffusion (installed and configured)
- PyTorch
- All dependencies required by RFdiffusion

## Troubleshooting

If you encounter errors related to model parameters, you may need to add more parameters to the `params_to_remove` list in the `setup_model` method of `RFDiffusionDistiller`. This is because the model configuration in the checkpoint might contain parameters that aren't expected by the model's `__init__` method.

## References

- RFdiffusion: [https://github.com/RosettaCommons/RFdiffusion](https://github.com/RosettaCommons/RFdiffusion)
- Original paper: [https://www.biorxiv.org/content/10.1101/2022.12.09.519842v1](https://www.biorxiv.org/content/10.1101/2022.12.09.519842v1)