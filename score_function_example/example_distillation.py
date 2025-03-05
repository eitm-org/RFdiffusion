import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from score_function_example.distill_model import RFDiffusionDistiller

def main():
    try:
        # Path to the model weights
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "model_weights/Base_ckpt.pt")
        
        # Note on GPU usage:
        # To manage GPU memory usage, it's best to use CUDA_VISIBLE_DEVICES before running the script.
        # Examples:
        #   - To use only GPU 0: CUDA_VISIBLE_DEVICES=0 python example_distillation.py
        #   - To use no GPU (CPU only): CUDA_VISIBLE_DEVICES= python example_distillation.py
        #   - To use multiple GPUs, use PyTorch DataParallel or DistributedDataParallel
        
        distiller = RFDiffusionDistiller(
            teacher_ckpt_path=model_path
        )
        
        # Use a smaller protein for faster execution
        protein_length = 25
        
        # Example 1: Access a single step of diffusion with score function
        print(f"\nGetting a single diffusion step for protein length {protein_length}...")
        batch = distiller.get_single_diffusion_step(protein_length=protein_length, timestep=10)
        print(f"Generated step data with shape: {batch['x_t'].shape}")
        
        # Example 2: Calculate teacher and student score functions
        print("\nCalculating teacher score function...")
        seq = batch['seq'][0]  # Get the sequence from the batch
        teacher_score = distiller.compute_teacher_score(seq, batch['x_t'], batch['timestep'])
        print(f"Teacher score function shape: {teacher_score.shape}")
        
        print("\nCalculating student score function...")
        student_score = distiller.compute_student_score(seq, batch['x_t'], batch['timestep'])
        print(f"Student score function shape: {student_score.shape}")
        
        # Example 3: Calculate score difference between teacher and student
        print("\nCalculating score difference...")
        score_diff = torch.norm(teacher_score - student_score, dim=-1).mean()
        print(f"Average score difference: {score_diff:.6f}")
        
        # Example 4: Apply the score update using teacher score
        print("\nApplying score update using teacher score...")
        x_prev_from_teacher = distiller.apply_score_update(batch['x_t'], teacher_score, batch['timestep'])
        print(f"Updated coordinates shape: {x_prev_from_teacher.shape}")
        
        # Example 5: Apply the score update using student score
        print("\nApplying score update using student score...")
        x_prev_from_student = distiller.apply_score_update(batch['x_t'], student_score, batch['timestep'])
        print(f"Updated coordinates shape: {x_prev_from_student.shape}")
        
        # Example 6: Get teacher model prediction of x0
        print("\nGetting teacher model prediction of x0...")
        teacher_pred = distiller.compute_teacher_prediction(batch)
        print(f"Teacher x0 prediction shape: {teacher_pred.shape}")
        
        # Example 7: Get student model prediction of x0
        print("\nGetting student model prediction of x0...")
        student_pred = distiller.compute_student_prediction(batch)
        print(f"Student x0 prediction shape: {student_pred.shape}")
        
        # Example 8: Calculate prediction difference between teacher and student
        print("\nCalculating x0 prediction difference...")
        pred_diff = torch.norm(teacher_pred - student_pred, dim=-1).mean()
        print(f"Average x0 prediction difference: {pred_diff:.6f}")
        
        # Example 9: Train for a few steps using x0 prediction matching
        print("\nTraining the student model on x0 prediction for a few steps...")
        optimizer = torch.optim.Adam(distiller.student_model.parameters(), lr=1e-4)
        x0_losses = []
        for i in range(2):  # Just a few steps for demonstration
            loss = distiller.train_step(optimizer, protein_length=protein_length, train_on_score=False)
            x0_losses.append(loss)
            print(f"Step {i}, X0 Prediction Loss: {loss:.6f}")
        
        # Example 10: Train for a few steps using score matching
        print("\nTraining the student model on score matching for a few steps...")
        optimizer = torch.optim.Adam(distiller.student_model.parameters(), lr=1e-4)
        score_losses = []
        for i in range(2):  # Just a few steps for demonstration
            loss = distiller.train_step(optimizer, protein_length=protein_length, train_on_score=True)
            score_losses.append(loss)
            print(f"Step {i}, Score Matching Loss: {loss:.6f}")
            
        # Combine losses for plotting
        losses = x0_losses + score_losses
        
        # Example 11: Generate full trajectory for encoding
        print("\nGenerating short trajectory for encoding...")
        trajectory = distiller.encode_trajectory(protein_length=protein_length, num_steps=5)
        print(f"Trajectory contains {len(trajectory['x_t'])} timesteps")
        
        # Example 12: Save student model
        print("\nSaving student model checkpoint...")
        os.makedirs("checkpoints", exist_ok=True)
        distiller.save_student_model("checkpoints/student_model.pt")
        print("Saved student model to checkpoints/student_model.pt")
        
        # Show loss curve
        if losses:
            plt.figure(figsize=(10, 5))
            plt.plot(losses)
            plt.title("Training Loss")
            plt.xlabel("Steps")
            plt.ylabel("MSE Loss")
            plt.savefig("training_loss.png")
            print("Saved loss curve to training_loss.png")
        
        print("\nDistillation finished!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()