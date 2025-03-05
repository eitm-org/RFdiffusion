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
        
        distiller = RFDiffusionDistiller(
            teacher_ckpt_path=model_path
        )
        
        # Use a smaller protein for faster execution
        protein_length = 25
        
        # Example 1: Access a single step of diffusion with score function
        print(f"\nGetting a single diffusion step for protein length {protein_length}...")
        batch = distiller.get_single_diffusion_step(protein_length=protein_length, timestep=10)
        print(f"Generated step data with shape: {batch['x_t'].shape}")
        
        # Example 2: Directly calculate the score function
        print("\nCalculating score function...")
        seq = batch['seq'][0]  # Get the sequence from the batch
        score = distiller.compute_score(seq, batch['x_t'], batch['timestep'])
        print(f"Score function shape: {score.shape}")
        
        # Example 3: Apply the score update
        print("\nApplying score update...")
        x_prev_from_score = distiller.apply_score_update(batch['x_t'], score, batch['timestep'])
        print(f"Updated coordinates shape: {x_prev_from_score.shape}")
        
        # Example 4: Get teacher model prediction
        print("\nGetting teacher model prediction...")
        teacher_pred = distiller.compute_teacher_prediction(batch)
        print(f"Teacher prediction shape: {teacher_pred.shape}")
        
        # Example 5: Get student model prediction
        print("\nGetting student model prediction...")
        student_pred = distiller.compute_student_prediction(batch)
        print(f"Student prediction shape: {student_pred.shape}")
        
        # Example 6: Train for a few steps
        print("\nTraining the student model for a few steps...")
        optimizer = torch.optim.Adam(distiller.student_model.parameters(), lr=1e-4)
        losses = []
        for i in range(3):  # Just a few steps for demonstration
            loss = distiller.train_step(optimizer, protein_length=protein_length)
            losses.append(loss)
            print(f"Step {i}, Loss: {loss:.6f}")
        
        # Example 7: Generate full trajectory for encoding
        print("\nGenerating short trajectory for encoding...")
        trajectory = distiller.encode_trajectory(protein_length=protein_length, num_steps=5)
        print(f"Trajectory contains {len(trajectory['x_t'])} timesteps")
        
        # Example 8: Save student model
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