import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from score_function_example.distill_model import RFDiffusionDistiller

def main():
    # Initialize the distiller
    print("Initializing RFDiffusionDistiller...")
    
    # Try different possible paths for the model weights
    possible_paths = [
        "model_weights/Base_ckpt.pt",
        "../model_weights/Base_ckpt.pt",
        "/home/kmitchell/RFdiffusion/model_weights/Base_ckpt.pt",
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print(f"Model weights not found in any of the expected locations.")
        print("Please make sure the RFdiffusion model weights are downloaded.")
        print("You can download them using the scripts/download_models.sh script.")
        return
    
    print(f"Found model weights at: {model_path}")
    
    try:
        distiller = RFDiffusionDistiller(
            teacher_ckpt_path=model_path
        )
        
        # Use a smaller protein for faster execution
        protein_length = 25
        
        # Example 1: Access a single step of diffusion
        print(f"\nGetting a single diffusion step for protein length {protein_length}...")
        batch = distiller.get_single_diffusion_step(protein_length=protein_length, timestep=10)
        print(f"Generated step data with shape: {batch['x_t'].shape}")
        
        # Example 2: Get teacher model prediction
        print("\nGetting teacher model prediction...")
        teacher_pred = distiller.compute_teacher_prediction(batch)
        print(f"Teacher prediction shape: {teacher_pred.shape}")
        
        # Example 3: Get student model prediction
        print("\nGetting student model prediction...")
        student_pred = distiller.compute_student_prediction(batch)
        print(f"Student prediction shape: {student_pred.shape}")
        
        # Example 4: Train for a few steps
        print("\nTraining the student model for a few steps...")
        optimizer = torch.optim.Adam(distiller.student_model.parameters(), lr=1e-4)
        losses = []
        for i in range(3):  # Just a few steps for demonstration
            loss = distiller.train_step(optimizer, protein_length=protein_length)
            losses.append(loss)
            print(f"Step {i}, Loss: {loss:.6f}")
        
        # Example 5: Generate full trajectory for encoding
        print("\nGenerating short trajectory for encoding...")
        trajectory = distiller.encode_trajectory(protein_length=protein_length, num_steps=5)
        print(f"Trajectory contains {len(trajectory['x_t'])} timesteps")
        
        # Example 6: Save student model
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
        
        print("\nScript completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()