import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import logging

# Add parent directory to path to import RFdiffusion modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import DictConfig, OmegaConf
from rfdiffusion.inference import utils as iu
from rfdiffusion.diffusion import Diffuser
from rfdiffusion.RoseTTAFoldModel import RoseTTAFoldModule
from rfdiffusion.util_module import ComputeAllAtomCoords

class RFDiffusionDistiller:
    """
    Class for distilling/fine-tuning the RFdiffusion model into a single-shot model
    """
    
    def __init__(self, config_path=None, teacher_ckpt_path=None, student_ckpt_path=None):
        """
        Initialize the distiller with teacher and student models
        
        Args:
            config_path: Path to the config file
            teacher_ckpt_path: Path to the teacher checkpoint
            student_ckpt_path: Path to the student checkpoint (optional)
        """
        # Configure logging
        self._log = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._log.info(f"Using device: {self.device}")
        
        # Path for teacher model
        ckpt_path = teacher_ckpt_path or "models/Base_ckpt.pt"
        self._log.info(f"Loading checkpoint from {ckpt_path}")
        
        # Load teacher checkpoint directly
        self.ckpt = torch.load(ckpt_path, map_location=self.device)
        
        # Setup configuration with values from the model checkpoint
        self._log.info("Setting up configuration from checkpoint")
        self.config_dict = self.ckpt['config_dict']
        
        # Initialize diffusion objects
        self._log.info("Setting up diffuser")
        self.setup_diffuser()
        
        # Create the teacher model
        self._log.info("Creating teacher model")
        self.teacher_model = self.setup_model(is_student=False)
        
        # Create a student model (initially a copy of the teacher)
        self._log.info("Creating student model")
        self.student_model = self.setup_model(is_student=True)
        
        # Either load provided checkpoint or copy teacher weights
        if student_ckpt_path is not None:
            self._log.info(f"Loading student weights from {student_ckpt_path}")
            student_ckpt = torch.load(student_ckpt_path, map_location=self.device)
            self.student_model.load_state_dict(student_ckpt['model_state_dict'], strict=True)
        else:
            self._log.info("Initializing student with teacher weights")
            self.student_model.load_state_dict(self.teacher_model.state_dict())
        
        # Set model modes
        self.student_model.train()
        self.teacher_model.eval()
        
        # Initialize the all-atom coordinate calculator
        self.allatom = ComputeAllAtomCoords().to(self.device)
        
        self._log.info("Initialized RFDiffusionDistiller successfully")
    
    def setup_model(self, is_student=False):
        """
        Setup the model with parameters from the checkpoint
        
        Args:
            is_student: Whether this is a student model
            
        Returns:
            The initialized model
        """
        # Get model config from checkpoint
        model_config = self.config_dict['model'].copy()
        
        # Get input dimensions from the checkpoint
        d_t1d = self.config_dict['preprocess']['d_t1d']
        d_t2d = self.config_dict['preprocess']['d_t2d']
        T = self.config_dict['diffuser']['T']
        
        # Remove parameters that are not part of the model's __init__
        # This list might need to be extended if more incompatible parameters are found
        params_to_remove = [
            'd_time_emb', 
            'd_time_emb_proj',
            'clear_previous_cfg', 
            'model_only_neighbors',
            'time_emb_type',
            'time_emb_dim',
        ]
        
        for param in params_to_remove:
            if param in model_config:
                del model_config[param]
                
        # Print the model config for debugging
        print("Model config parameters after filtering:", model_config.keys())
        
        # Create model
        model = RoseTTAFoldModule(**model_config, d_t1d=d_t1d, d_t2d=d_t2d, T=T).to(self.device)
        
        # Load weights for teacher model only
        if not is_student:
            model.load_state_dict(self.ckpt['model_state_dict'], strict=True)
        
        # Set mode
        if is_student:
            model.train()
        else:
            model.eval()
            
        return model
        
    def setup_diffuser(self):
        """
        Setup the diffuser with parameters from the checkpoint
        """
        # Get diffuser config
        diffuser_config = self.config_dict['diffuser']
        
        # Create diffuser
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schedules")
        os.makedirs(cache_dir, exist_ok=True)
        
        self.diffuser = Diffuser(**diffuser_config, cache_dir=cache_dir)
        
        # Store important parameters
        self.T = diffuser_config['T']
        self.crd_scale = diffuser_config['crd_scale']
    
    def save_student_model(self, path):
        """
        Save the student model checkpoint
        
        Args:
            path: Path to save the checkpoint
        """
        torch.save({
            'model_state_dict': self.student_model.state_dict(),
            'config_dict': self.config_dict
        }, path)
    
    def get_noise_level(self, timestep):
        """
        Helper method to get the noise level at a specific timestep
        
        Args:
            timestep: Current timestep (1-indexed)
            
        Returns:
            Noise level for this timestep
        """
        # Convert to 0-indexed for scheduler
        t_idx = timestep - 1
        return torch.sqrt(self.diffuser.eucl_diffuser.beta_schedule[t_idx])
        
    def _get_diffusion_step(self, protein_length=150, timestep=10):
        """
        Get a proper diffusion step using the diffuser and score function
        
        Args:
            protein_length: Length of protein
            timestep: Current timestep
            
        Returns:
            x_t, x_prev, seq
        """
        # Create sequence (all masked)
        seq = torch.full((protein_length,), 21, dtype=torch.long, device=self.device)
        seq = F.one_hot(seq, num_classes=22).float()  # [L,22]
        
        # Create initial noise for x_t
        # Scale by crd_scale as in the RFDiffusion model
        x_t = torch.randn(protein_length, 14, 3, device=self.device) * self.crd_scale
        
        # Compute score function (gradient of log probability) at timestep t
        score = self.compute_score(seq, x_t, timestep)
        
        # Apply score to get x_{t-1} using the proper posterior sampling rule
        x_prev = self.apply_score_update(x_t, score, timestep)
        
        return x_t, x_prev, seq
        
    def compute_score(self, seq, x_t, timestep):
        """
        Compute score function (gradient of log probability) at current state
        
        Args:
            seq: One-hot encoded sequence
            x_t: Coordinates at time t
            timestep: Current timestep
            
        Returns:
            Score function output (gradient)
        """
        # Create diffusion mask (all False to diffuse all residues)
        L = seq.shape[0]
        diffusion_mask = torch.zeros(L, dtype=torch.bool, device=self.device)
        
        # Preprocess for model input
        batch = self._preprocess(seq, x_t, timestep, diffusion_mask)
        
        # Get model prediction
        with torch.no_grad():
            msa_masked = batch['msa_masked']
            msa_full = batch['msa_full']
            seq_batch = batch['seq']
            xyz_prev = batch['xyz_prev']
            idx_pdb = batch['idx_pdb']
            t1d = batch['t1d']
            t2d = batch['t2d']
            xyz_t_batch = batch['xyz_t']
            alpha_t = batch['alpha_t']
            
            msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = self.teacher_model(
                msa_masked,
                msa_full,
                seq_batch,
                xyz_prev,
                idx_pdb,
                t1d=t1d,
                t2d=t2d,
                xyz_t=xyz_t_batch,
                alpha_t=alpha_t,
                msa_prev=None,
                pair_prev=None,
                state_prev=None,
                t=torch.tensor(timestep, device=self.device),
                return_infer=True,
                motif_mask=diffusion_mask
            )
            
            # Process the output to get full atom coordinates
            _, px0_full = self.allatom(torch.argmax(seq_batch, dim=-1), px0, alpha)
            px0_full = px0_full.squeeze()[:, :14]
            
            # The model predicts x0, and we need to calculate the score function
            # In diffusion models, the score is the gradient of the log probability
            # For variance exploding (VE) SDE, the score can be approximated as:
            # score = (predicted_x0 - x_t) / sigma_t^2
            # For variance preserving (VP) SDE, it would be different
            
            # For RFdiffusion's Euclidean diffusion, they use a linear noise schedule
            # Get the noise level at this timestep
            t_idx = timestep - 1  # Convert to 0-indexed
            beta_t = self.diffuser.eucl_diffuser.beta_schedule[t_idx]
            alpha_t = self.diffuser.eucl_diffuser.alpha_schedule[t_idx]
            alpha_bar_t = self.diffuser.eucl_diffuser.alphabar_schedule[t_idx]
            
            # Calculate the score estimate (gradient of log probability)
            # For this type of diffusion model:
            # score = (x_0 - sqrt(1-alpha_bar_t) * x_t) / (sigma_t^2)
            # where sigma_t^2 = beta_t * (1-alpha_bar_{t-1})/(1-alpha_bar_t)
            
            score = (px0_full - x_t) / beta_t
            
            return score
    
    def apply_score_update(self, x_t, score, timestep):
        """
        Apply score function update to get x_{t-1} following the reverse diffusion process
        
        Args:
            x_t: Coordinates at time t
            score: Score function output
            timestep: Current timestep
            
        Returns:
            x_{t-1}
        """
        # Get diffusion parameters for this timestep (0-indexed)
        t_idx = timestep - 1  # Convert to 0-indexed
        beta_t = self.diffuser.eucl_diffuser.beta_schedule[t_idx]
        alpha_t = self.diffuser.eucl_diffuser.alpha_schedule[t_idx]
        alpha_bar_t = self.diffuser.eucl_diffuser.alphabar_schedule[t_idx]
        
        # Get parameters for previous timestep
        alpha_bar_prev = self.diffuser.eucl_diffuser.alphabar_schedule[t_idx-1] if t_idx > 0 else torch.tensor(1.0, device=self.device)
        
        # Calculate posterior variance (beta_tilde)
        # Formula: beta_tilde_t = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t
        posterior_variance = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t
        
        # Calculate posterior mean coefficient for x_t
        # Formula: (sqrt(alpha_t) * (1 - alpha_bar_{t-1})/(1 - alpha_bar_t))
        posterior_mean_coef1 = torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        
        # Calculate posterior mean coefficient for x_0
        # Formula: (sqrt(alpha_bar_{t-1}) * beta_t) / (1 - alpha_bar_t)
        posterior_mean_coef2 = torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t)
        
        # Calculate predicted x_0 from score and current x_t
        # For Euclidean diffusion with linear schedule
        # x_0 = x_t + beta_t * score
        predicted_x0 = x_t + beta_t * score
        
        # Calculate the mean of the posterior distribution
        # mu_t = posterior_mean_coef1 * x_t + posterior_mean_coef2 * x_0
        posterior_mean = posterior_mean_coef1 * x_t + posterior_mean_coef2 * predicted_x0
        
        # Sample from the posterior distribution
        # x_{t-1} ~ N(posterior_mean, posterior_variance * I)
        posterior_noise = torch.randn_like(x_t) * torch.sqrt(posterior_variance)
        x_prev = posterior_mean + posterior_noise
        
        return x_prev
        
    def get_single_diffusion_step(self, protein_length=150, timestep=10):
        """
        Generate a single step of the diffusion process for training
        
        Args:
            protein_length: Length of the protein
            timestep: The timestep t to generate
            
        Returns:
            Dictionary containing x_t, x_t-1, and other data needed for training
        """
        with torch.no_grad():
            # Create a diffusion mask (all False to diffuse all residues)
            diffusion_mask = torch.zeros(protein_length, dtype=torch.bool, device=self.device)
            
            # Get proper diffusion step with score function
            x_t, x_t_prev, seq_t = self._get_diffusion_step(protein_length, timestep)
            
            # Preprocess for model input
            batch = self._preprocess(seq_t, x_t, timestep, diffusion_mask)
            batch.update({
                'x_t': x_t,
                'x_t_prev': x_t_prev,
                'timestep': timestep,
                'diffusion_mask': diffusion_mask
            })
            
            return batch
    
    def _preprocess(self, seq, xyz_t, t, diffusion_mask):
        """
        Preprocess inputs for the model
        
        Args:
            seq: Sequence
            xyz_t: Coordinates at time t
            t: Timestep
            diffusion_mask: Diffusion mask
            
        Returns:
            Dictionary of preprocessed inputs
        """
        L = seq.shape[0]
        T = self.T
                
        # MSA features 
        msa_masked = torch.zeros((1, 1, L, 48), device=self.device)
        msa_masked[:, :, :, :22] = seq[None, None]
        msa_masked[:, :, :, 22:44] = seq[None, None]
        msa_masked[:, :, 0, 46] = 1.0
        msa_masked[:, :, -1, 47] = 1.0
        
        msa_full = torch.zeros((1, 1, L, 25), device=self.device)
        msa_full[:, :, :, :22] = seq[None, None]
        msa_full[:, :, 0, 23] = 1.0
        msa_full[:, :, -1, 24] = 1.0
        
        # T1D features
        t1d = torch.zeros((1, 1, L, 21), device=self.device)
        seqt1d = torch.clone(seq)
        for idx in range(L):
            if seqt1d[idx, 21] == 1:
                seqt1d[idx, 20] = 1
                seqt1d[idx, 21] = 0
        
        t1d[:, :, :, :21] = seqt1d[None, None, :, :21]
        
        # Timestep feature (1-t/T for non-motif residues, 1 for motif residues)
        timefeature = torch.zeros(L, dtype=torch.float32, device=self.device)
        timefeature[diffusion_mask] = 1
        timefeature[~diffusion_mask] = 1 - t/T
        timefeature = timefeature[None, None, ..., None]
        
        t1d = torch.cat((t1d, timefeature), dim=-1).float()
        
        # XYZ processing
        xyz_t_clone = xyz_t.clone()
        # Set sidechains to NaN for masked residues
        mask_indices = torch.where(seq[:, 21] == 1)
        if len(mask_indices[0]) > 0:
            xyz_t_clone[mask_indices[0], 3:, :] = float('nan')
        
        xyz_t_clone = xyz_t_clone[None, None]
        xyz_t_clone = torch.cat((xyz_t_clone, torch.full((1, 1, L, 13, 3), float('nan'), device=self.device)), dim=3)
        
        # T2D features
        t2d = self.xyz_to_t2d(xyz_t_clone)
        
        # Index features
        idx = torch.arange(L, device=self.device)[None]
        
        # Alpha features (placeholder for torsions)
        alpha_t = torch.zeros((1, 1, L, 30), device=self.device)
        
        return {
            'msa_masked': msa_masked,
            'msa_full': msa_full,
            'seq': seq[None],
            'xyz_prev': torch.squeeze(xyz_t_clone, dim=0),
            'idx_pdb': idx,
            't1d': t1d,
            't2d': t2d,
            'xyz_t': xyz_t_clone, 
            'alpha_t': alpha_t
        }
    
    def xyz_to_t2d(self, xyz):
        """
        Convert xyz coordinates to 2D distance and orientation features
        
        Args:
            xyz: Coordinates
            
        Returns:
            2D features
        """
        # This is a simplified placeholder
        # In practice, use the actual implementation from RFdiffusion
        B, T, L = xyz.shape[:3]
        t2d = torch.zeros((B, T, L, L, 44), device=self.device)
        return t2d
    
    def compute_teacher_prediction(self, batch):
        """
        Get the teacher model's prediction for a given batch
        
        Args:
            batch: Batch data from get_single_diffusion_step
            
        Returns:
            The teacher's prediction of x0
        """
        with torch.no_grad():
            msa_masked = batch['msa_masked']
            msa_full = batch['msa_full']
            seq = batch['seq']
            xyz_prev = batch['xyz_prev']
            idx_pdb = batch['idx_pdb']
            t1d = batch['t1d']
            t2d = batch['t2d']
            xyz_t = batch['xyz_t']
            alpha_t = batch['alpha_t']
            t = batch['timestep']
            diffusion_mask = batch['diffusion_mask']
            
            msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = self.teacher_model(
                msa_masked,
                msa_full,
                seq,
                xyz_prev,
                idx_pdb,
                t1d=t1d,
                t2d=t2d,
                xyz_t=xyz_t,
                alpha_t=alpha_t,
                msa_prev=None,
                pair_prev=None,
                state_prev=None,
                t=torch.tensor(t, device=self.device),
                return_infer=True,
                motif_mask=diffusion_mask
            )
            
            # Process the output to get full atom coordinates
            _, px0_full = self.allatom(torch.argmax(seq, dim=-1), px0, alpha)
            px0_full = px0_full.squeeze()[:, :14]
            
            return px0_full
    
    def compute_student_prediction(self, batch):
        """
        Get the student model's prediction for a given batch
        
        Args:
            batch: Batch data from get_single_diffusion_step
            
        Returns:
            The student's prediction
        """
        msa_masked = batch['msa_masked']
        msa_full = batch['msa_full']
        seq = batch['seq']
        xyz_prev = batch['xyz_prev']
        idx_pdb = batch['idx_pdb']
        t1d = batch['t1d']
        t2d = batch['t2d']
        xyz_t = batch['xyz_t']
        alpha_t = batch['alpha_t']
        t = batch['timestep']
        diffusion_mask = batch['diffusion_mask']
        
        msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = self.student_model(
            msa_masked,
            msa_full,
            seq,
            xyz_prev,
            idx_pdb,
            t1d=t1d,
            t2d=t2d,
            xyz_t=xyz_t,
            alpha_t=alpha_t,
            msa_prev=None,
            pair_prev=None,
            state_prev=None,
            t=torch.tensor(t, device=self.device),
            return_infer=True,
            motif_mask=diffusion_mask
        )
        
        # Process the output to get full atom coordinates
        _, px0_full = self.allatom(torch.argmax(seq, dim=-1), px0, alpha)
        px0_full = px0_full.squeeze()[:, :14]
        
        return px0_full
    
    def train_step(self, optimizer, protein_length=150, timestep=None):
        """
        Perform a single training step
        
        Args:
            optimizer: PyTorch optimizer
            protein_length: Length of the protein
            timestep: Specific timestep to train on (random if None)
            
        Returns:
            Loss value
        """
        # Sample a random timestep if not provided
        if timestep is None:
            timestep = torch.randint(1, self.T, (1,)).item()
            
        # Get a batch of data
        batch = self.get_single_diffusion_step(protein_length, timestep)
        
        # Get teacher prediction
        teacher_pred = self.compute_teacher_prediction(batch)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Get student prediction
        student_pred = self.compute_student_prediction(batch)
        
        # Compute loss 
        # MSE loss between teacher and student predictions
        loss = F.mse_loss(student_pred, teacher_pred)
        
        # Backpropagate
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        return loss.item()
    
    def train(self, num_steps=1000, lr=1e-4, save_path=None, protein_length=150):
        """
        Train the student model
        
        Args:
            num_steps: Number of training steps
            lr: Learning rate
            save_path: Path to save the final model
            protein_length: Length of the protein to generate
            
        Returns:
            List of loss values
        """
        # Create optimizer
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=lr)
        
        # Training loop
        losses = []
        for step in range(num_steps):
            loss = self.train_step(optimizer, protein_length)
            losses.append(loss)
            
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss:.6f}")
                
            if save_path is not None and step % 100 == 0:
                self.save_student_model(f"{save_path}_step_{step}.pt")
                
        # Save final model
        if save_path is not None:
            self.save_student_model(save_path)
            
        return losses

    def encode_trajectory(self, protein_length=150, num_steps=None, include_x0=True):
        """
        Generate a full diffusion trajectory for encoding using the score function
        
        Args:
            protein_length: Length of the protein
            num_steps: Number of diffusion steps (uses T if None)
            include_x0: Whether to include the ground truth x0
            
        Returns:
            Dictionary with trajectory data
        """
        with torch.no_grad():
            if num_steps is None:
                num_steps = self.T
                
            # Create initial noised state x_T
            x_T = torch.randn(protein_length, 14, 3, device=self.device) * self.crd_scale
            
            # Create sequence (all masked)
            seq = torch.full((protein_length,), 21, dtype=torch.long, device=self.device)
            seq = F.one_hot(seq, num_classes=22).float()
            
            # Setup trajectory dict
            trajectory = {
                'x_t': [x_T],  # Start with x_T
                'timesteps': list(range(self.T, 0, -1))[:num_steps],  # Count down from T to 1
                'seq': seq,
                'diffusion_mask': torch.zeros(protein_length, dtype=torch.bool, device=self.device)
            }
            
            # Starting from x_T, generate states x_{T-1}, x_{T-2}, etc. using score function
            x_t = x_T
            for t in range(self.T, max(self.T - num_steps, 0), -1):
                # Compute score for current state
                score = self.compute_score(seq, x_t, t)
                
                # Apply score to get x_{t-1}
                x_t = self.apply_score_update(x_t, score, t)
                
                # Add to trajectory if within the desired number of steps
                if t > self.T - num_steps:
                    trajectory['x_t'].append(x_t)
            
            # Reverse the order to go from t=1 to t=T for consistency with timesteps
            trajectory['x_t'] = trajectory['x_t'][::-1]
            
            if include_x0:
                # For x_0, we use the model's prediction at t=1
                x_1 = trajectory['x_t'][0]  # First element is now t=1
                score_1 = self.compute_score(seq, x_1, 1)
                
                # The score at t=1 gives us information about x_0
                noise_level_1 = self.diffuser.get_noise_level(1)
                x_0 = x_1 + (noise_level_1**2) * score_1
                
                trajectory['x_0'] = x_0
                
            return trajectory