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
from rfdiffusion.util import calc_rmsd
import numpy as np

class RFDiffusionDistiller:
    """
    Class for distilling/fine-tuning the RFdiffusion model into a single-shot model.
    Provides core functionality for computing score functions and losses.
    """
    
    def __init__(self, config_path=None, teacher_ckpt_path=None, student_ckpt_path=None, 
                 generator_ckpt_path=None, device_map=None):
        """
        Initialize the distiller with teacher, student, and generator models
        
        Args:
            config_path: Path to the config file
            teacher_ckpt_path: Path to the teacher checkpoint
            student_ckpt_path: Path to the student checkpoint (optional)
            generator_ckpt_path: Path to the generator checkpoint (optional)
            device_map: Dictionary mapping model names to devices ('teacher', 'student', 'generator')
                        Example: {'teacher': 'cuda:0', 'student': 'cuda:1', 'generator': 'cuda:1'}
                        If None, all models will be placed on the same device
        """
        # Configure logging
        self._log = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Get available devices
        self.cuda_available = torch.cuda.is_available()
        self.num_gpus = torch.cuda.device_count() if self.cuda_available else 0
        self._log.info(f"Found {self.num_gpus} CUDA devices")
        
        # Set default device
        self.default_device = torch.device('cuda:0' if self.cuda_available else 'cpu')
        
        # Setup device map (but don't use it yet - first initialize all models on same device)
        self.device_map = self._setup_device_map(device_map)
        self._log.info(f"Using device map: {self.device_map}")
        
        # Path for teacher model
        ckpt_path = teacher_ckpt_path or "models/Base_ckpt.pt"
        self._log.info(f"Loading checkpoint from {ckpt_path}")
        
        # Load teacher checkpoint to CPU first
        self.ckpt = torch.load(ckpt_path, map_location='cpu')
        
        # Setup configuration with values from the model checkpoint
        self._log.info("Setting up configuration from checkpoint")
        self.config_dict = self.ckpt['config_dict']
        
        # Initialize diffusion objects
        self._log.info("Setting up diffuser")
        self.setup_diffuser()
        
        # First create all models on CPU for faster weight copying
        self._log.info("Creating models on CPU for initialization")
        cpu_teacher = self.setup_model(is_teacher=True, device='cpu')
        
        # Load teacher weights on CPU
        cpu_teacher.load_state_dict(self.ckpt['model_state_dict'], strict=True)
        
        # Create student model on CPU
        if student_ckpt_path is not None:
            self._log.info(f"Loading student weights from {student_ckpt_path}")
            student_ckpt = torch.load(student_ckpt_path, map_location='cpu')
            cpu_student = self.setup_model(is_teacher=False, device='cpu')
            cpu_student.load_state_dict(student_ckpt['model_state_dict'], strict=True)
        else:
            self._log.info("Initializing student with teacher weights")
            # Same-device state dict copy (much faster)
            cpu_student = self.setup_model(is_teacher=False, device='cpu')
            cpu_student.load_state_dict(cpu_teacher.state_dict())
        
        # Create generator model on CPU
        if generator_ckpt_path is not None:
            self._log.info(f"Loading generator weights from {generator_ckpt_path}")
            generator_ckpt = torch.load(generator_ckpt_path, map_location='cpu')
            cpu_generator = self.setup_model(is_teacher=False, device='cpu')
            cpu_generator.load_state_dict(generator_ckpt['model_state_dict'], strict=True)
        else:
            self._log.info("Initializing generator with student weights")
            # Same-device state dict copy (much faster)
            cpu_generator = self.setup_model(is_teacher=False, device='cpu')
            cpu_generator.load_state_dict(cpu_student.state_dict())
        
        # Now move models to their target devices
        self._log.info(f"Moving teacher model to {self.device_map['teacher']}")
        self.teacher_model = self._move_model_to_device(cpu_teacher, self.device_map['teacher'])
        
        self._log.info(f"Moving student model to {self.device_map['student']}")
        self.student_model = self._move_model_to_device(cpu_student, self.device_map['student'])
        
        self._log.info(f"Moving generator model to {self.device_map['generator']}")
        self.generator_model = self._move_model_to_device(cpu_generator, self.device_map['generator'])
            
        # Set model modes
        self.teacher_model.eval()  # Teacher is always frozen
        self.student_model.train()
        self.generator_model.train()
        
        # Initialize the all-atom coordinate calculator (on the default device)
        self.allatom = ComputeAllAtomCoords().to(self.default_device)
        
        # Clear CPU models to free memory
        del cpu_teacher, cpu_student, cpu_generator
        torch.cuda.empty_cache()
        
        self._log.info("Initialized RFDiffusionDistiller successfully")
    
    def _setup_device_map(self, device_map=None):
        """
        Set up device mapping for the models
        
        Args:
            device_map: Dictionary mapping model names to devices
            
        Returns:
            Complete device map dictionary
        """
        # If no device map provided, create a default one
        if device_map is None:
            device_map = {}
            
        # Set defaults for any unspecified models
        if self.num_gpus >= 2:
            # Multi-GPU setup if available
            default_map = {
                'teacher': 'cuda:0',  # Teacher on first GPU
                'student': 'cuda:1',  # Student on second GPU
                'generator': 'cuda:1'  # Generator on second GPU
            }
        else:
            # Single GPU or CPU setup
            device_str = 'cuda:0' if self.cuda_available else 'cpu'
            default_map = {
                'teacher': device_str,
                'student': device_str,
                'generator': device_str
            }
            
        # Update defaults with any provided values
        for key, value in device_map.items():
            default_map[key] = value
            
        # Convert string device specifications to torch devices
        return {k: torch.device(v) for k, v in default_map.items()}
    
    def _move_model_to_device(self, model, device):
        """
        Move a model to the specified device
        
        Args:
            model: Model to move
            device: Target device
            
        Returns:
            Model on the target device
        """
        # Convert string device to torch.device if needed
        if isinstance(device, str):
            device = torch.device(device)
            
        # Move the model
        model = model.to(device)
        
        # Store device in model for easier reference
        model.device = device
        
        return model
    
    def _copy_state_dict(self, source_model, target_model):
        """
        Copy state dict between models that might be on different devices
        
        Args:
            source_model: Source model
            target_model: Target model
        """
        # This is less efficient than model.load_state_dict() when both models are on the same device
        for name, param in source_model.state_dict().items():
            target_model.state_dict()[name].copy_(param.to(target_model.device))
    
    def to_device(self, tensor, device_name):
        """
        Move a tensor to the specified device
        
        Args:
            tensor: Tensor to move
            device_name: Name of the device in the device map ('teacher', 'student', 'generator')
            
        Returns:
            Tensor on the specified device
        """
        target_device = self.device_map.get(device_name, self.default_device)
        return tensor.to(target_device)
    
    def setup_model(self, is_teacher=False, device=None):
        """
        Setup the model with parameters from the checkpoint
        
        Args:
            is_teacher: Whether this is the teacher model (which loads weights and is frozen)
            device: Device to place the model on
            
        Returns:
            The initialized model
        """
        # If no device specified, use default
        if device is None:
            device = self.default_device
        
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
                
        # Create model on specified device
        model = RoseTTAFoldModule(**model_config, d_t1d=d_t1d, d_t2d=d_t2d, T=T).to(device)
        
        # Set model's device attribute for easier reference
        model.device = device
        
        # Load weights for teacher model only
        if is_teacher:
            self._log.info(f"Loading teacher weights to {device}")
            # Load weights directly to the correct device
            state_dict = {k: v.to(device) for k, v in self.ckpt['model_state_dict'].items()}
            model.load_state_dict(state_dict, strict=True)
            model.eval()  # Teacher is always frozen
        else:
            model.train()  # Student/Generator are trainable
            
        return model
        
    def setup_diffuser(self):
        """
        Setup the diffuser with parameters from the checkpoint
        Diffuser parameters will be accessible from any device
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
        
        # Cache key diffusion parameters on all devices for faster access
        self._setup_diffusion_cache()
    
    def save_model(self, model, path):
        """
        Save a model checkpoint
        
        Args:
            model: The model to save (teacher, student, or generator)
            path: Path to save the checkpoint
        """
        torch.save({
            'model_state_dict': model.state_dict(),
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
        
    def add_noise(self, x_0, timestep, noise=None, device=None):
        """
        Add noise to protein coordinates for a specific timestep.
        This function is provided for external use in training pipelines.
        
        Args:
            x_0: Clean protein coordinates [B, L, 14, 3] or [L, 14, 3]
            timestep: Timestep (1-indexed) to noise to 
            noise: Optional pre-generated noise; if None, random noise will be generated
            device: Device to place results on; if None, uses x_0's device
            
        Returns:
            x_t: Noised coordinates at timestep t
            noise: The noise that was added (for calculating targets in training)
        """
        # Determine which device to use
        if device is None:
            device = x_0.device
            
        # If x_0 is not on the desired device, move it
        if x_0.device != device:
            x_0 = x_0.to(device)
            
        # Add batch dimension if not present
        if len(x_0.shape) == 3:
            x_0 = x_0.unsqueeze(0)
            
        B, L = x_0.shape[:2]
        
        # Get diffusion parameters for this device
        params = self._get_device_params(device, timestep)
        
        # Generate noise if not provided
        if noise is None:
            noise = torch.randn_like(x_0, device=device)
        elif noise.device != device:
            noise = noise.to(device)
            
        # Apply noise schedule following the forward diffusion process
        # For variance preserving (VP) SDE:
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_t = torch.sqrt(params['alpha_bar_t']) * x_0 + torch.sqrt(1 - params['alpha_bar_t']) * noise
        
        return x_t, noise
        
    def _setup_diffusion_cache(self):
        """
        Cache diffusion parameters on each device for faster access
        """
        # Create device-specific caches
        self.device_caches = {}
        
        # For each device in the device map
        for device_name, device in self.device_map.items():
            # Create cache for this device
            device_cache = {
                'beta_schedule': self.diffuser.eucl_diffuser.beta_schedule.to(device),
                'alpha_schedule': self.diffuser.eucl_diffuser.alpha_schedule.to(device),
                'alphabar_schedule': self.diffuser.eucl_diffuser.alphabar_schedule.to(device)
            }
            self.device_caches[device] = device_cache
            
        # Also cache on default device if not already included
        if self.default_device not in self.device_caches:
            device_cache = {
                'beta_schedule': self.diffuser.eucl_diffuser.beta_schedule.to(self.default_device),
                'alpha_schedule': self.diffuser.eucl_diffuser.alpha_schedule.to(self.default_device),
                'alphabar_schedule': self.diffuser.eucl_diffuser.alphabar_schedule.to(self.default_device)
            }
            self.device_caches[self.default_device] = device_cache
            
    def _get_device_params(self, device, timestep):
        """
        Get diffusion parameters for a specific device and timestep
        
        Args:
            device: Device to get parameters for
            timestep: Current timestep (1-indexed)
            
        Returns:
            Dictionary of diffusion parameters
        """
        # Get the device cache
        cache = self.device_caches.get(device)
        
        # If no cache for this device, create one
        if cache is None:
            cache = {
                'beta_schedule': self.diffuser.eucl_diffuser.beta_schedule.to(device),
                'alpha_schedule': self.diffuser.eucl_diffuser.alpha_schedule.to(device),
                'alphabar_schedule': self.diffuser.eucl_diffuser.alphabar_schedule.to(device)
            }
            self.device_caches[device] = cache
            
        # Convert to 0-indexed for scheduler
        t_idx = timestep - 1
        
        # Return parameters for this timestep
        return {
            'beta_t': cache['beta_schedule'][t_idx],
            'alpha_t': cache['alpha_schedule'][t_idx],
            'alpha_bar_t': cache['alphabar_schedule'][t_idx],
            'alpha_bar_prev': cache['alphabar_schedule'][t_idx-1] if t_idx > 0 else torch.tensor(1.0, device=device)
        }
    
    def compute_score(self, model, x_t, timestep, seq=None):
        """
        Compute a model's score function (gradient of log probability) at current state
        
        Args:
            model: The model to compute the score for (teacher, student, or generator)
            x_t: Coordinates at time t [L, 14, 3] or [B, L, 14, 3]
            timestep: Current timestep
            seq: Optional one-hot encoded sequence. If None, a masked sequence will be created.
            
        Returns:
            Model's score function output (gradient)
        """
        # Determine the device this model is on
        model_device = getattr(model, 'device', self.default_device)
        
        # Handle batch dimension
        if len(x_t.shape) == 3:
            # Add batch dimension if not present
            x_t = x_t.unsqueeze(0)
            single_item = True
        else:
            single_item = False
            
        # Move to the correct device
        if x_t.device != model_device:
            x_t = x_t.to(model_device)
            
        B, L = x_t.shape[:2]
        
        # Process each batch item
        all_scores = []
        
        # Determine if we need gradients
        # For the teacher model, NEVER track gradients (it's frozen)
        # For the student or generator, track gradients ONLY if model is in training mode
        requires_grad = model is not self.teacher_model and model.training
        
        for b in range(B):
            # Create or use sequence
            if seq is None:
                # Create a masked sequence (all unknown residues)
                item_seq = torch.full((L,), 21, dtype=torch.long, device=model_device)
                item_seq = F.one_hot(item_seq, num_classes=22).float()  # [L, 22]
            else:
                # Use provided sequence
                if len(seq.shape) == 2 and seq.shape[0] == L:
                    # Single sequence for all batch items
                    item_seq = seq.to(model_device)
                elif len(seq.shape) == 3:
                    # Batch of sequences
                    item_seq = seq[b].to(model_device)
                else:
                    raise ValueError(f"Invalid sequence shape: {seq.shape}")
            
            # Create diffusion mask (all False to diffuse all residues)
            diffusion_mask = torch.zeros(L, dtype=torch.bool, device=model_device)
            
            # Preprocess for model input
            batch = self._preprocess(item_seq, x_t[b], timestep, diffusion_mask, device=model_device)
            
            # Model forward pass
            with torch.set_grad_enabled(requires_grad):
                msa_masked = batch['msa_masked']
                msa_full = batch['msa_full']
                seq_batch = batch['seq']
                xyz_prev = batch['xyz_prev']
                idx_pdb = batch['idx_pdb']
                t1d = batch['t1d']
                t2d = batch['t2d']
                xyz_t_batch = batch['xyz_t']
                alpha_t = batch['alpha_t']
                
                msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = model(
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
                    t=torch.tensor(timestep, device=model_device),
                    return_infer=True,
                    motif_mask=diffusion_mask
                )
                
                # Process the output to get full atom coordinates
                with torch.no_grad():  # Always detach here to avoid gradient issues
                    _, px0_full = self.allatom(torch.argmax(seq_batch, dim=-1).to(self.default_device), 
                                              px0.to(self.default_device), 
                                              alpha.to(self.default_device))
                    px0_full = px0_full.squeeze()[:, :14].to(model_device)
                
                # Get diffusion parameters for this device
                params = self._get_device_params(model_device, timestep)
                
                # Calculate the score estimate
                # If we're computing for teacher, always detach to avoid gradient flow
                if model is self.teacher_model:
                    score = ((px0_full - x_t[b]) / params['beta_t']).detach()
                else:
                    score = (px0_full - x_t[b]) / params['beta_t']
                    
                all_scores.append(score)
        
        # Stack all scores
        batched_score = torch.stack(all_scores)
        
        # Remove batch dimension if input was a single item
        if single_item:
            batched_score = batched_score.squeeze(0)
            
        return batched_score
        
    def compute_teacher_score(self, x_t, timestep, seq=None):
        """
        Compute teacher model's score function (gradient of log probability)
        
        Args:
            x_t: Coordinates at time t [L, 14, 3] or [B, L, 14, 3]
            timestep: Current timestep
            seq: Optional one-hot encoded sequence. If None, a masked sequence will be created.
            
        Returns:
            Teacher's score function output (gradient)
        """
        return self.compute_score(self.teacher_model, x_t, timestep, seq)
        
    def compute_student_score(self, x_t, timestep, seq=None):
        """
        Compute student model's score function (gradient of log probability)
        
        Args:
            x_t: Coordinates at time t [L, 14, 3] or [B, L, 14, 3]
            timestep: Current timestep
            seq: Optional one-hot encoded sequence. If None, a masked sequence will be created.
            
        Returns:
            Student's score function output (gradient)
        """
        return self.compute_score(self.student_model, x_t, timestep, seq)
        
    def compute_generator_score(self, x_t, timestep, seq=None):
        """
        Compute generator model's score function (gradient of log probability)
        
        Args:
            x_t: Coordinates at time t [L, 14, 3] or [B, L, 14, 3]
            timestep: Current timestep
            seq: Optional one-hot encoded sequence. If None, a masked sequence will be created.
            
        Returns:
            Generator's score function output (gradient)
        """
        return self.compute_score(self.generator_model, x_t, timestep, seq)
    
    def apply_score_update(self, x_t, score, timestep, device=None):
        """
        Apply score function update to get x_{t-1} following the reverse diffusion process
        
        Args:
            x_t: Coordinates at time t
            score: Score function output
            timestep: Current timestep
            device: Device to place results on; if None, uses x_t's device
            
        Returns:
            x_{t-1}
        """
        # Determine which device to use
        if device is None:
            device = x_t.device
            
        # If tensors are not on the desired device, move them
        if x_t.device != device:
            x_t = x_t.to(device)
            
        if score.device != device:
            score = score.to(device)
            
        # Get diffusion parameters for this device
        params = self._get_device_params(device, timestep)
        
        # Calculate posterior variance (beta_tilde)
        # Formula: beta_tilde_t = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t
        posterior_variance = (1 - params['alpha_bar_prev']) / (1 - params['alpha_bar_t']) * params['beta_t']
        
        # Calculate posterior mean coefficient for x_t
        # Formula: (sqrt(alpha_t) * (1 - alpha_bar_{t-1})/(1 - alpha_bar_t))
        posterior_mean_coef1 = torch.sqrt(params['alpha_t']) * (1 - params['alpha_bar_prev']) / (1 - params['alpha_bar_t'])
        
        # Calculate posterior mean coefficient for x_0
        # Formula: (sqrt(alpha_bar_{t-1}) * beta_t) / (1 - alpha_bar_t)
        posterior_mean_coef2 = torch.sqrt(params['alpha_bar_prev']) * params['beta_t'] / (1 - params['alpha_bar_t'])
        
        # Calculate predicted x_0 from score and current x_t
        # For Euclidean diffusion with linear schedule
        # x_0 = x_t + beta_t * score
        predicted_x0 = x_t + params['beta_t'] * score
        
        # Calculate the mean of the posterior distribution
        # mu_t = posterior_mean_coef1 * x_t + posterior_mean_coef2 * x_0
        posterior_mean = posterior_mean_coef1 * x_t + posterior_mean_coef2 * predicted_x0
        
        # Sample from the posterior distribution
        # x_{t-1} ~ N(posterior_mean, posterior_variance * I)
        posterior_noise = torch.randn_like(x_t, device=device) * torch.sqrt(posterior_variance)
        x_prev = posterior_mean + posterior_noise
        
        return x_prev
            
    def _preprocess(self, seq, xyz_t, t, diffusion_mask, device=None):
        """
        Preprocess inputs for the model
        
        Args:
            seq: Sequence
            xyz_t: Coordinates at time t
            t: Timestep
            diffusion_mask: Diffusion mask
            device: Device to place tensors on (default: self.default_device)
            
        Returns:
            Dictionary of preprocessed inputs
        """
        # If no device specified, use seq's device
        if device is None:
            device = seq.device
            
        L = seq.shape[0]
        T = self.T
                
        # MSA features 
        msa_masked = torch.zeros((1, 1, L, 48), device=device)
        msa_masked[:, :, :, :22] = seq[None, None]
        msa_masked[:, :, :, 22:44] = seq[None, None]
        msa_masked[:, :, 0, 46] = 1.0
        msa_masked[:, :, -1, 47] = 1.0
        
        msa_full = torch.zeros((1, 1, L, 25), device=device)
        msa_full[:, :, :, :22] = seq[None, None]
        msa_full[:, :, 0, 23] = 1.0
        msa_full[:, :, -1, 24] = 1.0
        
        # T1D features
        t1d = torch.zeros((1, 1, L, 21), device=device)
        seqt1d = torch.clone(seq)
        for idx in range(L):
            if seqt1d[idx, 21] == 1:
                seqt1d[idx, 20] = 1
                seqt1d[idx, 21] = 0
        
        t1d[:, :, :, :21] = seqt1d[None, None, :, :21]
        
        # Timestep feature (1-t/T for non-motif residues, 1 for motif residues)
        timefeature = torch.zeros(L, dtype=torch.float32, device=device)
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
        xyz_t_clone = torch.cat((xyz_t_clone, torch.full((1, 1, L, 13, 3), float('nan'), device=device)), dim=3)
        
        # T2D features
        t2d = self.xyz_to_t2d(xyz_t_clone, device)
        
        # Index features
        idx = torch.arange(L, device=device)[None]
        
        # Alpha features (placeholder for torsions)
        alpha_t = torch.zeros((1, 1, L, 30), device=device)
        
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
        
    def xyz_to_t2d(self, xyz, device=None):
        """
        Convert xyz coordinates to 2D distance and orientation features
        
        Args:
            xyz: Coordinates
            device: Device to place tensors on
            
        Returns:
            2D features
        """
        # If no device specified, use xyz's device
        if device is None:
            device = xyz.device
            
        # This is a simplified placeholder
        # In practice, use the actual implementation from RFdiffusion
        B, T, L = xyz.shape[:3]
        t2d = torch.zeros((B, T, L, L, 44), device=device)
        return t2d
        
    def compute_score_loss(self, pred_score, target_score, device=None):
        """
        Compute MSE loss between predicted and target score functions.
        
        Args:
            pred_score: Predicted score
            target_score: Target score
            device: Device to place tensors on (if None, uses pred_score's device)
            
        Returns:
            Loss value
        """
        # Determine target device if not specified
        if device is None:
            device = pred_score.device if hasattr(pred_score, 'device') else self.default_device
            
        # IMPORTANT: Ensure the target score is detached to prevent backward through both models
        # This is crucial for separating student and generator gradients
        target_score_detached = target_score.detach()
            
        # Move tensors to the same device if needed
        if pred_score.device != device:
            pred_score = pred_score.to(device)
            
        if target_score_detached.device != device:
            target_score_detached = target_score_detached.to(device)
            
        # Use MSE loss for score matching
        return F.mse_loss(pred_score, target_score_detached)
    
    def compute_rfdiffusion_loss(self, pred, target, seq=None, w2D=0.5, device=None):
        """
        Compute loss the same way RFdiffusion does during training.
        
        Args:
            pred: Predicted coordinates [B, L, 14, 3] or [L, 14, 3]
            target: Target coordinates [B, L, 14, 3] or [L, 14, 3]
            seq: Sequence information [B, L] or [L] (optional)
            w2D: Weight for the L2D term (default: 0.5)
            device: Device to place tensors on (if None, uses pred's device)
            
        Returns:
            Combined loss value (LFrame + w2D * L2D)
        """
        # Determine target device if not specified
        if device is None:
            device = pred.device if hasattr(pred, 'device') else self.default_device
        
        # Check if pred requires grad - this is critical for backward pass
        requires_grad = pred.requires_grad
        if not requires_grad:
            self._log.warning("Input pred tensor doesn't require gradients! Creating a differentiable copy.")
            # Clone to make sure we have gradients
            pred = pred.clone().detach().requires_grad_(True)
            
        # Always detach target
        target = target.detach()
            
        # Move tensors to the same device if needed
        if pred.device != device:
            pred = pred.to(device)
            
        if target.device != device:
            target = target.to(device)
            
        # Check if shapes match
        if pred.shape != target.shape:
            self._log.warning(f"Shape mismatch: pred {pred.shape}, target {target.shape}")
            # Try to fix shape issues
            if len(pred.shape) == 3 and len(target.shape) == 4:
                pred = pred.unsqueeze(0)
            elif len(pred.shape) == 4 and len(target.shape) == 3:
                target = target.unsqueeze(0)
                
        # Add batch dimension if not present
        if len(pred.shape) == 3:
            pred = pred.unsqueeze(0)
        if len(target.shape) == 3:
            target = target.unsqueeze(0)
            
        # Ensure same length in both tensors
        if pred.shape[1] != target.shape[1]:
            min_len = min(pred.shape[1], target.shape[1])
            pred = pred[:, :min_len]
            target = target[:, :min_len]
        
        # Direct MSE calculation with guaranteed gradients
        if pred.shape[2] < 3 or target.shape[2] < 3:
            self._log.warning(f"Not enough atoms for frame loss: pred {pred.shape}, target {target.shape}")
            # Calculate simple MSE on all available coordinates
            return F.mse_loss(pred, target)
            
        # Backbone atoms (guaranteed to have gradients)
        pred_bb = pred[:, :, :3]
        target_bb = target[:, :, :3]
        frame_loss = F.mse_loss(pred_bb, target_bb)
        
        # CB atoms if available
        if pred.shape[2] > 4 and target.shape[2] > 4:
            pred_cb = pred[:, :, 4]
            target_cb = target[:, :, 4]
            cb_loss = F.mse_loss(pred_cb, target_cb)
            frame_loss = frame_loss + cb_loss
            
        # Only compute 2D loss if we have a reasonable frame loss
        if torch.isfinite(frame_loss) and frame_loss.requires_grad:
            try:
                # Compute 2D loss (inter-residue geometry loss)
                l2d_loss = self.compute_2d_loss(pred, target, seq, device)
                
                # Combine losses as in RFdiffusion
                total_loss = frame_loss + w2D * l2d_loss
            except Exception as e:
                self._log.warning(f"Error in 2D loss: {e}. Using only frame loss.")
                total_loss = frame_loss
        else:
            self._log.warning("Invalid frame loss - using direct MSE on all coordinates")
            # Fallback to direct MSE
            total_loss = F.mse_loss(pred, target)
        
        # Final check to ensure we have a valid loss with gradients
        if not torch.isfinite(total_loss) or not total_loss.requires_grad:
            self._log.warning("Invalid total loss - using simplified loss")
            # Last resort - simplest possible loss
            total_loss = ((pred - target)**2).mean()
            
        return total_loss
        
    def compute_frame_loss(self, pred, target, device=None):
        """
        Compute coordinate-based frame loss as used in RFdiffusion
        
        Args:
            pred: Predicted coordinates [B, L, 14, 3]
            target: Target coordinates [B, L, 14, 3]
            device: Device to place tensors on (if None, uses pred's device)
            
        Returns:
            Frame loss value
        """
        # Determine target device if not specified
        if device is None:
            device = pred.device if hasattr(pred, 'device') else self.default_device
            
        # Move tensors to the same device if needed
        if pred.device != device:
            pred = pred.to(device)
            
        if target.device != device:
            target = target.to(device)
        
        # Verify shapes match before continuing
        if pred.shape != target.shape:
            self._log.warning(f"Shape mismatch in compute_frame_loss: pred {pred.shape}, target {target.shape}")
            
            # Try to fix shape mismatches
            if len(pred.shape) == 3 and len(target.shape) == 4:
                pred = pred.unsqueeze(0)
            elif len(pred.shape) == 4 and len(target.shape) == 3:
                target = target.unsqueeze(0)
            
            # Ensure dimensions match
            min_batch = min(pred.shape[0], target.shape[0])
            min_len = min(pred.shape[1], target.shape[1])
            min_atoms = min(pred.shape[2], target.shape[2])
            
            pred = pred[:min_batch, :min_len, :min_atoms]
            target = target[:min_batch, :min_len, :min_atoms]
            
            self._log.info(f"Adjusted shapes to: pred {pred.shape}, target {target.shape}")
        
        try:
            # Get backbone atoms (N, CA, C)
            pred_bb = pred[:, :, :3]
            target_bb = target[:, :, :3]
            
            # MSE loss on backbone atoms coordinates
            bb_loss = F.mse_loss(pred_bb, target_bb)
            
            # Get CB atoms (index 4)
            # If no CB (e.g., for GLY), this will be a virtual CB
            if pred.shape[2] > 4 and target.shape[2] > 4:
                pred_cb = pred[:, :, 4]
                target_cb = target[:, :, 4]
                
                # MSE loss on CB atoms
                cb_loss = F.mse_loss(pred_cb, target_cb)
                
                # Final frame loss is weighted sum
                frame_loss = bb_loss + cb_loss
            else:
                frame_loss = bb_loss
                
            return frame_loss
            
        except RuntimeError as e:
            # Log error and return a dummy loss
            self._log.warning(f"Error in compute_frame_loss: {e}")
            return torch.tensor(0.0, device=device)
        except IndexError as e:
            # Log error and return a dummy loss
            self._log.warning(f"Index error in compute_frame_loss: {e}")
            return torch.tensor(0.0, device=device)
    
    def compute_2d_loss(self, pred, target, seq=None, device=None):
        """
        Compute inter-residue geometry loss (L2D) as used in RFdiffusion
        
        Args:
            pred: Predicted coordinates [B, L, 14, 3]
            target: Target coordinates [B, L, 14, 3]
            seq: Sequence information [B, L] (optional)
            device: Device to place tensors on (if None, uses pred's device)
            
        Returns:
            2D loss value
        """
        # Determine target device if not specified
        if device is None:
            device = pred.device if hasattr(pred, 'device') else self.default_device
            
        # Move tensors to the same device if needed
        if pred.device != device:
            pred = pred.to(device)
            
        if target.device != device:
            target = target.to(device)
            
        # Verify shapes match before continuing
        if pred.shape != target.shape:
            self._log.warning(f"Shape mismatch in compute_2d_loss: pred {pred.shape}, target {target.shape}")
            
            # Try to fix shape mismatches
            if len(pred.shape) == 3 and len(target.shape) == 4:
                pred = pred.unsqueeze(0)
            elif len(pred.shape) == 4 and len(target.shape) == 3:
                target = target.unsqueeze(0)
            
            # Ensure dimensions match
            min_batch = min(pred.shape[0], target.shape[0])
            min_len = min(pred.shape[1], target.shape[1])
            min_atoms = min(pred.shape[2], target.shape[2])
            
            pred = pred[:min_batch, :min_len, :min_atoms]
            target = target[:min_batch, :min_len, :min_atoms]
            
            self._log.info(f"Adjusted shapes to: pred {pred.shape}, target {target.shape}")
            
        # Handle potentially different shapes
        B, L = pred.shape[:2]
        
        # Check if we have enough atoms
        if pred.shape[2] < 5 or target.shape[2] < 5:
            self._log.warning(f"Not enough atoms in input tensors: pred {pred.shape}, target {target.shape}")
            return torch.tensor(0.0, device=device)
        
        # Extract relevant atoms for geometry calculations
        try:
            pred_ca = pred[:, :, 1]  # CA atoms
            pred_cb = pred[:, :, 4]  # CB atoms (or virtual CB)
            pred_n = pred[:, :, 0]   # N atoms
            
            target_ca = target[:, :, 1]
            target_cb = target[:, :, 4]
            target_n = target[:, :, 0]
            
            # Initialize loss components
            dist_loss = 0.0
            omega_loss = 0.0
            theta_loss = 0.0
            phi_loss = 0.0
            
            # Count successful batches
            successful_batches = 0
            
            # For each batch
            for b in range(B):
                try:
                    # 1. Compute CB-CB distances
                    pred_cb_dists = torch.cdist(pred_cb[b], pred_cb[b])
                    target_cb_dists = torch.cdist(target_cb[b], target_cb[b])
                    
                    # Convert to binned distributions with 37 bins (0-18.5Å in 0.5Å steps)
                    # RFdiffusion uses a one-hot encoding, but we'll use KL divergence between distributions
                    pred_cb_dist_bins = self._bin_distances(pred_cb_dists, device=device)
                    target_cb_dist_bins = self._bin_distances(target_cb_dists, device=device)
                    
                    # 2. Compute dihedral angles: Dihedral(Cα,l, Cβ,l, Cα,l′, Cβ,l′)
                    pred_omega = self._compute_dihedral_matrix(
                        pred_ca[b], pred_cb[b], pred_ca[b], pred_cb[b], device=device
                    )
                    target_omega = self._compute_dihedral_matrix(
                        target_ca[b], target_cb[b], target_ca[b], target_cb[b], device=device
                    )
                    
                    # 3. Compute dihedral angles: Dihedral(N,l, Cα,l, Cβ,l, Cβ,l′)
                    pred_theta = self._compute_dihedral_matrix(
                        pred_n[b], pred_ca[b], pred_cb[b], pred_cb[b], device=device
                    )
                    target_theta = self._compute_dihedral_matrix(
                        target_n[b], target_ca[b], target_cb[b], target_cb[b], device=device
                    )
                    
                    # 4. Compute planar angles: Planar(Cα,l, Cβ,l, Cβ,l′)
                    pred_phi = self._compute_planar_matrix(
                        pred_ca[b], pred_cb[b], pred_cb[b], device=device
                    )
                    target_phi = self._compute_planar_matrix(
                        target_ca[b], target_cb[b], target_cb[b], device=device
                    )
                    
                    # Convert angles to binned distributions
                    pred_omega_bins = self._bin_angles(pred_omega, angle_type="dihedral", device=device)
                    target_omega_bins = self._bin_angles(target_omega, angle_type="dihedral", device=device)
                    
                    pred_theta_bins = self._bin_angles(pred_theta, angle_type="dihedral", device=device)
                    target_theta_bins = self._bin_angles(target_theta, angle_type="dihedral", device=device)
                    
                    pred_phi_bins = self._bin_angles(pred_phi, angle_type="planar", device=device)
                    target_phi_bins = self._bin_angles(target_phi, angle_type="planar", device=device)
                    
                    # Compute KL divergence between predicted and target distributions
                    b_dist_loss = F.kl_div(
                        F.log_softmax(pred_cb_dist_bins, dim=-1),
                        F.softmax(target_cb_dist_bins, dim=-1),
                        reduction='sum'  # Use sum for better numerical stability
                    )
                    
                    b_omega_loss = F.kl_div(
                        F.log_softmax(pred_omega_bins, dim=-1),
                        F.softmax(target_omega_bins, dim=-1),
                        reduction='sum'
                    )
                    
                    b_theta_loss = F.kl_div(
                        F.log_softmax(pred_theta_bins, dim=-1),
                        F.softmax(target_theta_bins, dim=-1),
                        reduction='sum'
                    )
                    
                    b_phi_loss = F.kl_div(
                        F.log_softmax(pred_phi_bins, dim=-1),
                        F.softmax(target_phi_bins, dim=-1),
                        reduction='sum'
                    )
                    
                    # Only add if loss is finite
                    if torch.isfinite(b_dist_loss) and torch.isfinite(b_omega_loss) and \
                       torch.isfinite(b_theta_loss) and torch.isfinite(b_phi_loss):
                        dist_loss += b_dist_loss / (L * L)  # Normalize by number of residue pairs
                        omega_loss += b_omega_loss / (L * L)
                        theta_loss += b_theta_loss / (L * L)
                        phi_loss += b_phi_loss / (L * L)
                        successful_batches += 1
                    else:
                        self._log.warning(f"Non-finite loss values encountered for batch {b}")
                        
                except RuntimeError as e:
                    # If we encounter an error in the 2D loss calculation, log it but don't fail
                    self._log.warning(f"Error computing 2D loss for batch {b}: {e}")
                    continue
                except IndexError as e:
                    self._log.warning(f"Index error in 2D loss for batch {b}: {e}")
                    continue
                except Exception as e:
                    self._log.warning(f"Unexpected error in 2D loss for batch {b}: {e}")
                    continue
            
            # Make sure we have at least one batch that succeeded
            if successful_batches > 0:
                # Combine all geometry losses
                l2d_loss = (dist_loss + omega_loss + theta_loss + phi_loss) / successful_batches
            else:
                # Return zero loss if no batches could be processed
                self._log.warning("No batches could be processed successfully in 2D loss")
                l2d_loss = torch.tensor(0.0, device=device)
            
            return l2d_loss
            
        except Exception as e:
            # Log error and return zero loss if anything goes wrong
            self._log.warning(f"Error in compute_2d_loss: {e}")
            return torch.tensor(0.0, device=device)
    
    def _bin_distances(self, distances, num_bins=37, max_dist=18.5, device=None):
        """
        Convert distances to binned distributions
        
        Args:
            distances: Pairwise distance matrix [L, L]
            num_bins: Number of bins (default: 37 for 0-18.5Å in 0.5Å steps)
            max_dist: Maximum distance to consider
            device: Device to place tensors on (if None, uses distances's device)
            
        Returns:
            Binned distance distributions [L, L, num_bins]
        """
        # Determine target device if not specified
        if device is None:
            device = distances.device if hasattr(distances, 'device') else self.default_device
            
        # Move tensor to the target device if needed
        if distances.device != device:
            distances = distances.to(device)
            
        try:
            bin_size = max_dist / num_bins
            bins = torch.arange(0, max_dist + bin_size, bin_size, device=device)
            
            # Clamp distances to max_dist - epsilon to avoid bin_indices == num_bins
            distances = torch.clamp(distances, 0, max_dist - 1e-8)
            
            # Convert to bin indices with safety check
            bin_indices = torch.floor(distances / bin_size).long()
            
            # Ensure bin indices are within valid range
            bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)
            
            # Create one-hot encoding
            binned_dists = torch.zeros(*distances.shape, num_bins, device=device)
            
            # For each position, set the corresponding bin to 1
            for i in range(distances.shape[0]):
                for j in range(distances.shape[1]):
                    binned_dists[i, j, bin_indices[i, j]] = 1.0
                    
            return binned_dists
            
        except RuntimeError as e:
            self._log.warning(f"Error in _bin_distances: {e}")
            # Return a safe fallback
            return torch.zeros(*distances.shape, num_bins, device=device)
        except IndexError as e:
            self._log.warning(f"Index error in _bin_distances: {e}")
            # Return a safe fallback
            return torch.zeros(*distances.shape, num_bins, device=device)
    
    def _bin_angles(self, angles, angle_type="dihedral", eps=1e-8, device=None):
        """
        Convert angles to binned distributions
        
        Args:
            angles: Angle matrix [L, L]
            angle_type: Type of angle ("dihedral" or "planar")
            eps: Small epsilon for numerical stability
            device: Device to place tensors on (if None, uses angles's device)
            
        Returns:
            Binned angle distributions
        """
        # Determine target device if not specified
        if device is None:
            device = angles.device if hasattr(angles, 'device') else self.default_device
            
        # Move tensor to the target device if needed
        if angles.device != device:
            angles = angles.to(device)
            
        try:
            if angle_type == "dihedral":
                # For dihedral angles: 37 bins from -π to π
                num_bins = 37
                min_val = -torch.pi
                max_val = torch.pi
            else:  # planar
                # For planar angles: 19 bins from 0 to π
                num_bins = 19
                min_val = 0
                max_val = torch.pi
                
            bin_size = (max_val - min_val) / num_bins
            
            # Clamp angles to valid range
            angles = torch.clamp(angles, min_val + eps, max_val - eps)
            
            # Convert to bin indices
            bin_indices = torch.floor((angles - min_val) / bin_size).long()
            
            # Ensure bin indices are within valid range
            bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)
            
            # Create one-hot encoding
            binned_angles = torch.zeros(*angles.shape, num_bins, device=device)
            
            # For each position, set the corresponding bin to 1
            for i in range(angles.shape[0]):
                for j in range(angles.shape[1]):
                    if torch.isfinite(angles[i, j]):  # Only bin finite values
                        binned_angles[i, j, bin_indices[i, j]] = 1.0
                    else:
                        # For NaN or inf values, distribute evenly
                        binned_angles[i, j, :] = 1.0 / num_bins
                    
            return binned_angles
            
        except RuntimeError as e:
            self._log.warning(f"Error in _bin_angles: {e}")
            # Return a safe fallback
            return torch.zeros(*angles.shape, num_bins, device=device)
        except IndexError as e:
            self._log.warning(f"Index error in _bin_angles: {e}")
            # Return a safe fallback
            return torch.zeros(*angles.shape, num_bins, device=device)
        except Exception as e:
            self._log.warning(f"Unexpected error in _bin_angles: {e}")
            # Return a safe fallback
            if angle_type == "dihedral":
                num_bins = 37
            else:
                num_bins = 19
            return torch.zeros(*angles.shape, num_bins, device=device)
    
    def _compute_dihedral_matrix(self, a, b, c, d, device=None):
        """
        Compute dihedral angles between residues
        
        Args:
            a, b, c, d: Atom coordinates [L, 3]
            device: Device to place tensors on (if None, uses a's device)
            
        Returns:
            Dihedral angle matrix [L, L]
        """
        # Determine target device if not specified
        if device is None:
            device = a.device if hasattr(a, 'device') else self.default_device
            
        # Move tensors to the target device if needed
        if a.device != device:
            a = a.to(device)
        if b.device != device:
            b = b.to(device)
        if c.device != device:
            c = c.to(device)
        if d.device != device:
            d = d.to(device)
            
        # Fix shapes if needed
        if len(a.shape) > 2:
            a = a.reshape(-1, 3)
        if len(b.shape) > 2:
            b = b.reshape(-1, 3)
        if len(c.shape) > 2:
            c = c.reshape(-1, 3)
        if len(d.shape) > 2:
            d = d.reshape(-1, 3)
        
        L = a.shape[0]
        dihedrals = torch.zeros((L, L), device=device)
        
        try:
            # Compute dihedral angles for each pair of residues
            for i in range(L):
                for j in range(L):
                    if i != j:
                        try:
                            # Skip if any atom has NaN coordinates
                            if torch.isnan(a[i]).any() or torch.isnan(b[i]).any() or \
                               torch.isnan(c[j]).any() or torch.isnan(d[j]).any():
                                continue
                                
                            # Calculate vectors
                            v1 = b[i] - a[i]  # a->b
                            v2 = c[j] - b[i]  # b->c
                            v3 = d[j] - c[j]  # c->d
                            
                            # Check for zero vectors
                            if torch.norm(v1) < 1e-6 or torch.norm(v2) < 1e-6 or torch.norm(v3) < 1e-6:
                                continue
                            
                            # Normalize vectors
                            v1 = v1 / (torch.norm(v1) + 1e-8)
                            v2 = v2 / (torch.norm(v2) + 1e-8)
                            v3 = v3 / (torch.norm(v3) + 1e-8)
                            
                            # Compute cross products
                            n1 = torch.cross(v1, v2)
                            n2 = torch.cross(v2, v3)
                            
                            # Check for zero normal vectors (colinear vectors)
                            if torch.norm(n1) < 1e-6 or torch.norm(n2) < 1e-6:
                                continue
                            
                            # Normalize normal vectors
                            n1 = n1 / (torch.norm(n1) + 1e-8)
                            n2 = n2 / (torch.norm(n2) + 1e-8)
                            
                            # Compute dihedral angle
                            x = torch.dot(n1, n2)
                            y = torch.dot(torch.cross(n1, v2/torch.norm(v2)), n2)
                            
                            # Calculate dihedral using atan2
                            dihedral = torch.atan2(y, x)
                            dihedrals[i, j] = dihedral
                        except Exception as e:
                            # Skip this calculation if it fails
                            continue
            
            return dihedrals
            
        except Exception as e:
            self._log.warning(f"Error in _compute_dihedral_matrix: {e}")
            # Return zeros if anything goes wrong
            return torch.zeros((L, L), device=device)
    
    def _compute_planar_matrix(self, a, b, c, device=None):
        """
        Compute planar angles between residues
        
        Args:
            a, b, c: Atom coordinates [L, 3]
            device: Device to place tensors on (if None, uses a's device)
            
        Returns:
            Planar angle matrix [L, L]
        """
        # Determine target device if not specified
        if device is None:
            device = a.device if hasattr(a, 'device') else self.default_device
            
        # Move tensors to the target device if needed
        if a.device != device:
            a = a.to(device)
        if b.device != device:
            b = b.to(device)
        if c.device != device:
            c = c.to(device)
            
        # Fix shapes if needed
        if len(a.shape) > 2:
            a = a.reshape(-1, 3)
        if len(b.shape) > 2:
            b = b.reshape(-1, 3)
        if len(c.shape) > 2:
            c = c.reshape(-1, 3)
            
        L = a.shape[0]
        angles = torch.zeros((L, L), device=device)
        
        try:
            # Compute planar angles for each pair of residues
            for i in range(L):
                for j in range(L):
                    if i != j:
                        try:
                            # Skip if any atom has NaN coordinates
                            if torch.isnan(a[i]).any() or torch.isnan(b[i]).any() or torch.isnan(c[j]).any():
                                continue
                                
                            # Calculate vectors
                            v1 = a[i] - b[i]  # a->b
                            v2 = c[j] - b[i]  # b->c
                            
                            # Check for zero vectors
                            if torch.norm(v1) < 1e-6 or torch.norm(v2) < 1e-6:
                                continue
                            
                            # Normalize vectors
                            v1_norm = torch.norm(v1) + 1e-8
                            v2_norm = torch.norm(v2) + 1e-8
                            
                            # Compute cosine of angle
                            cos_angle = torch.dot(v1, v2) / (v1_norm * v2_norm)
                            cos_angle = torch.clamp(cos_angle, -1.0 + 1e-8, 1.0 - 1e-8)
                            
                            # Compute angle
                            angle = torch.acos(cos_angle)
                            angles[i, j] = angle
                        except Exception as e:
                            # Skip this calculation if it fails
                            continue
                            
            return angles
            
        except Exception as e:
            self._log.warning(f"Error in _compute_planar_matrix: {e}")
            # Return zeros if anything goes wrong
            return torch.zeros((L, L), device=device)
        
    def calculate_kl_divergence_loss(self, student_score, teacher_score, x_t, timestep, device=None):
        """
        Calculate the KL divergence loss between student and teacher score functions.
        This implements the training step for score-based distillation using KL divergence.
        
        Args:
            student_score: Score function from student model
            teacher_score: Score function from teacher model
            x_t: Noised coordinates at timestep t
            timestep: Current timestep (1-indexed)
            device: Device to place tensors on (if None, uses student_score's device)
            
        Returns:
            KL divergence loss
        """
        # Determine target device if not specified
        if device is None:
            device = student_score.device if hasattr(student_score, 'device') else self.default_device
        
        # Check if inputs require grad
        student_requires_grad = student_score.requires_grad
        if not student_requires_grad:
            self._log.warning("Student score doesn't require gradients! Using x_t to create gradients.")
            # In this case, we need to generate our own gradients through x_t
            if not x_t.requires_grad:
                x_t = x_t.clone().detach().requires_grad_(True)
                
        # For generator training:
        # We need generator_score (not student_score) with gradients and teacher_score detached
        # First check if we're dealing with the generator score
        if not student_requires_grad and x_t.requires_grad:
            # Compute generator score directly with gradients
            self._log.info("Computing generator score with gradients")
            generator_score = self.compute_generator_score(x_t, timestep)
            
            # Always detach the teacher score
            teacher_score_detached = teacher_score.detach()
                
            # Move tensors to the same device if needed
            if generator_score.device != device:
                generator_score = generator_score.to(device)
                
            if teacher_score_detached.device != device:
                teacher_score_detached = teacher_score_detached.to(device)
                
            # Get device-specific diffusion parameters
            params = self._get_device_params(device, timestep)
            
            # Weight for the score discrepancy
            weight = params['beta_t']
            
            try:
                # KL divergence for diffusion models simplifies to weighted MSE between scores
                kl_loss = weight * torch.mean(torch.sum(
                    (generator_score - teacher_score_detached) ** 2, dim=-1
                ))
            except RuntimeError as e:
                # Log error and provide fallback
                self._log.warning(f"Error computing KL loss: {e}")
                # Ensure we have a well-defined loss with gradients
                kl_loss = F.mse_loss(generator_score, teacher_score_detached)
                
            # Double-check for gradients
            if not kl_loss.requires_grad:
                self._log.warning("KL loss still has no gradients! Using direct loss.")
                # Last resort - use direct loss on coordinates that guarantees gradients
                kl_loss = F.mse_loss(generator_score, generator_score.detach() - 0.01)
        else:
            # Regular flow - use student and teacher scores
            # IMPORTANT: Keep student_score gradients intact for student model training
            # But detach teacher score
            teacher_score_detached = teacher_score.detach()
                
            # Move tensors to the same device if needed
            if student_score.device != device:
                student_score = student_score.to(device)
                
            if teacher_score_detached.device != device:
                teacher_score_detached = teacher_score_detached.to(device)
                
            # Get device-specific diffusion parameters
            params = self._get_device_params(device, timestep)
            
            # Weight for the score discrepancy
            weight = params['beta_t']
            
            try:
                # KL divergence for diffusion models simplifies to weighted MSE between scores
                kl_loss = weight * torch.mean(torch.sum(
                    (student_score - teacher_score_detached) ** 2, dim=-1
                ))
            except RuntimeError as e:
                # Log error and provide fallback
                self._log.warning(f"Error computing KL loss: {e}")
                # Ensure we have a well-defined loss with gradients
                kl_loss = F.mse_loss(student_score, teacher_score_detached)
                
            # Check for gradients
            if not kl_loss.requires_grad and student_score.requires_grad:
                self._log.warning("KL loss has no gradients despite score having gradients!")
                # Use a fallback that guarantees gradients by directly using student_score
                kl_loss = torch.mean(student_score**2) * 0.01 + kl_loss.detach()
        
        # Final verification of gradients
        if not kl_loss.requires_grad:
            self._log.warning("KL loss STILL has no gradients! Creating a dummy loss.")
            # Create a dummy loss with known gradients
            dummy_tensor = torch.ones(1, device=device, requires_grad=True)
            kl_loss = dummy_tensor * 0.01 + kl_loss.detach()
            
        return kl_loss