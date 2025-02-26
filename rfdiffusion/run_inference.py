#!/usr/bin/env python
"""
Inference script with data saving for consistency model training.
"""

import re
import os
import time
import pickle
import torch
from omegaconf import OmegaConf
import logging
from rfdiffusion.util import writepdb_multi, writepdb
from rfdiffusion.inference import utils as iu
import numpy as np
import random
import glob

def make_deterministic(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def run_inference(
    config_file="../config.yml",  # Path to the YAML configuration file
    save_data=True,   # Option to save intermediate states
    save_dir='diffusion_data'  # Directory to save data if enabled
):
    log = logging.getLogger(__name__)
    
    # Load configuration
    conf = OmegaConf.load(config_file)

    # Extract relevant configuration sections
    inf_conf = conf.rfdiffusion.inference
    num_designs = inf_conf.num_designs
    design_startnum = inf_conf.design_startnum
    cautious = inf_conf.cautious
    write_trajectory = inf_conf.write_trajectory
    deterministic = inf_conf.deterministic

    if deterministic:
        make_deterministic()

    # Check for available GPU and print result of check
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        log.info(f"Found GPU with device_name {device_name}. Will run RFdiffusion on {device_name}")
    else:
        log.info("////////////////////////////////////////////////")
        log.info("///// NO GPU DETECTED! Falling back to CPU /////")
        log.info("////////////////////////////////////////////////")

    # Initialize sampler and target/contig.
    sampler = iu.sampler_selector(conf)

    # Loop over number of designs to sample.
    if design_startnum == -1:
        existing = glob.glob(sampler.inf_conf.output_prefix + "*.pdb")
        indices = [-1]
        for e in existing:
            m = re.match(".*_(\d+)\.pdb$", e)
            if not m:
                continue
            m = m.groups()[0]
            indices.append(int(m))
        design_startnum = max(indices) + 1

    # Create directory to save data
    if save_data:
        os.makedirs(save_dir, exist_ok=True)

    for i_des in range(design_startnum, design_startnum + num_designs):
        if deterministic:
            make_deterministic(i_des)

        start_time = time.time()
        out_prefix = f"{sampler.inf_conf.output_prefix}_{i_des}"
        log.info(f"Making design {out_prefix}")
        if cautious and os.path.exists(out_prefix + ".pdb"):
            log.info(f"(cautious mode) Skipping this design because {out_prefix}.pdb already exists.")
            continue

        x_init, seq_init = sampler.sample_init()
        denoised_xyz_stack = []
        px0_xyz_stack = []
        seq_stack = []
        plddt_stack = []

        x_t = torch.clone(x_init)
        seq_t = torch.clone(seq_init)
        
        # Loop over number of reverse diffusion time steps.
        for t in range(int(sampler.t_step_input), sampler.inf_conf.final_step - 1, -1):
            px0, x_t, seq_t, plddt = sampler.sample_step(
                t=t, x_t=x_t, seq_init=seq_t, final_step=sampler.inf_conf.final_step
            )
            px0_xyz_stack.append(px0)
            denoised_xyz_stack.append(x_t)
            seq_stack.append(seq_t)
            plddt_stack.append(plddt[0])  # remove singleton leading dimension

            # Save intermediate noisy (x_t) and denoised (px0) states
            if save_data:
                torch.save(x_t, f"{save_dir}/intermediate_state_design_{i_des}_t_{t}.pt")
                torch.save(px0, f"{save_dir}/denoised_state_design_{i_des}_t_{t}.pt")

        # Flip order for better visualization in pymol
        denoised_xyz_stack = torch.stack(denoised_xyz_stack)
        denoised_xyz_stack = torch.flip(denoised_xyz_stack, [0])
        px0_xyz_stack = torch.stack(px0_xyz_stack)
        px0_xyz_stack = torch.flip(px0_xyz_stack, [0])

        # Save final outputs (optional)
        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
        final_seq = seq_stack[-1]

        # Output glycines, except for motif region
        final_seq = torch.where(
            torch.argmax(seq_init, dim=-1) == 21, 7, torch.argmax(seq_init, dim=-1)
        )  # 7 is glycine

        bfacts = torch.ones_like(final_seq.squeeze())
        # make bfact=0 for diffused coordinates
        bfacts[torch.where(torch.argmax(seq_init, dim=-1) == 21, True, False)] = 0

        # pX0 last step
        out = f"{out_prefix}.pdb"

        # Now don't output sidechains
        writepdb(
            out,
            denoised_xyz_stack[0, :, :4],
            final_seq,
            sampler.binderlen,
            chain_idx=sampler.chain_idx,
            bfacts=bfacts,
        )

        # run metadata
        trb = dict(
            config=OmegaConf.to_container(sampler._conf, resolve=True),
            plddt=torch.stack(plddt_stack).cpu().numpy(),
            device=torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "CPU",
            time=time.time() - start_time,
        )
        if hasattr(sampler, "contig_map"):
            for key, value in sampler.contig_map.get_mappings().items():
                trb[key] = value
        with open(f"{out_prefix}.trb", "wb") as f_out:
            pickle.dump(trb, f_out)

        log.info(f"Finished design in {(time.time()-start_time)/60:.2f} minutes")