# CV-SMD/MAF
This project aims to enhance the sampling of non-equilibrium protein unfolding by performing steered molecular dynamics (SMD) simulations using the constant-velocity method, and by implementing a conditional masked autoregressive flow model for the generation of new samples.

# Overview
This study explores the non-equilibrium mecahnical unfolding of villin headpiece (PDB: 2RJY) by employing steered molecular dynamics simulations using the constant-velocity pulling method, through the TorchMD framework. A pulling protocol similar to that described by Bustamante et al. (2020) in Single-Molecule Studies of Protein Folding with Optical Tweezers was implemented. In this CV-SMD configuration, the pulling mechanism is defined by two reference points that move in opposite directions along the x-axis at constant velocity. Each of the two terminal atoms is harmonically restrained to one of these moving reference points, generating time-dependent forces that drive the unfolding process. This configuration is designed to mimic the physical conditions of Bustamante’s optical tweezers experiment, which is performed in a thermal bath allowing continuous energy exchange between the system and its environment. In TorchMD, temperature control required for the canonical (NVT) ensemble is achieved through its built-in integrator class, and a Langevin integrator was employed to maintain constant temperature throughout the simulation.

Several constant-velocity steered molecular dynamics (CV-SMD) simulations were performed at different pulling velocities. Analysis of the resulting force–extension profiles revealed noise associated with the sampling of similar conformations throughout the trajectories, while fewer samples were observed during unfolding events. These unfolding events correspond primarily to the disruption of secondary structure elements (α-helices) within the villin headpiece. 
<img width="1211" height="598" alt="image" src="https://github.com/user-attachments/assets/fa0a26e2-b5d1-48bc-83fd-2fcb5f349c0a" />

To better capture and generate rare conformational states, different architectures of conditional normalising flow models were employed, specifically variants of the Conditional Masked Autoregressive Flow (MAF) trained on trajectory data conditioned on molecular extension and the restoring forces acting on the terminal atoms. Four models were implemented: a 5-layer MAF conditioned on force (Model A), a 5-layer MAF conditioned on both force and molecular extension (Model B), a 6-layer MAF conditioned on both features (Model C), and a 7-layer MAF conditioned on both features (Model D).

After evaluating the models based on validation and training loss, Model A, which was conditioned on force only, achieved the best validation loss. During generation, it displayed greater stochasticity, producing a wider range of molecular extensions at comparable force values, including higher extensions at lower forces, a pattern that was not present in the training data. In contrast, Model D, which incorporated additional layers and conditioning features, showed the worst validation loss but generated samples in a more controlled mannerand with higher structural accuracy compared to the other models. These results suggest that conditioning on fewer features allows greater generative diversity, while additional conditioning constrains variability, leads to more controlled generation, and may increase the risk of overfitting given the limited size of the available dataset.

# Results 

<img width="825" height="507" alt="image" src="https://github.com/user-attachments/assets/6942fad3-f8ab-45a2-bfd0-a2c351a3ba8f" />







<img width="1062" height="733" alt="image" src="https://github.com/user-attachments/assets/6fdd88b8-0084-43a8-8cb7-e346a7198d4e" />


 
## Repository Structure
.
├──            
├── 

## Requirements
The project was developed and tested in a **conda environment (Miniconda3)** with **Python 3.7** on a Linux-based HPC cluster.

Main dependencies:
- torch==1.13.1  
- torchmd==1.0.2  
- moleculekit==1.3.4  
- numpy==1.21.5  
- tqdm==4.66.5
- nflows == 0.14
- pandas==1.3.5

All packages can be installed using the provided `requirements.txt` within a conda environment.
