# -----------------------------------------------------------------------------
# This script is a continuation and modification of an example contributed by
# Saurav Maheshkar to the TorchMD framework repository.
#
# Original framework reference:
# @misc{doerr2020torchmd,
#   title        = {TorchMD: A deep learning framework for molecular simulations},
#   author       = {Stefan Doerr and Maciej Majewski and Adrià Pérez and
#                   Andreas Krämer and Cecilia Clementi and Frank Noé and
#                   Toni Giorgino and Gianni De Fabritiis},
#   year         = {2020},
#   eprint       = {2012.12106},
#   archivePrefix= {arXiv},
#   primaryClass = {physics.chem-ph}
# }
# -----------------------------------------------------------------------------

import os
import numpy as np
from tqdm import tqdm
import torch
from moleculekit.molecule import Molecule
from torchmd.forcefields.forcefield import ForceField
from torchmd.parameters import Parameters
from torchmd.systems import System
from torchmd.forces import Forces
from torchmd.integrator import Integrator, maxwell_boltzmann
from torchmd.utils import LogWriter

os.environ['NUMEXPR_MAX_THREADS'] = '16'

# Directory and device setup
testdir = "/path/to/project"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
precision = torch.float
FS2NS = 1E-6

# Load the molecule
mol = Molecule(os.path.join(testdir, "villin15.prmtop"))
mol.read(os.path.join(testdir, "villin15_output.pdb"))
print(f"Molecule loaded with {mol.numAtoms} atoms.")

# Set up the system with no periodic boundary conditions (PBC)
ff = ForceField.create(mol, os.path.join(testdir, "villin15.prmtop"))
parameters = Parameters(ff, mol, precision=precision, device=device)
system = System(mol.numAtoms, 1, precision, device)
system.set_positions(mol.coords)
system.set_velocities(maxwell_boltzmann(parameters.masses, T=330, replicas=1))
system.pos = torch.tensor(mol.coords).permute(2, 0, 1).type(precision)
print("System initialised. Starting positions and velocities are set.")

# Pulling force setup
class PullingForce:
    def __init__(self, k, terminal_atom_indices, initial_reference_x_positions, positive_velocity, negative_velocity):
        self.k = k  
        self.terminal_atom_indices = terminal_atom_indices
        self.reference_x_positions = torch.tensor(initial_reference_x_positions, device=device)
        self.positive_velocity = positive_velocity   
        self.negative_velocity = negative_velocity  
        self.last_forces = torch.zeros_like(system.pos)
        self.dt_fs = 1.0  # timestep in femtoseconds

    def calculate(self, pos, box=None):
        forces = torch.zeros_like(pos)
        energy = torch.zeros(1)
# Calculation of current reference positions
        self.reference_x_positions[0] += self.positive_velocity * self.dt_fs   
        self.reference_x_positions[1] += self.negative_velocity * self.dt_fs   

# Pulling is applied only along the x-axis, and only x-coordinates are used for reference and displacement
        for i, target_x in zip(self.terminal_atom_indices, self.reference_x_positions):
            displacement_x = pos[0][i][0] - target_x
            forces[0][i][0] = -self.k * displacement_x
            # Harmonic restraint energy term 
            energy += 0.5 * self.k * displacement_x ** 2

        self.last_forces = forces
        return energy, forces[0]

# Initialise reference points
terminal_atom_indices = [0, 1026]
initial_reference_x_positions = [
    system.pos[0][terminal_atom_indices[0]][0].item(),
    system.pos[0][terminal_atom_indices[1]][0].item()
]

pulling_force = PullingForce(
    k=0.4,  # Force constant
    terminal_atom_indices=terminal_atom_indices,
    initial_reference_x_positions=initial_reference_x_positions,
    # Velocity is given in Å/fs (Angstroms per femtosecond)
    positive_velocity=7.65e-6,
    negative_velocity=-7.65e-6
)

# Potential energy includes bonded, non-bonded, and a harmonic restraint,
# with a switching function at 7.5 Å and a 9 Å cutoff.
forces = Forces(
    parameters,
    cutoff=9.0,
    rfa=False,
    switch_dist=7.5,
    terms=["bonds", "angles", "dihedrals", "impropers", "1-4", "electrostatics", "lj"],
    external=pulling_force
)

# Integrator setup including friction (gamma) and temperature (T) parameters
integrator = Integrator(system, forces, timestep=1, device=device, gamma=1, T=330)  # T in K, timestep in fs
trajectory_out = "7p65e-6.npy"

logger = LogWriter(
    path="logs/",
    keys=(
        'iter', 'ns', 'epot', 'ekin', 'etot', 'T',
        'pulling_force_x_terminal_atom_positive',
        'pulling_force_x_terminal_atom_negative',
        'molecular_extension'
    ),
    name='7p65e-6.csv'
)

# Simulation loop
print("Starting simulation...")
traj = []
steps = 20000000  # Simulation runs for 20 million steps
logging_period = 5000  # Log data every 5,000 steps
trajectory_save_period = 100000  # Save trajectory every 100,000 steps to avoid saving highly correlated conformations

# Initialise simulation step count
current_step = 0

# Start the simulation
iterator = tqdm(range(steps))
for _ in iterator:
    # Perform one simulation step
    Ekin, Epot, T = integrator.step(niter=1)
    current_step += 1

    # Save the trajectory periodically
    if current_step % trajectory_save_period == 0:
        currpos = system.pos.detach().cpu().numpy().copy()
        traj.append(currpos)
        np.save(trajectory_out, np.stack(traj, axis=2))

    # Log simulation data
    if current_step % logging_period == 0:
        i_pos = pulling_force.terminal_atom_indices[0]
        i_neg = pulling_force.terminal_atom_indices[1]
        molecular_extension = system.pos[0][i_pos][0] - system.pos[0][i_neg][0]

        # Save current data to log file
        logger.write_row({
            'iter': current_step,  # Simulation step
            'ns': FS2NS * current_step * 1,  # Simulation time (ns)
            'epot': Epot,  # Potential energy
            'ekin': Ekin,  # Kinetic energy
            'etot': Epot + Ekin,  # Total energy
            'T': T,  # Instantaneous temperature
            'pulling_force_x_terminal_atom_positive': pulling_force.last_forces[0][i_pos][0].item(),
            'pulling_force_x_terminal_atom_negative': pulling_force.last_forces[0][i_neg][0].item(),
            'molecular_extension': molecular_extension.item()
        })

print("Simulation complete.")
