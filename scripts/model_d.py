import os
import glob
import random
import numpy as np
import pandas as pd
import torch
from torch import optim
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.distributions.normal import StandardNormal
from sklearn.preprocessing import StandardScaler
import sklearn

# fixed random seed (42) for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cpu")

num_flows = 7
hidden_features = 128
condition_dim = 2
num_epochs = 30
batch_size = 128
GOOD_EPOCH = 11

# The conditioning variables are the positively directed pulling force
# recorded for one of the terminal atoms and the corresponding molecular extension
FORCE_COL = "pulling_force_x_terminal_atom_positive"
EXT_COL   = "molecular_extension"

# Directory paths
npy_dir = "/path/to/trajectories"
csv_dir = "/path/to/logs"
base_save_path = "/path/to/project"
save_path = os.path.join(base_save_path, "model_d")
os.makedirs(save_path, exist_ok=True)

# The CV-SMD with pulling velocity of 7.64 Å/ns was reserved for validation
val_sim_name = "7p64e-6"


# Load and align NumPy trajectory files and CSV log files
# Trajectories were truncated at 18.2 ns (or 17.5 ns for the 7.70 Å/ns test case) to exclude overstretched states
# CSV logs (sampled every 5,000 steps) were downsampled to match trajectory sampling (every 100,000 steps)
X_train_list, cond_train_list = [], []
X_val_list, cond_val_list = [], []

for npy_file in sorted(glob.glob(os.path.join(npy_dir, "*.npy"))):
    sim_name = os.path.basename(npy_file).replace(".npy", "")
    csv_file = os.path.join(csv_dir, sim_name + ".csv")
    if not os.path.exists(csv_file):
        continue

    # Load trajectory data, originally stored as a 3D array of shape (num_atoms, timesteps, 3)
    # where the last dimension corresponds to Cartesian coordinates (x, y, z)
    raw = np.load(npy_file)
    coords = raw[0] if raw.ndim == 4 else raw
    n_atoms, T = coords.shape[0], coords.shape[1]

    # Each trajectory is reshaped to (timesteps, num_atoms × 3),
    # where each row corresponds to one molecular conformation used as a training sample
    traj = coords.transpose(1, 0, 2).reshape(T, -1).astype(np.float32)

    # keep only timesteps present in the trajectory after truncation
    df = pd.read_csv(csv_file)
    if ("iter" not in df.columns) or (FORCE_COL not in df.columns) or (EXT_COL not in df.columns):
        continue

    # Truncate at 18.2 ns equivalent and keep rows every 100,000 steps
    df = df[df["iter"] <= 18_200_000]
    df = df[df["iter"] % 100_000 == 0]

    forces = df[FORCE_COL].values[:T].astype(np.float32)
    exts = df[EXT_COL].values[:T].astype(np.float32)
    if (len(forces) != T) or (len(exts) != T):
        continue

    cond = np.stack([forces, exts], axis=1)  

    # Split into training and validation according to simulation name
    if sim_name == val_sim_name:
        X_val_list.append(traj)
        cond_val_list.append(cond)
    else:
        X_train_list.append(traj)
        cond_train_list.append(cond)

if not X_train_list or not cond_train_list:
    raise RuntimeError("No training data assembled.")
if not X_val_list or not cond_val_list:
    X_val_list = [np.empty((0, X_train_list[0].shape[1]), dtype=np.float32)]
    cond_val_list = [np.empty((0, 2), dtype=np.float32)]


X_train = np.vstack(X_train_list)
conditions_train = np.vstack(cond_train_list)
X_val = np.vstack(X_val_list)
conditions_val = np.vstack(cond_val_list)

# StandardScaler is fitted on the training data and applied to all splits
# to ensure zero mean and unit variance for all input and conditioning features
X_scaler = StandardScaler().fit(X_train)
cond_scaler = StandardScaler().fit(conditions_train)

X_train_tensor = torch.tensor(X_scaler.transform(X_train), dtype=torch.float32, device=device)
cond_train_tensor = torch.tensor(cond_scaler.transform(conditions_train), dtype=torch.float32, device=device)
X_val_tensor = torch.tensor(X_scaler.transform(X_val), dtype=torch.float32, device=device)
cond_val_tensor = torch.tensor(cond_scaler.transform(conditions_val), dtype=torch.float32, device=device)


# Conditional normalising flow (CNF) implemented using the nflows library
# Each flow step consists of a ReversePermutation followed by a MaskedAffineAutoregressiveTransform
# with 128 hidden features and 2 conditioning inputs. The base distribution is multivariate standard normal
def create_conditional_flow(data_dim, condition_dim, num_flows, hidden_features):
    transforms = []
    for _ in range(num_flows):
        transforms.append(ReversePermutation(features=data_dim))
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=data_dim,
                hidden_features=hidden_features,
                context_features=condition_dim,
            )
        )
    transform = CompositeTransform(transforms)
    base_distribution = StandardNormal(shape=[data_dim])
    return Flow(transform, base_distribution)

data_dim = X_train_tensor.shape[1]
flow = create_conditional_flow(data_dim, condition_dim, num_flows, hidden_features).to(device)
optimizer = optim.Adam(flow.parameters(), lr=1e-3)


# Training minimises the negative log-likelihood using Adam (lr=1e-3)
# A batch size of 128 is used and data are reshuffled each epoch
# The model is trained for up to 30 epochs
loss_log = []
saved_flag = False
N_train = X_train_tensor.shape[0]
N_val = X_val_tensor.shape[0]

for epoch in range(1, num_epochs + 1):
    flow.train()
    perm = torch.randperm(N_train)
    X_shuffled = X_train_tensor[perm]
    cond_shuffled = cond_train_tensor[perm]

    train_loss_sum, seen_train = 0.0, 0

    for i in range(0, N_train, batch_size):
        x_batch = X_shuffled[i:i + batch_size]
        c_batch = cond_shuffled[i:i + batch_size]
        bsz = x_batch.shape[0]

        loss = -flow.log_prob(inputs=x_batch, context=c_batch).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * bsz
        seen_train += bsz

    mean_train_loss = train_loss_sum / max(1, seen_train)

    # Validation loss for monitoring convergence
    flow.eval()
    val_loss_sum, seen_val = 0.0, 0
    with torch.no_grad():
        for j in range(0, N_val, batch_size):
            x_val_b = X_val_tensor[j:j + batch_size]
            c_val_b = cond_val_tensor[j:j + batch_size]
            bsz = x_val_b.shape[0]
            if bsz == 0:
                continue
            vloss = -flow.log_prob(inputs=x_val_b, context=c_val_b).mean().item()
            val_loss_sum += vloss * bsz
            seen_val += bsz

    mean_val_loss = val_loss_sum / seen_val if seen_val > 0 else float("nan")
    loss_log.append((epoch, float(mean_train_loss), float(mean_val_loss)))

    # Save model and scalers at the chosen epoch
    # Save architecture parameters and library versions for reproducibility
    if epoch == GOOD_EPOCH and not saved_flag:
        try:
            import nflows as _nflows
            nflows_ver = getattr(_nflows, "__version__", "unknown")
        except Exception:
            nflows_ver = "unknown"

        ckpt = {
            "model_state_dict": flow.state_dict(),
            "X_scaler": X_scaler,
            "cond_scaler": cond_scaler,
            "epoch": epoch,
            "data_dim": data_dim,
            "condition_dim": condition_dim,
            "num_flows": num_flows,
            "hidden_features": hidden_features,
            "library_versions": {
                "torch": torch.__version__,
                "nflows": nflows_ver,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sklearn": sklearn.__version__,
            },
            "cond_cols": [FORCE_COL, EXT_COL],
        }

        model_path = os.path.join(save_path, f"model_epoch{epoch}.pt")
        torch.save(ckpt, model_path)
        saved_flag = True

# Training and validation losses are saved as a CSV file
loss_log_path = os.path.join(save_path, "training_losslog_30epochs.csv")
pd.DataFrame(loss_log, columns=["epoch", "train_loss", "val_loss"]).to_csv(loss_log_path, index=False)
