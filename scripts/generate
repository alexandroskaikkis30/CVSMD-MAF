import os
import numpy as np
import pandas as pd
import torch

from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.distributions.normal import StandardNormal

# using the trained CNF model to generate new molecular conformations
# and compare them to the CV-SMD run at 7.70 Å/ns

# file paths (change to your own)
ckpt_path = "/path/to/checkpoint/model"
csv_path = "/path/to/test/logs"
npy_path = "/path/to/test/trajectories"
out_dir = "/path/to/directory/outputs"
os.makedirs(out_dir, exist_ok=True)

# save each generated frame separately if needed
SAVE_EACH_FRAME = False
if SAVE_EACH_FRAME:
    frame_dir = os.path.join(out_dir, "best_frames")
    os.makedirs(frame_dir, exist_ok=True)

ITER_MAX = 17_500_000     # total steps (17.5 ns)
ITER_STEP = 100_000       # trajectory saved every 100k steps
MAX_FRAMES = 175          # number of frames to use
N_SAMPLES_PER_FRAME = 1   # how many samples per frame to generate

# thresholds for invalid data
BAD_RMSD_THRESHOLD = np.float32(50.0)   # skip if RMSD > 50 Å
MAX_ABS_COORD_ALLOWED = np.float32(1e6) # skip if |coord| > 1e6 Å
USE_NAN_FOR_FAILED = True                # if failed, fill with NaN

# random seed so results are reproducible
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cpu")
np.seterr(all='warn')


# build the same model used during training
def create_conditional_flow(data_dim, condition_dim, num_flows, hidden_features):
    layers = []
    for _ in range(num_flows):
        layers.append(ReversePermutation(features=data_dim))
        layers.append(
            MaskedAffineAutoregressiveTransform(
                features=data_dim,
                hidden_features=hidden_features,
                context_features=condition_dim,
            )
        )
    transform = CompositeTransform(layers)
    base = StandardNormal(shape=[data_dim])
    return Flow(transform, base)


def load_coords_T_atoms3(npy_file):
    # load trajectory (T, n_atoms, 3)
    raw = np.load(npy_file)
    if raw.ndim == 4:
        raw = raw[0]
    if raw.ndim != 3:
        raise ValueError(f"unexpected .npy shape {raw.shape}")
    return np.transpose(raw, (1, 0, 2)).astype(np.float32, copy=False)


def kabsch_rmsd_safe(P, Q):
    # compute RMSD after alignment (Kabsch)
    P = np.asarray(P, dtype=np.float32)
    Q = np.asarray(Q, dtype=np.float32)
    if not (np.all(np.isfinite(P)) and np.all(np.isfinite(Q))):
        return (np.float32(np.nan), "nonfinite")

    # center both conformations
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    C = Pc.T @ Qc

    try:
        V, S, Wt = np.linalg.svd(C, full_matrices=False)
    except np.linalg.LinAlgError:
        rmsd_fb = np.sqrt(np.mean((Pc - Qc) ** 2)).astype(np.float32)
        return (rmsd_fb, "svd_fail")

    if np.linalg.det(V @ Wt) < 0:
        V[:, -1] *= -1

    R = V @ Wt
    P_rot = Pc @ R
    rmsd = np.sqrt(np.mean((P_rot - Qc) ** 2)).astype(np.float32)
    return (rmsd, None)


# load checkpoint and rebuild model
ckpt = torch.load(ckpt_path, map_location="cpu")
if "X_scaler" not in ckpt or "cond_scaler" not in ckpt:
    raise RuntimeError("checkpoint missing scalers")

X_scaler = ckpt["X_scaler"]
cond_scaler = ckpt["cond_scaler"]
data_dim = ckpt.get("data_dim", X_scaler.mean_.shape[0])
condition_dim = ckpt.get("condition_dim", 1)
num_flows = ckpt.get("num_flows", 5)
hidden_features = ckpt.get("hidden_features", 128)

flow = create_conditional_flow(data_dim, condition_dim, num_flows, hidden_features)
flow.load_state_dict(ckpt["model_state_dict"])
flow.eval().to(device)

# load the test simulation (7.70 Å/ns) preprocessed same as training data
coords_true = load_coords_T_atoms3(npy_path)
df = pd.read_csv(csv_path)
df = df[(df["iter"] <= ITER_MAX) & (df["iter"] % ITER_STEP == 0)].copy()
df.sort_values("iter", inplace=True)

# conditioning variables depend on the model
if condition_dim == 1:
    cond_cols = ["pulling_force_x_terminal_atom_positive"]
elif condition_dim == 2:
    cond_cols = ["pulling_force_x_terminal_atom_positive", "molecular_extension"]
else:
    raise RuntimeError(f"unsupported condition_dim={condition_dim}")

# get conditions and make sure everything lines up
conditions = df[cond_cols].to_numpy(dtype=np.float32, copy=False)
T_csv = conditions.shape[0]
T_npy = coords_true.shape[0]
T = min(T_csv, T_npy, MAX_FRAMES)

conditions = conditions[:T]
coords_true = coords_true[:T]
iters = df["iter"].values[:T]
forces_logged = df["pulling_force_x_terminal_atom_positive"].values[:T].astype(np.float32)

# make sure input size matches model setup
expected_data_dim = coords_true.shape[1] * 3
if data_dim != expected_data_dim:
    raise RuntimeError("data_dim doesn't match the number of atoms")

# scale conditioning vars same as training
generated_best = np.zeros((T, coords_true.shape[1], 3), dtype=np.float32)
rmsd_best_list, rmsd_mean_list, rmsd_std_list, bad_counts = [], [], [], []
all_samples = []
cond_scaled = cond_scaler.transform(conditions).astype(np.float32)

# sample from standard Gaussian (tau=1) and transform with trained flow
with torch.no_grad():
    for i in range(T):
        context_i = torch.tensor(cond_scaled[i:i+1], dtype=torch.float32, device=device)
        good_rmsds, good_coords = [], []
        bad = 0

        for k in range(N_SAMPLES_PER_FRAME):
            samp = flow.sample(1, context=context_i)
            samp = samp.cpu().numpy().reshape(1, -1).astype(np.float32)
            samp_real = X_scaler.inverse_transform(samp).astype(np.float32)

            # reshape to (n_atoms, 3)
            n_atoms = samp_real.shape[1] // 3
            coords_pred_k = samp_real.reshape(n_atoms, 3)

            # skip invalid coordinates
            if (not np.all(np.isfinite(coords_pred_k))) or (np.max(np.abs(coords_pred_k)) > MAX_ABS_COORD_ALLOWED):
                bad += 1
                continue

            # compute RMSD after Kabsch alignment, discard if too large
            r, status = kabsch_rmsd_safe(coords_pred_k, coords_true[i])
            if (not np.isfinite(r)) or (r > BAD_RMSD_THRESHOLD):
                bad += 1
                continue

            # store valid sample
            all_samples.append({
                "frame_idx": int(i),
                "iter": int(iters[i]),
                "force": float(forces_logged[i]),
                "sample_idx": int(k),
                "rmsd": float(r),
            })
            good_rmsds.append(np.float32(r))
            good_coords.append(coords_pred_k.astype(np.float32))

        # keep lowest RMSD one per frame
        if len(good_rmsds) == 0:
            best_rmsd = np.float32(np.nan)
            mean_r = np.float32(np.nan)
            std_r = np.float32(np.nan)
            best_coords = np.full_like(coords_true[i], np.nan, dtype=np.float32) if USE_NAN_FOR_FAILED else coords_true[i]
        else:
            arr = np.array(good_rmsds, dtype=np.float32)
            best_idx = int(np.argmin(arr))
            best_rmsd = np.float32(arr[best_idx])
            mean_r = np.mean(arr, dtype=np.float32)
            std_r = np.std(arr, dtype=np.float32)
            best_coords = good_coords[best_idx]

        generated_best[i] = best_coords
        rmsd_best_list.append(best_rmsd)
        rmsd_mean_list.append(mean_r)
        rmsd_std_list.append(std_r)
        bad_counts.append(int(bad))

        if SAVE_EACH_FRAME:
            np.save(os.path.join(frame_dir, f"frame_{i:04d}.npy"), generated_best[i])

# save results
all_samples_df = pd.DataFrame(all_samples)
all_samples_df.to_csv(os.path.join(out_dir, "all_rmsd_samples.csv"), index=False)

np.save(os.path.join(out_dir, "generated_best.npy"), generated_best.astype(np.float32))

rmsd_best_arr = np.array(rmsd_best_list, dtype=np.float32)
rmsd_mean_arr = np.array(rmsd_mean_list, dtype=np.float32)
rmsd_std_arr = np.array(rmsd_std_list, dtype=np.float32)
valid_counts = (np.array([N_SAMPLES_PER_FRAME] * T) - np.array(bad_counts, dtype=int))
reject_rates = np.where(N_SAMPLES_PER_FRAME > 0,
                        np.array(bad_counts, dtype=np.float32) / N_SAMPLES_PER_FRAME,
                        np.nan)

rmsd_df = pd.DataFrame({
    "frame_idx": np.arange(T, dtype=int),
    "iter": iters,
    "force": forces_logged,
    "rmsd_best": rmsd_best_arr,
    "rmsd_mean": rmsd_mean_arr,
    "rmsd_std": rmsd_std_arr,
    "valid_count": valid_counts,
    "bad_count": bad_counts,
    "reject_rate": reject_rates.astype(np.float32),
    "K": [N_SAMPLES_PER_FRAME] * T,
    "bad_rmsd_threshold_A": [float(BAD_RMSD_THRESHOLD)] * T,
})
rmsd_df.to_csv(os.path.join(out_dir, "rmsd_stats.csv"), index=False)
