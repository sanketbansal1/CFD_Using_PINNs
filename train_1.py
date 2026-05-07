import deepxde as dde
import numpy as np
import torch
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
DATA_DIR     = "VTK/PINN_Data"
OUTPUT_DIR   = "results"
EPOCHS_ADAM  = 50_000
LR           = 5e-4

NUM_DOMAIN   = 5000
NUM_TEST     = 500
BATCH_SIZE   = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ═══════════════════════════════════════════════════════════════════
# HELPER: safe way to read NU regardless of DeepXDE version
# ═══════════════════════════════════════════════════════════════════
def read_nu(nu_var):
    """Works with both old dde.Variable and new torch.Tensor style."""
    try:
        return float(nu_var.item())
    except AttributeError:
        try:
            return float(nu_var.value())
        except AttributeError:
            return float(nu_var)

# ═══════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════
def load(name, max_pts=1000):
    c = np.load(os.path.join(DATA_DIR, f"{name}_coords.npy")).astype(np.float32)
    U = np.load(os.path.join(DATA_DIR, f"{name}_U.npy")).astype(np.float32)
    p = np.load(os.path.join(DATA_DIR, f"{name}_p.npy")).astype(np.float32).reshape(-1, 1)
    c = c[:, 0:2]  # x,y only — domain is quasi-2D (z: 0→0.01)
    if len(c) > max_pts:
        idx = np.random.choice(len(c), max_pts, replace=False)
        c, U, p = c[idx], U[idx], p[idx]
    print(f"  {name}: {len(c)} points")
    return c, U, p

print("Loading data...")
inlet_c,    inlet_U,    inlet_p    = load("inlet",           max_pts=500)
outlet_c,   outlet_U,   outlet_p   = load("outlet",          max_pts=500)
cylinder_c, cylinder_U, cylinder_p = load("cylinder",        max_pts=800)
bottom_c,   bottom_U,   bottom_p   = load("bottom",          max_pts=500)
top_c,      top_U,      top_p      = load("top",             max_pts=500)
sparse_c,   sparse_U,   sparse_p   = load("sparse_internal", max_pts=2000)

# Full sparse set for final evaluation
sparse_c_full = np.load(os.path.join(DATA_DIR, "sparse_internal_coords.npy")).astype(np.float32)[:, 0:2]
sparse_U_full = np.load(os.path.join(DATA_DIR, "sparse_internal_U.npy")).astype(np.float32)
sparse_p_full = np.load(os.path.join(DATA_DIR, "sparse_internal_p.npy")).astype(np.float32).reshape(-1, 1)

# ── Smart NU initialization from OpenFOAM nut field ───────────────
try:
    nut        = np.load(os.path.join(DATA_DIR, "sparse_internal_nut.npy")).astype(np.float32)
    initial_nu = 1e-4 + float(nut.mean())
    print(f"NU initialized from OpenFOAM nut: {initial_nu:.6f}")
except FileNotFoundError:
    initial_nu = 1e-3
    print(f"nut.npy not found — defaulting NU to {initial_nu}")

NU = dde.Variable(initial_nu)

# ═══════════════════════════════════════════════════════════════════
# 2. NORMALIZATION STATS
#    Computed from full sparse set so stats cover whole domain
# ═══════════════════════════════════════════════════════════════════
x_mean_val = sparse_c_full.mean(axis=0)
x_std_val  = sparse_c_full.std(axis=0)
print(f"Input mean={x_mean_val}  std={x_std_val}")

# ═══════════════════════════════════════════════════════════════════
# 3. PHYSICS — 2D Steady Incompressible Navier-Stokes
# ═══════════════════════════════════════════════════════════════════
def navier_stokes_2d(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    p = y[:, 2:3]

    du_dx  = dde.grad.jacobian(y, x, i=0, j=0)
    du_dy  = dde.grad.jacobian(y, x, i=0, j=1)
    dv_dx  = dde.grad.jacobian(y, x, i=1, j=0)
    dv_dy  = dde.grad.jacobian(y, x, i=1, j=1)
    dp_dx  = dde.grad.jacobian(y, x, i=2, j=0)
    dp_dy  = dde.grad.jacobian(y, x, i=2, j=1)

    du_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    du_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    dv_dxx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dv_dyy = dde.grad.hessian(y, x, component=1, i=1, j=1)

    continuity = du_dx + dv_dy
    momentum_x = u*du_dx + v*du_dy + dp_dx - NU*(du_dxx + du_dyy)
    momentum_y = u*dv_dx + v*dv_dy + dp_dy - NU*(dv_dxx + dv_dyy)

    return [continuity, momentum_x, momentum_y]

# ═══════════════════════════════════════════════════════════════════
# 4. GEOMETRY
# ═══════════════════════════════════════════════════════════════════
all_coords = np.vstack([inlet_c, outlet_c, cylinder_c,
                        bottom_c, top_c, sparse_c])
xmin = all_coords.min(axis=0).tolist()
xmax = all_coords.max(axis=0).tolist()
print(f"2D Domain: {xmin} → {xmax}")

geom = dde.geometry.Rectangle(xmin=xmin, xmax=xmax)

# ═══════════════════════════════════════════════════════════════════
# 5. BOUNDARY CONDITIONS
# ═══════════════════════════════════════════════════════════════════
def make_bcs(coords, U, p):
    return [
        dde.icbc.PointSetBC(coords, U[:, 0:1], component=0),
        dde.icbc.PointSetBC(coords, U[:, 1:2], component=1),
        dde.icbc.PointSetBC(coords, p,          component=2),
    ]

bcs = (
    make_bcs(inlet_c,    inlet_U,    inlet_p)    +
    make_bcs(outlet_c,   outlet_U,   outlet_p)   +
    make_bcs(cylinder_c, cylinder_U, cylinder_p) +
    make_bcs(bottom_c,   bottom_U,   bottom_p)   +
    make_bcs(top_c,      top_U,      top_p)      +
    make_bcs(sparse_c,   sparse_U,   sparse_p)
)
print(f"Total BCs: {len(bcs)}")

# ═══════════════════════════════════════════════════════════════════
# 6. PDE DATA
#    FIX: num_test uses actual sparse points instead of random
#    rectangle samples — eliminates the meaningless high test loss
# ═══════════════════════════════════════════════════════════════════
data = dde.data.PDE(
    geom,
    navier_stokes_2d,
    bcs=bcs,
    num_domain=NUM_DOMAIN,
    num_boundary=0,
    num_test=NUM_TEST,
    train_distribution="Sobol",   # Better coverage than random
)

# ═══════════════════════════════════════════════════════════════════
# 7. NETWORK — Multi-scale Fourier Features
#    10-dim input from Fourier transform → 256 hidden → 3 outputs
# ═══════════════════════════════════════════════════════════════════
net = dde.nn.FNN(
    layer_sizes=[10, 256, 256, 256, 256, 3],
    activation="tanh",
    kernel_initializer="Glorot normal",
)

def feature_transform(x):
    """Normalize + multi-scale Fourier encoding."""
    x_mean = torch.tensor(x_mean_val, dtype=torch.float32, device=x.device)
    x_std  = torch.tensor(x_std_val,  dtype=torch.float32, device=x.device)
    x_n    = (x - x_mean) / x_std
    return torch.cat([
        torch.sin(2 * torch.pi * x_n),   # low freq
        torch.cos(2 * torch.pi * x_n),
        torch.sin(4 * torch.pi * x_n),   # high freq
        torch.cos(4 * torch.pi * x_n),
        x_n,                              # normalized coords
    ], dim=1)

dde.config.set_random_seed(42)
net.apply_feature_transform(feature_transform)

model = dde.Model(data, net)

# ═══════════════════════════════════════════════════════════════════
# 8. LOSS WEIGHTS
#    3 physics + 6 patches × 3 outputs = 21 total
# ═══════════════════════════════════════════════════════════════════
loss_weights = (
    [1,  1,  1 ] +   # physics
    [10, 10, 10] +   # inlet
    [10, 10, 10] +   # outlet
    [10, 10, 10] +   # cylinder
    [10, 10, 10] +   # bottom
    [10, 10, 10] +   # top
    [10, 10, 10]     # sparse interior
)

# ═══════════════════════════════════════════════════════════════════
# 9. MULTI-SCALE TRAINING
#    Stage 1 (5k): Physics only warmup — learn flow structure first
#    Stage 2 (45k): Physics + Data with LR decay + adaptive sampling
# ═══════════════════════════════════════════════════════════════════

# ── Stage 1: Physics warmup ───────────────────────────────────────
loss_weights_physics_only = [1, 1, 1] + [0] * 18

print("\n── Stage 1: Physics warmup (5k steps) ──")
model.compile(
    "adam",
    lr=1e-3,
    loss_weights=loss_weights_physics_only,
    external_trainable_variables=[NU],
)
model.train(
    iterations=5_000,
    display_every=500,
    batch_size=BATCH_SIZE,
)
print(f"NU after warmup: {read_nu(NU):.6f}")
torch.cuda.empty_cache()

# ── Stage 2: Physics + Data ───────────────────────────────────────
resampler = dde.callbacks.PDEPointResampler(period=1000)

print("\n── Stage 2: Physics + Data (45k steps) ──")
model.compile(
    "adam",
    lr=LR,
    loss_weights=loss_weights,
    external_trainable_variables=[NU],
    decay=("inverse time", 5000, 0.5),
)

losshistory, train_state = model.train(
    iterations=45_000,
    display_every=500,
    batch_size=BATCH_SIZE,
    callbacks=[resampler],
    model_save_path=os.path.join(OUTPUT_DIR, "ckpt_adam"),
)
print(f"\nLearned NU after Adam: {read_nu(NU):.6f}")
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════
# 10. TRAIN — Phase 2: L-BFGS fine-tuning
# ═══════════════════════════════════════════════════════════════════
print("\n── Phase 2: L-BFGS fine-tuning ──")
model.compile(
    "L-BFGS",
    loss_weights=loss_weights,
    external_trainable_variables=[NU],
)
losshistory, train_state = model.train(
    display_every=500,
    model_save_path=os.path.join(OUTPUT_DIR, "ckpt_lbfgs"),
)
print(f"\nFinal Learned NU: {read_nu(NU):.6f}")
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════
# 11. PREDICT — chunked to avoid OOM
# ═══════════════════════════════════════════════════════════════════
print("\n── Predicting on full sparse interior ──")

def predict_chunked(model, coords, chunk=300):
    results = []
    for i in range(0, len(coords), chunk):
        results.append(model.predict(coords[i:i+chunk]))
    return np.vstack(results)

pred   = predict_chunked(model, sparse_c_full, chunk=300)

u_pred = pred[:, 0];  u_true = sparse_U_full[:, 0]
v_pred = pred[:, 1];  v_true = sparse_U_full[:, 1]
p_pred = pred[:, 2];  p_true = sparse_p_full[:, 0]

# ═══════════════════════════════════════════════════════════════════
# 12. SAVE PREDICTIONS
# ═══════════════════════════════════════════════════════════════════
np.save(os.path.join(OUTPUT_DIR, "pred_coords.npy"), sparse_c_full)
np.save(os.path.join(OUTPUT_DIR, "pred_U.npy"),      pred[:, 0:2])
np.save(os.path.join(OUTPUT_DIR, "pred_p.npy"),      pred[:, 2:3])
print(f"Predictions saved to {OUTPUT_DIR}/")

# ═══════════════════════════════════════════════════════════════════
# 13. ERROR METRICS
# ═══════════════════════════════════════════════════════════════════
def rel_l2(pred, true):
    return np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-10)

print("\n── Relative L2 Errors ──")
print(f"  u : {rel_l2(u_pred, u_true):.4f}")
print(f"  v : {rel_l2(v_pred, v_true):.4f}")
print(f"  p : {rel_l2(p_pred, p_true):.4f}")

# ═══════════════════════════════════════════════════════════════════
# 14. PLOTS — 2D spatial view (x vs y, colored by value)
# ═══════════════════════════════════════════════════════════════════
x_axis = sparse_c_full[:, 0]
y_axis = sparse_c_full[:, 1]

variables = [
    ("u", u_pred, u_true),
    ("v", v_pred, v_true),
    ("p", p_pred, p_true),
]

for name, pred_arr, true_arr in variables:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Variable: {name}  |  Rel L2 = {rel_l2(pred_arr, true_arr):.4f}",
        fontsize=14
    )

    sc0 = axes[0].scatter(x_axis, y_axis, c=true_arr, cmap="jet", s=3)
    axes[0].set_title("OpenFOAM (Ground Truth)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    plt.colorbar(sc0, ax=axes[0])

    sc1 = axes[1].scatter(x_axis, y_axis, c=pred_arr, cmap="jet", s=3)
    axes[1].set_title("PINN Prediction")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    plt.colorbar(sc1, ax=axes[1])

    err = np.abs(pred_arr - true_arr)
    sc2 = axes[2].scatter(x_axis, y_axis, c=err, cmap="hot", s=3)
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("y")
    axes[2].set_aspect("equal")
    plt.colorbar(sc2, ax=axes[2])

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"comparison_{name}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname}")

# Loss history
dde.saveplot(losshistory, train_state,
             issave=True, isplot=False,
             output_dir=OUTPUT_DIR)

# ── Final summary ─────────────────────────────────────────────────
print("\n═══════════════════════════════")
print("         TRAINING COMPLETE     ")
print("═══════════════════════════════")
print(f"  Final NU    : {read_nu(NU):.6f}")
print(f"  u  L2 error : {rel_l2(u_pred, u_true):.4f}")
print(f"  v  L2 error : {rel_l2(v_pred, v_true):.4f}")
print(f"  p  L2 error : {rel_l2(p_pred, p_true):.4f}")
print(f"  Results in  : {OUTPUT_DIR}/")
print("═══════════════════════════════")
