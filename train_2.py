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

# Memory budget for RTX 2050 4GB
# k as extra output adds ~100MB so we slightly reduce domain points
NUM_DOMAIN   = 4000     # reduced from 5000 to compensate for k output
NUM_TEST     = 500
BATCH_SIZE   = 512

# Cylinder geometry
CYL_CENTER   = [0.0, 0.0]
CYL_RADIUS   = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ═══════════════════════════════════════════════════════════════════
# HELPER: safe NU reading across DeepXDE versions
# ═══════════════════════════════════════════════════════════════════
def read_log_nu(log_nu_var):
    try:
        return float(torch.exp(log_nu_var).item())
    except Exception:
        return float(np.exp(float(log_nu_var)))

# ═══════════════════════════════════════════════════════════════════
# 1. LOAD DATA
#    IMPROVEMENT 6: load k field alongside U and p
# ═══════════════════════════════════════════════════════════════════
def load(name, max_pts=1000):
    c = np.load(os.path.join(DATA_DIR, f"{name}_coords.npy")).astype(np.float32)
    U = np.load(os.path.join(DATA_DIR, f"{name}_U.npy")).astype(np.float32)
    p = np.load(os.path.join(DATA_DIR, f"{name}_p.npy")).astype(np.float32).reshape(-1, 1)
    k = np.load(os.path.join(DATA_DIR, f"{name}_k.npy")).astype(np.float32).reshape(-1, 1)
    c = c[:, 0:2]  # x,y only — domain is quasi-2D (z: 0→0.01)
    if len(c) > max_pts:
        idx = np.random.choice(len(c), max_pts, replace=False)
        c, U, p, k = c[idx], U[idx], p[idx], k[idx]
    print(f"  {name}: {len(c)} points")
    return c, U, p, k

print("Loading data...")
inlet_c,    inlet_U,    inlet_p,    inlet_k    = load("inlet",           max_pts=500)
outlet_c,   outlet_U,   outlet_p,   outlet_k   = load("outlet",          max_pts=500)
cylinder_c, cylinder_U, cylinder_p, cylinder_k = load("cylinder",        max_pts=800)
bottom_c,   bottom_U,   bottom_p,   bottom_k   = load("bottom",          max_pts=500)
top_c,      top_U,      top_p,      top_k      = load("top",             max_pts=500)
sparse_c,   sparse_U,   sparse_p,   sparse_k   = load("sparse_internal", max_pts=2000)

# Full sparse set for final evaluation
sparse_c_full = np.load(os.path.join(DATA_DIR, "sparse_internal_coords.npy")).astype(np.float32)[:, 0:2]
sparse_U_full = np.load(os.path.join(DATA_DIR, "sparse_internal_U.npy")).astype(np.float32)
sparse_p_full = np.load(os.path.join(DATA_DIR, "sparse_internal_p.npy")).astype(np.float32).reshape(-1, 1)
sparse_k_full = np.load(os.path.join(DATA_DIR, "sparse_internal_k.npy")).astype(np.float32).reshape(-1, 1)

# ── Smart NU initialization ────────────────────────────────────────
# IMPROVEMENT 1: Use log parameterization so NU is always positive
# LOG_NU is the trainable variable; actual NU = exp(LOG_NU) > 0 always
try:
    nut        = np.load(os.path.join(DATA_DIR, "sparse_internal_nut.npy")).astype(np.float32)
    initial_nu = 1e-4 + float(nut.mean())
except FileNotFoundError:
    # Estimate from k and omega if nut not available
    # nu_t = k / omega (simplified)
    try:
        k_data     = sparse_k_full.ravel()
        omega_data = np.load(os.path.join(DATA_DIR, "sparse_internal_omega.npy")).astype(np.float32).ravel()
        # clip to avoid division by zero
        nu_t_est   = np.mean(k_data / np.clip(omega_data, 1e-6, None))
        initial_nu = 1e-4 + nu_t_est
        print(f"NU estimated from k/omega: {initial_nu:.6f}")
    except FileNotFoundError:
        initial_nu = 1e-3
        print(f"Defaulting NU to {initial_nu}")

# Store as log so exp(LOG_NU) is always positive — IMPROVEMENT 1
LOG_NU = dde.Variable(float(np.log(initial_nu)))
print(f"Initial NU = {initial_nu:.6f}  →  LOG_NU = {float(np.log(initial_nu)):.4f}")

# ═══════════════════════════════════════════════════════════════════
# 2. NORMALIZATION STATS
#    Input normalization from full domain coverage
# ═══════════════════════════════════════════════════════════════════
x_mean_val = sparse_c_full.mean(axis=0)
x_std_val  = sparse_c_full.std(axis=0)
print(f"Input mean={x_mean_val}  std={x_std_val}")

# IMPROVEMENT 4: Output normalization scales
# Approximate magnitudes from OpenFOAM data
u_scale = float(np.abs(sparse_U_full[:, 0]).mean()) + 1e-6
v_scale = float(np.abs(sparse_U_full[:, 1]).mean()) + 1e-6
p_scale = float(np.abs(sparse_p_full).mean())       + 1e-6
k_scale = float(np.abs(sparse_k_full).mean())       + 1e-6
print(f"Output scales — u:{u_scale:.3f}  v:{v_scale:.4f}  p:{p_scale:.3f}  k:{k_scale:.6f}")

# ═══════════════════════════════════════════════════════════════════
# 3. PHYSICS — 2D Steady Incompressible Navier-Stokes
#    IMPROVEMENT 1: NU = exp(LOG_NU) — always positive
#    Outputs: [u, v, p, k]  (indices 0,1,2,3)
# ═══════════════════════════════════════════════════════════════════
def navier_stokes_2d(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    p = y[:, 2:3]
    # k is output[:,3] but not used in NS equations directly
    # It provides a physics-informed anchor for turbulent kinetic energy

    # IMPROVEMENT 1: positive viscosity via exp
    nu_eff = torch.exp(LOG_NU)

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
    momentum_x = u*du_dx + v*du_dy + dp_dx - nu_eff*(du_dxx + du_dyy)
    momentum_y = u*dv_dx + v*dv_dy + dp_dy - nu_eff*(dv_dxx + dv_dyy)

    return [continuity, momentum_x, momentum_y]

# ═══════════════════════════════════════════════════════════════════
# 4. GEOMETRY
#    IMPROVEMENT 3: Subtract cylinder disk from rectangle
#    Ensures NO collocation points land inside the solid cylinder
# ═══════════════════════════════════════════════════════════════════
all_coords = np.vstack([inlet_c, outlet_c, cylinder_c,
                        bottom_c, top_c, sparse_c])
xmin = all_coords.min(axis=0).tolist()
xmax = all_coords.max(axis=0).tolist()
print(f"2D Domain: {xmin} → {xmax}")

outer_rect = dde.geometry.Rectangle(xmin=xmin, xmax=xmax)
inner_disk = dde.geometry.Disk(CYL_CENTER, CYL_RADIUS)
geom       = outer_rect - inner_disk   # fluid domain only
print("Geometry: Rectangle minus Cylinder disk")

# ═══════════════════════════════════════════════════════════════════
# 5. BOUNDARY CONDITIONS
#    IMPROVEMENT 2: v loss weight = 20 (doubled vs u and p)
#    IMPROVEMENT 6: k added as 4th BC component
#
#    Loss layout per patch: [u, v, p, k]
#    Weights:               [10, 20, 10, 5]
#    k weight is lower (5) because k is a secondary quantity
# ═══════════════════════════════════════════════════════════════════
def make_bcs(coords, U, p, k):
    return [
        dde.icbc.PointSetBC(coords, U[:, 0:1], component=0),  # u
        dde.icbc.PointSetBC(coords, U[:, 1:2], component=1),  # v
        dde.icbc.PointSetBC(coords, p,          component=2),  # p
        dde.icbc.PointSetBC(coords, k,          component=3),  # k
    ]

bcs = (
    make_bcs(inlet_c,    inlet_U,    inlet_p,    inlet_k)    +
    make_bcs(outlet_c,   outlet_U,   outlet_p,   outlet_k)   +
    make_bcs(cylinder_c, cylinder_U, cylinder_p, cylinder_k) +
    make_bcs(bottom_c,   bottom_U,   bottom_p,   bottom_k)   +
    make_bcs(top_c,      top_U,      top_p,      top_k)      +
    make_bcs(sparse_c,   sparse_U,   sparse_p,   sparse_k)
)
print(f"Total BCs: {len(bcs)}  (3 physics + 6 patches × 4 outputs = {3 + 6*4})")

# ═══════════════════════════════════════════════════════════════════
# 6. PDE DATA
# ═══════════════════════════════════════════════════════════════════
data = dde.data.PDE(
    geom,
    navier_stokes_2d,
    bcs=bcs,
    num_domain=NUM_DOMAIN,
    num_boundary=0,
    num_test=NUM_TEST,
    train_distribution="Sobol",
)

# ═══════════════════════════════════════════════════════════════════
# 7. NETWORK
#    IMPROVEMENT 5: MsFFN with residual connections
#    Input: 10 (Fourier features) → 4 outputs [u, v, p, k]
#    sigmas control the multi-scale frequency encoding per layer
# ═══════════════════════════════════════════════════════════════════
net = dde.nn.pytorch.PFNN(
    layer_sizes=[10, [256, 256, 256, 256], [1, 1, 1, 1]],
    activation="tanh",
    kernel_initializer="Glorot normal",
)
print("Network: PFNN (parallel subnet per output)")
# ── Feature transform: Normalization + Fourier encoding ───────────
def feature_transform(x):
    x_mean = torch.tensor(x_mean_val, dtype=torch.float32, device=x.device)
    x_std  = torch.tensor(x_std_val,  dtype=torch.float32, device=x.device)
    x_n    = (x - x_mean) / x_std
    return torch.cat([
        torch.sin(2 * torch.pi * x_n),
        torch.cos(2 * torch.pi * x_n),
        torch.sin(4 * torch.pi * x_n),
        torch.cos(4 * torch.pi * x_n),
        x_n,
    ], dim=1)

# IMPROVEMENT 4: Output transform scales each output to its
# approximate physical magnitude so all outputs train equally
def output_transform(x, y):
    scales = torch.tensor(
        [u_scale, v_scale, p_scale, k_scale],
        dtype=torch.float32, device=y.device
    )
    return y * scales

dde.config.set_random_seed(42)
net.apply_feature_transform(feature_transform)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)

# ═══════════════════════════════════════════════════════════════════
# 8. LOSS WEIGHTS
#    Layout: [continuity, mom_x, mom_y,          ← 3 physics
#             u, v, p, k  × 6 patches]           ← 24 data
#    Total: 27 terms
#
#    IMPROVEMENT 2: v weight = 20 (all others = 10)
#    k weight = 5 (secondary quantity, lower priority)
# ═══════════════════════════════════════════════════════════════════
patch_weights = [10, 20, 10, 5]   # u, v, p, k

loss_weights = (
    [1, 1, 1]          +   # physics
    patch_weights      +   # inlet
    patch_weights      +   # outlet
    patch_weights      +   # cylinder
    patch_weights      +   # bottom
    patch_weights      +   # top
    patch_weights          # sparse interior
)
print(f"Loss weights: {len(loss_weights)} terms")

# ═══════════════════════════════════════════════════════════════════
# 9. MULTI-SCALE TRAINING
# ═══════════════════════════════════════════════════════════════════

# Stage 1: Physics warmup — all data weights = 0
loss_weights_warmup = [1, 1, 1] + [0] * 24

print("\n── Stage 1: Physics warmup (5k steps) ──")
model.compile(
    "adam",
    lr=1e-3,
    loss_weights=loss_weights_warmup,
    external_trainable_variables=[LOG_NU],
)
model.train(
    iterations=5_000,
    display_every=500,
    batch_size=BATCH_SIZE,
)
print(f"NU after warmup: {read_log_nu(LOG_NU):.6f}")
torch.cuda.empty_cache()

# Stage 2: Physics + Data with LR decay + adaptive resampler
resampler = dde.callbacks.PDEPointResampler(period=1000)

print("\n── Stage 2: Physics + Data (45k steps) ──")
model.compile(
    "adam",
    lr=LR,
    loss_weights=loss_weights,
    external_trainable_variables=[LOG_NU],
    decay=("inverse time", 5000, 0.5),
)

losshistory, train_state = model.train(
    iterations=45_000,
    display_every=500,
    batch_size=BATCH_SIZE,
    callbacks=[resampler],
    model_save_path=os.path.join(OUTPUT_DIR, "ckpt_adam"),
)
print(f"\nLearned NU after Adam: {read_log_nu(LOG_NU):.6f}")
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════
# 10. L-BFGS FINE-TUNING
# ═══════════════════════════════════════════════════════════════════
print("\n── Phase 2: L-BFGS fine-tuning ──")
model.compile(
    "L-BFGS",
    loss_weights=loss_weights,
    external_trainable_variables=[LOG_NU],
)
losshistory, train_state = model.train(
    display_every=500,
    model_save_path=os.path.join(OUTPUT_DIR, "ckpt_lbfgs"),
)
print(f"\nFinal NU: {read_log_nu(LOG_NU):.6f}  (always positive ✓)")
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

pred = predict_chunked(model, sparse_c_full, chunk=300)

u_pred = pred[:, 0];  u_true = sparse_U_full[:, 0]
v_pred = pred[:, 1];  v_true = sparse_U_full[:, 1]
p_pred = pred[:, 2];  p_true = sparse_p_full[:, 0]
k_pred = pred[:, 3];  k_true = sparse_k_full[:, 0]

# ═══════════════════════════════════════════════════════════════════
# 12. SAVE PREDICTIONS
# ═══════════════════════════════════════════════════════════════════
np.save(os.path.join(OUTPUT_DIR, "pred_coords.npy"), sparse_c_full)
np.save(os.path.join(OUTPUT_DIR, "pred_U.npy"),      pred[:, 0:2])
np.save(os.path.join(OUTPUT_DIR, "pred_p.npy"),      pred[:, 2:3])
np.save(os.path.join(OUTPUT_DIR, "pred_k.npy"),      pred[:, 3:4])
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
print(f"  k : {rel_l2(k_pred, k_true):.4f}")

# ═══════════════════════════════════════════════════════════════════
# 14. PLOTS
# ═══════════════════════════════════════════════════════════════════
x_axis = sparse_c_full[:, 0]
y_axis = sparse_c_full[:, 1]

variables = [
    ("u", u_pred, u_true),
    ("v", v_pred, v_true),
    ("p", p_pred, p_true),
    ("k", k_pred, k_true),
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

dde.saveplot(losshistory, train_state,
             issave=True, isplot=False,
             output_dir=OUTPUT_DIR)

# ── Final summary ─────────────────────────────────────────────────
print("\n═══════════════════════════════════════")
print("         TRAINING COMPLETE             ")
print("═══════════════════════════════════════")
print(f"  Final NU    : {read_log_nu(LOG_NU):.6f}  (positive ✓)")
print(f"  u  L2 error : {rel_l2(u_pred, u_true):.4f}")
print(f"  v  L2 error : {rel_l2(v_pred, v_true):.4f}")
print(f"  p  L2 error : {rel_l2(p_pred, p_true):.4f}")
print(f"  k  L2 error : {rel_l2(k_pred, k_true):.4f}")
print(f"  Results in  : {OUTPUT_DIR}/")
print("═══════════════════════════════════════")
