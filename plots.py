import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
DATA_DIR   = "VTK/PINN_Data"
PRED_DIR   = "results_train_improved"
PLOT_DIR   = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Cylinder parameters (adjust if different)
CYL_CENTER = (0.0, 0.0)
CYL_RADIUS = 0.5
D          = 2 * CYL_RADIUS    # diameter = 1.0
U_INF      = 10.0              # freestream velocity (check your inlet U)

# ═══════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════
print("Loading data...")

# Predicted fields (from train_deepxde.py output)
coords = np.load(os.path.join(PRED_DIR, "pred_coords.npy"))   # (N,2) x,y
U_pred = np.load(os.path.join(PRED_DIR, "pred_U.npy"))        # (N,2) u,v
p_pred = np.load(os.path.join(PRED_DIR, "pred_p.npy"))        # (N,1) p

x_pred = coords[:, 0]
y_pred = coords[:, 1]
u_pred = U_pred[:, 0]
v_pred = U_pred[:, 1]
p_pred = p_pred[:, 0]

# Ground truth (OpenFOAM sparse internal)
coords_gt = np.load(os.path.join(DATA_DIR, "sparse_internal_coords.npy"))[:, 0:2]
U_gt      = np.load(os.path.join(DATA_DIR, "sparse_internal_U.npy"))
p_gt      = np.load(os.path.join(DATA_DIR, "sparse_internal_p.npy")).reshape(-1)
nut_gt    = np.load(os.path.join(DATA_DIR, "sparse_internal_k.npy")).reshape(-1)

x_gt = coords_gt[:, 0]
y_gt = coords_gt[:, 1]
u_gt = U_gt[:, 0]
v_gt = U_gt[:, 1]

print(f"Loaded {len(x_pred)} prediction points, {len(x_gt)} ground truth points")

# ═══════════════════════════════════════════════════════════════════
# HELPER: mask points inside cylinder
# ═══════════════════════════════════════════════════════════════════
def outside_cylinder(x, y, cx=CYL_CENTER[0], cy=CYL_CENTER[1], r=CYL_RADIUS):
    return np.sqrt((x - cx)**2 + (y - cy)**2) > r

def cylinder_patch(ax):
    """Draw grey cylinder on a plot."""
    circle = plt.Circle(CYL_CENTER, CYL_RADIUS, color="grey",
                        zorder=5, linewidth=1.5)
    ax.add_patch(circle)

# ═══════════════════════════════════════════════════════════════════
# HELPER: interpolate scattered points onto a regular grid
# ═══════════════════════════════════════════════════════════════════
def to_grid(x, y, z, nx=300, ny=200):
    """Triangulate scattered data onto a regular grid."""
    mask = outside_cylinder(x, y)
    xi   = np.linspace(x.min(), x.max(), nx)
    yi   = np.linspace(y.min(), y.max(), ny)
    Xi, Yi = np.meshgrid(xi, yi)

    triang = tri.Triangulation(x[mask], y[mask])
    interp = tri.LinearTriInterpolator(triang, z[mask])
    Zi     = interp(Xi, Yi)

    # Mask inside cylinder on grid
    cyl_mask = ~outside_cylinder(Xi.ravel(), Yi.ravel())
    Zi.ravel()[cyl_mask] = np.nan

    return Xi, Yi, Zi

# ═══════════════════════════════════════════════════════════════════
# PLOT 1 — Wake Centerline Velocity Profiles (Image 1 style)
# ═══════════════════════════════════════════════════════════════════
print("\nGenerating wake centerline velocity profiles...")

x_slices = [1.0 * D, 2.0 * D, 3.0 * D]   # x = 1D, 2D, 3D downstream
tol       = 0.3                             # window around each x slice

fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
fig.suptitle("Wake Centerline Velocity Profiles — PINN vs OpenFOAM", fontsize=14)

for ax, x_loc in zip(axes, x_slices):
    # Extract points near this x slice — ground truth
    mask_gt   = np.abs(x_gt - x_loc) < tol
    y_slice_gt = y_gt[mask_gt]
    u_slice_gt = u_gt[mask_gt]
    sort_gt    = np.argsort(y_slice_gt)

    # Extract points near this x slice — PINN
    mask_pred   = np.abs(x_pred - x_loc) < tol
    y_slice_pred = y_pred[mask_pred]
    u_slice_pred = u_pred[mask_pred]
    sort_pred    = np.argsort(y_slice_pred)

    # Relative L2 at this slice
    if mask_gt.sum() > 0 and mask_pred.sum() > 0:
        # Interpolate PINN to GT y locations for fair comparison
        u_interp = np.interp(
            np.sort(y_slice_gt),
            np.sort(y_slice_pred),
            u_slice_pred[sort_pred]
        )
        rel_l2 = (np.linalg.norm(u_interp - u_slice_gt[sort_gt]) /
                  (np.linalg.norm(u_slice_gt[sort_gt]) + 1e-10))
    else:
        rel_l2 = float("nan")

    ax.plot(u_slice_pred[sort_pred], y_slice_pred[sort_pred],
            "b-", linewidth=2, label="PINN")
    ax.plot(u_slice_gt[sort_gt],     y_slice_gt[sort_gt],
            "r--", linewidth=1.5, label="OpenFOAM")
    ax.axvline(U_INF, color="grey", linestyle=":", linewidth=1, label=r"U$_\infty$")

    ax.set_title(f"x = {x_loc/D:.1f}D  |  rel.L2 = {rel_l2:.3f}")
    ax.set_xlabel("u [m/s]")
    ax.set_ylabel("y [m]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)

plt.tight_layout()
fname = os.path.join(PLOT_DIR, "wake_centerline_profiles.png")
plt.savefig(fname, dpi=150)
plt.close()
print(f"  Saved {fname}")

# ═══════════════════════════════════════════════════════════════════
# PLOT 2 — Full 2D Flow Field: OpenFOAM vs PINN vs |Error| (Image 2)
# ═══════════════════════════════════════════════════════════════════
print("\nGenerating 2D flow field comparison...")

# Interpolate all fields onto grids
print("  Interpolating U (GT)...")
Xi, Yi, u_gt_grid   = to_grid(x_gt,   y_gt,   u_gt)
print("  Interpolating V (GT)...")
_,  _,  v_gt_grid   = to_grid(x_gt,   y_gt,   v_gt)
print("  Interpolating P (GT)...")
_,  _,  p_gt_grid   = to_grid(x_gt,   y_gt,   p_gt)
print("  Interpolating nut (GT)...")
_,  _,  nut_gt_grid = to_grid(x_gt,   y_gt,   nut_gt)

print("  Interpolating U (PINN)...")
_,  _,  u_pr_grid   = to_grid(x_pred, y_pred, u_pred)
print("  Interpolating V (PINN)...")
_,  _,  v_pr_grid   = to_grid(x_pred, y_pred, v_pred)
print("  Interpolating P (PINN)...")
_,  _,  p_pr_grid   = to_grid(x_pred, y_pred, p_pred)

# PINN has no nut — show zeros (model didn't predict it)
nut_pr_grid = np.zeros_like(u_pr_grid)

rows = [
    ("U velocity [m/s]",      u_gt_grid,   u_pr_grid,   "RdBu_r"),
    ("V velocity [m/s]",      v_gt_grid,   v_pr_grid,   "RdBu_r"),
    ("Pressure [m²/s²]",      p_gt_grid,   p_pr_grid,   "RdYlGn"),
    ("Turb. kinetic energy k", nut_gt_grid, nut_pr_grid, "plasma"),
]

fig, axes = plt.subplots(4, 3, figsize=(18, 20))
fig.suptitle("OpenFOAM vs PINN — Flow Past Cylinder Re=10000 (t=5.0s)",
             fontsize=14, y=1.01)

for row_idx, (label, gt_grid, pr_grid, cmap) in enumerate(rows):
    ax_gt  = axes[row_idx, 0]
    ax_pr  = axes[row_idx, 1]
    ax_err = axes[row_idx, 2]

    # Shared color scale for GT and PINN panels
    vmin = np.nanmin(gt_grid)
    vmax = np.nanmax(gt_grid)

    # GT panel
    im0 = ax_gt.pcolormesh(Xi, Yi, gt_grid, cmap=cmap,
                            vmin=vmin, vmax=vmax, shading="auto")
    ax_gt.set_title("OpenFOAM" if row_idx == 0 else "")
    ax_gt.set_ylabel(label, fontsize=9)
    cylinder_patch(ax_gt)
    plt.colorbar(im0, ax=ax_gt, fraction=0.03, pad=0.02)

    # PINN panel
    im1 = ax_pr.pcolormesh(Xi, Yi, pr_grid, cmap=cmap,
                            vmin=vmin, vmax=vmax, shading="auto")
    ax_pr.set_title("PINN" if row_idx == 0 else "")
    cylinder_patch(ax_pr)
    plt.colorbar(im1, ax=ax_pr, fraction=0.03, pad=0.02)

    # Error panel
    err_grid = np.abs(gt_grid - pr_grid)
    im2 = ax_err.pcolormesh(Xi, Yi, err_grid, cmap="hot_r",
                             shading="auto")
    ax_err.set_title("|Error|" if row_idx == 0 else "")
    cylinder_patch(ax_err)
    plt.colorbar(im2, ax=ax_err, fraction=0.03, pad=0.02)

    # Formatting
    for ax in [ax_gt, ax_pr, ax_err]:
        ax.set_xlim(Xi.min(), Xi.max())
        ax.set_ylim(Yi.min(), Yi.max())
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.tick_params(labelsize=7)

plt.tight_layout()
fname = os.path.join(PLOT_DIR, "flow_field_comparison.png")
plt.savefig(fname, dpi=120, bbox_inches="tight")
plt.close()
print(f"  Saved {fname}")

print("\n✓ All plots saved to plots/ folder")
print("  wake_centerline_profiles.png")
print("  flow_field_comparison.png")
