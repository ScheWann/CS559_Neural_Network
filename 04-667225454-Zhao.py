import numpy as np
import matplotlib.pyplot as plt


n = 300
TARGET_MSE = 0.01
N_START = 6
N_MAX = 128
CAPACITY_GROWTH = 2.0

MAX_EPOCHS = 10000
PATIENCE = 600
PRINT_EVERY = 100

CENTER_X = True 

MAX_TRIES = 200


data_rng = np.random.default_rng(42)
x_raw = data_rng.uniform(0.0, 1.0, size=n)
noise = data_rng.uniform(-0.1, 0.1, size=n)

f_clean = np.sin(20 * x_raw) + 3 * x_raw
d_true = f_clean + noise

# Tanh
x = 2.0 * x_raw - 1.0 if CENTER_X else x_raw.copy()


def init_params(N, rng):
    a = rng.normal(0.0, 1.0, size=N)
    b = rng.uniform(-0.5, 0.5, size=N)
    c = rng.normal(0.0, 1.0 / np.sqrt(N), size=N)
    d = 0.0
    return a, b, c, d


def forward_batch(x_vec, a, b, c, d):
    z = np.outer(x_vec, a) + b
    h = np.tanh(z)
    y = h @ c + d
    return y, h, z


def batch_mse(a, b, c, d, x_vec, d_vec):
    y, _, _ = forward_batch(x_vec, a, b, c, d)
    return float(np.mean((d_vec - y) ** 2))


def train_batch_auto(x_vec, d_vec, N, seed, max_epochs, patience, print_every=None):
    rng = np.random.default_rng(seed)
    a, b, c, d = init_params(N, rng)

    eta = 0.02
    ETA_MIN, ETA_MAX = 1e-6, 0.2

    best_mse = np.inf
    best_params = (a.copy(), b.copy(), c.copy(), d)
    history = []
    no_improve = 0

    n = len(x_vec)

    for epoch in range(1, max_epochs + 1):
        # Forward
        y, h, _ = forward_batch(x_vec, a, b, c, d)
        e = d_vec - y

        # dL/dy = -(1/n) * e
        dy = -(1.0 / n) * e

        # Gradients
        dc = h.T @ dy
        dd = np.sum(dy)
        dh = 1.0 - h**2
        temp = dy[:, None] * dh
        da = (temp * (x_vec[:, None] * c[None, :])).sum(axis=0)
        db = (temp * c[None, :]).sum(axis=0)

        # Update
        a -= eta * da
        b -= eta * db
        c -= eta * dc
        d -= eta * dd

        # Evaluate
        epoch_mse = batch_mse(a, b, c, d, x_vec, d_vec)
        history.append(epoch_mse)

        if epoch_mse < best_mse - 1e-12:
            best_mse = epoch_mse
            best_params = (a.copy(), b.copy(), c.copy(), d)
            no_improve = 0
            eta = min(ETA_MAX, eta * 1.02)
        else:
            eta = max(ETA_MIN, eta * 0.5)
            no_improve += 1

        if print_every and (epoch % print_every == 0 or epoch == 1):
            print(
                f"[N={N:3d} seed={seed:3d}] Epoch {epoch:5d} | "
                f"MSE={epoch_mse:.6f} | best={best_mse:.6f} | eta={eta:.6f}"
            )

        if no_improve >= patience or best_mse <= TARGET_MSE:
            break

    return best_mse, best_params, history, epoch


global_best_mse = np.inf
global_best = None

seed_counter = 0
tries = 0
N = N_START

while N <= N_MAX:
    print(f"\n=== Trying capacity N = {N} ===")

    max_restarts = 6
    for r in range(max_restarts):
        if tries >= MAX_TRIES:
            print("Hit MAX_TRIES; stopping search.")
            break

        best_mse, params, hist, epochs_run = train_batch_auto(
            x_vec=x,
            d_vec=d_true,
            N=N,
            seed=seed_counter,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            print_every=PRINT_EVERY,
        )
        seed_counter += 1
        tries += 1

        print(
            f"Restart {r+1}/{max_restarts}: best MSE={best_mse:.6f} (epochs={epochs_run})"
        )

        if best_mse < global_best_mse:
            global_best_mse = best_mse
            global_best = (N, seed_counter - 1, params, hist)

        if global_best_mse <= TARGET_MSE or tries >= MAX_TRIES:
            break

    # stop if target hit exhausted
    if global_best_mse <= TARGET_MSE:
        print(f"Reached target MSE {TARGET_MSE} at N={N}. Stopping.")
        break
    if tries >= MAX_TRIES:
        print("Reached MAX_TRIES without hitting target; stopping.")
        break

    next_N = int(min(N_MAX, max(N + 1, int(N * CAPACITY_GROWTH))))

    if next_N == N:
        print("Reached N_MAX and target not met; stopping search.")
        break

    N = next_N

if global_best is None:
    raise RuntimeError("No training run completed (unexpected).")

N_star, seed_star, (a_star, b_star, c_star, d_star), hist_star = global_best

print(f"N: {N_star}")
print(f"seed: {seed_star}")
print(f"Best MSE: {global_best_mse:.6f}")

plt.figure(figsize=(7, 4.5))
plt.plot(np.arange(1, len(hist_star) + 1), hist_star, linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("MSE")
title_extra = "x∈[-1,1]" if CENTER_X else "x∈[0,1]"
plt.title(
    f"Best — N={N_star}, seed={seed_star}, {title_extra}, MSE={global_best_mse:.6f}"
)
plt.tight_layout()

xx_raw = np.linspace(0, 1, 1000)
xx = 2.0 * xx_raw - 1.0 if CENTER_X else xx_raw


def forward_for_plot(xv):
    y, _, _ = forward_batch(xv, a_star, b_star, c_star, d_star)
    return y


yy = forward_for_plot(xx)

plt.figure(figsize=(7, 4.5))
plt.scatter(x_raw, d_true, s=10, alpha=0.6, label="Data (xi, di)")
plt.plot(xx_raw, yy, linewidth=2.0, label="Fit f(x)")
plt.xlabel("x")
plt.ylabel("d / f(x)")
plt.title("Curve Fitting")
plt.legend()
plt.tight_layout()
plt.show()
