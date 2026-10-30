"""Spectral loss and FFT-weight sweep for the Swin2SR downscaling model.

Loss: L = MSE(pred, target) + lambda_fft * sum((log|FFT(pred)| - log|FFT(target)|)^2). # the use of log within the loss function might lead to the spurious values observed in the resulting outputs, something to be improved.

SpectralSRLoss: the loss. run_fft_parameter_sweep: trains one model per lambda_fft value, evaluates
it on the test set and writes the metrics to param_sweep_fft.csv, one folder per value in
<results_dir>/param_loss_function, and a sweep summary figure. run_fft_lambda_sweep: older
call signature of the same sweep. Running the file executes a short smoke test.
Author: Marion Falkenauer (with the help of Claude)
"""

import os
import gc
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

from ML_Methods_ERA5_CERRA import plot_comparison_transformer

from Swin2SR_tensorflow import infer

from skimage.metrics import structural_similarity as sk_ssim

_PYWT_AVAILABLE = False


class SpectralSRLoss:
    """MSE plus a log-amplitude FFT term, for tensors in BHWC layout."""

    def __init__(
        self,
        lambda_fft:    float = 1.0,
        lambda_wav:    float = 1.0,
        use_wavelet:   bool  = False,
        wavelet:       str   = "bior2.2",
        wavelet_levels:int   = 3,
        eps:           float = 1e-12,
    ):
        """Store the loss weights.

        Args:
            lambda_fft: weight of the FFT term (0 gives pure MSE).
            lambda_wav: weight of the wavelet term (not implemented, the term is always 0).
            use_wavelet: True raises ImportError (the wavelet term is not available).
            wavelet, wavelet_levels: wavelet settings.
            eps: constant added inside the logarithm.
        """
        self.lambda_fft     = lambda_fft
        self.lambda_wav     = lambda_wav
        self.use_wavelet    = use_wavelet
        self.wavelet        = wavelet
        self.wavelet_levels = wavelet_levels
        self.eps            = eps

        if self.use_wavelet and not _PYWT_AVAILABLE:
            raise ImportError(
                "PyWavelets is required for the wavelet loss. "
                "Install with:  pip install PyWavelets"
            )

    def _fft_loss(self, pred: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        """Sum of squared differences of the log FFT amplitudes.

        Args:
            pred, target: tensors (B, H, W, C).
        Returns:
            Scalar tensor.
        """
        pred_c   = tf.cast(tf.transpose(pred,   [0, 3, 1, 2]), tf.float32)
        target_c = tf.cast(tf.transpose(target, [0, 3, 1, 2]), tf.float32)

        F_pred   = tf.signal.rfft2d(pred_c)  # (B, C, H, W//2+1) complex64
        F_target = tf.signal.rfft2d(target_c)

        log_amp_pred   = tf.math.log(tf.abs(F_pred)   + self.eps)
        log_amp_target = tf.math.log(tf.abs(F_target) + self.eps)

        return tf.reduce_sum(tf.square(log_amp_pred - log_amp_target))

    def __call__(self, pred: tf.Tensor, target: tf.Tensor):
        """Compute the total loss and its terms.

        Args:
            pred, target: tensors (B, H, W, C).
        Returns:
            (total_loss, loss_mse, loss_fft, loss_wav), scalar tensors.
        """
        pred   = tf.cast(pred,   tf.float32)
        target = tf.cast(target, tf.float32)

        loss_mse = tf.reduce_mean(tf.square(pred - target))

        loss_fft = tf.constant(0.0)
        if self.lambda_fft > 0.0:
            loss_fft = self._fft_loss(pred, target)

        loss_wav   = tf.constant(0.0)
        total_loss = loss_mse + self.lambda_fft * loss_fft

        return total_loss, loss_mse, loss_fft, loss_wav


def train_step_spectral(model, optimizer, lr_input, hr_input,
                        criterion: SpectralSRLoss) -> dict:
    """One training step with SpectralSRLoss and gradient clipping at norm 1.

    Args:
        model: Swin2SR Keras model.
        optimizer: Keras optimizer.
        lr_input: LR batch (B, h, w, C) scaled to [0, 1].
        hr_input: HR batch (B, H, W, C) scaled to [0, 1].
        criterion: SpectralSRLoss instance.
    Returns:
        Dictionary of floats total, mse, fft and wav.
    """
    lr = tf.cast(lr_input, tf.float32)
    hr = tf.cast(hr_input, tf.float32)

    with tf.GradientTape() as tape:
        pred                      = model(lr, training=True)
        total, mse, fft, wav      = criterion(pred, hr)

    grads, _ = tf.clip_by_global_norm(
        tape.gradient(total, model.trainable_variables), 1.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return {
        "total": float(total.numpy()),
        "mse":   float(mse.numpy()),
        "fft":   float(fft.numpy()),
        "wav":   float(wav.numpy()),
    }


def _validation_loss(model, val_ds, criterion: SpectralSRLoss) -> float:
    """Mean total loss over a validation dataset, using infer for the predictions.

    Args:
        model: Swin2SR Keras model.
        val_ds: dataset of (lr, hr) batches.
        criterion: SpectralSRLoss instance.
    Returns:
        Mean loss as a float.
    """
    losses = []
    for lr_batch, hr_batch in val_ds:
        pred      = tf.convert_to_tensor(
            infer(model, lr_batch), dtype=tf.float32)
        hr        = tf.cast(hr_batch, tf.float32)
        total, _, _, _ = criterion(pred, hr)
        losses.append(float(total.numpy()))
    return float(np.mean(losses))


class LossTermTracker:
    """Collects the loss terms of each training step and saves their plot and CSV."""

    def __init__(self, results_dir: str, run_tag: str = ""):
        """Create empty histories.

        Args:
            results_dir: output directory.
            run_tag: suffix of the output file names.
        """
        self.results_dir = results_dir
        self.run_tag     = run_tag
        self.history: dict = {"total": [], "mse": [], "fft": [], "wav": []}

    def update(self, loss_dict: dict):
        """Append the values of one step.

        Args:
            loss_dict: dictionary with total, mse, fft and wav.
        """
        for key in self.history:
            self.history[key].append(loss_dict.get(key, 0.0))

    def plot(self):
        """Save loss_terms_<run_tag>.png with one panel per non-zero term."""
        active = [k for k in ("total", "mse", "fft", "wav")
                  if any(v != 0.0 for v in self.history[k])]
        if not active:
            return

        fig, axes = plt.subplots(1, len(active), figsize=(5 * len(active), 4))
        if len(active) == 1:
            axes = [axes]

        colors = {"total": "#333333", "mse": "#1f77b4",
                  "fft":   "#d62728", "wav": "#2ca02c"}
        labels = {"total": "Total loss", "mse": "MSE term",
                  "fft":   "FFT spectral term", "wav": "Wavelet term"}

        for ax, term in zip(axes, active):
            ax.plot(self.history[term], color=colors[term], linewidth=1.2)
            ax.set_title(labels[term], fontsize=13)
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.4)
            vals = np.array(self.history[term])
            vals = vals[vals > 0]
            if len(vals) > 0 and (vals.max() / (vals.min() + 1e-12)) > 100:
                ax.set_yscale("log")

        tag = f" — {self.run_tag}" if self.run_tag else ""
        fig.suptitle(f"Loss term history{tag}", fontsize=14)
        plt.tight_layout()
        fname = f"loss_terms_{self.run_tag}.png" if self.run_tag else "loss_terms.png"
        out   = os.path.join(self.results_dir, fname)
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[LossTermTracker] Saved loss-term plot -> {out}")

    def save_csv(self):
        """Save loss_terms_<run_tag>.csv with one row per step."""
        fname = f"loss_terms_{self.run_tag}.csv" if self.run_tag else "loss_terms.csv"
        out   = os.path.join(self.results_dir, fname)
        with open(out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "total", "mse", "fft", "wav"])
            for i, (t, m, ff, w) in enumerate(zip(
                    self.history["total"], self.history["mse"],
                    self.history["fft"],  self.history["wav"])):
                writer.writerow([i, t, m, ff, w])
        print(f"[LossTermTracker] Saved loss-term CSV  -> {out}")


def _evaluate_test_set(model, test_ds):
    """Evaluate a model on a test dataset with infer.

    Args:
        model: Swin2SR Keras model.
        test_ds: dataset of (lr, hr) batches scaled to [0, 1].
    Returns:
        Dictionary with mse_mean, psnr_mean, ssim_mean, the mean and standard deviation of the spatial
        SSIM maps of u and v (ssim_mean_u, ssim_std_u, ssim_mean_v, ssim_std_v) and the last batch
        (last_lr_batch, last_hr_batch, last_pred).
    """
    psnr_all        = []
    ssim_all        = []
    mse_all         = []
    ssim_map_list_u = []
    ssim_map_list_v = []
    last_lr_batch   = None
    last_hr_batch   = None
    last_pred       = None

    for lr_batch, hr_batch in test_ds:

        # Inference
        hr_pred = infer(model, lr_batch)
        y_pred  = tf.convert_to_tensor(hr_pred, dtype=tf.float32)
        y_true  = tf.cast(hr_batch, tf.float32)

        # Scalar metrics
        mse_all.append(float(tf.reduce_mean((y_pred - y_true) ** 2).numpy()))

        mse_u = tf.reduce_mean((y_pred[..., 0:1] - y_true[..., 0:1]) ** 2)
        mse_v = tf.reduce_mean((y_pred[..., 1:2] - y_true[..., 1:2]) ** 2)
        psnr_all.append(
            10.0 * tf.math.log(1.0 / ((mse_u + mse_v) / 2.0))
            / tf.math.log(10.0)
        )

        ssim_u = tf.image.ssim(y_true[..., 0:1], y_pred[..., 0:1], max_val=1.0)
        ssim_v = tf.image.ssim(y_true[..., 1:2], y_pred[..., 1:2], max_val=1.0)
        ssim_all.append((ssim_u + ssim_v) / 2.0)

        # Spatial SSIM maps (per sample, per channel)
        for b in range(hr_pred.shape[0]):
            _, map_u = sk_ssim(
                y_true[b, ..., 0].numpy(), y_pred[b, ..., 0].numpy(),
                full=True, data_range=1.0, win_size=11,
            )
            _, map_v = sk_ssim(
                y_true[b, ..., 1].numpy(), y_pred[b, ..., 1].numpy(),
                full=True, data_range=1.0, win_size=11,
            )
            ssim_map_list_u.append(map_u)
            ssim_map_list_v.append(map_v)

        last_lr_batch = lr_batch
        last_hr_batch = hr_batch
        last_pred     = hr_pred

    ssim_mean_u = np.stack(ssim_map_list_u).mean(axis=0)
    ssim_std_u  = np.stack(ssim_map_list_u).std(axis=0)
    ssim_mean_v = np.stack(ssim_map_list_v).mean(axis=0)
    ssim_std_v  = np.stack(ssim_map_list_v).std(axis=0)

    return {
        "mse_mean":    float(np.mean(mse_all)),
        "psnr_mean":   float(tf.reduce_mean(tf.stack(psnr_all)).numpy()),
        "ssim_mean":   float(tf.reduce_mean(tf.concat(ssim_all, axis=0)).numpy()),
        "ssim_mean_u": ssim_mean_u,
        "ssim_std_u":  ssim_std_u,
        "ssim_mean_v": ssim_mean_v,
        "ssim_std_v":  ssim_std_v,
        "last_lr_batch": last_lr_batch,
        "last_hr_batch": last_hr_batch,
        "last_pred":     last_pred,
    }


_CASE_DURATION = {
    "1year":  2920,
    "5year":  14608,
    "10year": 29216,
    "20year": 58440,
}


class ResetTrainLoss(keras.callbacks.Callback):
    """Callback resetting the loss_tracker of the model at the start of each epoch (no-op if absent)."""

    def on_epoch_begin(self, epoch, logs=None):
        """Reset the loss tracker of the model."""
        if hasattr(self.model, "loss_tracker"):
            self.model.loss_tracker.reset_state()


def run_fft_parameter_sweep(
    build_model_fn,
    datasets:          dict,
    minmax_stats:      dict,
    results_dir:       str,
    case:              str   = "1year",
    n_steps:           int   = 10,
    lambda_min:        float = 1e-7,
    lambda_max:        float = 1e-5,
    epochs:            int   = 150,
    lr_rate:           float = 1e-5,
    patience:          int   = 10,
    min_delta:         float = 1e-7,
    batch_size:        int   = 2,
    validation_steps:  int   = 2807,
    lat_hr=None, lon_hr=None,
    lat_lr=None, lon_lr=None,
):
    """Train, evaluate and plot one model for each FFT-loss weight.

    For each of n_steps values of lambda_fft between lambda_min and lambda_max, a new model is
    built, compiled, trained with checkpointing and early stopping, evaluated on the test set,
    and its metrics are appended to param_sweep_fft.csv. A sweep summary figure is saved at the end.

    Args:
        build_model_fn: function taking lambda_fft and returning a Keras model. A model with a
            loss_tracker attribute computes its own loss, otherwise SpectralSRLoss is used.
        datasets: dictionary with train_ds, val_ds and test_ds.
        minmax_stats: dictionary {"lr": (mins, maxs), "hr": (mins, maxs)} used to unscale the data.
        results_dir: output directory.
        case: training-set case, sets the steps per epoch.
        n_steps: number of lambda_fft values.
        lambda_min, lambda_max: range of lambda_fft (linear spacing).
        epochs: maximum number of epochs per model.
        lr_rate: Adam learning rate.
        patience: early-stopping patience in epochs.
        min_delta: minimum validation-loss improvement.
        batch_size: batch size used for the steps per epoch.
        validation_steps: validation steps per epoch.
        lat_hr, lon_hr, lat_lr, lon_lr: coordinate vectors of the HR and LR grids for the figures.
    """
    os.makedirs(results_dir, exist_ok=True)
    print("Model count:")

    # Denormalisation constants
    hr_mins = np.array([float(v) for v in minmax_stats["hr"][0]], dtype=np.float32)
    hr_maxs = np.array([float(v) for v in minmax_stats["hr"][1]], dtype=np.float32)
    lr_mins = np.array([float(v) for v in minmax_stats["lr"][0]], dtype=np.float32)
    lr_maxs = np.array([float(v) for v in minmax_stats["lr"][1]], dtype=np.float32)

    # Steps per epoch from the training-case duration
    steps_per_epoch = int(_CASE_DURATION.get(case, 2920) / batch_size)
    print(f"  steps_per_epoch = {steps_per_epoch}  (case='{case}', "
          f"batch_size={batch_size})")

    # CSV setup
    csv_path   = os.path.join(results_dir, "param_sweep_fft.csv")
    csv_header = [
        "run_time", "lambda_fft",
        "best_val_loss", "test_mse_scaled",
        "test_psnr", "test_ssim", "epochs_trained",
    ]
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_header)

    lambda_values = np.linspace(lambda_min, lambda_max, n_steps)

    for step_idx, lam_fft in enumerate(lambda_values):
        run_tag = f"fft_lambda_{lam_fft:.9f}".replace(".", "p")
        out_dir = os.path.join(results_dir, "param_loss_function", run_tag)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'='*62}")
        print(f"  Sweep step {step_idx+1}/{n_steps}  |  lambda_fft = {lam_fft:.9f}")
        print(f"  run_tag : {run_tag}")
        print(f"{'='*62}")

        # Fresh model
        model = build_model_fn(float(lam_fft))

        # Mode A: the model computes its own loss (loss_tracker attribute); mode B: wrap SpectralSRLoss as a Keras loss
        criterion = SpectralSRLoss(lambda_fft=float(lam_fft),
                                   lambda_wav=0.0, use_wavelet=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_rate)

        if hasattr(model, "loss_tracker"):
            # Mode A - model handles its own loss internally
            model.compile(optimizer=optimizer)
            print("  Compiled in Mode A (model has internal loss_tracker).")
        else:
            # Mode B - Keras-compatible spectral loss wrapper
            def _spectral_loss_fn(y_true, y_pred, _crit=criterion):
                """Keras loss returning the total SpectralSRLoss."""
                total, _, _, _ = _crit(y_pred, y_true)
                return total
            model.compile(optimizer=optimizer, loss=_spectral_loss_fn)
            print("  Compiled in Mode B (external spectral loss wrapper).")

        # Callbacks
        model_path = os.path.join(out_dir, "Swin2SR_weights_Checkpoint_TF2.weights.h5")

        callbacks = [
            ResetTrainLoss(),
            keras.callbacks.ModelCheckpoint(
                filepath          = model_path,
                monitor           = "val_loss",
                save_best_only    = True,
                save_weights_only = True,
            ),
            keras.callbacks.EarlyStopping(
                monitor              = "val_loss",
                patience             = patience,
                min_delta            = min_delta,
                mode                 = "min",
                restore_best_weights = True,
            ),
        ]

        history = model.fit(
            datasets["train_ds"],
            validation_data  = datasets["val_ds"],
            epochs           = epochs,
            steps_per_epoch  = steps_per_epoch,
            validation_steps = validation_steps,
            verbose          = 1,
            callbacks        = callbacks,
        )

        best_val_loss  = min(history.history["val_loss"])
        epochs_trained = len(history.history["val_loss"])

        # Loss-history plot
        plt.figure(figsize=(8, 4))
        plt.plot(history.history["loss"],     label="train")
        plt.plot(history.history["val_loss"], label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(f"Loss history — lambda_fft = {lam_fft:.9e}", fontsize=12)
        plt.legend()
        plt.tight_layout()
        loss_hist_path = os.path.join(out_dir, "loss_history.png")
        plt.savefig(loss_hist_path, dpi=300)
        plt.close()
        print(f"  Saved loss history -> {loss_hist_path}")

        # Evaluate on the test set
        metrics = _evaluate_test_set(model, datasets["test_ds"])

        print(f"  Test MSE  = {metrics['mse_mean']:.8f}")
        print(f"  Test PSNR = {metrics['psnr_mean']:.3f} dB")
        print(f"  Test SSIM = {metrics['ssim_mean']:.4f}")

        # Append to sweep CSV
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(),
                round(float(lam_fft),         9),
                round(best_val_loss,           9),
                round(metrics["mse_mean"],     5),
                round(metrics["psnr_mean"],    5),
                round(metrics["ssim_mean"],    5),
                int(epochs_trained),
            ])
        print(f"  Saved metrics row --> {csv_path}")

        # Comparison figures of the last test batch
        if metrics["last_pred"] is not None:
            pred_unscaled = metrics["last_pred"] * (hr_maxs - hr_mins) + hr_mins
            hr_unscaled   = (metrics["last_hr_batch"].numpy()
                             * (hr_maxs - hr_mins) + hr_mins)
            lr_unscaled   = (metrics["last_lr_batch"].numpy()
                             * (lr_maxs - lr_mins) + lr_mins)

            plot_comparison_transformer(
                lr_unscaled,
                pred_unscaled,
                hr_unscaled,
                ssim_mean_u = metrics["ssim_mean_u"],
                ssim_std_u  = metrics["ssim_std_u"],
                ssim_mean_v = metrics["ssim_mean_v"],
                ssim_std_v  = metrics["ssim_std_v"],
                lat_hr      = lat_hr,
                lon_hr      = lon_hr,
                lat_lr      = lat_lr,
                lon_lr      = lon_lr,
                index       = 0,
                results_dir = out_dir,
            )

        # Free memory between runs
        del model, optimizer, criterion
        tf.keras.backend.clear_session()
        gc.collect()

    # Summary sweep plot
    _plot_sweep_summary(csv_path, results_dir,
                        x_col="lambda_fft", sweep_label="FFT lambda")


def run_fft_lambda_sweep(
    build_model_fn,
    datasets,
    minmax_stats,
    results_dir,
    n_steps      = 10,
    lambda_min   = 1e-7,
    lambda_max   = 1e-5,
    epochs       = 30,
    lr_rate      = 1e-5,
    patience     = 10,
    min_delta    = 1e-7,
    lat_hr=None, lon_hr=None,
    lat_lr=None, lon_lr=None,
    batch_size       = 2,
    validation_steps = 2807,
    case             = "1year",
):
    """Same sweep as run_fft_parameter_sweep with the older argument order.

    A build_model_fn without argument is accepted and called without lambda_fft.
    """
    import inspect
    sig = inspect.signature(build_model_fn)
    n_params = len(sig.parameters)

    if n_params == 0:
        _orig_fn = build_model_fn
        def _adapted_fn(lam):
            """Call the zero-argument model builder, ignoring lambda_fft."""
            return _orig_fn()
        adapted_fn = _adapted_fn
    else:
        adapted_fn = build_model_fn

    run_fft_parameter_sweep(
        build_model_fn   = adapted_fn,
        datasets         = datasets,
        minmax_stats     = minmax_stats,
        results_dir      = results_dir,
        case             = case,
        n_steps          = n_steps,
        lambda_min       = lambda_min,
        lambda_max       = lambda_max,
        epochs           = epochs,
        lr_rate          = lr_rate,
        patience         = patience,
        min_delta        = min_delta,
        batch_size       = batch_size,
        validation_steps = validation_steps,
        lat_hr           = lat_hr,
        lon_hr           = lon_hr,
        lat_lr           = lat_lr,
        lon_lr           = lon_lr,
    )


def plot_comparison_param(
    lr_unscaled:  np.ndarray,
    sr_unscaled:  np.ndarray,
    hr_unscaled:  np.ndarray,
    ssim_mean_u=None, ssim_std_u=None,
    ssim_mean_v=None, ssim_std_v=None,
    run_tag:    str = "",
    results_dir:str = "",
    lat_hr=None, lon_hr=None,
    lat_lr=None, lon_lr=None,
    index: int = 0,
    model_name: str = "Swin2SR",
):
    """Save the arrays and the comparison figures of one sample of a sweep run.

    Writes plot_arrays_<run_tag>_<index>.npz and the figures of plot_comparison_transformer in
    <results_dir>/param_loss_function/<run_tag>.

    Args:
        lr_unscaled, sr_unscaled, hr_unscaled: arrays (N, H, W, 2) of inputs, predictions and targets.
        ssim_mean_u, ssim_std_u, ssim_mean_v, ssim_std_v: spatial SSIM maps.
        run_tag: name of the run folder.
        results_dir: base output directory.
        lat_hr, lon_hr, lat_lr, lon_lr: coordinate vectors of the HR and LR grids.
        index: sample index.
        model_name: unused.
    """
    out_dir = os.path.join(results_dir, "param_loss_function", run_tag)
    os.makedirs(out_dir, exist_ok=True)

    # Save raw arrays for post-processing
    npz_path = os.path.join(out_dir, f"plot_arrays_{run_tag}_{index:03d}.npz")
    np.savez_compressed(
        npz_path,
        lr_img=lr_unscaled[index], sr_img=sr_unscaled[index],
        hr_img=hr_unscaled[index],
        lat_hr=lat_hr, lon_hr=lon_hr,
        lat_lr=lat_lr, lon_lr=lon_lr,
    )
    print(f"[plot_comparison_param] Saved arrays -> {npz_path}")

    # Delegate the figures to plot_comparison_transformer
    plot_comparison_transformer(
        lr_unscaled,
        sr_unscaled,
        hr_unscaled,
        ssim_mean_u = ssim_mean_u,
        ssim_std_u  = ssim_std_u,
        ssim_mean_v = ssim_mean_v,
        ssim_std_v  = ssim_std_v,
        lat_hr      = lat_hr,
        lon_hr      = lon_hr,
        lat_lr      = lat_lr,
        lon_lr      = lon_lr,
        index       = index,
        results_dir = out_dir,
    )


def _plot_sweep_summary(csv_path: str, results_dir: str,
                        x_col: str, sweep_label: str):
    """Plot PSNR, SSIM and MSE against the swept coefficient from the sweep CSV.

    Saves sweep_summary_<x_col>.png in results_dir.

    Args:
        csv_path: sweep CSV file.
        results_dir: output directory.
        x_col: CSV column of the swept coefficient.
        sweep_label: label of the x axis.
    """
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[_plot_sweep_summary] Could not read {csv_path}: {e}")
        return

    if x_col not in df.columns:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [
        ("test_psnr",       "PSNR (dB)",   "#1f77b4"),
        ("test_ssim",       "SSIM",         "#2ca02c"),
        ("test_mse_scaled", "MSE (scaled)", "#d62728"),
    ]
    for ax, (col, ylabel, color) in zip(axes, metrics):
        if col not in df.columns:
            continue
        ax.plot(df[x_col], df[col], "o-", color=color,
                linewidth=1.5, markersize=6)
        ax.set_xlabel(sweep_label, fontsize=12)
        ax.set_ylabel(ylabel,      fontsize=12)
        ax.set_title(f"{ylabel} vs {sweep_label}", fontsize=12)
        ax.grid(True, alpha=0.4)

    fig.suptitle(f"Parametric study — {sweep_label}", fontsize=14)
    plt.tight_layout()
    out = os.path.join(results_dir, f"sweep_summary_{x_col}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[_plot_sweep_summary] Saved sweep summary -> {out}")


if __name__ == "__main__":
    print("SpectralLoss_Swin2SR_tensorflow — smoke test")
    print("-" * 40)

    B, H_hr, W_hr = 2, 320, 400

    pred_dummy = tf.random.uniform([B, H_hr, W_hr, 1])
    true_dummy = tf.random.uniform([B, H_hr, W_hr, 1])

    for lam in [0.0, 0.01, 0.1, 1.0, 10.0]:
        criterion = SpectralSRLoss(lambda_fft=lam)
        total, mse, fft, wav = criterion(pred_dummy, true_dummy)
        print(f"  lambda_fft={lam:7.1f}  total={float(total):.6f}  "
              f"mse={float(mse):.6f}  fft={float(fft):.6f}")

    try:
        from Swin2SR_tensorflow import build_swin2sr
        model_test = build_swin2sr(embed_dim=48, depths=(2, 2))
        opt_test   = tf.keras.optimizers.Adam(learning_rate=1e-4)
        crit_test  = SpectralSRLoss(lambda_fft=1.0)
        lr_dummy   = np.random.rand(B, 64, 80, 1).astype(np.float32)
        hr_dummy   = np.random.rand(B, H_hr, W_hr, 1).astype(np.float32)

        loss_dict = train_step_spectral(
            model_test, opt_test, lr_dummy, hr_dummy, crit_test)
        print(f"\n  train_step_spectral loss dict: {loss_dict}")
    except ImportError:
        print("\n  Swin2SR_tensorflow not importable — skipping step 2.")

    tracker = LossTermTracker(".", run_tag="smoke_test")
    for _ in range(5):
        tracker.update({"total": np.random.rand(), "mse": np.random.rand(),
                        "fft": np.random.rand() * 10, "wav": 0.0})
    tracker.plot()
    tracker.save_csv()

    cb = ResetTrainLoss()
    print("\n  ResetTrainLoss callback created (no-op if model has no loss_tracker).")

    results_dir = "specify the folder in which the results will appear"
    csv_path   = os.path.join(results_dir, "param_sweep_fft.csv")
    _plot_sweep_summary(csv_path, results_dir, x_col="lambda_fft",
                        sweep_label="FFT lambda")
    print("\nSmoke test complete.")
