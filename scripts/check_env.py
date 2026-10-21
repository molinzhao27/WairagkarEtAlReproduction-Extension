"""Environment check for the brain-to-voice reproduction (macOS CPU dev env and Linux CUDA training env).

Builds the vendored, unmodified Wairagkar et al. 2025 model from its own config with its own functions, runs a
forward pass (eager and graph mode) and a one-step loss gradient on CPU and, if present, on GPU (CUDA or Metal),
compares the devices, and smoke-tests the LPCNet binaries. Exits nonzero on any failure.

    mamba run -n wvx python scripts/check_env.py           # Mac dev env
    TF_USE_LEGACY_KERAS=1 .venv/bin/python scripts/check_env.py   # GPU-host venv
"""

import sys

sys.dont_write_bytecode = True  # never write __pycache__ into the vendored third_party tree

import contextlib
import io
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Their code uses `tensorflow.keras` with the Keras 2 API; on TF >= 2.16 that needs tf-keras + this flag.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

BATCH = 64
REL_TOL = 1e-4  # max|GPU - CPU| / max|CPU| allowed for outputs and gradients (TF32 disabled for the comparison)

R = Path(__file__).resolve().parents[1]
B2V = R / "third_party" / "brain_to_voice"
LPCNET_DIR = R / "third_party" / "LPCNet"
sys.path.insert(0, str(B2V))
sys.path.insert(0, str(R / "src"))

from wvx.config import load_config  # noqa: E402
from wvx.seeding import set_seed  # noqa: E402

SEED = int(load_config()["seed"])  # configs/default.yaml (AGENTS.md rule 7)
set_seed(SEED)

failures: list[str] = []


def check(ok: bool, what: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {what}")
    if not ok:
        failures.append(what)
    return ok


def section(title: str) -> None:
    print(f"\n== {title}")


# ------------------------------------------------------------------ versions and devices
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import yaml  # noqa: E402

section("versions")
print(f"  python      {platform.python_version()} ({sys.executable})")
print(f"  platform    {platform.platform()} {platform.machine()}")
print(f"  tensorflow  {tf.__version__}  (built with CUDA: {tf.test.is_built_with_cuda()})")
keras2_version = getattr(tf.keras, "__version__", None) or getattr(sys.modules.get("tf_keras"), "__version__", "?")
print(f"  tf.keras    {keras2_version}  (module {tf.keras.__name__}; TF_USE_LEGACY_KERAS={os.environ.get('TF_USE_LEGACY_KERAS')})")
try:
    import keras as _standalone_keras

    print(f"  keras       {_standalone_keras.__version__}  (standalone `import keras`)")
    if int(_standalone_keras.__version__.split(".")[0]) >= 3:
        print("              note: lib/inference_utils.py does `from keras.models import load_model`; alias")
        print("              sys.modules['keras'] = tf_keras (and 'keras.models') before importing it.")
except ImportError:
    print("  keras       (not installed)")
print(f"  numpy       {np.__version__}")
check(str(keras2_version).startswith("2."), f"tf.keras is the Keras 2 API (got {keras2_version})")

section("devices")
cpus = tf.config.list_physical_devices("CPU")
gpus = tf.config.list_physical_devices("GPU")
for d in cpus + gpus:
    details = tf.config.experimental.get_device_details(d) if d.device_type == "GPU" else {}
    extra = f"  {details}" if details else ""
    print(f"  {d.name}  ({d.device_type}){extra}")
if not gpus:
    print("  no GPU visible: CPU-only run")

# ------------------------------------------------------------------ build their model from their config
section("model (vendored models/brain2voice.py via lib/model_utils.build_model)")
from lib import dataloader, model_utils, speech_processor  # noqa: E402

with open(B2V / "configs" / "config.yaml") as f:
    cfg = yaml.safe_load(f)
_, session_layers = dataloader.get_sessions(str(B2V / cfg["paths"]["training_data"]))
n_sessions = session_layers[-1] + 1
time_steps = cfg["data"]["time_steps"]  # 60 bins of 10 ms; the model avg-pools x2 -> transformer_input_shape[0]
n_channels = 512                          # 256 spikepow + 256 threshcross; hard-coded in day_embedding_model((30, 512))
out_dim = speech_processor._N_CEPS + speech_processor._N_PITCH  # 18 cepstra + 2 pitch = 20 decoder targets
assert time_steps // 2 == cfg["model"]["transformer_input_shape"][0]

tf.keras.utils.set_random_seed(SEED)
rng = np.random.default_rng(SEED)
X = rng.standard_normal((BATCH, time_steps, n_channels)).astype("float32")
Y = rng.uniform(-1, 1, (BATCH, out_dim)).astype("float32")
S = np.full((BATCH, 1), n_sessions - 1, dtype="int32")
print(f"  input X {X.shape} float32, session {S.shape} int32 (session layer {n_sessions - 1} of {n_sessions}); target {Y.shape}")

t0 = time.time()
with contextlib.redirect_stdout(io.StringIO()):  # build_model prints 22 model summaries
    model = model_utils.build_model(cfg, X, Y, S, n_sessions)
print(f"  built in {time.time() - t0:.1f} s: {model.count_params():,} parameters, strategy {type(model.distribute_strategy).__name__}")
huber = tf.keras.losses.Huber(delta=cfg["training"]["huber_delta"])
train_vars = model.base_transformer.trainable_variables + model.day_embedding[n_sessions - 1].trainable_variables


def run_on(device: str) -> dict:
    """Eager forward, graph-mode forward and graph-mode loss gradient, all placed on `device`."""
    with tf.device(device):
        @tf.function
        def fwd(x, s):
            return model([x, s], training=False)

        @tf.function
        def grad(x, s, y):
            with tf.GradientTape() as tape:
                loss = huber(y, model([x, s], training=False))
            return tape.gradient(loss, train_vars)

        out = {"eager": model([X, S], training=False).numpy(),
               "graph": fwd(tf.constant(X), tf.constant(S)).numpy()}
        out["grad"] = np.concatenate([g.numpy().ravel() for g in grad(tf.constant(X), tf.constant(S), tf.constant(Y))])
    return out


section("forward pass")
tf.config.experimental.enable_tensor_float_32_execution(False)  # strict fp32 for the CPU/GPU comparison
ref = run_on("/CPU:0")
print(f"  CPU output shape {ref['eager'].shape}, |out| max {np.abs(ref['eager']).max():.4f}")
check(ref["eager"].shape == (BATCH, out_dim), f"output shape is (batch, {out_dim})")
check(bool(np.isfinite(ref["eager"]).all() and np.isfinite(ref["grad"]).all()), "CPU outputs and gradients are finite")
d_eg = float(np.abs(ref["eager"] - ref["graph"]).max())
check(d_eg < 1e-5, f"CPU eager vs graph max abs diff {d_eg:.2e}")

for i, _ in enumerate(gpus):
    dev = f"/GPU:{i}"
    t0 = time.time()
    res = run_on(dev)
    print(f"  {dev} output shape {res['eager'].shape} ({time.time() - t0:.1f} s incl. tracing / kernel JIT)")
    for k in ("eager", "graph", "grad"):
        d = float(np.abs(res[k] - ref[k]).max())
        rel = d / max(float(np.abs(ref[k]).max()), 1e-12)
        check(rel < REL_TOL, f"{dev} vs CPU [{k:5s}] max abs diff {d:.3e} (relative {rel:.2e}, tol {REL_TOL:g})")
if not gpus:
    print("  CPU-vs-GPU max abs diff: n/a (no GPU)")

# ------------------------------------------------------------------ LPCNet
section(f"LPCNet ({LPCNET_DIR.relative_to(R)})")
their_path = B2V / cfg["paths"]["lpcnet_path"]
print(f"  note: their config lpcnet_path resolves to {their_path.relative_to(R)} "
      f"({'exists' if their_path.exists() else 'absent; pass our path explicitly'})")
bins_ok = True
for exe in ("lpcnet_demo", "dump_data"):
    p = LPCNET_DIR / exe
    bins_ok &= check(p.is_file() and os.access(p, os.X_OK), f"{p.relative_to(R)} exists and is executable")
if bins_ok:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fs, n = 16000, 16000  # 1 s, 16 kHz, 16-bit mono raw PCM: the format their speech_processor feeds LPCNet
        t = np.arange(n) / fs
        sig = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 20)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
        sig = sig + 0.01 * rng.standard_normal(n)
        (0.3 * 32767 * sig / np.abs(sig).max()).astype("<i2").tofile(tmp / "in.pcm")
        demo = str(LPCNET_DIR / "lpcnet_demo")
        r1 = subprocess.run([demo, "-features", str(tmp / "in.pcm"), str(tmp / "f.f32")], capture_output=True)
        feats = np.fromfile(tmp / "f.f32", dtype=np.float32) if (tmp / "f.f32").exists() else np.zeros(0)
        ok = check(r1.returncode == 0 and feats.size > 0 and feats.size % 36 == 0,
                   f"lpcnet_demo -features -> {feats.size // 36} frames x 36 (exit {r1.returncode})")
        if ok:
            r2 = subprocess.run([demo, "-synthesis", str(tmp / "f.f32"), str(tmp / "out.pcm")], capture_output=True)
            out = np.fromfile(tmp / "out.pcm", dtype="<i2") if (tmp / "out.pcm").exists() else np.zeros(0)
            rms = float(np.sqrt(np.mean((out / 32768.0) ** 2))) if out.size else 0.0
            check(r2.returncode == 0 and out.size == (feats.size // 36) * 160 and rms > 1e-3,
                  f"lpcnet_demo -synthesis -> {out.size} samples ({out.size / fs:.2f} s), RMS {rms:.4f} (exit {r2.returncode})")

# ------------------------------------------------------------------ summary
section("summary")
if failures:
    print(f"  {len(failures)} check(s) FAILED:")
    for f_ in failures:
        print(f"   - {f_}")
    sys.exit(1)
print("  all checks passed")
