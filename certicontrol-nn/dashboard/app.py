"""CertiControl-NN: interactive Safety Verification Inspector.

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402

from src.models.certificate import CertificateModule  # noqa: E402
from src.verification.lirpa_verifier import METHOD_CODES, BranchAndBoundVerifier  # noqa: E402
from src.verification.proof_checker import load_certificate  # noqa: E402
from src.visualization.phase_portrait import (plot_dubins_slice, plot_dubins_trajectories,  # noqa: E402
                                              plot_pendulum, simulate)

st.set_page_config(page_title="CertiControl-NN Inspector", layout="wide")
st.title("CertiControl-NN: Safety Verification Inspector")
st.caption("Neural controller + neural control barrier function, certified by "
           "IBP / auto_LiRPA (CROWN) branch-and-bound with Z3 counterexample extraction.")

RESULTS = ROOT / "results"
available = sorted({p.parent.name for p in list(RESULTS.glob("*/certificate.pt")) + list(RESULTS.glob("*/certificate.pt.gz"))})
if not available:
    st.error("No certificates found. Run `python scripts/train_and_certify.py` first.")
    st.stop()


@st.cache_resource(show_spinner=False)
def _load(name: str):
    cert, payload = load_certificate(RESULTS / name / "certificate.pt")
    check = RESULTS / name / "proof_check.json"
    meta = RESULTS / name / "certificate.json"
    return cert.double(), payload, (json.loads(check.read_text()) if check.exists() else None), \
        (json.loads(meta.read_text()) if meta.exists() else None)


# ------------------------------------------------------------------ sidebar
system_name = st.sidebar.selectbox("System", available)
cert, payload, check, meta = _load(system_name)
system = cert.system
summ = payload["verification"]["summary"]

st.sidebar.markdown("### Re-verification")
gamma_val = st.sidebar.slider("Barrier alpha parameter  gamma", 0.1, 5.0, float(payload["gamma"]), 0.1,
                              help="alpha(B) = gamma * B.  For B >= 0 a larger gamma only makes the "
                                   "condition easier, so the stored proof also covers every gamma >= its own.")
depth = st.sidebar.slider("Max bisection depth (resolution)", 4, 14, int(payload["config"]["verify"]["max_depth"]))
use_z3 = st.sidebar.checkbox("Use Z3 on unresolved boxes", True)
st.sidebar.markdown("### Display")
show_leaves = st.sidebar.checkbox("Show branch-and-bound leaves", True)
n_traj = st.sidebar.slider("Simulated rollouts", 0, 40, 12)

# ------------------------------------------------------------------ status
c1, c2, c3, c4 = st.columns(4)
status = summ["status"]
c1.metric("Certificate status", status)
c2.metric("Domain leaves proven", f"{summ['domain_leaves']:,}")
c3.metric("Z3-proven leaves", summ["leaves_by_method"].get("z3", 0))
c4.metric("Proof checker", "ACCEPTED" if (check and check["ok"]) else ("REJECTED" if check else "n/a"))
if status == "CERTIFIED":
    st.success("All properties P0-P3 are proven for **every** state in the continuous domain: "
               "C = {B >= 0} is forward invariant under the neural controller, avoids the unsafe set, "
               "stays inside D, and contains the initial set X0.")
else:
    st.warning(f"The stored result is {status}: see counterexamples / unknown boxes below.")

with st.expander("Certificate details (weights hashes, B&B statistics, CGIS history)"):
    if meta:
        st.json({k: meta[k] for k in ["system", "status", "controller_sha256", "barrier_sha256", "properties"]})
    st.json(summ)
    hist = [h for h in payload.get("history", []) if isinstance(h.get("round"), int)]
    if hist:
        fig, ax = plt.subplots(figsize=(6, 2.4))
        ax.plot([h["round"] for h in hist], [h["counterexamples"] for h in hist], "o-", label="counterexamples")
        ax.plot([h["round"] for h in hist], [h["unknown_boxes"] for h in hist], "s--", label="unknown boxes")
        ax.set_xlabel("CGIS round")
        ax.set_yscale("symlog")
        ax.legend()
        st.pyplot(fig)
    if check:
        st.json(check["details"])

# ---------------------------------------------------------- phase portrait
st.subheader("Certified safe state-space domain")
if system.name == "inverted_pendulum":
    fig, ax = plt.subplots(figsize=(7, 5.4))
    _, handles = plot_pendulum(cert, payload["verification"] if show_leaves else None,
                               trajectories=n_traj, ax=ax, show_leaves=show_leaves)
    ax.legend(handles=handles, fontsize=7, loc="upper right")
    st.pyplot(fig)
else:
    cc1, cc2 = st.columns(2)
    theta = cc1.slider("Heading slice theta [rad]", -3.14, 3.14, 0.0, 0.05)
    v = cc2.slider("Speed slice v", float(system.domain.lo[3]), float(system.domain.hi[3]), 0.0, 0.05)
    col_a, col_b = st.columns(2)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    plot_dubins_slice(cert, theta, v, ax=ax, verification=payload["verification"] if show_leaves else None)
    col_a.pyplot(fig)
    fig2, ax2 = plt.subplots(figsize=(5.5, 5.5))
    if n_traj:
        plot_dubins_trajectories(cert, n=n_traj, ax=ax2)
    col_b.pyplot(fig2)

# ------------------------------------------------------------- simulation
st.subheader("Simulate a trajectory from a custom initial state")
cols = st.columns(system.state_dim + 1)
lo, hi = system.domain.lo, system.domain.hi
x0 = [cols[i].number_input(system.state_names[i], float(lo[i]), float(hi[i]),
                           float(system.init_set.lo[i] * 0.5 + system.init_set.hi[i] * 0.5 + 0.1 * (hi[i] - lo[i]) * (i == 0)),
                           format="%.3f") for i in range(system.state_dim)]
T = cols[-1].number_input("horizon T [s]", 1.0, 30.0, 8.0)
t, xs = simulate(cert, x0, T=T)
with torch.no_grad():
    out = cert(torch.from_numpy(xs))
B_t = out[:, 0].numpy()
fig, axes = plt.subplots(1, 2, figsize=(11, 3))
axes[0].plot(t, B_t)
axes[0].axhline(0, color="k", lw=0.8)
axes[0].set_xlabel("t [s]")
axes[0].set_ylabel("B(x(t))")
for i in range(system.state_dim):
    axes[1].plot(t, xs[:, i], label=system.state_names[i])
axes[1].legend(fontsize=8)
axes[1].set_xlabel("t [s]")
st.pyplot(fig)
unsafe_hit = bool(system.is_unsafe(torch.from_numpy(xs)).any())
if B_t[0] >= 0:
    st.info(f"x0 lies in the certified set (B = {B_t[0]:.4f}); min B along the rollout = {B_t.min():.4f}; "
            f"unsafe set reached: {unsafe_hit}.")
else:
    st.info(f"x0 is outside the certified set (B = {B_t[0]:.4f}); no guarantee applies. "
            f"Unsafe set reached: {unsafe_hit}.")

# ---------------------------------------------------------- re-verification
st.subheader("Re-run the formal verifier")
st.write("Runs IBP -> CROWN -> Z3 branch-and-bound on the stored networks with the parameters from the sidebar.")
if st.button("Verify now"):
    test_cert = CertificateModule(system, cert.controller, cert.barrier, gamma_val)
    ver = BranchAndBoundVerifier(test_cert, use_z3=use_z3, max_depth=depth,
                                 z3_depth=min(depth, int(payload["config"]["verify"]["z3_depth"])),
                                 max_boxes=int(payload["config"]["verify"]["max_boxes"]))
    t0 = time.time()
    with st.spinner("verifying ..."):
        res = ver.verify()
    s2 = res.summary()
    (st.success if res.certified else st.error)(f"{s2['status']} with gamma = {gamma_val} in {time.time() - t0:.1f}s")
    st.json(s2)
    cex = res.counterexamples()
    if cex.shape[0]:
        st.write("First counterexamples (state, B, dB/dt + gamma B):")
        with torch.no_grad():
            vals = test_cert.double()(cex[:10])
        st.dataframe(np.concatenate([cex[:10].numpy(), vals.numpy()], 1))
