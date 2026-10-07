"""End-to-end CGIS: co-train controller + barrier, certify, save the proof.

Usage:
    python scripts/train_and_certify.py                       # inverted pendulum
    python scripts/train_and_certify.py system=dubins_car     # 4-D Dubins car
    python scripts/train_and_certify.py gamma=2.0 train.steps_initial=4000
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.dynamics.continuous_system import make_system  # noqa: E402
from src.verification.cgis_loop import CGISTrainer, TrainConfig, VerifyConfig  # noqa: E402
from src.verification.proof_checker import ProofChecker, load_certificate  # noqa: E402


def _dc(cls, cfg):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in cfg.items() if k in names})


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    c = OmegaConf.to_container(cfg, resolve=True)
    out_dir = ROOT / c["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(OmegaConf.to_yaml(cfg))

    system = make_system(c["system"])
    trainer = CGISTrainer(system, c["gamma"], _dc(TrainConfig, c["train"]), _dc(VerifyConfig, c["verify"]),
                          tuple(c["model"]["ctrl_hidden"]), tuple(c["model"]["barrier_hidden"]), seed=c["seed"])
    t0 = time.time()
    res = trainer.run(max_rounds=c["max_rounds"], checkpoint=out_dir / "checkpoint.pt", resume=c.get("resume", False))
    print(f"\n[result] {res.status} after {len(trainer.history)} CGIS round(s), {time.time() - t0:.1f}s")
    trainer.save(out_dir, res, c)

    # independent re-check of the stored proof
    cert, payload = load_certificate(out_dir / "certificate.pt")
    report = ProofChecker(cert, tol=c["verify"]["tol"]).check(payload["verification"])
    (out_dir / "proof_check.json").write_text(json.dumps({"ok": report.ok, "seconds": report.seconds,
                                                          "details": report.details}, indent=2))
    print(f"[proof checker] {'ACCEPTED' if report.ok else 'REJECTED'} ({report.seconds:.1f}s)")
    for k, v in report.details.items():
        print(f"   {k}: ok={v.get('ok')} leaves={v.get('leaves')} failed={v.get('failed_leaves')}")


if __name__ == "__main__":
    main()
