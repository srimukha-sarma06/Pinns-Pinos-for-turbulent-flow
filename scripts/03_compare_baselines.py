"""Run the conventional baselines (time-of-flight, MOC grid-search + LM polish) across the
hackathon's test cases (clean / noisy / small-leak / location sweep). These are the working,
validated leak locators in this repo right now -- use them as the reference the PINN is judged
against once its inverse-training convergence (see leakpinn/pinn.py + README 'Known limitation')
is finished.

Run: python scripts/03_compare_baselines.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from leakpinn.synth import make_dataset
from leakpinn.baselines import time_of_flight, moc_search, estimate_wave_speed

CASES = [
    dict(name="clean, mid-pipe leak",      leak_x=37.3, CdA=3.6e-6, noise_std=0.02, seed=0),
    dict(name="realistic sensor noise",     leak_x=37.3, CdA=3.6e-6, noise_std=0.10, seed=1),
    dict(name="small leak (1% of Q0)",      leak_x=37.3, CdA=0.9e-6, noise_std=0.10, seed=2),
    dict(name="leak near reservoir",        leak_x=15.0, CdA=3.6e-6, noise_std=0.10, seed=3),
    dict(name="leak near valve",            leak_x=85.0, CdA=3.6e-6, noise_std=0.10, seed=4),
    dict(name="unknown wave speed (PVC-ish)", leak_x=50.0, CdA=3.6e-6, a_true=420.0, noise_std=0.10, seed=5),
]

results = []
for c in CASES:
    kw = {k: v for k, v in c.items() if k != "name"}
    d = make_dataset(**kw)
    print(f"\n=== {c['name']} ===  truth: x_L={d.truth['leak_x']:.1f} m, "
          f"CdA={d.truth['CdA']:.2e} m^2, a={d.truth['a']:.0f} m/s")

    t0 = time.time()
    tof = time_of_flight(d)
    tof_t = time.time() - t0
    print(f"  time-of-flight : x_L={tof['x_L']:6.1f} m  CdA={tof['CdA']:.2e}  a={tof['a']:6.0f}  "
          f"({tof_t:.2f} s)   err = {tof['x_L']-d.truth['leak_x']:+.1f} m")

    t0 = time.time()
    moc = moc_search(d)
    print(f"  MOC search     : x_L={moc['x_L']:6.1f} m  CdA={moc['CdA']:.2e}  a={moc['a']:6.0f}  "
          f"({moc['seconds']:.1f} s)   err = {moc['x_L']-d.truth['leak_x']:+.1f} m")

    results.append(dict(case=c["name"], truth=dict(x_L=d.truth["leak_x"], CdA=d.truth["CdA"], a=d.truth["a"]),
                        time_of_flight=tof, moc_search={k: v for k, v in moc.items() if k != "cost"}))

os.makedirs("results", exist_ok=True)
json.dump(results, open("results/03_baseline_comparison.json", "w"), indent=2)
print("\nSaved results/03_baseline_comparison.json")
