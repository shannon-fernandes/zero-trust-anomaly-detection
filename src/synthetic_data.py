"""
synthetic_data.py - FAKE data shaped like UNSW-NB15, for TESTING THE CODE ONLY.

!! Results from this data are NOT real experimental results and must never be
!! reported as such. It exists so tests and demos run without the real dataset.
"""
import numpy as np
import pandas as pd

import config

SYNTHETIC_NOTICE = "SYNTHETIC DEMO DATA - NOT REAL RESULTS"
CATS = ["Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance"]


def make_synthetic_unsw(n_rows=2000, attack_fraction=0.5, seed=None):
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    label = (rng.random(n_rows) < attack_fraction).astype(int)
    a = label == 1

    def pick(normal_lam, attack_lam):
        return rng.poisson(np.where(a, attack_lam, normal_lam)).astype(float)

    df = pd.DataFrame({
        "id": np.arange(1, n_rows + 1),
        "dur": rng.exponential(np.where(a, 0.5, 1.0)),
        "proto": rng.choice(["tcp", "udp", "icmp"], n_rows),
        "service": rng.choice(["-", "http", "dns", "ftp"], n_rows),
        "state": rng.choice(["FIN", "CON", "INT"], n_rows),
        "spkts": pick(10, 30), "dpkts": pick(10, 3),
        "sbytes": pick(500, 3000), "dbytes": pick(2000, 200),
        "rate": rng.exponential(np.where(a, 8000, 2000)),
        "sttl": rng.choice([31, 62, 254], n_rows),
        "sload": rng.exponential(np.where(a, 3e6, 1e6)),
        "ct_srv_src": pick(5, 15), "ct_dst_ltm": pick(4, 12),
        "ct_src_dport_ltm": pick(2, 7), "ct_dst_sport_ltm": pick(1, 4),
        "ct_dst_src_ltm": pick(4, 14), "ct_src_ltm": pick(4, 11),
        "ct_srv_dst": pick(5, 14),
    })
    df["attack_cat"] = np.where(a, rng.choice(CATS, n_rows), "Normal")
    df["label"] = label
    return df
