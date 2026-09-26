"""PTB-XL multi-hot labels over the full statement set, and E3 shared labels."""
import ast
import numpy as np
import pandas as pd

# E3 primary shared labels -> constituent PTB-XL statements (E3_protocol_v1.md).
# Merged labels are scored as the max of constituent outputs (D4, D9).
E3_PRIMARY = {
    "AF": ["AFIB"], "PR_PROL": ["1AVB", "LPR"], "CRBBB": ["CRBBB"], "IRBBB": ["IRBBB"],
    "CLBBB": ["CLBBB"], "LAFB": ["LAFB"], "LVH": ["LVH"], "IMI": ["IMI"], "ASMI": ["ASMI"],
}
E3_GROUPS = {"criteria": ["AF", "PR_PROL", "CRBBB", "IRBBB", "CLBBB", "LAFB"],
             "interpretive": ["LVH", "IMI", "ASMI"]}
# SPH AHA primary codes for the same labels (E3 protocol table)
E3_SPH_CODES = {"AF": {"50"}, "PR_PROL": {"82"}, "CRBBB": {"106"}, "IRBBB": {"105"}, "CLBBB": {"104"},
                "LAFB": {"101"}, "LVH": {"142"}, "IMI": {"161"}, "ASMI": {"165"}}


def statement_list(scp):
    """All statements in scp_statements.csv file order (71 in PTB-XL v1.0.3)."""
    return list(scp.index)


def ptbxl_multihot(meta, scp, min_likelihood=0.0):
    """(N, S) float32. Threshold applies to diagnostic statements only (D11)."""
    stmts = statement_list(scp)
    col = {s: i for i, s in enumerate(stmts)}
    diag = set(scp.index[scp["diagnostic"] == 1])
    Y = np.zeros((len(meta), len(stmts)), dtype=np.float32)
    for r, codes in enumerate(meta["scp_codes"]):
        d = ast.literal_eval(codes) if isinstance(codes, str) else codes
        for c, lik in d.items():
            if c in col and (c not in diag or lik >= min_likelihood):
                Y[r, col[c]] = 1.0
    return Y, stmts


def e3_scores(P, stmts):
    """Collapse full-statement probabilities P (N, S) to E3 shared-label scores (N, 9) by max."""
    col = {s: i for i, s in enumerate(stmts)}
    return np.stack([P[:, [col[s] for s in cons]].max(axis=1) for cons in E3_PRIMARY.values()], axis=1)


def sph_primary(aha_field):
    return {p.strip() for stmt in str(aha_field).split(";") for p in stmt.split("+")
            if p.strip().isdigit() and int(p) < 300}


def sph_e3_labels(sph_meta):
    prim = sph_meta["AHA_Code"].apply(sph_primary)
    return np.stack([prim.apply(lambda s: int(bool(s & codes))).to_numpy()
                     for codes in E3_SPH_CODES.values()], axis=1).astype(np.float32)
