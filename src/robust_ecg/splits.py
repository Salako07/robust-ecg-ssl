"""Patient-level nested label budgets (decision D10; E3 protocol)."""
import numpy as np

PRIMARY_BUDGETS = (0.05, 0.10, 0.25, 0.50, 1.0)
EXPLORATORY_BUDGETS = (0.01,)


def nested_patient_budgets(patients, seed, budgets=EXPLORATORY_BUDGETS + PRIMARY_BUDGETS):
    """One random patient order per seed; budget b = first ceil(b*N) patients, so budgets are nested."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.asarray(sorted(set(patients))))
    return {b: set(order[: int(np.ceil(b * len(order)))]) for b in budgets}


def budget_indices(meta, budget, seed, train_folds=range(1, 9)):
    """Row positions (into `meta`) of PTB-XL training records for a budget/seed."""
    train = meta["strat_fold"].isin(list(train_folds)).to_numpy()
    pats = nested_patient_budgets(meta.loc[train, "patient_id"], seed, budgets=(budget,))[budget]
    return np.flatnonzero(train & meta["patient_id"].isin(pats).to_numpy())
