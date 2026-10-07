"""SPH file helpers shared by sph_dedup.py and prepare_sph.py."""
import os


def record_path(records_dir, ecg_id):
    """SPH metadata lists IDs without extension (e.g. 'A00001'); files are 'A00001.h5'.
    Accept either form. Returns None if no file exists."""
    for name in (ecg_id, f"{ecg_id}.h5"):
        p = os.path.join(records_dir, name)
        if os.path.isfile(p):
            return p
    return None


def resolve_all(records_dir, ecg_ids):
    """Map every ID to its file; raise listing the first misses if any are missing."""
    paths, missing = {}, []
    for e in ecg_ids:
        p = record_path(records_dir, e)
        (missing.append(e) if p is None else paths.__setitem__(e, p))
    if missing:
        present = sorted(os.listdir(records_dir))[:3] if os.path.isdir(records_dir) else "folder missing"
        raise FileNotFoundError(
            f"{len(missing)} of {len(ecg_ids)} SPH records not found in {records_dir}, e.g. {missing[:3]}; "
            f"files present look like {present}")
    return paths
