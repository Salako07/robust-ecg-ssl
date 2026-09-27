"""Helpers for Colab sessions. Kept free of google.colab imports so they can be tested locally."""
import os
import shutil
import time


def copy_dir_with_retry(src_dir, dst_dir, remount=None, attempts=4, wait=10, log=print):
    """Copy every file of src_dir to dst_dir, skipping files already copied with the same size.

    Google Drive's FUSE mount in Colab can drop during large reads ("OSError: [Errno 107] Transport endpoint is not
    connected"). Each file is copied to `<name>.part` and renamed only when complete, so an interrupted copy never
    leaves a truncated file that looks finished. On OSError, `remount()` is called (e.g. a forced drive.mount) and
    the whole pass is retried; files already complete are skipped.
    """
    os.makedirs(dst_dir, exist_ok=True)
    for attempt in range(1, attempts + 1):
        try:
            for f in sorted(os.listdir(src_dir)):
                s, d = os.path.join(src_dir, f), os.path.join(dst_dir, f)
                if not os.path.isfile(s):
                    continue
                size = os.path.getsize(s)
                if os.path.exists(d) and os.path.getsize(d) == size:
                    continue
                t0 = time.time()
                shutil.copyfile(s, d + ".part")
                if os.path.getsize(d + ".part") != size:
                    raise OSError(f"size mismatch after copying {f}")
                os.replace(d + ".part", d)
                log(f"copied {f} ({size / 1e9:.2f} GB, {time.time() - t0:.0f} s)")
            return
        except OSError as e:
            log(f"attempt {attempt}/{attempts} failed: {e}")
            if attempt == attempts:
                raise
            if remount is not None:
                remount()
            time.sleep(wait)
