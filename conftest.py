#!/usr/bin/env python3

"""pins every HuggingFace cache into a throwaway dir before those libs import"""

import atexit
import os
import shutil
import tempfile

# `huggingface_hub`, `datasets`, and `transformers` freeze their cache
# locations and their offline flag into module-level constants at *import*
# time, so this has to run before the first `import datasets` in the session.
# A rootdir conftest is the earliest plugin-free hook there is: pytest imports
# conftests outermost-directory-first, ahead of `tests/conftest.py` and of
# every test module. `tmp_path_factory` does not exist yet, hence `mkdtemp`.
CACHE_HOME = tempfile.mkdtemp(prefix="cotorra-test-cache-")
atexit.register(shutil.rmtree, CACHE_HOME, ignore_errors=True)

# every one of these outranks (or side-steps) `HF_HOME`, and a developer may
# have any of them pointed at a real cache, so clear them and let `HF_HOME`
# alone decide; `HF_TOKEN` is dropped so nothing can authenticate as the user
for var in (
    "HF_ASSETS_CACHE",
    "HF_DATASETS_CACHE",
    "HF_DATASETS_DOWNLOADED_DATASETS_PATH",
    "HF_DATASETS_EXTRACTED_DATASETS_PATH",
    "HF_HUB_CACHE",
    "HF_MODULES_CACHE",
    "HF_TOKEN",
    "HF_TOKEN_PATH",
    "HF_XET_CACHE",
    "HUGGINGFACE_ASSETS_CACHE",
    "HUGGINGFACE_CO_STAGING",
    "HUGGINGFACE_HUB_CACHE",
    "HUGGING_FACE_HUB_TOKEN",
):
    os.environ.pop(var, None)

os.environ.update(
    {
        # one root for the hub cache, the datasets cache, the dynamic-module
        # cache, the token file, and the xet chunk cache
        "HF_HOME": CACHE_HOME,
        # no network: this is the master switch huggingface_hub's http hook
        # honors, and every transformers `from_pretrained` defers to it.
        # Local directories still resolve -- `cached_file` returns them before
        # it ever looks at the cache or the hub
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        # `datasets` keeps its own copy of the offline flag
        "HF_DATASETS_OFFLINE": "1",
        # ... and its download-count telemetry (a HEAD to s3.amazonaws.com on
        # every `load_dataset`) is gated separately
        "HF_UPDATE_DOWNLOAD_COUNTS": "0",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_UPDATE_CHECK": "1",
        # the xet downloader is a rust extension that reads the process
        # environment directly, so it cannot be monkeypatched from python
        "HF_HUB_DISABLE_XET": "1",
        # nothing here should ever reach wandb; `Trainer.__init__` sets
        # WANDB_PROJECT/WANDB_NAME unconditionally, and the shipped default is
        # `report_to: wandb`, so keep the run dirs local and the mode inert
        "WANDB_MODE": "disabled",
        "WANDB_DIR": CACHE_HOME,
        "WANDB_CACHE_DIR": os.path.join(CACHE_HOME, "wandb-cache"),
        "WANDB_CONFIG_DIR": os.path.join(CACHE_HOME, "wandb-config"),
        "WANDB_DATA_DIR": os.path.join(CACHE_HOME, "wandb-data"),
        # torch would otherwise scatter compile caches through $TMPDIR
        "TORCHINDUCTOR_CACHE_DIR": os.path.join(CACHE_HOME, "torchinductor"),
        "TRITON_CACHE_DIR": os.path.join(CACHE_HOME, "triton"),
    }
)
