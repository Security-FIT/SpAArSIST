"""Dataset locations and protocol file names.

Edit ``DATA_DIR`` (or set the ``SPAARSIST_DATA_DIR`` environment variable) to
point at the parent folder that contains the ASVspoof5 and In-the-Wild corpora.
"""

import os

DATA_DIR = os.environ.get("SPAARSIST_DATA_DIR", "/path/to/datasets")
RIR_ROOT = os.environ.get("SPAARSIST_RIR_ROOT", "")  # optional, for RIR augmentation

DATASETS = {
    "asvspoof5": {
        "root": os.path.join(DATA_DIR, "ASVspoof5"),
        "train_protocol": "ASVspoof5.train.tsv",
        "dev_protocol": "ASVspoof5.dev.track_1.tsv",
        "eval_protocol": "ASVspoof5.eval.track_1.tsv",
    },
    "inthewild": {
        "root": os.path.join(DATA_DIR, "release_in_the_wild"),
        "eval_protocol": "meta.csv",
    },
}

# Two-stage training recipe (Firc et al., Section 4.2).
TRAIN = {
    "stage1_epochs": 10,      # frozen XLS-R front-end
    "stage1_batch_size": 64,
    "stage2_epochs": 5,       # end-to-end fine-tuning
    "stage2_batch_size": 32,
    "lr": 1e-4,
    "num_workers": 8,
}
