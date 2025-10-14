import os
import logging
from datetime import datetime
from cellpose_omni.models import CellposeModel

# I recommend using cli to train omnipose models, but this demonstrates how to train a model programmatically.

# ========== CONFIG ==========
TRAIN_DIR = '/Users/lucas/Documents/GitHub/ECT_template_1/data/processed/train'
MODEL_SAVE_DIR = os.path.join(TRAIN_DIR, 'models')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

LOGFILE = os.path.join(MODEL_SAVE_DIR, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

# ========== LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOGFILE),
        logging.StreamHandler()
    ],
    format='%(asctime)s [%(levelname)s] %(message)s'
)

logging.info("=== Starting Omnipose grayscale training ===")

try:
    logging.info("Initializing Omnipose model...")
    model = CellposeModel(
        gpu=False,
        omni=True,
        nchan=1,
        nclasses=2,
        residual_on=True,
        style_on=True,
        concatenation=False,
        diam_mean=30,
    )

    logging.info(f"Starting training from directory: {TRAIN_DIR}")

    model.train(
        train_data=TRAIN_DIR,
        train_labels=None,
        channels=[0, 0],    # Grayscale mode
        save_path=MODEL_SAVE_DIR,
        learning_rate=0.1,
        n_epochs=300,
        batch_size=8,
        dataloader=True,
        num_workers=8,
        save_every=5,
        min_train_masks=5,
        rescale=False
    )

    logging.info("Training completed successfully.")

except Exception as e:
    logging.exception("Training failed due to an unexpected error.")
