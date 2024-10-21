# Data paths
DETEXIFY_DUMP_PATH = 'detexify.sql'
IAM_DATASET_PATH = 'path/to/iam_dataset/'

# Image settings
IMAGE_SIZE = 28

# Model parameters
NUM_CLASSES = None  # To be set after label encoding
BATCH_SIZE = 64
EPOCHS = 20

# Multiprocessing
NUM_WORKERS = 4  # Number of parallel processes
CHUNK_SIZE = 1000  # Number of lines to process per worker
