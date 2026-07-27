from dotenv import load_dotenv
load_dotenv()
import os
import json
from utility.embedding import lora_fine_tune
from utility.pg import get_training_samples
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


if __name__ == '__main__':
    lora_fine_tune(get_training_samples())