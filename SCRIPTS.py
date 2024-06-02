import logging
import os
from pathlib import Path


def create_symlink(source_dir_name: str, target_path: str):
    """
    Creates a symbolic link from the source path to the target path.

    Args:
        source_dir_name (str): The name of the directory in the local instance.
        target_path (str): The path in the host machine where a dataset is located.
    """
    source_path = Path(os.getcwd()) / source_dir_name  # noqa: F821

    logging.debug(f"Creating symlink from {source_path} to {target_path}")

    # Check if the symlink already exists
    if not os.path.islink(source_path):
        if os.path.exists(source_path):
            logging.info(f"File or directory {source_path} already exists and is not a symlink.")
        else:
            os.symlink(target_path, str(source_path))
    else:
        logging.info(f"Symlink {source_path} already exists.")


source_dir_name = "data"
target_path = os.path.join('/datasets', 'RomanianDB_v.0.8.1')
create_symlink(source_dir_name, target_path)