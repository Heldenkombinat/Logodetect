"""Main training function to be called from the terminal."""

from .constants import CONFIG_FILE
from .utils import load_config, create_configs
from .module_loader import load_modules


if __name__ == "__main__":
    # Create multiple configs if exploring hyperparams:
    config = load_config(CONFIG_FILE)
    configs = create_configs(config)
    for config_file in configs:
        trainer = load_modules(config_file)
        trainer.train()
