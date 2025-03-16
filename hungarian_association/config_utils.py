import yaml
import os
from ament_index_python.packages import get_package_share_directory

def load_hungarian_config():
    """Load configuration for the hungarian_association package."""
    config_file = os.path.join(
        get_package_share_directory('hungarian_association'),
        'config',
        'hungarian_config.yaml'
    )

    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading hungarian_association config: {e}")
        return None 