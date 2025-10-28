import sys

from typing import Dict, List, Any

from pathlib import Path
parent_dir = Path(__file__).parent.parent.parent
sys.path.append(str(parent_dir))
import config


print(parent_dir)

class Preeprocess:
    def __init__(self):
        pass
        # self.pipeline = 