import os
import torch
from .pycore import *
from .common import *
from .pyoqmc import *

Configuration.set_macro("PATH_SE_PYTHON", os.path.dirname(__file__))