import copy
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from .config import matrix_game3

WAN_CONFIGS = { 
    'matrix_game3': matrix_game3,
}

MAX_AREA_CONFIGS = {
    '704*1280': 704 * 1280,
}