import os

from ffldraft import SASNUT, DEMO, LAMBO

HERE_DIR = os.path.realpath(os.path.dirname(__file__))
FYAML = LAMBO
REPL_PLOT_NUM = 25

SECRET_KEY = os.environ.get(
    'SECRET_KEY',
    '\xa6\xf5r\xc9\xce\xfd2\x0b8\xf4~Y\x15\xaf\x07HmGR\xa4%\x11%h'
)
