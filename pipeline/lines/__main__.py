"""注册表自检入口:python -m pipeline.lines(逻辑收敛在 __init__._selftest,避免两处各抄一份)。"""
import sys
from pipeline.lines import _selftest

sys.exit(_selftest())
