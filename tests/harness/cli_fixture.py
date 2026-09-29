"""Real portable subprocess fixtures, including a Windows executable launcher."""
from pathlib import Path
from pip._vendor.distlib.scripts import ScriptMaker


def write_cli(path: Path, source: str) -> Path:
    source = ("import sys\nif '--version' in sys.argv:\n"
              "    print('offline-cli-fixture-1')\n    sys.exit(0)\n" + source)
    maker = ScriptMaker(None, str(path.parent))
    maker.set_mode = True
    filenames = []
    maker._write_script([path.name], maker._get_shebang("utf-8"),
                        source.encode("utf-8"), filenames, "py")
    return Path(filenames[0])
