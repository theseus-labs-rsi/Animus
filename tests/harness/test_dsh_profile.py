"""dsh profile 的可选回归测试：需要本机装有 dsh，缺失则整类跳过。

`configs/dsh/profiles/headless/` 里只保留 `package.json`：它显式声明这个 harness
启动哪些 dsh bundle。其余文件（`cordis.yml`、`cordis.patch.yml`、
`pnpm-workspace.yaml`）由 dsh 自己在 `$DSH_HOME/profiles/<name>/` 下按模板生成，
其中 `cordis.yml` 每次 boot 都会被无条件重写。

这个测试用离线的 `dsh --profile <name> --dump-config` 校验「我们钉住的 profile」
与「dsh 完全自己 bootstrap 的 profile」组合出的插件树一致；dsh 升级导致组合变化时
会在这里暴露，而不是等到 run 结果变化。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from agent_harnesses.config import REPOSITORY_ROOT
from agent_harnesses.runners.native_cli import DSH_PROFILE_NAME, dsh_profile_path

DUMP_TIMEOUT_S = 120
# DSH 自己生成的三个文件；我们不再随仓库分发它们。
GENERATED_BY_DSH = ("cordis.yml", "cordis.patch.yml", "pnpm-workspace.yaml")


def _dsh_binary() -> Path | None:
    """按 run 时的解析顺序找 dsh：DSH_BIN → PATH → 仓库 .npm-global。"""
    override = os.environ.get("DSH_BIN", "").strip()
    if override:
        path = Path(override).expanduser()
        return path if path.is_file() else None
    found = shutil.which("dsh")
    if found:
        return Path(found)
    local = REPOSITORY_ROOT / ".npm-global" / "bin" / "dsh"
    return local if local.is_file() else None


DSH_BINARY = _dsh_binary()


@unittest.skipUnless(DSH_BINARY, "本机没有 dsh，跳过 profile 组合回归")
class DshProfileTests(unittest.TestCase):
    def _dump_config(self, home: Path) -> str:
        env = {**os.environ, "DSH_HOME": str(home)}
        completed = subprocess.run(
            [str(DSH_BINARY), "--profile", DSH_PROFILE_NAME, "--dump-config"],
            cwd=home,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=DUMP_TIMEOUT_S,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        return completed.stdout

    def test_shipped_profile_matches_dsh_generated_default(self):
        source_dir = dsh_profile_path()
        self.assertTrue(
            (source_dir / "package.json").is_file(), f"缺少 {source_dir}/package.json"
        )
        # 仓库里只应保留 package.json；其余由 dsh 生成。
        shipped = sorted(p.name for p in source_dir.iterdir())
        self.assertEqual(shipped, ["package.json"], shipped)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pinned_home = root / "pinned"
            (pinned_home / "profiles" / DSH_PROFILE_NAME).mkdir(parents=True)
            shutil.copy2(
                source_dir / "package.json",
                pinned_home / "profiles" / DSH_PROFILE_NAME / "package.json",
            )
            pinned_dump = self._dump_config(pinned_home)
            profile_dir = pinned_home / "profiles" / DSH_PROFILE_NAME
            generated = sorted(p.name for p in profile_dir.iterdir())
            # `cordis.yml` 由 prepareProfile 每次 boot 无条件重写，所以必定出现；
            # profile manifest 已存在时 dsh 不会跑完整 bootstrap，patch/pnpm 模板不会补写。
            self.assertIn("cordis.yml", generated, generated)
            self.assertIn("package.json", generated, generated)

            # 完全不提供 profile：dsh 按 profile 名自己 bootstrap。
            bare_home = root / "bare"
            bare_home.mkdir()
            bare_dump = self._dump_config(bare_home)

            self.assertEqual(
                pinned_dump,
                bare_dump,
                "我们钉住的 profile 与 dsh 自行生成的组合不一致；"
                "说明继续分发这些文件已经改变了 harness 行为",
            )
            # bootstrap 路径会写全 manifest + patch + pnpm workspace 模板。
            bootstrapped = sorted(
                p.name for p in (bare_home / "profiles" / DSH_PROFILE_NAME).iterdir()
            )
            for name in GENERATED_BY_DSH:
                self.assertIn(name, bootstrapped, bootstrapped)


if __name__ == "__main__":
    unittest.main()
