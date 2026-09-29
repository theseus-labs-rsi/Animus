# DSH profile

安装 Node.js 22.19.0 及以上版本，然后安装仓库固定的 CLI：

```bash
npm install --global @deepseek-ai/dsh@0.1.2-rc.1
dsh --version
```

复制 `configs/env/secrets.env.example` 为 `configs/env/secrets.env`，填写 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_BASE_URL`。运行器从 `PATH` 查找 `dsh`；需要指定位置时设置 `DSH_BIN`。

运行器将 `profiles/headless/package.json` 复制到每次评测的 DSH 工作目录，并用 `dsh --profile headless` 启动。这个文件固定了 `@deepseek-ai/dsh-base` 和 `@deepseek-ai/dsh-headless` 两个 bundle。修改 bundle 后请开启新 run 验证。
