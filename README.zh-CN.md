> English documentation is canonical. See [README.md](README.md).

# AllCallAll Agent Runtime 简介

本仓库是 AllCallAll 的独立 Python Agent 与 RAG 运行时，负责 LangGraph
编排、检索规划、重排、证据组织、引用、工具提案、追踪和确定性评测。

## 与主仓的边界

- [AllCallAll 主仓](https://github.com/XianingY/allcallall)负责产品界面、
  用户、组织、会话、会议、权限、审批、审计和业务写入。
- Python 运行时只通过既有 HTTP、JSON Schema 和 Tool Bridge 契约协作。
- 写工具只返回需要审批的提案，不直接修改产品数据。

## 快速开始

```bash
python3 -m venv .venv
. .venv/bin/activate
make install-dev
make test
make lint
make typecheck
make contracts-check
```

启动两个核心服务：

```bash
make run-agent-runtime
make run-rag-runtime
```

## 文档与参与

- [文档索引](docs/README.md)
- [贡献指南](CONTRIBUTING.md)
- [安全策略](SECURITY.md)
- [支持方式](SUPPORT.md)
- [行为准则](CODE_OF_CONDUCT.md)

评测数据只表示固定样例上的回归结果，不代表开放领域模型质量。项目以
[MIT License](LICENSE) 发布。
