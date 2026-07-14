# 🎯 智能直播带货辅助系统

> **git init**
>
> 实时分析抖音直播弹幕，智能推荐话术策略，助力主播提升转化率。

---

## 📸 项目亮点

| 模块 | 技术 | 说明 |
|------|------|------|
| 🤖 **LLM 引擎** | DeepSeek-V4-Flash | 情感分析、意图识别、话术生成 |
| 🧠 **Agent 决策** | LangChain Agent | 智能选择最优话术策略 |
| 📚 **RAG 知识库** | 关键词检索 | 商品知识库 + 话术策略库 |
| 🔄 **工作流引擎** | LangGraph StateGraph | 5 节点串行处理管道 |
| 🌐 **实时接入** | WebSocket + 弹窗桥接 | 绕过 CSP 限制，实时捕获抖音弹幕 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    抖音直播间 (HTTPS)                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │  F12 Console → 粘贴捕获脚本                      │   │
│  │  ┌──────────────┐    postMessage    ┌────────┐  │   │
│  │  │ MutationObserver│ ──────────────→ │桥接弹窗│  │   │
│  │  │ 监听 DOM 弹幕  │                  │(HTTP)  │  │   │
│  │  └──────────────┘                    └───┬────┘  │   │
│  └───────────────────────────────────────────────│────┘   │
└──────────────────────────────────────────────────│─────────┘
                                                   │ WebSocket
                                                   ▼
┌─────────────────────────────────────────────────────────┐
│             本地 Python 后端 (localhost)                    │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────┐  │
│  │ 接收弹幕  │ → │ LLM 分析  │ → │ RAG 检索  │ → │Agent │  │
│  │ 节点 1   │   │ 节点 2   │   │ 节点 3   │   │决策  │  │
│  └──────────┘   └──────────┘   └──────────┘   └──┬───┘  │
│                                                   ▼      │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐            │
│  │ 生成话术  │ ← │ 策略推荐  │ ← │ 数据看板  │            │
│  │ 节点 5   │   │ 节点 4   │   │ (实时)   │            │
│  └──────────┘   └──────────┘   └──────────┘            │
└─────────────────────────────────────────────────────────┘
```

---

## ✨ 功能特性

### 🎯 核心功能
- **实时弹幕捕获**：通过桥接页面绕过抖音 CSP 安全策略，实时捕获弹幕
- **情感分析 + 意图识别**：LLM 自动分析每条弹幕的情感倾向和用户意图
- **RAG 知识检索**：基于关键词匹配的商品知识库和话术策略库检索
- **Agent 智能决策**：根据弹幕上下文和商品信息，自动选择最优话术策略
- **话术自动生成**：生成自然口语化的直播话术，直接可用

### 📊 数据看板
- 实时统计弹幕总数、情感分布、意图分布
- 近 5 条弹幕处理记录回溯
- 控制台带颜色标记的实时输出

### 🚀 三种运行模式
| 模式 | 命令 | 用途 |
|------|------|------|
| 🎭 **演示模式** | `python Intelligent-live-streaming-sales-assistance.py` | 快速体验系统功能 |
| 💬 **交互模式** | `python Intelligent-live-streaming-sales-assistance.py --interactive` | 手动输入弹幕测试 |
| 📡 **直播模式** | `python Intelligent-live-streaming-sales-assistance.py --live` | 接入抖音直播实时使用 |

---

## 🛠️ 技术栈

| 技术 | 用途 |
|------|------|
| [Python 3.11+](https://www.python.org/) | 编程语言 |
| [LangChain](https://www.langchain.com/) | LLM 应用框架 |
| [LangGraph](https://langchain-ai.github.io/langgraph/) | 工作流编排引擎 |
| [LangChain DeepSeek](https://pypi.org/project/langchain-deepseek/) | DeepSeek 模型集成 |
| [DeepSeek-V4-Flash](https://platform.deepseek.com/) | 大语言模型 |
| [WebSockets](https://websockets.readthedocs.io/) | 实时双向通信 |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | 环境变量管理 |

---

## 📦 安装

### 前置条件
- Python 3.11 或更高版本
- DeepSeek API Key（[注册获取](https://platform.deepseek.com/)）

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/你的用户名/Intelligent-live-streaming-sales-assistance.git
cd Intelligent-live-streaming-sales-assistance

# 2. 创建虚拟环境（推荐）
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# 3. 安装依赖
pip install langchain langchain-core langchain-deepseek langgraph websockets python-dotenv

# 4. 配置 API Key
cp .env.example .env
# 编辑 .env 文件，填入你的 DeepSeek API Key
```

---

## 🚀 快速开始

### 1️⃣ 演示模式

```bash
python Intelligent-live-streaming-sales-assistance.py
```

### 2️⃣ 交互模式

```bash
python Intelligent-live-streaming-sales-assistance.py --interactive
```

输入弹幕内容，系统自动分析并生成话术推荐。

### 3️⃣ 抖音直播实时模式

```bash
python Intelligent-live-streaming-sales-assistance.py --live
```

然后在抖音直播间网页：

1. 按 **F12** 打开开发者工具
2. 点击 **Console** 选项卡
3. 复制终端输出的注入脚本，粘贴到 Console 后回车
4. 浏览器弹出桥接窗口（如果被拦截，请允许弹窗）
5. 弹幕开始实时分析，话术推荐实时显示！

---

## 📁 项目结构

```
Intelligent-live-streaming-sales-assistance/
├── Intelligent-live-streaming-sales-assistance.py  # 主程序
├── .env.example                                     # 环境变量模板
├── .gitignore                                       # Git 忽略规则
└── README.md                                        # 项目文档
```

---

## 🔧 配置说明

### 环境变量

在 `.env` 文件中配置：

```env
DEEPSEEK_API_KEY=your_api_key_here
```

### 自定义知识库

编辑 `PRODUCT_KNOWLEDGE_BASE` 和 `STRATEGY_KNOWLEDGE_BASE` 变量，添加你的商品和话术策略。

---

## 📊 工作流详解

系统使用 LangGraph 构建了 5 节点串行工作流：

| 节点 | 功能 | 输入 | 输出 |
|------|------|------|------|
| 1️⃣ `receive_danmaku` | 接收弹幕 | 弹幕队列 | 当前弹幕 |
| 2️⃣ `analyze_danmaku` | 情感分析 | 弹幕文本 | 情感/意图 JSON |
| 3️⃣ `rag_retrieve` | 知识检索 | 弹幕+意图 | 商品/策略列表 |
| 4️⃣ `agent_decision` | 策略决策 | 分析+商品+策略 | 最佳策略 |
| 5️⃣ `generate_script` | 生成话术 | 策略+商品+上下文 | 直播话术 |

---

## ⚠️ 注意事项

1. **API Key 安全**：不要将 `.env` 文件提交到 GitHub
2. **弹窗权限**：使用直播模式时，需要允许浏览器弹出窗口
3. **网络要求**：本地服务需要运行在 `localhost`，确保端口 8080 和 8765 未被占用
4. **抖音 DOM 结构**：如果抖音更新页面结构，可能需要调整注入脚本中的 CSS 选择器

---

## 📄 开源协议

[MIT License](LICENSE)

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📧 联系方式

如有问题，请提交 GitHub Issue。