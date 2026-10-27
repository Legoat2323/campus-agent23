# 🎓 校园智能 Agent 问答服务 (Campus AI Agent)

## 📖 项目简介
基于大语言模型（LLM）与 RAG 架构的校园智能问答后端服务。旨在解决高校信息分散及大模型幻觉问题。支持私有知识库精准检索、外部工具（如天气）自动调用、多轮对话记忆以及 SSE 流式输出。

## 🛠️ 技术栈
- **Web 框架**：FastAPI, Uvicorn
- **大模型能力**：智谱 GLM-4 (兼容 OpenAI SDK)
- **RAG 架构**：LangChain, ChromaDB, Embedding-2
- **工程化**：Pydantic, python-dotenv, SSE流式输出

## ✨ 核心功能
1. **私有知识库问答**：自动加载文档、切片、向量化入库；提问时自动检索最相似的上下文喂给大模型。
2. **Agent 工具调用**：利用 Function Calling 自动判断用户意图，调用外部 API。
3. **多轮对话记忆**：基于 Session ID 实现轻量级历史上下文管理。
4. **打字机流式输出**：解决工具调用与流式输出的时序冲突，实现“先执行、后流式”的丝滑体验。

## 🚀 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 在根目录创建 `.env` 文件，填入智谱 API Key：`ZHIPU_API_KEY=你的key`
3. 构建向量库：`python build_db.py`
4. 启动服务：`python main.py`
5. 访问接口文档：`http://127.0.0.1:8000/docs`