"""校园智能 Agent 问答服务。

改造说明（相对上一版）：
1. 会话记忆从内存 dict 换成 SQLite —— 重启不丢，多进程也能共享
2. 错误处理改用 HTTPException —— 客户端能区分成功/失败，不再一律 200
3. print 换成 logging —— 能落盘、能分级、带时间戳
"""
import json
import logging
import os
import sqlite3
import time
from contextlib import closing

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI, RateLimitError
from pydantic import BaseModel

from weather import get_weather
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# ----------------------------------------------------------------------------
# 日志：按时间戳 + 级别输出，以后要落盘只需再加一个 FileHandler
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("campus")


# ----------------------------------------------------------------------------
# 模型与向量库
# ----------------------------------------------------------------------------
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

embeddings = OpenAIEmbeddings(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    model="embedding-2",
    check_embedding_ctx_length=False,
)
log.info("正在连接向量数据库 ...")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

app = FastAPI(title="我的校园Agent")

# 检索距离阈值：超过它的片段视为不相关，直接丢弃。
# 实测库内问题最高 0.87、库外最低 1.18，取中间值。
RELEVANCE_THRESHOLD = 1.0

# ----------------------------------------------------------------------------
# 会话记忆：SQLite
#
# 为什么不用全局 dict：
#   - 进程重启就丢
#   - uvicorn 开多个 worker 时各存各的
#   - 只有上限没有清理，内存只涨不降
# ----------------------------------------------------------------------------
DB_PATH = "memory.db"
HISTORY_LIMIT = 6  # 只保留最近 3 轮（user + assistant 各 3 条）


def init_db() -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id, id)"
        )
        conn.commit()


def load_history(session_id: str, limit: int = HISTORY_LIMIT) -> list[dict]:
    """取最近 limit 条消息，按时间正序返回，可直接喂给模型。"""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [{"role": role, "content": text} for role, text in reversed(rows)]


def append_message(session_id: str, role: str, content: str) -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        conn.commit()


def trim_history(session_id: str, keep: int = HISTORY_LIMIT) -> None:
    """只保留最近 keep 条，避免数据库无限增长。"""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            "DELETE FROM messages WHERE session_id=? AND id NOT IN "
            "(SELECT id FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?)",
            (session_id, session_id, keep),
        )
        conn.commit()


init_db()


class ChatRequest(BaseModel):
    session_id: str = "default_user"
    query: str


# ----------------------------------------------------------------------------
# 检索
# ----------------------------------------------------------------------------
SYSTEM_PROMPT_TMPL = """你是桂林电子科技大学的校园助手，服务对象是新入学的大一新生。

【必须遵守的规则】
1. 只依据下面【资料】里的内容回答，不要使用你自己的知识。
2. 如果【资料】里没有答案，直接回复："这个问题我这边没有相关资料，建议咨询辅导员或查看学校官网。"
3. 禁止编造店名、价格、电话、时间等具体信息。
4. 回答简洁，不超过 5 句话。
5. 如果【资料】里没有内容，但用户问的是天气这类可以用工具查询的信息，请调用工具，不要直接拒答。

【资料】
{context}"""

NO_DATA_ANSWER = "这个问题我这边没有相关资料，建议咨询辅导员或查看学校官网。"


def get_school_context(question: str) -> str:
    """检索知识库，只返回相关度够高的片段。"""
    try:
        results = vectorstore.similarity_search_with_score(question, k=3)
        useful = [doc for doc, score in results if score < RELEVANCE_THRESHOLD]
        log.info("检索命中 %d/%d 条（阈值 %.1f）", len(useful), len(results), RELEVANCE_THRESHOLD)
        return "\n".join(doc.page_content for doc in useful)
    except Exception:
        log.exception("检索出错，按无资料处理")
        return ""


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------
# 天气工具的真实实现在 weather.py：
#   - 数据来自 Open-Meteo（免费、无需 API key）
#   - 内置常用城市别名表，解决中文地名匹配不准的问题
#   - 带 10 分钟缓存，避免重复请求


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气（真实数据，来自 Open-Meteo）",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，例如：桂林、北京"}
                },
                "required": ["city"],
            },
        },
    }
]


def build_messages(session_id: str, query: str, context: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT_TMPL.format(context=context)}]
    messages.extend(load_history(session_id))
    messages.append({"role": "user", "content": query})
    return messages


# ----------------------------------------------------------------------------
# 接口
# ----------------------------------------------------------------------------
@app.get("/health")
async def health():
    """健康检查：部署时用来确认服务活着。"""
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    # 注意：这里不能因为"知识库没资料"就直接兜底。
    # 天气这类问题本来就不在知识库里，答案是靠工具查的，
    # 提前返回会让工具永远没机会被调用。兜底放在"模型没调工具"之后。
    context = get_school_context(query)
    messages = build_messages(request.session_id, query, context)
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="glm-4-flash", messages=messages, tools=tools, tool_choice="auto"
            )
            response_message = response.choices[0].message
            final_answer = None

            if response_message.tool_calls:
                messages.append(response_message)
                for tool_call in response_message.tool_calls:
                    if tool_call.function.name == "get_weather":
                        args = json.loads(tool_call.function.arguments)
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": get_weather(args.get("city")),
                        })
                second = client.chat.completions.create(
                    model="glm-4-flash", messages=messages
                )
                final_answer = second.choices[0].message.content
            else:
                # 没调工具：知识库也没有资料，说明这个问题我们答不了
                if not context.strip():
                    log.info("无资料且未调用工具，返回兜底 session=%s", request.session_id)
                    return {"answer": NO_DATA_ANSWER}
                final_answer = response_message.content

            if not final_answer:
                final_answer = "抱歉，我这边没有生成有效回答，请再问一次。"

            # 写入记忆并裁剪
            append_message(request.session_id, "user", query)
            append_message(request.session_id, "assistant", final_answer)
            trim_history(request.session_id)

            return {"answer": final_answer}

        except RateLimitError:
            if attempt == max_retries - 1:
                log.warning("触发限流且重试耗尽 session=%s", request.session_id)
                raise HTTPException(status_code=429, detail="请求太频繁，请稍后再试")
            wait = 2 ** attempt
            log.warning("触发限流，%d 秒后重试（第 %d 次）", wait, attempt + 1)
            time.sleep(wait)

        except HTTPException:
            raise

        except Exception:
            log.exception("处理请求失败 session=%s", request.session_id)
            raise HTTPException(status_code=500, detail="服务内部错误，请稍后再试")


@app.post("/chat_stream")
async def chat_stream(request: ChatRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    # 和 /chat 一样：不能因为知识库没资料就提前返回，否则工具永远调不到。
    # 兜底放在"模型决定不调工具"之后（见 generate 里的 elif）。
    context = get_school_context(query)
    messages = build_messages(request.session_id, query, context)

    def generate():
        full_answer = ""
        try:
            # 第一阶段：非流式，判断要不要调工具
            # （流式响应里拿不到完整的 tool_calls，所以必须先用非流式问一次）
            first = client.chat.completions.create(
                model="glm-4-flash", messages=messages, tools=tools, tool_choice="auto"
            )
            response_message = first.choices[0].message

            if response_message.tool_calls:
                messages.append(response_message)
                for tool_call in response_message.tool_calls:
                    if tool_call.function.name == "get_weather":
                        args = json.loads(tool_call.function.arguments)
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": get_weather(args.get("city")),
                        })
                yield "data: [系统提示：已获取实时天气，正在生成回答...]\n\n"
            elif not context.strip():
                log.info("无资料且未调用工具，流式返回兜底 session=%s", request.session_id)
                yield f"data: {NO_DATA_ANSWER}\n\n"
                return

            # 第二阶段：带上工具结果，开启流式
            stream = client.chat.completions.create(
                model="glm-4-flash", messages=messages, stream=True
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_answer += delta
                    yield f"data: {delta}\n\n"

            append_message(request.session_id, "user", query)
            append_message(request.session_id, "assistant", full_answer)
            trim_history(request.session_id)
            log.info("流式回答完成 session=%s 长度=%d", request.session_id, len(full_answer))

        except Exception:
            log.exception("流式回答失败 session=%s", request.session_id)
            yield "data: [ERROR] 服务出错了，请稍后再试\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# 前端页面（必须在所有接口之后挂载）
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

