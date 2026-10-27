from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI, RateLimitError
import os
from dotenv import load_dotenv
import time
import json

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 加载环境变量
load_dotenv()

# 1. 初始化聊天模型（智谱）
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

# 2. 初始化向量模型并连接本地数据库
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    model="embedding-2",
    check_embedding_ctx_length=False
)
print("正在连接向量数据库...")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

app = FastAPI(title="我的校园Agent")

# 3. 全局变量：用来存各个用户的聊天记录
chat_histories = {}


# 4. 定义请求体（新增了 session_id）
class ChatRequest(BaseModel):
    session_id: str = "default_user"
    query: str


# --- RAG 检索函数 ---
def get_school_context(question: str) -> str:
    try:
        docs = vectorstore.similarity_search(question, k=2)
        return "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"检索出错: {e}")
        return ""


# --- Agent 工具定义 ---
def get_weather(city: str) -> str:
    print(f"🚨 [Agent正在干活] 大模型调用了天气工具，参数：{city}")
    if "桂林" in city:
        return "桂林今天中雨，气温 20-25 度，记得带伞。"
    else:
        return f"{city}今天晴，气温 22-28 度。"


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气情况",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：桂林、北京",
                    }
                },
                "required": ["city"],
            },
        },
    }
]


# --- 主接口 ---
@app.post("/chat")
async def chat(request: ChatRequest):
    # 1. 获取该用户的历史记忆（如果不存在则创建）
    if request.session_id not in chat_histories:
        chat_histories[request.session_id] = []
    history = chat_histories[request.session_id]

    # 2. 去向量库拿背景知识
    context = get_school_context(request.query)
    system_prompt = f"你是一个校园助手。请结合历史对话和以下资料回答问题：\n{context}"

    # 3. 组装发送给大模型的消息：系统提示 + 历史记忆 + 当前问题
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": request.query})

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 【第一次请求大模型】：决定是否调用工具
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message
            final_answer = None

            # 判断是否调用了工具
            if response_message.tool_calls:
                # 把大模型的“调用决定”加入当前上下文
                messages.append(response_message)

                for tool_call in response_message.tool_calls:
                    if tool_call.function.name == "get_weather":
                        args = json.loads(tool_call.function.arguments)
                        weather_result = get_weather(args.get("city"))

                        # 把工具结果塞回上下文
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": weather_result,
                        })

                # 【第二次请求大模型】：根据工具结果生成最终回答
                second_response = client.chat.completions.create(
                    model="glm-4-flash",
                    messages=messages,
                )
                final_answer = second_response.choices[0].message.content
            else:
                # 没调用工具，直接拿结果
                final_answer = response_message.content

            # 兜底保护：防止大模型真的吐空字符串
            if not final_answer:
                final_answer = "抱歉，我大脑短路了一下，请重新问一次。"

            # 4. 存入历史记忆（只存纯文本，保持记忆干净）
            history.append({"role": "user", "content": request.query})
            history.append({"role": "assistant", "content": final_answer})

            # 控制记忆长度，只保留最近 6 条（3轮对话），防止 Token 爆炸
            if len(history) > 6:
                chat_histories[request.session_id] = history[-6:]

            return {"answer": final_answer}

        except RateLimitError:
            if attempt == max_retries - 1:
                return {"error": "请求太频繁了，请等一分钟后再试"}
            wait_time = 2 ** attempt
            print(f"触发限流了，正在等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)

        except Exception as e:
            print(f"运行报错: {e}")
            return {"error": f"发生错误: {str(e)}"}


@app.post("/chat_stream")
async def chat_stream(request: ChatRequest):
    # 1. 准备基础上下文和记忆
    if request.session_id not in chat_histories:
        chat_histories[request.session_id] = []
    history = chat_histories[request.session_id]

    context = get_school_context(request.query)
    system_prompt = f"你是一个校园助手。请结合历史对话和以下资料回答问题：\n{context}"

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": request.query})

    def generate():
        full_answer = ""
        try:
            # 【第一阶段：非流式判断是否需要调用工具】
            # 注意这里没有 stream=True，而且带上了 tools
            first_response = client.chat.completions.create(
                model="glm-4-flash",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            response_message = first_response.choices[0].message

            # 如果大模型决定调用工具
            if response_message.tool_calls:
                messages.append(response_message)

                for tool_call in response_message.tool_calls:
                    if tool_call.function.name == "get_weather":
                        args = json.loads(tool_call.function.arguments)
                        weather_result = get_weather(args.get("city"))

                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": weather_result,
                        })

                # 告诉前端，工具跑完了，准备开始输出最终答案
                yield "data: [系统提示：已获取实时天气，正在生成回答...]\n\n"

            # 【第二阶段：开启流式输出】
            # 现在 messages 里已经包含了工具的结果（或者模型一开始就决定不需要工具）
            # 此时加上 stream=True，让它一个字一个字吐出来
            stream_response = client.chat.completions.create(
                model="glm-4-flash",
                messages=messages,
                stream=True
            )

            for chunk in stream_response:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_answer += text
                    yield f"data: {text}\n\n"

            # 3. 流式结束后，把这一轮对话存入记忆
            history.append({"role": "user", "content": request.query})
            history.append({"role": "assistant", "content": full_answer})
            if len(history) > 6:
                chat_histories[request.session_id] = history[-6:]

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)