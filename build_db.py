import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 加载环境变量
load_dotenv()

print("1. 开始初始化智谱 Embedding 模型...")
# 智谱兼容了 OpenAI 的接口，用于把文字变成向量
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    model="embedding-2",
    check_embedding_ctx_length=False # 防止因为文本长度检查报错
)

print("2. 正在加载本地文档 school_info.txt...")
# 确保你的项目里有一个叫 school_info.txt 的文件
loader = TextLoader("school_info.txt", encoding="utf-8")
documents = loader.load()

print("3. 正在切分文本...")
# 把长文本切成小块，这就是你网课里学的 Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,      # 每块最多 100 个字符
    chunk_overlap=20,    # 块与块之间重叠 20 个字符
    length_function=len,
)
chunks = text_splitter.split_documents(documents)
print(f"✅ 切分完成，共产生 {len(chunks)} 个文本块。")

print("4. 正在存入向量数据库 ChromaDB...")
# 将切片和向量存入本地文件夹
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db" # 数据会存到当前目录的 chroma_db 文件夹下     langchain自动就完成了向量化
)

print("🎉 知识库构建成功！数据已持久化到 ./chroma_db 文件夹。")