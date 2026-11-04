"""构建向量库 v2：按 `## ` 标题切分，一条问答一个 chunk。

相比原版的三处改动：
1. **切分方式**：不再用 chunk_size=100 硬切，而是按 `## 问题` 切 —— 一条事实一块，
   不会被切碎，也不会把不相关的内容混在一起。
2. **幂等**：每次重建前先删除 chroma_db，避免重复写入（原版跑 3 次会有 3 份数据）。
3. **质量检查**：统计「待补充」和「请登录…查询」这类空答案，写到答案里的提示。

用法：
    python build_db.py
"""
import os
import re
import shutil

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()

DATA_DIR = "data"
DB_DIR = "./chroma_db"

# 判定「空答案」的句式：这些回答等于没回答
VAGUE_PATTERNS = ["请登录", "请向", "请直接", "咨询学校", "咨询老师", "自行查询"]
PLACEHOLDER = "（待补充）"


def load_faq(data_dir: str):
    """把 data/*.md 按 `## ` 切成一条条 Document。"""
    docs = []
    empty_titles = []
    vague_titles = []

    if not os.path.isdir(data_dir):
        raise SystemExit(f"找不到数据目录：{data_dir}")

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".md"):
            continue

        path = os.path.join(data_dir, filename)
        with open(path, encoding="utf-8") as f:
            text = f.read()

        # 按行首的 "## " 切分；parts[0] 是文件头说明，丢掉
        parts = re.split(r"(?m)^##\s+", text)
        for block in parts[1:]:
            lines = block.strip().splitlines()
            if not lines:
                continue
            title = lines[0].strip()
            body = "\n".join(lines[1:]).strip()

            if not body or body.startswith(PLACEHOLDER):
                empty_titles.append(title)
                continue

            if any(p in body for p in VAGUE_PATTERNS):
                vague_titles.append(title)

            # 把问题和答案拼成一段，检索时语义更完整
            docs.append(
                Document(
                    page_content=f"问：{title}\n答：{body}",
                    metadata={
                        "source": filename,
                        "title": title,
                        "category": filename[:2],  # 01/02/03/04
                    },
                )
            )

    return docs, empty_titles, vague_titles


def main():
    print("1. 读取并切分 data/*.md ...")
    docs, empty, vague = load_faq(DATA_DIR)
    print(f"   有效问答：{len(docs)} 条")

    if empty:
        print(f"   ⚠ 空答案（被跳过）：{len(empty)} 条")
        for t in empty[:5]:
            print(f"      - {t}")

    if vague:
        print(f"   ⚠ 疑似空答案（答案是「去官网查」）：{len(vague)} 条")
        for t in vague[:5]:
            print(f"      - {t}")

    if not docs:
        raise SystemExit("没有可用的问答数据，先填 data/*.md")

    # 幂等：每次重建前清空旧库
    if os.path.isdir(DB_DIR):
        print(f"2. 删除旧向量库 {DB_DIR} ...")
        shutil.rmtree(DB_DIR)

    print("3. 初始化智谱 Embedding（embedding-2）...")
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        model="embedding-2",
        check_embedding_ctx_length=False,
    )

    print(f"4. 写入向量库（{len(docs)} 条）...")
    Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=DB_DIR)

    print(f"✅ 完成：{len(docs)} 条问答已入库 -> {DB_DIR}")
    if vague:
        print(f"   建议先把那 {len(vague)} 条「去官网查」改成具体答案，RAG 效果会明显变好。")


if __name__ == "__main__":
    main()
