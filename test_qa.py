"""校园问答效果测试：8 个问题，验证「检索命中」和「答案准确」。

用法（campus-agent 目录下）：
    .\.venv\Scripts\python.exe test_qa.py

输出两张表：
    1) 检索层：top-2 里有没有含正确答案的片段（衡量 RAG 的召回）
    2) 生成层：最终回答里有没有正确答案（衡量端到端效果）
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()

import main  # 复用 main.py 里已经初始化好的 vectorstore 和 client

# (问题, 答案中必须出现的关键词, 归属分类)
CASES = [
    ("新生报到是几号？", "9月5", "02"),
    ("学费大概多少？", "6000", "01"),
    ("军训要多久？", "三周", "03"),
    ("快递点在哪里？", "前街", "04"),
    ("图书馆能借几本书？", "三本", "04"),
    ("宿舍几点关门？", "十一点半", "04"),
    ("校园网要钱吗？", "免费", "04"),
    ("什么时候可以转专业？", "一年级", "03"),
]


import re

def hit(text: str, keyword: str) -> bool:
    norm = lambda s: re.sub(r"\s+", "", s or "")
    return norm(keyword) in norm(text)

def main_test():
    retrieve_ok = 0
    answer_ok = 0

    print("=" * 78)
    print(f"{'问题':<22}{'检索命中':<10}{'回答命中':<10}实际答案（截断）")
    print("=" * 78)

    for question, keyword, category in CASES:
        # 1) 检索层
        docs = main.vectorstore.similarity_search(question, k=2)
        context = "\n".join(d.page_content for d in docs)
        r_ok = hit(context, keyword)

        # 2) 生成层
        try:
            resp = main.client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": f"你是一个校园助手。请只根据以下资料回答：\n{context}"},
                    {"role": "user", "content": question},
                ],
                temperature=0.1,
            )
            answer = resp.choices[0].message.content or ""
        except Exception as e:
            answer = f"[调用失败] {e}"

        a_ok = hit(answer, keyword)

        retrieve_ok += r_ok
        answer_ok += a_ok

        print(f"{question:<22}{'✅' if r_ok else '❌':<10}{'✅' if a_ok else '❌':<10}"
              f"{answer.replace(chr(10), ' ')[:40]}")

    total = len(CASES)
    print("=" * 78)
    print(f"检索命中率：{retrieve_ok}/{total} = {retrieve_ok / total:.0%}")
    print(f"回答命中率：{answer_ok}/{total} = {answer_ok / total:.0%}")
    print("=" * 78)
    print("\n说明：")
    print("  - 「检索命中 ❌」说明知识库里没这条，或切分/检索没捞到")
    print("  - 「检索 ✅ 但回答 ❌」说明资料找到了，但模型没用上（prompt 问题）")


if __name__ == "__main__":
    main_test()
