"""诊断脚本：看「库内问题」和「库外问题」的检索分数差多少。

用途：找一个阈值，低于它的才算"相关资料"，高于它的直接当作没有资料。

用法（campus-agent 目录下）：
    .\.venv\Scripts\python.exe test_scores.py
"""
import main  # 复用已初始化的 vectorstore

# (问题, 类型)  in = 知识库里应该有；out = 知识库里没有
QUERIES = [
    ("什么时候报到？", "in"),
    ("快递点在哪里？", "in"),
    ("图书馆能借几本书？", "in"),
    ("宿舍几点关门？", "in"),
    ("学校附近有啥好吃的？", "out"),
    ("今天天气怎么样？", "out"),
    ("帮我写一首诗", "out"),
]

print("=" * 80)
print("Chroma 返回的是「距离」，越小越相关")
print("=" * 80)

for question, kind in QUERIES:
    results = main.vectorstore.similarity_search_with_score(question, k=2)
    print(f"\n[{kind:>3}] {question}")
    for doc, score in results:
        preview = doc.page_content.replace("\n", " ")[:46]
        print(f"        score={score:.4f}  {preview}")

print("\n" + "=" * 80)
print("怎么用这个结果：")
print("  1. 看 in 类和 out 类的分数有没有明显分界")
print("  2. 如果有（比如 in 都 < 0.9，out 都 > 1.3），取中间值做阈值")
print("  3. 如果重叠严重，说明向量检索区分不了，需要换方法（关键词过滤/LLM 判定）")
print("=" * 80)
