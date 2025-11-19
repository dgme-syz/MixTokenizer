from MixTokenizer.core.decode import ACAutomaton
import pickle

# 创建 AC 自动机
ac = ACAutomaton()

# 添加模式
ac.add_pattern([1, 2, 3, 5, 6], 100)
ac.add_pattern([4, 7], 200)

# 构建 AC 自动机
ac.build()

# 测试搜索
text = [1, 2, 3, 4, 7, 8]
res = ac.search(text)
print("Before Pickle:", res)
# 输出应该是: [(-1, 0, 1), (100, 1, 4), (200, 4, 6), (-1, 6, 7)]

# 序列化并保存到字节流
serialized_ac = pickle.dumps(ac)

# 尝试反序列化并捕获异常
try:
    ac2 = pickle.loads(serialized_ac)
    # 再次进行搜索，验证反序列化后的对象是否正确
    res2 = ac2.search(text)
    print("After Pickle:", res2)
except Exception as e:
    print("Error during unpickling:", e)
