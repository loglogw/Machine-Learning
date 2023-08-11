import ctypes

# 加载动态库
mylib = ctypes.cdll.LoadLibrary("mylib.so")

# 调用动态库中的函数
result = mylib.myfunction(42)
print(result)
