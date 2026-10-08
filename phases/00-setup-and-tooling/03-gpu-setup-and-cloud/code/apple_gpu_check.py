import mlx.core as mx
import time

size = 5000
a = mx.random.normal((size, size))
b = mx.random.normal((size, size))
mx.eval(a, b)

_ = mx.eval(a @ b)  # warmup
start = time.perf_counter()
c = a @ b
mx.eval(c)  # MLX is lazy; eval forces computation
print(f"MLX GPU: {time.perf_counter() - start:.3f}s")