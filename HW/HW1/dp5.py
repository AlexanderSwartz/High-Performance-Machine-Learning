import sys
import numpy as np
import time

def dp(A, B):
    R = 0.0
    R = np.dot(A, B)
    return R

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <size> [measurements]")
        return 1

    N = int(sys.argv[1])
    measurements = int(sys.argv[2])

    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)

    half = measurements // 2
    timed = measurements - half

    # warm-up
    result = 0.0
    for i in range(half):
        result = dp(A, B)

    start_ns = time.monotonic_ns()
    for i in range(timed):
        result = dp(A, B)
    end_ns = time.monotonic_ns()

    elapsed = (end_ns - start_ns) / 1e9
    average_time = elapsed / timed

    bytes_per_iteration = 2 * 4  # 2 loads (A,B) * 4 bytes each
    total_bytes = bytes_per_iteration * N * timed
    total_bytes_gb = total_bytes / 1e9
    bandwidth = total_bytes_gb / elapsed

    flops_per_iteration = 2 # 1 add + 1 mult
    total_flops = flops_per_iteration * N * timed
    total_gflops = total_flops / 1e9
    total_glops = total_gflops / elapsed

    print(f"N: {N}  <T>: {average_time:.3f} sec  B: {bandwidth:.3f} GB/sec  F: {total_glops:.3f} GFLOP/sec")
    return 0

if __name__ == '__main__':
    sys.exit(main())
