// Independent control-only reproduction of the cache-counter protocol.
// No Triton/PyTorch/CUPTI collector; ncu owns profiling and no latency is fitted.
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void check(cudaError_t error) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
    std::exit(1);
  }
}

__global__ void initialize_control(float *x, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = 1.0f;
}

__global__ void native_cache_read_control(const float *x, float *y, size_t n) {
  size_t base = blockIdx.x * 1024 + threadIdx.x;
  for (int offset = 0; offset < 1024; offset += blockDim.x) {
    size_t i = base + offset;
    if (i < n) {
      float value;
      asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(value) : "l"(x + i));
      y[i] = value + 1.0f;
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  int mib = std::atoi(argv[1]);
  bool evict = std::strcmp(argv[2], "zero") == 0;
  if ((mib != 3 && mib != 12 && mib != 48) ||
      (!evict && std::strcmp(argv[2], "none") != 0)) return 2;
  size_t n = static_cast<size_t>(mib) * 1024 * 1024 / sizeof(float);
  float *x, *y;
  void *sweep;
  int l2;
  check(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
  check(cudaMalloc(&x, n * sizeof(float)));
  check(cudaMalloc(&y, n * sizeof(float)));
  check(cudaMalloc(&sweep, 2 * static_cast<size_t>(l2)));
  initialize_control<<<(n + 255) / 256, 256>>>(x, n);
  native_cache_read_control<<<(n + 1023) / 1024, 128>>>(x, y, n);
  check(cudaDeviceSynchronize());
  check(cudaProfilerStart());
  for (int launch = 0; launch < 3; ++launch) {
    if (launch == 2 && evict) check(cudaMemset(sweep, 0, 2 * static_cast<size_t>(l2)));
    native_cache_read_control<<<(n + 1023) / 1024, 128>>>(x, y, n);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
  }
  check(cudaProfilerStop());
  float *host = static_cast<float *>(std::malloc(n * sizeof(float)));
  if (!host) return 3;
  check(cudaMemcpy(host, y, n * sizeof(float), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < n; ++i) {
    if (host[i] != 2.0f) return 4;
  }
  std::printf("native control %d MiB eviction=%s numerical=passed\n", mib, argv[2]);
  std::free(host);
  check(cudaFree(sweep));
  check(cudaFree(y));
  check(cudaFree(x));
}
