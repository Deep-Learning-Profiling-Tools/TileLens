// Control-only HES delivery reproduction without Python, Torch, or Triton.
// Precompiled kernels, explicit L2 eviction, no CUDA events or latency fitting.
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <chrono>
#include <thread>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

struct Timestamp {
  uint64_t start_ns, end_ns;
  uint32_t device, context, stream, correlation;
  char name[256];
};

static void check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(1);
  }
}

__global__ void native_hes_eviction(uint32_t *buffer, size_t words) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < words; i += gridDim.x * blockDim.x) buffer[i] = 0;
}

__global__ void native_hes_compute(float *output, int iterations) {
  float value = 1.0f + threadIdx.x;
  for (int i = 0; i < iterations; ++i)
    asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                 : "+f"(value) : "f"(1.000001f), "f"(0.000001f));
  output[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

int main(int argc, char **argv) {
  if (argc != 4 && argc != 5) return 2;
  int pairs = std::atoi(argv[2]), iterations = std::atoi(argv[3]);
  int replays = argc == 5 ? std::atoi(argv[4]) : 1;
  if (replays != 1 && replays != 16) return 2;
  if ((pairs != 11 && pairs != 32 && pairs != 352) ||
      (iterations != 16 && iterations != 65536)) return 2;
  void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!library) { std::fprintf(stderr, "%s\n", dlerror()); return 3; }
  auto method = reinterpret_cast<int (*)()>(dlsym(library, "tv_cupti_hardware_trace"));
  std::printf("hardware_trace=%d,diagnostic_only=true\n", method ? method() : -1);
  auto init = reinterpret_cast<int (*)()>(dlsym(library, "tv_cupti_init"));
  auto flush = reinterpret_cast<int (*)()>(dlsym(library, "tv_cupti_flush"));
  auto count = reinterpret_cast<size_t (*)()>(dlsym(library, "tv_cupti_count"));
  auto dropped = reinterpret_cast<size_t (*)()>(dlsym(library, "tv_cupti_dropped"));
  auto get = reinterpret_cast<int (*)(size_t, Timestamp *)>(dlsym(library, "tv_cupti_get"));
  auto clear = reinterpret_cast<void (*)()>(dlsym(library, "tv_cupti_clear"));
  auto close = reinterpret_cast<int (*)()>(dlsym(library, "tv_cupti_close"));
  if (!init || !flush || !count || !dropped || !get || !clear || !close) return 3;
  int status = init();
  if (status) { std::fprintf(stderr, "collector_init=%d\n", status); return 4; }
  auto dump = [&](const char *phase) {
    std::printf("phase=%s,count=%zu,dropped=%zu\n", phase, count(), dropped());
    for (size_t i = 0; i < count(); ++i) {
      Timestamp r{};
      if (get(i, &r)) std::exit(5);
      std::printf("%s,%zu,%llu,%llu,%u,%u,%u,%s\n", phase, i,
                  static_cast<unsigned long long>(r.start_ns),
                  static_cast<unsigned long long>(r.end_ns),
                  r.device, r.context, r.stream, r.name);
    }
    std::fflush(stdout);
  };
  auto drain = [&](size_t expected, const char *phase) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    do {
      if (flush()) return false;
      if (count() >= expected) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    dump(phase);
    bool timestamps_valid = true;
    for (size_t i = 0; i < count(); ++i) {
      Timestamp r{};
      if (get(i, &r) || !r.start_ns || r.end_ns <= r.start_ns) timestamps_valid = false;
    }
    std::printf("phase=%s,timestamps_valid=%d\n", phase, timestamps_valid);
    return count() == expected && dropped() == 0 && timestamps_valid;
  };
  int l2 = 0;
  check(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
  if (l2 <= 0) return 6;
  uint32_t *sweep;
  float *output;
  check(cudaMalloc(&sweep, 2 * static_cast<size_t>(l2)));
  check(cudaMalloc(&output, 48 * 128 * sizeof(float)));
  cudaStream_t stream;
  check(cudaStreamCreate(&stream));
  auto launch = [&]() {
    native_hes_eviction<<<192, 256, 0, stream>>>(sweep, 2 * static_cast<size_t>(l2) / 4);
    native_hes_compute<<<48, 128, 0, stream>>>(output, iterations);
    check(cudaGetLastError());
  };
  launch();
  check(cudaStreamSynchronize(stream));
  bool valid = drain(2, "warmup");
  if (valid) {
    clear();
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < pairs; ++i) launch();
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, 0));
    for (int replay = 0; replay < replays && valid; ++replay) {
      clear();
      check(cudaGraphLaunch(executable, stream));
      check(cudaStreamSynchronize(stream));
      char phase[32];
      std::snprintf(phase, sizeof(phase), "graph_%d", replay);
      valid = drain(2 * pairs, phase);
    }
    check(cudaGraphExecDestroy(executable));
    check(cudaGraphDestroy(graph));
  }
  status = close();
  dump("teardown");
  std::printf("delivery_complete=%d,close_status=%d,eligible_for_fit=false\n", valid, status);
  check(cudaStreamDestroy(stream));
  check(cudaFree(output));
  check(cudaFree(sweep));
  dlclose(library);
  return valid && !status ? 0 : 7;
}
