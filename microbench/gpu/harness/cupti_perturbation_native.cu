// Declared control-only profiler perturbation diagnostic, never fit data.
// Same precompiled instrumented binary in "none" and "software_serial" modes.
// globaltimer envelopes omit kernel prelude/epilogue; not ground-truth latency.
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <chrono>
#include <thread>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#ifndef TV_PERTURBATION_WORKLOAD
#define TV_PERTURBATION_WORKLOAD 0
#endif
#ifndef TV_LOCAL_SLOTS
#define TV_LOCAL_SLOTS 128
#endif
#ifndef TV_RECORD_SM_ID
#define TV_RECORD_SM_ID 0
#endif
#if TV_LOCAL_SLOTS != 32 && TV_LOCAL_SLOTS != 64 && TV_LOCAL_SLOTS != 128 && TV_LOCAL_SLOTS != 256
#error Unsupported declared local slot count
#endif
#if TV_PERTURBATION_WORKLOAD == 0
static const char *workload = "fma";
#elif TV_PERTURBATION_WORKLOAD == 1
static const char *workload = "tensor";
#elif TV_PERTURBATION_WORKLOAD == 2
static const char *workload = "local";
#else
#error Unknown perturbation workload
#endif

struct Timestamp {
  uint64_t start_ns, end_ns;
  uint32_t device, context, stream, correlation;
  char name[256];
};
struct BodyInterval {
  uint64_t start, end;
#if TV_RECORD_SM_ID
  uint32_t start_sm, end_sm, sm_id_count;
#endif
};
static void check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cuda_error=%s\n", cudaGetErrorString(status));
    std::exit(1);
  }
}
__global__ void perturbation_eviction(uint32_t *buffer, size_t words) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
       i += gridDim.x * blockDim.x) buffer[i] = 0;
}
__global__ void perturbation_body(float *output, BodyInterval *intervals,
                                  int iterations) {
  uint64_t start = 0, end = 0;
#if TV_RECORD_SM_ID
  uint32_t start_sm = 0, end_sm = 0, sm_id_count = 0;
  if (threadIdx.x == 0) {
    asm volatile("mov.u32 %0, %%smid;" : "=r"(start_sm) :: "memory");
    asm volatile("mov.u32 %0, %%nsmid;" : "=r"(sm_id_count));
  }
#endif
  if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start) :: "memory");
  __syncthreads();
  float value = 1.0f + threadIdx.x;
#if TV_PERTURBATION_WORKLOAD == 0
  for (int i = 0; i < iterations; ++i)
    asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                 : "+f"(value) : "f"(0.999999f), "f"(0.000001f));
#elif TV_PERTURBATION_WORKLOAD == 1
  float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  const unsigned ones = 0x3c003c00;  // two exactly represented FP16 ones
  for (int i = 0; i < iterations; ++i)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(ones));
  value = c0 + c1 + c2 + c3;
#else
  // Deliberate local-memory traffic, not a claim about compiler-inferred spills.
  volatile float local_values[TV_LOCAL_SLOTS];
  for (int j = 0; j < TV_LOCAL_SLOTS; ++j) local_values[j] = value;
  for (int i = 0; i < iterations; ++i) {
    float x = local_values[i & (TV_LOCAL_SLOTS - 1)];
    asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                 : "+f"(x) : "f"(0.999999f), "f"(0.000001f));
    local_values[i & (TV_LOCAL_SLOTS - 1)] = x;
  }
  value = 0;
  for (int j = 0; j < TV_LOCAL_SLOTS; ++j) value += local_values[j];
#endif
  output[blockIdx.x * blockDim.x + threadIdx.x] = value;
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end) :: "memory");
#if TV_RECORD_SM_ID
    asm volatile("mov.u32 %0, %%smid;" : "=r"(end_sm) :: "memory");
    intervals[blockIdx.x] = {start, end, start_sm, end_sm, sm_id_count};
#else
    intervals[blockIdx.x] = {start, end};
#endif
  }
}
int main(int argc, char **argv) {
  // MODE LIBRARY PROGRAMS ITERATIONS [counter_only]. Timing uses 11 x 32 launches.
  if (argc != 5 && argc != 6) return 2;
  const bool counter_only = argc == 6 && std::strcmp(argv[5], "counter_only") == 0;
#if TV_RECORD_SM_ID
  if (!counter_only) return 2;
#endif
  if (argc == 6 && (!counter_only || std::strcmp(argv[1], "none") != 0)) return 2;
  const int samples = counter_only ? 1 : 352;
  const bool enabled = std::strcmp(argv[1], "software_serial") == 0;
  if (!enabled && std::strcmp(argv[1], "none") != 0) return 2;
  const int programs = std::atoi(argv[3]), iterations = std::atoi(argv[4]);
  if ((programs != 48 && programs != 96 && programs != 192 && programs != 384) ||
      (iterations != 16 && iterations != 65536)) return 2;
  if (programs == 192 && !counter_only) return 2;
  if (counter_only && (TV_PERTURBATION_WORKLOAD != 2 || iterations != 65536)) return 2;
  void *library = nullptr;
  int (*flush)() = nullptr, (*close)() = nullptr;
  size_t (*count)() = nullptr, (*dropped)() = nullptr;
  int (*get)(size_t, Timestamp *) = nullptr;
  void (*clear)() = nullptr;
  if (enabled) {
    library = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (!library) return 3;
    auto init = reinterpret_cast<int (*)()>(dlsym(library, "tv_cupti_init"));
    auto hardware = reinterpret_cast<int (*)()>(dlsym(library, "tv_cupti_hardware_trace"));
    auto kind = reinterpret_cast<int (*)()>(dlsym(library, "tv_cupti_activity_kind"));
    flush = reinterpret_cast<int (*)()>(dlsym(library, "tv_cupti_flush"));
    close = reinterpret_cast<int (*)()>(dlsym(library, "tv_cupti_close"));
    count = reinterpret_cast<size_t (*)()>(dlsym(library, "tv_cupti_count"));
    dropped = reinterpret_cast<size_t (*)()>(dlsym(library, "tv_cupti_dropped"));
    get = reinterpret_cast<int (*)(size_t, Timestamp *)>(dlsym(library, "tv_cupti_get"));
    clear = reinterpret_cast<void (*)()>(dlsym(library, "tv_cupti_clear"));
    if (!init || !hardware || !kind || !flush || !close || !count || !dropped ||
        !get || !clear || hardware() != 0 || kind() != 3 || init()) return 3;
  }
  std::printf("role=control,eligible_for_fit=false,mode=%s,programs=%d,iterations=%d\n",
              argv[1], programs, iterations);
  std::printf("workload=%s\n", workload);
#if TV_PERTURBATION_WORKLOAD == 2
  std::printf("local_slots=%d\n", TV_LOCAL_SLOTS);
#endif
  if (counter_only) std::printf("purpose=counter_only,measurement_samples=1\n");
#if TV_RECORD_SM_ID
  std::printf("sm_observation=endpoint_v1\n");
#endif
  auto drain = [&](int sample) {
    if (!enabled) return true;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    do {
      if (flush()) return false;
      if (count() >= 2) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    bool valid = count() == 2 && dropped() == 0;
    uint64_t prior_end = 0;
    uint32_t context = 0, stream_id = 0;
    for (size_t i = 0; i < count(); ++i) {
      Timestamp r{};
      if (get(i, &r)) return false;
      std::printf("cupti,%d,%zu,%llu,%llu,%u,%u,%u,%s\n", sample, i,
          (unsigned long long)r.start_ns, (unsigned long long)r.end_ns,
          r.device, r.context, r.stream, r.name);
      const char *expected = i == 0 ? "perturbation_eviction" : "perturbation_body";
      if (i == 0) { context = r.context; stream_id = r.stream; }
      valid &= r.start_ns > 0 && r.end_ns > r.start_ns && r.start_ns >= prior_end &&
               r.device == 0 && r.context == context && r.stream == stream_id &&
               std::strstr(r.name, expected) != nullptr;
      prior_end = r.end_ns;
    }
    std::printf("delivery,%d,%zu,%zu,%d\n", sample, count(), dropped(), valid);
    std::fflush(stdout);
    if (valid) clear();
    return valid;
  };
  int l2 = 0, sms = 0;
  check(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
  check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  if (l2 <= 0 || sms <= 0) return 4;
  std::printf("l2_bytes=%d,sm_count=%d,eviction_bytes=%zu\n", l2, sms, 2 * size_t(l2));
  uint32_t *sweep;
  float *output;
  BodyInterval *intervals;
  check(cudaMalloc(&sweep, 2 * size_t(l2)));
  check(cudaMalloc(&output, programs * 128 * sizeof(float)));
  check(cudaMalloc(&intervals, programs * (samples + 1) * sizeof(BodyInterval)));
  cudaStream_t stream;
  check(cudaStreamCreate(&stream));
  auto launch = [&](int slot) {
    perturbation_eviction<<<sms * 4, 256, 0, stream>>>(sweep, 2 * size_t(l2) / 4);
    perturbation_body<<<programs, 128, 0, stream>>>(output, intervals + slot * programs, iterations);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));
  };
  // Precompiled kernel warmup, same duration policy in both modes.
  auto start = std::chrono::steady_clock::now();
  int warmups = 0;
  bool valid = true;
  if (!counter_only) do {
    launch(samples);
    valid = drain(-1);
    ++warmups;
  } while (valid && std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < 0.5);
  std::printf("warmup_launches=%d\n", warmups);
  int completed = 0;
  for (int sample = 0; sample < samples && valid; ++sample) {
    launch(sample);
    ++completed;
    valid = drain(sample);
  }
  std::vector<BodyInterval> host(completed * programs);
  check(cudaMemcpy(host.data(), intervals, host.size() * sizeof(BodyInterval), cudaMemcpyDeviceToHost));
  for (int sample = 0; sample < completed; ++sample)
    for (int block = 0; block < programs; ++block) {
      auto r = host[sample * programs + block];
      std::printf("body,%d,%d,%llu,%llu\n", sample, block,
                  (unsigned long long)r.start, (unsigned long long)r.end);
      valid &= r.start > 0 && r.end > r.start;
#if TV_RECORD_SM_ID
      std::printf("cta_sm,%d,%d,%u,%u,%u\n", sample, block, r.start_sm, r.end_sm, r.sm_id_count);
      valid &= r.sm_id_count > 0 && r.start_sm < r.sm_id_count && r.end_sm < r.sm_id_count;
#endif
    }
  std::vector<float> values(programs * 128), reference(128);
  check(cudaMemcpy(values.data(), output, values.size() * sizeof(float), cudaMemcpyDeviceToHost));
  for (int t = 0; t < 128; ++t) {
    float x = 1.0f + t;
#if TV_PERTURBATION_WORKLOAD == 0
    for (int i = 0; i < iterations; ++i) x = std::fma(x, 0.999999f, 0.000001f);
#elif TV_PERTURBATION_WORKLOAD == 1
    x = float(64 * iterations);
#else
    float slots[TV_LOCAL_SLOTS];
    for (int j = 0; j < TV_LOCAL_SLOTS; ++j) slots[j] = x;
    for (int i = 0; i < iterations; ++i) slots[i & (TV_LOCAL_SLOTS - 1)] = std::fma(slots[i & (TV_LOCAL_SLOTS - 1)], 0.999999f, 0.000001f);
    x = 0;
    for (int j = 0; j < TV_LOCAL_SLOTS; ++j) x += slots[j];
#endif
    reference[t] = x;
  }
  for (size_t i = 0; i < values.size(); ++i) valid &= values[i] == reference[i % 128];
  const int close_status = enabled ? close() : 0;
  std::printf("completed=%d,valid=%d,close_status=%d,eligible_for_fit=false\n", completed, valid, close_status);
  check(cudaStreamDestroy(stream));
  check(cudaFree(intervals)); check(cudaFree(output)); check(cudaFree(sweep));
  if (library) dlclose(library);
  return valid && completed == samples && close_status == 0 ? 0 : 5;
}
