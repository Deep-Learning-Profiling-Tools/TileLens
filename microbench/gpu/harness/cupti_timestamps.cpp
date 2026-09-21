// Control-only CUPTI 13.1 HES collector. Build against matching SDK headers/libs.
// No injected CUDA events, profiler counters, or CPU API durations are timed.
#include <cuda.h>
#include <cupti_activity.h>
#include <cupti_version.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include <thread>
#include <chrono>
#include <cstdio>

#ifndef TV_CUPTI_API_VERSION
#define TV_CUPTI_API_VERSION 130100
#endif
#ifndef TV_CUPTI_KERNEL_RECORD
#define TV_CUPTI_KERNEL_RECORD CUpti_ActivityKernel11
#endif
#ifndef TV_CUPTI_FLUSH_PERIOD_MS
#define TV_CUPTI_FLUSH_PERIOD_MS 0
#endif
#ifndef TV_CUPTI_PER_THREAD_BUFFERS
#define TV_CUPTI_PER_THREAD_BUFFERS 1
#endif
#ifndef TV_CUPTI_POLL_MS
#define TV_CUPTI_POLL_MS 0
#endif
#ifndef TV_CUPTI_HARDWARE_TRACE
#define TV_CUPTI_HARDWARE_TRACE 1
#endif
#ifndef TV_CUPTI_ACTIVITY_KIND
#define TV_CUPTI_ACTIVITY_KIND CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL
#endif

struct KernelTimestamp {
  uint64_t start_ns, end_ns;
  uint32_t device, context, stream, correlation;
  char name[256];
};

static std::mutex records_mutex;
static std::vector<KernelTimestamp> records;
static std::atomic<int> callback_error{0};
static std::atomic<size_t> dropped{0};
static bool initialized = false;
static std::atomic<bool> stop_polling{false};
static std::thread polling_thread;
static CUpti_SubscriberHandle state_subscriber;

static void CUPTIAPI state_callback(void *, CUpti_CallbackDomain domain,
                                    CUpti_CallbackId id, const void *data) {
  if (domain != CUPTI_CB_DOMAIN_STATE) return;
  auto *state = static_cast<const CUpti_StateData *>(data);
  // State notifications can report unsupported HES/fallback or data loss
  // asynchronously. Fail closed even when API return codes report success.
  callback_error = state && state->notification.result != CUPTI_SUCCESS
                     ? state->notification.result : -10;
  std::fprintf(stderr, "CUPTI state id=%u result=%d message=%s\n",
               static_cast<unsigned>(id), callback_error.load(),
               state && state->notification.message ? state->notification.message : "(none)");
}

static void CUPTIAPI request_buffer(uint8_t **buffer, size_t *size,
                                    size_t *max_records) {
  *size = 8 * 1024 * 1024;
  *max_records = 0;
  *buffer = static_cast<uint8_t *>(std::malloc(*size));
  if (!*buffer) {
    *size = 0;
    callback_error = -2;
  }
}

static void CUPTIAPI complete_buffer(CUcontext context, uint32_t stream,
                                     uint8_t *buffer, size_t, size_t valid) {
  CUpti_Activity *record = nullptr;
  CUptiResult status;
  while ((status = cuptiActivityGetNextRecord(buffer, valid, &record)) == CUPTI_SUCCESS) {
    if (record->kind != TV_CUPTI_ACTIVITY_KIND) continue;
    auto *kernel = reinterpret_cast<TV_CUPTI_KERNEL_RECORD *>(record);
    if (!kernel->name || std::strlen(kernel->name) >= 256) {
      callback_error = -3;
      continue;
    }
    KernelTimestamp item{kernel->start, kernel->end, kernel->deviceId,
                         kernel->contextId, kernel->streamId, kernel->correlationId, {0}};
    std::strcpy(item.name, kernel->name);
    try {
      std::lock_guard<std::mutex> lock(records_mutex);
      records.push_back(item);
    } catch (...) {
      callback_error = -2;
    }
  }
  if (status != CUPTI_ERROR_MAX_LIMIT_REACHED) callback_error = status;
  size_t count = 0;
  status = cuptiActivityGetNumDroppedRecords(context, stream, &count);
  if (status != CUPTI_SUCCESS) callback_error = status;
  dropped += count;
  std::free(buffer);
}

extern "C" int tv_cupti_init() {
  if (initialized) return -4;
  if (cuInit(0) != CUDA_SUCCESS) return -5;
  CUcontext context = nullptr;
  if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || context != nullptr) return -6;
  uint32_t version = 0;
  CUptiResult status = cuptiGetVersion(&version);
  if (status != CUPTI_SUCCESS) return status;
  // ABI and record type must be selected together from the matching SDK.
  // Reject any runtime version other than the explicitly compiled version.
  if (version != TV_CUPTI_API_VERSION) return -7;
  status = cuptiSubscribe(&state_subscriber, state_callback, nullptr);
  if (status != CUPTI_SUCCESS) return status;
  status = cuptiEnableDomain(1, state_subscriber, CUPTI_CB_DOMAIN_STATE);
  if (status != CUPTI_SUCCESS) return status;
  if (TV_CUPTI_HARDWARE_TRACE) {
    status = cuptiActivityEnableHWTrace(1);
    if (status != CUPTI_SUCCESS) return status;
  }
  // Configure buffer policy before callbacks/kinds, as required by CUPTI.
  // Diagnostic global-buffer builds have separate fingerprints and datasets.
  uint8_t per_thread_buffers = TV_CUPTI_PER_THREAD_BUFFERS;
  size_t attribute_size = sizeof(per_thread_buffers);
  status = cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_PER_THREAD_ACTIVITY_BUFFER,
                                    &attribute_size, &per_thread_buffers);
  if (status != CUPTI_SUCCESS) return status;
  status = cuptiActivityRegisterCallbacks(request_buffer, complete_buffer);
  if (status != CUPTI_SUCCESS) return status;
  // Default to on-demand delivery (0). Periodic-delivery diagnostic builds use
  // an explicit macro and separate fingerprint; do not merge their datasets.
  status = cuptiActivityFlushPeriod(TV_CUPTI_FLUSH_PERIOD_MS);
  if (status != CUPTI_SUCCESS) return status;
  // Retain launch correlation for graph nodes. API records are consumed only
  // as collector metadata; CPU API intervals never become kernel timings.
  status = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER);
  if (status != CUPTI_SUCCESS) return status;
  status = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
  if (status != CUPTI_SUCCESS) return status;
  status = cuptiActivityEnable(TV_CUPTI_ACTIVITY_KIND);
  if (status != CUPTI_SUCCESS) return status;
  initialized = true;
  // Optional, separately fingerprinted diagnostic: drain while CUDA work is
  // running, rather than waiting for a long graph synchronization to return.
  // This differs from CUPTI's periodic full-buffer delivery policy.
  if (TV_CUPTI_POLL_MS > 0) {
    stop_polling = false;
    polling_thread = std::thread([] {
      while (!stop_polling) {
        CUptiResult result = cuptiActivityFlushAll(0);
        if (result != CUPTI_SUCCESS) { callback_error = result; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(TV_CUPTI_POLL_MS));
      }
    });
  }
  return 0;
}

extern "C" int tv_cupti_flush() {
  // Caller synchronizes CUDA first and polls for the declared record count.
  // Do not force-deliver incomplete records during an active profiling session.
  CUptiResult status = cuptiActivityFlushAll(0);
  if (status != CUPTI_SUCCESS) return status;
  return callback_error;
}

extern "C" size_t tv_cupti_dropped() { return dropped; }
extern "C" int tv_cupti_hardware_trace() { return TV_CUPTI_HARDWARE_TRACE; }
extern "C" int tv_cupti_activity_kind() { return TV_CUPTI_ACTIVITY_KIND; }

extern "C" size_t tv_cupti_count() {
  std::lock_guard<std::mutex> lock(records_mutex);
  return records.size();
}

extern "C" int tv_cupti_get(size_t index, KernelTimestamp *result) {
  std::lock_guard<std::mutex> lock(records_mutex);
  if (index >= records.size()) return -8;
  *result = records[index];
  return 0;
}

extern "C" void tv_cupti_clear() {
  std::lock_guard<std::mutex> lock(records_mutex);
  records.clear();
  // Dropped records and callback failures are sticky: never erase audit errors.
}

extern "C" int tv_cupti_close() {
  if (!initialized) return 0;
  stop_polling = true;
  if (polling_thread.joinable()) polling_thread.join();
  CUcontext context = nullptr;
  if (cuCtxGetCurrent(&context) != CUDA_SUCCESS) return -6;
  if (context && cuCtxSynchronize() != CUDA_SUCCESS) return -9;
  // Forced flushing belongs only to teardown, not measured-record validation.
  // Finalize while the context is still alive, before process/DSO destruction.
  CUptiResult status = cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
  if (status != CUPTI_SUCCESS) return status;
  status = cuptiFinalize();
  if (status != CUPTI_SUCCESS) return status;
  initialized = false;
  return callback_error;
}
