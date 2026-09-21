// Control-only CUPTI 13.1 HES collector. Build against matching SDK headers/libs.
// No injected CUDA events, profiler counters, or CPU API durations are timed.
#include <cuda.h>
#include <cupti.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

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
    if (record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) continue;
    auto *kernel = reinterpret_cast<CUpti_ActivityKernel11 *>(record);
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
  // The record layout below is specifically the CUDA 13.1 ABI.
  if (version != 130100) return -7;
  status = cuptiActivityEnableHWTrace(1);
  if (status != CUPTI_SUCCESS) return status;
  status = cuptiActivityRegisterCallbacks(request_buffer, complete_buffer);
  if (status != CUPTI_SUCCESS) return status;
  // Retain launch correlation for graph nodes. API records are consumed only
  // as collector metadata; CPU API intervals never become kernel timings.
  status = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER);
  if (status != CUPTI_SUCCESS) return status;
  status = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
  if (status != CUPTI_SUCCESS) return status;
  status = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  if (status != CUPTI_SUCCESS) return status;
  initialized = true;
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
