#include "millstability/eigensolver.h"

#include <ATen/core/grad_mode.h>
#include <ATen/Parallel.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <condition_variable>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#include <sched.h>
#endif

namespace millstability::detail {
namespace {

constexpr int64_t kParallelThreshold = 4096;
constexpr int64_t kMatricesPerWorker = 256;
constexpr int64_t kSolverBatch = 4096;
// Matrix-copy budgets are shared across workers and capped per worker.
constexpr int64_t kInputCopyBytes = 256 * 1024 * 1024;

struct CpuAvailability {
    int workers = 1;
#if defined(__linux__)
    cpu_set_t affinity;
    bool has_affinity = false;
#endif
};

#if defined(__linux__)
int quota_limit(const std::filesystem::path& directory) {
    int64_t quota = -1, period = 0;
    std::ifstream v2(directory / "cpu.max");
    std::string token;
    if (v2 >> token >> period) {
        if (token == "max") return std::numeric_limits<int>::max();
        try { quota = std::stoll(token); }
        catch (...) { return std::numeric_limits<int>::max(); }
    } else {
        std::ifstream q(directory / "cpu.cfs_quota_us");
        std::ifstream p(directory / "cpu.cfs_period_us");
        if (!(q >> quota) || !(p >> period)) return std::numeric_limits<int>::max();
    }
    if (quota <= 0 || period <= 0) return std::numeric_limits<int>::max();
    return static_cast<int>(std::min<int64_t>(1 + (quota - 1) / period,
                                             std::numeric_limits<int>::max()));
}

bool contains_controller(const std::string& controllers, const std::string& name) {
    std::istringstream stream(controllers);
    std::string token;
    while (std::getline(stream, token, ',')) if (token == name) return true;
    return false;
}

int cgroup_capacity(int capacity) {
    // Resolve cgroup mount roots and apply every ancestor CPU quota.
    std::ifstream memberships("/proc/self/cgroup");
    std::string membership;
    while (std::getline(memberships, membership)) {
        const auto first = membership.find(':');
        const auto second = membership.find(':', first + 1);
        if (first == std::string::npos || second == std::string::npos) continue;
        const auto controllers = membership.substr(first + 1, second - first - 1);
        const bool unified = controllers.empty();
        if (!unified && !contains_controller(controllers, "cpu")) continue;
        const std::filesystem::path member = membership.substr(second + 1);
        std::ifstream mounts("/proc/self/mountinfo");
        std::string mount;
        while (std::getline(mounts, mount)) {
            const auto separator = mount.find(" - ");
            if (separator == std::string::npos) continue;
            std::istringstream left(mount.substr(0, separator));
            std::istringstream right(mount.substr(separator + 3));
            std::string id, parent, device, root, point, kind, source, options;
            if (!(left >> id >> parent >> device >> root >> point) ||
                !(right >> kind >> source >> options)) continue;
            if (unified ? kind != "cgroup2"
                        : (kind != "cgroup" || !contains_controller(options, "cpu"))) continue;
            auto relative = member.lexically_relative(std::filesystem::path(root));
            if (relative.empty() || *relative.begin() == "..") continue;
            const auto stop = std::filesystem::path(point).lexically_normal();
            auto current = (stop / relative).lexically_normal();
            for (;;) {
                capacity = std::min(capacity, quota_limit(current));
                if (current == stop || current == current.root_path()) break;
                const auto next = current.parent_path();
                if (next == current) break;
                current = next;
            }
        }
    }
    return std::max(1, capacity);
}
#endif

CpuAvailability cpu_availability() {
    CpuAvailability cpu{};
    cpu.workers = std::max(1u, std::thread::hardware_concurrency());
#if defined(__linux__)
    if (sched_getaffinity(0, sizeof(cpu.affinity), &cpu.affinity) == 0) {
        cpu.has_affinity = true;
        int logical = 0;
        bool topology_complete = true;
        std::set<std::pair<int, int>> physical;
        // Count physical cores in the caller's CPU affinity mask.
        for (int id = 0; id < CPU_SETSIZE; ++id) {
            if (!CPU_ISSET(id, &cpu.affinity)) continue;
            ++logical;
            const auto prefix = "/sys/devices/system/cpu/cpu" + std::to_string(id) + "/topology/";
            int package = -1, core = -1;
            std::ifstream package_file(prefix + "physical_package_id");
            std::ifstream core_file(prefix + "core_id");
            if (!(package_file >> package) || !(core_file >> core) || package < 0 || core < 0) {
                topology_complete = false;
            } else {
                physical.emplace(package, core);
            }
        }
        cpu.workers = std::max(1, topology_complete
            ? static_cast<int>(physical.size()) : logical);
    }
    cpu.workers = cgroup_capacity(cpu.workers);
#endif
    return cpu;
}

struct Dispatch {
    CpuAvailability cpu;
    int workers;
    int64_t batch_limit;
};

Dispatch dispatch_for(int64_t count, int64_t order, int device_id) {
    auto cpu = cpu_availability();
    const c10::cuda::CUDAGuard guard(device_id);
    size_t free_bytes = 0, total_bytes = 0;
    C10_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const int64_t bytes_per_matrix = order * order * int64_t(sizeof(float) + sizeof(double));
    const int64_t budget = std::max<int64_t>(1, free_bytes / 4);
    int workers = 1;
    if (count > kParallelThreshold) {
        const int64_t useful = 1 + (count - 1) / kMatricesPerWorker;
        const int64_t memory_workers = std::max<int64_t>(1,
            budget / (std::min(count, kMatricesPerWorker) * bytes_per_matrix));
        workers = static_cast<int>(std::min<int64_t>(cpu.workers,
                                                   std::min(useful, memory_workers)));
    }
    const int64_t copy_budget = std::min(kInputCopyBytes, budget / workers);
    const int64_t batch_limit = std::max<int64_t>(1,
        std::min(kSolverBatch, copy_budget / bytes_per_matrix));
    return {cpu, workers, batch_limit};
}

// Thread-local OMP/MKL limits preserve the caller's settings.
class WorkerThreadLimits {
public:
    WorkerThreadLimits() {
        (void)at::get_num_threads();  // Initialize thread-local state before setting limits.
#if defined(__linux__)
        struct Controls {
            int (*omp_get)();
            void (*omp_set)(int);
            int (*mkl_set)(int);
        };
        static const auto controls = [] {
            // Resolve symbols from ATen's loaded library to select its MKL runtime.
            Dl_info library{};
            void* handle = nullptr;
            if (dladdr(reinterpret_cast<void*>(&at::get_num_threads), &library) && library.dli_fname) {
                handle = dlopen(library.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
            }
            void* symbols = handle ? handle : RTLD_DEFAULT;
            Controls found{
                reinterpret_cast<int (*)()>(dlsym(symbols, "omp_get_max_threads")),
                reinterpret_cast<void (*)(int)>(dlsym(symbols, "omp_set_num_threads")),
                // This symbol uses C int(int); lowercase may use Fortran int(int*).
                reinterpret_cast<int (*)(int)>(dlsym(symbols, "MKL_Set_Num_Threads_Local"))};
            if (handle) dlclose(handle);
            return found;
        }();
        omp_set_ = controls.omp_set;
        mkl_set_ = controls.mkl_set;
        if (controls.omp_get && omp_set_) {
            old_omp_ = controls.omp_get();
            omp_set_(1);
        }
        if (mkl_set_) old_mkl_ = mkl_set_(1);
#endif
    }
    ~WorkerThreadLimits() {
#if defined(__linux__)
        if (mkl_set_) mkl_set_(old_mkl_);
        if (omp_set_ && old_omp_ > 0) omp_set_(old_omp_);
#endif
    }
private:
#if defined(__linux__)
    void (*omp_set_)(int) = nullptr;
    int (*mkl_set_)(int) = nullptr;
    int old_omp_ = 0, old_mkl_ = 0;
#endif
};

// Reused workers adopt the current caller's CPU affinity for each task.
class WorkerAffinity {
public:
    explicit WorkerAffinity(const CpuAvailability& cpu) {
#if defined(__linux__)
        if (cpu.has_affinity && sched_getaffinity(0, sizeof(old_), &old_) == 0) {
            changed_ = sched_setaffinity(0, sizeof(cpu.affinity), &cpu.affinity) == 0;
        }
#endif
    }
    ~WorkerAffinity() {
#if defined(__linux__)
        if (changed_) (void)sched_setaffinity(0, sizeof(old_), &old_);
#endif
    }
private:
#if defined(__linux__)
    cpu_set_t old_;
    bool changed_ = false;
#endif
};

class HostWorkers {
public:
    ~HostWorkers() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        ready_.notify_all();
        for (auto& thread : threads_) thread.join();
    }
    void ensure_capacity(int count) {
        std::lock_guard<std::mutex> lock(mutex_);
        while (threads_.size() < static_cast<size_t>(count)) {
            threads_.emplace_back([this] {
                for (;;) {
                    std::function<void()> job;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        ready_.wait(lock, [this] { return stopped_ || !jobs_.empty(); });
                        if (stopped_ && jobs_.empty()) return;
                        job = std::move(jobs_.front());
                        jobs_.pop();
                    }
                    job();
                }
            });
        }
    }
    std::future<void> submit(std::function<void()> job) {
        auto task = std::make_shared<std::packaged_task<void()>>(std::move(job));
        auto result = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            jobs_.push([task] { (*task)(); });
        }
        ready_.notify_one();
        return result;
    }
private:
    std::mutex mutex_;
    std::condition_variable ready_;
    std::queue<std::function<void()>> jobs_;
    std::vector<std::thread> threads_;
    bool stopped_ = false;
};

HostWorkers& host_workers() {
    static HostWorkers pool;
    return pool;
}

struct PrincipalBlock {
    at::Tensor matrices;
    at::Tensor output_indices;
};

at::Tensor cuda_indices(const std::vector<int64_t>& values,
                        const at::Tensor& like) {
    return at::tensor(values, at::TensorOptions().dtype(at::kLong))
        .to(like.device());
}

at::Tensor torch_spectral_radius(const at::Tensor& matrices) {
    // Nonsymmetric matrices can have complex eigenvalues.
    const auto eigenvalues = at::linalg_eigvals(matrices.to(at::kDouble));
    const auto radius = eigenvalues.abs().amax({-1});
    TORCH_CHECK(at::isfinite(radius).all().item<bool>(),
                "PyTorch produced NaN or Inf eigenvalues/EI");
    auto result = radius.to(at::kFloat);
    // Preserve EI > 1 when FP32 rounding yields exactly 1.
    const auto rounded_unstable = radius.gt(1.0).logical_and(result.le(1.0f));
    const auto one = at::ones_like(result);
    const auto above_one = at::nextafter(
        one, at::full_like(one, std::numeric_limits<float>::infinity()));
    result = at::where(rounded_unstable, above_one, result);
    TORCH_CHECK(at::isfinite(result).all().item<bool>(),
                "PyTorch EI exceeds the finite float32 range");
    return result;
}

void reduce_and_solve(const at::Tensor& matrices,
                      const at::Tensor& output_indices,
                      const at::Tensor& output) {
    std::vector<PrincipalBlock> pending;
    pending.push_back({matrices.contiguous(), output_indices});
    while (!pending.empty()) {
        auto block = std::move(pending.back());
        pending.pop_back();
        const int64_t count = block.matrices.size(0);
        const int64_t order = block.matrices.size(-1);

        // Inspect FP32 bits so only signed zeros are removed; subnormals remain.
        const auto support = block.matrices.view(at::kInt)
            .bitwise_and(0x7fffffff).ne(0).any(-2).cpu();
        const bool* active_columns = support.const_data_ptr<bool>();
        std::map<std::vector<int64_t>, std::vector<int64_t>> groups;
        for (int64_t row = 0; row < count; ++row) {
            std::vector<int64_t> active;
            for (int64_t column = 0; column < order; ++column) {
                if (active_columns[row * order + column]) active.push_back(column);
            }
            groups[std::move(active)].push_back(row);
        }

        for (const auto& group : groups) {
            const auto& active = group.first;
            const auto& rows = group.second;
            const bool whole_batch = static_cast<int64_t>(rows.size()) == count;
            const auto row_indices = whole_batch ? at::Tensor()
                : cuda_indices(rows, block.matrices);
            const auto destinations = whole_batch ? block.output_indices
                : block.output_indices.index_select(0, row_indices);
            if (active.empty()) {
                output.index_fill_(0, destinations, 0.0f);
                continue;
            }
            auto input = whole_batch ? block.matrices
                : block.matrices.index_select(0, row_indices);
            if (static_cast<int64_t>(active.size()) == order) {
                output.index_copy_(0, destinations, torch_spectral_radius(input));
                continue;
            }

            // Matching row/column permutations give [P 0; C 0], retaining P's
            // eigenvalues plus zeros. Repeat restriction to remove zero chains.
            const auto columns = cuda_indices(active, input);
            auto reduced = input.index_select(1, columns).index_select(2, columns);
            pending.push_back({std::move(reduced), destinations});
        }
    }
}

void solve_range(const at::Tensor& flat, const at::Tensor& output,
                  int64_t start, int64_t end, int64_t batch_limit) {
    for (; start < end; start += batch_limit) {
        const int64_t batch = std::min(batch_limit, end - start);
        const auto input = flat.narrow(0, start, batch);
        // Check finite values before removing rows or columns.
        TORCH_CHECK(at::isfinite(input).all().item<bool>(),
                    "matrices contain NaN or Inf before eigenvalue computation");
        const auto indices = at::arange(start, start + batch,
                                       flat.options().dtype(at::kLong));
        reduce_and_solve(input, indices, output);
    }
}

class WorkerCudaStream {
public:
    explicit WorkerCudaStream(int device)
        : device_(device), raw_(create(device)),
          stream_(c10::cuda::getStreamFromExternal(raw_, device)) {}
    ~WorkerCudaStream() {
        try {
            const c10::cuda::CUDAGuard guard(device_);
            C10_CUDA_CHECK(cudaStreamDestroy(raw_));
        } catch (...) {}
    }
    const c10::cuda::CUDAStream& get() const { return stream_; }
private:
    static cudaStream_t create(int device) {
        const c10::cuda::CUDAGuard guard(device);
        cudaStream_t result;
        C10_CUDA_CHECK(cudaStreamCreateWithFlags(&result, cudaStreamNonBlocking));
        return result;
    }
    int device_;
    cudaStream_t raw_;
    c10::cuda::CUDAStream stream_;
};

struct WorkerResult {
    std::shared_ptr<WorkerCudaStream> stream;
    at::cuda::CUDAEvent complete;
    std::future<void> future;
    std::exception_ptr error;
};

std::shared_ptr<WorkerCudaStream> worker_cuda_stream(int device) {
    // Each worker reuses one stream per device for allocator-cache reuse.
    thread_local std::map<int, std::shared_ptr<WorkerCudaStream>> streams;
    auto& stream = streams[device];
    if (!stream) stream = std::make_shared<WorkerCudaStream>(device);
    return stream;
}

void parallel_solve(const at::Tensor& flat, const at::Tensor& output,
                    const Dispatch& dispatch) {
    auto& pool = host_workers();
    pool.ensure_capacity(dispatch.workers);
    const auto caller = c10::cuda::getCurrentCUDAStream(flat.get_device());
    auto producer = std::make_shared<at::cuda::CUDAEvent>();
    producer->record(caller);

    std::vector<std::shared_ptr<WorkerResult>> results;
    results.reserve(dispatch.workers);
    std::exception_ptr error;
    try {
        for (int worker = 0; worker < dispatch.workers; ++worker) {
            auto result = std::make_shared<WorkerResult>();
            const int64_t start = flat.size(0) * worker / dispatch.workers;
            const int64_t end = flat.size(0) * (worker + 1) / dispatch.workers;
            result->future = pool.submit([flat, output, producer, result, dispatch, start, end] {
                try {
                    const WorkerAffinity affinity(dispatch.cpu);
                    const at::NoGradGuard no_grad;
                    const WorkerThreadLimits limits;
                    result->stream = worker_cuda_stream(flat.get_device());
                    const c10::cuda::CUDAStreamGuard guard(result->stream->get());
                    // Keep input/output storage alive through queued worker operations.
                    c10::cuda::CUDACachingAllocator::recordStream(
                        flat.storage().data_ptr(), result->stream->get());
                    c10::cuda::CUDACachingAllocator::recordStream(
                        output.storage().data_ptr(), result->stream->get());
                    // Wait for the caller's queued matrix construction.
                    producer->block(result->stream->get());
                    solve_range(flat, output, start, end, dispatch.batch_limit);
                } catch (...) {
                    result->error = std::current_exception();
                }
                try {
                    // Record completion after successful or failed tasks.
                    if (result->stream) result->complete.record(result->stream->get());
                } catch (...) {
                    if (!result->error) result->error = std::current_exception();
                    // Synchronize on event failure to protect tensor storage.
                    try { if (result->stream) result->stream->get().synchronize(); } catch (...) {}
                }
            });
            results.push_back(std::move(result));
        }
    } catch (...) {
        error = std::current_exception();
    }

    // Drain tasks and queue caller-stream waits before propagating errors.
    for (const auto& result : results) {
        try { result->future.get(); }
        catch (...) { if (!error) error = std::current_exception(); }
        if (result->error && !error) error = result->error;
        try { result->complete.block(caller); }
        catch (...) {
            if (!error) error = std::current_exception();
            try { if (result->stream) result->stream->get().synchronize(); } catch (...) {}
        }
    }
    if (error) std::rethrow_exception(error);
}

}  // namespace

at::Tensor spectral_radius(const at::Tensor& matrices) {
    TORCH_CHECK(matrices.is_cuda(), "matrices must be a CUDA tensor");
    TORCH_CHECK(matrices.scalar_type() == at::kFloat,
                "matrices must have dtype float32");
    TORCH_CHECK(matrices.dim() >= 2 && matrices.size(-2) == matrices.size(-1),
                "matrices must have shape [..., n, n] (square matrices)");
    const int64_t n = matrices.size(-1);
    TORCH_CHECK(n > 0 && n <= 46340, "matrix order n must be in [1, 46340]");
    TORCH_CHECK(!matrices.requires_grad(), "spectral_radius does not support autograd");
    const at::NoGradGuard no_grad;
    const c10::cuda::CUDAGuard guard(matrices.device());

    auto shape = matrices.sizes().vec();
    shape.resize(shape.size() - 2);
    auto ei = at::empty(shape, matrices.options());
    const int64_t count = ei.numel();
    if (count == 0) return ei;

    const auto flat = matrices.reshape({count, n, n});
    const auto output = ei.reshape({count});
    const auto dispatch = dispatch_for(count, n, matrices.get_device());
    if (dispatch.workers == 1) {
        const WorkerThreadLimits limits;
        solve_range(flat, output, 0, count, dispatch.batch_limit);
    } else {
        parallel_solve(flat, output, dispatch);
    }
    return ei;
}

}  // namespace millstability::detail
