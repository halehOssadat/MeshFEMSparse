#pragma once
#include <cuda_tool/cuda_tool.h>
#include <MeshFEMCore/GlobalBenchmark.hh>
#include <optional>
#include <string>

namespace uipc::backend::cuda
{
// A MeshFEM benchmark timer section that waits for the GPU when it starts and when it stops, so that it
// measures the GPU work launched inside it (not just the asynchronous launches). This matches libuipc's
// timers: its SimEngine installs cuda_tool::wait_device() as Timer's sync function, which every Timer
// calls on construction and destruction.
// Never create one during CUDA graph capture (synchronizing is not allowed there).
class SyncedTimer
{
  public:
    explicit SyncedTimer(const std::string& name)
    {
        cuda_tool::wait_device();
        m_timer.emplace(name);
    }

    ~SyncedTimer()
    {
        cudaDeviceSynchronize();  // no CUDA_TOOL_CHECK: a destructor must not throw
        m_timer.reset();
    }

    SyncedTimer(const SyncedTimer&)            = delete;
    SyncedTimer& operator=(const SyncedTimer&) = delete;

  private:
    std::optional<MeshFEM::BENCHMARK_SCOPED_TIMER_SECTION> m_timer;
};
}  // namespace uipc::backend::cuda
