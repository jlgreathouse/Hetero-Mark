#ifndef SHALLOW_WATER_H
#define SHALLOW_WATER_H

#include "src/common/cl_util/cl_util.h"
#include "src/common/benchmark/benchmark.h"

using namespace clHelper;

class ShallowWater : public Benchmark {
  clRuntime *runtime_;
  clFile *file_;

  cl_platform_id platform_;
  cl_device_id device_;
  cl_context context_;
  cl_command_queue cmdQueue_;

  cl_program program_;
  cl_kernel kernel_sw_init_psi_p_;
  cl_kernel kernel_sw_init_velocities_;
  cl_kernel kernel_sw_compute0_;
  cl_kernel kernel_sw_update0_;
  cl_kernel kernel_sw_compute1_;
  cl_kernel kernel_sw_update1_;
  cl_kernel kernel_sw_time_smooth_;

  // Size
  unsigned m_;
  unsigned n_;
  unsigned m_len_;
  unsigned n_len_;
  unsigned itmax_;

  // Params
  double dt_, tdt_, dx_, dy_, a_, alpha_, el_, pi_;
  double tpi_, di_, dj_, pcf_;
  double tdts8_, tdtsdx_, tdtsdy_, fsdx_, fsdy_;

  // OpenCL 1.2 style buffers
  cl_mem u_curr_;
  cl_mem u_next_;

  cl_mem v_curr_;
  cl_mem v_next_;

  cl_mem p_curr_;
  cl_mem p_next_;

  cl_mem u_;
  cl_mem v_;
  cl_mem p_;

  cl_mem cu_;
  cl_mem cv_;

  cl_mem z_;
  cl_mem h_;
  cl_mem psi_;

  // Initialize
  void InitKernel();
  void InitBuffer();
  void InitPsiP();
  void InitVelocities();

  // Cleanup
  void FreeKernel();
  void FreeBuffer();

  // Run
  void Compute0();
  void PeriodicUpdate0();
  void Compute1();
  void PeriodicUpdate1();
  void TimeSmooth(int ncycle);

public:
  ShallowWater(unsigned m = 2048, unsigned n = 2048);
  ~ShallowWater();

  void Initialize() override;
  void Run() override;
  void Verify() override {}
  void Cleanup() override;
  void Summarize() override {}
};

#endif
