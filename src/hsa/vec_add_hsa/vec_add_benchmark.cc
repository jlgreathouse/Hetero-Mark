/*
 * Hetero-mark
 *
 * Copyright (c) 2015 Northeastern University
 * All rights reserved.
 *
 * Developed by:
 *   Northeastern University Computer Architecture Research (NUCAR) Group
 *   Northeastern University
 *   http://www.ece.neu.edu/groups/nucar/
 *
 * Author: Yifan Sun (yifansun@coe.neu.edu)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a 
 * copy of this software and associated documentation files (the "Software"), 
 * to deal with the Software without restriction, including without limitation 
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, 
 * and/or sell copies of the Software, and to permit persons to whom the 
 * Software is furnished to do so, subject to the following conditions:
 * 
 *   Redistributions of source code must retain the above copyright notice, 
 *   this list of conditions and the following disclaimers.
 *
 *   Redistributions in binary form must reproduce the above copyright 
 *   notice, this list of conditions and the following disclaimers in the 
 *   documentation and/or other materials provided with the distribution.
 *
 *   Neither the names of NUCAR, Northeastern University, nor the names of 
 *   its contributors may be used to endorse or promote products derived 
 *   from this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS WITH THE SOFTWARE.
 */

#include "src/hsa/vec_add_hsa/vec_add_benchmark.h"
#include <cstdlib>
#include <cstdio>

VecAddBenchmark::VecAddBenchmark(HsaRuntimeHelper *runtime_helper) :
    runtime_helper_(runtime_helper) {
}

void VecAddBenchmark::Initialize() {
  in1_ = new float[size_];
  in2_ = new float[size_];
  out_ = new float[size_];
  for (uint64_t i = 0; i < size_; i++) {
    in1_[i] = rand();
    in2_[i] = rand();
  }

  runtime_helper_->InitializeOrDie();
  agent_ = runtime_helper_->FindGpuOrDie();
  executable_ = runtime_helper_->CreateProgramFromSourceOrDie(
      "kernels.brig", agent_);
  kernel_ = executable_->GetKernel("&vec_add_kernel", 
      agent_);
  queue_ = agent_->CreateQueueOrDie();
  
  kernel_->SetDimension(1);
  kernel_->SetLocalSize(1, 64);
  kernel_->SetGlobalSize(1, size_);
  kernel_->SetKernelArgument(1, sizeof(in1_), &in1_);
  kernel_->SetKernelArgument(2, sizeof(in2_), &in2_); 
  kernel_->SetKernelArgument(3, sizeof(out_), &out_); 

}

void VecAddBenchmark::Run() {
  kernel_->ExecuteKernel(agent_, queue_);
}

void VecAddBenchmark::Verify() {
  for (uint64_t i = 0; i < size_; i++) {
    float result = in1_[i] + in2_[i];
    if (result != out_[i]) {
      std::cerr << "Failed at " << i << ", expect " << result
        << " but get " << out_[i] << "\n";
      exit(1);
    }
  }
}

void VecAddBenchmark::Summarize() {
}

void VecAddBenchmark::Cleanup() {
  delete[] in1_;
  delete[] in2_;
  delete[] out_;
}
