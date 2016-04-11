/* Copyright (c) 2015 Northeastern University
 * All rights reserved.
 *
 * Developed by:Northeastern University Computer Architecture Research (NUCAR)
 * Group, Northeastern University, http://www.ece.neu.edu/groups/nucar/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 *  with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/
 * or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *   Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimers. Redistributions in binary
 *   form must reproduce the above copyright notice, this list of conditions and
 *   the following disclaimers in the documentation and/or other materials
 *   provided with the distribution. Neither the names of NUCAR, Northeastern
 *   University, nor the names of its contributors may be used to endorse or
 *   promote products derived from this Software without specific prior written
 *   permission.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *   DEALINGS WITH THE SOFTWARE.
 */

#include <string.h>
#include <stdio.h>
#include <cstdlib>
#include "src/sw/cl20/sw_cl20_benchmark.h"

void SwCl20Benchmark::Initialize() {
  SwBenchmark::Initialize();

  ClBenchmark::InitializeCl();

  InitializeKernels();
  InitializeBuffers();
  InitializeData();
}

void SwCl20Benchmark::InitializeKernels() {
  cl_int err;
  file_->open("kernels.cl");

  const char *source = file_->getSourceChar();
  program_ = clCreateProgramWithSource(context_, 1, (const char **)&source,
                                       NULL, &err);
  checkOpenCLErrors(err, "Failed to create program with source...\n");

  err =
      clBuildProgram(program_, 1, &device_, "-I ./ -cl-std=CL2.0", NULL, NULL);
  checkOpenCLErrors(err, "Failed to create program...\n");

  CREATE_KERNEL
  sw_kernel_ = clCreateKernel(program_, "XXX", &err);
  checkOpenCLErrors(err, "Failed to create kernel XXX\n");
}

void SwCl20Benchmark::InitializeBuffers() {}

void SwCl20Benchmark::InitializeData() {}

void SwCl20Benchmark::Run() {}

void SwCl20Benchmark::Cleanup() {
  SwBenchmark::Cleanup();

  cl_int ret;
  ret = clReleaseKernel(sw_kernel_);
  ret = clReleaseProgram(program_);

  OTHER_CLEANUPS

  checkOpenCLErrors(ret, "Release objects.\n");
}
