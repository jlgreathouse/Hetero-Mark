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

#ifndef SRC_SW_HSA_SW_HSA_BENCHMARK_H_
#define SRC_SW_HSA_SW_HSA_BENCHMARK_H_

#include "src/sw/sw_benchmark.h"
#include "src/common/time_measurement/time_measurement.h"

class SwHsaBenchmark : public SwBenchmark {
 private:
  // HSA style buffers
  double *u_curr_;
  double *u_next_;

  double *v_curr_;
  double *v_next_;

  double *p_curr_;
  double *p_next_;

  double *u_;
  double *v_;
  double *p_;

  double *cu_;
  double *cv_;

  double *z_;
  double *h_;
  double *psi_;

  // Initialize
  void InitializeParams();
  void InitializeData();
  void InitializeBuffers();
  void InitPsiP();
  void InitVelocities();

  // Free
  void FreeBuffers();

  // Run
  void Compute0();
  void PeriodicUpdate0();
  void Compute1();
  void PeriodicUpdate1();
  void TimeSmooth(int ncycle);

 public:
  void Initialize() override;
  void Run() override;
  void Cleanup() override;
};

#endif  // SRC_SW_HSA_SW_HSA_BENCHMARK_H_
