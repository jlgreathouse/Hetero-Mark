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

#include "src/common/runtime_helper/hsa_runtime_helper/hsa_signal.h"

#include <cstring>
#include <iostream>

HsaSignal::HsaSignal(hsa_signal_t signal) : signal_(signal) {
}

int64_t HsaSignal::WaitForCondition(const char *condition, int64_t value) {
  hsa_signal_condition_t signal_condition = HSA_SIGNAL_CONDITION_EQ;
  
  // Translate signal condition
  if (strcmp(condition, "EQ") == 0) {
    signal_condition = HSA_SIGNAL_CONDITION_EQ;
  } else if (strcmp(condition, "NE") == 0) {
    signal_condition = HSA_SIGNAL_CONDITION_NE;
  } else if (strcmp(condition, "LT") == 0) {
    signal_condition = HSA_SIGNAL_CONDITION_LT;
  } else if (strcmp(condition, "GTE") == 0) {
    signal_condition = HSA_SIGNAL_CONDITION_GTE;
  } else {
    std::cerr << "Unsupported HSA signal condition " << condition << "\n";
    exit(1);
  }

  // Wait for signal value
  int64_t signal_value = hsa_signal_wait_relaxed(signal_, signal_condition, 
      value, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
  return signal_value;
}

void HsaSignal::SetValue(int64_t value) {
  hsa_signal_store_relaxed(signal_, value);
}

