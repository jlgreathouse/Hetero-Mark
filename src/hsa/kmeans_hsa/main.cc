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

#include "src/common/Benchmark/BenchmarkRunner.h"
#include "src/common/Timer/TimeMeasurement.h"
#include "src/common/Timer/TimeMeasurementImpl.h"
#include "src/common/CommandLineOption/CommandLineOption.h"
#include "src/hsa/kmeans_hsa/kmeans_benchmark.h"

int main(int argc, const char **argv) {
  // Setup command line option
  CommandLineOption command_line_option(
    "====== Hetero-Mark KMEANS Benchmark "
    "(HSA mode) ======",
    "This benchmarks runs kmeans algorithm.");
  command_line_option.addArgument("Help", "bool", "false",
      "-h", "--help", "Dump help information");
  command_line_option.addArgument("Length", "integer", "1024",
      "-x", "--length",
      "Length of data");
  command_line_option.addArgument("Verify", "bool", "false",
      "-v", "--verify",
      "Verify the calculation result");

  command_line_option.parse(argc, argv);
  if (command_line_option.getArgumentValue("Help")->asBool()) {
    command_line_option.help();
    return 0;
  }

  //uint32_t length = command_line_option.getArgumentValue("Length")->asUInt32();
  bool verify = command_line_option.getArgumentValue("Verify")->asBool();

  // Create and run benchmarks
  std::unique_ptr<KmeansBenchmark> benchmark(new KmeansBenchmark());
  std::unique_ptr<TimeMeasurement> timer(new TimeMeasurementImpl());
  BenchmarkRunner runner(benchmark.get(), timer.get());
  runner.setVerificationMode(verify);
  runner.run();
  runner.summarize();
}
