/*
* Copyright (c) 2015 Northeastern University
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
__kernel void PageRankUpdateGpu(uint num_rows, __global uint* rowOffset,
                                __global uint* col, __global float* val,
                                __local float* vals, __global float* x,
                                __global float* y) {
  int thread_id = get_global_id(0);
  int local_id = get_local_id(0);
  int warp_id = thread_id >> 6;
  int lane = thread_id & (64 - 1);
  int row = warp_id;

  if (row < num_rows) {
    y[row] = 0.0;
    int row_A_start = rowOffset[row];
    int row_A_end = rowOffset[row + 1];

    vals[local_id] = 0;
    for (int jj = row_A_start + lane; jj < row_A_end; jj += 64)
      vals[local_id] += val[jj] * x[col[jj]];

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (lane < 32) vals[local_id] += vals[local_id + 32];
    if (lane < 16) vals[local_id] += vals[local_id + 16];
    if (lane < 8) vals[local_id] += vals[local_id + 8];
    if (lane < 4) vals[local_id] += vals[local_id + 4];
    if (lane < 2) vals[local_id] += vals[local_id + 2];
    if (lane < 1) vals[local_id] += vals[local_id + 1];
    if (lane == 0) y[row] += vals[local_id];
  }
}
