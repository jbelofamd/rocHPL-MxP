/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    HPL - 2.2 - February 24, 2016
 *    Antoine P. Petitet
 *    University of Tennessee, Knoxville
 *    Innovative Computing Laboratory
 *    (C) Copyright 2000-2008 All Rights Reserved
 *
 *    Modified by: Noel Chalmers
 *    (C) 2018-2022 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

#include "hplmxp_comm_mpi.hpp"
#include <cstdlib>
#include <vector>

/* Pipelined ring: up to ring_depth() chunks of ring_chunk_bytes() each are
 * in flight on both the receive and the send side, so a hop's wire time is
 * not serialized behind one chunk's rendezvous round trip.
 *   HPLMXP_RING_DEPTH     chunks in flight per direction (1..16, default 1,
 *                         the original one-recv/one-send ring; 4 is a good
 *                         starting point when pipelining)
 *   HPLMXP_RING_CHUNK_KB  chunk size in KiB (default 2048) */
static int ring_env(const char* name, int dflt, int lo, int hi) {
  const char* v = std::getenv(name);
  if(v == nullptr || *v == '\0') return dflt;
  int x = std::atoi(v);
  return std::max(lo, std::min(hi, x));
}

static int ring_depth() {
  static const int depth = ring_env("HPLMXP_RING_DEPTH", 1, 1, 16);
  return depth;
}

static size_t ring_chunk_bytes() {
  static const size_t bytes =
      size_t(ring_env("HPLMXP_RING_CHUNK_KB", 2048, 4, 1 << 20)) << 10;
  return bytes;
}

template <typename T>
int HPLMXP_bcast_1ring(T* SBUF, int SCOUNT, int ROOT, MPI_Comm COMM) {

  int rank, size;
  MPI_Comm_rank(COMM, &rank);
  MPI_Comm_size(COMM, &size);

  if(size <= 1 || SCOUNT <= 0) return (MPI_SUCCESS);

  /*One ring exchange to rule them all*/
  const int chunk_size =
      int(std::min<size_t>(ring_chunk_bytes() / sizeof(T), size_t(SCOUNT)));
  const int Nchunks = (SCOUNT + chunk_size - 1) / chunk_size;
  const int depth   = std::min(ring_depth(), Nchunks);

  const int tag  = rank;
  const int next = MModAdd1(rank, size);
  const int prev = MModSub1(rank, size);

  /*Shift to ROOT=0*/
  rank = MModSub(rank, ROOT, size);

  const bool do_recv = (rank != 0);
  const bool do_send = (rank != size - 1);

  /* request[0..depth) are receives, request[depth..2*depth) sends; chunk c
   * uses slot c % depth on either side. Sends are posted strictly in chunk
   * order so that they match the receiver's in-order posted receives. */
  std::vector<MPI_Request> request(2 * depth, MPI_REQUEST_NULL);
  std::vector<int>         slot_chunk(depth, -1);
  std::vector<int>         done_idx(2 * depth);
  std::vector<char>        have(Nchunks, do_recv ? 0 : 1);

  auto chunk_ptr = [&](int c) { return SBUF + size_t(c) * chunk_size; };
  auto chunk_len = [&](int c) {
    return std::min(chunk_size, SCOUNT - c * chunk_size);
  };

  int nrecv_posted = do_recv ? 0 : Nchunks;
  int nsend_posted = do_send ? 0 : Nchunks;

  while(true) {
    /*Recv from left into every free slot*/
    while(nrecv_posted < Nchunks &&
          request[nrecv_posted % depth] == MPI_REQUEST_NULL) {
      const int c           = nrecv_posted++;
      slot_chunk[c % depth] = c;
      MPI_Irecv(chunk_ptr(c),
                chunk_len(c),
                T2MPI<T>::type,
                prev,
                prev,
                COMM,
                &request[c % depth]);
    }

    /*Send to right, in order, every chunk that is present*/
    while(nsend_posted < Nchunks && have[nsend_posted] &&
          request[depth + nsend_posted % depth] == MPI_REQUEST_NULL) {
      const int c = nsend_posted++;
      MPI_Isend(chunk_ptr(c),
                chunk_len(c),
                T2MPI<T>::type,
                next,
                tag,
                COMM,
                &request[depth + c % depth]);
    }

    int ndone = 0;
    MPI_Waitsome(2 * depth,
                 request.data(),
                 &ndone,
                 done_idx.data(),
                 MPI_STATUSES_IGNORE);
    if(ndone == MPI_UNDEFINED) break; /*nothing left in flight*/

    for(int i = 0; i < ndone; ++i) {
      const int idx = done_idx[i];
      if(idx < depth) have[slot_chunk[idx]] = 1;
    }
  }

  return MPI_SUCCESS;
}

template int HPLMXP_bcast_1ring(double*  SBUF,
                                int      SCOUNT,
                                int      ROOT,
                                MPI_Comm COMM);

template int HPLMXP_bcast_1ring(float*   SBUF,
                                int      SCOUNT,
                                int      ROOT,
                                MPI_Comm COMM);

template int HPLMXP_bcast_1ring(__half*  SBUF,
                                int      SCOUNT,
                                int      ROOT,
                                MPI_Comm COMM);
