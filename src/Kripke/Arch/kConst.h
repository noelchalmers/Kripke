//
// Copyright (c) 2014-19, Lawrence Livermore National Security, LLC
// and Kripke project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

#ifndef KRIPKE_ARCH_KCONST
#define KRIPKE_ARCH_KCONST

#include <Kripke.h>
#include <Kripke/VarTypes.h>

namespace Kripke {
namespace Arch {


template<typename A>
struct Policy_kConst;

template<>
struct Policy_kConst<ArchT_Sequential> {
  using ExecPolicy = RAJA::seq_exec;
};

#ifdef KRIPKE_USE_OPENMP
template<>
struct Policy_kConst<ArchT_OpenMP> {
  using ExecPolicy = RAJA::omp_parallel_for_exec;
};
#endif // KRIPKE_USE_OPENMP

#ifdef KRIPKE_USE_CUDA
template<>
struct Policy_kConst<ArchT_CUDA> {
  using ExecPolicy = RAJA::cuda_exec<256>;
};
#endif // KRIPKE_USE_CUDA

#ifdef KRIPKE_USE_HIP
template<>
struct Policy_kConst<ArchT_HIP> {
  using ExecPolicy = RAJA::hip_exec<256>;
};
#endif // KRIPKE_USE_HIP

}
}
#endif
