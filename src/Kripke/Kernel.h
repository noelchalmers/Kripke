//
// Copyright (c) 2014-19, Lawrence Livermore National Security, LLC
// and Kripke project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

#ifndef KRIPKE_KERNEL_H__
#define KRIPKE_KERNEL_H__

#include <Kripke.h>
#include <Kripke/Core/DataStore.h>
#include <Kripke/Arch/kConst.h>
#include <utility>

namespace Kripke {

  namespace Kernel {

    void LPlusTimes(Kripke::Core::DataStore &data_store);


    void LTimes(Kripke::Core::DataStore &data_store);


    double population(Kripke::Core::DataStore &data_store);


    void scattering(Kripke::Core::DataStore &data_store);


    void source(Kripke::Core::DataStore &data_store);


    void sweepSubdomain(Kripke::Core::DataStore &data_store, Kripke::SdomId sdom_id);

    template<typename Arch, typename FieldType>
    RAJA_INLINE
    void kConstKernel(Arch arch, FieldType &field, Kripke::SdomId sdom_id, typename FieldType::ElementType value){
      auto view1d = field.getView1d(sdom_id);
      int num_elem = field.size(sdom_id);

      using ExecPolicy = typename Kripke::Arch::Policy_kConst<Arch>::ExecPolicy;

      RAJA::forall<ExecPolicy>(
        RAJA::RangeSegment(0, num_elem),
        [=](RAJA::Index_type i){
          view1d(i) = value;
      });
    }


    template<typename FieldType>
    RAJA_INLINE
    void kConst(ArchV arch_v, FieldType &field, Kripke::SdomId sdom_id, typename FieldType::ElementType value){
      switch(arch_v){
        case ArchV_Sequential: kConstKernel(ArchT_Sequential{}, field, sdom_id, value); break;

      #ifdef KRIPKE_USE_OPENMP
        case ArchV_OpenMP: kConstKernel(ArchT_OpenMP{}, field, sdom_id, value); break;
      #endif

      #ifdef KRIPKE_USE_CUDA
        case ArchV_CUDA: kConstKernel(ArchT_CUDA{}, field, sdom_id, value); break;
      #endif

      #ifdef KRIPKE_USE_HIP
        case ArchV_HIP: kConstKernel(ArchT_HIP{}, field, sdom_id, value); break;
      #endif
        default: KRIPKE_ABORT("Unknown arch_v=%d\n", (int)arch_v); break;
      }
    }

    template<typename FieldType>
    RAJA_INLINE
    void kConst(ArchV arch_v, FieldType &field, typename FieldType::ElementType value){
      for(Kripke::SdomId sdom_id : field.getWorkList()){
        kConst(arch_v, field, sdom_id, value);
      }
    }


    template<typename Arch, typename FieldType>
    RAJA_INLINE
    void kCopyKernel(Arch arch, FieldType &field_dst, Kripke::SdomId sdom_id_dst,
               FieldType &field_src, Kripke::SdomId sdom_id_src){
      auto view_src = field_src.getView1d(sdom_id_src);
      auto view_dst = field_dst.getView1d(sdom_id_dst);
      int num_elem = field_src.size(sdom_id_src);

      using ExecPolicy = typename Kripke::Arch::Policy_kConst<Arch>::ExecPolicy;

      RAJA::forall<ExecPolicy>(
        RAJA::RangeSegment(0, num_elem),
        [=](RAJA::Index_type i){
          view_src(i) = view_dst(i);
      });
    }

    template<typename FieldType>
    RAJA_INLINE
    void kCopy(ArchV arch_v, FieldType &field_dst, Kripke::SdomId sdom_id_dst,
               FieldType &field_src, Kripke::SdomId sdom_id_src){
      switch(arch_v){
        case ArchV_Sequential: kCopyKernel(ArchT_Sequential{}, field_dst, sdom_id_dst, field_src, sdom_id_src); break;

      #ifdef KRIPKE_USE_OPENMP
        case ArchV_OpenMP: kCopyKernel(ArchT_OpenMP{}, field_dst, sdom_id_dst, field_src, sdom_id_src); break;
      #endif

      #ifdef KRIPKE_USE_CUDA
        case ArchV_CUDA: kCopyKernel(ArchT_CUDA{}, field_dst, sdom_id_dst, field_src, sdom_id_src); break;
      #endif

      #ifdef KRIPKE_USE_HIP
        case ArchV_HIP: kCopyKernel(ArchT_HIP{}, field_dst, sdom_id_dst, field_src, sdom_id_src); break;
      #endif
        default: KRIPKE_ABORT("Unknown arch_v=%d\n", (int)arch_v); break;
      }
    }

    template<typename FieldType>
    RAJA_INLINE
    void kCopy(ArchV arch_v, FieldType &field_dst, FieldType &field_src){

      for(Kripke::SdomId sdom_id : field_dst.getWorkList()){
        kCopy(arch_v, field_dst, sdom_id, field_src, sdom_id);
      }
    }
  }
}

#endif
