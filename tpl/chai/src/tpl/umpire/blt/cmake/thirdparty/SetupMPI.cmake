# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level COPYRIGHT file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

################################
# MPI
################################

# CMake changed some of the output variables that we use from Find(MPI)
# in 3.10+.  This toggles the variables based on the CMake version
# the user is running.
if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.10.0" )
    if (NOT MPIEXEC_EXECUTABLE AND MPIEXEC)
        set(MPIEXEC_EXECUTABLE ${MPIEXEC} CACHE PATH "" FORCE)
    endif()

    set(_mpi_includes_suffix "INCLUDE_DIRS")
    set(_mpi_compile_flags_suffix "COMPILE_OPTIONS")
else()
    if (MPIEXEC_EXECUTABLE AND NOT MPIEXEC)
        set(MPIEXEC ${MPIEXEC_EXECUTABLE} CACHE PATH "" FORCE)
    endif()

    set(_mpi_includes_suffix "INCLUDE_PATH")
    set(_mpi_compile_flags_suffix "COMPILE_FLAGS")
endif()

set(_mpi_compile_flags )
set(_mpi_includes )
set(_mpi_libraries )
set(_mpi_link_flags )

message(STATUS "Enable FindMPI:  ${ENABLE_FIND_MPI}")

if (ENABLE_FIND_MPI)
    find_package(MPI REQUIRED)

    #-------------------
    # Merge found MPI info and remove duplication
    #-------------------
    # Compile flags
    set(_c_flg ${MPI_C_${_mpi_compile_flags_suffix}})
    if (_c_flag AND ENABLE_CUDA)
        list(APPEND _mpi_compile_flags   
                    "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:${c_flg}>"
                    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_c_flg}>")
    else()
        list(APPEND _mpi_compile_flags ${_c_flg})
    endif()
    
    set(_cxx_flg ${MPI_CXX_${_mpi_compile_flags_suffix}})
    if (_cxx_flg AND NOT "${_c_flg}" STREQUAL "${_cxx_flg}")
        if (ENABLE_CUDA)
            list(APPEND _mpi_compile_flags
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:${_cxx_flg}>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_cxx_flg}>")
        else()
            list(APPEND _mpi_compile_flags ${_cxx_flg})
        endif()
    endif()
    
    if (ENABLE_FORTRAN)
        set(_f_flg ${MPI_Fortran_${_mpi_compile_flags_suffix}})
        if (_f_flg AND NOT "${c_flg}" STREQUAL "${_f_flg}")
            list(APPEND _mpi_compile_flags ${_f_flg})
        endif()
    endif()
    unset(_c_flg)
    unset(_cxx_flg)
    unset(_f_flg)

    # Include paths
    list(APPEND _mpi_includes ${MPI_C_${_mpi_includes_suffix}}
                              ${MPI_CXX_${_mpi_includes_suffix}})
    if (ENABLE_FORTRAN)
        list(APPEND _mpi_includes ${MPI_Fortran_${_mpi_includes_suffix}})
    endif()
    blt_list_remove_duplicates(TO _mpi_includes)

    # Link flags
    set(_mpi_link_flags ${MPI_C_LINK_FLAGS})
    if (NOT "${MPI_C_LINK_FLAGS}" STREQUAL "${MPI_CXX_LINK_FLAGS}")
        list(APPEND _mpi_link_flags ${MPI_CXX_LINK_FLAGS})
    endif()
    if (ENABLE_FORTRAN)
        if (NOT "${MPI_C_LINK_FLAGS}" STREQUAL "${MPI_Fortran_LINK_FLAGS}")
            list(APPEND _mpi_link_flags ${MPI_CXX_LINK_FLAGS})
        endif()
    endif()

    # Libraries
    set(_mpi_libraries ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
    if (ENABLE_FORTRAN)
        list(APPEND _mpi_libraries ${MPI_Fortran_LIBRARIES})
    endif()
    blt_list_remove_duplicates(TO _mpi_libraries)
endif()

# Allow users to override CMake's FindMPI
if (BLT_MPI_COMPILE_FLAGS)
    set(_mpi_compile_flags ${BLT_MPI_COMPILE_FLAGS})
endif()
if (BLT_MPI_INCLUDES)
    set(_mpi_includes ${BLT_MPI_INCLUDES})
endif()
if (BLT_MPI_LIBRARIES)
    set(_mpi_libraries ${BLT_MPI_LIBRARIES})
endif()
if (BLT_MPI_LINK_FLAGS)
    set(_mpi_link_flags ${BLT_MPI_LINK_FLAGS})
endif()


# Output all MPI information
message(STATUS "BLT MPI Compile Flags:  ${_mpi_compile_flags}")
message(STATUS "BLT MPI Include Paths:  ${_mpi_includes}")
message(STATUS "BLT MPI Libraries:      ${_mpi_libraries}")
message(STATUS "BLT MPI Link Flags:     ${_mpi_link_flags}")

if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.10.0" )
    message(STATUS "MPI Executable:       ${MPIEXEC_EXECUTABLE}")
else()
    message(STATUS "MPI Executable:       ${MPIEXEC}")
endif()
message(STATUS "MPI Num Proc Flag:    ${MPIEXEC_NUMPROC_FLAG}")
message(STATUS "MPI Command Append:   ${BLT_MPI_COMMAND_APPEND}")

if (ENABLE_FORTRAN)
    # Determine if we should use fortran mpif.h header or fortran mpi module
    find_path(mpif_path
        NAMES "mpif.h"
        PATHS ${_mpi_includes}
        NO_DEFAULT_PATH
        )
    
    if(mpif_path)
        set(MPI_Fortran_USE_MPIF ON CACHE PATH "")
        message(STATUS "Using MPI Fortran header: mpif.h")
    else()
        set(MPI_Fortran_USE_MPIF OFF CACHE PATH "")
        message(STATUS "Using MPI Fortran module: mpi.mod")
    endif()
endif()

# Create the registered library
blt_register_library(NAME          mpi
                     INCLUDES      ${_mpi_includes}
                     TREAT_INCLUDES_AS_SYSTEM ON
                     LIBRARIES     ${_mpi_libraries}
                     COMPILE_FLAGS ${_mpi_compile_flags}
                     LINK_FLAGS    ${_mpi_link_flags} )