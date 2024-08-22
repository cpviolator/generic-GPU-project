# enable tests build a common library for all test utilities
add_library(quda_test googletest/src/gtest-all.cc)
target_include_directories(quda_test SYSTEM PUBLIC  googletest/include googletest)
target_link_libraries(quda_test PUBLIC Eigen)
target_link_libraries(quda_test PUBLIC QUDA::quda)
set_target_properties(quda_test PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(
  quda_test PUBLIC
  $<IF:$<CONFIG:RELEASE>,-w,-Wall -Wextra>
  $<$<CONFIG:STRICT>:-Werror>
)

# ignore any unkown pragmas if not using OpenMP
if(NOT ${QUDA_OPENMP})
  target_compile_options(quda_test PUBLIC $<$<COMPILE_LANGUAGE:CXX>:
    $<$<CXX_COMPILER_ID:Clang>:-Wno-unknown-pragmas>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-unknown-pragmas>
    >)
endif()

if(BUILD_SHARED_LIBS)
  install(TARGETS quda_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

add_subdirectory(utils)
add_subdirectory(host_reference)
# enable_language(Fortran)

if(QUDA_NVSHMEM AND QUDA_DOWNLOAD_NVSHMEM)
  add_dependencies(quda_test NVSHMEM)
endif()

set(TEST_LIBS quda_test)

# custom target for building all tests (useful if QUDA_BUILD_ALL_TESTS is OFF)
add_custom_target(all_tests)

macro(QUDA_CHECKBUILDTEST mytarget qudabuildtests)
  if(NOT ${qudabuildtests})
    set_property(TARGET ${mytarget} PROPERTY EXCLUDE_FROM_ALL 1)
    set(QUDA_EXCLUDE_FROM_INSTALL "EXCLUDE_FROM_ALL")
  endif()
  add_dependencies(all_tests ${mytarget})
endmacro()

if(NOT ${QUDA_INSTALL_ALL_TESTS})
  set(QUDA_EXCLUDE_FROM_INSTALL "EXCLUDE_FROM_ALL")
endif()

if(QUDA_ARPACK)
  list(APPEND TEST_LIBS ${ARPACK})
  if(QUDA_MPI OR QUDA_QMP)
    list(APPEND TEST_LIBS ${PARPACK})
  endif()
endif()

# define tests
add_executable(c_interface_test c_interface_test.c)
target_link_libraries(c_interface_test ${TEST_LIBS} "-lstdc++")
quda_checkbuildtest(c_interface_test QUDA_BUILD_ALL_TESTS)

add_executable(scale_vector_example scale_vector_example.cpp)
target_link_libraries(scale_vector_example ${TEST_LIBS} "-lstdc++")
quda_checkbuildtest(scale_vector_example QUDA_BUILD_ALL_TESTS)

#add_executable(vector_reduce_example vector_reduce_example.cpp)
#target_link_libraries(vector_reduce_example ${TEST_LIBS} "-lstdc++")
#quda_checkbuildtest(vector_reduce_example QUDA_BUILD_ALL_TESTS)

if(QUDA_BUILD_NATIVE_LAPACK)
  add_executable(blas_interface_test blas_interface_test.cpp)
  target_link_libraries(blas_interface_test ${TEST_LIBS})
  quda_checkbuildtest(blas_interface_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS blas_interface_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(QUDA_QIO)
  add_executable(io_test io_test.cpp)
  target_link_libraries(io_test ${TEST_LIBS})
  quda_checkbuildtest(io_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS io_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

add_executable(tune_test tune_test.cpp)
target_link_libraries(tune_test ${TEST_LIBS})
quda_checkbuildtest(tune_test QUDA_BUILD_ALL_TESTS)
install(TARGETS tune_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

if(QUDA_MPI OR QUDA_QMP)
  #helper function to extract required number of ranks from environment variable
  function(get_test_ranks TEST_GRID RANKS)
    separate_arguments(TEST_GRID)
    LIST(LENGTH TEST_GRID TEST_LENGTH)
    if(NOT ${TEST_LENGTH} EQUAL 4)
      message(SEND_ERROR "QUDA_TEST_GRID_SIZE takes 4 INTEGERS. GOT: ${TEST_GRID_SIZE}")
    endif()
      LIST(JOIN TEST_GRID * TEST_GRID_EXPR)
    math(EXPR TEST_RANKS ${TEST_GRID_EXPR} OUTPUT_FORMAT DECIMAL)
    set(${RANKS} ${TEST_RANKS} PARENT_SCOPE )
  endfunction()

  set(QUDA_TEST_NUM_PROCS 1)
  if(DEFINED ENV{QUDA_TEST_GRID_SIZE})
    get_test_ranks($ENV{QUDA_TEST_GRID_SIZE} QUDA_TEST_NUM_PROCS)
  endif()
  message(STATUS "ctest will run on ${QUDA_TEST_NUM_PROCS} processes")
  set(QUDA_CTEST_LAUNCH ${MPIEXEC_EXECUTABLE};${MPIEXEC_NUMPROC_FLAG};${QUDA_TEST_NUM_PROCS};${MPIEXEC_PREFLAGS}
    CACHE STRING "CTest Launcher command for QUDA's tests")
endif()


#BLAS interface test
if(QUDA_BUILD_NATIVE_LAPACK)
  add_test(NAME blas_interface_test
    COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:blas_interface_test> ${MPIEXEC_POSTFLAGS}
    --blas-gemm-mnk 64 64 64
    --blas-gemm-leading-dims 128 128 128
    --blas-gemm-offsets 16 16 16
    --blas-gemm-alpha 1.0 2.0
    --blas-gemm-beta -3.0 1.5
    --blas-gemm-trans-a T
    --blas-gemm-trans-b C
    --blas-lu-inv-mat-size 96
    --blas-data-type Z
    --blas-data-order row
    --blas-batch 20
    --enable-testing true
    --gtest_output=xml:blas_interface_test.xml)
endif()

# enable the precisions that are compiled
math(EXPR double_prec  "${QUDA_PRECISION} & 8")
math(EXPR single_prec  "${QUDA_PRECISION} & 4")
math(EXPR half_prec    "${QUDA_PRECISION} & 2")
math(EXPR quarter_prec "${QUDA_PRECISION} & 1")

if(double_prec AND single_prec)
  set(TEST_PRECS single double)
elseif(double_prec)
  set(TEST_PRECS double)
elseif(single_prec)
  set(TEST_PRECS single)
endif()


if(QUDA_QIO)
  add_test(NAME io_test
           COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:io_test> ${MPIEXEC_POSTFLAGS}
                   --dim 4 6 8 10
                   --gtest_output=xml:io_test.xml)
endif()

add_test(NAME tune_test
         COMMAND  ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:tune_test> ${MPIEXEC_POSTFLAGS}
                   --gtest_output=xml:tune_test.xml)
