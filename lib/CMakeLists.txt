# cmake-format: off

# this allows simplified running of clang-tidy
if(${CMAKE_BUILD_TYPE} STREQUAL "DEVEL")
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# build up git version add -debug to GITVERSION if we build with debug options enabled
string(REGEX MATCH [Dd][Ee][Bb][Uu][Gg] DEBUG_BUILD ${CMAKE_BUILD_TYPE})
if(DEBUG_BUILD)
  if(GITVERSION)
    set(GITVERSION ${GITVERSION}-debug)
  else()
    set(GITVERSION debug)
  endif()
endif()

set (QUDA_OBJS
  # cmake-format: sortable
  monitor.cpp 
  timer.cpp
  quda_ptr.cpp
  util_quda.cpp
  comm_common.cpp
  communicator_stack.cpp
  tune.cpp
  interface_quda.cpp
  transform_reduce.cu
  scale_vector.cu
  reduce_vector.cu
  #instantiate.cpp
  version.cpp)
# cmake-format: on

# split source into cu and cpp files
foreach(item ${QUDA_OBJS})
  string(REGEX MATCH ".+\\.cu$" item_match ${item})
  if(item_match)
    list(APPEND QUDA_CU_OBJS ${item})
  endif(item_match)
endforeach(item ${QUDA_OBJS})

list(REMOVE_ITEM QUDA_OBJS ${QUDA_CU_OBJS})

if(BUILD_FORTRAN_INTERFACE)
  list(APPEND QUDA_OBJS quda_fortran.F90)

  # set_source_files_properties(quda_fortran.F90 PROPERTIES OBJECT_OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/quda_fortran.mod)
endif()

# QUDA_CU_OBJS should contain all cuda files now QUDA_OBJS all c, cpp, fortran sources
# if we have a git version make version.cpp depend on git head so that it is rebuild if the git sha changed
if(${CMAKE_BUILD_TYPE} STREQUAL "DEVEL")
  if(GITVERSION)
    find_path(
      QUDA_GITDIR NAME HEAD
      PATHS ${CMAKE_SOURCE_DIR}/.git/logs
      NO_DEFAULT_PATH)
    include(AddFileDependencies)
    if(QUDA_GITDIR)
      add_file_dependencies(version.cpp ${QUDA_GITDIR}/HEAD)
    endif()
  endif()
  mark_as_advanced(QUDA_GITDIR)
endif()

# generate a cmake object library for all cpp files first
add_library(quda_cpp OBJECT ${QUDA_OBJS})

# add comms and QIO
target_sources(
  quda_cpp
  PRIVATE
    $<IF:$<BOOL:${QUDA_MPI}>,communicator_mpi.cpp,$<IF:$<BOOL:${QUDA_QMP}>,communicator_qmp.cpp,communicator_single.cpp>>
)

target_sources(quda_cpp PRIVATE $<$<BOOL:${QUDA_QIO}>:qio_field.cpp layout_hyper.cpp>)

if(QUDA_BUILD_SHAREDLIB)
  set_target_properties(quda_cpp PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  add_library(quda SHARED)
else()
  add_library(quda STATIC)
endif()
add_library(QUDA::quda ALIAS quda)

# add target specific files
if(${QUDA_TARGET_TYPE} STREQUAL "CUDA")
  include(targets/cuda/target_cuda.cmake)
endif()
if(${QUDA_TARGET_TYPE} STREQUAL "HIP")
  include(targets/hip/target_hip.cmake)
endif()
if(${QUDA_TARGET_TYPE} STREQUAL "SYCL")
  include(targets/sycl/target_sycl.cmake)
endif()

# Set the maximum multi-RHS per kernel if not already set by the target
if(NOT DEFINED QUDA_MAX_MULTI_RHS)
  set(QUDA_MAX_MULTI_RHS "16" CACHE STRING "maximum number of simultaneous RHS in a kernel")
  message(STATUS "Max number of rhs per kernel: ${QUDA_MAX_MULTI_RHS}")
endif()

# make one library
target_sources(quda PRIVATE $<TARGET_OBJECTS:quda_cpp> $<$<TARGET_EXISTS:quda_pack>:$<TARGET_OBJECTS:quda_pack>>
                            ${QUDA_CU_OBJS})

# for a develop build reduce the size by compressing the debug information
include(CheckLinkerFlag)
check_linker_flag(CXX "-Wl,--compress-debug-sections=zlib" QUDA_LINKER_COMPRESS)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-gz QUDA_COMPRESS_DEBUG)

if(NOT QUDA_BACKWARDS)
  if(${QUDA_LINKER_COMPRESS})
    target_link_options(quda PRIVATE $<$<CONFIG:DEVEL>:$<${QUDA_LINKER_COMPRESS}:-Wl,--compress-debug-sections=zlib>>)
  endif()
  if(${QUDA_COMPRESS_DEBUG})
    target_compile_options(quda_cpp PRIVATE $<$<CONFIG:DEVEL>:$<${QUDA_COMPRESS_DEBUG}:-gz>>)
    target_compile_options(quda PRIVATE $<$<CONFIG:DEVEL>:$<${QUDA_COMPRESS_DEBUG}:-gz>>)
  endif()
endif()

# set up QUDA compile options, put them before target specific options
target_compile_options(
  quda BEFORE PRIVATE $<$<COMPILE_LANGUAGE:CXX,C,Fortran>:
               $<IF:$<CONFIG:RELEASE>,-w,-Wall -Wextra>
               $<$<CONFIG:STRICT>:-Werror>
  >)

target_compile_definitions(
  quda PRIVATE $<$<CONFIG:DEVEL>:DEVEL> $<$<CONFIG:HOSTDEBUG>:HOST_DEBUG> $<$<CONFIG:DEBUG>:HOST_DEBUG>
               $<$<CONFIG:DEBUG>:DEVICE_DEBUG> $<$<CONFIG:SANITIZE>:HOST_DEBUG>)
target_include_directories(quda PRIVATE .)
target_include_directories(quda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(quda SYSTEM PRIVATE ../include/externals)

target_link_libraries(quda PRIVATE $<BUILD_INTERFACE:Eigen>)
target_include_directories(quda PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/>
                                       $<INSTALL_INTERFACE:include/>)

target_include_directories(quda PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include> $<INSTALL_INTERFACE:include>)

target_include_directories(quda_cpp SYSTEM PRIVATE ../include/externals)

target_link_libraries(quda_cpp PRIVATE $<BUILD_INTERFACE:Eigen>)
target_compile_definitions(quda_cpp PRIVATE $<TARGET_PROPERTY:quda,COMPILE_DEFINITIONS>)
target_include_directories(quda_cpp PRIVATE $<TARGET_PROPERTY:quda,INCLUDE_DIRECTORIES>)
target_compile_options(quda_cpp PRIVATE $<TARGET_PROPERTY:quda,COMPILE_OPTIONS>)

add_subdirectory(targets/generic)
target_include_directories(quda PRIVATE ../include/targets/generic)
add_subdirectory(interface)

# ignore any unknown pragmas
target_compile_options(quda_cpp PUBLIC $<$<COMPILE_LANGUAGE:CXX>:
  $<$<CXX_COMPILER_ID:Clang>:-Wno-unknown-pragmas>
  $<$<CXX_COMPILER_ID:GNU>:-Wno-unknown-pragmas>
  >)

# add some definitions that cause issues with cmake 3.7 and nvcc only to cpp files
target_compile_definitions(quda_cpp PUBLIC -DQUDA_HASH="${HASH}")
if(GITVERSION)
  target_compile_definitions(quda_cpp PUBLIC -DGITVERSION="${GITVERSION}")
endif()

# some clang warnings should be warning even when turning warnings into errors
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(quda_cpp PUBLIC $<$<COMPILE_LANGUAGE:CXX>: -Wno-error=unused-function>)
  # this is a hack to get colored diagnostics back when using Ninja and clang
  if(CMAKE_GENERATOR MATCHES "Ninja")
    target_compile_options(quda PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fcolor-diagnostics>)
  endif()
endif()

target_link_libraries(quda PUBLIC Threads::Threads)

if(${QUDA_BUILD_NATIVE_LAPACK} STREQUAL "ON")
  target_compile_definitions(quda PRIVATE NATIVE_LAPACK_LIB)
endif()

if(${QUDA_BUILD_NATIVE_FFT} STREQUAL "ON")
  target_compile_definitions(quda PRIVATE NATIVE_FFT_LIB)
endif()

# MULTI GPU AND USQCD
if(QUDA_MPI OR QUDA_QMP)
  target_compile_definitions(quda PUBLIC MULTI_GPU)
endif()

if(QUDA_MPI)
  target_compile_definitions(quda PUBLIC MPI_COMMS)
  target_link_libraries(quda PUBLIC MPI::MPI_CXX)
endif()

if(QUDA_QMP)
  target_compile_definitions(quda PUBLIC QMP_COMMS)
  target_link_libraries(quda PUBLIC QMP::qmp)
  target_link_libraries(quda PUBLIC MPI::MPI_CXX)
endif()

if(QUDA_QIO)
  target_compile_definitions(quda PUBLIC HAVE_QIO)
  target_link_libraries(quda PUBLIC QIO::qio)
endif()


if(QUDA_OPENBLAS)
  target_compile_definitions(quda PRIVATE OPENBLAS_LIB)

  if(QUDA_DOWNLOAD_OPENBLAS)
    target_link_libraries(quda PUBLIC openblas)
    target_link_libraries(quda_cpp PUBLIC openblas)
  else()
    target_link_libraries(quda INTERFACE ${OPENBLAS})
  endif()
endif()

if(QUDA_USE_EIGEN)
  target_compile_definitions(quda PRIVATE EIGEN)
endif()

if(QUDA_OPENMP)
  target_link_libraries(quda PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(quda PUBLIC QUDA_OPENMP)
endif()

# set which precisions to enable
target_compile_definitions(quda PUBLIC QUDA_PRECISION=${QUDA_PRECISION})
target_compile_definitions(quda PUBLIC QUDA_RECONSTRUCT=${QUDA_RECONSTRUCT})

if(QUDA_ALTERNATIVE_I_TO_F)
  target_compile_definitions(quda PRIVATE QUDA_ALTERNATIVE_I_TO_F)
endif()

if(QUDA_BACKWARDS)
  target_include_directories(quda_cpp SYSTEM PRIVATE ${backward-cpp_SOURCE_DIR})

  set_property(
    SOURCE comm_common.cpp
    APPEND
    PROPERTY COMPILE_DEFINITIONS ${BACKWARD_DEFINITIONS})
  set_property(
    SOURCE comm_common.cpp
    APPEND
    PROPERTY COMPILE_DEFINITIONS QUDA_BACKWARDSCPP)
  target_link_libraries(quda PUBLIC ${BACKWARD_LIBRARIES})
endif()

configure_file(../include/quda_define.h.in ../include/quda_define.h @ONLY)
install(FILES "${CMAKE_BINARY_DIR}/include/quda_define.h" DESTINATION include/)

# until we define an install step copy the include directory to the build directory
add_custom_command(
  TARGET quda
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)

# some hackery to prevent having old shared / static builds of quda messing with the current build
add_custom_command(
  TARGET quda
  PRE_LINK
  COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/libquda.a ${CMAKE_CURRENT_BINARY_DIR}/libquda.so)

install(
  TARGETS quda
  EXPORT QUDATargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include)

install(
  EXPORT QUDATargets
  FILE QUDATargets.cmake
  NAMESPACE QUDA::
  DESTINATION lib/cmake/QUDA)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/qudaConfigVersion.cmake"
  VERSION ${QUDA_VERSION}
  COMPATIBILITY AnyNewerVersion)

include(CMakePackageConfigHelpers)

configure_package_config_file(${CMAKE_SOURCE_DIR}/QUDAConfig.cmake.in QUDAConfig.cmake INSTALL_DESTINATION lib/cmake/QUDA)

write_basic_package_version_file(
  QUDAConfigVersion.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY AnyNewerVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/QUDAConfigVersion.cmake ${CMAKE_CURRENT_BINARY_DIR}/QUDAConfig.cmake
	DESTINATION lib/cmake/QUDA)
