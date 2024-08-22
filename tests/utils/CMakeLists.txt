# add utils files to quda_test
target_sources(
  quda_test PRIVATE
  command_line_params.cpp
  host_utils.cpp
  )

target_include_directories(quda_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(quda_test SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/include/externals)
target_include_directories(quda_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(quda_test PRIVATE ${CMAKE_BINARY_DIR}/include)
