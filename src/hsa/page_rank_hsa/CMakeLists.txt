set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )

set( FOLDER_NAME page_rank_hsa )
set( SUBDIRECTORIES )
set( SRC_FILES 
  main.cc
  page_rank_benchmark.cc
)
set( HEADER_FILES 
  page_rank_benchmark.h
)
set( TEST_FILES )
set( EXTRA_FILES 
  kernels.cl
)

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

# Unit test
if (${TEST_FILES})
  add_executable(${FOLDER_NAME}_unittest ${TEST_FILES} ${SRC_FILES} ${HEADER_FILES})
  target_link_libraries(${FOLDER_NAME}_unittest gtest gtest_main)
  add_test(${FOLDER_NAME}_unittest ${FOLDER_NAME}_unittest)
endif (${TEST_FILES})

###############################################################

if (have_snack AND HSA_RUNTIME)
  foreach( extra_file ${EXTRA_FILES} )
    add_custom_command(
      OUTPUT ${extra_file}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/${extra_file}  ${CMAKE_CURRENT_BINARY_DIR}
    )
  endforeach( extra_file )
  add_custom_command(
    OUTPUT kernels.o kernels.h
    COMMAND snack.sh -c kernels.cl
    DEPENDS kernels.cl
    COMMENT "Using snack to generate kernel library"
    WORKING_DIRECTORY ""
    VERBATIM
  )
  add_executable(${FOLDER_NAME} kernels.h kernels.o ${HEADER_FILES} ${SRC_FILES})
  target_link_libraries(${FOLDER_NAME}
    time_measurement command_line_option benchmark ${HSA_RUNTIME})
endif (have_snack AND HSA_RUNTIME)

