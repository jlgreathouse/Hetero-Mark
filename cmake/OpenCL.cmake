# Auto-select bitness based on platform
if( NOT BITNESS )
  if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(BITNESS 64)
  else()
    set(BITNESS 32)
  endif()
endif()

# Unset OPENCL_LIBRARIES, so that corresponding arch specific libs are found when bitness is changed
unset(OPENCL_LIBRARIES CACHE)
unset(OPENCL_INCLUDE_DIRS CACHE)

if( BITNESS EQUAL 64 )
  set(BITNESS_SUFFIX x86_64)
elseif( BITNESS EQUAL 32 )
  set(BITNESS_SUFFIX x86)
else()
  message( FATAL_ERROR "Bitness specified is invalid" )
endif()

############################################################################

# Find OpenCL include and libs
find_path( OPENCL_INCLUDE_DIRS
  NAMES OpenCL/cl.h CL/cl.h
  HINTS $ENV{AMDAPPSDKROOT}/include
)

find_library( OPENCL_LIBRARIES
  NAMES OpenCL
  HINTS $ENV{AMDAPPSDKROOT}/lib
  PATH_SUFFIXES ${PLATFORM}${BITNESS} ${BITNESS_SUFFIX}
)

if( OPENCL_INCLUDE_DIRS AND OPENCL_LIBRARIES )
else( OPENCL_INCLUDE_DIRS AND OPENCL_LIBRARIES )
  message("OpenCL include file and libraries not found. OpenCL benchmarks will be skipped.")
endif()
