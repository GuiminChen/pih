function(pih_add_qwen_cubin target_name architecture)
  set(output_dir "${CMAKE_CURRENT_BINARY_DIR}/cubin/sm_${architecture}")
  set(output_file "${output_dir}/qwen_bf16_primitives.cubin")
  set(manifest_file "${output_dir}/qwen_bf16_primitives.cubin.json")
  # Device compilation must use the selected compiler's own driver syntax.
  # Preserve toolchain flags (target/sysroot/includes) for cross builds.
  separate_arguments(device_flags NATIVE_COMMAND "${CMAKE_CUDA_FLAGS}")
  set(PIH_CUDA_DEVICE_INCLUDE_DIRS "" CACHE STRING "Additional CUDA device header directories")
  foreach(directory IN LISTS CUDAToolkit_INCLUDE_DIRS PIH_CUDA_DEVICE_INCLUDE_DIRS)
    list(APPEND device_flags "-I${directory}")
  endforeach()
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    list(APPEND device_flags --cubin "--gpu-architecture=sm_${architecture}"
        --std=c++20 --fmad=false --ftz=false --prec-div=true --prec-sqrt=true)
  elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    if(CMAKE_CUDA_COMPILER_TARGET)
      list(APPEND device_flags "--target=${CMAKE_CUDA_COMPILER_TARGET}")
    endif()
    if(CMAKE_SYSROOT)
      list(APPEND device_flags "--sysroot=${CMAKE_SYSROOT}")
    endif()
    list(APPEND device_flags --cuda-device-only -c
        "--cuda-gpu-arch=sm_${architecture}" "--cuda-path=${CUDAToolkit_LIBRARY_ROOT}"
        -std=c++20 -ffp-contract=off -fno-fast-math -Xcuda-ptxas --fmad=false)
  else()
    message(FATAL_ERROR "Qwen cubin requires NVIDIA or Clang CUDA compiler")
  endif()
  list(APPEND device_flags -O3 "-I${CMAKE_CURRENT_SOURCE_DIR}/include")
  list(JOIN device_flags "," producer_flags)
  add_custom_command(
      OUTPUT "${output_file}" "${manifest_file}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_dir}"
      COMMAND
          "${CMAKE_CUDA_COMPILER}"
          ${device_flags}
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda/qwen_bf16_primitives.cu"
          -o "${output_file}"
      COMMAND
          "${CMAKE_COMMAND}"
          "-DCUBIN_PATH=${output_file}"
          "-DMANIFEST_PATH=${manifest_file}"
          "-DTARGET_SM=sm_${architecture}"
          "-DPRODUCER_TOOLKIT=${CMAKE_CUDA_COMPILER_ID}-${CMAKE_CUDA_COMPILER_VERSION}+CUDA-${CUDAToolkit_VERSION}"
          "-DPRODUCER_FLAGS=${producer_flags}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/WriteCubinManifest.cmake"
      DEPENDS
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda/qwen_bf16_primitives.cu"
          "${CMAKE_CURRENT_SOURCE_DIR}/include/pih/model/qwen3_cuda_invariant.h"
          "${CMAKE_CURRENT_SOURCE_DIR}/cmake/WriteCubinManifest.cmake"
      VERBATIM
  )
  add_custom_target(
      ${target_name} ALL DEPENDS "${output_file}" "${manifest_file}"
  )
  install(
      FILES "${output_file}" "${manifest_file}"
      DESTINATION "pih/cubin/sm_${architecture}"
  )
endfunction()
