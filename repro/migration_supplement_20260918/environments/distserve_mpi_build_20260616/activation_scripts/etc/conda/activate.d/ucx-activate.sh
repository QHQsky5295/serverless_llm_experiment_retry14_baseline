if [ "${CONDA_BUILD:-}" = "1" ]; then
  if [[ "${target_platform:-}" == "linux-aarch64" || "${target_platform:-}" == "linux-ppc64le" ]]; then
    # disable UCX_MEM_EVENTS during build on likely emulated platforms,
    # as it's known to segfault sometimes in these situations
    export UCX_MEM_EVENTS="${UCX_MEM_EVENTS:-no}"
  fi
fi
