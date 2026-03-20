#!/usr/bin/env bash

detect_openssl3_libdir() {
  if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q 'libssl\.so\.3'; then
    return 1
  fi

  local candidates=()
  if [[ -n "${OPENSSL3_LIB_DIR:-}" ]]; then
    candidates+=("${OPENSSL3_LIB_DIR}")
  fi
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    candidates+=("${CONDA_PREFIX}/lib")
  fi
  candidates+=(
    "${HOME}/miniforge3/lib"
    "${HOME}/miniforge3/envs/simple_py310/lib"
    "${HOME}/miniconda3/lib"
    "${HOME}/mambaforge/lib"
  )

  local libdir
  for libdir in "${candidates[@]}"; do
    if [[ -n "${libdir}" && -f "${libdir}/libssl.so.3" && -f "${libdir}/libcrypto.so.3" ]]; then
      printf '%s\n' "${libdir}"
      return 0
    fi
  done

  return 1
}

append_library_path() {
  local libdir="$1"
  if [[ -z "${libdir}" ]]; then
    return 0
  fi

  case ":${LD_LIBRARY_PATH:-}:" in
    *":${libdir}:"*) ;;
    *) export LD_LIBRARY_PATH="${libdir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
}

if libdir="$(detect_openssl3_libdir)"; then
  append_library_path "${libdir}"
fi
