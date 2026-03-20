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

lib_supports_symbol_version() {
  local libpath="$1"
  local symbol_version="$2"
  [[ -f "${libpath}" ]] || return 1
  command -v strings >/dev/null 2>&1 || return 1
  strings "${libpath}" 2>/dev/null | grep -Fqx "${symbol_version}"
}

current_libstdcpp_satisfies() {
  local required_version="$1"
  local search_dir libpath

  IFS=':' read -r -a search_dirs <<< "${LD_LIBRARY_PATH:-}"
  for search_dir in "${search_dirs[@]}"; do
    [[ -n "${search_dir}" ]] || continue
    libpath="${search_dir}/libstdc++.so.6"
    if lib_supports_symbol_version "${libpath}" "${required_version}"; then
      return 0
    fi
  done

  if command -v ldconfig >/dev/null 2>&1; then
    while IFS= read -r libpath; do
      if lib_supports_symbol_version "${libpath}" "${required_version}"; then
        return 0
      fi
    done < <(ldconfig -p 2>/dev/null | awk '/libstdc\+\+\.so\.6/{print $NF}')
  fi

  for libpath in \
    /lib64/libstdc++.so.6 \
    /usr/lib64/libstdc++.so.6 \
    /usr/lib/x86_64-linux-gnu/libstdc++.so.6
  do
    if lib_supports_symbol_version "${libpath}" "${required_version}"; then
      return 0
    fi
  done

  return 1
}

detect_libstdcpp_libdir() {
  local required_version="${LIBSTDCXX_REQUIRED_GLIBCXX:-GLIBCXX_3.4.26}"
  if current_libstdcpp_satisfies "${required_version}"; then
    return 1
  fi

  local candidates=()
  if [[ -n "${LIBSTDCXX_LIB_DIR:-}" ]]; then
    candidates+=("${LIBSTDCXX_LIB_DIR}")
  fi
  if [[ -n "${GCC_HOME:-}" ]]; then
    candidates+=("${GCC_HOME}/lib64" "${GCC_HOME}/lib")
  fi
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    candidates+=("${CONDA_PREFIX}/lib")
  fi

  local compiler libpath
  for compiler in g++ gcc c++; do
    if command -v "${compiler}" >/dev/null 2>&1; then
      libpath="$("${compiler}" -print-file-name=libstdc++.so.6 2>/dev/null || true)"
      if [[ -n "${libpath}" && "${libpath}" != "libstdc++.so.6" ]]; then
        candidates+=("$(dirname "${libpath}")")
      fi
    fi
  done

  local nullglob_was_set=0
  if shopt -q nullglob; then
    nullglob_was_set=1
  fi
  shopt -s nullglob
  candidates+=(
    /cm/local/apps/gcc/*/lib64
    /cm/local/apps/gcc/*/lib
    /opt/rh/*/root/usr/lib64
    "${HOME}/miniforge3/lib"
    "${HOME}/miniconda3/lib"
    "${HOME}/mambaforge/lib"
  )
  if [[ "${nullglob_was_set}" -eq 0 ]]; then
    shopt -u nullglob
  fi

  local libdir
  for libdir in "${candidates[@]}"; do
    if lib_supports_symbol_version "${libdir}/libstdc++.so.6" "${required_version}"; then
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

if libdir="$(detect_libstdcpp_libdir)"; then
  append_library_path "${libdir}"
fi
