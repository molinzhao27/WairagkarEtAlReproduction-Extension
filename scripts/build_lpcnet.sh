#!/usr/bin/env bash
# Build the LPCNet vocoder (xiph/LPCNet @ 7dc99429e246cca614708b5aa4990d827bacce2c, vendored without .git)
# in place under third_party/LPCNet, producing third_party/LPCNet/lpcnet_demo, the executable that the
# vendored brain_to_voice code calls as `<lpcnet_path>/lpcnet_demo -features|-synthesis <in> <out>`.
#
# Usage, from the repository root:
#     mamba run -n wvx bash scripts/build_lpcnet.sh            # macOS arm64 (NEON) or Linux x86_64 (AVX2)
#     FORCE_RECONFIGURE=1 mamba run -n wvx bash scripts/build_lpcnet.sh
#
# Idempotent: the model download/extraction, autoreconf and configure steps are skipped when their outputs are
# already present and valid; `make` is incremental. LPCNet's own autogen.sh is NOT used because it calls `wget`
# (absent on macOS); this script does the same steps (fetch model tarball, extract, autoreconf -isf) with curl and
# a pinned SHA-256. Only the src/ members of the tarball are extracted (models/*.h5 are Keras training
# checkpoints the C build does not use). No LPCNet source file is modified.
#
# Requires: autoconf, automake, libtool, make, pkg-config (from the env), a C compiler (Xcode clang on macOS,
# conda-forge c-compiler / gcc on Linux), curl, tar.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LPC="$REPO/third_party/LPCNet"

MODEL_COMMIT=2ddc476   # "first commit compatible with the current model", from LPCNet's autogen.sh
MODEL_TGZ="lpcnet_data-${MODEL_COMMIT}.tar.gz"
MODEL_URL="https://media.xiph.org/lpcnet/data/${MODEL_TGZ}"
MODEL_SHA256="2d8866d60bbc6e15839c913b3c026799fea3f039e4e4e2e73826c3391ca787f0"
MODEL_BYTES=102536511
MODEL_STAMP="$LPC/.model_stamp"       # records the verified tarball + sha256 of every extracted file
BUILD_STAMP="$LPC/.build_flags_stamp" # records the configure flags of the current build tree

log() { printf '[build_lpcnet] %s\n' "$*"; }
die() { printf '[build_lpcnet] ERROR: %s\n' "$*" >&2; exit 1; }

sha256() {
    if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
    else sha256sum "$1" | cut -d' ' -f1; fi
}

[ -f "$LPC/configure.ac" ] && [ -f "$LPC/src/lpcnet_demo.c" ] || die "LPCNet sources not found at $LPC"
for tool in autoreconf libtoolize make curl tar; do
    command -v "$tool" >/dev/null 2>&1 || die "'$tool' not on PATH (run inside the env: mamba run -n wvx bash scripts/build_lpcnet.sh)"
done
CC="${CC:-cc}"
command -v "$CC" >/dev/null 2>&1 || die "C compiler '$CC' not found"

# ---------------------------------------------------------------- platform flags
case "$(uname -s)-$(uname -m)" in
    Darwin-arm64|Linux-aarch64)
        # NEON (and the 8-bit dot-product path, __ARM_FEATURE_DOTPROD) are enabled by default on arm64;
        # -mfpu=neon from the LPCNet README is an ARMv7-only flag. -O3 -ffast-math == upstream's -Ofast.
        ARCH_CFLAGS="" ;;
    Linux-x86_64|Darwin-x86_64)
        # x86-64-v3 = AVX2 + FMA (fixed target instead of upstream's -march=native, for reproducibility)
        ARCH_CFLAGS="-march=x86-64-v3" ;;
    *) die "unsupported platform $(uname -s)-$(uname -m)" ;;
esac
CFLAGS_LPC="-O3 -ffast-math -g ${ARCH_CFLAGS}"
CONFIGURE_ARGS="--disable-shared --enable-static --disable-doc"   # static: lpcnet_demo is a real binary, not a libtool wrapper

# ---------------------------------------------------------------- 1. model data (src/nnet_data.c etc.)
model_ok() {
    [ -f "$MODEL_STAMP" ] || return 1
    grep -q "^tarball_sha256 ${MODEL_SHA256}\$" "$MODEL_STAMP" || return 1
    local f h
    while read -r tag h f; do
        [ "$tag" = "file" ] || continue
        [ -f "$LPC/$f" ] && [ "$(sha256 "$LPC/$f")" = "$h" ] || return 1
    done < "$MODEL_STAMP"
    return 0
}

if model_ok; then
    log "model data present and verified (stamp: $MODEL_STAMP)"
else
    TGZ="$LPC/$MODEL_TGZ"
    if [ ! -f "$TGZ" ] || [ "$(sha256 "$TGZ")" != "$MODEL_SHA256" ]; then
        log "downloading $MODEL_URL ($MODEL_BYTES bytes)"
        curl -fL --retry 3 -o "$TGZ.part" "$MODEL_URL"
        mv "$TGZ.part" "$TGZ"
    fi
    got="$(sha256 "$TGZ")"
    [ "$got" = "$MODEL_SHA256" ] || die "SHA-256 mismatch for $TGZ: got $got, expected $MODEL_SHA256"
    # Refuse anything outside src/ or with path traversal before extracting into the source tree.
    MEMBERS="$(tar -tzf "$TGZ" | grep '^src/' || true)"
    [ -n "$MEMBERS" ] || die "no src/ members in $TGZ"
    if printf '%s\n' "$MEMBERS" | grep -Eq '(^/|\.\.)'; then die "unsafe path in $TGZ"; fi
    log "extracting src/ members of $MODEL_TGZ"
    # shellcheck disable=SC2086
    (cd "$LPC" && tar -xzof "$TGZ" $MEMBERS)
    touch "$LPC"/src/nnet_data.[ch] "$LPC"/src/plc_data.[ch]   # as LPCNet's download_model.sh does
    {
        echo "# LPCNet model data, written by scripts/build_lpcnet.sh"
        echo "url $MODEL_URL"
        echo "tarball_sha256 $MODEL_SHA256"
        echo "tarball_bytes $MODEL_BYTES"
        for f in $MEMBERS; do echo "file $(sha256 "$LPC/$f") $f"; done
    } > "$MODEL_STAMP"
    rm -f "$TGZ"   # verified + extracted; the stamp keeps its URL and hash (saves ~100 MB)
    model_ok || die "model data verification failed after extraction"
fi

# ---------------------------------------------------------------- 2. autotools (equivalent of autogen.sh's `autoreconf -isf`)
cd "$LPC"
if [ -n "${FORCE_RECONFIGURE:-}" ] || [ ! -x configure ] || [ ! -f Makefile.in ] || [ configure.ac -nt configure ]; then
    log "autoreconf -isf"
    autoreconf -isf
fi

# ---------------------------------------------------------------- 3. configure
WANT="CC=$CC CFLAGS=$CFLAGS_LPC ARGS=$CONFIGURE_ARGS"
if [ -n "${FORCE_RECONFIGURE:-}" ] || [ ! -f config.status ] || [ ! -f Makefile ] \
   || [ ! -f "$BUILD_STAMP" ] || [ "$(cat "$BUILD_STAMP")" != "$WANT" ]; then
    log "configure: $WANT"
    [ -f Makefile ] && make distclean >/dev/null 2>&1 || true
    # shellcheck disable=SC2086
    CC="$CC" CFLAGS="$CFLAGS_LPC" ./configure $CONFIGURE_ARGS
    echo "$WANT" > "$BUILD_STAMP"
else
    log "configure up to date ($WANT)"
fi

# ---------------------------------------------------------------- 4. build
JOBS="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
log "make -j$JOBS"
make -j"$JOBS"

# ---------------------------------------------------------------- 5. verify
for exe in lpcnet_demo dump_data; do
    [ -x "$LPC/$exe" ] || die "$exe was not built"
done
log "built: $(file -b "$LPC/lpcnet_demo" 2>/dev/null || echo lpcnet_demo)"
log "OK: $LPC/lpcnet_demo"
