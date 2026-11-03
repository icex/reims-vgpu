#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=$(realpath -m -- "${1:?supply an output directory}")
: "${CARGO_TARGET_DIR:?set a private Cargo target directory}"
JOBS=${JOBS:-6}
mkdir -p "$OUT" "$OUT/bundle/lib/pkgconfig" "$OUT/bundle/bin" "$OUT/bundle/include" "$OUT/bundle/share/qemu"
export RUSTFLAGS=-Dwarnings
cargo build --manifest-path "$ROOT/Cargo.toml" --locked --release -p reims-vgpu-c --features vulkan -j "$JOBS"
cp "$CARGO_TARGET_DIR/release/libreims_vgpu_c.a" "$OUT/bundle/lib/libreims_vgpu.a"
cp "$ROOT/crates/reims-vgpu-c/include/reims_vgpu.h" "$OUT/bundle/include/"
make -C "$ROOT/firmware" -j "$JOBS" BUILD="$OUT/firmware"
cp "$OUT/firmware/ReimsVgpu.rom" "$OUT/bundle/share/qemu/"
cat > "$OUT/bundle/lib/pkgconfig/reims-vgpu.pc" <<'PC'
prefix=${pcfiledir}/../..
libdir=${prefix}/lib
includedir=${prefix}/include
Name: reims-vgpu
Description: Reims paravirtual GPU host
Version: 0.1.0
Libs: ${libdir}/libreims_vgpu.a -lm -ldl -lgcc_s -lutil -lrt -lpthread -lc
Cflags: -I${includedir}
PC
if [[ ! -d "$OUT/libslirp/.git" ]]; then
 git clone https://gitlab.freedesktop.org/slirp/libslirp.git "$OUT/libslirp"
 git -C "$OUT/libslirp" checkout --detach 390848f47391f412451b2d259d8527fabb51e5e3
 git -C "$OUT/libslirp" apply "$ROOT/contrib/best-release/libslirp-eof.patch"
fi
meson setup "$OUT/slirp-build" "$OUT/libslirp" --prefix="$OUT/slirp-prefix" --libdir=lib --buildtype=release
ninja -C "$OUT/slirp-build" -j "$JOBS"
meson test -C "$OUT/slirp-build" --print-errorlogs
meson install -C "$OUT/slirp-build"
cp -a "$OUT/slirp-prefix/lib/"libslirp.so* "$OUT/bundle/lib/"
if [[ ! -f "$ROOT/vendor/qemu/configure" ]]; then git -C "$ROOT" submodule update --init vendor/qemu; fi
if [[ ! -d "$ROOT/vendor/qemu/subprojects/keycodemapdb/.git" ]]; then
 git clone https://github.com/qemu/keycodemapdb.git "$ROOT/vendor/qemu/subprojects/keycodemapdb"
 git -C "$ROOT/vendor/qemu/subprojects/keycodemapdb" checkout --detach f5772a62ec52591ff6870b7e8ef32482371f22c6
fi
export PKG_CONFIG_PATH="$OUT/bundle/lib/pkgconfig:$OUT/slirp-prefix/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
mkdir -p "$OUT/qemu-build"
cd "$OUT/qemu-build"
"$ROOT/vendor/qemu/configure" --target-list=x86_64-softmmu --enable-kvm --enable-pvg --disable-sdl --enable-gtk --enable-spice --enable-usb-redir --enable-pa --enable-slirp --disable-docs --disable-tests --disable-werror --disable-rust --disable-download "--python=${PYTHON:-python3}"
ninja -j "$JOBS" qemu-system-x86_64
cp qemu-system-x86_64 "$OUT/bundle/bin/"
cp "$ROOT/vendor/qemu/scripts/reims-unix-tcp-relay.py" "$OUT/bundle/bin/"
cc -std=c11 -Wall -Wextra -Werror "$ROOT/vendor/qemu/tests/unit/test-reims-halt-poll.c" -o "$OUT/test-halt-poll"
"$OUT/test-halt-poll"
