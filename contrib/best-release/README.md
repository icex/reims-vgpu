# DAILY-based fork release

This branch retains the DAILY host sources, with VRAM backing admission and
device-selection review fixes. Its pinned metal2vulkan fork has the same source
tree as the former local runtime override. QEMU includes the matching desktop
fixes and a VM-local `-accel kvm,halt-poll-ns=20000` control. Omit that property
to use the kernel default. `REIMS_VGPU_VRAM_SURFACES=1` opts into VRAM backings.

`build.sh OUTPUT_DIRECTORY` builds the release archive, source ROM, patched
libslirp 4.9.5 and the pinned QEMU into OUTPUT_DIRECTORY/bundle. It requires
Cargo/Rust 1.98.1, C/C++ and clang/lld, Meson/Ninja, pkg-config, and QEMU GTK,
SPICE, PulseAudio and USB-redirection development dependencies. Set
CARGO_TARGET_DIR to a separate writable build cache, and JOBS to bound CPU use.
The script requires a fresh output directory for its first run. The libslirp
patch preserves the tested EOF polling fix; its base is pinned by commit.

Controlled drag/video trials informed the selection of a 20 us halt-poll cap.
One paired video gap count regressed. Neither sustained 240 presents/s nor
physical 240 Hz was qualified. These sources preserve a selected trade-off;
they do not claim all-workload performance qualification.
