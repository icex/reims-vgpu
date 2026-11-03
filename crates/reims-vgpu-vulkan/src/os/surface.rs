//! What this platform reads of a surface beyond core's `Surface` trait, written against it and
//! each OS's record of the surfaces it made (`recorded_mut`): a plane's layout and YCbCr facts,
//! the host memory a surface's bytes are over, whether the GPU's writes through it reach that
//! memory, and the alias its buffers lie over.

use std::sync::{Arc, Weak};

use metal2vulkan_runtime as rt;
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::os::surface::Surface;

use super::{OsSurface, host_os};
use crate::gpu::backing::Alias;
use crate::gpu::timing;

/// Plane `plane` of `surface`, as a texture on it reads it: its bytes' offset from the surface's
/// base address, its bytes per row, and whether the surface has planes. A surface without planes
/// is its plane 0. `None` for a plane the surface lacks, of which Metal makes no texture
/// (`textures_backing_records`). The plane names no `kIOSurfaceClientDirection`, which the
/// runtime refuses: where the GPU's writes through a surface never reach its memory
/// ([`gpu_read_only`]), the buffer under the texture lies over a read-only alias of that memory
/// (the device's `buffer_over_surface_at`), and the GPU's writes never reach it there either.
pub(crate) fn plane(surface: &OsSurface, plane: u64) -> Option<rt::IoSurfacePlane> {
    if surface.plane_count() == 0 {
        return (plane == 0).then(|| rt::IoSurfacePlane::new(0, surface.bytes_per_row(), false));
    }
    let (offset, bytes_per_row) = host_os::plane_at(surface, plane)?;
    Some(rt::IoSurfacePlane::new(offset, bytes_per_row, true))
}

/// What a texture over plane `plane` of `surface`, a surface of two planes, reads of its layout,
/// which Metal reads from it when the texture is made: its OSType, its size, and its plane 1's
/// offset from its base address and bytes per row. `None` for a surface of fewer than two planes,
/// over which the runtime then refuses the texture. A texture of a two-plane XR10 format (550,
/// 551) takes these alone, plane 1 its alpha bytes ([`crate::gpu::device`]).
pub(crate) fn plane_facts(surface: &OsSurface, plane: u64) -> Option<rt::IoSurfaceYCbCr> {
    if surface.plane_count() < 2 {
        return None;
    }
    let (chroma, bytes_per_row) = host_os::plane_at(surface, 1)?;
    Some(rt::IoSurfaceYCbCr::new(
        surface.pixel_format(),
        plane,
        [surface.width(), surface.height()],
        chroma,
        bytes_per_row,
    ))
}

/// What a texture of a two-plane YCbCr format the runtime reads
/// ([`crate::gpu::ycbcr::runtime_makes`]) over plane `plane` of `surface` reads of the surface:
/// [`plane_facts`], and its YCbCr matrix as it is now, where it names one
/// (`kIOSurfaceYCbCrMatrix`).
pub(crate) fn ycbcr_facts(surface: &OsSurface, plane: u64) -> Option<rt::IoSurfaceYCbCr> {
    let facts = plane_facts(surface, plane)?;
    Some(match host_os::ycbcr_matrix(surface) {
        Some(matrix) => facts.with_matrix(matrix),
        None => facts,
    })
}

/// The host memory a surface's bytes are over: its allocation in its own mapping (its
/// allocation size from its base address less its `kIOSurfaceOffset`), and, for one
/// `OsSurfaces::create` made over the guest's pages (`kIOSurfaceClientAddressRanges`),
/// those pages as the host maps them, where the guest's other objects over the same pages lie.
/// Each range is a first byte and an end.
///
/// The surface's base address is its offset into its allocation, and a plane's
/// (`IOSurfaceGetBaseAddressOfPlane`) its `kIOSurfacePlaneOffset` into it. A surface over one
/// run of the guest's pages at offset 0 has them as its base address; at another offset, its
/// allocation is another mapping of the pages, the base address that offset into it, and a
/// write through it shows in the pages at that offset (measured on the M3 Max, offsets 0x40,
/// 0x1000 and 0x4000).
pub(crate) fn surface_memory(surface: &OsSurface) -> Vec<std::ops::Range<u64>> {
    let start = timing::span_start();
    let base = surface.base_address().as_ptr::<u8>().addr().to_u64();
    let made = made_over(surface);
    let first = base.saturating_sub(made.offset);
    let own = std::iter::once(first..first.saturating_add(surface.alloc_size()));
    let pages = made.pages.len().to_u64();
    let memory: Vec<_> = own.chain(made.pages).collect();
    // sp 20: the runs of the guest's pages the surface was made over, the ranges answered
    timing::span(20, start, &[pages, memory.len().to_u64()]);
    memory
}

/// The host memory of `surface`'s bytes at `offsets`, each a first byte and an end counted from
/// the surface's base address, where [`surface_memory`] has them: from its base address, then in
/// the guest's pages it was made over, in their order, from its offset into them.
pub(crate) fn surface_bytes_memory(
    surface: &OsSurface,
    offsets: &[std::ops::Range<u64>],
) -> Vec<std::ops::Range<u64>> {
    let start = timing::span_start();
    let base = surface.base_address().as_ptr::<u8>().addr().to_u64();
    let own = offsets
        .iter()
        .map(|o| base.saturating_add(o.start)..base.saturating_add(o.end));
    let made = made_over(surface);
    let at = |o: &std::ops::Range<u64>| {
        made.offset.saturating_add(o.start)..made.offset.saturating_add(o.end)
    };
    let paged: Vec<_> = offsets
        .iter()
        .flat_map(|o| in_runs(&made.pages, &at(o)))
        .collect();
    let memory: Vec<_> = own.chain(paged).collect();
    // sp 21: the offsets asked, the runs of the guest's pages the surface was made over, the
    // ranges answered
    let counts = [offsets.len(), made.pages.len(), memory.len()].map(ToU64::to_u64);
    timing::span(21, start, &counts);
    memory
}

/// The host ranges that hold the bytes `offsets` of memory laid out over `runs` in their order
/// (each a host first byte and an end).
pub(crate) fn in_runs(
    runs: &[std::ops::Range<u64>],
    offsets: &std::ops::Range<u64>,
) -> Vec<std::ops::Range<u64>> {
    let mut held = Vec::new();
    let mut at = 0u64;
    for run in runs {
        let length = run.end - run.start;
        let (first, end) = (offsets.start.max(at), offsets.end.min(at + length));
        if first < end {
            held.push(run.start + (first - at)..run.start + (end - at));
        }
        at += length;
        if at >= offsets.end {
            break;
        }
    }
    held
}

/// The offset of `surface`'s bytes into its allocation (`kIOSurfaceOffset`), as
/// `OsSurfaces::create` made it: its base address lies that far into the allocation, a
/// page-aligned mapping of `IOSurfaceGetAllocSize` bytes (measured on the M3 Max at 0x40, 0x1000
/// and 0x4000, over the guest's pages); 0 for a surface made otherwise.
pub(crate) fn surface_offset(surface: &OsSurface) -> u64 {
    made_over(surface).offset
}

/// Whether `surface` was made over the guest's pages (`OsSurfaces::create`).
pub(crate) fn over_guest_pages(surface: &OsSurface) -> bool {
    recorded(surface, |made| !made.pages.is_empty()).unwrap_or(false)
}

/// What `surface` was made over, as its OS records it: no pages and offset 0 for a surface
/// made over neither the guest's pages nor an offset into its allocation.
fn made_over(surface: &OsSurface) -> MadeOver {
    let start = timing::span_start();
    let made = recorded(surface, |made| made.clone()).unwrap_or_default();
    // sp 23: the runs of the guest's pages copied
    timing::span(23, start, &[made.pages.len().to_u64()]);
    made
}

/// The `kIOSurfaceClientDirection` of a surface core makes over the guest's pages read-only for
/// its clients (core's `iosurface::descriptor`).
pub(crate) const READ_ONLY_DIRECTION: u32 = 2;

/// Whether the GPU's writes through `surface` never reach its memory: it was made over the
/// guest's pages with `kIOSurfaceClientDirection` 2, which Metal maps read-only for the GPU.
///
/// Measured on the M3 Max (macOS 26.7, one case per process, BGRA8Unorm surfaces of 64 by 16, 256
/// by 256 and 1024 by 1024 texels, `tests/client_direction.rs`): through a texture on such a
/// surface, a blit's copy or colour fill and a kernel's writes of one or two texels reach the later
/// commands of their command buffer and never the pages, which later command buffers and the CPU
/// read unchanged; of a row of 64 texels or more, they fail the command buffer with a GPU page
/// fault (`MTLCommandBufferErrorPageFault`), the encoders after them running nothing. So does a
/// blit's copy of a texture of a row of 64 texels into a buffer made on the surface with
/// `newBufferWithIOSurface:`, where one of two texels is dropped as above. A blit's copy from a
/// buffer or colour fill of 1 to 8 texels each way into the texture is dropped at every origin
/// tried (0,0, 1,1 and one far inside, on surfaces of 64 by 16 and 256 by 256 texels); of a row of
/// 16 to 63 texels it faults at some origins and is dropped at others (16 at 0,0 faults, at 1,1 is
/// dropped). A blit's fill of a buffer made on the surface, its copy into one from another buffer,
/// a kernel's stores into one, plain or by an atomic exchange, and a vertex or fragment function's,
/// of 8 bytes to 4 MiB, are dropped as above, the later commands of the command buffer reading what
/// of the write the GPU's caches still hold (all of 4 MiB with nothing else on the GPU, less beside
/// other GPU work, none after 64 MiB of it; a draw's, all, part or none of it, case to case, in one
/// process). A render pass that clears it or draws into it and stores fails its command buffer so;
/// one that stores nothing, or loads, stores and draws nothing, completes, writing nothing.
/// `replaceRegion:` writes through the surface's base address: the page of a surface over one page,
/// which lies there; a surface over several (4, 16 and 256) has a mapping of its own there,
/// read-only, and the write ends the process with a bus error, at every extent measured
/// ([`base_in_pages`]). The key is not read back (`IOSurfaceCopyValue` answers none), so the
/// surfaces made here record it. A surface of its own memory takes the key and the GPU's writes
/// alike; one over the guest's pages takes 3 and the writes land, and none of 0, 1, 4 or 255
/// (`IOSurfaceCreate` answers NULL); core sets 2 alone.
pub(crate) fn gpu_read_only(surface: &OsSurface) -> bool {
    recorded(surface, |made| made.gpu_read_only).unwrap_or(false)
}

/// Dies at `sel`, a command that may write `what` into the memory of a surface whose GPU writes
/// never reach it ([`gpu_read_only`]): Metal's GPU drops some such writes and faults the command
/// buffer at others by their extent, which nothing here can tell before the GPU runs the
/// command, and the runtime's device loses itself at the same fault, so the write is never sent.
/// A blit's fill of a buffer made on the surface, its copy into one from another buffer, and a
/// kernel's, vertex function's or fragment function's stores into one, which Metal drops at
/// every extent measured, as the runtime does, are sent; a tile function's, which nothing
/// measured, are not. So is a blit's copy from a buffer or colour fill into a `BGRA8Unorm` 2D
/// texture made on the surface, of 1 to 8 texels each way inside level 0, on a surface whose rows
/// hold 256 bytes or more, which Metal drops at every origin measured
/// (`gpu::encode_blit::dropped_where_measured`).
pub(crate) fn dropped_write(sel: &'static str, what: &str) -> ! {
    crate::answer::unrepresentable(
        sel,
        &format!(
            "{what} a surface made read-only for the GPU (kIOSurfaceClientDirection 2), whose \
             writes Metal's GPU drops or faults on by their extent"
        ),
    )
}

/// The alias of the allocation of `surface`, one at an offset into it ([`surface_offset`]) or
/// whose GPU writes never reach its memory, read-only then ([`gpu_read_only`]), that every buffer
/// over it lies over while one lives: the live one, or the one `make` makes, kept for the next;
/// `None` where `make` makes none. The buffers then lie in one import of the runtime's, as those
/// over a surface at no offset whose writes land lie over its base address: a write through one
/// reaches the later commands of its command buffer that read another, as on Metal
/// (`tests/client_direction.rs`), and a buffer over the surface can share the range of another
/// over the same first byte (the device's `new_buffer_on_surface`).
pub(crate) fn surface_alias(
    surface: &OsSurface,
    make: impl FnOnce() -> Option<Alias>,
) -> Option<Arc<Alias>> {
    let mut make = Some(make);
    let kept = host_os::recorded_mut(surface, |made| {
        if let Some(alias) = made.alias.upgrade() {
            return Some(alias);
        }
        let alias = Arc::new(make.take()?()?);
        made.alias = Arc::downgrade(&alias);
        Some(alias)
    });
    match kept {
        Some(alias) => alias,
        None => make.take()?().map(Arc::new),
    }
}

/// Whether `surface`'s base address lies in the guest's pages it was made over, so that the CPU's
/// writes through it reach them: as for a surface over one page; one over several, contiguous or
/// not, has a mapping of its own (measured on the M3 Max: one of 4 contiguous pages), read-only
/// for a surface of [`gpu_read_only`].
pub(crate) fn base_in_pages(surface: &OsSurface) -> bool {
    let base = surface.base_address().as_ptr::<u8>().addr().to_u64();
    recorded(surface, |made| {
        made.pages.iter().any(|run| run.contains(&base))
    })
    .unwrap_or(false)
}

/// The memory `surface`'s bytes are in, as a diagnostic names it: its own, for one not made over
/// the guest's pages; the guest's pages, where its base address lies in them
/// ([`base_in_pages`]); or a mapping of its own of the guest's pages, as of several pages, or of
/// one run at an offset on macOS, which IOSurface makes, and of several ranges on Linux, the
/// stand-in's view of them.
pub(crate) fn memory_kind(surface: &OsSurface) -> &'static str {
    if !over_guest_pages(surface) {
        "its own memory"
    } else if base_in_pages(surface) {
        "the guest's pages"
    } else {
        "a mapping of its own of the guest's pages"
    }
}

/// Writes `bytes` at `offset` in the bytes of `surface`, counted from its first byte, into the
/// guest's pages it was made over, where those bytes lie (from its `kIOSurfaceOffset` in the
/// pages, in their order): what the CPU writes through Metal's texture on such a surface
/// reaches (`replaceRegion:`, measured: `tests/client_direction.rs`). False, writing nothing,
/// for a surface not made over the guest's pages, and for bytes past them.
pub(crate) fn write_surface_bytes(surface: &OsSurface, offset: u64, bytes: &[u8]) -> bool {
    let Some((pages, start)) = recorded(surface, |made| (made.pages.clone(), made.offset)) else {
        return false;
    };
    let first = start.saturating_add(offset);
    let held = in_runs(&pages, &(first..first.saturating_add(bytes.len().to_u64())));
    let total: u64 = held.iter().map(|r| r.end - r.start).sum();
    if total != bytes.len().to_u64() {
        return false;
    }
    let mut from = bytes;
    for run in held {
        let (now, rest) = from.split_at((run.end - run.start).to_usize());
        // SAFETY: `run` lies in the guest's pages the surface was made over, which the host maps
        // readable and writable while the surface lives, as core holds them; `now` is the
        // caller's, apart from them.
        unsafe {
            std::ptr::copy_nonoverlapping(
                now.as_ptr(),
                std::ptr::with_exposed_provenance_mut::<u8>(run.start.to_usize()),
                now.len(),
            );
        }
        from = rest;
    }
    true
}

/// What the OS's record of `surface` holds, read by `read`; `None` for a surface it did not
/// record.
fn recorded<T>(surface: &OsSurface, read: impl FnOnce(&MadeOver) -> T) -> Option<T> {
    host_os::recorded_mut(surface, |made| read(made))
}

/// What a surface was made over: the guest's pages, in its bytes' order, merged into runs, and
/// its bytes' offset into its allocation (`kIOSurfaceOffset`); whether the GPU's writes through
/// it never reach them ([`gpu_read_only`]), and the alias of its allocation its buffers lie
/// over while one lives, where they lie over one ([`surface_alias`]).
#[derive(Debug, Clone, Default)]
pub(crate) struct MadeOver {
    pub(crate) pages: Vec<std::ops::Range<u64>>,
    pub(crate) offset: u64,
    pub(crate) gpu_read_only: bool,
    pub(crate) alias: Weak<Alias>,
}

/// The runs of `pages`, each a host first byte and a length, those that follow on in the host's
/// memory merged, each a first byte and an end.
pub(crate) fn page_runs(pages: impl IntoIterator<Item = (u64, u64)>) -> Vec<std::ops::Range<u64>> {
    let mut runs: Vec<std::ops::Range<u64>> = Vec::new();
    for (first, length) in pages {
        let end = first.saturating_add(length);
        match runs.last_mut() {
            Some(run) if run.end == first => run.end = end,
            _ => runs.push(first..end),
        }
    }
    runs
}

#[cfg(test)]
pub(crate) mod tests {
    use reims_vgpu_core::host::{HostPtr, PhysicalMemoryRange};
    use reims_vgpu_core::platform::Handle as _;
    use reims_vgpu_core::platform::os::surface::{IoSurfaces as _, SurfaceSpec};
    use reims_vgpu_core::platform::os::vm::HostVm as _;

    use super::*;
    use crate::os::{OsSurfaces, OsVm, allocate_aliasable};

    /// The bytes of memory laid out over runs are found in each run they reach, in order
    #[test]
    #[expect(clippy::single_range_in_vec_init, reason = "lists of host ranges")]
    fn in_runs_finds_bytes_in_each_run_they_reach() {
        let runs = [0x10_000..0x14_000, 0x40_000..0x48_000, 0x20_000..0x24_000];
        assert_eq!(in_runs(&runs, &(0x100..0x200)), [0x10_100..0x10_200]);
        assert_eq!(
            in_runs(&runs, &(0x3f00..0x4100)),
            [0x13_f00..0x14_000, 0x40_000..0x40_100]
        );
        assert_eq!(
            in_runs(&runs, &(0x3000..0xd000)),
            [0x13_000..0x14_000, 0x40_000..0x48_000, 0x20_000..0x21_000]
        );
        assert_eq!(in_runs(&runs, &(0x10_000..0x11_000)), []);
        assert_eq!(in_runs(&[], &(0..0x100)), []);
    }

    /// A 'BGRA' surface of `width` × `height`, 4 bytes an element, rows of `width × 4`, made over
    /// fresh pages of the host's (`kIOSurfaceClientAddressRanges`) `offset` bytes into them
    /// (`kIOSurfaceOffset`), and those pages: their first byte and length, which the caller frees
    /// once the surface is gone.
    pub(crate) fn bgra_surface_over_at(
        width: u64,
        height: u64,
        offset: u64,
    ) -> (OsSurface, HostPtr, u64) {
        let page = OsVm::new().page_size();
        let length = (width * height * 4 + offset).next_multiple_of(page);
        let pages = allocate_aliasable(length).unwrap();
        let first = pages.addr().to_u64();
        let spec = SurfaceSpec {
            pixel_format: u32::from_be_bytes(*b"BGRA"),
            offset,
            alloc_size: length,
            width,
            height,
            bytes_per_element: 4,
            bytes_per_row: width * 4,
            element_width: 1,
            element_height: 1,
            client_address_ranges: Some(
                (0..length / page)
                    .map(|i| PhysicalMemoryRange {
                        physical_address: first + i * page,
                        physical_length: page,
                    })
                    .collect(),
            ),
            ..SurfaceSpec::default()
        };
        (OsSurfaces::new().create(&spec).unwrap(), pages, length)
    }

    /// A host-only 'BGRA' surface of `width` × `height`, 4 bytes an element, rows of `width × 4`.
    pub(crate) fn bgra_surface(width: u64, height: u64) -> OsSurface {
        let spec = SurfaceSpec {
            pixel_format: u32::from_be_bytes(*b"BGRA"),
            alloc_size: width * height * 4,
            width,
            height,
            bytes_per_element: 4,
            bytes_per_row: width * 4,
            element_width: 1,
            element_height: 1,
            ..SurfaceSpec::default()
        };
        OsSurfaces::new().create(&spec).unwrap()
    }

    /// The pages of [`bgra_surface_over_at`], freed.
    pub(crate) fn free(pages: HostPtr, length: u64) {
        assert!(OsVm::new().deallocate(pages, length).is_success());
    }

    /// A lookup of a surface, as an isolated task's surface is (`lookup_by_port`), a reference
    /// of its own, answers what the surface was made over: its offset, memory and GPU writes
    #[test]
    fn a_lookup_of_a_surface_answers_its_record() {
        let (s, pages, length) = bgra_surface_over_at(16, 16, 0x100);
        let t = s.lookup_by_port().unwrap();
        assert_ne!(t.id(), s.id());
        assert_eq!(surface_offset(&t), 0x100);
        assert_eq!(surface_memory(&t), surface_memory(&s));
        let page = OsVm::new().page_size();
        let guest = allocate_aliasable(page).unwrap();
        let spec = SurfaceSpec {
            pixel_format: u32::from_be_bytes(*b"BGRA"),
            alloc_size: page,
            width: 16,
            height: 16,
            bytes_per_element: 4,
            bytes_per_row: 64,
            element_width: 1,
            element_height: 1,
            client_direction: Some(READ_ONLY_DIRECTION),
            client_address_ranges: Some(vec![PhysicalMemoryRange {
                physical_address: guest.addr().to_u64(),
                physical_length: page,
            }]),
            ..SurfaceSpec::default()
        };
        let read_only = OsSurfaces::new().create(&spec).unwrap();
        let looked_up = read_only.lookup_by_port().unwrap();
        assert!(gpu_read_only(&read_only) && gpu_read_only(&looked_up));
        assert!(!gpu_read_only(&s) && !gpu_read_only(&t));
        drop((t, s, looked_up, read_only));
        free(pages, length);
        free(guest, page);
    }

    /// The value of the `cfg` predicate `p` in every build of the product, where it is one: `test`
    /// is off there, and so is the `testkit` feature, which only dev-dependencies enable (xtask's
    /// testkit lint); any other option may be on or off, and a predicate it decides has none.
    fn in_product(p: &str) -> Option<bool> {
        let p = p.trim();
        let args = |name: &str| {
            let inner = p.strip_prefix(name)?.trim_start();
            let inner = inner.strip_prefix('(')?.strip_suffix(')')?;
            let (mut parts, mut depth, mut start) = (Vec::new(), 0usize, 0);
            for (i, c) in inner.char_indices() {
                match c {
                    '(' => depth += 1,
                    ')' => depth = depth.saturating_sub(1),
                    ',' if depth == 0 => {
                        parts.push(&inner[start..i]);
                        start = i + 1;
                    }
                    _ => {}
                }
            }
            parts.push(&inner[start..]);
            let parts = parts.into_iter().filter(|part| !part.trim().is_empty());
            Some(parts.map(in_product).collect::<Vec<_>>())
        };
        if let Some(values) = args("not") {
            return values.first().copied().flatten().map(|value| !value);
        }
        if let Some(values) = args("any") {
            return if values.contains(&Some(true)) {
                Some(true)
            } else if values.contains(&None) {
                None
            } else {
                Some(false)
            };
        }
        if let Some(values) = args("all") {
            return if values.contains(&Some(false)) {
                Some(false)
            } else if values.contains(&None) {
                None
            } else {
                Some(true)
            };
        }
        let atom: String = p.chars().filter(|c| !c.is_whitespace()).collect();
        (atom == "test" || atom == "feature=\"testkit\"").then_some(false)
    }

    /// `line`'s code: without its string and character literals' contents, or its comment, where
    /// `in_string` says whether a string continues into it, and then whether one continues past it.
    fn code_of(line: &str, in_string: &mut bool) -> String {
        let chars: Vec<char> = line.chars().collect();
        let (mut code, mut i) = (String::new(), 0);
        while i < chars.len() {
            let c = chars[i];
            i += 1;
            if *in_string {
                match c {
                    '\\' => i += 1,
                    '"' => *in_string = false,
                    _ => {}
                }
                continue;
            }
            match c {
                '"' => *in_string = true,
                '/' if chars.get(i) == Some(&'/') => break,
                // A character literal (a lifetime has no closing quote after one character).
                '\'' if chars.get(i) == Some(&'\\') => {
                    let rest = chars.get(i + 2..).unwrap_or_default();
                    i += rest.iter().position(|&c| c == '\'').map_or(1, |at| at + 3);
                }
                '\'' if chars.get(i + 1) == Some(&'\'') => i += 2,
                c => code.push(c),
            }
        }
        code
    }

    /// The name of the module `line` declares, and what follows it (`;` or `{`), whatever its
    /// visibility.
    fn module_declared(line: &str) -> Option<&str> {
        let rest = match line.strip_prefix("pub") {
            Some(rest) if rest.starts_with('(') => rest.split_once(')')?.1.trim_start(),
            Some(rest) if rest.starts_with(' ') => rest.trim_start(),
            Some(_) => return None,
            None => line,
        };
        rest.strip_prefix("mod ")
    }

    /// What a walk of the library's module tree saw ([`base_address_calls`]).
    #[derive(Default)]
    struct Walk {
        /// The calls of a surface's base address out of the tests, by file relative to the
        /// crates' directory.
        calls: std::collections::BTreeMap<String, usize>,
        /// The files of the modules it read.
        read: std::collections::BTreeSet<std::path::PathBuf>,
        /// The files of the modules it left as the tests' (declared under a `cfg` that holds in
        /// no build of the product, [`in_product`]), and the directories of their children.
        tests: Vec<std::path::PathBuf>,
    }

    /// The calls of a surface's base address in the library's code, out of its tests, from the
    /// module tree from `file`, whose child modules lie in `dir` or at the paths their attributes
    /// name, into `walk`: but those of the items under a `cfg` that holds in no build of the
    /// product, the bodies of such items and the modules declared under one.
    fn base_address_calls(
        root: &std::path::Path,
        (file, dir): (&std::path::Path, &std::path::Path),
        walk: &mut Walk,
    ) {
        let text =
            std::fs::read_to_string(file).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
        walk.read.insert(file.to_path_buf());
        let (mut test_only, mut in_string, mut depth, mut test_item) =
            (false, false, 0usize, false);
        let (mut attribute, mut brackets) = (String::new(), 0usize);
        let mut paths = Vec::new();
        // A test-only item whose body has not started starts it at its first brace, and one of
        // none ends with the line that ends in `;` or `,`.
        let ended = |code: &str| code.trim_end().ends_with([';', ',']);
        for line in text.lines().map(str::trim) {
            let starts_in_string = in_string;
            let code = code_of(line, &mut in_string);
            let (opens, closes) = (code.matches('{').count(), code.matches('}').count());
            if depth > 0 {
                depth = (depth + opens).saturating_sub(closes);
                continue;
            }
            if test_item {
                test_item = opens == 0 && !ended(&code);
                depth = opens.saturating_sub(closes);
                continue;
            }
            // Its attributes, over one line or several, and the item that may follow them.
            let mut item = line;
            while !starts_in_string
                && (brackets > 0 || item.starts_with("#[") || item.starts_with("#!["))
            {
                let mut end = None;
                for (i, c) in item.char_indices() {
                    match c {
                        '[' => brackets += 1,
                        ']' => {
                            brackets = brackets.saturating_sub(1);
                            if brackets == 0 {
                                end = Some(i);
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                let Some(end) = end else {
                    attribute.push_str(item);
                    attribute.push(' ');
                    item = "";
                    break;
                };
                attribute.push_str(&item[..=end]);
                item = item[end + 1..].trim();
                let whole = std::mem::take(&mut attribute);
                let whole: String = whole.chars().filter(|c| !c.is_whitespace()).collect();
                let cfg = |prefix: &str| {
                    let predicate = whole.strip_prefix(prefix)?.strip_suffix(")]")?;
                    Some(in_product(predicate) == Some(false))
                };
                if cfg("#![cfg(") == Some(true) {
                    walk.tests.extend([file.to_path_buf(), dir.to_path_buf()]);
                    return;
                }
                test_only |= cfg("#[cfg(") == Some(true);
                if whole.starts_with("#[")
                    && let Some((_, path)) = whole.split_once("path=\"")
                {
                    paths.push(path.split('"').next().unwrap().to_owned());
                }
            }
            if item.is_empty() || item.starts_with("//") {
                continue;
            }
            let (attributed, test) = (std::mem::take(&mut paths), std::mem::take(&mut test_only));
            let declared = (!starts_in_string).then(|| module_declared(item)).flatten();
            let children = |rest: &str| {
                let name = rest.trim_end_matches([';', '{']).trim();
                let own = dir.join(format!("{name}.rs"));
                let at = if own.exists() {
                    own
                } else {
                    dir.join(name).join("mod.rs")
                };
                (at, dir.join(name))
            };
            match declared {
                Some(rest) if test && rest.ends_with(';') => {
                    let attributed = attributed
                        .iter()
                        .map(|path| file.parent().unwrap().join(path));
                    walk.tests.extend(attributed);
                    let (at, children) = children(rest);
                    walk.tests.extend([at, children]);
                }
                Some(rest) if rest.ends_with(';') && !attributed.is_empty() => {
                    for path in attributed {
                        let at = file.parent().unwrap().join(path);
                        base_address_calls(root, (&at, at.parent().unwrap()), walk);
                    }
                }
                Some(rest) if rest.ends_with(';') => {
                    let (at, children) = children(rest);
                    base_address_calls(root, (&at, &children), walk);
                }
                _ if test => {
                    // An inline module's children lie under its name, as a declared one's.
                    if let Some(rest) = declared {
                        walk.tests.push(children(rest).1);
                    }
                    test_item = opens == 0 && !ended(&code);
                    depth = opens.saturating_sub(closes);
                }
                _ => {
                    let calls = item.matches(".base_address()").count()
                        + item.matches("Surface::base_address(").count();
                    if calls > 0 {
                        let at = file.strip_prefix(root).unwrap().display().to_string();
                        *walk.calls.entry(at).or_default() += calls;
                    }
                }
            }
        }
    }

    /// The `.rs` files under `dir`, into `files`.
    fn rust_files(dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
            let path = entry.unwrap().path();
            if path.is_dir() {
                rust_files(&path, files);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }

    /// The walk reads the items a `cfg` leaves in a build of the product and skips those it
    /// leaves in none, whatever their visibility or the lines they span
    #[test]
    fn the_base_address_walk_skips_only_the_tests() {
        for (predicate, expected) in [
            ("test", Some(false)),
            ("not(test)", Some(true)),
            ("all(test, not(loom))", Some(false)),
            ("all(test, target_os = \"linux\")", Some(false)),
            ("any(test, feature = \"testkit\")", Some(false)),
            ("not(any(test, feature = \"testkit\"))", Some(true)),
            ("not(any(test, feature = \"vulkan\"))", None),
            ("all(not(test), feature = \"vulkan\")", None),
            ("any(test, target_os = \"macos\")", None),
            ("target_os = \"linux\"", None),
        ] {
            assert_eq!(in_product(predicate), expected, "{predicate}");
        }
        assert_eq!(module_declared("pub(super) mod a;"), Some("a;"));
        assert_eq!(module_declared("pub(in crate::os) mod a {"), Some("a {"));
        assert_eq!(module_declared("pub fn a() {"), None);
        let mut in_string = false;
        let line = "let (a, b, c) = ('{', '\\'', \"}\"); // }";
        assert_eq!(code_of(line, &mut in_string), "let (a, b, c) = (, , ); ");
        assert_eq!(code_of("f::<'a>(\"{\\", &mut in_string), "f::<'a>(");
        assert!(in_string);
        assert_eq!(code_of("{\" }", &mut in_string), " }");
        assert!(!in_string);

        let dir = std::env::temp_dir().join(format!("reims-walk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("inner")).unwrap();
        let call = "    let _ = s.base_address();\n";
        let lib = [
            "#[cfg(not(test))]\npub(super) mod kept;\n",
            "#[cfg(\n    any(test, feature = \"testkit\")\n)]\npub(in crate) mod tested;\n",
            "pub(crate) mod inner;\n",
            "#[cfg(test)]\nfn helper(\n) {\n    let _ = '{';\n",
            call,
            "}\n",
            "#[cfg(test)] fn one() { let _ = s.base_address(); }\n",
            "#[cfg(not(test))]\nfn product() {\n",
            call,
            "}\n",
        ]
        .concat();
        std::fs::write(dir.join("lib.rs"), lib).unwrap();
        std::fs::write(dir.join("kept.rs"), call).unwrap();
        std::fs::write(dir.join("tested.rs"), call).unwrap();
        std::fs::write(dir.join("inner.rs"), "pub(super) mod deep;\n").unwrap();
        std::fs::write(dir.join("inner/deep.rs"), call).unwrap();
        let mut walk = Walk::default();
        base_address_calls(&dir, (&dir.join("lib.rs"), &dir), &mut walk);
        let calls: Vec<_> = walk
            .calls
            .iter()
            .map(|(file, &n)| (file.as_str(), n))
            .collect();
        assert_eq!(calls, [("inner/deep.rs", 1), ("kept.rs", 1), ("lib.rs", 1)]);
        assert!(walk.tests.contains(&dir.join("tested.rs")));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Every call of a surface's base address in the library's code is one of those listed, each
    /// with what it does with the memory: a CPU write of a surface of its own memory is
    /// declared to it (`Surface::cpu_wrote`), as the runtime keeps shadows of its bytes across
    /// command buffers on that declaration (`os::linux::surface`'s docs, "Declared writes"). A
    /// new call writes nothing there undeclared, and hands the address out of the library only
    /// after withdrawing the declaration; then it is listed here. The walk reaches every Rust
    /// file of the crates' sources, as a module it read or one it left as the tests'
    #[test]
    fn every_surface_base_address_call_is_listed() {
        let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap();
        let mut walk = Walk::default();
        let mut files = Vec::new();
        for name in [
            "reims-vgpu",
            "reims-vgpu-c",
            "reims-vgpu-core",
            "reims-vgpu-metal",
            "reims-vgpu-vulkan",
            "reims-vgpu-window",
        ] {
            let src = crates.join(name).join("src");
            base_address_calls(crates, (&src.join("lib.rs"), &src), &mut walk);
            rust_files(&src, &mut files);
        }
        let unreached: Vec<_> = (files.iter())
            .filter(|file| !walk.read.contains(*file))
            .filter(|file| !walk.tests.iter().any(|test| file.starts_with(test)))
            .collect();
        assert!(
            unreached.is_empty(),
            "files no module declaration reached: {unreached:?}"
        );
        let listed: std::collections::BTreeMap<String, usize> = [
            // A task's base address, not a surface's.
            ("reims-vgpu-core/src/device.rs", 1),
            // The test pattern, which declares its write (`cpu_wrote`).
            ("reims-vgpu-core/src/iosurface/table.rs", 1),
            // The runtime's import of the surface's memory, which carries the declaration.
            ("reims-vgpu-vulkan/src/gpu/device.rs", 1),
            // Address arithmetic: a plane's offset (macOS, whose surfaces declare nothing).
            ("reims-vgpu-vulkan/src/os/macos.rs", 1),
            // Address arithmetic: the surface's memory, its bytes', whether it lies in its pages.
            ("reims-vgpu-vulkan/src/os/surface.rs", 3),
        ]
        .into_iter()
        .map(|(file, calls)| (file.to_owned(), calls))
        .collect();
        assert_eq!(walk.calls, listed);
    }
}
