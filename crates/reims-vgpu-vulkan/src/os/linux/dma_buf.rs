//! Dma-bufs of the pages behind a range the backing table resolves ([`super::files`]), for a
//! device that imports no shared memory as host memory.
//!
//! RADV imports host memory as an amdgpu userptr, which Mesa asks for with
//! `AMDGPU_GEM_USERPTR_ANONONLY`, so the kernel refuses every page with a file behind it: the
//! guest's RAM (QEMU's `memory-backend-memfd`) and every range the library makes a view of are
//! shared mappings of memfds, and none of them imports. `/dev/udmabuf` makes a dma-buf of those
//! same memfd pages (`UDMABUF_CREATE_LIST`), and RADV imports that
//! (`VK_EXT_external_memory_dma_buf`). Measured on a Radeon 780M (RADV of Mesa 26.0.8, Linux 7.0,
//! 2026-10-11), with no runtime between:
//! - a sealed memfd mapped shared: `vkGetMemoryHostPointerPropertiesEXT` answers
//!   `VK_ERROR_UNKNOWN`; private anonymous memory imports;
//! - a udmabuf of two pieces of that memfd 6 MiB apart imports as types 5 and 10 (host-visible,
//!   coherent and cached); the GPU read the bytes the CPU wrote through the shared mapping, and a
//!   fill across both pieces reached the file's pages, seen through that mapping after the fence,
//!   with no map, flush or invalidate of the import;
//! - a udmabuf of one piece takes about 9 ms per 64 MiB, the kernel pinning its pages; one of
//!   16,201 scattered 4 KiB pieces (with the limits below raised) 19 ms the first time and 3 ms
//!   after, and RADV imports it in about 50 µs; two over the same pages both import.
//!
//! **Chunks.** udmabuf takes at most `list_limit` pieces (1024 by default) and `size_limit_mb` MiB
//! (64) in one dma-buf, parameters of the `udmabuf` module only root can raise, and the library
//! asks no one to. An import of a macOS 26 guest's boot took up to 16,201 runs of file pages
//! (`docs/qemu.md`, "Linux hosts: AMD GPUs and udmabuf"), so a window is cut into chunks, each a
//! dma-buf within the limits the host has now ([`limits`]), cut only where the runtime can bind
//! the next ([`chunks`]); the runtime binds them in order behind one buffer.
//!
//! **Costs** (measured on the same host, 2026-10-12): a udmabuf pins every page it takes,
//! allocating those never touched (64 MiB of untouched memory in 18 ms), and keeps showing a page
//! the file later drops. Each import the device holds adds about 0.55 µs to every submission
//! (RADV passes each in a list on every submit): 1,024 live imports cost 642 µs a submit, 4,096
//! about 2.5 ms. So the fewest chunks are made, and an import goes with its buffer.
//!
//! What udmabuf takes besides ([`Refusal`]): memfds sealed against shrinking (`F_SEAL_SHRINK`,
//! which QEMU's `seal=on`, its default, sets, and [`super::vm::memfd`] sets on the library's own),
//! at page offsets. The device node is the `kvm` group's on Ubuntu.
//!
//! **Device memory.** A backing IOSurface's memory under `REIMS_VGPU_VRAM_SURFACES` is a dma-buf
//! of device memory already ([`super::vram`]): a window of it from its start is given as one
//! duplicate of that dma-buf, which meets any granule, before anything here looks for
//! `/dev/udmabuf`, and is never kept; a window of it otherwise is refused
//! ([`Refusal::VramWindow`]).
//!
//! Pieces of the block of zeros read-only aliases share ([`super::import::allocate_zeros`]) are
//! made of a file of zeros of the import's own instead, as an import's twin maps private zeros
//! there: the GPU's writes stay in it and never reach the block.
//!
//! **Kept.** A paged texture's transfer buffer is made again at each bind, and the runtime imports
//! it anew, asking for dma-bufs of the same window of the guest's RAM each time (a Finder drag's
//! 1,860 binds covered 180 windows). So the dma-bufs of a window wholly in the guest's RAM, the
//! files the backing table's registrations name, with no zeros and no pad, are kept, by its runs
//! of file pages and its granule ([`KEPT_WINDOWS`], [`KEPT_BYTES`]), and the next call for the
//! same runs and granule is given duplicates of them. A udmabuf keeps the pages it pinned, while
//! the host may drop a page from its file (QEMU punches holes for a balloon, a discard or a
//! reset), and the library is told of none of it. So a kept window is given again only while
//! every file of it shows the change time it showed before its dma-bufs were made, a time older
//! than the coarse clock's tick read before that, so any change after it moves the time: on
//! Linux 7.2 (measured, `a_punched_hole_moves_the_files_change_time`) a hole punched in a memfd
//! moves its change time, as does the first write fault of a page through any mapping of it, which
//! then only makes the window anew. Only files of tmpfs (memfds, QEMU's `memory-backend-memfd`)
//! are kept. What could still pass unseen is a punch at a moment the realtime clock, set back,
//! gives the very nanosecond the file showed. All go once a lease of the guest's RAM goes
//! ([`forget_kept`]), so no file outlives the device that registered it here.

use std::fmt;
use std::io;
use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};
use std::sync::{Arc, OnceLock};

use metal2vulkan_runtime::DmaBufChunk;
use reims_vgpu_core::num::ToU64 as _;
use reims_vgpu_core::sync::Mutex;

use super::files::{self, Backing};
use super::vm;

/// `UDMABUF_FLAGS_CLOEXEC`.
const FLAGS_CLOEXEC: u32 = 0x01;

/// `UDMABUF_CREATE_LIST`, `_IOW('u', 0x43, struct udmabuf_create_list)`: the header's 8 bytes.
const CREATE_LIST: u64 = (1 << 30) | (8 << 16) | ((b'u' as u64) << 8) | 0x43;

/// udmabuf's defaults, taken where the system shows no parameters.
const DEFAULT_LIST_LIMIT: u64 = 1024;
const DEFAULT_SIZE_LIMIT_MB: u64 = 64;

/// Why no dma-bufs were made of a range.
#[derive(Debug)]
pub(crate) enum Refusal {
    /// A byte of the range has no file behind it in the backing table.
    Unresolved,
    /// `/dev/udmabuf` would not open: absent (the module not loaded), or not the process's to
    /// open (Ubuntu gives it to the `kvm` group).
    Device(Arc<io::Error>),
    /// The range cannot be cut into chunks within udmabuf's limits at the runtime's granule: a
    /// granule holds more pieces or bytes than one dma-buf may, or the runtime binds no chunks
    /// (its granule is the whole window) and the window is past the limits.
    Limit {
        pieces: u64,
        bytes: u64,
        granule: u64,
        limits: Limits,
    },
    /// The kernel refused the call otherwise (a file it takes no pages of, unsealed or not a
    /// memfd, or no memory to pin the pages).
    Kernel(io::Error),
    /// The range lies in a backing's device memory ([`super::vram`]) other than as one piece from
    /// its dma-buf's start: `offset` bytes into it, among `pieces` pieces.
    VramWindow { offset: u64, pieces: u64 },
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::Unresolved => write!(f, "a byte of the range has no file behind it"),
            Refusal::Device(e) => write!(
                f,
                "/dev/udmabuf would not open ({e}): load the udmabuf module, and put the user in \
                 the group that owns the node (kvm on Ubuntu)"
            ),
            Refusal::Limit {
                pieces,
                bytes,
                granule,
                limits,
            } => write!(
                f,
                "{pieces} pieces of {bytes:#x} bytes cut at {granule:#x} do not fit udmabuf's \
                 list_limit {} and size_limit_mb {}",
                limits.pieces,
                limits.bytes >> 20,
            ),
            Refusal::Kernel(e) => write!(f, "UDMABUF_CREATE_LIST failed: {e}"),
            Refusal::VramWindow { offset, pieces } => write!(
                f,
                "the range lies {offset:#x} bytes into a dma-buf of VRAM, among {pieces} pieces: \
                 the runtime's chunks carry no offset into a dma-buf, and udmabuf takes no pages \
                 of one"
            ),
        }
    }
}

/// What one dma-buf may hold: pieces, and bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Limits {
    pub(crate) pieces: u64,
    pub(crate) bytes: u64,
}

/// udmabuf's limits as the host has them now, its defaults where it shows none.
fn limits() -> Limits {
    let parameter = |name: &str, default: u64| {
        let path = format!("/sys/module/udmabuf/parameters/{name}");
        (std::fs::read_to_string(path).ok())
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(default)
    };
    Limits {
        pieces: parameter("list_limit", DEFAULT_LIST_LIMIT),
        bytes: parameter("size_limit_mb", DEFAULT_SIZE_LIMIT_MB) << 20,
    }
}

/// `/dev/udmabuf`, opened once for the process.
fn device() -> Result<&'static OwnedFd, Refusal> {
    static DEVICE: OnceLock<Result<OwnedFd, Arc<io::Error>>> = OnceLock::new();
    let opened = DEVICE.get_or_init(|| {
        // SAFETY: a NUL-terminated path.
        let raw = unsafe { libc::open(c"/dev/udmabuf".as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
        if raw < 0 {
            return Err(Arc::new(io::Error::last_os_error()));
        }
        // SAFETY: a descriptor just opened, owned from here on.
        Ok(unsafe { OwnedFd::from_raw_fd(raw) })
    });
    opened.as_ref().map_err(|e| Refusal::Device(Arc::clone(e)))
}

/// A run of file pages: `length` bytes of `file` from `offset`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Run<F> {
    file: F,
    offset: u64,
    length: u64,
}

/// `runs`, the window's pages in order, cut into chunks: each at most `limits.pieces` runs and
/// `limits.bytes` bytes, and each but the last a multiple of `granule` long, so every cut lies a
/// multiple of `granule` from the window's start. A chunk takes as much as fits, so the fewest
/// chunks are made; a run is cut where a chunk ends inside it. `None` where a granule does not fit
/// one chunk.
fn chunks<F: Clone>(runs: &[Run<F>], granule: u64, limits: Limits) -> Option<Vec<Vec<Run<F>>>> {
    let length: u64 = runs.iter().map(|r| r.length).sum();
    let mut made = Vec::new();
    // The next byte to place, as a run and an offset into it, and its offset in the window.
    let (mut run, mut into, mut at) = (0, 0, 0);
    while at < length {
        // The farthest the limits reach from `at`.
        let room = limits.bytes.min(length - at);
        let (mut reach, mut pieces, mut r, mut i) = (0, 0, run, into);
        while reach < room && pieces < limits.pieces {
            let take = (runs[r].length - i).min(room - reach);
            reach += take;
            pieces += 1;
            (r, i) = if i + take == runs[r].length {
                (r + 1, 0)
            } else {
                (r, i + take)
            };
        }
        // Cut there, or at the granule before it unless the window ends there.
        let size = if at + reach == length {
            reach
        } else {
            reach - reach % granule
        };
        if size == 0 {
            return None;
        }
        let mut chunk = Vec::new();
        let mut left = size;
        while left > 0 {
            let take = (runs[run].length - into).min(left);
            chunk.push(Run {
                file: runs[run].file.clone(),
                offset: runs[run].offset + into,
                length: take,
            });
            left -= take;
            (run, into) = if into + take == runs[run].length {
                (run + 1, 0)
            } else {
                (run, into + take)
            };
        }
        made.push(chunk);
        at += size;
    }
    Some(made)
}

/// [`chunks`] of `runs`, padded where they are more than one: then they come to the window's
/// length rounded up to `granule`, as the runtime binds each chunk at a multiple of it, and the
/// last ends with a run `pad` makes of the bytes past the window, a file of the import's own, so a
/// GPU write past the window's end reaches nothing else.
fn padded<F: Clone>(
    runs: &mut Vec<Run<F>>,
    granule: u64,
    limits: Limits,
    pad: impl FnOnce(u64) -> Run<F>,
) -> Option<Vec<Vec<Run<F>>>> {
    let cut = chunks(runs, granule, limits)?;
    let length: u64 = runs.iter().map(|r| r.length).sum();
    let past = length.next_multiple_of(granule) - length;
    if cut.len() == 1 || past == 0 {
        return Some(cut);
    }
    runs.push(pad(past));
    chunks(runs, granule, limits)
}

/// The most windows kept (module docs, "Kept").
const KEPT_WINDOWS: usize = 64;

/// The most bytes the kept windows pin, all together.
const KEPT_BYTES: u64 = 256 << 20;

/// A file's change time, and whether its file system is tmpfs, read now; `None` where `fstat`
/// or `fstatfs` refuses.
fn change_time(file: &OwnedFd) -> Option<((i64, i64), bool)> {
    // SAFETY: plain old data the calls fill.
    let mut stat: libc::stat = unsafe { std::mem::zeroed() };
    // SAFETY: an open descriptor and a stat to fill.
    if unsafe { libc::fstat(file.as_raw_fd(), &mut stat) } != 0 {
        return None;
    }
    // SAFETY: plain old data the call fills.
    let mut fs: libc::statfs = unsafe { std::mem::zeroed() };
    // SAFETY: an open descriptor and a statfs to fill.
    if unsafe { libc::fstatfs(file.as_raw_fd(), &mut fs) } != 0 {
        return None;
    }
    let tmpfs = fs.f_type == libc::TMPFS_MAGIC;
    Some(((stat.st_ctime, stat.st_ctime_nsec), tmpfs))
}

/// The coarse realtime clock now, the clock file times are read from (module docs, "Kept").
fn coarse_now() -> (i64, i64) {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: a timespec to fill.
    unsafe { libc::clock_gettime(libc::CLOCK_REALTIME_COARSE, &mut now) };
    (now.tv_sec, now.tv_nsec)
}

/// A file of a kept window and the change time it showed before its dma-bufs were made.
struct Stamp {
    file: Arc<OwnedFd>,
    changed: (i64, i64),
}

/// The stamps of the files of `runs`, read after the coarse clock: `None` where one is not of
/// tmpfs, or shows a change in the clock's tick or after it, which a later change could repeat.
fn stamps(runs: &[Run<Arc<OwnedFd>>]) -> Option<Vec<Stamp>> {
    let now = coarse_now();
    let mut stamps: Vec<Stamp> = Vec::new();
    for run in runs {
        if stamps.iter().any(|s| Arc::ptr_eq(&s.file, &run.file)) {
            continue;
        }
        let (changed, tmpfs) = change_time(&run.file)?;
        if !tmpfs || changed >= now {
            return None;
        }
        stamps.push(Stamp {
            file: Arc::clone(&run.file),
            changed,
        });
    }
    Some(stamps)
}

/// A kept window: its runs and granule, the stamps of its files, and its dma-bufs.
struct KeptWindow {
    runs: Vec<Run<Arc<OwnedFd>>>,
    granule: u64,
    stamps: Vec<Stamp>,
    chunks: Vec<DmaBufChunk>,
}

impl KeptWindow {
    /// Whether it is the window of `runs` at `granule`.
    fn is(&self, runs: &[Run<Arc<OwnedFd>>], granule: u64) -> bool {
        self.granule == granule
            && self.runs.len() == runs.len()
            && self.runs.iter().zip(runs).all(|(a, b)| {
                Arc::ptr_eq(&a.file, &b.file) && a.offset == b.offset && a.length == b.length
            })
    }

    /// The bytes its dma-bufs pin.
    fn bytes(&self) -> u64 {
        self.chunks.iter().map(|c| c.length).sum()
    }

    /// Whether every file shows the change time it was stamped with.
    fn unchanged(&self) -> bool {
        (self.stamps.iter())
            .all(|s| change_time(&s.file).is_some_and(|(changed, _)| changed == s.changed))
    }

    /// Duplicates of its dma-bufs; `None` where the kernel gives none.
    fn duplicates(&self) -> Option<Vec<DmaBufChunk>> {
        (self.chunks.iter())
            .map(|c| {
                Some(DmaBufChunk {
                    fd: c.fd.try_clone().ok()?,
                    length: c.length,
                })
            })
            .collect()
    }
}

/// The kept windows, most recently used last. A leaf lock: no call is made under it.
static KEPT: Mutex<Vec<KeptWindow>> = Mutex::new(Vec::new());

/// The kept window of `runs` at `granule`, taken out; the caller puts it back.
fn take_kept(runs: &[Run<Arc<OwnedFd>>], granule: u64) -> Option<KeptWindow> {
    let mut kept = KEPT.lock();
    let at = kept.iter().position(|w| w.is(runs, granule))?;
    Some(kept.remove(at))
}

/// Keeps `window` as the most recently used, letting the least recently used go past the limits,
/// outside the lock.
fn keep(window: KeptWindow) {
    let gone = {
        let mut kept = KEPT.lock();
        kept.push(window);
        let mut bytes: u64 = kept.iter().map(KeptWindow::bytes).sum();
        let mut gone = Vec::new();
        while kept.len() > KEPT_WINDOWS || bytes > KEPT_BYTES {
            let oldest = kept.remove(0);
            bytes -= oldest.bytes();
            gone.push(oldest);
        }
        gone
    };
    drop(gone);
}

/// Lets every kept window go: a lease of the guest's RAM goes ([`super::files`]).
pub(crate) fn forget_kept() {
    let gone = std::mem::take(&mut *KEPT.lock());
    drop(gone);
}

/// The windows kept now, for tests.
#[cfg(test)]
fn kept_windows() -> usize {
    KEPT.lock().len()
}

/// Dma-bufs of the `length` bytes from `first`, whole pages, each resolved through the backing
/// table, which in order are those bytes: one piece for each run of a file's pages that follow on,
/// the block of zeros' pieces from a file of zeros of the import's own, cut into chunks within
/// udmabuf's limits at multiples of `granule` from `first` (module docs). Those of a window of
/// the guest's RAM are kept for the next call over the same pages, and given again while its
/// files show no change (module docs, "Kept"). A window of a backing's device memory is a
/// duplicate of its dma-buf (module docs, "Device memory").
pub(crate) fn of_pages(first: u64, length: u64, granule: u64) -> Result<Vec<DmaBufChunk>, Refusal> {
    let page = vm::page_size();
    debug_assert!(first.is_multiple_of(page) && length.is_multiple_of(page));
    let resolved = files::pieces(first, length).ok_or(Refusal::Unresolved)?;
    if let Some(answer) = super::vram::chunks_of(&resolved, length) {
        // vdma: the window's length, the granule, whether its dma-buf was given
        reims_vgpu_core::trace::event("vdma", &[length, granule, u64::from(answer.is_ok())]);
        return answer;
    }
    let zero_bytes: u64 = (resolved.iter())
        .filter(|p| super::import::zeros(&p.backing.fd))
        .map(|p| p.length)
        .sum();
    let zeros = (zero_bytes > 0).then(|| vm::memfd(zero_bytes));
    let mut zeros_at = 0;
    let mut runs: Vec<Run<Arc<OwnedFd>>> = Vec::new();
    for piece in resolved {
        let backing = match &zeros {
            Some(zeros) if super::import::zeros(&piece.backing.fd) => {
                let backing = Backing {
                    fd: Arc::clone(zeros),
                    offset: zeros_at,
                };
                zeros_at += piece.length;
                backing
            }
            _ => piece.backing,
        };
        match runs.last_mut() {
            Some(last)
                if Arc::ptr_eq(&last.file, &backing.fd)
                    && last.offset + last.length == backing.offset =>
            {
                last.length += piece.length;
            }
            _ => runs.push(Run {
                file: backing.fd,
                offset: backing.offset,
                length: piece.length,
            }),
        }
    }
    let device = device()?;
    let limits = limits();
    let granule = granule.max(page);
    let keepable = zeros.is_none() && runs.iter().all(|r| files::guest_ram(&r.file));
    if keepable && let Some(window) = take_kept(&runs, granule) {
        let given = window.unchanged().then(|| window.duplicates()).flatten();
        // kdma: whether a kept window's dma-bufs were given again, its chunks
        reims_vgpu_core::trace::event(
            "kdma",
            &[u64::from(given.is_some()), window.chunks.len().to_u64()],
        );
        if let Some(chunks) = given {
            keep(window);
            return Ok(chunks);
        }
        drop(window);
    }
    let stamped = keepable.then(|| stamps(&runs)).flatten();
    let window_runs = stamped.as_ref().map(|_| runs.clone());
    let refused = |runs: &[Run<Arc<OwnedFd>>]| Refusal::Limit {
        pieces: runs.len() as u64,
        bytes: length,
        granule,
        limits,
    };
    let pad = |pad| Run {
        file: vm::memfd(pad),
        offset: 0,
        length: pad,
    };
    let unpadded = runs.len();
    let cut = padded(&mut runs, granule, limits, pad).ok_or_else(|| refused(&runs))?;
    let made = cut.iter().map(|chunk| {
        let fd = create(device, chunk)?;
        Ok(DmaBufChunk {
            fd,
            length: chunk.iter().map(|r| r.length).sum(),
        })
    });
    let chunks: Vec<DmaBufChunk> = made.collect::<Result<_, Refusal>>()?;
    // A window with a pad of its own is not kept: the pad is the import's alone.
    if runs.len() == unpadded
        && let (Some(stamps), Some(window_runs)) = (stamped, window_runs)
    {
        let mut window = KeptWindow {
            runs: window_runs,
            granule,
            stamps,
            chunks,
        };
        if let Some(given) = window.duplicates() {
            keep(window);
            return Ok(given);
        }
        return Ok(std::mem::take(&mut window.chunks));
    }
    Ok(chunks)
}

/// One dma-buf of `runs`, in order (`UDMABUF_CREATE_LIST`).
fn create(device: &OwnedFd, runs: &[Run<Arc<OwnedFd>>]) -> Result<OwnedFd, Refusal> {
    // `struct udmabuf_create_list` of Linux's `include/uapi/linux/udmabuf.h` as little-endian
    // words: `flags` and `count`, then each `struct udmabuf_create_item`'s `memfd` and its
    // padding, `offset` and `size`.
    let count = u32::try_from(runs.len()).unwrap_or(u32::MAX);
    let mut call = vec![u64::from(FLAGS_CLOEXEC) | (u64::from(count) << 32)];
    for run in runs {
        let memfd = u32::try_from(run.file.as_raw_fd()).unwrap_or(u32::MAX);
        call.extend([u64::from(memfd), run.offset, run.length]);
    }
    // SAFETY: the list describes open memfds the backing table and the import's zeros hold
    // through the call; the kernel reads the header and its items.
    let made = unsafe { libc::ioctl(device.as_raw_fd(), CREATE_LIST as _, call.as_ptr()) };
    if made < 0 {
        return Err(Refusal::Kernel(io::Error::last_os_error()));
    }
    // SAFETY: the dma-buf the call just made, owned from here on.
    Ok(unsafe { OwnedFd::from_raw_fd(made) })
}

#[cfg(test)]
mod tests {
    use reims_vgpu_core::num::ToUsize as _;
    use reims_vgpu_core::platform::os::vm::{HostAddr, HostVm as _};

    use super::*;

    /// Whether this host has a udmabuf device this process opens: where it has none, the tests
    /// say so and check nothing.
    fn udmabuf() -> bool {
        match device() {
            Ok(_) => true,
            Err(refusal) => {
                eprintln!("skipped: {refusal}");
                false
            }
        }
    }

    /// The dma-buf's byte at `offset`, read through a mapping of it.
    fn read_through(chunk: &DmaBufChunk, offset: u64) -> u8 {
        let length = usize::try_from(chunk.length).unwrap();
        // SAFETY: a fresh read-only mapping of the dma-buf, read once and unmapped.
        unsafe {
            let at = libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ,
                libc::MAP_SHARED,
                chunk.fd.as_raw_fd(),
                0,
            );
            assert_ne!(at, libc::MAP_FAILED);
            let byte = (at.cast::<u8>())
                .add(usize::try_from(offset).unwrap())
                .read_volatile();
            libc::munmap(at, length);
            byte
        }
    }

    fn run(file: u32, offset: u64, length: u64) -> Run<u32> {
        Run {
            file,
            offset,
            length,
        }
    }

    /// Runs within the limits make one chunk; past them, chunks as full as the limits let, cut
    /// at granules from the window's start, a run cut where a chunk ends in it; a granule past
    /// the limits makes none
    #[test]
    fn a_window_is_cut_into_chunks_within_the_limits_at_granules() {
        let limits = Limits {
            pieces: 3,
            bytes: 0x10000,
        };
        let one = [run(1, 0, 0x1000), run(2, 0x5000, 0x2000)];
        assert_eq!(
            chunks(&one, 0x1000, limits),
            Some(vec![vec![run(1, 0, 0x1000), run(2, 0x5000, 0x2000)]])
        );
        // Five one-page runs, three a chunk.
        let scattered: Vec<_> = (0..5u32)
            .map(|i| run(i, u64::from(i) * 0x3000, 0x1000))
            .collect();
        assert_eq!(
            chunks(&scattered, 0x1000, limits),
            Some(vec![
                vec![
                    run(0, 0, 0x1000),
                    run(1, 0x3000, 0x1000),
                    run(2, 0x6000, 0x1000)
                ],
                vec![run(3, 0x9000, 0x1000), run(4, 0xc000, 0x1000)],
            ])
        );
        // At a granule of two pages the first chunk ends at two pages; the last ends with the
        // window, at no granule.
        assert_eq!(
            chunks(&scattered, 0x2000, limits),
            Some(vec![
                vec![run(0, 0, 0x1000), run(1, 0x3000, 0x1000)],
                vec![
                    run(2, 0x6000, 0x1000),
                    run(3, 0x9000, 0x1000),
                    run(4, 0xc000, 0x1000)
                ],
            ])
        );
        // A run longer than a chunk's bytes is cut where each chunk ends.
        let long = [run(7, 0x1000, 0x25000)];
        assert_eq!(
            chunks(&long, 0x1000, limits),
            Some(vec![
                vec![run(7, 0x1000, 0x10000)],
                vec![run(7, 0x11000, 0x10000)],
                vec![run(7, 0x21000, 0x5000)],
            ])
        );
        // A run cut at a granule inside it: three runs reach 0x5000, the granule 0x2000 cuts
        // the third.
        let mixed = [
            run(1, 0, 0x1000),
            run(2, 0, 0x2000),
            run(3, 0, 0x2000),
            run(4, 0, 0x1000),
        ];
        assert_eq!(
            chunks(&mixed, 0x2000, limits),
            Some(vec![
                vec![run(1, 0, 0x1000), run(2, 0, 0x2000), run(3, 0, 0x1000)],
                vec![run(3, 0x1000, 0x1000), run(4, 0, 0x1000)],
            ])
        );
        // A granule of four single pages past three pieces fits no chunk; the whole window as
        // its granule fits only within the limits.
        assert_eq!(chunks(&scattered, 0x4000, limits), None);
        assert_eq!(chunks(&scattered, 0x5000, limits), None);
        assert!(chunks(&one, 0x3000, limits).is_some());
        // The most runs one import of a macOS 26 guest's boot took, at udmabuf's defaults.
        let boot: Vec<_> = (0..16_201u32)
            .map(|i| run(i, u64::from(i) * 0x2000, 0x1000))
            .collect();
        let defaults = Limits {
            pieces: DEFAULT_LIST_LIMIT,
            bytes: DEFAULT_SIZE_LIMIT_MB << 20,
        };
        let cut = chunks(&boot, 0x1000, defaults).unwrap();
        assert_eq!(cut.len(), 16);
        assert!(cut.iter().all(|c| c.len() as u64 <= DEFAULT_LIST_LIMIT));
        assert_eq!(cut.iter().map(Vec::len).sum::<usize>(), 16_201);
    }

    /// Chunks past one come to the window's length rounded up to the granule, the last ending
    /// with the pad's run; one chunk, or a window of whole granules, takes no pad
    #[test]
    fn several_chunks_are_padded_to_the_granule() {
        let limits = Limits {
            pieces: 2,
            bytes: 0x10000,
        };
        let pad = |length| run(99, 0, length);
        let mut one = vec![run(1, 0, 0x1000)];
        assert_eq!(
            padded(&mut one, 0x4000, limits, pad),
            Some(vec![vec![run(1, 0, 0x1000)]])
        );
        let mut whole: Vec<_> = (0..4u32).map(|i| run(i, 0, 0x2000)).collect();
        assert_eq!(padded(&mut whole, 0x4000, limits, pad).unwrap().len(), 2);
        assert_eq!(whole.len(), 4, "no pad");
        // Three single pages at a granule of two: two chunks, the last padded by a page.
        let mut three: Vec<_> = (0..3u32).map(|i| run(i, 0, 0x1000)).collect();
        assert_eq!(
            padded(&mut three, 0x2000, limits, pad),
            Some(vec![
                vec![run(0, 0, 0x1000), run(1, 0, 0x1000)],
                vec![run(2, 0, 0x1000), run(99, 0, 0x1000)],
            ])
        );
        // Five pages at two a chunk: the pad joins the last page's chunk.
        let mut five: Vec<_> = (0..5u32).map(|i| run(i, 0, 0x1000)).collect();
        let cut = padded(&mut five, 0x2000, limits, pad).unwrap();
        assert_eq!(cut.iter().flatten().map(|r| r.length).sum::<u64>(), 0x6000);
        assert_eq!(cut.last().unwrap().last().unwrap().file, 99);
    }

    /// The dma-bufs of a view show the pages the view shows, in the view's order, and a range
    /// with no file behind it is refused
    #[test]
    fn a_dma_buf_shows_the_pages_of_its_range() {
        if !udmabuf() {
            return;
        }
        let page = vm::page_size();
        let at = vm::allocate_aliasable(4 * page).unwrap().addr().to_u64();
        for i in 0..4 {
            // SAFETY: the range this test just allocated, read and write.
            unsafe {
                vm::ptr(at + i * page)
                    .cast::<u8>()
                    .write_volatile(0x10 + i as u8)
            };
        }
        let chunks = of_pages(at + page, 2 * page, page).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].length, 2 * page);
        assert_eq!(read_through(&chunks[0], 0), 0x11);
        assert_eq!(read_through(&chunks[0], page), 0x12);
        // SAFETY: as above.
        unsafe { vm::ptr(at + page).cast::<u8>().write_volatile(0x5a) };
        assert_eq!(
            read_through(&chunks[0], 0),
            0x5a,
            "the same pages, not a copy"
        );
        let private = vm::anonymous(None, page, "a test's private page").unwrap();
        assert!(matches!(
            of_pages(private, page, page),
            Err(Refusal::Unresolved)
        ));
    }

    /// A view of more scattered pages than one dma-buf may hold is made of several, in order,
    /// each within the host's limits
    #[test]
    fn a_view_past_the_limits_is_made_of_chunks() {
        if !udmabuf() {
            return;
        }
        let page = vm::page_size();
        let limits = limits();
        // Every other page of a range, so no two pieces follow on in the file.
        let count = limits.pieces + 2;
        let source = vm::allocate_aliasable(2 * count * page).unwrap();
        let view = vm::allocate_aliasable(count * page).unwrap();
        let host = super::super::LinuxVm::new();
        for i in 0..count {
            let from = source.addr().to_u64() + 2 * i * page;
            // SAFETY: the source range this test allocated, read and write.
            unsafe { vm::ptr(from).cast::<u8>().write_volatile((i % 251) as u8) };
            let to = view.addr().to_u64() + i * page;
            let (to, from) = (
                HostAddr::from_addr(to.to_usize()),
                HostAddr::from_addr(from.to_usize()),
            );
            assert!(host.remap_shared_overwrite(to, from, page).is_ok());
        }
        let chunks = of_pages(view.addr().to_u64(), count * page, page).unwrap();
        assert!(chunks.len() >= 2);
        assert_eq!(chunks.iter().map(|c| c.length).sum::<u64>(), count * page);
        let mut at = 0;
        for chunk in &chunks {
            assert!(chunk.length <= limits.bytes);
            assert!(chunk.length / page <= limits.pieces);
            for offset in (0..chunk.length).step_by(usize::try_from(page).unwrap()) {
                let i = at + offset / page;
                assert_eq!(read_through(chunk, offset), (i % 251) as u8);
            }
            at += chunk.length / page;
        }
    }

    /// A window of a backing's device memory from its start is one duplicate of its dma-buf,
    /// whatever the granule, showing the CPU's writes, with no udmabuf asked for: so where
    /// `/dev/udmabuf` does not open too
    #[test]
    fn a_window_at_the_start_of_vram_memory_is_its_dma_buf() {
        let page = vm::page_size();
        let exporter = super::super::vram::tests::test_exporter();
        let own = vm::OwnMemory::in_vram(2 * page, exporter).unwrap();
        let first = own.first();
        // SAFETY: the memory just mapped, read and write.
        unsafe { vm::ptr(first + page + 5).cast::<u8>().write_volatile(0x5c) };
        let duplicates = super::super::vram::live().duplicates;
        let whole = of_pages(first, 2 * page, page).unwrap();
        assert_eq!((whole.len(), whole[0].length), (1, 2 * page));
        assert_eq!(read_through(&whole[0], page + 5), 0x5c);
        let start = of_pages(first, page, 4 * page).unwrap();
        assert_eq!((start.len(), start[0].length), (1, page));
        assert_eq!(super::super::vram::live().duplicates, duplicates + 2);
    }

    /// A window inside a backing's device memory is refused, naming its offset, and no udmabuf
    /// is asked for
    #[test]
    fn a_window_inside_vram_memory_is_refused() {
        let page = vm::page_size();
        let exporter = super::super::vram::tests::test_exporter();
        let own = vm::OwnMemory::in_vram(2 * page, exporter).unwrap();
        let refused = of_pages(own.first() + page, page, page).unwrap_err();
        assert!(
            matches!(refused, Refusal::VramWindow { offset, pieces: 1 } if offset == page),
            "{refused}"
        );
    }

    /// Held by each test of kept windows: the windows are the process's.
    static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

    /// A memfd of `pages` pages, sealed as QEMU's, mapped shared and registered as the guest's
    /// RAM at its mapping, and its change time left a few ticks behind the coarse clock.
    fn guest_ram(pages: u64) -> (u64, Arc<OwnedFd>, super::super::files::GuestMemoryLease) {
        let page = vm::page_size();
        let fd = vm::memfd(pages * page);
        let backing = Backing {
            fd: Arc::clone(&fd),
            offset: 0,
        };
        let at = vm::file(
            None,
            pages * page,
            &backing,
            libc::PROT_READ | libc::PROT_WRITE,
        )
        .unwrap();
        for i in 0..pages {
            // SAFETY: the mapping this test just made, read and write.
            unsafe {
                vm::ptr(at + i * page)
                    .cast::<u8>()
                    .write_volatile(0x20 + i as u8)
            };
        }
        let lease = super::super::files::lease_guest_memory([crate::os::GuestRamFile {
            host: at,
            length: pages * page,
            fd: Arc::clone(&fd),
            offset: 0,
        }]);
        std::thread::sleep(std::time::Duration::from_millis(30));
        (at, fd, lease)
    }

    /// The dma-buf's inode: duplicates share it.
    fn inode(chunk: &DmaBufChunk) -> u64 {
        // SAFETY: plain old data the call fills.
        let mut stat: libc::stat = unsafe { std::mem::zeroed() };
        // SAFETY: an open descriptor and a stat to fill.
        assert_eq!(unsafe { libc::fstat(chunk.fd.as_raw_fd(), &mut stat) }, 0);
        stat.st_ino
    }

    /// Measured on Linux 7.2: a hole punched in a memfd moves its change time, read or not since
    /// the last change, as does the first write fault of a page through a mapping; a read of a
    /// page and a write of one written through the same mapping move nothing. Kept windows are
    /// given again only while their files' times stand (module docs, "Kept")
    #[test]
    fn a_punched_hole_moves_the_files_change_time() {
        let page = vm::page_size();
        let (at, fd, lease) = guest_ram(4);
        let changed = || change_time(&fd).unwrap();
        let (before, tmpfs) = changed();
        assert!(tmpfs, "a memfd is of tmpfs");
        assert!(before < coarse_now());
        // SAFETY: the test's mapping, read and write.
        unsafe {
            let _ = vm::ptr(at).cast::<u8>().read_volatile();
            vm::ptr(at + 1).cast::<u8>().write_volatile(1);
        }
        assert_eq!(changed(), (before, true));
        // SAFETY: a hole punched in the test's own file.
        let punched = unsafe {
            libc::fallocate(
                fd.as_raw_fd(),
                libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
                0,
                page as libc::off_t,
            )
        };
        assert_eq!(punched, 0);
        let after = changed();
        assert_ne!(after.0, before);
        // At once again, in the same tick
        // SAFETY: as above.
        let punched = unsafe {
            libc::fallocate(
                fd.as_raw_fd(),
                libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
                page as libc::off_t,
                page as libc::off_t,
            )
        };
        assert_eq!(punched, 0);
        assert_ne!(changed(), after);
        drop(lease);
        // SAFETY: the test's mapping, which nothing uses any more.
        unsafe { libc::munmap(vm::ptr(at), (4 * page).to_usize()) };
    }

    /// A window of the guest's RAM is kept and given again, its dma-bufs duplicates of the first,
    /// until its file changes: then made anew, of the pages the file shows now; a window of other
    /// memory is not kept; the lease's drop lets the kept go
    #[test]
    fn a_kept_window_is_given_again_until_its_file_changes() {
        if !udmabuf() {
            return;
        }
        let _one = ONE_AT_A_TIME.lock();
        forget_kept();
        let page = vm::page_size();
        let (at, fd, lease) = guest_ram(4);
        let first = of_pages(at + page, 2 * page, page).unwrap();
        assert_eq!(kept_windows(), 1);
        let again = of_pages(at + page, 2 * page, page).unwrap();
        assert_eq!(inode(&again[0]), inode(&first[0]), "the same dma-buf");
        assert_eq!(read_through(&again[0], 0), 0x21);
        // Another granule is another window
        let other = of_pages(at + page, 2 * page, 2 * page).unwrap();
        assert_ne!(inode(&other[0]), inode(&first[0]));
        assert_eq!(kept_windows(), 2);
        // The host drops the second page from the file: the kept window is not given again
        // SAFETY: a hole punched in the test's own file.
        let punched = unsafe {
            libc::fallocate(
                fd.as_raw_fd(),
                libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
                page as libc::off_t,
                page as libc::off_t,
            )
        };
        assert_eq!(punched, 0);
        assert_eq!(
            read_through(&first[0], 0),
            0x21,
            "a udmabuf keeps the page it pinned"
        );
        let fresh = of_pages(at + page, 2 * page, page).unwrap();
        assert_ne!(inode(&fresh[0]), inode(&first[0]));
        assert_eq!(read_through(&fresh[0], 0), 0, "the file's page now");
        // The changed window went; the fresh one is kept unless made in the change's tick
        let kept = kept_windows();
        assert!((1..=2).contains(&kept), "{kept}");
        // Memory of the library's own is never kept
        let own = vm::allocate_aliasable(2 * page).unwrap().addr().to_u64();
        drop(of_pages(own, 2 * page, page).unwrap());
        assert_eq!(kept_windows(), kept);
        drop(lease);
        assert_eq!(kept_windows(), 0);
        // SAFETY: the test's mapping, which nothing uses any more.
        unsafe { libc::munmap(vm::ptr(at), (4 * page).to_usize()) };
    }
}
