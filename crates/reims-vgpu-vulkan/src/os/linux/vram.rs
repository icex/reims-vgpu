//! IOSurface backings in device memory, on request: with `REIMS_VGPU_VRAM_SURFACES=1`, the memory
//! of a backing IOSurface core makes (`IoSurfaces::create_backing`) of one plane at its base is a
//! dma-buf of device-local, host-visible, coherent memory (resizable BAR), in place of a memfd
//! ([`super::vm::OwnMemory::in_vram`]). Unset, or any value but `1`, nothing here runs.
//!
//! **Why.** A memfd's pages are host RAM, which the GPU reads over PCIe at every use. Measured on
//! an RX 9070 XT (RADV of Mesa 26.2.4, Linux 7.2.10, 2026-10-22), with no library between, over a
//! 3840 × 2160 BGRA8 surface the runtime made a texture of: a full-screen sample of every texel
//! took 640 µs from host RAM and 130 to 170 µs from this memory; a copy of the whole surface into
//! an optimal texture 628 µs and 49 to 63 µs; the first opaque draw into a quarter of it 175 to
//! 194 µs and 73 µs.
//!
//! **Engagement** ([`exporter`]), decided once per process, at the first backing made with the
//! switch on, with no GPU work and no wait, so nothing on the thread making the backing waits
//! behind the guest's queued work. Each step that fails leaves every backing in a memfd, with one
//! warning ([`Disengaged`]):
//! - the runtime lists exactly one non-CPU device (`MTLCopyAllDevices`), the system default,
//!   so no import of the memory is asked of another GPU's driver. CPU implementations do not
//!   count as another GPU;
//! - it loads `vkGetMemoryFdKHR`, which the runtime enables with `VK_EXT_external_memory_dma_buf`;
//! - a transfer buffer may export a dma-buf (`vkGetPhysicalDeviceExternalBufferProperties`), and
//!   not only from a dedicated allocation;
//! - a memory type is device-local, host-visible and host-coherent, and none of AMD's
//!   device-coherent or uncached, protected or lazily allocated (as the runtime's own allocations
//!   avoid), among those such a buffer takes. A quarter of its heap is the backings' budget,
//!   counted in the dma-bufs' sizes as the kernel gives them;
//! - a self-test over 64 KiB of such memory, registered as a backing's is ([`self_test`]): a
//!   buffer over its pointer alone is refused (`Unsupported`), and one with a dma-buf source is
//!   made, the source called once.
//!
//! **A backing** ([`export`]): an allocation of that type exporting a dma-buf, its descriptor
//! taken, and the allocation freed at once, so the dma-buf is the only reference to the memory
//! from then on. The memory is the dma-buf's mapping, read and write and shared, registered in the
//! backing table as its file, so views, `files::pieces` and a failed remap's put-back work as over
//! a memfd. Each import of it is a duplicate of the dma-buf ([`chunks_of`], from
//! `super::dma_buf::of_pages`), never a udmabuf, so no `/dev/udmabuf` is needed. The kernel frees
//! the memory once the mapping, the descriptor and the device's last import of it go.
//!
//! **Measured** on the same host (gate 2 and the driver probe, with no library between):
//! - memory type 3 (heap 1, flags 0x7) exports a dma-buf with no buffer bound, and its `lseek`
//!   size is the 33,177,600 bytes asked; it outlives `vkFreeMemory` through its descriptor;
//! - a transfer buffer's dma-buf export features are 0x6, exportable and importable, not
//!   dedicated-only; `vkGetMemoryFdPropertiesKHR` offers type 3 alone for it;
//! - its mapping (`mmap`, `MAP_SHARED`) reads zeros before any write, as a fresh memfd does;
//! - `vkGetMemoryHostPointerPropertiesEXT` answers that mapping `VK_SUCCESS` with type bits 0x20,
//!   but `vkAllocateMemory` with `VkImportMemoryHostPointerInfoEXT` over it is refused with
//!   `VK_ERROR_INVALID_EXTERNAL_HANDLE`, which the runtime takes as a refusal of the pointer and
//!   asks the dma-buf source then. Read from code (Linux 7.3-rc3's `amdgpu_ttm.c`), matching it:
//!   Mesa's userptr asks for anonymous pages only, and the mapping has a file;
//! - an import of fewer bytes than the dma-buf holds works; a buffer import and a
//!   `DRM_FORMAT_MOD_LINEAR` image import (3840 × 2160 B8G8R8A8 at a pitch of 15,360) of two
//!   duplicates of one dma-buf work in one submission, and the CPU's writes through the mapping
//!   and the GPU's through the imports are each seen by the other;
//! - with one such dma-buf live, `fdinfo` showed `drm-shared-vram` 32,400 KiB and
//!   `amd-evicted-vram` 0.
//!
//! **Rules.**
//! - Never `DMA_BUF_IOCTL_SYNC`: amdgpu's `begin_cpu_access` moves an unpinned buffer object to
//!   GTT (read from code, `amdgpu_dma_buf.c`), which would undo the placement.
//! - The CPU's reads through the mapping are uncached reads across the BAR: the runtime measured a
//!   1080p `get_bytes` from RADV's uncached device-local memory at 108 ms on a Radeon 780M. No path
//!   of a backing reads it by CPU: core pages a backing in and out by blits, its test pattern
//!   writes only surfaces of the IOSurface table, and a texture over the memory is an image over a
//!   duplicate of the dma-buf, which the runtime acquires from the foreign queue family with no
//!   read of its bytes. A guest's `getBytes` of a texture over a backing would read it slowly.
//! - A GPU reset that loses device memory loses the backings' bytes, which a memfd's keep: the
//!   switch stays off for the suite and for runs that fault the GPU.
//!
//! **What the guest and host can see**, with the switch on only: device memory the backings take
//! (at most the budget) can make a later allocation refused for lack of it one call sooner, a nil
//! or a death at an earlier op; and a buffer over such memory that is not the dma-buf's start,
//! which no path makes today ([`super::dma_buf::Refusal::VramWindow`]), dies under `Policy::Dies`
//! where a memfd's would have been imported. A backing the switch cannot place
//! ([`Refused`]) is a memfd's, logged.
//!
//! **Interim.** The allocation, the export and the free are raw Vulkan calls on the runtime's
//! device, on the thread making the backing, outside the runtime's release door: their race with
//! the process's exit is the program's, as the runtime's exit guard has the program's own calls,
//! checked once before them ([`rt::process_exiting`]). An export of device memory by the runtime
//! itself is to replace them.
//!
//! The log lines' target is this module's. Engaged, one info line says so; each backing placed,
//! one info line, with its base, so a run untraced shows which went to device memory; each
//! refusal's kind, one warning, then debug lines. With the trace on, `vsrf` marks each backing
//! made with the switch engaged, and `vdma` each window `of_pages` answered here.

use std::fmt;
use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use ash::{khr, vk};
use metal2vulkan_runtime as rt;
use metal2vulkan_runtime::DmaBufChunk;
use reims_vgpu_core::num::ToU64 as _;
use reims_vgpu_core::platform::os::vm::KernReturn;
use reims_vgpu_core::sync::Mutex;

use super::dma_buf::{self, Refusal};
use super::files;
use super::vm::{self, OwnMemory};

/// The switch (module docs): `1` turns it on.
const ENV_VAR: &str = "REIMS_VGPU_VRAM_SURFACES";

/// The bytes of the self-test's memory ([`self_test`]).
const SELF_TEST_BYTES: u64 = 64 << 10;

/// The external handle every export here is.
const DMA_BUF: vk::ExternalMemoryHandleTypeFlags = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;

/// Whether the switch's value turns it on: `1` alone, as the library's other `1` switches.
fn switched_on(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|v| v == "1")
}

/// Why the switch places no backing in device memory for the process (module docs,
/// "Engagement").
#[derive(Debug)]
pub(crate) enum Disengaged {
    /// The runtime made no device.
    NoDevice(String),
    /// The runtime lists other than one non-CPU device, or its one is not the system default.
    SeveralDevices(usize),
    /// The device has no `vkGetMemoryFdKHR`.
    NoExportFunctions,
    /// A transfer buffer exports no dma-buf, or only from a dedicated allocation: its features.
    NotExportable(u32),
    /// No memory type is device-local, host-visible and coherent, of those such a buffer takes.
    NoMemoryType,
    /// The self-test's memory was refused.
    SelfTestMemory(Refused),
    /// The device imported the mapping's pointer as host memory.
    PointerImported,
    /// The device failed the pointer's import otherwise than refusing it.
    PointerFailed(String),
    /// The device refused a buffer over the memory with a dma-buf source.
    DmaBufRefused(String),
    /// The source of that buffer was called other than once.
    SourceNotCalled(u32),
}

impl fmt::Display for Disengaged {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Disengaged::NoDevice(e) => write!(f, "the runtime made no device: {e}"),
            Disengaged::SeveralDevices(n) => write!(
                f,
                "the runtime lists {n} non-CPU devices, where the memory is made on its one device"
            ),
            Disengaged::NoExportFunctions => write!(f, "the device has no vkGetMemoryFdKHR"),
            Disengaged::NotExportable(features) => write!(
                f,
                "a transfer buffer's dma-buf export features are {features:#x}: not exportable, \
                 or dedicated-only"
            ),
            Disengaged::NoMemoryType => write!(
                f,
                "no memory type is device-local, host-visible and coherent for an exportable \
                 buffer"
            ),
            Disengaged::SelfTestMemory(r) => write!(f, "the self-test's memory: {r}"),
            Disengaged::PointerImported => write!(
                f,
                "the device imported the mapping of device memory as host memory, where its \
                 imports must be dma-bufs"
            ),
            Disengaged::PointerFailed(e) => write!(
                f,
                "the device failed the import of the mapping's pointer without refusing it: {e}"
            ),
            Disengaged::DmaBufRefused(e) => write!(
                f,
                "the device refused a buffer over the memory with its dma-buf: {e}"
            ),
            Disengaged::SourceNotCalled(calls) => write!(
                f,
                "a buffer over the memory called its dma-buf source {calls} times, not once"
            ),
        }
    }
}

/// Why one backing is not in device memory: it is a memfd's (module docs).
#[derive(Debug)]
pub(crate) enum Refused {
    /// The backing has planes other than one at its base.
    Planes,
    /// Its dma-buf would take the backings' bytes past the budget.
    Budget { size: u64, budget: u64 },
    /// `vkAllocateMemory` failed.
    Allocation(vk::Result),
    /// `vkGetMemoryFdKHR` failed.
    Export(vk::Result),
    /// The dma-buf is shorter than the memory asked for, or its size does not read.
    ShortFile { size: i64, length: u64 },
    /// The kernel mapped none of it.
    Map(KernReturn),
    /// The process is exiting.
    Exiting,
}

impl Refused {
    /// The kind, numbered for the trace's `vsrf` and the warning given once per kind.
    fn code(&self) -> usize {
        match self {
            Refused::Planes => 1,
            Refused::Budget { .. } => 2,
            Refused::Allocation(_) => 3,
            Refused::Export(_) => 4,
            Refused::ShortFile { .. } => 5,
            Refused::Map(_) => 6,
            Refused::Exiting => 7,
        }
    }
}

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refused::Planes => write!(f, "not one plane at its base"),
            Refused::Budget { size, budget } => write!(
                f,
                "{size:#x} bytes more would pass the budget of {budget:#x} bytes, a quarter of \
                 the heap"
            ),
            Refused::Allocation(e) => write!(f, "vkAllocateMemory failed: {e}"),
            Refused::Export(e) => write!(f, "vkGetMemoryFdKHR failed: {e}"),
            Refused::ShortFile { size, length } => {
                write!(f, "a dma-buf of {size} bytes for {length:#x}")
            }
            Refused::Map(kr) => write!(f, "the kernel mapped none of it ({kr:?})"),
            Refused::Exiting => write!(f, "the process is exiting"),
        }
    }
}

/// What makes the backings' device memory, once engaged (module docs).
pub(crate) struct Exporter {
    /// The runtime's device, kept so its handles stay live.
    device: rt::Device,
    /// Its commands, as the runtime loaded them.
    commands: ash::Device,
    /// `vkGetMemoryFdKHR`.
    fd: khr::external_memory_fd::DeviceFn,
    /// The memory type, its flags and its heap's size.
    memory_type: u32,
    flags: vk::MemoryPropertyFlags,
    heap: u64,
    /// The most bytes the backings' dma-bufs take together.
    budget: u64,
    /// The bytes they take now, given back by each [`Listed`]'s drop.
    reserved: Arc<AtomicU64>,
}

/// The process's exporter, decided at the first call ([`exporter`]).
static EXPORTER: OnceLock<Option<Exporter>> = OnceLock::new();

#[cfg(test)]
thread_local! {
    /// A test's exporter, in place of the process's, on the thread that sets it.
    static TEST_EXPORTER: std::cell::Cell<Option<&'static Exporter>> =
        const { std::cell::Cell::new(None) };
}

/// Runs `call` with `exporter` as this thread's ([`exporter`]), the thread's own put back after.
#[cfg(test)]
pub(crate) fn with_exporter<T>(exporter: &'static Exporter, call: impl FnOnce() -> T) -> T {
    /// Puts back the thread's exporter on drop, an unwinding call's included.
    struct Restore(Option<&'static Exporter>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_EXPORTER.with(|e| e.set(self.0));
        }
    }
    let _restore = Restore(TEST_EXPORTER.with(|e| e.replace(Some(exporter))));
    call()
}

/// What makes the backings' device memory: `None` with the switch off, or where it disengaged
/// (module docs, "Engagement"), decided at the first call and logged then.
pub(crate) fn exporter() -> Option<&'static Exporter> {
    #[cfg(test)]
    if let Some(exporter) = TEST_EXPORTER.with(std::cell::Cell::get) {
        return Some(exporter);
    }
    let engaged = EXPORTER.get_or_init(|| {
        if !switched_on(std::env::var_os(ENV_VAR).as_deref()) {
            return None;
        }
        match the_device().and_then(|device| engage(&device)) {
            Ok(exporter) => {
                log::info!(
                    "IOSurface backings of one plane go to VRAM: memory type {} (flags {:#x}) of \
                     a heap of {} MiB, up to {:#x} bytes, device {:#x}",
                    exporter.memory_type,
                    exporter.flags.as_raw(),
                    exporter.heap >> 20,
                    exporter.budget,
                    exporter.device.registry_id()
                );
                Some(exporter)
            }
            Err(why) => {
                log::warn!("{ENV_VAR}: backings stay in memfds: {why}");
                None
            }
        }
    });
    engaged.as_ref()
}

/// The runtime's one device, the system default (module docs, "Engagement").
fn the_device() -> Result<rt::Device, Disengaged> {
    let options = crate::platform::device_options();
    let all = rt::Device::all_with(options).map_err(|e| Disengaged::NoDevice(e.to_string()))?;
    let default = rt::Device::system_default_with(options)
        .map_err(|e| Disengaged::NoDevice(e.to_string()))?;
    require_one_device(
        all.iter()
            .map(|device| (device.properties().kind, device.registry_id())),
        default.registry_id(),
    )?;
    Ok(default)
}

/// Only non-CPU devices participate in the single-GPU admission rule.
fn require_one_device(
    devices: impl IntoIterator<Item = (rt::DeviceKind, u64)>,
    default: u64,
) -> Result<(), Disengaged> {
    let ids: Vec<_> = devices
        .into_iter()
        .filter_map(|(kind, id)| (kind != rt::DeviceKind::Cpu).then_some(id))
        .collect();
    match ids.as_slice() {
        [one] if *one == default => Ok(()),
        _ => Err(Disengaged::SeveralDevices(ids.len())),
    }
}

/// The exporter over `device`, its self-test passed (module docs, "Engagement").
pub(crate) fn engage(device: &rt::Device) -> Result<Exporter, Disengaged> {
    device
        .device_proc_addr(c"vkGetMemoryFdKHR")
        .ok_or(Disengaged::NoExportFunctions)?;
    let fd = khr::external_memory_fd::DeviceFn::load(|name| {
        device
            .device_proc_addr(name)
            .map_or(std::ptr::null(), |f| f as *const std::ffi::c_void)
    });
    // SAFETY: the handles are the runtime's, used while the exporter holds the device; nothing
    // here destroys any of them.
    let raw = unsafe { device.raw() };
    let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    let asked = vk::PhysicalDeviceExternalBufferInfo::default()
        .usage(usage)
        .handle_type(DMA_BUF);
    let mut answered = vk::ExternalBufferProperties::default();
    // SAFETY: a query of the device's physical device, core at the runtime's Vulkan 1.2 floor.
    unsafe {
        raw.instance.get_physical_device_external_buffer_properties(
            raw.physical_device,
            &asked,
            &mut answered,
        );
    }
    let features = answered.external_memory_properties.external_memory_features;
    if !features.contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE)
        || features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY)
    {
        return Err(Disengaged::NotExportable(features.as_raw()));
    }
    let commands = raw.device.clone();
    let takes = buffer_memory_types(&commands, usage).ok_or(Disengaged::NoMemoryType)?;
    // SAFETY: a query of the device's physical device.
    let memory = unsafe {
        raw.instance
            .get_physical_device_memory_properties(raw.physical_device)
    };
    let (memory_type, flags, heap) =
        exportable_type(&memory, takes).ok_or(Disengaged::NoMemoryType)?;
    let exporter = Exporter {
        device: device.clone(),
        commands,
        fd,
        memory_type,
        flags,
        heap,
        budget: heap / 4,
        reserved: Arc::new(AtomicU64::new(0)),
    };
    self_test(&exporter)?;
    Ok(exporter)
}

/// The memory types a 64 KiB buffer of `usage` exporting a dma-buf takes on `commands`' device,
/// made and destroyed to ask; `None` where the device makes none.
fn buffer_memory_types(commands: &ash::Device, usage: vk::BufferUsageFlags) -> Option<u32> {
    let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(DMA_BUF);
    let info = vk::BufferCreateInfo::default()
        .size(SELF_TEST_BYTES)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external);
    // SAFETY: a valid create info; the buffer is this call's, destroyed before it returns, with
    // no memory bound and no command recorded.
    unsafe {
        let buffer = commands.create_buffer(&info, None).ok()?;
        let requirements = commands.get_buffer_memory_requirements(buffer);
        commands.destroy_buffer(buffer, None);
        Some(requirements.memory_type_bits)
    }
}

/// The first memory type of `takes` (a mask of type indices) that is device-local, host-visible
/// and coherent, and none of AMD's device-coherent or uncached, protected or lazily allocated: its
/// index, its flags and its heap's size (module docs, "Engagement").
fn exportable_type(
    memory: &vk::PhysicalDeviceMemoryProperties,
    takes: u32,
) -> Option<(u32, vk::MemoryPropertyFlags, u64)> {
    use vk::MemoryPropertyFlags as F;
    let wanted = F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_COHERENT;
    let unwanted =
        F::DEVICE_COHERENT_AMD | F::DEVICE_UNCACHED_AMD | F::PROTECTED | F::LAZILY_ALLOCATED;
    let count = usize::try_from(memory.memory_type_count).unwrap_or(0);
    (memory.memory_types.iter().zip(0u32..))
        .take(count)
        .find_map(|(ty, index)| {
            let fits = takes & (1 << index) != 0
                && ty.property_flags.contains(wanted)
                && !ty.property_flags.intersects(unwanted);
            let heap = memory
                .memory_heaps
                .get(usize::try_from(ty.heap_index).ok()?)?;
            fits.then_some((index, ty.property_flags, heap.size))
        })
}

/// What the self-test's buffers answered ([`self_test`]).
#[derive(Debug)]
enum Answer {
    /// The buffer was made.
    Made,
    /// The device refused the memory (`Unsupported`).
    Refused,
    /// The runtime failed the buffer otherwise.
    Failed(String),
}

impl Answer {
    /// The answer of a buffer's making.
    fn of<T>(made: &Result<T, rt::Error>) -> Answer {
        match made {
            Ok(_) => Answer::Made,
            Err(rt::Error::Unsupported(_)) => Answer::Refused,
            Err(e) => Answer::Failed(e.to_string()),
        }
    }
}

/// The self-test's verdict on a buffer over the memory's pointer alone: refused, as every
/// import of the memory must be a dma-buf's (module docs, "Engagement").
fn pointer_verdict(answer: Answer) -> Result<(), Disengaged> {
    match answer {
        Answer::Refused => Ok(()),
        Answer::Made => Err(Disengaged::PointerImported),
        Answer::Failed(e) => Err(Disengaged::PointerFailed(e)),
    }
}

/// The self-test's verdict on a buffer over the memory with a dma-buf source, which it called
/// `calls` times: made, the source called once.
fn source_verdict(answer: Answer, calls: u32) -> Result<(), Disengaged> {
    match answer {
        Answer::Made if calls == 1 => Ok(()),
        Answer::Made => Err(Disengaged::SourceNotCalled(calls)),
        Answer::Refused => Err(Disengaged::DmaBufRefused(
            "refused (Unsupported)".to_owned(),
        )),
        Answer::Failed(e) => Err(Disengaged::DmaBufRefused(e)),
    }
}

/// The self-test (module docs, "Engagement"), with no GPU work: over [`SELF_TEST_BYTES`] of the
/// exporter's memory, registered as a backing's is, a buffer of the platform's own (outside the
/// Metal address space) over its pointer alone is refused, and one with a dma-buf source counting
/// its own calls, which gives what `super::dma_buf::of_pages` gives, is made, the source called
/// once. Each buffer's keep-alive holds the memory until the runtime has freed its import.
fn self_test(exporter: &Exporter) -> Result<(), Disengaged> {
    const SEL: &str = "REIMS_VGPU_VRAM_SURFACES";
    let own = Arc::new(
        OwnMemory::in_vram(SELF_TEST_BYTES, exporter).map_err(Disengaged::SelfTestMemory)?,
    );
    let Some(at) = NonNull::new(vm::ptr(own.first()).cast::<u8>()) else {
        return Err(Disengaged::SelfTestMemory(Refused::Map(
            vm::INVALID_ADDRESS,
        )));
    };
    let buffer = |memory: rt::HostMemory| {
        let contents = rt::BufferContents::NoCopy {
            memory,
            deallocator: None,
            keep_alive: Some(crate::gpu::backing::keep_alive(Arc::clone(&own))),
        };
        let options = rt::ResourceOptions::default();
        let descriptor = crate::gpu::own_descriptor(SELF_TEST_BYTES, options, contents);
        crate::answer::unstopped_at(SEL, || {
            exporter.device.new_buffer_with_descriptor(descriptor)
        })
    };
    // SAFETY: the memory stays mapped, read and write, until the runtime drops the keep-alive,
    // after it freed the import; nothing frees or remaps it meanwhile.
    let pointer = buffer(unsafe { rt::HostMemory::new(at) });
    pointer_verdict(Answer::of(&pointer))?;
    drop(pointer);
    let calls = Arc::new(AtomicU32::new(0));
    let counted = Arc::clone(&calls);
    let source: rt::DmaBufSource = Arc::new(move |request: rt::DmaBufRequest| {
        counted.fetch_add(1, Ordering::Relaxed);
        let first = request.start.as_ptr().addr().to_u64();
        dma_buf::of_pages(first, request.length, request.granule)
            .map_err(|refusal| std::io::Error::other(refusal.to_string()))
    });
    // SAFETY: as above; the source's dma-bufs are duplicates of the memory's own, its pages in
    // order ([`chunks_of`]).
    let sourced = buffer(unsafe { rt::HostMemory::with_dma_buf(at, source) });
    let verdict = source_verdict(Answer::of(&sourced), calls.load(Ordering::Relaxed));
    drop(sourced);
    verdict
}

/// The device memory of a backing's stand-in, listed while it lives (module docs, "A backing"):
/// its dma-buf, whose size, as the kernel gives it, it holds against the exporter's budget.
#[derive(Debug)]
pub(crate) struct Listed {
    fd: Arc<OwnedFd>,
    size: u64,
    reserved: Reservation,
}

/// Admission precedes allocation; every early refusal returns its reservation.
#[derive(Debug)]
struct Reservation {
    counter: Arc<AtomicU64>,
    bytes: u64,
}

impl Reservation {
    fn new(counter: &Arc<AtomicU64>, bytes: u64, budget: u64) -> Result<Self, Refused> {
        counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |before| {
                before.checked_add(bytes).filter(|&after| after <= budget)
            })
            .map_err(|_| Refused::Budget {
                size: bytes,
                budget,
            })?;
        Ok(Self {
            counter: Arc::clone(counter),
            bytes,
        })
    }

    /// Account for a kernel export larger than the Vulkan allocation requested.
    #[allow(
        clippy::expect_used,
        reason = "The kernel export must cover the requested allocation"
    )]
    fn grow_to(&mut self, size: u64, budget: u64) -> Result<(), Refused> {
        let additional = size
            .checked_sub(self.bytes)
            .expect("the export is at least the requested allocation size");
        if additional == 0 {
            return Ok(());
        }
        self.counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |before| {
                before
                    .checked_add(additional)
                    .filter(|&after| after <= budget)
            })
            .map_err(|_| Refused::Budget { size, budget })?;
        self.bytes = size;
        Ok(())
    }

    fn release(&mut self) {
        if self.bytes != 0 {
            self.counter
                .fetch_sub(std::mem::take(&mut self.bytes), Ordering::Relaxed);
        }
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.release();
    }
}

impl Listed {
    /// The dma-buf.
    pub(super) fn fd(&self) -> Arc<OwnedFd> {
        Arc::clone(&self.fd)
    }
}

impl Drop for Listed {
    /// Unlists the dma-buf and gives its size back to the budget, once its mapping is gone
    /// (`OwnMemory`'s drop).
    fn drop(&mut self) {
        VRAM_FILES
            .lock()
            .retain(|(fd, _)| !Arc::ptr_eq(fd, &self.fd));
        self.reserved.release();
        LIVE_BACKINGS.fetch_sub(1, Ordering::Relaxed);
        LIVE_BYTES.fetch_sub(self.size, Ordering::Relaxed);
        log::debug!("a backing's {:#x} bytes of VRAM went", self.size);
    }
}

/// The dma-bufs of the backings in device memory and their sizes. A leaf lock: no call is made
/// under it.
static VRAM_FILES: Mutex<Vec<(Arc<OwnedFd>, u64)>> = Mutex::new(Vec::new());

/// The backings in device memory now, and their bytes ([`live`]).
static LIVE_BACKINGS: AtomicU64 = AtomicU64::new(0);
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
/// The duplicates of their dma-bufs given to imports, and the backings refused, since the start.
static DUPLICATES: AtomicU64 = AtomicU64::new(0);
static REFUSALS: AtomicU64 = AtomicU64::new(0);

/// The counts of the backings in device memory, a diagnostic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Counts {
    /// Backings live.
    pub(crate) backings: u64,
    /// Their dma-bufs' bytes.
    pub(crate) bytes: u64,
    /// Duplicates of their dma-bufs given to imports.
    pub(crate) duplicates: u64,
    /// Backings refused, left in memfds.
    pub(crate) refusals: u64,
}

/// The counts now.
pub(crate) fn live() -> Counts {
    Counts {
        backings: LIVE_BACKINGS.load(Ordering::Relaxed),
        bytes: LIVE_BYTES.load(Ordering::Relaxed),
        duplicates: DUPLICATES.load(Ordering::Relaxed),
        refusals: REFUSALS.load(Ordering::Relaxed),
    }
}

/// A new dma-buf of `length` bytes of the exporter's memory, whole pages, listed (module docs,
/// "A backing"); its zeros are the kernel's, which clears device memory it gives.
pub(super) fn export(exporter: &Exporter, length: u64) -> Result<Listed, Refused> {
    if rt::process_exiting() {
        return Err(Refused::Exiting);
    }
    let mut reserved = Reservation::new(&exporter.reserved, length, exporter.budget)?;
    let mut exported = vk::ExportMemoryAllocateInfo::default().handle_types(DMA_BUF);
    let info = vk::MemoryAllocateInfo::default()
        .allocation_size(length)
        .memory_type_index(exporter.memory_type)
        .push_next(&mut exported);
    // SAFETY: a valid allocate info of the device's own memory type, freed below on every path.
    let memory =
        unsafe { exporter.commands.allocate_memory(&info, None) }.map_err(Refused::Allocation)?;
    let asked = vk::MemoryGetFdInfoKHR::default()
        .memory(memory)
        .handle_type(DMA_BUF);
    let mut raw = -1;
    // SAFETY: the memory just made, exportable as a dma-buf; `raw` takes the descriptor.
    let answer =
        unsafe { (exporter.fd.get_memory_fd_khr)(exporter.commands.handle(), &asked, &mut raw) };
    // SAFETY: the memory is this call's, bound to nothing and used by no command; the dma-buf,
    // where one was made, keeps the memory from here on.
    unsafe { exporter.commands.free_memory(memory, None) };
    if answer != vk::Result::SUCCESS {
        return Err(Refused::Export(answer));
    }
    // SAFETY: the descriptor `vkGetMemoryFdKHR` just gave, owned from here on.
    let fd = unsafe { OwnedFd::from_raw_fd(raw) };
    // SAFETY: a query of an open descriptor's size.
    let size = unsafe { libc::lseek(fd.as_raw_fd(), 0, libc::SEEK_END) };
    let Some(size) = u64::try_from(size).ok().filter(|&s| s >= length) else {
        return Err(Refused::ShortFile { size, length });
    };
    reserved.grow_to(size, exporter.budget)?;
    let fd = Arc::new(fd);
    VRAM_FILES.lock().push((Arc::clone(&fd), size));
    LIVE_BACKINGS.fetch_add(1, Ordering::Relaxed);
    LIVE_BYTES.fetch_add(size, Ordering::Relaxed);
    Ok(Listed { fd, size, reserved })
}

/// The dma-bufs of a window `super::dma_buf::of_pages` resolved to `resolved`, `length` bytes,
/// where it lies in a backing's device memory (module docs, "A backing"): one duplicate of its
/// dma-buf, which meets any granule, where the window is one piece from the dma-buf's start
/// within its size; refused where the window lies in such memory otherwise, as the runtime's chunks
/// carry no offset into a dma-buf, and udmabuf takes no pages of one. `None` where no piece is of
/// such memory.
pub(super) fn chunks_of(
    resolved: &[files::Piece],
    length: u64,
) -> Option<Result<Vec<DmaBufChunk>, Refusal>> {
    let sizes: Vec<Option<u64>> = {
        let listed = VRAM_FILES.lock();
        (resolved.iter())
            .map(|p| {
                (listed.iter())
                    .find(|(fd, _)| Arc::ptr_eq(fd, &p.backing.fd))
                    .map(|&(_, size)| size)
            })
            .collect()
    };
    let at = sizes.iter().position(Option::is_some)?;
    match (resolved, sizes.as_slice()) {
        ([one], [Some(size)]) if one.backing.offset == 0 && length <= *size => {
            let given = one.backing.fd.try_clone().map(|fd| {
                DUPLICATES.fetch_add(1, Ordering::Relaxed);
                vec![DmaBufChunk { fd, length }]
            });
            Some(given.map_err(Refusal::Kernel))
        }
        _ => Some(Err(Refusal::VramWindow {
            offset: resolved[at].backing.offset,
            pieces: resolved.len().to_u64(),
        })),
    }
}

/// Logs a backing of `length` bytes at `base` placed in device memory, and traces it (module
/// docs).
pub(super) fn placed(base: u64, length: u64) {
    let now = live();
    log::info!(
        "a backing of {length:#x} bytes at {base:#x} is in VRAM: {} live, {:#x} bytes",
        now.backings,
        now.bytes
    );
    // vsrf: whether the backing is in device memory, its length, its base, why not (0: it is)
    reims_vgpu_core::trace::event("vsrf", &[1, length, base, 0]);
}

/// Counts and logs a backing of `length` bytes at `base` the switch could not place, left in a
/// memfd: a warning the first time of each kind, debug lines after; and traces it (module docs).
pub(super) fn refused(why: &Refused, base: u64, length: u64) {
    /// Whether each kind was warned of, by [`Refused::code`].
    static WARNED: [AtomicBool; 8] = [const { AtomicBool::new(false) }; 8];
    REFUSALS.fetch_add(1, Ordering::Relaxed);
    let first = WARNED
        .get(why.code())
        .is_some_and(|warned| !warned.swap(true, Ordering::Relaxed));
    if first {
        log::warn!("a backing of {length:#x} bytes stays in a memfd: {why}");
    } else {
        log::debug!("a backing of {length:#x} bytes at {base:#x} stays in a memfd: {why}");
    }
    // vsrf: as above
    let code = why.code().to_u64();
    reims_vgpu_core::trace::event("vsrf", &[0, length, base, code]);
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::LazyLock;

    use super::*;

    /// An exporter over the tests' device, engaged once for the process, its self-test passed.
    pub(crate) fn test_exporter() -> &'static Exporter {
        static ENGAGED: LazyLock<Exporter> = LazyLock::new(|| {
            engage(&crate::gpu::device::tests::dev().raw).expect("the switch engages here")
        });
        &ENGAGED
    }

    /// A fresh exporter over the tests' device with a budget of `budget` bytes, its bytes reserved
    /// counted apart from any other exporter's.
    pub(crate) fn exporter_with_budget(budget: u64) -> &'static Exporter {
        let mut exporter =
            engage(&crate::gpu::device::tests::dev().raw).expect("the switch engages here");
        exporter.budget = budget;
        Box::leak(Box::new(exporter))
    }

    /// The bytes an exporter's backings take now.
    pub(crate) fn reserved(exporter: &Exporter) -> u64 {
        exporter.reserved.load(Ordering::Relaxed)
    }

    /// Whether `fd` is a backing's dma-buf of device memory.
    pub(crate) fn listed(fd: &Arc<OwnedFd>) -> bool {
        VRAM_FILES.lock().iter().any(|(f, _)| Arc::ptr_eq(f, fd))
    }

    /// Only `1` turns the switch on; unset, `0`, `true` and the empty string leave it off
    #[test]
    fn the_switch_is_on_only_when_set_to_1() {
        use std::ffi::OsStr;
        assert!(switched_on(Some(OsStr::new("1"))));
        for off in [None, Some("0"), Some("true"), Some(""), Some("1 ")] {
            assert!(!switched_on(off.map(OsStr::new)), "{off:?}");
        }
    }

    /// The self-test engages only where a buffer over the memory's pointer alone is refused and
    /// one with a dma-buf source is made, calling it once: an imported pointer, another failure
    /// of it, a refused dma-buf or a source called other than once each leave it disengaged
    #[test]
    fn engagement_refuses_where_the_pointer_imports() {
        assert!(pointer_verdict(Answer::Refused).is_ok());
        assert!(matches!(
            pointer_verdict(Answer::Made),
            Err(Disengaged::PointerImported)
        ));
        assert!(matches!(
            pointer_verdict(Answer::Failed("lost".to_owned())),
            Err(Disengaged::PointerFailed(_))
        ));
        assert!(source_verdict(Answer::Made, 1).is_ok());
        for calls in [0, 2] {
            assert!(matches!(
                source_verdict(Answer::Made, calls),
                Err(Disengaged::SourceNotCalled(c)) if c == calls
            ));
        }
        assert!(matches!(
            source_verdict(Answer::Refused, 1),
            Err(Disengaged::DmaBufRefused(_))
        ));
    }

    /// The memory type taken is the first device-local, host-visible and coherent one a buffer
    /// takes, of none of the kinds the runtime's allocations avoid, with its heap's size
    #[test]
    fn the_memory_type_is_device_local_host_visible_and_coherent() {
        use vk::MemoryPropertyFlags as F;
        let mut memory = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: 4,
            memory_heap_count: 2,
            ..Default::default()
        };
        memory.memory_heaps[0].size = 1 << 30;
        memory.memory_heaps[1].size = 16 << 30;
        let rebar = F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_COHERENT;
        for (i, (flags, heap)) in [
            (F::DEVICE_LOCAL, 1),
            (rebar | F::DEVICE_COHERENT_AMD, 1),
            (rebar, 1),
            (F::HOST_VISIBLE | F::HOST_COHERENT, 0),
        ]
        .into_iter()
        .enumerate()
        {
            memory.memory_types[i] = vk::MemoryType {
                property_flags: flags,
                heap_index: heap,
            };
        }
        let taken = |takes| exportable_type(&memory, takes).map(|(i, f, h)| (i, f.as_raw(), h));
        assert_eq!(taken(0xf), Some((2, rebar.as_raw(), 16 << 30)));
        assert_eq!(taken(0b1011), None, "type 2 not taken");
    }

    /// Measured on the RX 9070 XT (module docs): a buffer over the mapping of the switch's
    /// memory with no dma-buf source is refused, so every import of it is a dma-buf's
    #[test]
    fn the_pointer_of_vram_memory_is_refused_as_host_memory() {
        let exporter = test_exporter();
        let own = OwnMemory::in_vram(SELF_TEST_BYTES, exporter).unwrap();
        let at = NonNull::new(vm::ptr(own.first()).cast::<u8>()).unwrap();
        // SAFETY: the memory outlives the call, which makes no buffer.
        let memory = unsafe { rt::HostMemory::new(at) };
        let contents = rt::BufferContents::NoCopy {
            memory,
            deallocator: None,
            keep_alive: None,
        };
        let descriptor =
            crate::gpu::own_descriptor(SELF_TEST_BYTES, rt::ResourceOptions::default(), contents);
        let made = exporter.device.new_buffer_with_descriptor(descriptor);
        assert!(
            matches!(made, Err(rt::Error::Unsupported(_))),
            "{:?}",
            made.map(|_| ())
        );
    }

    /// A backing the switch places, made as core makes it, takes a texture over its plane that
    /// is an image over a duplicate of its dma-buf: a blit into the texture reaches the CPU
    /// through the backing's base, and the CPU's writes there reach a blit out of a texture made
    /// after them, with no shadow of the bytes. At a small shape, at rows of 256 bytes, and at
    /// 3840 × 2160, rows of 15,360 bytes
    #[test]
    fn a_backing_in_vram_pages_in_and_reads_back() {
        use reims_vgpu_core::platform::Handle as _;
        use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
        use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use reims_vgpu_core::platform::device::GpuDevice as _;
        use reims_vgpu_core::platform::os::surface::{
            BackingPlaneSpec, BackingSurfaceSpec, IoSurfaces as _, Surface as _,
        };
        use reims_vgpu_core::platform::resource::Buffer as _;
        use reims_vgpu_core::platform::types::{MtlOrigin, MtlSize};
        use reims_vgpu_core::raise::OnFault;

        use crate::gpu::resource::tests::bgra;

        let d = crate::gpu::device::tests::dev();
        let queue = d.new_command_queue().unwrap();
        for (width, height) in [(64u32, 32u32), (64, 4), (3840, 2160)] {
            let (w, h) = (u64::from(width), u64::from(height));
            let (row, length) = (w * 4, w * h * 4);
            let spec = BackingSurfaceSpec {
                pixel_format: u32::from_be_bytes(*b"BGRA"),
                planes: vec![BackingPlaneSpec {
                    width,
                    height,
                    bytes_per_element: 4,
                    bytes_per_row: i32::try_from(row).unwrap(),
                    base: 0,
                    offset: 0,
                }],
                width,
                height,
            };
            let surface = with_exporter(test_exporter(), || {
                super::super::LinuxSurfaces::new()
                    .create_backing(&spec)
                    .unwrap()
            });
            let base = surface.base_address().addr().to_u64();
            let pieces = files::pieces(base, vm::page_size()).unwrap();
            assert!(listed(&pieces[0].backing.fd), "{w} x {h}: in VRAM");
            let texels =
                |seed: u32| (0..w * h).map(move |i| seed ^ (i as u32).wrapping_mul(0x9e37_79b9));
            // A Shared buffer of `length` bytes, which holds `bytes` where given.
            let shared = |bytes: Option<&[u8]>| {
                let b = d
                    .new_buffer_of_length(OnFault::Die, length, 0)
                    .unwrap()
                    .unwrap();
                if let Some(bytes) = bytes {
                    // SAFETY: a Shared buffer of `length` bytes no command uses; `bytes` is as
                    // long.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            bytes.as_ptr(),
                            b.contents().as_ptr::<u8>(),
                            bytes.len(),
                        );
                    }
                }
                b
            };
            let words =
                |seed: u32| -> Vec<u8> { texels(seed).flat_map(u32::to_le_bytes).collect() };
            // Every 997th texel and the last, read through the backing's base: uncached reads.
            let sampled = |seed: u32| {
                let expected: Vec<u32> = texels(seed).collect();
                (0..w * h).step_by(997).chain([w * h - 1]).all(|i| {
                    // SAFETY: the backing's memory, mapped while it lives, which no command uses
                    // now.
                    let at = unsafe { vm::ptr(base + 4 * i).cast::<u32>().read_volatile() };
                    at == expected[usize::try_from(i).unwrap()]
                })
            };
            let start = d.raw.work_counts();
            let duplicates = live().duplicates;
            let run = |cmd: BlitCmd<'_, crate::platform::VulkanPlatform>| {
                let cb = queue.command_buffer(OnFault::Die).unwrap().unwrap();
                let b = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
                b.encode(OnFault::Die, cmd).unwrap();
                b.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
                cb.commit(OnFault::Die).unwrap();
                cb.wait_until_completed(OnFault::Die).unwrap();
                assert!(cb.error().is_none(), "{w} x {h}: {:?}", cb.error());
            };
            // The GPU's writes into the texture reach the CPU through the backing's base.
            let texture = d
                .new_texture_on_surface(&bgra(w, h, 0), Some(&surface), 0)
                .unwrap()
                .unwrap();
            let source = shared(Some(&words(0x1234_5678)));
            run(BlitCmd::CopyBufferToTexture {
                source_buffer: &source,
                source_offset: 0,
                source_bytes_per_row: row,
                source_bytes_per_image: length,
                source_size: MtlSize::new(w, h, 1),
                destination_texture: &texture,
                destination_slice: 0,
                destination_level: 0,
                destination_origin: MtlOrigin::new(0, 0, 0),
            });
            assert!(sampled(0x1234_5678), "{w} x {h}: GPU to CPU");
            drop(texture);
            // The CPU's writes through the base reach a texture made after them.
            let written = words(0x0bad_cafe);
            // SAFETY: as above; `written` is `length` bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    written.as_ptr(),
                    vm::ptr(base).cast::<u8>(),
                    written.len(),
                );
            }
            surface.cpu_wrote();
            let texture = d
                .new_texture_on_surface(&bgra(w, h, 0), Some(&surface), 0)
                .unwrap()
                .unwrap();
            let out = shared(None);
            run(BlitCmd::CopyTextureToBuffer {
                source_texture: &texture,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(w, h, 1),
                destination_buffer: &out,
                destination_offset: 0,
                destination_bytes_per_row: row,
                destination_bytes_per_image: length,
            });
            // SAFETY: a Shared buffer of `length` bytes the GPU's copy completed.
            let read = unsafe {
                std::slice::from_raw_parts(
                    out.contents().as_ptr::<u8>(),
                    usize::try_from(length).unwrap(),
                )
            };
            assert!(read == written.as_slice(), "{w} x {h}: CPU to GPU");
            let now = d.raw.work_counts();
            assert_eq!(
                now.dma_buf_image_fallbacks, start.dma_buf_image_fallbacks,
                "{w} x {h}: no shadow"
            );
            assert!(now.dma_buf_images > start.dma_buf_images, "{w} x {h}");
            assert!(
                live().duplicates > duplicates,
                "{w} x {h}: its dma-buf's duplicates"
            );
            assert!(surface.id().0 != 0);
            drop((texture, surface));
        }
    }

    /// The switch's memory is zeros first, a dma-buf the kernel sized at least as asked, listed
    /// while it lives and its bytes held against the budget, both given back when it goes; past
    /// the budget, it is refused with nothing held
    #[test]
    fn vram_memory_is_listed_and_budgeted_while_it_lives() {
        let page = vm::page_size();
        let exporter = exporter_with_budget(4 * page);
        let own = OwnMemory::in_vram(2 * page, exporter).unwrap();
        // SAFETY: the memory just mapped, read and write.
        let first = unsafe { vm::ptr(own.first()).cast::<u8>().read_volatile() };
        assert_eq!(first, 0);
        let fd = files::pieces(own.first(), 2 * page).unwrap()[0]
            .backing
            .fd
            .clone();
        assert!(listed(&fd));
        assert_eq!(reserved(exporter), 2 * page);
        assert!(matches!(
            OwnMemory::in_vram(3 * page, exporter),
            Err(Refused::Budget { .. })
        ));
        assert_eq!(reserved(exporter), 2 * page, "the refusal holds nothing");
        drop(own);
        assert!(!listed(&fd));
        assert_eq!(reserved(exporter), 0);
    }

    /// Allocation refusals, rounded export sizes and overflow never leak the budget.
    #[test]
    fn vram_budget_reservations_release_on_failure_and_growth() {
        let counter = Arc::new(AtomicU64::new(0));
        {
            let mut first = Reservation::new(&counter, 6, 10).unwrap();
            assert!(matches!(
                Reservation::new(&counter, 5, 10),
                Err(Refused::Budget { .. })
            ));
            assert!(matches!(first.grow_to(11, 10), Err(Refused::Budget { .. })));
            assert_eq!(counter.load(Ordering::Relaxed), 6);
        }
        assert_eq!(counter.load(Ordering::Relaxed), 0);
        {
            let mut rounded = Reservation::new(&counter, 6, 10).unwrap();
            rounded.grow_to(10, 10).unwrap();
            assert_eq!(counter.load(Ordering::Relaxed), 10);
        }
        assert_eq!(counter.load(Ordering::Relaxed), 0);
        {
            let _large = Reservation::new(&counter, u64::MAX - 1, u64::MAX).unwrap();
            assert!(matches!(
                Reservation::new(&counter, 2, u64::MAX),
                Err(Refused::Budget { .. })
            ));
            assert_eq!(counter.load(Ordering::Relaxed), u64::MAX - 1);
        }
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    /// Simultaneous callers cannot all pass an advisory pre-allocation load.
    #[test]
    fn vram_budget_admission_is_atomic() {
        use std::sync::Barrier;

        let counter = Arc::new(AtomicU64::new(0));
        let admitted = AtomicU64::new(0);
        let reserved = Barrier::new(9);
        let release = Barrier::new(9);
        let held = std::thread::scope(|scope| {
            for _ in 0..8 {
                let (counter, admitted, reserved, release) =
                    (&counter, &admitted, &reserved, &release);
                scope.spawn(move || {
                    let reservation = Reservation::new(counter, 1, 4);
                    if reservation.is_ok() {
                        admitted.fetch_add(1, Ordering::Relaxed);
                    }
                    reserved.wait();
                    release.wait();
                    drop(reservation);
                });
            }
            reserved.wait();
            let held = (
                admitted.load(Ordering::Relaxed),
                counter.load(Ordering::Relaxed),
            );
            release.wait();
            held
        });
        assert_eq!(held, (4, 4));
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    /// Software rendering is ignored; a second hardware device or wrong default still refuses.
    #[test]
    fn vram_device_rule_ignores_cpu_devices_only() {
        use rt::DeviceKind::{Cpu, Discrete, Integrated};

        assert!(require_one_device([(Cpu, 1), (Discrete, 2)], 2).is_ok());
        assert!(matches!(
            require_one_device([(Cpu, 1), (Discrete, 2), (Integrated, 3)], 2),
            Err(Disengaged::SeveralDevices(2))
        ));
        assert!(matches!(
            require_one_device([(Cpu, 1)], 1),
            Err(Disengaged::SeveralDevices(0))
        ));
        assert!(matches!(
            require_one_device([(Cpu, 1), (Discrete, 2)], 1),
            Err(Disengaged::SeveralDevices(1))
        ));
    }
}
