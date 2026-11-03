//! The tests of a surface's base anchor (`crate::os::wants_base_anchor`, `base_anchor`): Linux
//! only, where the IOSurface stand-in keeps one. Each test runs in a process of its own, so the
//! device's counts and the anchors live move only by what the test does.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
use reims_vgpu_core::platform::os::surface::Surface as _;
use reims_vgpu_core::platform::resource::{Buffer as _, Library as _};
use reims_vgpu_core::platform::types::{CommandBufferStatus, MtlOrigin, MtlSize, NsRange};

use super::*;
use crate::gpu::command::VulkanCommandBuffer;
use crate::gpu::resource::tests::bgra;
use crate::os::tests::{bgra_surface, bgra_surface_over_at, free};

/// The kernel that keeps the GPU busy (`src/gpu/spin.metal`), as the device's tests use it.
const SPIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spin.metallib"));
/// The spin's rounds: about 60 ms on the M3 Max.
const ROUNDS: u32 = 1_000;
/// The test surfaces' size: 64 x 32 BGRA texels.
const WIDTH: u64 = 64;
const HEIGHT: u64 = 32;
const LEN: u64 = WIDTH * HEIGHT * 4;

/// Whether `done` came true within 30 s, polled each millisecond.
fn within(done: &dyn Fn() -> bool) -> bool {
    let wait = Duration::from_secs(30);
    (0..wait.as_millis()).any(|_| {
        done() || {
            std::thread::sleep(Duration::from_millis(1));
            false
        }
    }) || done()
}

/// The device's ID, which its anchors are kept by.
fn id(d: &VulkanDevice) -> u64 {
    d.id().0
}

/// The device's live imports of host pages: by pointer, or (RADV, where a memfd's pages are refused
/// by pointer) as dma-buf memories.
fn live(c: &rt::WorkCounts) -> u64 {
    c.host_imports_live + c.dma_buf_memories_live
}

/// The device's counts now.
fn counts(d: &VulkanDevice) -> rt::WorkCounts {
    d.raw.work_counts()
}

/// A buffer over `s`, made with `newBufferWithIOSurface:`.
fn on_surface(d: &VulkanDevice, s: &OsSurface) -> VulkanBuffer {
    d.new_buffer_on_surface(Some(s)).unwrap().unwrap()
}

/// `rounds` buffers over `s`, each released before the next is made, as a guest that makes one
/// at each frame does.
fn churn(d: &VulkanDevice, s: &OsSurface, rounds: u32) {
    for _ in 0..rounds {
        drop(on_surface(d, s));
    }
}

/// A Shared buffer of `length` bytes.
fn shared(d: &VulkanDevice, length: u64) -> VulkanBuffer {
    d.new_buffer_of_length(OnFault::Die, length, 0)
        .unwrap()
        .unwrap()
}

/// The first `length` bytes of a Shared buffer whose command buffer completed.
fn bytes_of(b: &VulkanBuffer, length: u64) -> Vec<u8> {
    let length = usize::try_from(length).unwrap();
    // SAFETY: a Shared buffer of at least `length` bytes the GPU's writes completed.
    unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>(), length) }.to_vec()
}

/// Writes `byte` over the `length` bytes at `s`'s base address, a write declared to it.
fn fill_surface(s: &OsSurface, byte: u8, length: u64) {
    // SAFETY: the surface's memory, mapped and writable while it lives, which no command uses.
    unsafe {
        std::ptr::write_bytes(
            s.base_address().as_ptr::<u8>(),
            byte,
            usize::try_from(length).unwrap(),
        );
    }
    s.cpu_wrote();
}

/// A command buffer on a new queue of `d` that spins the GPU for about 60 ms and then copies
/// `length` bytes of `source` into a new buffer, committed and scheduled: it, the copy's buffer,
/// and whether the spin still ran as this returned.
fn spin_then_copy(
    d: &VulkanDevice,
    source: &VulkanBuffer,
    length: u64,
) -> (VulkanCommandBuffer, VulkanBuffer, bool) {
    let library = d.new_library(SPIN).unwrap().unwrap().unwrap();
    let function = library.new_function(OnFault::Die, "spin").unwrap().unwrap();
    let state = d
        .new_compute_pipeline_state_for_function(Some(&function))
        .unwrap()
        .unwrap();
    let words = shared(d, 16);
    let word = |i: usize| {
        // SAFETY: one of the buffer's four words, which the kernel reads and writes.
        unsafe { AtomicU32::from_ptr(words.contents().as_ptr::<u32>().add(i)) }
    };
    word(0).store(ROUNDS, Ordering::SeqCst);
    let out = shared(d, length);
    let cb = (d.new_command_queue().unwrap())
        .command_buffer(OnFault::Die)
        .unwrap()
        .unwrap();
    let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
    let one = MtlSize::new(1, 1, 1);
    for cmd in [
        ComputeCmd::SetComputePipelineState {
            state: Some(&state),
        },
        ComputeCmd::SetBuffers {
            buffers: &[Some(&words)],
            offsets: &[0],
            range: NsRange::new(0, 1),
        },
        ComputeCmd::DispatchThreads {
            threads_per_grid: one,
            threads_per_threadgroup: one,
        },
        ComputeCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    let b = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    for cmd in [
        BlitCmd::CopyBufferToBuffer {
            source_buffer: Some(source),
            source_offset: 0,
            destination_buffer: Some(&out),
            destination_offset: 0,
            size: length,
        },
        BlitCmd::EndEncoding,
    ] {
        b.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_scheduled().unwrap();
    let running = word(1).load(Ordering::SeqCst) == 0;
    (cb, out, running)
}

/// Whether `cb` completed within 30 s, without an error.
fn completes(cb: &VulkanCommandBuffer) -> bool {
    let ended = [CommandBufferStatus::COMPLETED, CommandBufferStatus::ERROR];
    within(&|| ended.contains(&cb.status())) && cb.status() == CommandBufferStatus::COMPLETED
}

/// Buffers made over a surface one after another, each released before the next, share one
/// import from the second on, which the surface's base anchor keeps: the device's imports live
/// stay one more than before the surface, and its dma-buf imports stop growing. Each lies at the
/// address the first, made before any anchor, took, and each release gives the surface's retain
/// count back to 1: the anchor counts no reference. One anchor lives
#[test]
fn buffers_made_over_a_surface_one_after_another_share_one_import() {
    let d = dev();
    let anchors = crate::os::anchors_live();
    let before = counts(&d);
    let s = bgra_surface(WIDTH, HEIGHT);
    let mut at = None;
    let mut dma_bufs = None;
    for round in 0..6 {
        let b = on_surface(&d, &s);
        assert_eq!(s.retain_count(), 2, "round {round}");
        let address = b.gpu_address();
        assert_eq!(*at.get_or_insert(address), address, "round {round}");
        drop(b);
        assert_eq!(s.retain_count(), 1, "round {round}");
        let now = counts(&d);
        if round >= 1 {
            assert_eq!(live(&now), live(&before) + 1, "round {round}");
            let made = *dma_bufs.get_or_insert(now.dma_buf_imports);
            assert_eq!(now.dma_buf_imports, made, "round {round}");
        }
    }
    assert!(crate::os::has_base_anchor(&s, id(&d)));
    assert_eq!(crate::os::anchors_live(), anchors + 1);
}

/// The anchor lies in the import of the buffer it is made after, and makes none of its own: the
/// second buffer over a surface, made while none lives, imports the pages as the first did, and
/// the anchor made after it adds no import live and no dma-buf import. This pins the runtime's
/// join (the same first byte and length, of the same memory class); where it stops joining, the
/// anchor, whose memory has no dma-buf source, is refused on a device that refuses the pages'
/// pointer, and this fails there
#[test]
fn an_anchor_joins_and_makes_no_import() {
    let d = dev();
    let s = bgra_surface(WIDTH, HEIGHT);
    let start = counts(&d);
    let first = on_surface(&d, &s);
    let one = counts(&d);
    drop(first);
    assert!(!crate::os::has_base_anchor(&s, id(&d)), "after the first");
    let between = counts(&d);
    let second = on_surface(&d, &s);
    let two = counts(&d);
    assert!(crate::os::has_base_anchor(&s, id(&d)), "after the second");
    assert_eq!(
        live(&two) - live(&between),
        live(&one) - live(&start),
        "imports live"
    );
    assert_eq!(
        two.dma_buf_imports - between.dma_buf_imports,
        one.dma_buf_imports - start.dma_buf_imports,
        "dma-buf imports"
    );
    drop(second);
    assert_eq!(live(&counts(&d)), live(&two));
}

/// A surface's base anchor goes with its last object, a lookup's included: with every buffer
/// over it released, the surface's last object takes the device's imports live back to what
/// they were before it, the anchors live too, and unmaps its memory
#[test]
fn a_surface_anchor_goes_with_the_surfaces_last_reference() {
    let d = dev();
    let anchors = crate::os::anchors_live();
    let before = live(&counts(&d));
    let s = bgra_surface(WIDTH, HEIGHT);
    let base = s.base_address();
    churn(&d, &s, 3);
    assert_eq!(crate::os::anchors_live(), anchors + 1);
    assert_eq!(live(&counts(&d)), before + 1);
    let lookup = s.lookup_by_port().unwrap();
    drop(s);
    assert!(crate::os::mapped(base), "the lookup holds the surface");
    assert_eq!(crate::os::anchors_live(), anchors + 1, "the lookup's");
    drop(lookup);
    assert!(
        within(&|| live(&counts(&d)) == before),
        "the anchor's import was not freed"
    );
    assert_eq!(crate::os::anchors_live(), anchors);
    assert!(within(&|| !crate::os::mapped(base)), "unmapped");
}

/// An anchor dropped in a guest buffer's deallocator: a buffer over an anchored surface holds its
/// last reference and is released while a command buffer that reads it runs, so as that command
/// buffer completes the runtime releases the buffer, its deallocator drops the surface's last
/// reference, and the surface drops its anchor, inside the runtime's release, on its thread.
/// Nothing hangs or aborts; the memory stays mapped until the command buffer completed, the
/// command buffer read the surface's bytes, the import is freed and the memory unmapped after,
/// and no deallocator is delayed
#[test]
fn an_anchor_dropped_in_a_deallocator_frees_after_it() {
    const BYTE: u8 = 0x3c;
    let d = dev();
    let anchors = crate::os::anchors_live();
    let before = live(&counts(&d));
    let s = bgra_surface(WIDTH, HEIGHT);
    let base = s.base_address();
    churn(&d, &s, 2);
    assert!(crate::os::has_base_anchor(&s, id(&d)));
    fill_surface(&s, BYTE, LEN);
    let buffer = on_surface(&d, &s);
    let (cb, out, running) = spin_then_copy(&d, &buffer, LEN);
    drop((buffer, s));
    assert!(running, "released while the kernel ran");
    assert!(
        crate::os::mapped(base),
        "mapped while the command buffer runs"
    );
    assert!(completes(&cb), "{:?}", cb.error());
    assert!(
        bytes_of(&out, LEN).iter().all(|&b| b == BYTE),
        "the command buffer read the surface's bytes"
    );
    assert!(
        within(&|| live(&counts(&d)) == before),
        "the import was not freed"
    );
    assert!(within(&|| !crate::os::mapped(base)), "unmapped after");
    assert_eq!(crate::os::anchors_live(), anchors);
    assert_eq!(counts(&d).host_deallocators_delayed, 0);
}

/// A buffer that joined an anchored import runs its deallocator at its release, Metal's moment:
/// the surface's retain count falls at once, its keep-alive is dropped then, as the anchor keeps
/// the import's pages mapped, and no deallocator is delayed
#[test]
fn a_joined_buffers_deallocator_runs_at_release() {
    let d = dev();
    let s = bgra_surface(WIDTH, HEIGHT);
    churn(&d, &s, 2);
    assert!(crate::os::has_base_anchor(&s, id(&d)));
    let b = on_surface(&d, &s);
    assert_eq!(s.retain_count(), 2);
    let covered = counts(&d).host_keep_alives_covered;
    drop(b);
    assert_eq!(s.retain_count(), 1, "the deallocator ran at the release");
    assert_eq!(counts(&d).host_keep_alives_covered, covered + 1);
    assert_eq!(counts(&d).host_deallocators_delayed, 0);
}

/// Textures made over a surface one after another, each released before the next: from the
/// third, each lies in the import the anchor keeps and takes the image an earlier one made, which
/// the import holds, so the device's shared images grow by one a texture and its dma-buf images'
/// memories live stay as they were. Each reads the texels the CPU wrote to the surface before it
/// was made
#[test]
fn textures_over_a_churned_surface_share_one_image() {
    let d = dev();
    let s = bgra_surface(WIDTH, HEIGHT);
    let queue = d.new_command_queue().unwrap();
    let mut imaged = false;
    let mut last: Option<rt::WorkCounts> = None;
    for round in 0..6u8 {
        let byte = 0x10 + round;
        fill_surface(&s, byte, LEN);
        let start = counts(&d);
        let texture = d
            .new_texture_on_surface(&bgra(WIDTH, HEIGHT, 0), Some(&s), 0)
            .unwrap()
            .unwrap();
        let out = shared(&d, LEN);
        let cb = queue.command_buffer(OnFault::Die).unwrap().unwrap();
        let b = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        for cmd in [
            BlitCmd::CopyTextureToBuffer {
                source_texture: &texture,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(WIDTH, HEIGHT, 1),
                destination_buffer: &out,
                destination_offset: 0,
                destination_bytes_per_row: WIDTH * 4,
                destination_bytes_per_image: LEN,
            },
            BlitCmd::EndEncoding,
        ] {
            b.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "round {round}: {:?}", cb.error());
        assert!(
            bytes_of(&out, LEN).iter().all(|&b| b == byte),
            "round {round}: the texture reads the surface's texels"
        );
        drop((texture, cb));
        let now = counts(&d);
        // Whether the texture took an image aliasing the surface's bytes, which the runtime
        // shares: on a device that imports them as dma-bufs, or by their pointer at Metal's
        // row pitch. A device that makes neither shadows the texture, and shares nothing.
        if round == 1 {
            imaged = now.dma_buf_images + now.pitched_host_images
                > start.dma_buf_images + start.pitched_host_images;
        }
        if round >= 2
            && imaged
            && let Some(last) = &last
        {
            assert_eq!(
                now.shared_alias_images,
                last.shared_alias_images + 1,
                "round {round}"
            );
            assert_eq!(
                now.dma_buf_image_memories_live, last.dma_buf_image_memories_live,
                "round {round}"
            );
        }
        last = Some(now);
    }
    assert!(crate::os::has_base_anchor(&s, id(&d)));
}

/// [`a_surface_textures_image_keeps_the_surfaces_memory`](super::a_surface_textures_image_keeps_the_surfaces_memory)
/// over a surface anchored first, by two buffers over it made one after another: the anchor
/// goes with the surface's last reference, at Metal's moment, and the import it held stays while
/// the command buffer that may read the texture's image runs, so the surface's memory stays
/// mapped until it completed, and is unmapped after
#[test]
fn a_churned_surface_textures_image_keeps_the_surfaces_memory() {
    super::surface_textures_image_keeps_the_surfaces_memory(2);
}

/// A surface over a single range of the guest's pages takes no anchor: its buffers lie over the
/// range itself, whose hold is an unmap hold an anchor would extend. Each release takes the
/// device's imports live back to what they were, and no anchor lives. A protected surface has
/// no base address to make a buffer at (`os::linux::surface`'s tests check that it takes none)
#[test]
fn single_range_surfaces_take_no_anchor() {
    let d = dev();
    let anchors = crate::os::anchors_live();
    let before = live(&counts(&d));
    let (s, pages, length) = bgra_surface_over_at(16, 16, 0);
    for round in 0..3 {
        drop(on_surface(&d, &s));
        assert!(within(&|| live(&counts(&d)) == before), "round {round}");
    }
    assert!(!crate::os::has_base_anchor(&s, id(&d)));
    assert_eq!(crate::os::anchors_live(), anchors);
    drop(s);
    free(pages, length);
}

/// Two threads that each make and release buffers over one surface at once make one anchor
/// between them: the surface keeps one, one more anchor lives, and with every buffer gone the
/// device holds one import more than before the surface. A buffer made first, before the
/// threads, lives throughout, so each of theirs shares its Metal range, as one made while
/// another's range is held must
#[test]
fn racing_buffers_make_one_anchor() {
    let d = dev();
    let anchors = crate::os::anchors_live();
    let before = live(&counts(&d));
    let s = bgra_surface(WIDTH, HEIGHT);
    let first = on_surface(&d, &s);
    std::thread::scope(|scope| {
        for _ in 0..2 {
            scope.spawn(|| churn(&d, &s, 50));
        }
    });
    assert!(crate::os::has_base_anchor(&s, id(&d)));
    assert_eq!(crate::os::anchors_live(), anchors + 1);
    drop(first);
    assert!(within(&|| live(&counts(&d)) == before + 1));
}

/// A remap reported over an anchored surface's base ends the join: a buffer made over it after
/// imports the pages there then, an import of its own beside the anchor's, which goes at its
/// release, and reads the surface's bytes now, as without an anchor
#[test]
fn a_remap_ends_the_anchors_join() {
    const BYTE: u8 = 0x5a;
    let d = dev();
    let s = bgra_surface(WIDTH, HEIGHT);
    churn(&d, &s, 2);
    assert!(crate::os::has_base_anchor(&s, id(&d)));
    let anchored = live(&counts(&d));
    fill_surface(&s, BYTE, LEN);
    crate::gpu::device::host_range_remapped(s.base_address(), s.alloc_size());
    let b = on_surface(&d, &s);
    assert_eq!(live(&counts(&d)), anchored + 1, "imported again");
    let (cb, out, _) = spin_then_copy(&d, &b, LEN);
    assert!(completes(&cb), "{:?}", cb.error());
    assert!(bytes_of(&out, LEN).iter().all(|&b| b == BYTE));
    drop((b, cb));
    assert!(within(&|| live(&counts(&d)) == anchored));
}
