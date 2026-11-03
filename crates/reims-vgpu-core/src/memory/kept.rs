//! The mappings a task's transfer buffers were made over, kept for the next buffer over the same
//! pages ([`KeptMappings`]).
//!
//! In the map setup a paged texture's, buffer's or heap's bind makes a no-copy buffer over a fresh
//! mapping of the task's pages each time (`buffer_for_page`, `new_buffer_for_virtual_page`): an
//! allocation, a remap of each run of the pages and a protect, then an unmap when the buffer
//! goes. On Linux that took about 2.7 µs a page in a Finder drag on macOS 26 (328 µs a bind on
//! average), and the drag's 1,860 binds covered 180 distinct runs of pages. So where the platform
//! lets it ([`HostVm::kept_mapping_limits`]), a task keeps the mappings it made last, by the
//! physical runs they show and their writability, and a bind whose page-table walk yields the same
//! runs and writability, of the same memory map, takes the kept mapping in place of a fresh one.
//!
//! What stays as it was:
//! - **The guest's memory**: the walk runs at every bind, so the page table is read as before,
//!   and a run or a page's writability the guest changed is another key, a fresh mapping.
//! - **What Metal and the guest see**: the buffer is made anew at each bind, through the same
//!   call, so Metal's space, the buffer's GPU address and its deallocator's moment are as before.
//!   Only its host address may repeat an earlier buffer's, which no answer of the API carries.
//! - **The pages**: the platform keeps mappings only where a fresh one shows nothing a kept one
//!   does not ([`HostVm::kept_mapping_limits`]: on Linux both are views of the guest RAM's files
//!   through the backing table's registrations, fixed while the device lives, so a page the host
//!   drops from a file shows in both alike).
//! - **A mapping whose protect failed** (deallocated, then made anyway, as Mach's caller does) is
//!   never kept, so its freed range is never reused.
//!
//! What a kept mapping holds is the platform's to bound: entries, bytes of address space and runs
//! (each at most a kernel mapping). Past any of them the least recently used go. All go at the
//! task's teardown ([`KeptMappings::clear`]), and on the platform's word before a view would pass
//! its budget of mappings ([`release_all`]), so a kept mapping never takes a mapping a fresh view
//! would need. Each goes as a released mapping does, its teardown the platform's VM's
//! deallocation, which the host never sees; a kept mapping is never one of the host's (the root
//! task's or a host task's).
//!
//! [`HostVm::kept_mapping_limits`]: crate::platform::os::vm::HostVm::kept_mapping_limits

use std::sync::{Arc, Weak};

use crate::host::PhysRange;
use crate::memory::map::MemoryMap;
use crate::memory::mapping::Mapping;
use crate::num::ToU64;
use crate::platform::Platform;
use crate::sync::Mutex;

/// How much a task's kept mappings may hold ([`HostVm::kept_mapping_limits`]).
///
/// [`HostVm::kept_mapping_limits`]: crate::platform::os::vm::HostVm::kept_mapping_limits
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeptLimits {
    /// Mappings.
    pub entries: usize,
    /// Bytes of address space, all mappings together.
    pub bytes: u64,
    /// Runs of physical pages, all mappings together: each is at most one kernel mapping.
    pub runs: u64,
}

impl KeptLimits {
    /// None kept: every bind makes a fresh mapping.
    pub const NONE: KeptLimits = KeptLimits {
        entries: 0,
        bytes: 0,
        runs: 0,
    };

    /// Whether a mapping of `bytes` and `runs` fits alone.
    fn fits(&self, bytes: u64, runs: u64) -> bool {
        self.entries != 0 && bytes <= self.bytes && runs <= self.runs
    }
}

/// One kept mapping and what it shows.
struct Kept<P: Platform> {
    /// The memory map it was made of.
    map: Weak<MemoryMap<P>>,
    /// The physical runs, in order.
    ranges: Vec<PhysRange>,
    /// Its length.
    length: u64,
    /// The memory map's `read_only` it was made with.
    read_only: u32,
    /// The mapping.
    mapping: Arc<Mapping<P>>,
}

impl<P: Platform> Kept<P> {
    /// Whether it is the mapping `map` would make of `ranges`.
    fn shows(
        &self,
        map: &Arc<MemoryMap<P>>,
        ranges: &[PhysRange],
        length: u64,
        read_only: u32,
    ) -> bool {
        self.length == length
            && self.read_only == read_only
            && self.ranges == ranges
            && std::ptr::eq(self.map.as_ptr(), Arc::as_ptr(map))
    }

    fn runs(&self) -> u64 {
        self.ranges.len().to_u64()
    }
}

/// The kept mappings, most recently used last, and their sums.
struct Entries<P: Platform> {
    kept: Vec<Kept<P>>,
    bytes: u64,
    runs: u64,
}

impl<P: Platform> Entries<P> {
    /// Takes out the least recently used until `limits` hold, answering them for the caller to
    /// let go outside the lock.
    fn trim(&mut self, limits: KeptLimits) -> Vec<Kept<P>> {
        let mut gone = Vec::new();
        while !self.kept.is_empty()
            && (self.kept.len() > limits.entries
                || self.bytes > limits.bytes
                || self.runs > limits.runs)
        {
            let oldest = self.kept.remove(0);
            self.bytes = self.bytes.wrapping_sub(oldest.length);
            self.runs = self.runs.wrapping_sub(oldest.runs());
            gone.push(oldest);
        }
        gone
    }
}

/// A task's kept mappings (module docs). Its lock is a leaf: no mapping is made or released under
/// it.
pub struct KeptMappings<P: Platform> {
    entries: Mutex<Entries<P>>,
}

/// What [`release_all`] reaches: every task's kept mappings.
trait Release: Send + Sync {
    /// Lets every kept mapping go.
    fn release(&self);
}

impl<P: Platform> Release for KeptMappings<P> {
    fn release(&self) {
        self.clear();
    }
}

/// Every live task's kept mappings, for [`release_all`].
static ALL: Mutex<Vec<Weak<dyn Release>>> = Mutex::new(Vec::new());

/// Lets every task's kept mappings go, as the platform asks before a view would pass its budget
/// of mappings (module docs). Under no lock of the caller's that a released mapping's teardown
/// takes.
pub fn release_all() {
    let live: Vec<Arc<dyn Release>> = ALL.lock().iter().filter_map(Weak::upgrade).collect();
    for kept in live {
        kept.release();
    }
}

impl<P: Platform> KeptMappings<P> {
    /// None kept yet, reached by [`release_all`] while it lives.
    pub fn new() -> Arc<Self> {
        let kept = Arc::new(KeptMappings {
            entries: Mutex::new(Entries {
                kept: Vec::new(),
                bytes: 0,
                runs: 0,
            }),
        });
        let weak = Arc::downgrade(&kept);
        let weak: Weak<dyn Release> = weak;
        let mut all = ALL.lock();
        all.retain(|w| w.strong_count() != 0);
        all.push(weak);
        kept
    }

    /// The mapping `map` makes of `ranges` (`MemoryMap::new_range_mapping`), or the one kept of
    /// the same runs, length and writability of the same map, which then is the most recently
    /// used. A fresh one is kept where the platform's limits let it and its protect held; the
    /// least recently used go past them.
    pub fn mapping(
        &self,
        map: &Arc<MemoryMap<P>>,
        ranges: &[PhysRange],
        length: u64,
        read_only: u32,
    ) -> Option<Arc<Mapping<P>>> {
        let limits = map.kept_limits();
        let runs = ranges.len().to_u64();
        if !limits.fits(length, runs) {
            return map.new_range_mapping(ranges, length, read_only);
        }
        let hit = {
            let mut entries = self.entries.lock();
            let at = (entries.kept.iter()).position(|k| k.shows(map, ranges, length, read_only));
            at.map(|at| {
                let hit = entries.kept.remove(at);
                let mapping = Arc::clone(&hit.mapping);
                entries.kept.push(hit);
                mapping
            })
        };
        // kmap: whether a transfer buffer's mapping was a kept one, its length
        crate::trace::event("kmap", &[u64::from(hit.is_some()), length]);
        if hit.is_some() {
            return hit;
        }
        let (mapping, protected) = map.range_mapping(ranges, length, read_only)?;
        if !protected {
            return Some(mapping);
        }
        let gone = {
            let mut entries = self.entries.lock();
            entries.bytes = entries.bytes.wrapping_add(length);
            entries.runs = entries.runs.wrapping_add(runs);
            entries.kept.push(Kept {
                map: Arc::downgrade(map),
                ranges: ranges.to_vec(),
                length,
                read_only,
                mapping: Arc::clone(&mapping),
            });
            entries.trim(limits)
        };
        drop(gone);
        Some(mapping)
    }

    /// Lets every kept mapping go, outside the lock: the task's teardown.
    pub fn clear(&self) {
        let gone = {
            let mut entries = self.entries.lock();
            entries.bytes = 0;
            entries.runs = 0;
            std::mem::take(&mut entries.kept)
        };
        drop(gone);
    }

    /// How many mappings are kept, for tests.
    #[cfg(test)]
    pub fn kept(&self) -> usize {
        self.entries.lock().kept.len()
    }
}
