//! The per-kind caches, the lookups over them, and the address and pipeline registries.
//!
//! A lookup of a cached reference reads its cache under the leaf lock only; a miss takes the
//! recursive creation lock, reads the guest's entry and record, and has the kind make the
//! object. A creator may look up other references (re-entering the lock) and is never checked
//! for cycles.

use std::sync::Arc;

use crate::collections::{ListOrderMap, LookupMap};
use crate::guest::GuestPtr;
use crate::guest::records::ObjectEntry;
use crate::host::HostPtr;
use crate::memory::cursor::Cursor;
use crate::platform::Platform;
use crate::raise::RaiseResult;
use crate::resources::manager::create::Resources;
use crate::resources::manager::records::word;
use crate::resources::manager::states::{
    ComputePipelineStates, DepthStencilStates, Fences, Functions, RasterizationRateMaps,
    RenderPipelineStates, SamplerStates,
};
use crate::resources::manager::{DeathKind, ResourceManager};
use crate::resources::resource::AnyResource;
use crate::resources::resource::Resource;
use crate::rt::Hold;

/// One kind's cached objects, keyed by reference, in list order (the order they release in).
pub struct Cache<T> {
    objects: ListOrderMap<T>,
}

impl<T> Default for Cache<T> {
    fn default() -> Self {
        Cache {
            objects: ListOrderMap::new(),
        }
    }
}

impl<T> Cache<T> {
    /// The object cached for `reference`.
    pub fn get(&self, reference: u32) -> Option<&T> {
        self.objects.get(reference)
    }

    /// Caches `object` for `reference`, or hands it back if the reference is taken.
    pub fn try_insert(&mut self, reference: u32, object: T) -> Result<(), T> {
        self.objects.try_insert(reference, object)
    }

    /// Takes the object cached for `reference` out, for the caller to release.
    pub fn remove(&mut self, reference: u32) -> Option<T> {
        self.objects.remove(reference)
    }

    /// Takes every object out, in list order, for the caller to release.
    pub fn drain(&mut self) -> Vec<T> {
        self.objects.drain()
    }

    /// The references and objects, in list order.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &T)> {
        self.objects.iter()
    }
}

/// The manager's caches and registries, reached only under the leaf lock in short closures that
/// never call out (a nested creation, a Metal call, a task call or a release can re-enter).
/// Each [`ManagedKind`] names its cache.
pub struct Caches<P: Platform> {
    /// The sampler states.
    pub sampler_states: Cache<P::SamplerState>,
    /// The depth-stencil states.
    pub depth_stencil_states: Cache<P::DepthStencilState>,
    /// The render pipeline states.
    pub render_pipeline_states: Cache<P::RenderPipeline>,
    /// The compute pipeline states.
    pub compute_pipeline_states: Cache<P::ComputePipeline>,
    /// The fences.
    pub fences: Cache<P::Fence>,
    /// The rasterization rate maps.
    pub rasterization_rate_maps: Cache<P::RateMap>,
    /// The functions.
    pub functions: Cache<P::Function>,
    /// The resources: buffers, heaps, ICBs and textures.
    pub resources: Cache<Arc<Resource<P>>>,
    /// Registered GPU addresses and their references, looked up by the greatest address at or
    /// below a given one.
    gpu_addresses: GpuAddresses,
    /// Registered pipeline identifiers and their references.
    pipelines: LookupMap<u64, u32>,
}

/// The GPU-address registry: its only query is the greatest registered address at or below an
/// address, so its order is never observed as an iteration.
#[derive(Default)]
struct GpuAddresses(
    #[allow(clippy::disallowed_types)] // lookup-only: the floor query (range, next_back)
    std::collections::BTreeMap<u64, u32>,
);

impl<P: Platform> Caches<P> {
    /// Every cache and registry empty.
    pub(crate) fn new() -> Self {
        Caches {
            sampler_states: Cache::default(),
            depth_stencil_states: Cache::default(),
            render_pipeline_states: Cache::default(),
            compute_pipeline_states: Cache::default(),
            fences: Cache::default(),
            rasterization_rate_maps: Cache::default(),
            functions: Cache::default(),
            resources: Cache::default(),
            gpu_addresses: GpuAddresses::default(),
            pipelines: LookupMap::new(),
        }
    }

    /// Empties the registries (the manager's teardown).
    pub(crate) fn clear_registries(&mut self) {
        self.pipelines.clear();
        self.gpu_addresses.0.clear();
    }
}

/// One kind of object the manager caches. Each kind's module implements it for its marker type,
/// so the lookups stay one generic body and the creators stay with their kind.
pub trait ManagedKind<P: Platform>: 'static {
    /// The object.
    type Obj: Send + 'static;
    /// The kind's cache.
    fn sub_manager(c: &Caches<P>) -> &Cache<Self::Obj>;
    /// The kind's cache, mutably.
    fn sub_manager_mut(c: &mut Caches<P>) -> &mut Cache<Self::Obj>;
    /// Another handle to the same object.
    fn retain(o: &Self::Obj) -> Self::Obj;
    /// Makes the object `entry` and `cursor` (its record) describe, switching on the record's
    /// operation header.
    ///
    /// `Obj` cannot be nil, so a creator that would succeed with none calls [`nil_object`].
    fn create_object(
        hold: Hold<'_>,
        m: &ResourceManager<P>,
        reference: u32,
        entry: &ObjectEntry,
        cursor: Cursor,
        placement: Option<GuestPtr>,
    ) -> RaiseResult<Result<Self::Obj, String>>;
}

/// A creator succeeded with no object: the registration's assert.
pub fn nil_object() -> ! {
    crate::fatal!(DeathKind::NilObject)
}

/// The serialized delete's operation codes and what each deletes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Deletion {
    /// 1000, 1001, 1012: a resource.
    Resource,
    /// 1002.
    DepthStencil,
    /// 1003.
    Sampler,
    /// 1005.
    Function,
    /// 1006.
    ComputePipeline,
    /// 1007.
    RenderPipeline,
    /// 1009.
    Fence,
    /// 1014.
    RateMap,
}

impl Deletion {
    fn of(op: u32) -> Option<Deletion> {
        Some(match op {
            1000 | 1001 | 1012 => Deletion::Resource,
            1002 => Deletion::DepthStencil,
            1003 => Deletion::Sampler,
            1005 => Deletion::Function,
            1006 => Deletion::ComputePipeline,
            1007 => Deletion::RenderPipeline,
            1009 => Deletion::Fence,
            1014 => Deletion::RateMap,
            _ => return None,
        })
    }
}

impl<P: Platform> ResourceManager<P> {
    /// Runs `f` on the caches under the leaf lock only, never the creation lock, so it never
    /// waits for another thread's creation. `f` must not call out.
    pub(crate) fn with_caches<R>(&self, f: impl FnOnce(&mut Caches<P>) -> R) -> R {
        let mut caches = self.caches.lock();
        f(&mut caches)
    }

    /// Another handle to the object cached for `reference`, if there is one.
    fn probe<K: ManagedKind<P>>(&self, reference: u32) -> Option<K::Obj> {
        self.with_caches(|c| K::sub_manager(c).get(reference).map(K::retain))
    }

    /// The object for `reference`, made on a miss; reference 0 is `Ok(None)`. The caller owns
    /// the returned handle, which keeps the object alive past a delete while it holds it.
    pub fn get_object<K: ManagedKind<P>>(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Result<Option<K::Obj>, String>> {
        // Reference 0 is Ok(nil) for every kind, and flows on.
        if reference == 0 {
            return Ok(Ok(None));
        }
        if let Some(object) = self.probe::<K>(reference) {
            return Ok(Ok(Some(object)));
        }
        self.new_object::<K>(hold, reference)
    }

    /// The cached object for `reference`, never made: `None` when nothing is cached, and
    /// `Some(None)` for reference 0.
    pub fn get_existing_object<K: ManagedKind<P>>(&self, reference: u32) -> Option<Option<K::Obj>> {
        if reference == 0 {
            return Some(None);
        }
        Some(Some(self.probe::<K>(reference)?))
    }

    /// The resource's texture: a resource that has none raises, and reference 0 is nil.
    pub fn get_object_texture(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Result<Option<P::Texture>, String>> {
        self.resource_view(hold, reference, |r| r.get_texture())
    }

    /// The resource's buffer: a resource that has none raises, and reference 0 is nil.
    pub fn get_object_buffer(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Result<Option<P::Buffer>, String>> {
        self.resource_view(hold, reference, |r| r.get_buffer())
    }

    /// The resource's heap: a resource that has none raises, and reference 0 is nil.
    pub fn get_object_heap(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Result<Option<P::Heap>, String>> {
        self.resource_view(hold, reference, |r| r.get_heap())
    }

    /// The resource's Metal resource; reference 0 is nil.
    pub fn get_object_resource(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Result<Option<AnyResource<P>>, String>> {
        self.resource_view(hold, reference, |r| r.get_resource())
    }

    /// The resource's indirect command buffer: a resource that has none raises, and reference 0
    /// is nil.
    pub fn get_object_icb(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Result<Option<P::Icb>, String>> {
        self.resource_view(hold, reference, |r| r.get_indirect_command_buffer())
    }

    /// One view of the resource for `reference`; the resource's handle drops after the view is
    /// taken.
    fn resource_view<T>(
        &self,
        hold: Hold<'_>,
        reference: u32,
        view: impl FnOnce(&Resource<P>) -> RaiseResult<Option<T>>,
    ) -> RaiseResult<Result<Option<T>, String>> {
        match self.get_object::<Resources>(hold, reference)? {
            Ok(resource) => Ok(Ok(resource.as_deref().map_or(Ok(None), view)?)),
            Err(e) => Ok(Err(e)),
        }
    }

    /// Makes and caches the object for `reference` under the creation lock, unless another
    /// creation cached it first.
    ///
    /// The object returned is the creator's handle (the cache holds its own). A raise from the
    /// cursor, the placement or the creator leaves with nothing cached.
    pub fn new_object<K: ManagedKind<P>>(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Result<Option<K::Obj>, String>> {
        let _creation = self.creation.lock();
        if let Some(object) = self.probe::<K>(reference) {
            return Ok(Ok(Some(object)));
        }
        let entry = match self.object_lists.get_object_entry(reference) {
            Ok(entry) => entry,
            Err(e) => return Ok(Err(e)),
        };
        if entry.object_type() == 0 {
            return Ok(Err(format!("object {reference} has an empty entry")));
        }
        let offset = entry.offset;
        let Some(cursor) = self.new_memory_cursor(hold, offset, u64::from(entry.length()))? else {
            return Ok(Err(format!("no cursor over object {reference}'s record")));
        };
        let placement = match self.object_lists.get_placement_list_entry(reference) {
            Ok(placement) => placement,
            Err(e) => return Ok(Err(format!("no placement for object {reference}: {e}"))),
        };
        // No cycle detection. A creator that looks its own reference up again re-enters this lock
        // and recurses until the stack overflows.
        // The entry and record are read now only; a cached object never reads them again.
        let made = crate::trace::span_start();
        let object = K::create_object(hold, self, reference, &entry, cursor, placement)?;
        // 63: the creation: the reference, the object type
        let object_type = u64::from(entry.object_type());
        crate::trace::span(63, made, &[u64::from(reference), object_type]);
        let object = match object {
            Ok(object) => object,
            Err(e) => return Ok(Err(e)),
        };
        if !self.register_object::<K>(reference, K::retain(&object)) {
            drop(object);
            return Ok(Err(format!("object {reference} already cached")));
        }
        Ok(Ok(Some(object)))
    }

    /// Caches `object` (the cache's own handle) for `reference`. A reference already cached
    /// keeps its object, and `object` is released outside the lock.
    pub fn register_object<K: ManagedKind<P>>(&self, reference: u32, object: K::Obj) -> bool {
        let rejected = self.with_caches(|c| K::sub_manager_mut(c).try_insert(reference, object));
        match rejected {
            Ok(()) => true,
            Err(object) => {
                drop(object);
                false
            }
        }
    }

    /// Releases the object cached for `reference` under the creation lock, then clears its
    /// placement slot if asked.
    pub fn delete_object<K: ManagedKind<P>>(
        &self,
        hold: Hold<'_>,
        reference: u32,
        clear_placement: bool,
    ) {
        let _creation = self.creation.lock();
        let removed = self.with_caches(|c| K::sub_manager_mut(c).remove(reference));
        drop(removed);
        if clear_placement {
            self.clear_placement_list_entry(hold, reference);
        }
    }

    /// [`ResourceManager::delete_object`] of the resource `reference`, clearing its placement
    /// slot, but a resource whose last release reaches neither the host nor anything it sees
    /// ([`Resource::releases_internally`]) comes back, released by the caller after its own
    /// locks. Any other is released here, before the slot's clear, as `delete_object` does.
    pub fn delete_resource(&self, hold: Hold<'_>, reference: u32) -> Option<Arc<Resource<P>>> {
        let _creation = self.creation.lock();
        let removed = self.with_caches(|c| c.resources.remove(reference));
        let internal = match removed {
            Some(resource) if resource.releases_internally() => Some(resource),
            other => {
                drop(other);
                None
            }
        };
        self.clear_placement_list_entry(hold, reference);
        internal
    }

    /// The guest's serialized delete (opcode 40): the operation word at 0 and the reference at
    /// 8. A short command, or an operation that deletes nothing, is `false`.
    pub fn delete_serialized_object(&self, hold: Hold<'_>, command: &[u8]) -> bool {
        self.delete_serialized(hold, command).0
    }

    /// [`ResourceManager::delete_serialized_object`], and a resource it deleted whose release is
    /// the caller's ([`ResourceManager::delete_resource`]).
    pub fn delete_serialized(
        &self,
        hold: Hold<'_>,
        command: &[u8],
    ) -> (bool, Option<Arc<Resource<P>>>) {
        let (Some(op), Some(reference)) = (word(command, 0), word(command, 8)) else {
            return (false, None);
        };
        let Some(deletion) = Deletion::of(op) else {
            log::warn!("delete of object {reference} ignored: unknown operation {op}");
            return (false, None);
        };
        match deletion {
            Deletion::Resource => return (true, self.delete_resource(hold, reference)),
            Deletion::DepthStencil => {
                self.delete_object::<DepthStencilStates>(hold, reference, true)
            }
            Deletion::Sampler => self.delete_object::<SamplerStates>(hold, reference, true),
            Deletion::Function => self.delete_object::<Functions>(hold, reference, true),
            Deletion::ComputePipeline => {
                self.delete_object::<ComputePipelineStates>(hold, reference, true)
            }
            Deletion::RenderPipeline => {
                self.delete_object::<RenderPipelineStates>(hold, reference, true)
            }
            Deletion::Fence => self.delete_object::<Fences>(hold, reference, true),
            Deletion::RateMap => self.delete_object::<RasterizationRateMaps>(hold, reference, true),
        }
        (true, None)
    }

    /// Zeroes the placement slot of `reference`, mapping the placement list first (under the
    /// object lists' mutex, inside the creation lock) if it is not mapped. An unmappable list,
    /// no list or a reference out of range writes nothing.
    pub fn clear_placement_list_entry(&self, hold: Hold<'_>, reference: u32) {
        let lists = &self.object_lists;
        if !lists.is_placement_list_mapped() {
            // A caller holding the lists' mutex deadlocks here.
            let _lock = lists.mutex.lock();
            if !lists.map_placement_list(hold) {
                return;
            }
        }
        if lists.placement_list_length() == 0 {
            return;
        }
        if let Ok(slot) = lists.get_placement_list_entry(reference) {
            // Guest-visible: one 16-byte store (`stp xzr, xzr`), single-copy atomic for the
            // 16-aligned slot, so a guest's 16-byte load never sees half of it cleared
            slot.unwrap_or(GuestPtr::new(HostPtr::NULL))
                .write_zero_16(0, DeathKind::NullPlacement);
        }
    }

    /// Registers `address` for `reference`; the newest registration of an address wins.
    pub fn register_gpu_address(&self, address: u64, reference: u32) -> Result<(), String> {
        self.with_caches(|c| {
            c.gpu_addresses.0.remove(&address);
            c.gpu_addresses.0.insert(address, reference);
        });
        Ok(())
    }

    /// Registers the pipeline `identifier` for `reference`; the newest registration wins.
    pub fn register_pipeline(&self, identifier: u64, reference: u32) -> Result<(), String> {
        self.with_caches(|c| {
            c.pipelines.remove(&identifier);
            c.pipelines.insert(identifier, reference);
        });
        Ok(())
    }

    /// The reference registered for the pipeline `identifier`; all ones is never valid.
    pub fn get_pipeline_ref(&self, identifier: u64) -> Result<u32, String> {
        if identifier == u64::MAX {
            return Err(format!("pipeline identifier {identifier:#x} is not valid"));
        }
        self.with_caches(|c| c.pipelines.get(&identifier).copied())
            .ok_or_else(|| format!("no pipeline registered for identifier {identifier:#x}"))
    }

    /// The greatest registered address at or below `address`, and its reference.
    pub fn gpu_address_at_or_below(&self, address: u64) -> Option<(u64, u32)> {
        self.with_caches(|c| {
            c.gpu_addresses
                .0
                .range(..=address)
                .next_back()
                .map(|(a, r)| (*a, *r))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixture::*;
    use super::*;
    use crate::platform::Handle;
    use crate::resources::texture::views::SerializerTextureResource;
    use crate::tasks::object_lists::ObjectListKind;
    use crate::testkit::calllog::Arg;
    use crate::testkit::capture::catch_death;
    use crate::testkit::{Fk, Step, kind};
    use zerocopy::IntoBytes;

    fn fence(fx: &Fx) -> Fk<kind::Fence> {
        fx.platform
            .world()
            .new_obj::<kind::Fence>(Default::default())
    }

    fn texture_resource(fx: &Fx) -> Arc<Resource<P>> {
        let t = fx
            .platform
            .world()
            .new_obj::<kind::Texture>(Default::default());
        SerializerTextureResource::new(t, false)
    }

    fn entries3() -> Vec<(u8, u32, u64)> {
        vec![(0, 0, 0), (6, 0x20, REC), (6, 0x10, REC + 0x100)]
    }

    mod get_object {
        use super::*;

        /// Reference 0 is Ok(nil), for every kind.
        #[test]
        fn reference_zero_is_nil_without_touching_the_list() {
            let fx = Fx::new(&entries3(), Placement::Off);
            let mark = fx.clear_log();
            assert!(matches!(
                fx.m.get_object::<TestKind>(Hold::NONE, 0),
                Ok(Ok(None))
            ));
            assert!(fx.sels_since(mark).is_empty());
        }

        #[test]
        fn a_miss_creates_and_a_hit_returns_the_cached_object() {
            let fx = Fx::new(&entries3(), Placement::Off);
            fx.next_cursor(0, 0x20);
            let a =
                fx.m.get_object::<TestKind>(Hold::NONE, 1)
                    .unwrap()
                    .unwrap()
                    .unwrap();
            let b =
                fx.m.get_object::<TestKind>(Hold::NONE, 1)
                    .unwrap()
                    .unwrap()
                    .unwrap();
            assert_eq!(a.id(), b.id());
            // The cache's and the two handed out.
            assert_eq!(a.obj().handle_count(), 3);
            // Created once, from entry 1 and its cursor.
            assert_eq!(seen().len(), 1);
            assert_eq!(fx.log().calls_to("newFence").len(), 1);
            assert_eq!(
                fx.log().calls_to("cursor_from_virtual_offset")[0].args,
                vec![Arg::U64(REC), Arg::U64(0x20)]
            );
        }

        #[test]
        fn an_error_passes_through_verbatim() {
            let fx = Fx::new(&entries3(), Placement::Off);
            let r = fx.m.get_object::<TestKind>(Hold::NONE, 7).unwrap();
            assert_eq!(
                r.err().unwrap(),
                "reference 7 is past the object list's 3 entries"
            );
        }

        /// A handed-out object outlives its delete while its holder holds it
        #[test]
        fn a_handed_out_object_outlives_its_delete_while_held() {
            let fx = Fx::new(&entries3(), Placement::Off);
            fx.next_cursor(0, 0x20);
            let a =
                fx.m.get_object::<TestKind>(Hold::NONE, 1)
                    .unwrap()
                    .unwrap()
                    .unwrap();
            let obj = a.obj().clone();
            fx.m.delete_object::<TestKind>(Hold::NONE, 1, false);
            assert_eq!(obj.handle_count(), 1);
            drop(a);
            assert_eq!(obj.handle_count(), 0);
        }

        #[test]
        fn get_existing_object_zero_absent_and_present() {
            let fx = Fx::new(&entries3(), Placement::Off);
            assert!(matches!(
                fx.m.get_existing_object::<TestKind>(0),
                Some(None)
            ));
            assert!(fx.m.get_existing_object::<TestKind>(1).is_none());
            // Never creates.
            assert!(seen().is_empty());
            let f = fence(&fx);
            assert!(fx.m.register_object::<TestKind>(1, f.clone_ref()));
            let got = fx.m.get_existing_object::<TestKind>(1).unwrap().unwrap();
            assert_eq!(got.id(), f.id());
        }

        /// A view of a resource that has none raises its own unrecognized operation; one of
        /// reference 0 is nil with nothing asked.
        #[test]
        fn views_of_a_resource() {
            let fx = Fx::new(&entries3(), Placement::Off);
            assert!(fx.m.register_object::<Resources>(1, texture_resource(&fx)));
            assert!(
                fx.m.get_object_texture(Hold::NONE, 1)
                    .unwrap()
                    .unwrap()
                    .is_some()
            );
            assert!(
                fx.m.get_object_texture(Hold::NONE, 0)
                    .unwrap()
                    .unwrap()
                    .is_none()
            );
            let hold = Hold::NONE;
            assert!(fx.m.get_object_heap(hold, 1).is_err());
        }
    }

    mod new_object {
        use super::*;

        #[test]
        fn entry_error_is_returned_verbatim() {
            let fx = Fx::new(&entries3(), Placement::Off);
            fx.olm.unmap_object_list();
            let r = fx.m.new_object::<TestKind>(Hold::NONE, 1).unwrap();
            assert_eq!(r.err().unwrap(), "the object list is not mapped");
        }

        #[test]
        fn an_invalid_entry_is_refused_before_the_cursor() {
            let fx = Fx::new(&entries3(), Placement::Off);
            let mark = fx.clear_log();
            let r = fx.m.new_object::<TestKind>(Hold::NONE, 0).unwrap();
            assert_eq!(r.err().unwrap(), "object 0 has an empty entry");
            assert!(fx.sels_since(mark).is_empty());
        }

        #[test]
        fn a_null_cursor_fails() {
            let fx = Fx::new(&entries3(), Placement::Off);
            let r = fx.m.new_object::<TestKind>(Hold::NONE, 2).unwrap();
            assert_eq!(r.err().unwrap(), "no cursor over object 2's record");
            assert_eq!(
                fx.log().calls_to("cursor_from_virtual_offset")[0].args,
                vec![Arg::U64(REC + 0x100), Arg::U64(0x10)]
            );
            assert!(seen().is_empty());
        }

        #[test]
        fn a_cursor_throw_propagates_with_nothing_inserted() {
            let fx = Fx::new(&entries3(), Placement::Off);
            fx.platform.script().push(
                "cursor_from_virtual_offset",
                Step::raise("NSGenericException", "x"),
            );
            let hold = Hold::NONE;
            assert!(fx.m.new_object::<TestKind>(hold, 1).is_err());
            assert!(fx.m.get_existing_object::<TestKind>(1).is_none());
        }

        #[test]
        fn a_placement_error_is_prefixed() {
            let fx = Fx::new(&entries3(), Placement::Unmapped);
            fx.next_cursor(0, 0x20);
            let r = fx.m.new_object::<TestKind>(Hold::NONE, 1).unwrap();
            assert_eq!(
                r.err().unwrap(),
                "no placement for object 1: the placement list is not mapped"
            );
            assert!(seen().is_empty());
        }

        #[test]
        fn the_creator_gets_entry_cursor_and_placement() {
            let fx = Fx::new(&entries3(), Placement::Mapped);
            fx.next_cursor(0, 0x20);
            let obj =
                fx.m.new_object::<TestKind>(Hold::NONE, 2)
                    .unwrap()
                    .unwrap()
                    .unwrap();
            assert_eq!(
                seen(),
                vec![Seen {
                    reference: 2,
                    entry_type: 6,
                    cursor_len: 0x20,
                    placement: Some(fx.addr(PL + 0x20).addr()),
                }]
            );
            // Registered: the cache holds one handle, the caller the creator's.
            assert_eq!(obj.obj().handle_count(), 2);
            let cached = fx.m.get_existing_object::<TestKind>(2).unwrap().unwrap();
            assert_eq!(cached.id(), obj.id());
        }

        #[test]
        fn placement_is_null_without_argument_buffers() {
            let fx = Fx::new(&entries3(), Placement::Off);
            fx.next_cursor(0, 0x20);
            assert!(fx.m.new_object::<TestKind>(Hold::NONE, 1).unwrap().is_ok());
            assert_eq!(seen()[0].placement, None);
        }

        #[test]
        fn a_creator_error_is_verbatim_and_nothing_is_inserted() {
            let fx = Fx::new(&entries3(), Placement::Off);
            fx.next_cursor(0, 0x20);
            script_create(Create::Fail("unknown command type 9"));
            let r = fx.m.new_object::<TestKind>(Hold::NONE, 1).unwrap();
            assert_eq!(r.err().unwrap(), "unknown command type 9");
            assert!(fx.m.get_existing_object::<TestKind>(1).is_none());
        }

        #[test]
        fn a_creator_throw_propagates_and_the_lock_is_released() {
            let fx = Fx::new(&entries3(), Placement::Off);
            fx.next_cursor(0, 0x20);
            script_create(Create::Throw);
            {
                {
                    let hold = Hold::NONE;
                    assert!(fx.m.new_object::<TestKind>(hold, 1).is_err());
                }
            }
            assert!(fx.m.get_existing_object::<TestKind>(1).is_none());
            // Another thread can take the lock: it was released on the way out.
            let m = fx.m.clone();
            #[allow(clippy::disallowed_methods)] // test scaffolding: a second thread
            let t = std::thread::spawn(move || {
                let _creation = m.creation.lock();
                m.get_existing_object::<TestKind>(1).is_none()
            });
            assert!(t.join().unwrap());
        }

        /// The re-entrant creation: a creator's lookup of another reference takes the creation
        /// lock again on the same thread.
        #[test]
        fn a_nested_creation_reenters_the_lock() {
            let fx = Fx::new(&entries3(), Placement::Off);
            fx.next_cursor(0, 0x20);
            fx.next_cursor(0x100, 0x10);
            script_create(Create::Nested(2));
            let a =
                fx.m.new_object::<TestKind>(Hold::NONE, 1)
                    .unwrap()
                    .unwrap()
                    .unwrap();
            // The inner reference was created first, inside the outer creation.
            let refs: Vec<u32> = seen().iter().map(|s| s.reference).collect();
            assert_eq!(refs, vec![1, 2]);
            assert!(fx.m.get_existing_object::<TestKind>(2).unwrap().is_some());
            assert_eq!(
                fx.m.get_existing_object::<TestKind>(1)
                    .unwrap()
                    .unwrap()
                    .id(),
                a.id()
            );
        }

        #[test]
        fn a_duplicate_registration_fails_and_releases_the_new_object() {
            let fx = Fx::new(&entries3(), Placement::Off);
            fx.next_cursor(0, 0x20);
            script_create(Create::Duplicate);
            let r = fx.m.new_object::<TestKind>(Hold::NONE, 1).unwrap();
            assert_eq!(r.err().unwrap(), "object 1 already cached");
            // The object registered inside the creation stays.
            let fences = fx.log().calls_to("newFence");
            assert_eq!(fences.len(), 2);
            let kept = fx.m.get_existing_object::<TestKind>(1).unwrap().unwrap();
            let live: Vec<_> = fx.platform.world().find(kept.id()).into_iter().collect();
            assert_eq!(live.len(), 1);
        }

        /// A creation that looks its own reference up recurses (here once, then the inner creation
        /// ends it); the outer registration then finds the key taken.
        #[test]
        fn a_self_reference_recurses_without_detection() {
            let fx = Fx::new(&entries3(), Placement::Off);
            fx.next_cursor(0, 0x20);
            fx.next_cursor(0, 0x20);
            script_create(Create::Nested(1));
            let r = fx.m.new_object::<TestKind>(Hold::NONE, 1).unwrap();
            let refs: Vec<u32> = seen().iter().map(|s| s.reference).collect();
            assert_eq!(refs, vec![1, 1]);
            assert_eq!(r.err().unwrap(), "object 1 already cached");
            assert!(fx.m.get_existing_object::<TestKind>(1).unwrap().is_some());
        }

        /// A cached object is never rebuilt from a changed entry; after a delete the next lookup
        /// reads the entry as it is then.
        #[test]
        fn a_live_object_never_rereads_its_entry() {
            let fx = Fx::new(&entries3(), Placement::Off);
            fx.next_cursor(0, 0x20);
            let a =
                fx.m.get_object::<TestKind>(Hold::NONE, 1)
                    .unwrap()
                    .unwrap()
                    .unwrap();
            // The guest retypes entry 1 (type 7, record at REC + 0x40).
            let e = ObjectEntry {
                type_and_length: 7 | (0x30 << 8),
                offset: REC + 0x40,
            };
            fx.write(OL + 12, e.as_bytes());
            let b =
                fx.m.get_object::<TestKind>(Hold::NONE, 1)
                    .unwrap()
                    .unwrap()
                    .unwrap();
            assert_eq!(a.id(), b.id());
            assert_eq!(seen().len(), 1);
            fx.m.delete_object::<TestKind>(Hold::NONE, 1, false);
            fx.next_cursor(0x40, 0x30);
            let c =
                fx.m.get_object::<TestKind>(Hold::NONE, 1)
                    .unwrap()
                    .unwrap()
                    .unwrap();
            assert_ne!(c.id(), a.id());
            assert_eq!(seen()[1].entry_type, 7);
            assert_eq!(seen()[1].cursor_len, 0x30);
        }

        #[test]
        fn a_cached_object_is_returned_under_the_lock_without_creation() {
            let fx = Fx::new(&entries3(), Placement::Off);
            let f = fence(&fx);
            assert!(fx.m.register_object::<TestKind>(1, f.clone_ref()));
            let mark = fx.clear_log();
            let got =
                fx.m.new_object::<TestKind>(Hold::NONE, 1)
                    .unwrap()
                    .unwrap()
                    .unwrap();
            assert_eq!(got.id(), f.id());
            assert!(fx.sels_since(mark).is_empty());
        }
    }

    mod register_object {
        use super::*;

        #[test]
        fn first_wins_and_a_duplicate_is_released() {
            let fx = Fx::new(&entries3(), Placement::Off);
            let a = fence(&fx);
            let b = fence(&fx);
            assert!(fx.m.register_object::<TestKind>(4, a.clone_ref()));
            assert!(!fx.m.register_object::<TestKind>(4, b.clone_ref()));
            assert_eq!(b.obj().handle_count(), 1);
            assert_eq!(a.obj().handle_count(), 2);
            let got = fx.m.get_existing_object::<TestKind>(4).unwrap().unwrap();
            assert_eq!(got.id(), a.id());
        }

        #[test]
        fn nil_object_is_the_assert() {
            assert_eq!(
                catch_death(|| nil_object()),
                Err(DeathKind::NilObject.site())
            );
        }

        /// One cache per kind; a delete of one kind leaves the others.
        #[test]
        fn one_reference_in_several_caches() {
            let fx = Fx::new(&entries3(), Placement::Mapped);
            let s = fx
                .platform
                .world()
                .new_obj::<kind::SamplerState>(Default::default());
            let d = fx
                .platform
                .world()
                .new_obj::<kind::DepthStencilState>(Default::default());
            assert!(fx.m.register_object::<SamplerStates>(1, s.clone_ref()));
            assert!(fx.m.register_object::<DepthStencilStates>(1, d.clone_ref()));
            assert!(fx.m.register_object::<Resources>(1, texture_resource(&fx)));
            // 1006 (compute) on this reference forgets nothing it holds.
            let mut cmd = [0u8; 12];
            cmd[0..4].copy_from_slice(&1006u32.to_le_bytes());
            cmd[8..12].copy_from_slice(&1u32.to_le_bytes());
            assert!(fx.m.delete_serialized_object(Hold::NONE, &cmd));
            assert!(
                fx.m.get_existing_object::<SamplerStates>(1)
                    .unwrap()
                    .is_some()
            );
            assert!(
                fx.m.get_existing_object::<DepthStencilStates>(1)
                    .unwrap()
                    .is_some()
            );
            assert!(fx.m.get_existing_object::<Resources>(1).unwrap().is_some());
            // 1003 (sampler) forgets only the sampler.
            cmd[0..4].copy_from_slice(&1003u32.to_le_bytes());
            assert!(fx.m.delete_serialized_object(Hold::NONE, &cmd));
            assert!(fx.m.get_existing_object::<SamplerStates>(1).is_none());
            assert!(
                fx.m.get_existing_object::<DepthStencilStates>(1)
                    .unwrap()
                    .is_some()
            );
        }
    }

    mod delete_object {
        use super::*;

        #[test]
        fn erases_and_releases_without_clearing() {
            let fx = Fx::new(&entries3(), Placement::Mapped);
            fx.set_placement(1, 0x1234, 0x5678);
            let f = fence(&fx);
            assert!(fx.m.register_object::<TestKind>(1, f.clone_ref()));
            fx.m.delete_object::<TestKind>(Hold::NONE, 1, false);
            assert!(fx.m.get_existing_object::<TestKind>(1).is_none());
            assert_eq!(f.obj().handle_count(), 1);
            assert_eq!(fx.placement(1), (0x1234, 0x5678));
        }

        /// A delete that clears zeroes all 16 bytes.
        #[test]
        fn clears_the_placement_even_when_nothing_was_cached() {
            let fx = Fx::new(&entries3(), Placement::Mapped);
            fx.set_placement(2, 0x1234, 0x5678);
            fx.set_placement(1, 7, 8);
            fx.m.delete_object::<TestKind>(Hold::NONE, 2, true);
            assert_eq!(fx.placement(2), (0, 0));
            assert_eq!(fx.placement(1), (7, 8));
        }

        #[test]
        fn reference_zero_is_not_special() {
            let fx = Fx::new(&entries3(), Placement::Mapped);
            fx.set_placement(0, 9, 9);
            fx.m.delete_object::<TestKind>(Hold::NONE, 0, true);
            assert_eq!(fx.placement(0), (0, 0));
        }
    }

    mod delete_serialized_object {
        use super::*;

        fn cmd(op: u32, ignored: u32, reference: u32) -> Vec<u8> {
            [op, ignored, reference]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect()
        }

        #[test]
        fn a_short_command_is_false() {
            let fx = Fx::new(&entries3(), Placement::Mapped);
            fx.set_placement(1, 5, 5);
            assert!(
                !fx.m
                    .delete_serialized_object(Hold::NONE, &cmd(1000, 0, 1)[..11])
            );
            assert_eq!(fx.placement(1), (5, 5));
        }

        #[test]
        fn each_operation_deletes_its_kind_and_clears() {
            let fx = Fx::new(&entries3(), Placement::Mapped);
            let world = fx.platform.world();
            for op in [1000u32, 1001, 1012] {
                assert!(fx.m.register_object::<Resources>(1, texture_resource(&fx)));
                fx.set_placement(1, 3, 3);
                assert!(fx.m.delete_serialized_object(Hold::NONE, &cmd(op, 0xdead, 1)));
                assert!(fx.m.get_existing_object::<Resources>(1).is_none(), "{op}");
                assert_eq!(fx.placement(1), (0, 0));
            }
            macro_rules! one {
                ($op:expr, $k:ty, $kind:ty) => {{
                    let o = world.new_obj::<$kind>(Default::default());
                    assert!(fx.m.register_object::<$k>(2, o));
                    fx.set_placement(2, 3, 3);
                    assert!(fx.m.delete_serialized_object(Hold::NONE, &cmd($op, 0, 2)));
                    assert!(fx.m.get_existing_object::<$k>(2).is_none());
                    assert_eq!(fx.placement(2), (0, 0));
                }};
            }
            one!(1002, DepthStencilStates, kind::DepthStencilState);
            one!(1003, SamplerStates, kind::SamplerState);
            one!(1005, Functions, kind::Function);
            one!(1006, ComputePipelineStates, kind::ComputePipeline);
            one!(1007, RenderPipelineStates, kind::RenderPipeline);
            one!(1009, Fences, kind::Fence);
            one!(1014, RasterizationRateMaps, kind::RateMap);
        }

        #[test]
        fn an_invalid_operation_logs() {
            let fx = Fx::new(&entries3(), Placement::Mapped);
            let (r, lines) =
                fx.logged(|| fx.m.delete_serialized_object(Hold::NONE, &cmd(1004, 0, 1)));
            assert!(!r);
            assert_eq!(
                lines,
                vec!["delete of object 1 ignored: unknown operation 1004".to_owned()]
            );
            // A short command logs nothing.
            let (r, lines) = fx.logged(|| {
                fx.m.delete_serialized_object(Hold::NONE, &cmd(1004, 0, 1)[..8])
            });
            assert!(!r);
            assert!(lines.is_empty());
        }

        #[test]
        fn other_operations_are_false_and_touch_nothing() {
            let fx = Fx::new(&entries3(), Placement::Mapped);
            assert!(fx.m.register_object::<Resources>(1, texture_resource(&fx)));
            fx.set_placement(1, 5, 5);
            for op in [1004u32, 1008, 1010, 1011, 1013, 0, 999] {
                assert!(!fx.m.delete_serialized_object(Hold::NONE, &cmd(op, 0, 1)));
            }
            assert!(fx.m.get_existing_object::<Resources>(1).unwrap().is_some());
            assert_eq!(fx.placement(1), (5, 5));
        }
    }

    mod clear_placement {
        use super::*;

        #[test]
        fn a_mapped_list_is_zeroed_and_the_list_mutex_is_free_after() {
            let fx = Fx::new(&entries3(), Placement::Mapped);
            fx.set_placement(1, 1, 2);
            fx.m.clear_placement_list_entry(Hold::NONE, 1);
            assert_eq!(fx.placement(1), (0, 0));
            assert!(fx.olm.mutex.try_lock().is_some());
        }

        #[test]
        fn an_unmappable_list_writes_nothing() {
            let fx = Fx::new(&entries3(), Placement::Unmapped);
            // Mapping the placement list fails again.
            fx.platform.script().push("new_virtual_mapping", Step::Nil);
            fx.set_placement(1, 1, 2);
            fx.m.clear_placement_list_entry(Hold::NONE, 1);
            assert_eq!(fx.placement(1), (1, 2));
            assert!(fx.olm.mutex.try_lock().is_some());
        }

        #[test]
        fn an_unmapped_list_is_mapped_under_the_list_mutex_then_zeroed() {
            let fx = Fx::new(&entries3(), Placement::Unmapped);
            fx.platform.script().push(
                "new_virtual_mapping",
                Step::value(Some(crate::memory::mapping::Mapping::<P>::new(
                    fx.addr(PL),
                    0x4000,
                    true,
                    crate::memory::mapping::Teardown::None,
                ))),
            );
            fx.set_placement(2, 1, 2);
            fx.m.clear_placement_list_entry(Hold::NONE, 2);
            assert_eq!(fx.placement(2), (0, 0));
        }

        #[test]
        fn no_placement_list_is_a_no_op() {
            // Legacy: mapping the placement list is `length == 0`, true; then length 0 returns.
            let fx = Fx::with_kind(
                &entries3(),
                Placement::Off,
                ObjectListKind::Legacy {
                    object_list_mapped: false,
                },
            );
            fx.m.clear_placement_list_entry(Hold::NONE, 1);
        }

        #[test]
        fn out_of_range_writes_nothing() {
            let fx = Fx::new(&entries3(), Placement::Mapped);
            fx.set_placement(3, 1, 2); // just past the three entries
            fx.m.clear_placement_list_entry(Hold::NONE, 3);
            assert_eq!(fx.placement(3), (1, 2));
        }

        /// One `stp xzr, xzr`: a guest flipping entry 1 between two whole 16-byte values with one
        /// 16-byte store each, while the host clears it, reads zero or one of its own values
        /// with one 16-byte load, never `{field0, 0}` or `{0, id}`.
        #[test]
        fn the_clear_is_one_16_byte_store() {
            use std::sync::atomic::{AtomicBool, Ordering};
            let fx = Fx::new(&entries3(), Placement::Mapped);
            let at = PL + 16;
            let (a, b) = ((1u64, 2u64), (3u64, 4u64));
            let stop = AtomicBool::new(false);
            let torn = std::thread::scope(|sc| {
                let guest = sc.spawn(|| {
                    let mut torn = Vec::new();
                    while !stop.load(Ordering::Relaxed) {
                        for v in [a, b] {
                            assert!(fx.region.store_16_atomic(at, v.0, v.1));
                            let seen = fx.region.load_16_atomic(at).unwrap();
                            if seen != (0, 0) && seen != a && seen != b && torn.len() < 8 {
                                torn.push(seen);
                            }
                        }
                    }
                    torn
                });
                for _ in 0..200_000 {
                    fx.m.clear_placement_list_entry(Hold::NONE, 1);
                }
                stop.store(true, Ordering::Relaxed);
                guest.join().unwrap()
            });
            assert!(torn.is_empty(), "torn entries {torn:x?}");
            fx.m.clear_placement_list_entry(Hold::NONE, 1);
            assert_eq!(fx.placement(1), (0, 0));
        }
    }

    mod registries {
        use super::*;

        #[test]
        fn the_newest_address_wins_and_zero_is_accepted() {
            let fx = Fx::new(&entries3(), Placement::Off);
            assert_eq!(fx.m.register_gpu_address(0x1000, 1), Ok(()));
            assert_eq!(fx.m.register_gpu_address(0x1000, 2), Ok(()));
            assert_eq!(fx.m.register_gpu_address(0, 3), Ok(()));
            assert_eq!(fx.m.register_gpu_address(u64::MAX, 4), Ok(()));
            assert_eq!(fx.m.gpu_address_at_or_below(0x1000), Some((0x1000, 2)));
            assert_eq!(fx.m.gpu_address_at_or_below(0x1fff), Some((0x1000, 2)));
            assert_eq!(fx.m.gpu_address_at_or_below(0xfff), Some((0, 3)));
            assert_eq!(fx.m.gpu_address_at_or_below(u64::MAX), Some((u64::MAX, 4)));
        }

        /// Deletes never unregister an address.
        #[test]
        fn deletes_keep_the_address() {
            let fx = Fx::new(&entries3(), Placement::Off);
            fx.m.register_gpu_address(0x2000, 1).unwrap();
            fx.m.delete_object::<Resources>(Hold::NONE, 1, true);
            assert_eq!(fx.m.gpu_address_at_or_below(0x2000), Some((0x2000, 1)));
        }

        #[test]
        fn below_every_address_is_none() {
            let fx = Fx::new(&entries3(), Placement::Off);
            fx.m.register_gpu_address(0x2000, 1).unwrap();
            assert_eq!(fx.m.gpu_address_at_or_below(0x1fff), None);
        }

        #[test]
        fn a_pipeline_registration_replaces_and_is_found() {
            let fx = Fx::new(&entries3(), Placement::Off);
            assert_eq!(fx.m.register_pipeline(0x77, 1), Ok(()));
            assert_eq!(fx.m.get_pipeline_ref(0x77), Ok(1));
            assert_eq!(fx.m.register_pipeline(0x77, 2), Ok(()));
            assert_eq!(fx.m.get_pipeline_ref(0x77), Ok(2));
        }

        #[test]
        fn an_all_ones_pipeline_is_invalid_even_if_registered() {
            let fx = Fx::new(&entries3(), Placement::Off);
            fx.m.register_pipeline(u64::MAX, 1).unwrap();
            assert_eq!(
                fx.m.get_pipeline_ref(u64::MAX),
                Err("pipeline identifier 0xffffffffffffffff is not valid".to_owned())
            );
        }

        /// An identifier nothing registered is refused.
        #[test]
        fn a_missing_pipeline_is_refused() {
            let fx = Fx::new(&entries3(), Placement::Off);
            assert_eq!(
                fx.m.get_pipeline_ref(0),
                Err("no pipeline registered for identifier 0x0".to_owned())
            );
            assert_eq!(
                fx.m.get_pipeline_ref(0xab),
                Err("no pipeline registered for identifier 0xab".to_owned())
            );
        }
    }
}

/// A cached lookup never waits for a creation in progress on another thread.
#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod lock_scope_tests {
    use super::super::fixture::*;
    use super::*;
    use crate::platform::Handle;
    use crate::testkit::{Fk, kind};
    use std::sync::mpsc;
    use std::time::Duration;

    fn fence(fx: &Fx) -> Fk<kind::Fence> {
        fx.platform
            .world()
            .new_obj::<kind::Fence>(Default::default())
    }

    type Park = (mpsc::Sender<()>, mpsc::Receiver<()>);
    static PARK: parking_lot::Mutex<Option<Park>> = parking_lot::Mutex::new(None);

    /// A kind over `fences` whose creator parks until the test releases it.
    struct ParkKind;

    impl ManagedKind<P> for ParkKind {
        type Obj = Fk<kind::Fence>;
        fn sub_manager(c: &Caches<P>) -> &Cache<Self::Obj> {
            &c.fences
        }
        fn sub_manager_mut(c: &mut Caches<P>) -> &mut Cache<Self::Obj> {
            &mut c.fences
        }
        fn retain(o: &Self::Obj) -> Self::Obj {
            o.clone_ref()
        }
        fn create_object(
            _hold: Hold<'_>,
            m: &ResourceManager<P>,
            _reference: u32,
            _entry: &ObjectEntry,
            _cursor: Cursor,
            _placement: Option<GuestPtr>,
        ) -> RaiseResult<Result<Self::Obj, String>> {
            let (entered, release) = PARK.lock().take().unwrap();
            entered.send(()).unwrap();
            release.recv().unwrap();
            Ok(Ok(new_fence(m)))
        }
    }

    /// The probe takes the leaf lock only; a creation holds the creation lock throughout.
    #[test]
    fn a_cached_lookup_does_not_wait_for_another_threads_creation() {
        let fx = Fx::new(&[(0, 0, 0), (6, 0x10, REC), (6, 0x10, REC)], Placement::Off);
        let cached = fence(&fx);
        assert!(fx.m.register_object::<ParkKind>(2, cached.clone_ref()));
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        *PARK.lock() = Some((entered_tx, release_rx));
        fx.next_cursor(0, 0x10);
        let m = fx.m.clone();
        let a = std::thread::spawn(move || {
            let _ = m.get_object::<ParkKind>(Hold::NONE, 1);
        });
        entered_rx.recv().unwrap();
        let m = fx.m.clone();
        let (done_tx, done_rx) = mpsc::channel();
        let b = std::thread::spawn(move || {
            done_tx
                .send(
                    m.get_object::<ParkKind>(Hold::NONE, 2)
                        .unwrap()
                        .unwrap()
                        .is_some(),
                )
                .unwrap();
        });
        let r = done_rx.recv_timeout(Duration::from_millis(500));
        release_tx.send(()).unwrap();
        a.join().unwrap();
        b.join().unwrap();
        assert_eq!(r, Ok(true), "B's cached lookup blocked behind A's creation");
    }

    #[test]
    fn a_cache_hit_does_not_wait_for_a_creation_on_another_thread() {
        let fx = Fx::new(&[(0, 0, 0), (6, 0x10, REC)], Placement::Off);
        assert!(fx.m.register_object::<Fences>(1, fence(&fx)));
        let m = fx.m.clone();
        let guard = fx.m.creation.lock();
        let (tx, rx) = mpsc::channel();
        let h = std::thread::spawn(move || {
            let _ = tx.send(m.get_existing_object::<Fences>(1).map(|o| o.is_some()));
        });
        let r = rx.recv_timeout(Duration::from_millis(500));
        drop(guard);
        h.join().unwrap();
        assert_eq!(r.ok(), Some(Some(true)));
    }

    /// The pipeline registry and the flush's walk take no creation lock either.
    #[test]
    fn get_pipeline_ref_and_flush_do_not_wait_for_a_creation_on_another_thread() {
        let fx = Fx::new(&[(0, 0, 0), (6, 0x10, REC)], Placement::Off);
        fx.m.register_pipeline(7, 3).unwrap();
        let m = fx.m.clone();
        let guard = fx.m.creation.lock();
        let (tx, rx) = mpsc::channel();
        let h = std::thread::spawn(move || {
            let r = m.get_pipeline_ref(7);
            let flushed = m.flush_resources(Hold::NONE).is_ok();
            let _ = tx.send((r, flushed));
        });
        let r = rx.recv_timeout(Duration::from_millis(500));
        drop(guard);
        h.join().unwrap();
        assert_eq!(r.ok(), Some((Ok(3), true)));
    }
}
