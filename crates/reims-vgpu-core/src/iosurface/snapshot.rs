//! Local saved-machine state for a quiesced IOSurface mapper.
//!
//! The host must stop guest CPUs and finish graphics suspension first. That flushes GPU
//! resources before the host copies RAM and before host-only surface bytes are read here.
//! Restoring requires guest RAM to be loaded and precedes graphics `did_resume`.

#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

use crate::guest::GuestPtr;
use crate::iosurface::DeathKind;
use crate::iosurface::device::HostDeviceCore;
use crate::iosurface::regs::*;
use crate::iosurface::table::{HOST_ONLY_LIMIT, MAPPING_ID_COUNT};
use crate::platform::Platform;
use crate::platform::os::surface::Surface;
use std::collections::HashSet;

const MAGIC: &[u8; 8] = b"RVGIOM01";
const REGS: [u64; 6] = [
    REG_DESC_ENABLE,
    REG_DESCRIPTOR_BASE,
    REG_RING_RD_COUNTER,
    REG_RING_BASE,
    REG_RING_SIZE,
    REG_RING_WR_COUNTER,
];

struct Reader<'a>(&'a [u8]);
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let (head, rest) = self.0.split_at_checked(n).ok_or("truncated mapper state")?;
        self.0 = rest;
        Ok(head)
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().map_err(|_| "mapper integer")?,
        ))
    }
}

impl<P: Platform> HostDeviceCore<P> {
    /// Saves registers, live mapping IDs and host-only bytes, after all GPU tasks have stopped.
    /// Guest-backed surface bytes are already in the host's RAM image. No interrupt is raised.
    pub fn save_machine_state(&self) -> Result<Vec<u8>, String> {
        self.hw_queue.sync(|| {
            let st = self.state.lock();
            let registers = REGS.map(|r| self.mmio_read(r));
            if registers[2] != registers[5] {
                return Err("IOSurface command ring has not drained".into());
            }
            let mut bytes = MAGIC.to_vec();
            bytes.extend_from_slice(&self.page_size().to_le_bytes());
            for value in registers {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let count = st.slots.iter().filter(|s| s.surface.is_some()).count();
            bytes.extend_from_slice(
                &u64::try_from(count)
                    .map_err(|_| "mapping count")?
                    .to_le_bytes(),
            );
            let mut host_bytes = 0u64;
            for (id, slot) in st.slots.iter().enumerate() {
                let Some(surface) = &slot.surface else {
                    continue;
                };
                bytes
                    .extend_from_slice(&u64::try_from(id).map_err(|_| "mapping id")?.to_le_bytes());
                bytes.extend_from_slice(&u64::from(slot.host_only).to_le_bytes());
                let size = if slot.host_only {
                    surface.alloc_size()
                } else {
                    0
                };
                host_bytes = host_bytes
                    .checked_add(size)
                    .ok_or("mapper byte count overflow")?;
                if host_bytes > HOST_ONLY_LIMIT {
                    return Err("host-only surfaces exceed limit".into());
                }
                bytes.extend_from_slice(&size.to_le_bytes());
                if size != 0 {
                    let size = usize::try_from(size).map_err(|_| "surface size")?;
                    if surface.base_address().is_null() {
                        return Err("surface has no CPU backing".into());
                    }
                    let mut data = vec![0u8; size];
                    surface.lock(1);
                    GuestPtr::new(surface.base_address()).copy_to(
                        0,
                        &mut data,
                        DeathKind::SnapshotRead,
                    );
                    surface.unlock(1);
                    bytes.extend_from_slice(&data);
                }
            }
            Ok(bytes)
        })
    }

    /// Restores a saved mapper over loaded guest RAM, before any graphics task resumes.
    /// Refuses truncated/duplicate/oversized states before changing the device. A mapping
    /// failure is reported to the host, which must discard this incoming VM without running it.
    pub fn load_machine_state(&self, bytes: &[u8]) -> Result<(), String> {
        let mut r = Reader(bytes);
        if r.take(8)? != MAGIC || r.u64()? != self.page_size() {
            return Err("incompatible mapper state".into());
        }
        let mut registers = [0; 6];
        for reg in &mut registers {
            *reg = r.u64()?;
        }
        if registers[2] != registers[5] {
            return Err("saved IOSurface ring is not drained".into());
        }
        let count = r.u64()?;
        if count >= MAPPING_ID_COUNT {
            return Err("too many mapper surfaces".into());
        }
        let mut ids = HashSet::new();
        let mut slots = Vec::new();
        let mut host_bytes = 0u64;
        for _ in 0..count {
            let id = r.u64()?;
            let host_only = r.u64()?;
            let size = r.u64()?;
            if id == 0
                || id >= MAPPING_ID_COUNT
                || !ids.insert(id)
                || host_only > 1
                || (host_only == 0 && size != 0)
            {
                return Err("invalid saved mapper surface".into());
            }
            host_bytes = host_bytes
                .checked_add(size)
                .ok_or("mapper byte count overflow")?;
            if host_bytes > HOST_ONLY_LIMIT {
                return Err("oversized mapper surface data".into());
            }
            let data = r.take(usize::try_from(size).map_err(|_| "surface size")?)?;
            slots.push((id, host_only != 0, data));
        }
        if !r.0.is_empty() {
            return Err("trailing mapper state".into());
        }
        // Configure without a ring first; the read counter is writable only before ring mapping.
        self.mmio_write(REG_RING_BASE, 0);
        self.mmio_write(REG_DESC_ENABLE, registers[0]);
        self.mmio_write(REG_DESCRIPTOR_BASE, registers[1]);
        self.mmio_write(REG_RING_RD_COUNTER, registers[2]);
        self.hw_queue.sync(|| {
            let mut st = self.state.lock();
            for (id, host_only, data) in slots {
                let id32 = u32::try_from(id).map_err(|_| "mapping id")?;
                if self.map(&mut st, id32) != crate::iosurface::ring::IoReturn::Success {
                    return Err(format!("could not restore IOSurface mapping {id}"));
                }
                let slot = st
                    .slots
                    .get(usize::try_from(id).map_err(|_| "mapping id")?)
                    .ok_or("mapping slot")?;
                let surface = slot.surface.as_ref().ok_or("missing restored surface")?;
                if slot.host_only != host_only
                    || (host_only
                        && surface.alloc_size()
                            != u64::try_from(data.len()).map_err(|_| "surface size")?)
                {
                    return Err("restored surface differs from saved descriptor".into());
                }
                if !data.is_empty() {
                    if surface.base_address().is_null() {
                        return Err("surface has no CPU backing".into());
                    }
                    surface.lock(0);
                    GuestPtr::new(surface.base_address()).copy_from(
                        0,
                        data,
                        DeathKind::SnapshotWrite,
                    );
                    surface.cpu_wrote();
                    surface.unlock(0);
                }
            }
            Ok::<(), String>(())
        })?;
        self.mmio_write(REG_RING_BASE, registers[3]);
        self.mmio_write(REG_RING_SIZE, registers[4]);
        self.mmio_write(REG_RING_WR_COUNTER, registers[5]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iosurface::ring::IoReturn;
    use crate::iosurface::testing::{HOLE, ROOT, rig, v2};

    #[test]
    fn guest_backed_mapping_roundtrip() {
        let r = rig();
        r.setup(true);
        r.root(2);
        r.put_v2(5, &v2(2, ROOT));
        assert_eq!(r.map(5), IoReturn::Success);
        let bytes = r.dev.save_machine_state().unwrap();
        let regs = REGS.map(|off| r.dev.mmio_read(off));
        r.dev.mmio_write(REG_DESCRIPTOR_BASE, 0);
        r.dev.load_machine_state(&bytes).unwrap();
        assert_eq!(REGS.map(|off| r.dev.mmio_read(off)), regs);
        assert!(r.surface(5).is_some());
        assert_eq!(r.dev.save_machine_state().unwrap(), bytes);
    }
    #[test]
    fn host_only_contents_survive_recreation() {
        let r = rig();
        r.setup(true);
        let mut d = v2(3, HOLE);
        d.base.bits41 = 2;
        r.put_v2(5, &d);
        assert_eq!(r.map(5), IoReturn::Success);
        let surface = r.held(5);
        GuestPtr::new(surface.base_address()).copy_from(
            0,
            b"unsaved GPU surface",
            DeathKind::SnapshotWrite,
        );
        let bytes = r.dev.save_machine_state().unwrap();
        drop(surface);
        r.dev.mmio_write(REG_DESCRIPTOR_BASE, 0);
        r.dev.load_machine_state(&bytes).unwrap();
        assert_eq!(r.dev.save_machine_state().unwrap(), bytes);
    }
    #[test]
    fn malformed_states_do_not_change_registers() {
        let r = rig();
        r.setup(true);
        let bytes = r.dev.save_machine_state().unwrap();
        let regs = REGS.map(|off| r.dev.mmio_read(off));
        for n in 0..bytes.len() {
            assert!(r.dev.load_machine_state(&bytes[..n]).is_err());
        }
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(r.dev.load_machine_state(&trailing).is_err());
        assert_eq!(REGS.map(|off| r.dev.mmio_read(off)), regs);
    }
    #[test]
    fn duplicate_ids_refused_before_restore() {
        let r = rig();
        r.setup(true);
        r.root(2);
        r.put_v2(5, &v2(2, ROOT));
        assert_eq!(r.map(5), IoReturn::Success);
        let mut bytes = r.dev.save_machine_state().unwrap();
        bytes[64..72].copy_from_slice(&2u64.to_le_bytes());
        let entry = bytes[72..].to_vec();
        bytes.extend_from_slice(&entry);
        assert!(r.dev.load_machine_state(&bytes).is_err());
        assert!(r.surface(5).is_some());
    }
}
