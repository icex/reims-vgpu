//! `VulkanPlatform` as an `ApiPlatform` (feature `vulkan`): the runtime's devices, on macOS over
//! MoltenVK and on Linux natively.

use reims_vgpu_core::num::ToU64 as _;
use reims_vgpu_core::platform::Platform;
use reims_vgpu_vulkan::VulkanPlatform;
use reims_vgpu_vulkan::os::GuestRamFile;

use super::*;

/// The platform over metal2vulkan's runtime (architecture 2.2): the runtime's devices, and on
/// macOS windows whose frames Metal copies from the runtime's (`window::window_sink`).
impl ApiPlatform for VulkanPlatform {
    fn process() -> Arc<VulkanPlatform> {
        crate::singleton::vulkan_platform()
    }

    /// The runtime's system default device (`MTLCreateSystemDefaultDevice()`), or the device
    /// whose registry ID, as the platform answers it, is `id` (`Platform::open_device`): the
    /// default device answers Metal's `registryID` of the same GPU, which is what a host
    /// names it by, and any other device the runtime's own ID.
    fn metal_device(&self, registry_id: Option<u64>) -> Option<<Self as Platform>::Device> {
        match registry_id {
            None => VulkanPlatform::system_default_device(),
            Some(id) => self.open_device(id),
        }
    }

    /// The whole texture, through the runtime's `get_bytes`
    /// (`reims_vgpu_vulkan::api::read_pixels`).
    fn read_pixels(
        &self,
        texture: &<Self as Platform>::Texture,
        dst: &mut [u8],
        bytes_per_row: usize,
    ) {
        reims_vgpu_vulkan::api::read_pixels(texture, dst, bytes_per_row);
    }

    fn frame_pixels(
        &self,
        texture: &<Self as Platform>::Texture,
    ) -> Option<Box<dyn FramePixels<Self>>> {
        reims_vgpu_vulkan::api::FramePixels::new(texture)
            .map(|pixels| Box::new(VulkanFramePixels(pixels)) as Box<dyn FramePixels<Self>>)
    }

    /// The files behind the map's ranges, registered while the device lives
    /// (`reims_vgpu_vulkan::os::lease_guest_memory`): on Linux the platform makes its views of the
    /// guest's pages of them. A range without a file is not registered, and a view of its pages
    /// is refused as a failed remap is. On macOS nothing is registered, since Mach makes a view
    /// by remapping addresses.
    fn guest_memory(&self, map: &MemoryMap) -> Box<dyn Any + Send + Sync> {
        let ranges = map.ranges().iter().zip(map.files());
        let files = ranges.filter_map(|(range, file)| {
            let file = file.as_ref()?;
            Some(GuestRamFile {
                host: range.host_address.addr().to_u64(),
                length: range.physical_length,
                fd: Arc::clone(&file.fd),
                offset: file.offset,
            })
        });
        Box::new(reims_vgpu_vulkan::os::lease_guest_memory(files))
    }
}

struct VulkanFramePixels(reims_vgpu_vulkan::api::FramePixels);

impl FramePixels<VulkanPlatform> for VulkanFramePixels {
    fn encode(
        &self,
        command_buffer: &<VulkanPlatform as Platform>::CommandBuffer,
        texture: &<VulkanPlatform as Platform>::Texture,
    ) {
        self.0.encode(command_buffer, texture);
    }

    fn read(&self, dst: &mut [u8], bytes_per_row: usize) {
        self.0.read(dst, bytes_per_row);
    }
}
