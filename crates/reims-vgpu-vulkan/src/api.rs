//! What the Rust API (`reims-vgpu`, feature `vulkan`) asks of this platform beyond core's
//! `Platform`: a frame target's pixels ([`read_pixels`]).
//!
//! API frame targets use [`FramePixels`] where its buffer can be allocated: the GPU copies
//! pixels before the frame completes, and the host reads cached rows. Measured on an RX 9070 XT
//! with RADV 26.2.4, a native 3840x2160 BGRA8 read through `get_bytes` took 5.70–6.91 CPU seconds;
//! a GPU texture-to-buffer copy and buffer read took 0.75–0.96 CPU ms and 1.58–4.16 wall ms,
//! with every byte checked. Those are readback measurements, not desktop frame rates.

use metal2vulkan_runtime as rt;
use reims_vgpu_core::num::ToU64;

use crate::answer::{get, plain};
use crate::gpu::command::VulkanCommandBuffer;
use crate::gpu::resource::VulkanTexture;

/// A frame target's pixels in host-cached buffer memory. The frame's command buffer copies
/// the rendered image here before completing; reading it submits no work and waits for none.
/// On discrete GPUs a host image copy can otherwise read uncached VRAM a texel at a time.
pub struct FramePixels {
    buffer: rt::Buffer,
    width: u64,
    height: u64,
    pitch: u64,
}

impl FramePixels {
    /// Optional storage: if allocation fails, the API keeps its ordinary texture readback.
    /// Frame targets are non-mipmapped, single-slice textures with four bytes per pixel.
    pub fn new(texture: &VulkanTexture) -> Option<Self> {
        let texture = texture.copied("frame target pixels");
        let width = plain(|| texture.width());
        let height = plain(|| texture.height());
        let pitch = width.checked_mul(4)?.checked_add(255)? & !255;
        let size = pitch.checked_mul(height)?;
        let buffer = texture
            .device()
            .new_buffer(size, rt::ResourceOptions::shared())
            .ok()?;
        buffer.contents()?;
        Some(Self {
            buffer,
            width,
            height,
            pitch,
        })
    }

    /// Append a GPU copy after the target's render, on its own frame command buffer. It
    /// therefore waits for no unrelated queue or future event when the host reads the pixels.
    pub fn encode(&self, command_buffer: &VulkanCommandBuffer, texture: &VulkanTexture) {
        let texture = texture.copied("frame target pixels");
        // A private encoder of the API's frame, included in runtime failure numbering.
        command_buffer.made_encoder(false);
        self.encode_copy(&command_buffer.raw, texture);
    }

    fn encode_copy(&self, command_buffer: &rt::CommandBuffer, texture: &rt::Texture) {
        let mut blit = get("blitCommandEncoder", || {
            command_buffer.blit_command_encoder()
        });
        get(
            "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:",
            || {
                blit.copy_from_texture_to_buffer(
                    texture,
                    0,
                    0,
                    rt::Origin { x: 0, y: 0, z: 0 },
                    rt::Size {
                        width: self.width,
                        height: self.height,
                        depth: 1,
                    },
                    &self.buffer,
                    0,
                    self.pitch,
                    self.pitch * self.height,
                    rt::BlitOption::None,
                )
            },
        );
        get("endEncoding", || blit.end_encoding());
    }

    /// Copy rows whose layout the API checked. Like a direct read of the texture, the host
    /// waits for frames whose pixels it needs before calling; no buffer bytes are modified.
    #[allow(
        clippy::expect_used,
        reason = "Frame allocation and the C API validate these row bounds"
    )]
    pub fn read(&self, dst: &mut [u8], bytes_per_row: usize) {
        let source = self
            .buffer
            .contents()
            .expect("shared frame pixels stay mapped");
        let row = (self.width * 4) as usize;
        for y in 0..self.height as usize {
            let start = y.checked_mul(bytes_per_row).expect("frame row offset fits");
            let end = start.checked_add(row).expect("frame row extent fits");
            let output = &mut dst[start..end];
            // SAFETY: the buffer has height padded rows; output holds exactly one row.
            // Its own allocation cannot alias the host's output slice.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    source.as_ptr().add(y * self.pitch as usize),
                    output.as_mut_ptr(),
                    row,
                );
            }
        }
    }
}

/// Copies the whole of `texture`'s first level and slice into `dst`: `height` rows of `width * 4`
/// bytes, each `bytes_per_row` after the last, the bytes between rows left as they are
/// (`getBytes:bytesPerRow:fromRegion:mipmapLevel:`). The caller has checked that `dst` holds
/// them. A refusal of the runtime's dies at the selector, as a raise in Metal's `getBytes` dies
/// at its send.
pub fn read_pixels(texture: &VulkanTexture, dst: &mut [u8], bytes_per_row: usize) {
    let sel = "getBytes:bytesPerRow:fromRegion:mipmapLevel:";
    let t = texture.copied(sel);
    let region = rt::Region::new_2d(0, 0, plain(|| t.width()), plain(|| t.height()));
    get(sel, || {
        t.get_bytes(dst, bytes_per_row.to_u64(), 0, region, 0, 0)
    });
}

#[cfg(test)]
mod tests {
    use metal2vulkan::metal::PixelFormat;
    use reims_vgpu_core::platform::Handle as _;

    use super::*;
    use crate::gpu::device::tests::dev;
    use crate::gpu::resource::TextureObject;
    use crate::platform::Obj;

    /// A BGRA8 texture of `width` by `height` (shared storage, the descriptor's default), holding
    /// `px` (rows of `width * 4` bytes).
    fn texture(width: u64, height: u64, px: &[u8]) -> VulkanTexture {
        let d = dev();
        let desc = rt::TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, height, false);
        let raw = d.raw.new_texture(&desc).unwrap();
        let all = rt::Region::new_2d(0, 0, width, height);
        raw.replace_region(all, 0, 0, px, width * 4, 0).unwrap();
        Obj::new(TextureObject {
            raw,
            device: d.clone_ref(),
            surface: None,
            plane: None,
            under: None,
            private_usage: 0,
            space: std::sync::Arc::new(crate::gpu::metal_space::Hold::NONE),
        })
    }

    /// Every pixel of the texture, row by row; the bytes past each row's pixels are left alone
    #[test]
    fn read_pixels_reads_the_whole_texture() {
        let (w, h) = (3u64, 2u64);
        let px: Vec<u8> = (0u8..24).collect();
        let t = texture(w, h, &px);
        let mut tight = vec![0xaa; 24];
        read_pixels(&t, &mut tight, 12);
        assert_eq!(tight, px);
        let mut padded = vec![0xaa; 16 + 12];
        read_pixels(&t, &mut padded, 16);
        assert_eq!(&padded[..12], &px[..12]);
        assert_eq!(&padded[12..16], &[0xaa; 4]);
        assert_eq!(&padded[16..], &px[12..]);
    }

    /// The pixel copy follows a GPU write in the same command buffer. Odd widths need padded
    /// GPU rows; caller row padding stays untouched, and a later frame replaces the pixels.
    #[test]
    fn frame_pixels_follow_gpu_writes_and_preserve_row_padding() {
        for (w, h) in [(3u64, 2u64), (64, 3), (65, 5)] {
            let t = texture(w, h, &vec![0; (w * h * 4) as usize]);
            let pixels = FramePixels::new(&t).unwrap();
            let raw = t.copied("test frame pixels");
            let queue = raw.device().new_command_queue().unwrap();
            let source = raw
                .device()
                .new_buffer(pixels.pitch * h, rt::ResourceOptions::shared())
                .unwrap();
            let stride = (w * 4 + 12) as usize;
            let extent = (h as usize - 1) * stride + (w * 4) as usize;
            let mut output = vec![0xa5; extent];
            pixels.read(&mut output, stride);
            for y in 0..h as usize {
                assert!(
                    output[y * stride..y * stride + w as usize * 4]
                        .iter()
                        .all(|&v| v == 0)
                );
            }
            for phase in [17usize, 73] {
                let expected: Vec<u8> = (0..(w * h * 4) as usize)
                    .map(|i| ((i * 19 + phase) % 251) as u8)
                    .collect();
                let pointer = source.contents().unwrap();
                for y in 0..h as usize {
                    // SAFETY: this shared buffer holds h padded rows, with no GPU work yet.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            expected.as_ptr().add(y * w as usize * 4),
                            pointer.as_ptr().add(y * pixels.pitch as usize),
                            w as usize * 4,
                        );
                    }
                }
                let cb = queue.command_buffer().unwrap();
                let mut upload = cb.blit_command_encoder().unwrap();
                upload
                    .copy_from_buffer_to_texture(
                        &source,
                        0,
                        pixels.pitch,
                        pixels.pitch * h,
                        rt::Size {
                            width: w,
                            height: h,
                            depth: 1,
                        },
                        raw,
                        0,
                        0,
                        rt::Origin { x: 0, y: 0, z: 0 },
                        rt::BlitOption::None,
                    )
                    .unwrap();
                upload.end_encoding().unwrap();
                pixels.encode_copy(&cb, raw);
                cb.commit().unwrap();
                cb.wait_until_completed_for(std::time::Duration::from_secs(5))
                    .unwrap();
                assert!(cb.error().is_none());
                output.fill(0xa5);
                pixels.read(&mut output, stride);
                for y in 0..h as usize {
                    assert_eq!(
                        &output[y * stride..y * stride + w as usize * 4],
                        &expected[y * w as usize * 4..(y + 1) * w as usize * 4]
                    );
                    if y + 1 < h as usize {
                        assert_eq!(
                            &output[y * stride + w as usize * 4..(y + 1) * stride],
                            &[0xa5; 12]
                        );
                    }
                }
            }
        }
    }
}
