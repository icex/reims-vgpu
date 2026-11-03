//! `DeviceCaps` on the Vulkan platform: every answer core asks a device, read once per device.
//!
//! Each answer comes from one of three sources, in this order:
//! - **the runtime's answer** of the Metal query (`DeviceProperties`,
//!   `Device::heap_buffer_size_and_align`): the name, the threadgroup sizes, the public queries,
//!   and, in [`crate::gpu::format_info`], Apple's private pixel format description and level
//!   layout (`Device::pixel_format_info`, `Device::texture_level_info`);
//! - **computed from documented facts** the runtime gives (its device's Vulkan description, the
//!   compile layer's `MetalDeviceProperties`, Metal's documented argument tables), for Apple's
//!   private properties: [`computed`];
//! - **a stand-in** where no documented fact gives the answer, or where the runtime's answer is
//!   not Metal's and the guest or the host sees it: the Metal platform's answer on the host the
//!   stand-ins were read on, an Apple M3 Max (`stand_ins_are_the_metal_platforms_here` checks them
//!   against the Metal platform on the host it runs on), but for the registry ID, which Metal
//!   answers at run time on the host the library runs on.
//!
//! **This table lists every stand-in of the crate**, the `DeviceCaps` answers read here and the
//! answers other modules give from Metal on the M3 Max, each beside the code that answers it:
//!
//! | answer | stand-in | why not the runtime's or a fact |
//! |---|---|---|
//! | `registryID` of the runtime's system default device | Metal's system default device's, asked at run time on macOS ([`crate::os::default_registry_id`]; `0x10000043f` on the M3 Max) | the runtime's is folded from the device UUID, which MoltenVK fills with the vendor and device IDs alone (`0x0902061a6b100000` on the M3 Max), and a host names its device by Metal's (`open_device`, the handshake); another device, and a host where Metal has none, answer the runtime's |
//! | `supportsFamily:` of Apple5 to Apple10, -1009 | yes, yes, yes, yes, yes, no, no | the runtime claims a family only where every row of Apple's table is exact, and claims none on MoltenVK. **Pending support** (below): what a family promises that the runtime refuses dies at its selector |
//! | `supportsExtendedXR10Formats` | yes | **pending support** (below): the runtime makes textures of the XR10 formats (550 to 555), in heaps too, and of format 84 (plane 1's view format of the two-plane XR10 textures, 550 and 551), reads and writes them in kernels, samples them in 2D and gathers 552 to 555 as floats, unswizzled and under the four swizzles of the texture or a view the tests read through (`rgbx`'s `XR_SWIZZLES`, which move, repeat, drop and replace channels; no other swizzle is measured), as Metal does on the M3 (`rgbx`'s `extended_range_textures_read_as_metals`), reads 552 and 554 through an argument buffer as Metal does, and samples and gathers them so as Metal does through a sampler of one filter and no mip filter, samples other textures through one as Metal does while one lives that the command does not reach, and dies, naming the runtime's refusal, at a command that samples or gathers one it reaches (names or makes resident) through an argument buffer otherwise (a sampler of a mip filter among them), or reads an `_sRGB` twin or a swizzled one so (`rgbx`'s `extended_range_textures_an_argument_buffer_holds_read_as_metals`), copies 550's and 551's colour words to and from the CPU and buffers as Metal does, and renders to one where the fragment function writes all of its channels or none. It blends a `half4` output, and a `float4` one, into one in the shader, by its model of Metal's blend, and reads one it blends by framebuffer fetch. The model holds without fast math, each product rounded to `half` as Metal's compiler does there (`encode_render`'s `float4_extended_range_blends_are_metals_here` and `half4_extended_range_blends_are_metals_without_fast_math` compare `float4` and `half4` outputs with Metal across passes, into each linear form and each `_sRGB` twin), and under fast math for an output each of whose lanes is a load; where Metal may round the blend's source otherwise, under fast math an output the function computes, or a module that names no math mode, the render state announces the output's kind: for a `half4` one `extended-range-blend-source`, which the device relays for its reach ([`crate::answer`]): a computed `half4` misses Metal's by one code of a channel at 13 of the 66 blended configurations the tests draw, at 130 to 483 of 4096 texels each (`half4_extended_range_blends_announce_their_source`); for a `float4` one `extended-range-float-blend-source`, which the runtime announces at each draw that binds the state's variant, where the draw dies (the state is made), where a computed `float4` missed Metal's at 17 of the 66 (`float4_extended_range_blends_die_naming_their_source`). The tests draw the 22 blends of `XR_BLENDS` into each linear form and the 8 of `XR_SRGB_BLENDS` into each `_sRGB` twin; without fast math, where the runtime also draws a blend by the blend colour under a difference, and any blend into an `_sRGB` twin, they draw those of `XR_CONSTANT_DIFFERENCES` into each form and the rest of `XR_BLENDS` into each twin as Metal does (`extended_range_blends_drawn_without_fast_math_alone_are_metals`), and into each form and twin six blends under write masks of one colour channel and of two as Metal does, and a function's two outputs into colour 0, unblended, storing the first output's values as Metal does, the second nowhere, and without fast math under six blends of the second source as Metal does (`extended_range_masked_blends_and_two_output_states_are_metals`); the others are not measured. A draw that blends into or reads an attachment an earlier draw of its pass may have written, or that its `Clear` left between two codes, reads the `half` values the runtime holds through a pass of one sample and one layer without a rasterization-rate map, as Metal's tile memory does, announcing nothing, so its blends in one pass are Metal's (`encode_render`'s `float4_extended_range_blends_are_metals_here`, `a_blend_onto_a_clear_between_codes_is_metals`); otherwise (another such pass, after an execution of render commands or a tile dispatch, where an earlier render pass instance of the pass wrote the attachment, or a draw whose pipeline has no shadowed form) it sees the codes the runtime stored, where Metal holds the value through the pass, which the runtime announces and the device relays (`extended-range-blends-stored-codes`). The runtime refuses the rest, which dies at its selector; among it a render target of one resolved, unblended under a mask of some of its channels, blended under fast math under a mask of one colour channel, by a factor of the second source, by the blend colour under a difference, into an `_sRGB` form beyond the blends the runtime measured, or a `float4` output into one but by a blend whose colour lanes take one term alone, or from an output other than four `half` or `float` lanes; a state of a function that returns two outputs into colour 0 (a second, index 1, as a blend of the second source reads), blended by factors of the second source under fast math (`ExtendedRangeBlend(DualSource)`); a gather of 550 or 551, or as halves; and 550's and 551's views of another format, mipmaps, textures over a buffer and copies of a layout it has not measured; a copy between two textures of 550 or 551 dies at its selector too, whose alpha plane the runtime copies otherwise than Metal ([`crate::gpu::encode_blit`]) |
//! | `device_linear_texture_alignment_bytes`, `linear_texture_alignment_bytes`, `iosurface_texture_alignment_bytes`, `iosurface_read_only_texture_alignment_bytes` | 16 each | undocumented; the public linear alignment query answers 16 for every format a buffer backs |
//! | `is_large_mrt_supported`, `is_rgb10a2_gamma_supported`, `supports_conditional_load_store`, `supports_float16_bcubic_filtering` | yes | undocumented, unmeasured |
//! | `supports_compute_compressed_texture_write`, `supports_memory_order_atomics` | no | undocumented, unmeasured |
//! | `supports_texture_2d_multisample_array` | yes | undocumented |
//! | `supportsPrimitiveType:` of the raw values 5 to 9 | yes, yes, yes, yes, no | no public `MTLPrimitiveType` is one, and the runtime draws only the public types. **Pending support** (below): the render encoder draws a private type as Metal does where it is built to, and a draw it is not built for dies at the draw's selector (`encode_render`) |
//! | `supportsTextureWriteRoundingMode:` of 1 and 2 | yes, yes | undocumented. **Pending support** (below): a state of mode 1 writes as at the default; at mode 2 the runtime rounds a render state's half-float colour attachments to nearest even, as Metal does, where they store the fragment function's output as written, of 1, 2 or 4 samples (Metal's resolve of the samples does not take the mode); a render state of mode 2 whose fragment function may write a half-float attachment otherwise (of another sample count, whose resolve nothing measured, blended other than one and zero, or its alpha under alpha to coverage), or a texture under indirect command buffer support, dies as it is made ([`crate::gpu::serialized::rounded_writes`]); one whose fragment function may write a texture otherwise is made, and a draw under it that may write a texture of a half-float format (`R16Float`, `RG16Float`, `RGBA16Float`, `RGBX16Float`), of `RG11B10Float` or of `RGB9E5Float` dies at the draw ([`crate::gpu::serialized::MODE_2_TEXTURE_FORMATS`], [`crate::gpu::serialized::rounded_texture_writes`]): the runtime takes no mode for a fragment function's texture write, for every format, and Metal's mode 2 rounds the half-float ones measured to nearest even and `RG11B10Float` otherwise than its default, where `RGBX16Float`'s and `RGB9E5Float`'s are not measured; a kernel's writes are Metal's at every mode, as Metal's mode takes none of them |
//! | `maxComputeThreadgroupMemory` (Apple's private property, the guest's device-info key 31) | 65536 | undocumented, and not the public `maxThreadgroupMemoryLength` (32768 on both platforms, the runtime's answer) |
//! | `maxVertexAmplificationCount` (private, key 20) | 2, or the largest count the public `supportsVertexAmplificationCount:` takes where that is less | undocumented: the public query takes 1 to 8 on both platforms |
//! | `supportsTileShaders`, `supportsNonSquareTileShaders`, `supportsImageBlocks` (private, keys 23 and 24) | yes | the runtime makes tile pipelines and imageblocks, a `half4` member over `RGBA8Unorm` among them |
//! | an indirect command buffer's `gpuResourceID` and `resourceIndex` ([`crate::gpu::icb`]'s `Space`) | an ID of Metal's pool of them (`Pool`), a placement's low 32 bits, and a placement at `ID_LIMIT` (16384) or past it dies, as Metal writes past its table there | the runtime's ID, which its records' address follows, follows Metal's pool, placement and hold too, but after a reservation holding an ID of 2^23 or more, which it is not given: it refuses one of 2^30 or more, which Metal takes to 2^31 - 1, and a buffer whose own ID is 2^23 or more |
//! | an indirect command buffer's `uniqueIdentifier` ([`crate::gpu::icb`]) | `UNIQUE_IDENTIFIER_BASE` (`0x1000000000`) plus Metal's address of the buffer: the lowest free range from `FIRST_ADDRESS` (`0x18000`), aligned to `SLOT` (`0x8000`), that holds its `allocatedSize` (the runtime's, which is Metal's) and one more page | the runtime has no unique identifier of a buffer |
//! | the address an indirect command's arguments answer for a nil patch or control-point index buffer (`icb::NIL_BUFFER_ADDRESS`) | `0x1eeee1111` | Metal's private getters' answer, which no record holds |
//! | what an indirect command answers beside its record ([`crate::gpu::icb`]'s `CommandSide`) | of a buffer slot whose record names no buffer, one set nil or past the 32 the record holds: the address, the buffer's plus the offset's low 32 bits or those bits alone for nil, and the stride, all ones after a plain set and the stride after a stride form; 0 and 0 for a slot never set or reset; a threadgroup memory length set past the 31 the record holds | the runtime's record names no buffer at a nil set and holds no slot past its 32 buffers or 31 lengths, where Metal's getters answer them |
//! | a rate map's `parameterBufferSizeAndAlign` of a map not yet compiled ([`crate::gpu::rate_map`]'s `PARAMETER_DATA`) | 24,624 bytes aligned to 4, Metal's for every map, which a compiled map's runtime answers too (its parameter data is Metal's) | a map not yet compiled has no runtime map to answer it |
//! | a rate map's `physicalGranularity` (`rate_map::GRANULARITY`) | 32 by 32, depth 1, compiled or not | a map not yet compiled has no runtime map to answer it |
//! | what a mutable rate map not yet reset answers ([`crate::gpu::rate_map`]) | its descriptor's screen size and layer count, a physical size of 0 by 0 (depth 1) for every layer, screen-to-physical (0, 0), parameter data of zeros | the runtime has no map until the reset compiles one |
//!
//! **Pending support.** In the rows marked so, Metal has the capability and the runtime's support
//! for it is pending at the pinned runtime. The answer is kept Metal's, as the guest sees an M3
//! Max and the suite's expected files were recorded there. What the guest then does with it and
//! the runtime refuses dies at its selector ([`crate::answer::refused`],
//! [`crate::answer::unrepresentable`]), never with a faked result.
//!
//! **Command predication is the adapter's** ([`crate::gpu::predication`]), with no limit on
//! nesting, as Metal's: `supportsCommandBufferJump` (key 33's 0x4) is yes and
//! `maxPredicatedNestingDepth` (key 36) is `u32::MAX`, Metal's answers.
//!
//! **Where the device lacks any other capability, the answer is the device's truth**, never
//! Metal's:
//! - the two SIMD flags (key 33's 0x40, `supports_simd_reduction`, and 0x100,
//!   `supports_simd_shuffle_and_fill`) are yes only where the runtime takes every function of
//!   the classes they promise, in every stage, as Metal's device-wide flags do: where each row of
//!   Apple's feature tables the flag covers is exact in the runtime's audit of the Apple9 column
//!   (`Device::family_rows`, [`SIMD_ROWS`]): "SIMD-scoped reduction operations" and "Quad-scoped
//!   reduction operations" for 0x40, "SIMD shift and fill" for 0x100. The rows answer by the
//!   stages' classes and the quad members the compile layer decodes. At the pinned runtime every
//!   row is exact on this host, so both flags are yes and key 33 is Metal's: the quad
//!   reductions combine a quad's lanes in Metal's order, `(a + b) + (c + d)`, under the ballot
//!   and shuffle classes, which MoltenVK's vertex stage has too, and a fill's modulo is answered
//!   as Metal answers it, a constant one and a runtime one. A guest told yes where a row is not
//!   exact picks work that Metal runs and the runtime refuses or runs otherwise; told no, it
//!   never makes that pipeline;
//! - buffers and heaps over address ranges and detach backing (key 33's 0x8 and 0x10, and key 42,
//!   which core sets from detach backing, not from the argument buffer tier) are yes where the
//!   runtime imports host memory (`VK_EXT_external_memory_host`), which every heap over address
//!   ranges and every replace of a detached backing is made over ([`crate::gpu::backing`]), as
//!   on MoltenVK, where they are Metal's answers. A submission that uses a detached buffer or
//!   heap fails as Metal's does; [`crate::gpu::command`]'s docs list where it differs;
//! - `heapTextureSizeAndAlignWithDescriptor:` of a depth or stencil texture with
//!   `allowGPUOptimizedContents` off answers Metal's size, the runtime's model of Metal's answer,
//!   which answers an X32_Stencil8 texture as a Stencil8 one of its shape, else made of the
//!   runtime's answers for colour formats
//!   ([`crate::gpu::metal_space::metal_size_and_align`]), as the runtime's own is larger:
//!   MoltenVK keeps the texture's GPU-optimized layout, which the runtime announces
//!   (`OptimizedDepthStencilTexture`, [`crate::gpu::device`]'s query). It dies for a descriptor
//!   neither answers (one of `ShaderAtomic` usage the runtime refuses first, as Metal asserts),
//!   and so does a
//!   placement of such a texture in a heap where the runtime's size decides it
//!   (`crate::gpu::resource::Placements`), and
//!   `newTextureWithDescriptor:` of such a texture the runtime's model of Metal's answer has
//!   none for, at every shape; where the model has one, the texture takes it
//!   ([`crate::gpu::metal_space::own_heap_size`]);
//! - `heapTextureSizeAndAlignWithDescriptor:` of a texture whose layout in Metal's heaps the
//!   runtime does not model (a PVRTC format; 8 samples the runtime refuses first), whose
//!   size and alignment the runtime answers as its device's image's and announces so
//!   (`unmeasured-heap-texture-size`), dies at its selector, naming it ([`crate::answer::HEARD`]):
//!   a guest that sized a heap by such answers could fit more or fewer such textures in it than
//!   on Metal, at other offsets. The runtime models a mipmapped texture Metal keeps losslessly
//!   compressed and a one-sample `Depth32Float_Stencil8` one, which answer Metal's: the
//!   three-level `RGBA16Float` ones the macOS 26 boots ask among them, and a seven-level 64x64
//!   `R32Uint` render target, 0x5800 bytes (`resource`'s
//!   `lossless_mipmapped_heap_texture_sizes_are_metals`);
//! - a two-plane XR10 texture (550, 551) in a heap takes the size and alignment
//!   `heapTextureSizeAndAlignWithDescriptor:` answers, Metal's two planes. A 2D one of one
//!   level, slice and sample keeps its planes in the heap's bytes where Metal lays them out, but
//!   any other's texels lie in memory of their own, apart from the heap's bytes, which the
//!   runtime announces (`heap-two-plane-texture`), where Metal's texture keeps its two planes
//!   there. So a buffer or texture placed over such a texture's range of the heap while it
//!   lives dies at the placement, as such a texture placed over a living one's does, the heap's
//!   cover, the buffer over all of it core makes with each heap, included: such a texture dies
//!   in every heap core makes ([`crate::gpu::resource::Placements`]); the heap's bytes there,
//!   read through the heap's address, a replaced backing or the guest's pages under a heap over
//!   address ranges, are not its texels either;
//! - a two-plane XR10 texture (550, 551) over plane 0 of a 'b3a8' surface is the runtime's, its
//!   colour words in plane 0 and its alpha bytes in plane 1, read, sampled, cleared, filled and
//!   copied to and from another texture of the format as Metal's (the tests compare them over the
//!   guest's shape and others); one over another surface or over plane 1, or rotated, which Metal
//!   makes, dies where it is made, naming the runtime's refusal; a copy within one dies at the
//!   copy, naming the runtime's refusal, where Metal copies;
//! - a texture of a two-plane YCbCr format over an IOSurface is the runtime's for the 24 formats
//!   it reads ([`crate::gpu::ycbcr::runtime_makes`], the packed 508 to 510 and 528 to 530 among
//!   them), Metal's bit for bit in every comparison the tests make (`ycbcr/sampling.rs`:
//!   `read()`, nearest samples under every address mode, gathers under the clamping ones and
//!   each border colour, and linear samples, over every OSType each is made over, and constexpr
//!   samplers). One over a surface whose OSType, plane, chroma or matrix the runtime does not
//!   read dies where it is made, where Metal makes it; and a command that samples one through a
//!   sampler slot that samples a texture of another map in the same stage, by a function that
//!   samples through handles able to reach it, or through a constexpr sampler whose modes are
//!   not stated or map to one a constexpr sampler cannot address, which the runtime cannot map,
//!   dies there.
//!
//! **Unannounced runtime gaps**, reported to the runtime, which announces nothing to die at:
//! none at the pinned runtime. Each would differ from Metal silently, as nothing here dies at it
//! either, and its test would pin the runtime's answer until the runtime announces it, then it
//! is placed by its reach.
//!
//! **Differences this platform announces itself**, each a warning once per process, until the
//! runtime announces a typed kind for it, which then places it by its reach: none. On Linux,
//! read-only memory (a range core maps read-only, as the EFI framebuffer; the alias of a
//! read-only replace or heap; a surface the GPU only reads) the runtime imports through a
//! writable twin of its pages, as Linux drivers pin imported pages for writing: the GPU reads
//! what Metal's reads, and its writes would reach the pages, where Metal drops them. Every
//! command that may write through an object over such an import dies where it is encoded
//! (`crate::gpu::writes::kept_write`; `resource`'s
//! `a_fill_into_read_only_pages_dies_on_linux`), but a function's write through an address it
//! loads into one not made resident for writing, which names nothing.

use std::sync::Arc;

use metal2vulkan::device::{VkExtension, VkFeature};
use metal2vulkan::error::Feature;
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime as rt;
use reims_vgpu_core::num::ToU64;
use reims_vgpu_core::platform::caps::{
    DeviceCaps, FAMILIES, Flags, HEAP_QUERY_OPTIONS, Limits, PRIMITIVE_TYPE_COUNT, SAMPLE_COUNTS,
    WRITE_ROUNDING_MODES,
};
use reims_vgpu_core::platform::types::{MtlSize, SizeAndAlign};
use reims_vgpu_core::raise::RaiseResult;

use crate::answer::{Policy, refused, send};
use crate::gpu::resource_options;

/// Metal's documented argument table of one stage: 31 buffers, 128 textures, 16 samplers.
const TABLE: (u64, u64, u64) = (31, 128, 16);

/// The entries of Metal's documented threadgroup memory table (`maxComputeLocalMemorySizes`).
const LOCAL_MEMORY_SIZES: u64 = 31;

/// Metal's documented alignment of a threadgroup memory length.
const THREADGROUP_MEMORY_ALIGNMENT: u64 = 16;

/// The public `MTLPrimitiveType`s: point, line, line strip, triangle, triangle strip, which the
/// runtime draws.
const PUBLIC_PRIMITIVE_TYPES: u64 = 5;

/// The stand-ins: the Metal platform's answers on an Apple M3 Max, where no documented fact
/// gives one (the module's table). The registry ID, which Metal answers at run time, is the
/// OS's ([`crate::os::default_registry_id`]).
pub mod stand_in {
    /// `supportsFamily:` of each of core's `FAMILIES`.
    pub const FAMILIES: [bool; 7] = [true, true, true, true, true, false, false];

    /// Whether the device answers the Apple families and the tile memory an Apple GPU has
    /// (`FAMILIES`, `TILE_SHADERS`, `IMAGE_BLOCKS`): yes unless `REIMS_VGPU_APPLE_FAMILIES` is
    /// `0`, when it answers no family, no tile shaders and no imageblocks, as a Mac's discrete
    /// GPU does. A macOS 26.4 guest's driver asks the device; told an Apple GPU, its compositor
    /// draws with tile dispatches, framebuffer fetch and memoryless targets, which the runtime
    /// runs through scratch images and the pixel interlock. Read once per process.
    pub fn apple_families() -> bool {
        static APPLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *APPLE
            .get_or_init(|| std::env::var_os("REIMS_VGPU_APPLE_FAMILIES").is_none_or(|v| v != "0"))
    }
    /// `supportsExtendedXR10Formats`.
    pub const EXTENDED_XR10_FORMATS: bool = true;
    /// The four texture alignment properties.
    pub const TEXTURE_ALIGNMENT_BYTES: u64 = 16;
    /// `maxComputeThreadgroupMemory` (private), which is not `maxThreadgroupMemoryLength`.
    pub const MAX_COMPUTE_THREADGROUP_MEMORY: u64 = 65536;
    /// `maxVertexAmplificationCount` (private), where the device takes that many.
    pub const MAX_VERTEX_AMPLIFICATION_COUNT: u64 = 2;
    /// `supportsTileShaders` and `supportsNonSquareTileShaders` (private).
    pub const TILE_SHADERS: bool = true;
    /// `supportsImageBlocks` (private).
    pub const IMAGE_BLOCKS: bool = true;
    /// `isLargeMRTSupported`.
    pub const LARGE_MRT: bool = true;
    /// `isRGB10A2GammaSupported`.
    pub const RGB10A2_GAMMA: bool = true;
    /// `supportsComputeCompressedTextureWrite`.
    pub const COMPUTE_COMPRESSED_TEXTURE_WRITE: bool = false;
    /// `supportsConditionalLoadStore`.
    pub const CONDITIONAL_LOAD_STORE: bool = true;
    /// `supportsFloat16BCubicFiltering`.
    pub const FLOAT16_BCUBIC_FILTERING: bool = true;
    /// `supportsMemoryOrderAtomics`: false, as Metal answers at macOS 26.7. Metal at 27.0.1
    /// answers true on the M3 Max (measured), but the runtime has no memory-order atomics, so this
    /// stays false until it has them; a guest that reads false does without them.
    pub const MEMORY_ORDER_ATOMICS: bool = false;
    /// `supportsTexture2DMultisampleArray`.
    pub const TEXTURE_2D_MULTISAMPLE_ARRAY: bool = true;
    /// `supportsPrimitiveType:` of the raw values 5 to 9.
    pub const PRIVATE_PRIMITIVE_TYPES: [bool; 5] = [true, true, true, true, false];
    /// `supportsTextureWriteRoundingMode:` of 1 and 2.
    pub const WRITE_ROUNDING_MODES: [bool; 2] = [true, true];
}

/// The runtime's answer of a query whose runtime piece may not be built yet: a gap dies at the
/// query's selector, as every send the runtime refuses where Metal answers does.
fn query<T>(sel: &'static str, answer: Result<T, rt::RuntimeGap>) -> T {
    answer.unwrap_or_else(|gap| refused(sel, &rt::Error::NotImplemented(gap)))
}

/// The largest count in `1..=64` that `accepts` takes, counting up from 1 until it refuses one; 0
/// when it refuses 1.
fn largest(sel: &'static str, accepts: impl Fn(u64) -> Result<bool, rt::RuntimeGap>) -> u64 {
    (1..=64u64)
        .take_while(|&count| query(sel, accepts(count)))
        .last()
        .unwrap_or(0)
}

/// The rows of Apple's feature tables each SIMD flag of key 33 promises, in every stage:
/// `supports_simd_reduction` (0x40) the SIMD- and quad-scoped reductions, and
/// `supports_simd_shuffle_and_fill` (0x100) SIMD shift and fill, whose row covers its quad forms.
const SIMD_ROWS: [&[rt::FeatureRow]; 2] = [
    &[rt::FeatureRow::SimdReduction, rt::FeatureRow::QuadReduction],
    &[rt::FeatureRow::SimdShiftAndFill],
];

/// Whether every row of each of [`SIMD_ROWS`] is exact on `device`: offered in every stage,
/// refusing nothing Metal takes, in the runtime's audit of the Apple9 column, which lists all
/// three rows, as `supportsFamily:` judges them.
fn simd_rows_exact(device: &rt::Device) -> [bool; 2] {
    let audit = device.family_rows(rt::GpuFamily::Apple9);
    SIMD_ROWS.map(|rows| {
        rows.iter().all(|&row| {
            audit
                .iter()
                .any(|a| a.row == rt::FamilyRow::Feature(row) && a.state == rt::RowState::Exact)
        })
    })
}

/// The answers computed from documented facts, for Apple's private properties the runtime does
/// not answer: limits and flags, and `maxVertexAmplificationCount`.
fn computed(device: &rt::Device) -> (u64, Limits, Flags) {
    let props = device.properties();
    let metal = device.compile_device().metal_properties();
    let l = &props.description.limits;
    let (buffers, textures, samplers) = TABLE;
    let buffers = buffers.min(u64::from(l.max_per_stage_descriptor_storage_buffers));
    let textures = textures.min(u64::from(l.max_per_stage_descriptor_sampled_images));
    let samplers = samplers.min(u64::from(l.max_per_stage_descriptor_samplers));
    let amplification = largest("supportsVertexAmplificationCount:", |count| {
        props.supports_vertex_amplification_count(count)
    });
    let rate_map_layers = largest("supportsRasterizationRateMapWithLayerCount:", |count| {
        props.supports_rasterization_rate_map_with_layer_count(count)
    });
    let viewports = if props.description.has(VkFeature::MultiViewport) {
        u64::from(props.max_viewports)
    } else {
        1
    };
    let [simd_reduction, simd_shuffle_and_fill] = simd_rows_exact(device);
    let imports_host_memory = device
        .enabled()
        .extensions
        .contains(&VkExtension::ExtExternalMemoryHost);
    let refuses = |feature: Feature| {
        metal
            .unsupported_features
            .iter()
            .any(|u| u.feature == feature)
    };
    let limits = Limits {
        device_linear_texture_alignment_bytes: stand_in::TEXTURE_ALIGNMENT_BYTES,
        iosurface_read_only_texture_alignment_bytes: stand_in::TEXTURE_ALIGNMENT_BYTES,
        iosurface_texture_alignment_bytes: stand_in::TEXTURE_ALIGNMENT_BYTES,
        linear_texture_alignment_bytes: stand_in::TEXTURE_ALIGNMENT_BYTES,
        max_compute_buffers: buffers,
        max_compute_local_memory_sizes: LOCAL_MEMORY_SIZES,
        max_compute_samplers: samplers,
        max_compute_textures: textures,
        max_compute_threadgroup_memory: stand_in::MAX_COMPUTE_THREADGROUP_MEMORY,
        max_compute_threadgroup_memory_alignment_bytes: THREADGROUP_MEMORY_ALIGNMENT,
        max_fragment_buffers: buffers,
        max_fragment_samplers: samplers,
        max_fragment_textures: textures,
        // The adapter's predication nests without limit (`predication`).
        max_predicated_nesting_depth: u64::from(u32::MAX),
        max_rasterization_rate_layer_count: rate_map_layers,
        max_texture_height_2d: u64::from(l.max_image_dimension_2d),
        max_texture_layers: u64::from(l.max_image_array_layers),
        max_texture_width_2d: u64::from(l.max_image_dimension_2d),
        max_threadgroup_memory_length: props.max_threadgroup_memory_length,
        max_tile_buffers: buffers,
        max_tile_samplers: samplers,
        max_tile_textures: textures,
        max_total_compute_threads_per_threadgroup: u64::from(
            metal.max_total_threads_per_threadgroup,
        ),
        max_vertex_buffers: buffers,
        max_vertex_samplers: samplers,
        max_vertex_textures: textures,
        max_viewport_count: viewports,
    };
    let flags = Flags {
        are_raster_order_groups_supported: query(
            "areRasterOrderGroupsSupported",
            props.are_raster_order_groups_supported(),
        ),
        is_depth24_stencil8_pixel_format_supported: query(
            "isDepth24Stencil8PixelFormatSupported",
            props.is_depth24_stencil8_pixel_format_supported(),
        ),
        is_framebuffer_read_supported: !refuses(Feature::FramebufferFetch),
        is_large_mrt_supported: stand_in::LARGE_MRT,
        is_rgb10a2_gamma_supported: stand_in::RGB10A2_GAMMA,
        supports_buffer_with_address_ranges: imports_host_memory,
        // This platform makes a buffer at an IOSurface's base (`new_buffer_on_surface`).
        supports_buffer_with_iosurface: true,
        // The adapter builds command predication (`predication`).
        supports_command_buffer_jump: true,
        supports_compute_compressed_texture_write: stand_in::COMPUTE_COMPRESSED_TEXTURE_WRITE,
        supports_conditional_load_store: stand_in::CONDITIONAL_LOAD_STORE,
        supports_dynamic_attribute_stride: props.description.has(VkFeature::ExtendedDynamicState),
        supports_extended_xr10_formats: stand_in::EXTENDED_XR10_FORMATS,
        supports_float16_bcubic_filtering: stand_in::FLOAT16_BCUBIC_FILTERING,
        supports_heap_with_address_ranges: imports_host_memory,
        supports_image_blocks: stand_in::IMAGE_BLOCKS && stand_in::apple_families(),
        supports_memory_order_atomics: stand_in::MEMORY_ORDER_ATOMICS,
        supports_native_hardware_fp16: props.description.has(VkFeature::ShaderFloat16),
        supports_non_square_tile_shaders: stand_in::TILE_SHADERS && stand_in::apple_families(),
        supports_programmable_sample_positions: query(
            "areProgrammableSamplePositionsSupported",
            props.are_programmable_sample_positions_supported(),
        ),
        supports_resource_detach_backing: imports_host_memory,
        // The runtime's shared texture handle, in this process.
        supports_shared_texture_handles: true,
        supports_simd_reduction: simd_reduction,
        supports_simd_shuffle_and_fill: simd_shuffle_and_fill,
        supports_texture_2d_multisample_array: stand_in::TEXTURE_2D_MULTISAMPLE_ARRAY,
        supports_tile_shaders: stand_in::TILE_SHADERS && stand_in::apple_families(),
    };
    let amplification = amplification.min(stand_in::MAX_VERTEX_AMPLIFICATION_COUNT);
    (amplification, limits, flags)
}

/// `gpuCoreCount` (private, the guest's device-info key 34) of the device named `name`, whose
/// driver reports `reported` cores and which this platform answers registry ID `registry_id` for:
/// the device's own GPU's cores, never another GPU's.
/// - On macOS, for the device this platform answers Metal's registry ID for (the runtime's system
///   default device, which MoltenVK runs on Metal's default GPU, [`crate::os::default_registry_id`]):
///   Metal's own answer for that GPU ([`crate::os::default_gpu_core_count`]), 40 on the M3 Max.
/// - Else the driver's count, where it reports one (`rt::DeviceProperties::gpu_core_count`:
///   NVIDIA's streaming multiprocessors, 60 on an RTX 5080 Laptop GPU; AMD's compute units; Arm's
///   shader cores).
/// - Else none (Intel's drivers, lavapipe, and a device MoltenVK runs on another GPU than Metal's
///   default report no count): core writes no key 34 pair, as Apple's framework writes none for a
///   device that does not respond to `gpuCoreCount`, and the guest's device then answers 0. Logged
///   once, as the device's caps are read, naming the kind (`gpu-core-count-unknown`).
fn gpu_core_count(name: &str, reported: Option<u32>, registry_id: u64) -> Option<u64> {
    let metals = if crate::os::default_registry_id() == Some(registry_id) {
        crate::os::default_gpu_core_count()
    } else {
        None
    };
    let count = metals.or(reported.map(u64::from));
    if count.is_none() {
        log::warn!(
            target: "reims_vgpu_vulkan::caps",
            "gpuCoreCount (gpu-core-count-unknown): {name}: its driver reports no core count and              no Metal device stands for it, so the guest's device info carries none (key 34) and              its device answers 0"
        );
    }
    count
}

/// `heapBufferSizeAndAlignWithLength:options:` of (`length`, `options`).
fn heap_buffer_size_and_align(
    device: &rt::Device,
    length: u64,
    options: u64,
) -> RaiseResult<SizeAndAlign> {
    let sel = "heapBufferSizeAndAlignWithLength:options:";
    let options = resource_options(sel, options);
    send(sel, Policy::Caught, || {
        let answer = device.heap_buffer_size_and_align(length, options);
        Ok(SizeAndAlign {
            size: answer.size,
            align: answer.align,
        })
    })
}

/// Every answer of `device`, read once. `registry_id` is the one this platform answers for it
/// (`device::registry_id`), and `page_size` the length of the heap-buffer query
/// (`vm_page_size`).
pub fn read(device: &rt::Device, registry_id: u64, page_size: u64) -> Arc<DeviceCaps> {
    let props = device.properties();
    let (max_vertex_amplification_count, limits, flags) = computed(device);
    let rt::Size {
        width,
        height,
        depth,
    } = props.max_threads_per_threadgroup;
    Arc::new(DeviceCaps {
        name: Some(props.name.clone()),
        registry_id,
        gpu_core_count: gpu_core_count(&props.name, props.gpu_core_count, registry_id),
        max_threads_per_threadgroup: MtlSize::new(width, height, depth),
        max_vertex_amplification_count,
        limits,
        flags,
        sample_counts: SAMPLE_COUNTS.map(|count| {
            Ok(query(
                "supportsTextureSampleCount:",
                props.supports_texture_sample_count(count),
            ))
        }),
        primitive_types: std::array::from_fn(|t| {
            let t = t.to_u64();
            Ok(match t.checked_sub(PUBLIC_PRIMITIVE_TYPES) {
                None => true,
                Some(private) => stand_in::PRIVATE_PRIMITIVE_TYPES
                    .get(usize::try_from(private).unwrap_or(usize::MAX))
                    .copied()
                    .unwrap_or(false),
            })
        }),
        families: if stand_in::apple_families() {
            stand_in::FAMILIES
        } else {
            [false; 7]
        }
        .map(Ok),
        write_rounding_modes: stand_in::WRITE_ROUNDING_MODES.map(Ok),
        heap_buffer_page_private: heap_buffer_size_and_align(device, page_size, HEAP_QUERY_OPTIONS),
        min_linear_alignment_bgra8: send(
            "minimumLinearTextureAlignmentForPixelFormat:",
            Policy::Caught,
            || props.minimum_linear_texture_alignment_for_pixel_format(PixelFormat::BGRA8Unorm),
        ),
    })
}

// The tables core asks are as long as the answers here hold.
const _: () = assert!(FAMILIES.len() == stand_in::FAMILIES.len());
const _: () = assert!(PRIMITIVE_TYPE_COUNT == PUBLIC_PRIMITIVE_TYPES + 5);
const _: () = assert!(WRITE_ROUNDING_MODES.len() == stand_in::WRITE_ROUNDING_MODES.len());

#[cfg(test)]
mod tests {
    use reims_vgpu_core::platform::device::GpuDevice;

    use super::*;
    use crate::gpu::device::tests::dev;

    /// `gpuCoreCount` of a device no Metal device stands for is its driver's count, and none
    /// where its driver reports none: never another GPU's
    #[test]
    fn gpu_core_counts_are_the_devices_own() {
        let elsewhere = crate::os::default_registry_id().map_or(1, |id| id ^ 1);
        assert_eq!(
            gpu_core_count("an RTX 5080 Laptop GPU", Some(60), elsewhere),
            Some(60)
        );
        assert_eq!(gpu_core_count("llvmpipe", None, elsewhere), None);
    }

    /// The SIMD flags answer whether the runtime's audit finds their rows exact ([`SIMD_ROWS`]):
    /// yes for both at the pinned runtime, every row exact on this host
    #[test]
    fn simd_flags_answer_the_runtimes_family_rows() {
        let d = dev();
        let flags = &d.caps().flags;
        let answers = [
            flags.supports_simd_reduction,
            flags.supports_simd_shuffle_and_fill,
        ];
        assert_eq!(answers, simd_rows_exact(&d.raw));
        let audit = d.raw.family_rows(rt::GpuFamily::Apple9);
        for &row in SIMD_ROWS.iter().flat_map(|rows| rows.iter()) {
            let state = audit
                .iter()
                .find(|a| a.row == rt::FamilyRow::Feature(row))
                .map(|a| &a.state);
            assert_eq!(state, Some(&rt::RowState::Exact), "{row:?}");
        }
        assert_eq!(answers, [true, true]);
    }

    #[cfg(target_os = "macos")]
    mod metal_tests;
}
