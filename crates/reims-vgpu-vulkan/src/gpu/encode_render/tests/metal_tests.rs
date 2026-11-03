//! The tests of `encode_render` that compare with Metal on this host: macOS only.

use std::sync::Arc;

use reims_vgpu_core::display::{POS_DATA, TEX_DATA};
use reims_vgpu_core::tasks::launcher::{HostSide, ServiceLink, TaskLauncher};
use reims_vgpu_metal::MetalPlatform;
use reims_vgpu_metal::os::iosurface::MetalSurface;

use super::*;

/// `MTLPixelFormatRGBA8Unorm`, `BGRA8Unorm`, `RGBA16Float` and `R32Float`.
const RGBA8: u64 = 70;

const R32F: u64 = 55;

/// `MTLResourceStorageModeManaged`: the property buffer's options.
const MANAGED: u64 = 0x10;

/// A launcher no test launches through.
struct NoLauncher;

impl TaskLauncher<MetalPlatform> for NoLauncher {
    fn launch(
        &self,
        _platform: &Arc<MetalPlatform>,
        _host: Arc<dyn HostSide<MetalPlatform>>,
    ) -> Arc<dyn ServiceLink<MetalPlatform>> {
        panic!("no task is launched in these tests")
    }
}

/// The fragment properties as core writes them: the matrix's four columns of three (each
/// padded to four), then the slopes at 0 and at 1.
fn properties(matrix: [[f32; 4]; 3], at_zero: [f32; 3], at_one: [f32; 3]) -> [f32; 24] {
    let mut p = [0.0f32; 24];
    for (col, column) in p.chunks_mut(4).take(4).enumerate() {
        for (row, v) in column.iter_mut().take(3).enumerate() {
            *v = matrix[row][col];
        }
    }
    p[16..19].copy_from_slice(&at_zero);
    p[19..22].copy_from_slice(&at_one);
    p
}

/// A gamma pass's table (the same ramp in each of the three slices) and properties.
struct Gamma {
    ramp: Vec<f32>,
    properties: [f32; 24],
}

/// The gamma table texture core makes: a 1D array of three `R32Float` slices of `ramp`.
fn gamma_table<P: Platform>(p: &P, d: &P::Device, ramp: &[f32]) -> P::Texture {
    let n = u64::try_from(ramp.len()).unwrap();
    let desc = p.texture_descriptor(&TextureDesc {
        texture_type: 1,
        pixel_format: R32F,
        width: n,
        array_length: 3,
        usage: READ,
        ..TextureDesc::default()
    });
    let t = d.new_texture(OnFault::Die, &desc).unwrap().unwrap();
    let row = bytes(ramp);
    let region = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(n, 1, 1));
    for slice in 0..3 {
        t.replace_region(region, 0, slice, HostRange::of_slice(&row), n * 4, n * 4)
            .unwrap();
    }
    t
}

/// The present pass as core's `display/render.rs` encodes it, from `frame` into a BGRA8
/// target of its size, with the display's pipeline of core's cache (the plain copy with
/// `gamma` `None`, else the gamma shader): the target cleared to opaque black, the pipeline,
/// the viewport of the target with `znear = zfar = 0`, the strip's buffers at vertex 0 and 1,
/// the gamma table (nil without one) at fragment texture 1, the frame at 0, the Managed
/// property buffer at fragment buffer 0, a four-vertex strip. The target's pixels, read back
/// by a blit, as RGBA.
fn present<P: Platform>(
    p: &P,
    d: &P::Device,
    frame: &P::Texture,
    gamma: Option<&Gamma>,
) -> Vec<[u8; 4]> {
    present_as(p, d, frame, gamma, Variant::Strip)
}

/// How a test pass draws after the present pass's sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    /// The present pass's strip.
    Strip,
    /// The strip of 16-bit indices 0 to 3.
    Indexed,
    /// The strip of indices 1 to 4 at a base vertex of -1, one instance.
    IndexedBase,
    /// The strip as one instance of an instanced draw at base instance 0.
    Instances,
    /// The strip under a scissor rectangle of the target's left half.
    Scissor,
    /// The indexed strip with index type 2, which Metal draws nothing for.
    IndexType2,
    /// The strip's four vertices as primitive type 7, one quad.
    Primitive7,
    /// `count` of [`scattered`]'s vertices from `start` as private primitive type
    /// `primitive_type`, `instances` instances of them.
    Private {
        primitive_type: u64,
        start: u64,
        count: u64,
        instances: u64,
    },
    /// The strip as an instanced draw of no instances, direct or indexed.
    NoInstances { indexed: bool },
    /// The strip after a nil depth-stencil state, Metal's default.
    NilDepthStencil,
    /// The strip after a late colour resolve where the pass named none.
    UnnamedColorResolve,
}

/// The commands of `variant` that end a test pass before `endEncoding`.
fn variant_cmds<'a, P: Platform>(
    variant: Variant,
    indices: &'a P::Buffer,
    frame: &'a P::Texture,
    (width, height): (u64, u64),
) -> Vec<RenderCmd<'a, P>> {
    let strip = RenderCmd::DrawPrimitives {
        primitive_type: STRIP,
        vertex_start: 0,
        vertex_count: 4,
    };
    let indexed = |index_type| RenderCmd::DrawIndexedPrimitives {
        primitive_type: STRIP,
        index_count: 4,
        index_type,
        index_buffer: indices,
        index_buffer_offset: 0,
    };
    match variant {
        Variant::Strip => vec![strip],
        Variant::Indexed => vec![indexed(U16)],
        Variant::IndexedBase => vec![RenderCmd::DrawIndexedPrimitivesBase {
            primitive_type: STRIP,
            index_count: 4,
            index_type: U16,
            index_buffer: indices,
            index_buffer_offset: 8,
            instance_count: 1,
            base_vertex: -1,
            base_instance: 0,
        }],
        Variant::Instances => vec![RenderCmd::DrawPrimitivesBase {
            primitive_type: STRIP,
            vertex_start: 0,
            vertex_count: 4,
            instance_count: 1,
            base_instance: 0,
        }],
        Variant::Scissor => vec![
            RenderCmd::SetScissorRect {
                rect: MtlScissorRect {
                    x: 0,
                    y: 0,
                    width: width / 2,
                    height,
                },
            },
            strip,
        ],
        Variant::IndexType2 => vec![indexed(2)],
        Variant::Primitive7 => vec![RenderCmd::DrawPrimitives {
            primitive_type: 7,
            vertex_start: 0,
            vertex_count: 4,
        }],
        Variant::Private {
            primitive_type,
            start,
            count,
            instances,
        } => vec![RenderCmd::DrawPrimitivesInstances {
            primitive_type,
            vertex_start: start,
            vertex_count: count,
            instance_count: instances,
        }],
        Variant::NoInstances { indexed: false } => vec![RenderCmd::DrawPrimitivesInstances {
            primitive_type: STRIP,
            vertex_start: 0,
            vertex_count: 4,
            instance_count: 0,
        }],
        Variant::NoInstances { indexed: true } => {
            vec![RenderCmd::DrawIndexedPrimitivesInstances {
                primitive_type: STRIP,
                index_count: 4,
                index_type: U16,
                index_buffer: indices,
                index_buffer_offset: 0,
                instance_count: 0,
            }]
        }
        Variant::NilDepthStencil => vec![
            RenderCmd::SetDepthStencilState {
                depth_stencil_state: None,
            },
            strip,
        ],
        Variant::UnnamedColorResolve => vec![
            RenderCmd::SetColorResolveTexture {
                texture: Some(frame),
                slice: 0,
                depth_plane: 0,
                level: 0,
                y_invert: true,
                index: 0,
            },
            strip,
        ],
    }
}

/// [`present`]'s pass, drawn as `variant` says.
fn present_as<P: Platform>(
    p: &P,
    d: &P::Device,
    frame: &P::Texture,
    gamma: Option<&Gamma>,
    variant: Variant,
) -> Vec<[u8; 4]> {
    let (w, h) = (frame.width(), frame.height());
    let index_bytes: Vec<u8> = [0u16, 1, 2, 3, 1, 2, 3, 4]
        .iter()
        .flat_map(|i| i.to_ne_bytes())
        .collect();
    let indices = buffer::<P>(d, 16, 0, &index_bytes);
    let cache = DisplayPipelineCache::<P>::new(d.clone_ref()).unwrap();
    let pipeline = cache
        .pipeline_for(PipelineKey {
            format: BGRA8,
            gamma: gamma.is_some(),
        })
        .unwrap();
    let dst = texture(p, d, BGRA8, TARGET | READ, (w, h), None);
    let (pos_data, tex_data) = match variant {
        Variant::Private { .. } => scattered(),
        _ => (POS_DATA.to_vec(), TEX_DATA.to_vec()),
    };
    let length = |data: &[f32]| u64::try_from(data.len() * 4).unwrap();
    let pos = buffer::<P>(d, length(&pos_data), 0, &bytes(&pos_data));
    let tex = buffer::<P>(d, length(&tex_data), 0, &bytes(&tex_data));
    let identity = properties(IDENTITY, [0.0; 3], [0.0; 3]);
    let props = gamma.map_or(identity, |g| g.properties);
    let props = buffer::<P>(d, 96, MANAGED, &bytes(&props));
    props
        .did_modify_range(OnFault::Die, NsRange::new(0, 96))
        .unwrap();
    let table = gamma.map(|g| gamma_table(p, d, &g.ramp));
    let pass = p.render_pass_descriptor();
    pass.set_color_attachment(
        OnFault::Die,
        0,
        &ColorAttachmentSet {
            texture: Some(Some(&dst)),
            load_action: Some(CLEAR),
            store_action: Some(STORE),
            clear_color: Some(ClearColor {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha: 1.0,
            }),
            ..ColorAttachmentSet::default()
        },
    )
    .unwrap();
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    #[expect(clippy::as_conversions, reason = "small sizes, exact in f64")]
    let viewport = MtlViewport {
        origin_x: 0.0,
        origin_y: 0.0,
        width: w as f64,
        height: h as f64,
        znear: 0.0,
        zfar: 0.0,
    };
    for cmd in [
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&pipeline),
        },
        RenderCmd::SetViewport { viewport },
        RenderCmd::SetVertexBuffer {
            buffer: Some(&pos),
            offset: 0,
            index: 0,
        },
        RenderCmd::SetVertexBuffer {
            buffer: Some(&tex),
            offset: 0,
            index: 1,
        },
        RenderCmd::SetFragmentTexture {
            texture: table.as_ref(),
            index: 1,
        },
        RenderCmd::SetFragmentTexture {
            texture: Some(frame),
            index: 0,
        },
        RenderCmd::SetFragmentBuffer {
            buffer: Some(&props),
            offset: 0,
            index: 0,
        },
    ]
    .into_iter()
    .chain(variant_cmds::<P>(variant, &indices, frame, (w, h)))
    .chain([RenderCmd::EndEncoding])
    {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    let out = buffer::<P>(d, w * h * 4, 0, &[]);
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    blit.encode(
        OnFault::Die,
        BlitCmd::CopyTextureToBuffer {
            source_texture: &dst,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(w, h, 1),
            destination_buffer: &out,
            destination_offset: 0,
            destination_bytes_per_row: w * 4,
            destination_bytes_per_image: w * h * 4,
        },
    )
    .unwrap();
    blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    let n = usize::try_from(w * h * 4).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the blit completed.
    let texels = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) };
    texels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| [b[2], b[1], b[0], b[3]])
        .collect()
}

/// Runs `check` on this platform and, as the reference, on Metal when the host has it.
fn on_both(check: impl Fn(&dyn Fn(Option<&Gamma>, Frame) -> Vec<[u8; 4]>)) {
    let p = platform();
    let d = dev();
    check(&|gamma, frame| {
        let f = frame.make(&p, &d);
        present(&p, &d, &f, gamma)
    });
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new(Box::new(NoLauncher));
    check(&|gamma, frame| {
        let f = frame.make(&mp, &md);
        present(&mp, &md, &f, gamma)
    });
}

/// A frame to present: its format, size and texels.
#[derive(Clone, Copy)]
struct Frame<'a> {
    format: u64,
    size: (u64, u64),
    texels: &'a [u8],
    bytes_per_texel: u64,
}

impl Frame<'_> {
    fn make<P: Platform>(&self, p: &P, d: &P::Device) -> P::Texture {
        let row = self.size.0 * self.bytes_per_texel;
        texture(p, d, self.format, READ, self.size, Some((self.texels, row)))
    }
}

/// An RGBA8 frame of `px`, `w` × `h`.
fn rgba8(px: &[[u8; 4]], size: (u64, u64)) -> (Vec<u8>, (u64, u64)) {
    (px.as_flattened().to_vec(), size)
}

/// The bytes of half floats.
fn halves(h: &[[u16; 4]]) -> Vec<u8> {
    h.iter().flatten().flat_map(|v| v.to_ne_bytes()).collect()
}

/// The copy draws the frame upright, pixel for pixel, alpha included: neither premultiplied
/// nor blended (`shaders.rs`'s Metal expectation)
#[test]
fn the_copy_passes_pixels_and_alpha_through() {
    let px: Vec<[u8; 4]> = (0..32u8)
        .map(|i| {
            [
                i.wrapping_mul(37),
                200 - i,
                i.wrapping_mul(5) + 3,
                [0, 0x7f, 0xff, 0x80][usize::from(i % 4)],
            ]
        })
        .collect();
    let (texels, size) = rgba8(&px, (8, 4));
    on_both(|present| {
        let frame = Frame {
            format: RGBA8,
            size,
            texels: &texels,
            bytes_per_texel: 4,
        };
        assert_eq!(present(None, frame), px);
    });
}

/// An RGBA16Float frame is converted to unorm8 by the target: clamped, rounded to nearest
#[test]
fn the_copy_clamps_extended_range_in_the_target() {
    // -0.5, 0.5, 1.75, 0.25 and 1.0, 0.0, 0.5, 0.75, as half floats.
    let texels = halves(&[
        [0xb800, 0x3800, 0x3f00, 0x3400],
        [0x3c00, 0, 0x3800, 0x3a00],
    ]);
    on_both(|present| {
        let frame = Frame {
            format: RGBA16F,
            size: (2, 1),
            texels: &texels,
            bytes_per_texel: 8,
        };
        assert_eq!(
            present(None, frame),
            vec![[0, 0x80, 0xff, 0x40], [0xff, 0, 0x80, 0xbf]]
        );
    });
}

/// With identity ramps (256 entries) and the identity matrix, the gamma shader is the copy
/// for colour, and every alpha is 0xff
#[test]
fn identity_gamma_is_the_copy_with_opaque_alpha() {
    let px: Vec<[u8; 4]> = (0..=255u8)
        .map(|i| [i, 255 - i, i.wrapping_mul(7), i.wrapping_mul(3)])
        .collect();
    let (texels, size) = rgba8(&px, (16, 16));
    let gamma = Gamma {
        ramp: (0..256u16).map(|i| f32::from(i) / 255.0).collect(),
        properties: properties(IDENTITY, [1.0; 3], [1.0; 3]),
    };
    on_both(|present| {
        let frame = Frame {
            format: RGBA8,
            size,
            texels: &texels,
            bytes_per_texel: 4,
        };
        let out = present(Some(&gamma), frame);
        let opaque: Vec<_> = px.iter().map(|p| [p[0], p[1], p[2], 0xff]).collect();
        assert_eq!(out, opaque);
    });
}

/// A two-entry ramp is a threshold at 0.5, not an interpolation
#[test]
fn a_two_entry_table_is_a_threshold() {
    let (texels, size) = rgba8(
        &[[0x7f, 0x80, 0x00, 0x10], [0xff, 0x40, 0xc0, 0x20]],
        (2, 1),
    );
    let gamma = Gamma {
        ramp: vec![0.25, 0.75],
        properties: properties(IDENTITY, [0.5; 3], [0.5; 3]),
    };
    on_both(|present| {
        let frame = Frame {
            format: RGBA8,
            size,
            texels: &texels,
            bytes_per_texel: 4,
        };
        // 0.25 is 63.75, 0.75 191.25, each rounded to nearest.
        assert_eq!(
            present(Some(&gamma), frame),
            vec![[0x40, 0xbf, 0x40, 0xff], [0xbf, 0x40, 0xbf, 0xff]]
        );
    });
}

/// Row c of the matrix gives channel c from the looked-up colour and the fourth column
#[test]
fn the_matrix_follows_the_table() {
    let (texels, size) = rgba8(
        &[[0xff, 0x00, 0x80, 0xff], [0x00, 0xff, 0x00, 0x00]],
        (2, 1),
    );
    // Red and blue swapped; green is half of green plus 0.25.
    let matrix = [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.5, 0.0, 0.25],
        [1.0, 0.0, 0.0, 0.0],
    ];
    let gamma = Gamma {
        ramp: (0..256u16).map(|i| f32::from(i) / 255.0).collect(),
        properties: properties(matrix, [1.0; 3], [1.0; 3]),
    };
    on_both(|present| {
        let frame = Frame {
            format: RGBA8,
            size,
            texels: &texels,
            bytes_per_texel: 4,
        };
        assert_eq!(
            present(Some(&gamma), frame),
            vec![[0x80, 0x40, 0xff, 0xff], [0x00, 0xbf, 0x00, 0xff]]
        );
    });
}

/// Outside [0, 1] the table's end entry is extended by its slope at that end
#[test]
fn out_of_range_components_follow_the_end_slopes() {
    // -0.25, 1.5, 0.5 as half floats.
    let texels = halves(&[[0xb400, 0x3e00, 0x3800, 0x3c00]]);
    let gamma = Gamma {
        ramp: vec![0.2, 0.4, 0.6],
        // Red's slope at 0 is 0.4 (-0.25 gives 0.1); green's at 1 is 0.2 (1.5 gives 0.7).
        properties: properties(IDENTITY, [0.4, 9.0, 9.0], [9.0, 0.2, 9.0]),
    };
    on_both(|present| {
        let frame = Frame {
            format: RGBA16F,
            size: (1, 1),
            texels: &texels,
            bytes_per_texel: 8,
        };
        // 0.1 is 25.5, 0.7 178.5, 0.4 102.
        assert_eq!(present(Some(&gamma), frame), vec![[0x1a, 0xb3, 0x66, 0xff]]);
    });
}

/// `MTLPixelFormatR32Uint`.
const R32UINT: u64 = 53;

/// An RGBA8 frame of 8 × 4 distinct opaque pixels.
fn opaque_frame() -> (Vec<[u8; 4]>, Vec<u8>) {
    let px: Vec<[u8; 4]> = (0..32u8)
        .map(|i| [i.wrapping_mul(29), 250 - i, i.wrapping_mul(3) + 1, 0xff])
        .collect();
    let texels = px.as_flattened().to_vec();
    (px, texels)
}

/// Each variant on this platform, and on Metal where the host has it when `on_metal`.
fn variant_on_both(variant: Variant, on_metal: bool) -> Vec<[u8; 4]> {
    let (_, texels) = opaque_frame();
    let frame = Frame {
        format: RGBA8,
        size: (8, 4),
        texels: &texels,
        bytes_per_texel: 4,
    };
    let (p, d) = (platform(), dev());
    let vulkan = present_as(&p, &d, &frame.make(&p, &d), None, variant);
    if on_metal && let Some(md) = MetalPlatform::system_default_device() {
        let mp = MetalPlatform::new(Box::new(NoLauncher));
        let metal = present_as(&mp, &md, &frame.make(&mp, &md), None, variant);
        assert_eq!(vulkan, metal, "{variant:?}");
    }
    vulkan
}

/// Indexed, based and instanced draws of the strip draw what the strip draws, as on Metal
#[test]
fn draw_forms_draw_the_strip() {
    let (px, _) = opaque_frame();
    for variant in [
        Variant::Indexed,
        Variant::IndexedBase,
        Variant::Instances,
        Variant::NilDepthStencil,
    ] {
        assert_eq!(variant_on_both(variant, true), px, "{variant:?}");
    }
    assert_eq!(variant_on_both(Variant::UnnamedColorResolve, false), px);
}

/// A scissor rectangle keeps the draw to it; the rest keeps the clear colour, as on Metal
#[test]
fn a_scissor_rectangle_bounds_the_draw() {
    let (px, _) = opaque_frame();
    let out = variant_on_both(Variant::Scissor, true);
    for (i, (got, want)) in out.iter().zip(&px).enumerate() {
        let expect = if i % 8 < 4 { *want } else { [0, 0, 0, 0xff] };
        assert_eq!(*got, expect, "pixel {i}");
    }
}

/// An index type no `MTLIndexType` is, and a draw of no instances, draw nothing, as on Metal
#[test]
fn an_index_type_metal_lacks_and_no_instances_draw_nothing() {
    for variant in [
        Variant::IndexType2,
        Variant::NoInstances { indexed: false },
        Variant::NoInstances { indexed: true },
    ] {
        assert_eq!(
            variant_on_both(variant, true),
            vec![[0, 0, 0, 0xff]; 32],
            "{variant:?}"
        );
    }
}

/// Metal's private primitive types 5 to 8 (a line loop, a triangle fan, quads and a quad
/// strip) draw what Metal draws, pixel for pixel, at every count from 0 to 10, from vertex 0
/// and from vertex 2 as two instances; one of too few vertices draws nothing
#[test]
fn private_primitive_types_draw_as_on_metal() {
    let clear = vec![[0, 0, 0, 0xff]; 32];
    assert_ne!(variant_on_both(Variant::Primitive7, true), clear);
    let mut drawn = 0;
    for primitive_type in 5..=8 {
        for count in 0..=10 {
            for (start, instances) in [(0, 1), (2, 2)] {
                let variant = Variant::Private {
                    primitive_type,
                    start,
                    count,
                    instances,
                };
                let out = variant_on_both(variant, true);
                if count < 2
                    || (primitive_type >= 6 && count < 3)
                    || (primitive_type >= 7 && count < 4)
                {
                    assert_eq!(out, clear, "{variant:?}");
                }
                drawn += usize::from(out != clear);
            }
        }
    }
    assert!(drawn > 40, "{drawn} draws drew anything");
}

/// The pixels of [`pass_pixels`] after `instances` instances of `count` of [`scattered`]'s
/// vertices from `start`, as `primitive_type`, under `setProvokingVertexMode:` of `mode` and
/// a state of the fixtures `functions`, over [`flat_vertices`] in buffers 0 and 1.
fn flat_pass<P: Platform>(
    p: &P,
    d: &P::Device,
    functions: (&str, &str),
    mode: u64,
    (primitive_type, start, count, instances): (u64, u64, u64, u64),
) -> Vec<[u8; 4]> {
    let draw = RenderCmd::DrawPrimitivesInstances {
        primitive_type,
        vertex_start: start,
        vertex_count: count,
        instance_count: instances,
    };
    flat_draw(p, d, functions, mode, draw)
}

/// The pixels of [`flat_pass`] after `draw` in place of its own.
fn flat_draw<P: Platform>(
    p: &P,
    d: &P::Device,
    functions: (&str, &str),
    mode: u64,
    draw: RenderCmd<'_, P>,
) -> Vec<[u8; 4]> {
    set_draw(
        p,
        d,
        functions,
        RenderCmd::SetProvokingVertexMode { mode },
        draw,
    )
}

/// The pixels of [`flat_draw`] after `setter` in place of its `setProvokingVertexMode:`.
fn set_draw<P: Platform>(
    p: &P,
    d: &P::Device,
    (vertex, fragment): (&str, &str),
    setter: RenderCmd<'_, P>,
    draw: RenderCmd<'_, P>,
) -> Vec<[u8; 4]> {
    let state = fixture_state::<P>(d, vertex, fragment);
    let (pos, colours) = flat_vertices::<P>(d, 16);
    pass_pixels(
        p,
        d,
        vec![
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            },
            RenderCmd::SetVertexBuffer {
                buffer: Some(&pos),
                offset: 0,
                index: 0,
            },
            RenderCmd::SetVertexBuffer {
                buffer: Some(&colours),
                offset: 0,
                index: 1,
            },
            setter,
            draw,
        ],
    )
}

/// A list or strip of triangles under a state with a flat input draws what Metal draws at
/// every provoking vertex mode, the last vertex's colour filling each triangle after a
/// nonzero one, from vertex 0 and later, one instance and two
#[test]
fn a_flat_input_takes_the_provoking_vertex_metal_takes() {
    let (p, d) = (platform(), dev());
    let metal = MetalPlatform::system_default_device()
        .map(|md| (MetalPlatform::new(Box::new(NoLauncher)), md));
    let clear = vec![[0, 0, 0, 0xff]; 256];
    for draw in [
        (3, 0, 6, 1),
        (3, 1, 10, 2),
        (4, 0, 5, 1),
        (4, 3, 8, 2),
        (3, 0, 2, 1),
        (4, 0, 2, 1),
    ] {
        let first = flat_pass(&p, &d, FLAT, 0, draw);
        let last = flat_pass(&p, &d, FLAT, 1, draw);
        assert_eq!(flat_pass(&p, &d, FLAT, 7, draw), last, "{draw:?}");
        if draw.2 >= 3 {
            assert_ne!(first, last, "{draw:?}: the last vertex's colours");
            assert_ne!(last, clear, "{draw:?}");
        } else {
            assert_eq!(last, clear, "{draw:?}");
        }
        if let Some((mp, md)) = &metal {
            for (mode, ours) in [(0, &first), (1, &last)] {
                let theirs = flat_pass(mp, md, FLAT, mode, draw);
                assert_eq!(*ours, theirs, "{draw:?} at mode {mode}");
            }
        }
    }
}

/// At the last vertex, a state whose fragment function reads `[[barycentric_coord]]` and no
/// flat input draws a list of triangles as it stands, as Metal draws it, and dies at a strip,
/// whose odd triangles' weights Metal moves; one that also reads a flat input dies at both,
/// as its rotated draw would move them. At the first vertex all of them draw
#[test]
fn a_barycentric_reader_draws_only_lists_at_the_last_vertex() {
    let (p, d) = (platform(), dev());
    let smooth = ("barycentric:smooth_vertex", "barycentric:smooth_weights");
    let flat = ("barycentric:flat_vertex", "barycentric:flat_weights");
    let list = (3, 1, 9, 2);
    let ours = flat_pass(&p, &d, smooth, 1, list);
    assert_ne!(ours, vec![[0, 0, 0, 0xff]; 256]);
    if let Some(md) = MetalPlatform::system_default_device() {
        let mp = MetalPlatform::new(Box::new(NoLauncher));
        assert_eq!(ours, flat_pass(&mp, &md, smooth, 1, list));
    }
    let draw = "drawPrimitives:vertexStart:vertexCount:";
    for (functions, list_draws) in [(smooth, true), (flat, false)] {
        let (e, _indices) = drawing_encoder(functions.0, functions.1);
        send_all(&e, vec![three(3, None), three(STRIP, None)]);
        send_all(&e, vec![RenderCmd::SetProvokingVertexMode { mode: 1 }]);
        if list_draws {
            send_all(&e, vec![three(3, None)]);
        } else {
            assert_eq!(dies_at(&e, three(3, None)), draw, "{functions:?}");
        }
        assert_eq!(dies_at(&e, three(STRIP, None)), draw, "{functions:?}");
    }
}

/// A draw over indices of its own, of a private type or of a list or strip of triangles at
/// the last vertex under a state that reads a flat input, gives each vertex's
/// `[[vertex_id]]` as Metal's non-indexed draw gives it, its indices relative to its first
/// vertex, its base vertex
#[test]
fn re_indexed_draws_keep_metals_vertex_ids() {
    let (p, d) = (platform(), dev());
    let ids = ("vertex_ids:ids_vertex", "vertex_ids:ids_fragment");
    let metal = MetalPlatform::system_default_device()
        .map(|md| (MetalPlatform::new(Box::new(NoLauncher)), md));
    for (mode, draw) in [
        (0, (5, 3, 5, 1)),
        (0, (6, 2, 6, 2)),
        (0, (7, 1, 8, 1)),
        (0, (8, 4, 8, 1)),
        (1, (3, 3, 6, 1)),
        (1, (STRIP, 2, 7, 1)),
    ] {
        let ours = flat_pass(&p, &d, ids, mode, draw);
        let lit: Vec<_> = ours.iter().filter(|px| **px != [0, 0, 0, 0xff]).collect();
        let (start, count) = (u8::try_from(draw.1).unwrap(), u8::try_from(draw.2).unwrap());
        let drawn = |px: &[u8; 4]| (start..start + count).contains(&px[2]);
        assert!(
            !lit.is_empty() && lit.iter().all(|px| drawn(px)),
            "{draw:?}"
        );
        if let Some((mp, md)) = &metal {
            assert_eq!(ours, flat_pass(mp, md, ids, mode, draw), "{mode} {draw:?}");
        }
    }
}

/// The draws of [`re_indexed_draws_keep_metals_vertex_ids`] and a public list from a later
/// vertex, of instances from base instance 3, give `[[base_vertex]]` as their first vertex
/// and `[[base_instance]]` as 3, as Metal's non-indexed draws give them
#[test]
fn re_indexed_draws_keep_metals_draw_bases() {
    /// `draw`, `(type, start, count, instances)`, from base instance 3.
    fn based<'a, P: Platform>(
        (primitive_type, start, count, instances): (u64, u64, u64, u64),
    ) -> RenderCmd<'a, P> {
        RenderCmd::DrawPrimitivesBase {
            primitive_type,
            vertex_start: start,
            vertex_count: count,
            instance_count: instances,
            base_instance: 3,
        }
    }
    let (p, d) = (platform(), dev());
    let bases = ("vertex_ids:bases_vertex", "vertex_ids:ids_fragment");
    let metal = MetalPlatform::system_default_device()
        .map(|md| (MetalPlatform::new(Box::new(NoLauncher)), md));
    for (mode, draw) in [
        (0, (3, 2, 6, 2)),
        (0, (5, 3, 5, 1)),
        (0, (6, 2, 6, 2)),
        (0, (7, 1, 8, 1)),
        (0, (8, 4, 8, 2)),
        (1, (3, 3, 6, 1)),
        (1, (STRIP, 2, 7, 2)),
    ] {
        let ours = flat_draw(&p, &d, bases, mode, based(draw));
        let lit: Vec<_> = ours.iter().filter(|px| **px != [0, 0, 0, 0xff]).collect();
        let start = u8::try_from(draw.1).unwrap();
        assert!(
            !lit.is_empty() && lit.iter().all(|px| px[1] == start && px[0] == 3),
            "{mode} {draw:?}"
        );
        if let Some((mp, md)) = &metal {
            let theirs = flat_draw(mp, md, bases, mode, based(draw));
            assert_eq!(ours, theirs, "{mode} {draw:?}");
        }
    }
}

/// A guest's state of `vertex` and `ids_fragment` for a BGRA8 target, of the input primitive
/// topology `class`, made as core makes one of a descriptor, with indirect command buffer support
/// where `indirect` says.
fn classed_state(
    d: &crate::gpu::device::VulkanDevice,
    (vertex, indirect): (&str, bool),
    class: PrimitiveTopologyClass,
) -> crate::gpu::resource::VulkanRenderPipeline {
    use metal2vulkan::metal::PixelFormat;
    use metal2vulkan::pipeline::{
        ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource,
    };
    use reims_vgpu_core::sync::Mutex;

    let library = d.raw.new_library_with_data(VERTEX_IDS).unwrap();
    let function = |name| {
        library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut desc = RenderPipelineDescriptor::new(StageSource::Function(function(vertex)));
    desc.fragment = Some(StageSource::Function(function("ids_fragment")));
    desc.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
    desc.input_primitive_topology = class;
    desc.support_indirect_command_buffers = indirect;
    let descriptor = Obj::new(crate::gpu::desc::PipelineDescriptorObject {
        desc: Mutex::new(desc),
        resource_index: Mutex::new(None),
        ignored: Vec::new(),
        nearest_even: false,
        label: None,
        raw_format: None,
    });
    d.new_render_pipeline_state_from_descriptor(Some(&descriptor), 0)
        .unwrap()
        .unwrap()
        .unwrap()
}

/// Points die under a state whose vertex function writes no `[[point_size]]` outside the
/// point input primitive topology class, where Metal sizes none, drawn directly, over
/// indices or from indirect arguments, but a draw of none; under one that writes it, or in
/// the point class, they draw as Metal's do, a pixel at each vertex
#[test]
fn points_die_where_metal_sizes_none() {
    let (p, d) = (platform(), dev());
    let draw = "drawPrimitives:vertexStart:vertexCount:";
    let indexed = "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:";
    let unclassed = [
        PrimitiveTopologyClass::Unspecified,
        PrimitiveTopologyClass::Line,
        PrimitiveTopologyClass::Triangle,
    ];
    for class in unclassed {
        let (e, indices) = open_drawing(&classed_state(&d, ("point_vertex", false), class));
        let indirect = RenderCmd::DrawPrimitivesIndirect {
            primitive_type: 0,
            indirect_buffer: &indices,
            indirect_buffer_offset: 0,
        };
        assert_eq!(dies_at(&e, three(0, None)), draw, "{class:?}");
        assert_eq!(dies_at(&e, three(0, Some(&indices))), indexed, "{class:?}");
        assert_eq!(
            dies_at(&e, indirect),
            "drawPrimitives:indirectBuffer:indirectBufferOffset:"
        );
        let none = RenderCmd::DrawPrimitives {
            primitive_type: 0,
            vertex_start: 0,
            vertex_count: 0,
        };
        send_all(&e, vec![none, three(3, None), RenderCmd::EndEncoding]);
    }
    let sized = fixture_state::<VulkanPlatform>(&d, SIZED.0, SIZED.1);
    let ours = pass_pixels(&p, &d, points::<VulkanPlatform>(&sized));
    let lit: Vec<usize> = (0..256).filter(|&i| ours[i] != [0, 0, 0, 0xff]).collect();
    assert_eq!(lit, [4 * 16 + 4, 8 * 16 + 8, 12 * 16 + 12]);
    let classed = classed_state(&d, ("point_vertex", false), PrimitiveTopologyClass::Point);
    assert_eq!(
        pass_pixels(&p, &d, points::<VulkanPlatform>(&classed)),
        ours
    );
    if let Some(md) = MetalPlatform::system_default_device() {
        let mp = MetalPlatform::new(Box::new(NoLauncher));
        let theirs = fixture_state::<MetalPlatform>(&md, SIZED.0, SIZED.1);
        assert_eq!(
            pass_pixels(&mp, &md, points::<MetalPlatform>(&theirs)),
            ours
        );
    }
}

/// An indirect command buffer of two commands whose first draws `count` points from vertex 1
/// under `state`, set on the command, or under the encoder's where `state` is `None`.
fn points_icb(
    d: &crate::gpu::device::VulkanDevice,
    state: Option<&crate::gpu::resource::VulkanRenderPipeline>,
    count: u64,
) -> crate::gpu::icb::VulkanIcb {
    use reims_vgpu_core::platform::icb::{
        IcbRenderCmd, IndirectCommandBuffer as _, IndirectRenderCommand as _,
    };
    use reims_vgpu_core::platform::values::IcbDesc;

    let icb = d
        .new_indirect_command_buffer(
            &IcbDesc {
                command_types: 1,
                inherit_pipeline_state: state.is_none(),
                ..IcbDesc::default()
            },
            2,
            0,
        )
        .unwrap()
        .unwrap();
    let c = icb.render_command(0).unwrap().unwrap();
    if let Some(state) = state {
        let set = IcbRenderCmd::SetRenderPipelineState {
            pipeline_state: Some(state),
        };
        c.encode(set).unwrap();
    }
    c.encode(IcbRenderCmd::DrawPrimitives {
        primitive_type: 0,
        vertex_start: 1,
        vertex_count: count,
        instance_count: 1,
        base_instance: 0,
    })
    .unwrap();
    icb
}

/// The commands that execute the first command of `icb`, after `state` where one is given.
fn execute_first<'a>(
    icb: &'a crate::gpu::icb::VulkanIcb,
    state: Option<&'a crate::gpu::resource::VulkanRenderPipeline>,
) -> Vec<RenderCmd<'a, VulkanPlatform>> {
    let set = state.map(|state| RenderCmd::SetRenderPipelineState {
        pipeline_state: Some(state),
    });
    set.into_iter()
        .chain([RenderCmd::ExecuteCommandsInBuffer {
            indirect_command_buffer: icb,
            execution_range: NsRange::new(0, 1),
        }])
        .collect()
}

/// The line of the death at the commit of a pass of `cmds`, which must die there.
fn commit_dies(cmds: Vec<RenderCmd<'_, VulkanPlatform>>) -> String {
    let (p, d) = (platform(), dev());
    let (_, pass) = pass_of(&p, &d, (16, 16));
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    for cmd in cmds.into_iter().chain([RenderCmd::EndEncoding]) {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    *crate::answer::LAST_LINE.lock() = None;
    let site = catch_death(|| cb.commit(OnFault::Return)).unwrap_err();
    assert_eq!(site.file, "commit");
    crate::answer::LAST_LINE.lock().take().unwrap_or_default()
}

/// An execution's draws of points die at the commit where a direct draw's die, its commands'
/// records read as they stand then: under a state of its own that writes no `[[point_size]]`
/// outside the point input primitive topology class, or under such a state of the encoder's
/// that its commands inherit; and an execution of a buffer whose records the GPU may have
/// written (a blit's reset of another of its commands) dies there too, as its commands are not
/// read before they run. A draw of no points, and one under a state of the point class or one
/// that writes its point size, own or inherited, draw as the direct draws do, which are Metal's
/// ([`points_die_where_metal_sizes_none`]). A command buffer committed while a command buffer
/// executes the buffer as its commit read it, and before it ends, dies at a blit's reset of
/// the buffer's commands
#[test]
fn executed_points_die_where_metal_sizes_none() {
    use reims_vgpu_core::platform::event::EventRef;

    let (p, d) = (platform(), dev());
    let unsizing = classed_state(
        &d,
        ("point_vertex", true),
        PrimitiveTopologyClass::Unspecified,
    );
    let classed = classed_state(&d, ("point_vertex", true), PrimitiveTopologyClass::Point);
    let sized = classed_state(
        &d,
        ("sized_vertex", true),
        PrimitiveTopologyClass::Unspecified,
    );
    let undrawn = "command 0 of executeCommandsInBuffer:withRange:'s indirect command buffer, a \
                   draw of points under a vertex function that writes no [[point_size]]";
    let own = points_icb(&d, Some(&unsizing), 3);
    assert!(commit_dies(execute_first(&own, None)).contains(undrawn));
    let inherited = points_icb(&d, None, 3);
    let line = commit_dies(execute_first(&inherited, Some(&unsizing)));
    assert!(line.contains(undrawn), "{line}");
    let none = points_icb(&d, Some(&unsizing), 0);
    let (lit, unlit) = (
        pass_pixels(&p, &d, points::<VulkanPlatform>(&classed)),
        pass_pixels(&p, &d, Vec::new()),
    );
    assert_eq!(pass_pixels(&p, &d, execute_first(&none, None)), unlit);
    for state in [&classed, &sized] {
        let direct = pass_pixels(&p, &d, points::<VulkanPlatform>(state));
        assert_eq!(direct, lit);
        let own = points_icb(&d, Some(state), 3);
        assert_eq!(pass_pixels(&p, &d, execute_first(&own, None)), direct);
        let inherits = execute_first(&inherited, Some(state));
        assert_eq!(pass_pixels(&p, &d, inherits), direct);
    }
    // Another command reset: the GPU may have written the buffer's records.
    let written = points_icb(&d, Some(&classed), 3);
    let reset = BlitCmd::ResetCommandsInBuffer {
        buffer: &written,
        range: NsRange::new(1, 1),
    };
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    for cmd in [reset, BlitCmd::EndEncoding] {
        blit.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    let line = commit_dies(execute_first(&written, None));
    assert!(
        line.contains("whose records the GPU may have written"),
        "{line}"
    );
    // A reset while an execution that read the records waits.
    let read = points_icb(&d, Some(&classed), 3);
    let (_, pass) = pass_of(&p, &d, (16, 16));
    let event = d.new_shared_event().unwrap();
    let held = q.command_buffer(OnFault::Die).unwrap().unwrap();
    held.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
    let e = held.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    for cmd in execute_first(&read, None)
        .into_iter()
        .chain([RenderCmd::EndEncoding])
    {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    held.commit(OnFault::Die).unwrap();
    let cb = d
        .new_command_queue()
        .unwrap()
        .command_buffer(OnFault::Die)
        .unwrap()
        .unwrap();
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    let reset = BlitCmd::ResetCommandsInBuffer {
        buffer: &read,
        range: NsRange::new(1, 1),
    };
    let sel = reset.selector();
    let site = catch_death(|| blit.encode(OnFault::Return, reset)).unwrap_err();
    assert_eq!(site.file, sel);
    event.set_signaled_value(1);
    held.wait_until_completed(OnFault::Die).unwrap();
    assert_eq!(held.error(), None);
    // Once it ended, the reset goes on.
    let reset = BlitCmd::ResetCommandsInBuffer {
        buffer: &read,
        range: NsRange::new(1, 1),
    };
    blit.encode(OnFault::Die, reset).unwrap();
}

/// A draw of points, and an execution of commands that inherit the encoder's state, under a
/// state of the point class whose vertex function can encode indirect commands die in an encoder that made resident for writing
/// an indirect command buffer whose ID is not the runtime's (one made after a reservation the
/// runtime is not given), as a compute dispatch does (`crate::gpu::encode_compute`); an execution
/// whose command's own state can dies at the commit, which reads its record. A draw under a
/// state that cannot runs, and so does one under the state that can where the buffer made
/// resident for writing has the runtime's ID
#[test]
fn work_that_can_encode_after_an_icb_made_writable_dies() {
    use reims_vgpu_core::platform::icb::{
        IcbRenderCmd, IndirectCommandBuffer as _, IndirectRenderCommand as _,
    };
    use reims_vgpu_core::platform::values::IcbDesc;
    use reims_vgpu_core::raise::death::DeathKind;

    let (p, d) = (platform(), dev());
    d.reserve_resource_indices(5, &[2, (1 << 23) + 1]).unwrap();
    let make = |inherit_pipeline_state| {
        d.new_indirect_command_buffer(
            &IcbDesc {
                command_types: 1,
                inherit_pipeline_state,
                ..IcbDesc::default()
            },
            1,
            0,
        )
        .unwrap()
        .unwrap()
    };
    let (reaching, written) = (make(false), make(false));
    assert!(reaching.reaches_its_records() && !written.reaches_its_records());
    let encoding = classed_state(&d, ("encoding_vertex", true), PrimitiveTopologyClass::Point);
    let other = classed_state(&d, ("point_vertex", true), PrimitiveTopologyClass::Point);
    let drawn = || RenderCmd::DrawPrimitives {
        primitive_type: 0,
        vertex_start: 1,
        vertex_count: 3,
    };
    let points = |state: Option<&crate::gpu::resource::VulkanRenderPipeline>| {
        let icb = make(state.is_none());
        let c = icb.render_command(0).unwrap().unwrap();
        let set = state.map(|state| IcbRenderCmd::SetRenderPipelineState {
            pipeline_state: Some(state),
        });
        let draw = IcbRenderCmd::DrawPrimitives {
            primitive_type: 0,
            vertex_start: 1,
            vertex_count: 3,
            instance_count: 1,
            base_instance: 0,
        };
        for cmd in set.into_iter().chain([draw]) {
            c.encode(cmd).unwrap();
        }
        icb
    };
    let (own, inheriting) = (points(Some(&encoding)), points(None));
    let held = buffer::<VulkanPlatform>(&d, 16, 0, &[0; 16]);
    let (the_written, the_reaching) = (
        [Some(ResourceRef::Icb(&written))],
        [Some(ResourceRef::Icb(&reaching))],
    );
    let opening = |state, resident: &'static str| {
        let list = if resident == "written" {
            &the_written
        } else {
            &the_reaching
        };
        vec![
            RenderCmd::UseResources {
                resources: list,
                usage: 3,
            },
            RenderCmd::SetVertexBuffer {
                buffer: Some(&held),
                offset: 0,
                index: 1,
            },
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(state),
            },
        ]
    };
    let execution = |icb| RenderCmd::ExecuteCommandsInBuffer {
        indirect_command_buffer: icb,
        execution_range: NsRange::new(0, 1),
    };
    for (last, sel) in [
        (drawn(), "drawPrimitives:vertexStart:vertexCount:"),
        (execution(&inheriting), "executeCommandsInBuffer:withRange:"),
    ] {
        let (e, _dst) = open_encoder();
        send_all(&e, opening(&encoding, "written"));
        let site = catch_death(|| e.encode(OnFault::Return, last)).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
    }
    let mut executed = opening(&other, "written");
    executed.push(execution(&own));
    commit_dies(executed);
    let unlit = pass_pixels(&p, &d, Vec::new());
    for (state, resident) in [(&other, "written"), (&encoding, "reaching")] {
        let mut cmds = opening(state, resident);
        cmds.push(drawn());
        assert_ne!(pass_pixels(&p, &d, cmds), unlit);
    }
}

/// The vertices [`vertex_runs`] counts.
const RUNS: usize = 1024;

/// The runs of `counting_vertex` at each of the first [`RUNS`] vertex IDs, after `draw` as
/// [`flat_pass`] draws it, at `setProvokingVertexMode:` of `mode`.
fn vertex_runs<P: Platform>(
    p: &P,
    d: &P::Device,
    mode: u64,
    (primitive_type, start, count, instances): (u64, u64, u64, u64),
) -> Vec<u32> {
    let state = fixture_state::<P>(d, "vertex_ids:counting_vertex", SIZED.1);
    let (pos, _) = flat_vertices::<P>(d, RUNS);
    let runs = buffer::<P>(d, 4 * RUNS as u64, 0, &[0; 4 * RUNS]);
    pass_pixels(
        p,
        d,
        vec![
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            },
            RenderCmd::SetVertexBuffer {
                buffer: Some(&pos),
                offset: 0,
                index: 0,
            },
            RenderCmd::SetVertexBuffer {
                buffer: Some(&runs),
                offset: 0,
                index: 1,
            },
            RenderCmd::SetProvokingVertexMode { mode },
            RenderCmd::DrawPrimitivesInstances {
                primitive_type,
                vertex_start: start,
                vertex_count: count,
                instance_count: instances,
            },
        ],
    );
    // SAFETY: a Shared buffer of `4 * RUNS` bytes, read after the pass completed.
    let bytes = unsafe { std::slice::from_raw_parts(runs.contents().as_ptr::<u8>(), 4 * RUNS) };
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// A draw over indices of its own (a private type, or triangles the last vertex rotates)
/// runs its vertex function as many times at each vertex as Metal's draw, as a draw as it
/// stands does: once for each primitive the vertex is in, on both
#[test]
fn re_indexed_draws_run_the_vertex_function_as_often_as_metals() {
    let (p, d) = (platform(), dev());
    let metal = MetalPlatform::system_default_device()
        .map(|md| (MetalPlatform::new(Box::new(NoLauncher)), md));
    let at = |start: usize, runs: &[u32]| -> Vec<u32> {
        let mut out = vec![0; RUNS];
        out[start..start + runs.len()].copy_from_slice(runs);
        out
    };
    for (mode, draw, expected) in [
        (0, (3, 1, 6, 1), at(1, &[1; 6])),
        (0, (STRIP, 2, 7, 1), at(2, &[1, 2, 3, 3, 3, 2, 1])),
        (0, (2, 3, 4, 1), at(3, &[1, 2, 2, 1])),
        (0, (5, 3, 5, 1), at(3, &[2; 5])),
        (0, (6, 2, 6, 1), at(2, &[4, 1, 2, 2, 2, 1])),
        (0, (7, 1, 8, 1), at(1, &[1, 2, 1, 2, 1, 2, 1, 2])),
        (0, (8, 4, 8, 1), at(4, &[2, 1, 3, 3, 3, 3, 1, 2])),
        (1, (3, 3, 6, 1), at(3, &[1; 6])),
        (1, (STRIP, 2, 7, 1), at(2, &[1, 2, 3, 3, 3, 2, 1])),
    ] {
        assert_eq!(vertex_runs(&p, &d, mode, draw), expected, "{mode} {draw:?}");
        if let Some((mp, md)) = &metal {
            assert_eq!(
                vertex_runs(mp, md, mode, draw),
                expected,
                "Metal: {mode} {draw:?}"
            );
        }
    }
    for (mode, draw) in [
        (0, (6, 0, 1000, 3)),
        (0, (7, 0, 1000, 2)),
        (0, (8, 0, 1000, 2)),
        (0, (5, 0, 1000, 2)),
        (1, (STRIP, 0, 1000, 2)),
        (1, (3, 0, 999, 2)),
    ] {
        let ours = vertex_runs(&p, &d, mode, draw);
        assert!(ours.iter().any(|&n| n > 1), "{mode} {draw:?}");
        if let Some((mp, md)) = &metal {
            assert_eq!(ours, vertex_runs(mp, md, mode, draw), "{mode} {draw:?}");
        }
    }
}

/// [`ROUNDING`]'s functions compiled with fast math off (`-fno-fast-math`).
const ROUNDING_SAFE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rounding_safe.metallib"));

/// How [`swept`] draws: the fragment function, `sweep_fragment` (a `float4`),
/// `sweep_half_fragment` (a `half4`) or `sweep_four_fragment` (a `float4` at colours 0 to 3),
/// the colour attachment's entries past its index and pixel format (key 2 blending, 3 to 8
/// its factors and operations, 9 its write mask), the blob's render entries past its
/// fragment function's (key 5 alpha to coverage), the swept attachment's index (past 0,
/// colour 0 is an `RGBA8Unorm` attachment of the blob's defaults, cleared), and its load
/// action (`MTLLoadAction`: 2 clears it, 1 loads what the new texture holds).
#[derive(Clone, Copy, Debug)]
struct Drawn<'a> {
    fragment: &'a str,
    colour: &'a [(u8, u32)],
    render: &'a [&'a str],
    index: u32,
    load: u64,
}

/// `sweep_fragment` into an attachment of the blob's defaults: no blend, every channel.
const PLAIN: Drawn<'static> = Drawn {
    fragment: "sweep_fragment",
    colour: &[],
    render: &[],
    index: 0,
    load: CLEAR,
};

/// The bytes of a 256 by 256 texture of `format`, `bytes` a texel, after the sweep `kind`
/// (`rounding.metal`) is written into it under a state of `textureWriteRoundingMode` `mode`,
/// decoded from a guest's blob as core decodes one: by the kernel where `kernel`, else drawn
/// into it as the colour attachment `drawn` says, as it says.
fn swept<P: Platform>(
    p: &P,
    d: &P::Device,
    (kernel, mode): (bool, u32),
    (format, bytes): (u64, u64),
    kind: u32,
    drawn: Drawn<'_>,
) -> Vec<u8> {
    use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    use reims_vgpu_core::platform::resource::Library as _;
    use reims_vgpu_core::platform::types::NsRange;
    use reims_vgpu_core::rt::Hold;

    use crate::gpu::serialized::tests::{hex, le, rounded_render_of};

    let library = d.new_library(ROUNDING).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    let functions = [
        function("sweep_vertex"),
        function(drawn.fragment),
        function("sweep_kernel"),
    ];
    let mut resolve = |_: Hold<'_>, r: u64| {
        let at = usize::try_from(r - 1).unwrap();
        Ok(Some(functions[at].clone_ref()))
    };
    let t = texture(p, d, format, READ | 2 | TARGET, (256, 256), None);
    let kinds = buffer::<P>(d, 4, 0, &kind.to_le_bytes());
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    if kernel {
        let blob = hex(&format!("02 0a 04 {} 00 04 03000000", le(mode)));
        let desc = d
            .deserialize_compute_pipeline_descriptor(Hold::NONE, &blob, &mut resolve)
            .unwrap()
            .unwrap();
        let state = d
            .new_compute_pipeline_state(Some(&desc), 0)
            .unwrap()
            .unwrap()
            .unwrap();
        let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
        let buffers = [Some(&kinds)];
        for cmd in [
            ComputeCmd::SetComputePipelineState {
                state: Some(&state),
            },
            ComputeCmd::SetTexture {
                texture: Some(&t),
                index: 0,
            },
            ComputeCmd::SetBuffers {
                buffers: &buffers,
                offsets: &[0],
                range: NsRange::new(0, 1),
            },
            ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(256, 256, 1),
                threads_per_threadgroup: MtlSize::new(16, 16, 1),
            },
            ComputeCmd::EndEncoding,
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
    } else {
        let format = u32::try_from(format).unwrap();
        let mut colour = vec![(0, drawn.index), (1, format)];
        colour.extend(drawn.colour);
        let first: &[(u8, u32)] = &[(0, 0), (1, 70)];
        let colours: Vec<&[(u8, u32)]> = match drawn.index {
            0 => vec![&colour],
            _ => vec![first, &colour],
        };
        let mut entries = vec!["02 04 02000000"];
        entries.extend(drawn.render);
        let blob = rounded_render_of(&entries, &colours, mode);
        let desc = d
            .deserialize_render_pipeline_descriptor(Hold::NONE, &blob, &mut resolve)
            .unwrap()
            .unwrap();
        let state = d
            .new_render_pipeline_state_from_descriptor(Some(&desc), 0)
            .unwrap()
            .unwrap()
            .unwrap();
        let pass = p.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(&t)),
            load_action: Some(drawn.load),
            store_action: Some(STORE),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, u64::from(drawn.index), &attachment)
            .unwrap();
        let target = texture(p, d, 70, READ | TARGET, (256, 256), None);
        if drawn.index != 0 {
            let first = ColorAttachmentSet {
                texture: Some(Some(&target)),
                load_action: Some(CLEAR),
                store_action: Some(STORE),
                ..ColorAttachmentSet::default()
            };
            pass.set_color_attachment(OnFault::Die, 0, &first).unwrap();
        }
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        for cmd in [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            },
            RenderCmd::SetFragmentBuffer {
                buffer: Some(&kinds),
                offset: 0,
                index: 0,
            },
            RenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
            },
            RenderCmd::EndEncoding,
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
    }
    let length = 256 * 256 * bytes;
    let out = buffer::<P>(d, length, 0, &[]);
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    blit.encode(
        OnFault::Die,
        BlitCmd::CopyTextureToBuffer {
            source_texture: &t,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(256, 256, 1),
            destination_buffer: &out,
            destination_offset: 0,
            destination_bytes_per_row: 256 * bytes,
            destination_bytes_per_image: length,
        },
    )
    .unwrap();
    blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    let n = usize::try_from(length).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the blit completed.
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// Texture writes under each `textureWriteRoundingMode` that decodes are Metal's under it,
/// byte for byte, over sweeps through unorm values, past 1 by float steps and past the
/// largest half: a kernel's at every mode, and a fragment's into an unblended colour
/// attachment at every mode, where Metal's mode 2 rounds a half-float one's otherwise than
/// its default
#[test]
fn texture_writes_round_as_metals_at_the_modes_they_decode() {
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new(Box::new(NoLauncher));
    let (p, d) = (platform(), dev());
    let half = [(25, 2), (65, 4), (115, 8)];
    let rest = [(70, 4), (12, 1), (20, 2), (90, 4), (92, 4)];
    let srgb = (81, 4);
    for kind in [0, 1, 3] {
        for format in half.into_iter().chain(rest) {
            for mode in 0..=2 {
                let ours = swept(&p, &d, (true, mode), format, kind, PLAIN);
                let theirs = swept(&mp, &md, (true, mode), format, kind, PLAIN);
                assert!(ours == theirs, "kernel {format:?} mode {mode} sweep {kind}");
            }
        }
        for format in half.into_iter().chain(rest).chain([srgb]) {
            for mode in 0..=2 {
                let ours = swept(&p, &d, (false, mode), format, kind, PLAIN);
                let theirs = swept(&mp, &md, (false, mode), format, kind, PLAIN);
                assert!(ours == theirs, "render {format:?} mode {mode} sweep {kind}");
            }
        }
        for format in half {
            let rounded = swept(&mp, &md, (false, 2), format, kind, PLAIN);
            assert!(
                rounded != swept(&mp, &md, (false, 0), format, kind, PLAIN),
                "{format:?}"
            );
        }
    }
}

/// A render state whose fragment function reads its extended-range colour 0 by framebuffer
/// fetch, blended there, is made where the runtime blends it in the shader, as Metal makes it:
/// into `RGB10A8_2P_XR10`, a `half4` output under source factor zero and destination factor
/// one minus the source's alpha (a guest's WindowServer's), and under source factor the
/// source's alpha. The function computes each output from the fetched colour under fast
/// math, which the runtime notes `extended-range-blend-source` and
/// announces at each draw that binds the state's variant: a draw of either over an attachment
/// a pass clears dies there, naming the kind ([`crate::answer::DIES`]), where Metal draws it.
/// A `float4` output so into `RGB10A8_2P_XR10` under that
/// first blend and under source and destination factor one is made, and the runtime notes it
/// `extended-range-float-blend-source`, which it announces at each draw that binds
/// the state's variant: a draw of it over an attachment a pass clears dies there, naming the
/// kind ([`crate::answer::DIES`]), where Metal draws it. One into `RGB10A8_2P_XR10_sRGB`
/// under source and destination factor one, whose blend in the `_sRGB` form the runtime has
/// not fitted, dies as the state is made, naming its refusal,
/// `ExtendedRangeBlend(FloatOutput)`, where Metal makes it
#[test]
fn extended_range_fetches_are_made_where_the_runtime_blends_them() {
    use reims_vgpu_core::platform::resource::Library as _;
    use reims_vgpu_core::raise::death::DeathKind;
    use reims_vgpu_core::rt::Hold;

    use crate::gpu::serialized::tests::rounded_render_of;

    fn state<P: Platform>(
        d: &P::Device,
        (fragment, format): (&str, u32),
        (source, destination): (u32, u32),
    ) -> Option<P::RenderPipeline> {
        let library = d.new_library(ROUNDING).unwrap().unwrap().unwrap();
        let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
        let functions = [function("sweep_vertex"), function(fragment)];
        let mut resolve = |_: Hold<'_>, r: u64| {
            let at = usize::try_from(r - 1).unwrap();
            Ok(Some(functions[at].clone_ref()))
        };
        let colour: &[(u8, u32)] = &[
            (0, 0),
            (1, format),
            (2, 1),
            (3, source),
            (4, destination),
            (5, 0),
            (6, source),
            (7, destination),
            (8, 0),
        ];
        let blob = rounded_render_of(&["02 04 02000000"], &[colour], 0);
        let desc = d
            .deserialize_render_pipeline_descriptor(Hold::NONE, &blob, &mut resolve)
            .unwrap()
            .unwrap();
        d.new_render_pipeline_state_from_descriptor(Some(&desc), 0)
            .unwrap()
            .unwrap()
    }
    /// A draw of `state` over a Private 64 by 64 attachment of `format` that its pass clears.
    fn drawn<P: Platform>(p: &P, d: &P::Device, state: &P::RenderPipeline, format: u32) {
        let t = texture(p, d, u64::from(format), READ | TARGET, (64, 64), None);
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let pass = p.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(&t)),
            load_action: Some(CLEAR),
            store_action: Some(STORE),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, 0, &attachment)
            .unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        for cmd in [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(state),
            },
            RenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
            },
            RenderCmd::EndEncoding,
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
    }
    const STATE: &str = "newRenderPipelineStateWithDescriptor:options:reflection:error:";
    const DRAW: &str = "drawPrimitives:vertexStart:vertexCount:";
    let (p, d) = (platform(), dev());
    let metal = MetalPlatform::system_default_device();
    // Each death, the selector it dies at, and what its line names.
    let half_source = "(metal2vulkan-degrade-extended-range-blend-source)";
    let float_source = "extended-range-float-blend-source";
    let dies = [
        ((("fetch_half_fragment", 550), (0, 5)), DRAW, half_source),
        ((("fetch_half_fragment", 550), (4, 5)), DRAW, half_source),
        ((("fetch_float_fragment", 550), (0, 5)), DRAW, float_source),
        ((("fetch_float_fragment", 550), (1, 1)), DRAW, float_source),
        (
            (("fetch_float_fragment", 551), (1, 1)),
            STATE,
            "ExtendedRangeBlend(FloatOutput)",
        ),
    ];
    for ((fragment, blend), sel, named) in dies {
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(|| {
            let state = state::<VulkanPlatform>(&d, fragment, blend).unwrap();
            drawn(&p, &d, &state, fragment.1);
        })
        .unwrap_err();
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, sel),
            "{fragment:?} {blend:?}: {line}"
        );
        assert!(line.contains(named), "{line}");
    }
    if let Some(md) = &metal {
        let mp = MetalPlatform::new(Box::new(NoLauncher));
        for (fragment, blend) in dies.map(|(d, ..)| d) {
            let state = state::<MetalPlatform>(md, fragment, blend);
            let state = state.unwrap_or_else(|| panic!("{fragment:?}"));
            drawn(&mp, md, &state, fragment.1);
        }
    }
}

/// The blends the extended-range tests draw under, of every factor kind and operation: each the
/// colour's source factor, destination factor and operation, then alpha's (`MTLBlendFactor`,
/// `MTLBlendOperation`).
const XR_BLENDS: [[u32; 6]; 22] = [
    [1, 1, 0, 1, 1, 0],
    [0, 5, 0, 0, 5, 0],
    [1, 5, 0, 1, 5, 0],
    [4, 5, 0, 1, 5, 0],
    [4, 5, 0, 4, 5, 0],
    [6, 0, 0, 6, 0, 0],
    [6, 5, 0, 1, 5, 0],
    [3, 8, 0, 1, 1, 0],
    [3, 6, 0, 1, 1, 0],
    [3, 5, 0, 1, 5, 0],
    [6, 2, 0, 1, 1, 0],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 2, 1, 1, 2],
    [1, 1, 3, 1, 1, 3],
    [1, 1, 4, 1, 1, 4],
    [7, 1, 0, 1, 1, 0],
    [9, 1, 0, 9, 1, 0],
    [2, 0, 0, 2, 0, 0],
    [10, 1, 0, 1, 1, 0],
    [8, 9, 0, 8, 9, 0],
    [11, 12, 0, 13, 14, 0],
    [5, 4, 0, 5, 4, 0],
];

/// The linear extended-range formats: `RGB10A8_2P_XR10`, `BGRA10_XR` and `BGR10_XR`.
const XR_LINEAR: [u32; 3] = [550, 552, 554];

/// Their `_sRGB` twins: `RGB10A8_2P_XR10_sRGB`, `BGRA10_XR_sRGB` and `BGR10_XR_sRGB`.
const XR_SRGB: [u32; 3] = [551, 553, 555];

/// The blends the runtime models into an `_sRGB` twin, as [`XR_BLENDS`]: added
/// colour and alpha, each of source factor zero or one and destination factor zero, one or one
/// minus the source's alpha.
const XR_SRGB_BLENDS: [[u32; 6]; 8] = [
    [0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 5, 0, 0, 5, 0],
    [1, 0, 0, 1, 0, 0],
    [1, 1, 0, 1, 1, 0],
    [1, 5, 0, 1, 5, 0],
    [1, 5, 0, 1, 1, 0],
    [0, 5, 0, 1, 5, 0],
];

/// How many of the 4096 texels `XrSide::blended` read of `ours` differ from `theirs`, and the
/// first, both sides; `None` where none does.
fn xr_wrong(ours: &[u8], theirs: &[u8]) -> Option<String> {
    let wrong = (ours.chunks(16).zip(theirs.chunks(16)).enumerate()).filter(|(_, (o, t))| o != t);
    let count = wrong.clone().count();
    let (i, (o, t)) = wrong.into_iter().next()?;
    Some(format!(
        "{count} of 4096 texels differ, the first ({}, {}): ours {:?}, Metal {:?}",
        i % 64,
        i / 64,
        xr_lanes(o),
        xr_lanes(t)
    ))
}

/// The values `xr_loaded_fragment` loads: 4096 `float4`s a draw, of seeds 1 and 2, as
/// `xr_float_fragment` draws them (colour lanes in [-0.9375, 1.5) and alpha in
/// [-0.125, 1.125), each a multiple of 2^-11), from a fixed sequence.
fn xr_values() -> Vec<u8> {
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = |span: u64, low: i64| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let code = i64::try_from(x % span).unwrap() - low;
        code as f32 / 2048.0
    };
    (0..2 * 4096)
        .flat_map(|_| {
            [
                next(4992, 1920),
                next(4992, 1920),
                next(4992, 1920),
                next(2560, 256),
            ]
        })
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// One platform's side of the extended-range comparisons: the platform, its device, and the
/// one command queue a test makes of it for all its configurations ([`xr_sides`]), for the
/// reason [`test_queue`](crate::gpu::device::tests::metal_tests::test_queue) gives: a Metal
/// queue made per configuration lives to the test's end, and past 512 of them Metal's passes
/// stored nothing.
struct XrSide<P: Platform> {
    p: P,
    d: P::Device,
    q: P::CommandQueue,
}

impl<P: Platform> XrSide<P> {
    /// The side of `p`'s device `d`, with a queue of its own.
    fn new(p: P, d: P::Device) -> XrSide<P> {
        let q = d.new_command_queue().unwrap();
        XrSide { p, d, q }
    }

    /// The floats `xr_read` reads of a Private 64 by 64 attachment of `format` after a draw
    /// of `fragment` (`xr_float_fragment`, `xr_half_fragment` or `xr_loaded_fragment`, buffer
    /// 1 holding [`xr_values`]) of `library` ([`ROUNDING`] or [`ROUNDING_SAFE`]) of seed 1
    /// into it unblended, in a pass that clears it, then, under `blend` (the colour
    /// attachment's keys 3 to 8: the colour's source and destination factors and operation,
    /// then alpha's), one of seed 2 blended: in a pass of its own that loads what the first
    /// stored where `across`, else in the first's pass. With no `blend`, the first draw
    /// alone.
    fn blended(
        &self,
        (library, fragment): (&[u8], &str),
        format: u32,
        blend: Option<[u32; 6]>,
        across: bool,
    ) -> Vec<u8> {
        self.masked((library, fragment), format, (blend, None), across)
    }

    /// [`XrSide::blended`], the blended draw's state of the colour write mask `mask` (key 9,
    /// `MTLColorWriteMask`: red 8, green 4, blue 2, alpha 1) where one is given, the
    /// unblended one's of every channel ([`xr_blended_masked`]).
    fn masked(
        &self,
        drawn: (&[u8], &str),
        format: u32,
        shape: (Option<[u32; 6]>, Option<u32>),
        across: bool,
    ) -> Vec<u8> {
        xr_blended_masked(&self.p, &self.d, &self.q, drawn, format, shape, across)
    }
}

impl XrSide<MetalPlatform> {
    /// [`XrSide::masked`] on Metal, the configuration named `what`. Where every texel it
    /// reads is one texel and that texel is Metal's fresh pattern of `format`
    /// ([`xr_unwritten`]), it fails, naming "Metal's passes did not run": its command buffer
    /// completed, its read ran, and none of its passes stored anything, which no draw of
    /// these functions leaves (each texel a hash of its position), so it is never a
    /// difference to report or an answer to compare with
    fn drawn(
        &self,
        what: &str,
        drawn: (&[u8], &str),
        format: u32,
        shape: (Option<[u32; 6]>, Option<u32>),
        across: bool,
    ) -> Vec<u8> {
        let theirs = self.masked(drawn, format, shape, across);
        let first = theirs.get(..16).unwrap_or_default();
        if theirs.chunks(16).all(|t| t == first) {
            let fresh = xr_unwritten(&self.p, &self.d, &self.q, drawn.0, format);
            assert!(
                theirs != fresh,
                "{what}: Metal's passes did not run: every texel Metal read is its fresh \
                 pattern of format {format}, {:?}",
                xr_lanes(first)
            );
        }
        theirs
    }
}

/// This platform's side and Metal's, where the host has a Metal device: one command queue of
/// each for a test ([`XrSide`]).
fn xr_sides() -> XrSides {
    let ours = XrSide::new(platform(), dev());
    let metal = MetalPlatform::system_default_device()
        .map(|md| XrSide::new(MetalPlatform::new(Box::new(NoLauncher)), md));
    (ours, metal)
}

/// The four floats of each texel of `bytes`, as `xr_read` writes them.
fn xr_lanes(bytes: &[u8]) -> Vec<f32> {
    (bytes.as_chunks::<4>().0.iter())
        .map(|&l| f32::from_le_bytes(l))
        .collect()
}

/// [`XrSide::masked`] on `p`'s device `d`, its command buffer of `q`, in an autorelease
/// pool of its own: what Metal autoreleased (each command buffer and encoder, which a dev
/// build cannot elide) goes at the configuration's end.
fn xr_blended_masked<P: Platform>(
    p: &P,
    d: &P::Device,
    q: &P::CommandQueue,
    (library, fragment): (&[u8], &str),
    format: u32,
    (blend, mask): (Option<[u32; 6]>, Option<u32>),
    across: bool,
) -> Vec<u8> {
    use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
    use reims_vgpu_core::platform::resource::Library as _;
    use reims_vgpu_core::rt::{Hold, Pool};

    use crate::gpu::serialized::tests::rounded_render_of;

    let _pool = Pool::here();
    let library = d.new_library(library).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    let functions = [function("sweep_vertex"), function(fragment)];
    let state = |blend: Option<[u32; 6]>| {
        let mut resolve = |_: Hold<'_>, r: u64| {
            let at = usize::try_from(r - 1).unwrap();
            Ok(Some(functions[at].clone_ref()))
        };
        let mut colour = vec![(0, 0), (1, format)];
        if let Some(blend) = blend {
            colour.push((2, 1));
            colour.extend((3..).zip(blend));
            colour.extend(mask.map(|mask| (9, mask)));
        }
        let blob = rounded_render_of(&["02 04 02000000"], &[&colour], 0);
        let desc = d
            .deserialize_render_pipeline_descriptor(Hold::NONE, &blob, &mut resolve)
            .unwrap()
            .unwrap();
        d.new_render_pipeline_state_from_descriptor(Some(&desc), 0)
            .unwrap()
            .unwrap()
            .unwrap()
    };
    let states: Vec<_> = std::iter::once(None)
        .chain(blend.map(Some))
        .map(state)
        .collect();
    let t = d
        .new_texture(OnFault::Die, &xr_attachment(p, format))
        .unwrap()
        .unwrap();
    let seeds = [1u32, 2].map(|seed| buffer::<P>(d, 4, 0, &seed.to_le_bytes()));
    let values = xr_values();
    let values = buffer::<P>(d, u64::try_from(values.len()).unwrap(), 0, &values);
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    // `MTLLoadActionLoad`.
    const LOAD: u64 = 1;
    let draws: Vec<_> = states.iter().zip(&seeds).collect();
    let passes = match across {
        true => draws.chunks(1).collect::<Vec<_>>(),
        false => vec![&draws[..]],
    };
    for (n, draws) in passes.into_iter().enumerate() {
        let pass = p.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(&t)),
            load_action: Some(if n == 0 { CLEAR } else { LOAD }),
            store_action: Some(STORE),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, 0, &attachment)
            .unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        for &(state, seed) in draws {
            for cmd in [
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(state),
                },
                RenderCmd::SetFragmentBuffer {
                    buffer: Some(seed),
                    offset: 0,
                    index: 0,
                },
                RenderCmd::SetFragmentBuffer {
                    buffer: Some(&values),
                    offset: 0,
                    index: 1,
                },
                RenderCmd::DrawPrimitives {
                    primitive_type: 3,
                    vertex_start: 0,
                    vertex_count: 3,
                },
            ] {
                e.encode(OnFault::Die, cmd).unwrap();
            }
        }
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    }
    xr_read_back::<P>(d, cb, &library, &t)
}

/// The descriptor of the Private 64 by 64 attachment of `format` the extended-range
/// comparisons draw into and read.
fn xr_attachment<P: Platform>(p: &P, format: u32) -> P::TextureDescriptor {
    p.texture_descriptor(&TextureDesc {
        pixel_format: u64::from(format),
        width: 64,
        height: 64,
        usage: READ | TARGET,
        resource_options: Some(0x20),
        ..TextureDesc::default()
    })
}

/// What `xr_read` of `library` reads of the attachment of `format` ([`xr_attachment`]) that
/// nothing wrote, on `p`'s device `d`, its command buffer of `q`: on Metal, its fresh pattern
/// of the format, the texel every texel of a new texture it keeps losslessly compressed reads, or
/// the texel of zero bytes where a new texture reads zero (the two-plane `RGB10A8_2P_XR10`).
fn xr_unwritten<P: Platform>(
    p: &P,
    d: &P::Device,
    q: &P::CommandQueue,
    library: &[u8],
    format: u32,
) -> Vec<u8> {
    let _pool = reims_vgpu_core::rt::Pool::here();
    let library = d.new_library(library).unwrap().unwrap().unwrap();
    let t = d
        .new_texture(OnFault::Die, &xr_attachment(p, format))
        .unwrap()
        .unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    xr_read_back::<P>(d, cb, &library, &t)
}

/// Ends `cb` with `xr_read` of `library` reading `t` (64 by 64) into a Shared buffer of `d`,
/// a float4 a texel, commits it, waits for it, and answers the buffer's bytes: no error.
fn xr_read_back<P: Platform>(
    d: &P::Device,
    cb: P::CommandBuffer,
    library: &P::Library,
    t: &P::Texture,
) -> Vec<u8> {
    use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    use reims_vgpu_core::platform::resource::Library as _;

    let read = library
        .new_function(OnFault::Die, "xr_read")
        .unwrap()
        .unwrap();
    let read = d
        .new_compute_pipeline_state_for_function(Some(&read))
        .unwrap()
        .unwrap();
    let length = 64 * 64 * 16;
    let out = buffer::<P>(d, length, 0, &[]);
    let c = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
    let outs = [Some(&out)];
    for cmd in [
        ComputeCmd::SetComputePipelineState { state: Some(&read) },
        ComputeCmd::SetTexture {
            texture: Some(t),
            index: 0,
        },
        ComputeCmd::SetBuffers {
            buffers: &outs,
            offsets: &[0],
            range: NsRange::new(0, 1),
        },
        ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(64, 64, 1),
            threads_per_threadgroup: MtlSize::new(8, 8, 1),
        },
        ComputeCmd::EndEncoding,
    ] {
        c.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    let n = usize::try_from(length).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the kernel completed.
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// A blend of a pinned gap ([`xr_across`]): an attachment's format, the blend (as
/// [`XR_BLENDS`]), and how many of its 4096 texels differ from Metal's.
type XrGap = (u32, [u32; 6], usize);

/// What [`xr_across`] holds the runtime's blends across passes to.
#[derive(Clone, Copy)]
enum XrExpect<'a> {
    /// Metal's texels, but at a pinned gap's blends, each of which differs at exactly its
    /// count of texels.
    Exact(&'a [XrGap]),
    /// The blends whose texels differ from Metal's are returned, and where the list is
    /// given, only its blends may differ, at any count: the runtime announces
    /// `extended-range-blend-source` for the sweep's states, whose draws die there,
    /// which the caller's test requires.
    Announced(Option<&'a [XrGap]>),
}

/// The relayed kinds a test of [`xr_across`] and [`xr_within`] checks. The runtime announces
/// each once per device (`crate::platform::device_options`), so a test checks what the
/// process heard: a kind unheard after the sweeps that must not announce it, heard after
/// those that must.
fn heard(kind: rt::Degradation) -> bool {
    crate::answer::RELAYED.lock().contains(&kind)
}

/// This platform's side of an extended-range comparison and Metal's, where the host has a
/// Metal device ([`xr_sides`]).
type XrSides = (XrSide<VulkanPlatform>, Option<XrSide<MetalPlatform>>);

/// Each of `blends` and the unblended draw of `fragment` of `library` into an attachment of
/// each of `formats` ([`XrSide::blended`]) across passes, on this platform and Metal's, the
/// test's `sides`: what fails the caller's test, and the blends whose texels differ from
/// Metal's where `expect` allows it. Each state is made, the unblended draw's texels are
/// Metal's, and each blend is held to `expect`.
fn xr_across(
    sides: &XrSides,
    drawn: (&str, &[u8], &str),
    shape: (&[u32], &[[u32; 6]]),
    expect: XrExpect<'_>,
) -> (Vec<String>, Vec<String>) {
    xr_compared(sides, drawn, shape, expect, true)
}

/// [`xr_across`]'s comparison, the blended draw in a pass of its own where `across`, else in
/// the unblended draw's pass ([`XrSide::blended`]).
fn xr_compared(
    (ours, metal): &XrSides,
    (mode, library, fragment): (&str, &[u8], &str),
    (formats, blends): (&[u32], &[[u32; 6]]),
    expect: XrExpect<'_>,
    across: bool,
) -> (Vec<String>, Vec<String>) {
    let Some(metal) = metal else {
        return (Vec::new(), Vec::new());
    };
    let (mut differ, mut missed) = (Vec::new(), Vec::new());
    // The texels `list` gives the blend, where it lists it.
    let listed = |list: &[XrGap], format, blend| {
        (list.iter())
            .find(|&&(f, b, _)| (f, Some(b)) == (format, blend))
            .map(|&(_, _, texels)| texels)
    };
    for &format in formats {
        for blend in std::iter::once(None).chain(blends.iter().copied().map(Some)) {
            let within = if across { "" } else { " in one pass" };
            let what = format!("{mode} {fragment} format {format} blend {blend:?}{within}");
            let drawn = (library, fragment);
            let theirs = metal.drawn(&what, drawn, format, (blend, None), across);
            *crate::answer::LAST_LINE.lock() = None;
            let ours = catch_death(|| ours.blended(drawn, format, blend, across));
            let ours = match ours {
                Ok(ours) => ours,
                Err(site) => {
                    let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                    differ.push(format!("{what}: died at {site:?}: {line}"));
                    continue;
                }
            };
            let count = (ours.chunks(16).zip(theirs.chunks(16)))
                .filter(|(o, t)| o != t)
                .count();
            let wrong = xr_wrong(&ours, &theirs).map(|w| format!("{what}: {w}"));
            let allowed = match (expect, blend) {
                (_, None) => Some(0),
                (XrExpect::Exact(pinned), _) => Some(listed(pinned, format, blend).unwrap_or(0)),
                (XrExpect::Announced(None), _) => None,
                (XrExpect::Announced(Some(list)), _) => {
                    listed(list, format, blend).map_or(Some(0), |_| None)
                }
            };
            match allowed {
                Some(pinned) if count != pinned => differ.push(format!(
                    "{what}: {count} texels differ, pinned {pinned}: {}",
                    wrong.unwrap_or_default()
                )),
                Some(_) => {}
                None => missed.extend(wrong),
            }
        }
    }
    (differ, missed)
}

/// Each of [`XR_BLENDS`] of `fragment` of `library` into an attachment of each linear
/// extended-range format in the first draw's pass ([`XrSide::blended`]), on this platform and
/// Metal's, the test's `sides`, as [`xr_across`] compares them across passes: what fails the
/// caller's test, and the blends whose texels differ from Metal's where `expect` allows it. The
/// runtime holds such an attachment's `half` values through a pass of one sample and one layer,
/// as Metal's tile memory does, from the first draw that blends onto what an earlier draw of
/// the pass wrote, whose pass it records again, and announces no
/// `extended-range-blends-stored-codes` for a draw that read them, which the
/// caller's test requires unheard.
fn xr_within(
    sides: &XrSides,
    drawn: (&str, &[u8], &str),
    expect: XrExpect<'_>,
) -> (Vec<String>, Vec<String>) {
    xr_compared(
        sides,
        drawn,
        (&XR_LINEAR[..], &XR_BLENDS[..]),
        expect,
        false,
    )
}

/// A `float4` output blended into an attachment of each extended-range format where the
/// runtime's model is Metal's blend: compiled with fast math ([`ROUNDING`]), one each of whose
/// lanes is a load (`xr_loaded_fragment`), into each linear format (`RGB10A8_2P_XR10`,
/// `BGRA10_XR`, `BGR10_XR`) under each of [`XR_BLENDS`], of every factor kind and operation
/// (those of the runtime's model among them: one minus the source colour beside an
/// alpha, the destination colour or one minus the source's alpha; the source times the
/// destination twice over; `min` and `max`), over a first draw's values (colour lanes from
/// past the formats' low end to past their high end, alpha past 0 and 1); and compiled
/// without fast math ([`ROUNDING_SAFE`]), a computed one (`xr_float_fragment`) and a loaded
/// one, into each linear format under each of [`XR_BLENDS`] and into each `_sRGB` twin under
/// each of [`XR_SRGB_BLENDS`], whose products the runtime rounds to `half` as Metal's
/// compiler does there. Across passes ([`xr_across`]) each state is made and
/// stores Metal's values on this host, read back by a kernel, bit for bit, as does the first
/// draw alone, and none announces `extended-range-blend-source` or
/// `extended-range-blends-stored-codes`; in the first draw's pass ([`xr_within`]) each linear
/// one is made and stores Metal's values, bit for bit, and none announces the stored codes,
/// as the runtime holds the attachment's `half` values through the pass
#[test]
fn float4_extended_range_blends_are_metals_here() {
    let linear = (&XR_LINEAR[..], &XR_BLENDS[..]);
    let srgb = (&XR_SRGB[..], &XR_SRGB_BLENDS[..]);
    let sweeps = [
        (("fast math", ROUNDING, "xr_loaded_fragment"), linear),
        (
            ("fast math off", ROUNDING_SAFE, "xr_float_fragment"),
            linear,
        ),
        (
            ("fast math off", ROUNDING_SAFE, "xr_loaded_fragment"),
            linear,
        ),
        (("fast math off", ROUNDING_SAFE, "xr_float_fragment"), srgb),
        (("fast math off", ROUNDING_SAFE, "xr_loaded_fragment"), srgb),
    ];
    let sides = xr_sides();
    let mut differ = Vec::new();
    for (drawn, shape) in sweeps {
        differ.extend(xr_across(&sides, drawn, shape, XrExpect::Exact(&[])).0);
    }
    let source = rt::Degradation::ExtendedRangeBlendSource;
    let stored_codes = rt::Degradation::ExtendedRangeBlendsStoredCodes;
    for kind in [source, stored_codes] {
        if heard(kind) {
            differ.push(format!("{} across passes", kind.code()));
        }
    }
    for &(drawn, _) in &sweeps[..3] {
        differ.extend(xr_within(&sides, drawn, XrExpect::Exact(&[])).0);
    }
    if heard(stored_codes) {
        differ.push(format!("{} in one pass", stored_codes.code()));
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
}

/// A `float4` output blended into an attachment of each linear extended-range format where
/// Metal may round the blend's source otherwise than the runtime's model: one whose
/// lanes the function computes (`xr_float_fragment`), compiled with fast math ([`ROUNDING`]).
/// The runtime notes such a state `extended-range-float-blend-source`, apart from
/// a `half4` output's `extended-range-blend-source`, and announces it at each draw that binds
/// the state's variant: the state is made, and its draw dies there, naming the kind
/// ([`crate::answer::DIES`]), where Metal makes it and draws: 17 of these 66
/// blends missed Metal's by one code across passes at 35 to 409 of 4096 texels each (at the
/// runtime's 7647e5c22 and 173b29636): in each format (colour, then alpha) (4, 5, add; 1, 5,
/// add), (4, 5, add; 4, 5, add), (3, 5, add; 1, 5, add), (6, 2, add; 1, 1, add) and (5, 4,
/// add; 5, 4, add), and in `RGB10A8_2P_XR10` and `BGRA10_XR` (8, 9, add; 8, 9, add). The
/// function's unblended draw is Metal's, bit for bit
#[test]
fn float4_extended_range_blends_die_naming_their_source() {
    use reims_vgpu_core::raise::death::DeathKind;

    const DRAW: &str = "drawPrimitives:vertexStart:vertexCount:";
    let drawn = ("fast math", ROUNDING, "xr_float_fragment");
    let sides = xr_sides();
    let (mut differ, _) = xr_across(&sides, drawn, (&XR_LINEAR[..], &[]), XrExpect::Exact(&[]));
    let (ours, metal) = &sides;
    let named = rt::Degradation::ExtendedRangeFloatBlendSource.code();
    for format in XR_LINEAR {
        for blend in XR_BLENDS {
            let what = format!("format {format} blend {blend:?}");
            let drawn = (ROUNDING, "xr_float_fragment");
            if let Some(metal) = metal {
                metal.drawn(&what, drawn, format, (Some(blend), None), true);
            }
            *crate::answer::LAST_LINE.lock() = None;
            let ours = catch_death(|| ours.blended(drawn, format, Some(blend), true));
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            match ours {
                Ok(_) => differ.push(format!("{what}: drawn")),
                Err(site)
                    if (site.kind, site.file) != (DeathKind::Abort, DRAW)
                        || !line.contains(named) =>
                {
                    differ.push(format!("{what}: died at {site:?}: {line}"));
                }
                Err(_) => {}
            }
        }
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
}

/// Blends of the colour channels' blend colour with its one minus under a difference
/// (`MTLBlendOperationSubtract` 1, `MTLBlendOperationReverseSubtract` 2), as [`XR_BLENDS`],
/// which the runtime draws without fast math alone.
const XR_CONSTANT_DIFFERENCES: [[u32; 6]; 2] = [[11, 12, 1, 1, 1, 0], [12, 11, 2, 1, 1, 0]];

/// The blends the runtime draws into an extended-range attachment without fast math alone, of a
/// `half4` output (`xr_half_fragment`), a
/// computed `float4` one (`xr_float_fragment`) and a loaded one (`xr_loaded_fragment`),
/// compiled without fast math ([`ROUNDING_SAFE`]): [`XR_CONSTANT_DIFFERENCES`] into each
/// linear form, and each of [`XR_BLENDS`] and [`XR_CONSTANT_DIFFERENCES`] into each `_sRGB`
/// twin, those into `BGRA10_XR_sRGB` whose colour channels read the blend colour itself
/// (`MTLBlendFactorBlendColor`) among them, which the runtime draws. Across passes ([`xr_across`])
/// each state is made and stores Metal's values, bit for bit, and none announces
/// `extended-range-blend-source`, `extended-range-float-blend-source` or
/// `extended-range-blends-stored-codes`. Which Metal makes and dies as the state is made,
/// naming the runtime's refusal, compiled with fast math ([`ROUNDING`]):
/// [`XR_CONSTANT_DIFFERENCES`] into `BGRA10_XR` (`ExtendedRangeBlend(ConstantColorDifference)`)
/// and one of source factor the source's alpha into `BGRA10_XR_sRGB`, beyond the runtime's
/// measured blends (`ExtendedRangeBlend(SrgbForm)`), of the `half4` output. A blend under a write
/// mask of one colour channel, or by a factor of the second source, which the runtime draws
/// without fast math too, is
/// [`extended_range_masked_blends_and_two_output_states_are_metals`]'s
#[test]
fn extended_range_blends_drawn_without_fast_math_alone_are_metals() {
    use reims_vgpu_core::raise::death::DeathKind;

    const STATE: &str = "newRenderPipelineStateWithDescriptor:options:reflection:error:";
    let fragments = [
        "xr_half_fragment",
        "xr_float_fragment",
        "xr_loaded_fragment",
    ];
    let srgb: Vec<[u32; 6]> = XR_BLENDS
        .into_iter()
        .chain(XR_CONSTANT_DIFFERENCES)
        .collect();
    let sides = xr_sides();
    let mut differ = Vec::new();
    for fragment in fragments {
        let drawn = ("fast math off", ROUNDING_SAFE, fragment);
        let shape = (&XR_LINEAR[..], &XR_CONSTANT_DIFFERENCES[..]);
        differ.extend(xr_across(&sides, drawn, shape, XrExpect::Exact(&[])).0);
        for format in XR_SRGB {
            differ.extend(xr_across(&sides, drawn, (&[format], &srgb), XrExpect::Exact(&[])).0);
        }
    }
    for kind in [
        rt::Degradation::ExtendedRangeBlendSource,
        rt::Degradation::ExtendedRangeFloatBlendSource,
        rt::Degradation::ExtendedRangeBlendsStoredCodes,
    ] {
        if heard(kind) {
            differ.push(format!("{} across passes", kind.code()));
        }
    }
    let fast = ROUNDING;
    let named = "ExtendedRangeBlend(ConstantColorDifference)";
    let refused = (XR_CONSTANT_DIFFERENCES.map(|b| ((fast, "xr_half_fragment"), 552, b, named)))
        .into_iter()
        .chain(std::iter::once((
            (fast, "xr_half_fragment"),
            553,
            [4, 5, 0, 1, 5, 0],
            "ExtendedRangeBlend(SrgbForm)",
        )));
    let (ours, metal) = &sides;
    for ((library, fragment), format, blend, named) in refused {
        let what = format!("{fragment} format {format} blend {blend:?}");
        let drawn = (library, fragment);
        if let Some(metal) = metal {
            metal.drawn(&what, drawn, format, (Some(blend), None), true);
        }
        *crate::answer::LAST_LINE.lock() = None;
        let ours = catch_death(|| ours.blended(drawn, format, Some(blend), true));
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        match ours {
            Ok(_) => differ.push(format!("{what}: drawn")),
            Err(site)
                if (site.kind, site.file) != (DeathKind::Abort, STATE) || !line.contains(named) =>
            {
                differ.push(format!("{what}: died at {site:?}: {line}"));
            }
            Err(_) => {}
        }
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
}

/// The blends [`extended_range_masked_blends_and_two_output_states_are_metals`] draws
/// under a write mask, as [`XR_BLENDS`]: added terms of factors one, zero, the source's
/// alpha and colour and their one minus, the destination's alpha and its one minus.
const XR_MASKED_BLENDS: [[u32; 6]; 6] = [
    [1, 1, 0, 1, 1, 0],
    [0, 5, 0, 0, 5, 0],
    [4, 5, 0, 1, 5, 0],
    [3, 8, 0, 1, 1, 0],
    [5, 4, 0, 5, 4, 0],
    [8, 9, 0, 8, 9, 0],
];

/// The write masks (`MTLColorWriteMask`: red 8, green 4, blue 2, alpha 1) of one colour
/// channel, red, green and alpha, and blue, and of two, red and green, and green, blue and
/// alpha.
const XR_MASKS: [u32; 5] = [0x8, 0x5, 0x2, 0xc, 0x7];

/// Blends of the second source (`MTLBlendFactorSource1Color` 15, `OneMinusSource1Color` 16,
/// `Source1Alpha` 17, `OneMinusSource1Alpha` 18), as [`XR_BLENDS`]: in each factor, beside
/// one and the first source's alpha, under add, subtract (1) and reverse subtract (2).
const XR_DUAL_BLENDS: [[u32; 6]; 6] = [
    [15, 16, 0, 17, 18, 0],
    [1, 16, 0, 1, 18, 0],
    [17, 18, 0, 17, 18, 0],
    [15, 1, 0, 1, 1, 0],
    [4, 16, 0, 17, 5, 0],
    [16, 15, 2, 18, 17, 1],
];

/// The two blends the runtime says it draws into an extended-range attachment without
/// fast math alone that [`extended_range_blends_drawn_without_fast_math_alone_are_metals`]
/// leaves out, compiled without fast math ([`ROUNDING_SAFE`]), into each linear form and each
/// `_sRGB` twin, across passes ([`xr_blended_masked`]), the first draw unblended under every
/// channel. Each of [`XR_MASKED_BLENDS`] under each of [`XR_MASKS`], of one colour channel
/// and of two, of a `half4` output (`xr_half_fragment`), a computed `float4` one
/// (`xr_float_fragment`) and a loaded one (`xr_loaded_fragment`): nothing of this platform's
/// stands between such a state and the runtime, which makes it and draws it, announcing
/// nothing; each state is made and stores Metal's values, bit for bit, and none announces
/// `extended-range-blend-source`, `extended-range-float-blend-source` or
/// `extended-range-blends-stored-codes`. A state of a function that returns two outputs into
/// colour 0, two `half4` ones (`xr_dual_half_fragment`) or two `float4` ones
/// (`xr_dual_float_fragment`), into each form and twin, is made and stores Metal's values, bit
/// for bit: unblended, compiled without fast math and with it ([`ROUNDING`]), its first
/// output's, the runtime storing the second nowhere, as Metal's write reads none;
/// and compiled without fast math under each of [`XR_DUAL_BLENDS`], which read its second
/// output, which the runtime blends in the shader. Which Metal makes and dies as the
/// state is made, naming the runtime's refusal: compiled with fast math, such a state under a
/// blend of the second source into `BGRA10_XR` (`ExtendedRangeBlend(DualSource)`), and a
/// blend under a mask of one colour channel into `BGRA10_XR`
/// (`ExtendedRangeBlend(OneColorChannel)`)
#[test]
fn extended_range_masked_blends_and_two_output_states_are_metals() {
    use reims_vgpu_core::raise::death::DeathKind;

    const STATE: &str = "newRenderPipelineStateWithDescriptor:options:reflection:error:";
    let (ours, Some(metal)) = xr_sides() else {
        return;
    };
    let mut differ = Vec::new();
    for format in XR_LINEAR.into_iter().chain(XR_SRGB) {
        for fragment in [
            "xr_half_fragment",
            "xr_float_fragment",
            "xr_loaded_fragment",
        ] {
            for (blend, mask) in XR_MASKS
                .into_iter()
                .flat_map(|mask| XR_MASKED_BLENDS.map(|blend| (blend, mask)))
            {
                let shape = (Some(blend), Some(mask));
                let what = format!("{fragment} format {format} blend {blend:?} mask {mask:#x}");
                let library = (ROUNDING_SAFE, fragment);
                let theirs = metal.drawn(&what, library, format, shape, true);
                *crate::answer::LAST_LINE.lock() = None;
                let ours = catch_death(|| ours.masked(library, format, shape, true));
                match ours {
                    Ok(ours) => differ
                        .extend(xr_wrong(&ours, &theirs).map(|wrong| format!("{what}: {wrong}"))),
                    Err(site) => {
                        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                        differ.push(format!("{what}: died at {site:?}: {line}"));
                    }
                }
            }
        }
    }
    // Each two-output state compared: unblended with and without fast math, and blended by
    // each of `XR_DUAL_BLENDS` without.
    let dual = (std::iter::once((ROUNDING, None))).chain(
        std::iter::once(None)
            .chain(XR_DUAL_BLENDS.map(Some))
            .map(|b| (ROUNDING_SAFE, b)),
    );
    let dual: Vec<(&[u8], Option<[u32; 6]>)> = dual.collect();
    for format in XR_LINEAR.into_iter().chain(XR_SRGB) {
        for fragment in ["xr_dual_half_fragment", "xr_dual_float_fragment"] {
            for &(library, blend) in &dual {
                let fast = if library == ROUNDING {
                    "fast math"
                } else {
                    "fast math off"
                };
                let what = format!("{fragment} {fast} format {format} blend {blend:?}");
                let library = (library, fragment);
                let shape = (blend, None);
                let theirs = metal.drawn(&what, library, format, shape, true);
                *crate::answer::LAST_LINE.lock() = None;
                let ours = catch_death(|| ours.masked(library, format, shape, true));
                match ours {
                    Ok(ours) => differ
                        .extend(xr_wrong(&ours, &theirs).map(|wrong| format!("{what}: {wrong}"))),
                    Err(site) => {
                        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                        differ.push(format!("{what}: died at {site:?}: {line}"));
                    }
                }
            }
        }
    }
    for kind in [
        rt::Degradation::ExtendedRangeBlendSource,
        rt::Degradation::ExtendedRangeFloatBlendSource,
        rt::Degradation::ExtendedRangeBlendsStoredCodes,
    ] {
        if heard(kind) {
            differ.push(format!("{} across passes", kind.code()));
        }
    }
    // Each death: its library and function, format, blend and mask, and what its line names.
    type Refused = (
        (&'static [u8], &'static str),
        u32,
        (Option<[u32; 6]>, Option<u32>),
        &'static str,
    );
    let refused: [Refused; 2] = [
        (
            (ROUNDING, "xr_dual_half_fragment"),
            552,
            (Some([15, 16, 0, 17, 18, 0]), None),
            "ExtendedRangeBlend(DualSource)",
        ),
        (
            (ROUNDING, "xr_half_fragment"),
            552,
            (Some([4, 5, 0, 1, 5, 0]), Some(0x8)),
            "ExtendedRangeBlend(OneColorChannel)",
        ),
    ];
    for (library, format, shape, named) in refused {
        let what = format!("{} format {format} {shape:?}", library.1);
        metal.drawn(&what, library, format, shape, true);
        *crate::answer::LAST_LINE.lock() = None;
        let ours = catch_death(|| ours.masked(library, format, shape, true));
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        match ours {
            Ok(_) => differ.push(format!("{what}: drawn")),
            Err(site)
                if (site.kind, site.file) != (DeathKind::Abort, STATE) || !line.contains(named) =>
            {
                differ.push(format!("{what}: died at {site:?}: {line}"));
            }
            Err(_) => {}
        }
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
}

/// The blends of `xr_half_fragment` compiled with fast math across passes ([`xr_across`])
/// whose texels missed Metal's on this host's M3 Max at the runtime's 7647e5c22, each as
/// [`XrGap`]: its format, blend (as [`XR_BLENDS`]) and how many of its 4096 texels differed,
/// each by one code of a channel, at blends whose factors are other than zero and one. The
/// runtime announces a computed `half4` under fast math as `extended-range-blend-source`, which
/// dies at the draw
/// ([`crate::answer::DIES`]); the half4 test's sweeps let these alone differ, at any count, of
/// a draw that does not die. At the runtime's 173b29636 the same 13 differ.
const HALF4_FAST_GAP: [XrGap; 13] = [
    (550, [3, 8, 0, 1, 1, 0], 333),
    (550, [3, 6, 0, 1, 1, 0], 483),
    (550, [3, 5, 0, 1, 5, 0], 130),
    (550, [8, 9, 0, 8, 9, 0], 413),
    (550, [5, 4, 0, 5, 4, 0], 425),
    (552, [3, 8, 0, 1, 1, 0], 315),
    (552, [3, 6, 0, 1, 1, 0], 483),
    (552, [3, 5, 0, 1, 5, 0], 130),
    (552, [8, 9, 0, 8, 9, 0], 412),
    (552, [5, 4, 0, 5, 4, 0], 425),
    (554, [3, 6, 0, 1, 1, 0], 483),
    (554, [3, 5, 0, 1, 5, 0], 130),
    (554, [5, 4, 0, 5, 4, 0], 425),
];

/// A `half4` output (`xr_half_fragment`, each lane a value a `half` holds, computed from the
/// pixel), compiled with fast math ([`ROUNDING`]), blended into an attachment of each linear
/// extended-range format under each of [`XR_BLENDS`], across passes ([`xr_across`]) and in the
/// first draw's pass ([`xr_within`]): Metal's side draws each, and this platform's blended draw
/// dies at its selector naming `extended-range-blend-source` ([`crate::answer::DIES`]), which
/// the runtime announces for such an output; [`HALF4_FAST_GAP`]'s 13 configurations
/// are where its blends missed Metal's. Each unblended draw stores Metal's values, bit for bit,
/// and none announces `extended-range-blends-stored-codes`
#[test]
fn half4_extended_range_blends_announce_their_source() {
    let drawn = ("fast math", ROUNDING, "xr_half_fragment");
    let shape = (&XR_LINEAR[..], &XR_BLENDS[..]);
    let named = "(metal2vulkan-degrade-extended-range-blend-source)";
    let at = "\"drawPrimitives:vertexStart:vertexCount:\"";
    let expect = XrExpect::Announced(Some(&HALF4_FAST_GAP));
    let sides = xr_sides();
    let (across, _) = xr_across(&sides, drawn, shape, expect);
    let (within, _) = xr_within(&sides, drawn, expect);
    let (died, differ): (Vec<_>, Vec<_>) = (across.into_iter().chain(within))
        .partition(|what| what.contains(named) && what.contains(at));
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
    if sides.1.is_some() {
        let blended = 2 * XR_LINEAR.len() * XR_BLENDS.len();
        assert_eq!(died.len(), blended, "the blended draws that die: {died:#?}");
    }
}

/// A `half4` output (`xr_half_fragment`) compiled without fast math ([`ROUNDING_SAFE`]), as
/// RenderBox's `primitive_image` functions are, blended into an attachment of each linear
/// extended-range format under each of [`XR_BLENDS`] and of each `_sRGB` twin under each of
/// [`XR_SRGB_BLENDS`]: the runtime rounds each product to `half` as Metal's compiler does
/// there. Across passes ([`xr_across`]) each state is made and stores Metal's
/// values, bit for bit, and none announces `extended-range-blend-source` or
/// `extended-range-blends-stored-codes`; in the first draw's pass ([`xr_within`]) each linear
/// one is made and stores Metal's values, bit for bit, and none announces the stored codes,
/// as the runtime holds the attachment's `half` values through the pass
#[test]
fn half4_extended_range_blends_are_metals_without_fast_math() {
    let drawn = ("fast math off", ROUNDING_SAFE, "xr_half_fragment");
    let source = rt::Degradation::ExtendedRangeBlendSource;
    let stored_codes = rt::Degradation::ExtendedRangeBlendsStoredCodes;
    let sides = xr_sides();
    let mut differ = Vec::new();
    for shape in [
        (&XR_LINEAR[..], &XR_BLENDS[..]),
        (&XR_SRGB[..], &XR_SRGB_BLENDS[..]),
    ] {
        differ.extend(xr_across(&sides, drawn, shape, XrExpect::Exact(&[])).0);
    }
    for kind in [source, stored_codes] {
        if heard(kind) {
            differ.push(format!("{} across passes", kind.code()));
        }
    }
    differ.extend(xr_within(&sides, drawn, XrExpect::Exact(&[])).0);
    if heard(stored_codes) {
        differ.push(format!("{} in one pass", stored_codes.code()));
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
}

/// The floats `xr_read` reads of a Private 64 by 64 `BGRA10_XR` attachment after one pass on
/// platform `P` that clears it to `clear` and draws `xr_half_fragment` of seed 1 into it under
/// source and destination factor one, added, colour and alpha.
fn xr_blended_onto_clear<P: Platform>(p: &P, d: &P::Device, clear: [f64; 4]) -> Vec<u8> {
    use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
    use reims_vgpu_core::platform::resource::Library as _;
    use reims_vgpu_core::platform::types::ClearColor;
    use reims_vgpu_core::rt::Hold;

    use crate::gpu::serialized::tests::rounded_render_of;

    let library = d.new_library(ROUNDING).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    let functions = [function("sweep_vertex"), function("xr_half_fragment")];
    let mut resolve = |_: Hold<'_>, r: u64| {
        let at = usize::try_from(r - 1).unwrap();
        Ok(Some(functions[at].clone_ref()))
    };
    let colour = [
        (0, 0),
        (1, 552),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 0),
        (6, 1),
        (7, 1),
        (8, 0),
    ];
    let blob = rounded_render_of(&["02 04 02000000"], &[&colour], 0);
    let desc = d
        .deserialize_render_pipeline_descriptor(Hold::NONE, &blob, &mut resolve)
        .unwrap()
        .unwrap();
    let state = d
        .new_render_pipeline_state_from_descriptor(Some(&desc), 0)
        .unwrap()
        .unwrap()
        .unwrap();
    let read = d
        .new_compute_pipeline_state_for_function(Some(&function("xr_read")))
        .unwrap()
        .unwrap();
    let private = p.texture_descriptor(&TextureDesc {
        pixel_format: 552,
        width: 64,
        height: 64,
        usage: READ | TARGET,
        resource_options: Some(0x20),
        ..TextureDesc::default()
    });
    let t = d.new_texture(OnFault::Die, &private).unwrap().unwrap();
    let seed = buffer::<P>(d, 4, 0, &1u32.to_le_bytes());
    let length = 64 * 64 * 16;
    let out = buffer::<P>(d, length, 0, &[]);
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let pass = p.render_pass_descriptor();
    let [red, green, blue, alpha] = clear;
    let attachment = ColorAttachmentSet {
        texture: Some(Some(&t)),
        load_action: Some(CLEAR),
        store_action: Some(STORE),
        clear_color: Some(ClearColor {
            red,
            green,
            blue,
            alpha,
        }),
        ..ColorAttachmentSet::default()
    };
    pass.set_color_attachment(OnFault::Die, 0, &attachment)
        .unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    for cmd in [
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&state),
        },
        RenderCmd::SetFragmentBuffer {
            buffer: Some(&seed),
            offset: 0,
            index: 0,
        },
        RenderCmd::DrawPrimitives {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 3,
        },
        RenderCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    let c = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
    let outs = [Some(&out)];
    for cmd in [
        ComputeCmd::SetComputePipelineState { state: Some(&read) },
        ComputeCmd::SetTexture {
            texture: Some(&t),
            index: 0,
        },
        ComputeCmd::SetBuffers {
            buffers: &outs,
            offsets: &[0],
            range: NsRange::new(0, 1),
        },
        ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(64, 64, 1),
            threads_per_threadgroup: MtlSize::new(8, 8, 1),
        },
        ComputeCmd::EndEncoding,
    ] {
        c.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    let n = usize::try_from(length).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the kernel completed.
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// A `half4` blend of one and one (`xr_blended_onto_clear`, its output computed under fast
/// math) onto a `BGRA10_XR` attachment its pass cleared: onto a clear colour of codes (every
/// channel 0 but alpha 0.5, which `BGRA10_XR` holds exactly), and onto a clear colour whose red
/// lies between two codes (0.0005, between 0 and 1/510), where Metal's tile memory holds the
/// clear's halves. Metal's side draws each; this platform's draw dies at its selector naming
/// `extended-range-blend-source` ([`crate::answer::DIES`]), which the runtime announces for a
/// computed `half4` under fast math blended into an extended-range attachment, whatever the
/// factors, though a blend of one and one is among those it measured Metal's
/// ([`HALF4_FAST_GAP`] lists none such)
#[test]
fn a_blend_onto_a_clear_between_codes_is_metals() {
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new(Box::new(NoLauncher));
    let (p, d) = (platform(), dev());
    let kind = rt::Degradation::ExtendedRangeBlendSource;
    for clear in [[0.0, 0.0, 0.0, 0.5], [0.0005, 0.0, 0.0, 0.5]] {
        xr_blended_onto_clear(&mp, &md, clear);
        let site = crate::answer::dies_of(kind, || xr_blended_onto_clear(&p, &d, clear));
        assert_eq!(
            site.file, "drawPrimitives:vertexStart:vertexCount:",
            "{clear:?}"
        );
    }
}

/// The bytes of `target`, a Private 256 by 256 `BGRA10_XR` texture, after a command buffer on
/// platform `P` that copies `first` into it, then runs a pass that loads and stores it and
/// draws `state` once, a triangle of the three vertices of `positions` (`reach_vertex`), then
/// copies it into a buffer.
fn reach_pass<P: Platform>(
    p: &P,
    d: &P::Device,
    (state, target): (&P::RenderPipeline, &P::Texture),
    first: &[u8],
    positions: &P::Buffer,
) -> Vec<u8> {
    use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
    use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};

    // `MTLLoadActionLoad`.
    const LOAD: u64 = 1;
    let length = u64::try_from(first.len()).unwrap();
    let up = buffer::<P>(d, length, 0, first);
    let down = buffer::<P>(d, length, 0, &[]);
    let all = MtlSize::new(256, 256, 1);
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    for cmd in [
        BlitCmd::CopyBufferToTexture {
            source_buffer: &up,
            source_offset: 0,
            source_bytes_per_row: 256 * 8,
            source_bytes_per_image: length,
            source_size: all,
            destination_texture: target,
            destination_slice: 0,
            destination_level: 0,
            destination_origin: MtlOrigin::new(0, 0, 0),
        },
        BlitCmd::EndEncoding,
    ] {
        blit.encode(OnFault::Die, cmd).unwrap();
    }
    let pass = p.render_pass_descriptor();
    let attachment = ColorAttachmentSet {
        texture: Some(Some(target)),
        load_action: Some(LOAD),
        store_action: Some(STORE),
        ..ColorAttachmentSet::default()
    };
    pass.set_color_attachment(OnFault::Die, 0, &attachment)
        .unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    for cmd in [
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(state),
        },
        RenderCmd::SetVertexBuffer {
            buffer: Some(positions),
            offset: 0,
            index: 0,
        },
        RenderCmd::DrawPrimitives {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 3,
        },
        RenderCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    for cmd in [
        BlitCmd::CopyTextureToBuffer {
            source_texture: target,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: all,
            destination_buffer: &down,
            destination_offset: 0,
            destination_bytes_per_row: 256 * 8,
            destination_bytes_per_image: length,
        },
        BlitCmd::EndEncoding,
    ] {
        blit.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    let n = usize::try_from(length).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the command buffer completed.
    unsafe { std::slice::from_raw_parts(down.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// A pass that loads and stores a 256 by 256 `BGRA10_XR` target of random codes (unused bits
/// set, alpha codes past [384, 894]) and draws one small triangle, a direct triangle list,
/// whose primitive reaches a few of its 64 tiles of 32 by 32. Metal rewrites only the tiles
/// the primitive reaches (unused bits canonical, alpha clamped) and keeps the rest's bytes.
/// The runtime's first pass of the pipeline rewrites every tile of the draw's viewport and
/// announces `unreached-tiles-rewritten`, relayed; once the draw's capture form is made, on
/// the runtime's own threads, a pass finds the tiles from the positions
/// the vertex function computes and rewrites those alone, as Metal. The runtime announces a
/// relayed kind once per device (`crate::platform::device_options`), so the test requires the
/// first pass to announce it, runs the pass again, 100 ms apart, until its bytes are Metal's,
/// byte for byte (at most 50 passes), and requires the same of two passes after it
#[test]
fn captured_reach_rewrites_the_tiles_metal_does() {
    use reims_vgpu_core::platform::resource::Library as _;
    use reims_vgpu_core::rt::Hold;

    use crate::gpu::serialized::tests::rounded_render_of;

    fn made<P: Platform>(d: &P::Device) -> P::RenderPipeline {
        let library = d.new_library(ROUNDING).unwrap().unwrap().unwrap();
        let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
        let functions = [function("reach_vertex"), function("xr_constant_fragment")];
        let mut resolve = |_: Hold<'_>, r: u64| {
            let at = usize::try_from(r - 1).unwrap();
            Ok(Some(functions[at].clone_ref()))
        };
        let blob = rounded_render_of(&["02 04 02000000"], &[&[(0, 0), (1, 552)]], 0);
        let desc = d
            .deserialize_render_pipeline_descriptor(Hold::NONE, &blob, &mut resolve)
            .unwrap()
            .unwrap();
        d.new_render_pipeline_state_from_descriptor(Some(&desc), 0)
            .unwrap()
            .unwrap()
            .unwrap()
    }
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new(Box::new(NoLauncher));
    let (p, d) = (platform(), dev());
    let shape = TextureDesc {
        pixel_format: 552,
        width: 256,
        height: 256,
        usage: READ | TARGET,
        resource_options: Some(0x20),
        ..TextureDesc::default()
    };
    // A triangle over pixels (40, 40) to (100, 100): tiles (1, 1) to (3, 3).
    let corners: [[f32; 4]; 3] = [
        [-0.6875, 0.6875, 0.5, 1.0],
        [-0.21875, 0.6875, 0.5, 1.0],
        [-0.6875, 0.21875, 0.5, 1.0],
    ];
    let positions: Vec<u8> = corners
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let first = (0..256 * 256 * 8u32)
        .map(|i| u8::try_from((i.wrapping_mul(2_654_435_761) >> 24) & 0xff).unwrap())
        .collect::<Vec<u8>>();
    let length = u64::try_from(positions.len()).unwrap();
    let theirs = {
        let target = md.new_texture(OnFault::Die, &mp.texture_descriptor(&shape));
        let target = target.unwrap().unwrap();
        let at = buffer::<MetalPlatform>(&md, length, 0, &positions);
        reach_pass(
            &mp,
            &md,
            (&made::<MetalPlatform>(&md), &target),
            &first,
            &at,
        )
    };
    assert_ne!(theirs, first, "Metal rewrote no tile");
    let state = made::<VulkanPlatform>(&d);
    let target = d.new_texture(OnFault::Die, &p.texture_descriptor(&shape));
    let target = target.unwrap().unwrap();
    let at = buffer::<VulkanPlatform>(&d, length, 0, &positions);
    let kind = rt::Degradation::UnreachedTilesRewritten;
    // How many tiles of each pass differ from Metal's.
    let mut passes = Vec::new();
    for _ in 0..50 {
        let ours = reach_pass(&p, &d, (&state, &target), &first, &at);
        if passes.is_empty() {
            assert!(heard(kind), "the first pass announced no {}", kind.code());
        }
        let tiles = (ours.chunks(8).zip(theirs.chunks(8)).enumerate())
            .filter(|(_, (o, t))| o != t)
            .map(|(i, _)| (i % 256 / 32, i / 256 / 32))
            .collect::<std::collections::BTreeSet<_>>();
        passes.push(tiles.len());
        if tiles.is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    assert_eq!(
        passes.last(),
        Some(&0),
        "no pass's bytes were Metal's; tiles off Metal's by pass: {passes:?}"
    );
    for _ in 0..2 {
        let ours = reach_pass(&p, &d, (&state, &target), &first, &at);
        assert!(
            ours == theirs,
            "a later pass: bytes off Metal's ({passes:?})"
        );
    }
}

/// A render blob of `textureWriteRoundingMode` 2 decodes, and its state is decided once made,
/// as Metal's: a compile's refusal is the `NSError` (a float fragment's colour 0 of
/// `R32Uint`, beside a half-float colour 1); a half-float attachment of write mask none makes
/// its state, and so does one the function has no output for (colour 1 of `RGBA16Float`
/// beside a written colour 0 of `RGBA8Unorm`), and one it writes, unblended, of 1 sample, of 2
/// or of 4; one it writes blended by one and one, or whose alpha alpha to coverage reads dies
/// as the state is made, where Metal makes each
#[test]
fn mode_2_render_states_are_decided_once_made() {
    use reims_vgpu_core::platform::resource::Library as _;
    use reims_vgpu_core::raise::death::DeathKind;
    use reims_vgpu_core::rt::Hold;

    use crate::gpu::serialized::NEAREST_EVEN;
    use crate::gpu::serialized::tests::rounded_render_of;

    /// A blob's 4-byte render entries past its fragment ref, and its colour attachments'.
    type Blob<'a> = (&'a [&'a str], &'a [&'a [(u8, u32)]]);

    fn state<P: Platform>(
        d: &P::Device,
        entries: &[&str],
        colours: &[&[(u8, u32)]],
    ) -> Result<Option<P::RenderPipeline>, reims_vgpu_core::platform::types::NsErrorText> {
        let library = d.new_library(ROUNDING).unwrap().unwrap().unwrap();
        let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
        let functions = [function("sweep_vertex"), function("sweep_fragment")];
        let mut resolve = |_: Hold<'_>, r: u64| {
            let at = usize::try_from(r - 1).unwrap();
            Ok(Some(functions[at].clone_ref()))
        };
        let mut all = vec!["02 04 02000000"];
        all.extend(entries);
        let blob = rounded_render_of(&all, colours, NEAREST_EVEN);
        let desc = d
            .deserialize_render_pipeline_descriptor(Hold::NONE, &blob, &mut resolve)
            .unwrap()
            .unwrap();
        d.new_render_pipeline_state_from_descriptor(Some(&desc), 0)
            .unwrap()
    }
    let d = dev();
    let metal = MetalPlatform::system_default_device();
    let refused: [&[(u8, u32)]; 2] = [&[(0, 0), (1, 53)], &[(0, 1), (1, 65)]];
    let unwritten: [&[(u8, u32)]; 1] = [&[(0, 0), (1, 115), (9, 0)]];
    let no_output: [&[(u8, u32)]; 2] = [&[(0, 0), (1, 70)], &[(0, 1), (1, 115)]];
    let written: [&[(u8, u32)]; 1] = [&[(0, 0), (1, 115)]];
    assert!(state::<VulkanPlatform>(&d, &[], &refused).is_err());
    let made: [Blob<'_>; 5] = [
        (&[], &unwritten),
        (&[], &no_output),
        (&[], &written),
        (&["04 04 02000000"], &written),
        (&["04 04 04000000"], &written),
    ];
    let additive: [&[(u8, u32)]; 1] = [&[
        (0, 0),
        (1, 115),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 0),
        (6, 1),
        (7, 1),
        (8, 0),
    ]];
    let dies: [Blob<'_>; 2] = [(&[], &additive), (&["05 04 01000000"], &written)];
    for (entries, colours) in made {
        assert!(
            matches!(state::<VulkanPlatform>(&d, entries, colours), Ok(Some(_))),
            "{colours:?}"
        );
    }
    for (entries, colours) in dies {
        let site =
            catch_death(|| state::<VulkanPlatform>(&d, entries, colours).is_ok()).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (
                DeathKind::Abort,
                "newRenderPipelineStateWithDescriptor:options:reflection:error:"
            ),
            "{entries:?} {colours:?}"
        );
    }
    if let Some(md) = &metal {
        assert!(state::<MetalPlatform>(md, &[], &refused).is_err());
        for (entries, colours) in made.into_iter().chain(dies) {
            assert!(
                matches!(state::<MetalPlatform>(md, entries, colours), Ok(Some(_))),
                "{entries:?} {colours:?}"
            );
        }
    }
}

/// A render state of `textureWriteRoundingMode` 2 whose half-float colour attachment stores
/// what the fragment function writes as written rounds as Metal's, byte for byte, over the
/// sweeps through unorm values, past 1 by float steps and past the largest half: `R16Float`,
/// `RG16Float` and `RGBA16Float`, unblended and blended by one and zero, of a `float4` and of
/// a `half4` output, and of a write mask of red and green alone, and of none
#[test]
fn mode_2_half_float_attachments_stored_as_written_round_as_metals() {
    use crate::gpu::serialized::NEAREST_EVEN;

    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new(Box::new(NoLauncher));
    let (p, d) = (platform(), dev());
    let replace: &[(u8, u32)] = &[(2, 1), (3, 1), (4, 0), (5, 0), (6, 1), (7, 0), (8, 0)];
    let drawn = [
        PLAIN,
        Drawn {
            colour: replace,
            ..PLAIN
        },
        Drawn {
            fragment: "sweep_half_fragment",
            ..PLAIN
        },
        Drawn {
            colour: &[(9, 0xc)],
            ..PLAIN
        },
        Drawn {
            colour: &[(9, 0)],
            ..PLAIN
        },
    ];
    for format in [(25, 2), (65, 4), (115, 8)] {
        for (n, &how) in drawn.iter().enumerate() {
            for kind in [0, 1, 3] {
                let ours = swept(&p, &d, (false, NEAREST_EVEN), format, kind, how);
                let theirs = swept(&mp, &md, (false, NEAREST_EVEN), format, kind, how);
                assert!(ours == theirs, "{format:?} drawn {n} sweep {kind}");
            }
        }
    }
}

/// How a texel's bytes read as channels, for [`half_outputs_store_within_the_bound_or_die_named`]:
/// their codes, a float format's bits as ordered integers (one apart a unit in the last place), and
/// a shared-exponent format's values in units of the finer of the two codes compared.
#[derive(Clone, Copy, Debug)]
enum Channels {
    /// One byte each.
    U8,
    /// `RGB10A2Unorm`'s 10, 10, 10 and 2 bits.
    Packed1010102,
    /// Two bytes each, unsigned.
    U16,
    /// `RG11B10Float`'s 11, 11 and 10 bits, each ordered as its code.
    Packed111110,
    /// `RGB9E5Float`'s three 9-bit mantissas over a shared 5-bit exponent.
    Shared9e5,
    /// A `half` each.
    Half,
    /// A `float` each.
    Float,
}

impl Channels {
    /// The bytes per channel group the decoder steps over.
    fn decoded(self, bytes: &[u8]) -> Vec<f64> {
        let word = |c: &[u8]| u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        let ordered = |bits: i64, sign: i64| {
            if bits & sign != 0 {
                -(bits & (sign - 1))
            } else {
                bits
            }
        };
        match self {
            Channels::U8 => bytes.iter().map(|&b| f64::from(b)).collect(),
            Channels::U16 => bytes
                .chunks(2)
                .map(|c| f64::from(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            Channels::Packed1010102 => (bytes.chunks(4).map(word))
                .flat_map(|w| [w & 0x3ff, (w >> 10) & 0x3ff, (w >> 20) & 0x3ff, w >> 30])
                .map(f64::from)
                .collect(),
            Channels::Packed111110 => (bytes.chunks(4).map(word))
                .flat_map(|w| [w & 0x7ff, (w >> 11) & 0x7ff, w >> 22])
                .map(f64::from)
                .collect(),
            Channels::Shared9e5 => (bytes.chunks(4).map(word))
                .flat_map(|w| {
                    let scale = (2f64).powi(i32::try_from(w >> 27).unwrap() - 24);
                    [w & 0x1ff, (w >> 9) & 0x1ff, (w >> 18) & 0x1ff].map(|m| f64::from(m) * scale)
                })
                .collect(),
            Channels::Half => bytes
                .chunks(2)
                .map(|c| ordered(i64::from(u16::from_le_bytes([c[0], c[1]])), 0x8000) as f64)
                .collect(),
            Channels::Float => (bytes.chunks(4).map(word))
                .map(|w| ordered(i64::from(w), 0x8000_0000) as f64)
                .collect(),
        }
    }

    /// How far apart `ours` and `theirs` lie at most, in codes, units in the last place, or, for
    /// a shared exponent, the finer code of the two compared; and at how many channels they
    /// differ.
    fn apart(self, ours: &[u8], theirs: &[u8]) -> (f64, usize) {
        let (o, m) = (self.decoded(ours), self.decoded(theirs));
        let step = |b: &[u8]| -> Vec<f64> {
            (b.chunks(4))
                .flat_map(|c| {
                    let e = u32::from_le_bytes([c[0], c[1], c[2], c[3]]) >> 27;
                    [(2f64).powi(i32::try_from(e).unwrap() - 24); 3]
                })
                .collect()
        };
        let steps = match self {
            Channels::Shared9e5 => Some((step(ours), step(theirs))),
            _ => None,
        };
        let mut most = 0f64;
        let mut differ = 0;
        for (i, (&a, &b)) in o.iter().zip(&m).enumerate() {
            if a != b {
                differ += 1;
            }
            let unit = steps.as_ref().map_or(1.0, |(s, t)| s[i].min(t[i]));
            most = most.max((a - b).abs() / unit);
        }
        (most, differ)
    }
}

/// A `half` colour output (`rounding.metal`'s `sweep_half_fragment`, a `half4` converted from
/// `float`, every channel the sweep's value), which the runtime rounds to `half` before the blend
/// and the format's conversion, drawn under
/// `textureWriteRoundingMode` 0 over the sweeps from -0.25 to 1.25 (0) and from -2 to 2 (4) onto a
/// 256 by 256 attachment loaded with texels from 0 to 1 (`LOAD`; zeros for `RG11B10Float` and
/// `RGB9E5Float`), from a congruential sequence (seed 7919, `x * 1103515245 + 12345`, bits 8 to 23
/// of each step: a unorm byte its low byte, a `half` the step modulo 0x3c01, a `float` the step
/// over 65535), blended as each case says, the blend colour (0.3, 0.6, 0.9, 0.45). Compiled with
/// fast math ([`ROUNDING`]), the function notes `half-output-fast-math-blend`, relayed in the
/// accepted class of fast-math rewrites: every draw draws, and one the rule below relayed
/// stores within the bound, against Metal's bytes: one code, one unit in the
/// last place of a `half`, or a `float` channel's own bits. Compiled without it
/// ([`ROUNDING_SAFE`]), the function notes `half-output-precision`, and a draw into an attachment
/// that stores it past the bound under fast math (`encode_render`'s `half_apart`) dies at its
/// selector, naming the kind; the cases measured past it are the ones the record of
/// 2026-10-11 lists (`One` and `One` into `R16Float` 384 `half` units apart, into `RG16Float`
/// 3072, `DestinationColor` and `OneMinusDestinationColor` into `RGBA16Float` 2213, `SourceAlpha`
/// and `OneMinusSourceAlpha` into `RGBA32Float` 1 float unit, into an 8-bit `_sRGB` format 2
/// codes at `DestinationColor` and `OneMinusDestinationColor`), and those that die with the same
/// rule within it (`One` and `Zero` into `RGBA16Float`, whose zero factor's product Metal keeps
/// at texels this sweep does not load, premultiplied `One` and `OneMinusSourceAlpha` into
/// `RGBA16Float` and `RG16Float`, `BlendColor` and `OneMinusBlendColor` into `RGBA16Float`,
/// `OneMinusBlendColor`
/// and `Zero` into `R16Float`, `SourceColor` and `DestinationColor` into `RGBA32Float`,
/// `SourceAlpha` and `OneMinusSourceAlpha` into `R32Float` and `RG32Float`). That death is strict
/// mode's: by default (`answer::relaxed`) each such draw draws, relaying the difference, counted
/// (`answer::Inexact::HalfOutputBlend`). A
/// `float4` output (`sweep_fragment`) under the same state, which the runtime notes nothing of,
/// draws and stores Metal's bytes in every case
#[test]
fn half_outputs_store_within_the_bound_or_die_named() {
    use reims_vgpu_core::platform::resource::Library as _;
    use reims_vgpu_core::rt::Hold;

    use crate::gpu::serialized::tests::rounded_render_of;

    /// `MTLLoadActionLoad`.
    const LOAD: u64 = 1;
    /// `MTLBlendOperation`: add, minimum and maximum.
    const ADD: u32 = 0;
    const MIN: u32 = 3;
    const MAX: u32 = 4;

    /// The bytes of a 256 by 256 attachment of `format`, `bytes` a texel, loaded with `under`,
    /// after `fragment` draws the sweep `kind` into it, blended as `blend` says (each channel's
    /// source and destination factors and operation), unblended where `None`.
    fn drawn<P: Platform>(
        (p, d): (&P, &P::Device),
        (format, bytes): (u64, u64),
        (library, fragment, kind): (&[u8], &str, u32),
        blend: Option<(u32, u32, u32)>,
        under: &[u8],
    ) -> Vec<u8> {
        let library = d.new_library(library).unwrap().unwrap().unwrap();
        let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
        let functions = [function("sweep_vertex"), function(fragment)];
        let mut resolve = |_: Hold<'_>, r: u64| {
            let at = usize::try_from(r - 1).unwrap();
            Ok(Some(functions[at].clone_ref()))
        };
        let t = texture(
            p,
            d,
            format,
            READ | TARGET,
            (256, 256),
            Some((under, 256 * bytes)),
        );
        let kinds = buffer::<P>(d, 4, 0, &kind.to_le_bytes());
        let mut colour = vec![(0, 0), (1, u32::try_from(format).unwrap())];
        if let Some((source, destination, operation)) = blend {
            colour.extend([(2, 1), (3, source), (4, destination), (5, operation)]);
            colour.extend([(6, source), (7, destination), (8, operation)]);
        }
        let blob = rounded_render_of(&["02 04 02000000"], &[&colour], 0);
        let desc = d
            .deserialize_render_pipeline_descriptor(Hold::NONE, &blob, &mut resolve)
            .unwrap()
            .unwrap();
        let state = d
            .new_render_pipeline_state_from_descriptor(Some(&desc), 0)
            .unwrap()
            .unwrap()
            .unwrap();
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let pass = p.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(&t)),
            load_action: Some(LOAD),
            store_action: Some(STORE),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, 0, &attachment)
            .unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        for cmd in [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            },
            RenderCmd::SetFragmentBuffer {
                buffer: Some(&kinds),
                offset: 0,
                index: 0,
            },
            RenderCmd::SetBlendColorRed {
                red: 0.3,
                green: 0.6,
                blue: 0.9,
                alpha: 0.45,
            },
            RenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
            },
            RenderCmd::EndEncoding,
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        let length = 256 * 256 * bytes;
        let out = buffer::<P>(d, length, 0, &[]);
        let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        blit.encode(
            OnFault::Die,
            BlitCmd::CopyTextureToBuffer {
                source_texture: &t,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(256, 256, 1),
                destination_buffer: &out,
                destination_offset: 0,
                destination_bytes_per_row: 256 * bytes,
                destination_bytes_per_image: length,
            },
        )
        .unwrap();
        blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
        let n = usize::try_from(length).unwrap();
        // SAFETY: a Shared buffer's bytes, read after the blit completed.
        unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec()
    }

    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new(Box::new(NoLauncher));
    let (p, d) = (platform(), dev());
    let blends_relayed = || {
        (crate::answer::inexact_counts().into_iter())
            .find(|&(kind, _)| kind == crate::answer::Inexact::HalfOutputBlend)
            .map_or(0, |(_, n)| n)
    };
    let mut state = 7919u32;
    let (mut bytes, mut halves, mut floats) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..256 * 256 * 8 {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
        let step = (state >> 8) & 0xffff;
        bytes.push(u8::try_from(step & 0xff).unwrap());
        halves.extend(u16::try_from(step % 0x3c01).unwrap().to_le_bytes());
        floats.extend((f32::from(u16::try_from(step).unwrap()) / 65535.0).to_le_bytes());
    }
    let zeros = vec![0u8; 256 * 256 * 16];
    // Each format: its raw `MTLPixelFormat`, bytes a texel, channels and loaded texels; then the
    // blends relayed within the bound (`true`) or dying at the draw (`false`), as (source factor,
    // destination factor, operation) for every channel (`MTLBlendFactor`: 0 zero, 1 one, 2 source
    // colour, 3 one minus it, 4 source alpha, 5 one minus it, 6 destination colour, 7 one minus
    // it, 11 blend colour, 12 one minus it, 13 blend alpha, 14 one minus it).
    let unblended: Option<(u32, u32, u32)> = None;
    type Case<'a> = (
        (u64, u64),
        Channels,
        &'a [u8],
        Vec<(Option<(u32, u32, u32)>, bool)>,
    );
    let cases: Vec<Case<'_>> = vec![
        (
            (115, 8),
            Channels::Half,
            &halves,
            vec![
                (unblended, true),
                (Some((1, 0, ADD)), false),
                (Some((1, 5, ADD)), false),
                (Some((11, 12, ADD)), false),
                (Some((4, 5, MIN)), true),
                (Some((4, 5, MAX)), true),
                (Some((4, 5, ADD)), false),
                (Some((0, 5, ADD)), false),
                (Some((6, 7, ADD)), false),
                (Some((13, 14, ADD)), false),
            ],
        ),
        (
            (25, 2),
            Channels::Half,
            &halves,
            vec![
                (unblended, true),
                (Some((12, 0, ADD)), false),
                (Some((1, 1, ADD)), false),
                (Some((1, 5, ADD)), false),
            ],
        ),
        (
            (65, 4),
            Channels::Half,
            &halves,
            vec![(Some((1, 5, ADD)), false), (Some((1, 1, ADD)), false)],
        ),
        (
            (125, 16),
            Channels::Float,
            &floats,
            vec![
                (unblended, true),
                (Some((1, 5, ADD)), true),
                (Some((2, 6, ADD)), false),
                (Some((4, 5, ADD)), false),
                (Some((3, 3, ADD)), false),
                (Some((11, 5, ADD)), false),
            ],
        ),
        (
            (55, 4),
            Channels::Float,
            &floats,
            vec![(Some((4, 5, ADD)), false)],
        ),
        (
            (105, 8),
            Channels::Float,
            &floats,
            vec![(Some((4, 5, ADD)), false)],
        ),
        (
            (71, 4),
            Channels::U8,
            &bytes,
            vec![(Some((4, 5, ADD)), true), (Some((6, 7, ADD)), false)],
        ),
        (
            (81, 4),
            Channels::U8,
            &bytes,
            vec![(Some((6, 7, ADD)), false)],
        ),
        (
            (70, 4),
            Channels::U8,
            &bytes,
            vec![
                (unblended, true),
                (Some((4, 5, ADD)), true),
                (Some((6, 7, ADD)), true),
                (Some((0, 3, ADD)), true),
            ],
        ),
        (
            (90, 4),
            Channels::Packed1010102,
            &bytes,
            vec![(Some((4, 5, ADD)), true), (Some((6, 7, ADD)), true)],
        ),
        (
            (110, 8),
            Channels::U16,
            &bytes,
            vec![(Some((4, 5, ADD)), true)],
        ),
        (
            (92, 4),
            Channels::Packed111110,
            &zeros,
            vec![(unblended, true), (Some((4, 5, ADD)), true)],
        ),
        (
            (93, 4),
            Channels::Shared9e5,
            &zeros,
            vec![(unblended, true)],
        ),
    ];
    let mut wrong = Vec::new();
    for (format, channels, under, blends) in &cases {
        for &(blend, relayed) in blends {
            for kind in [0, 4] {
                let float4 = (ROUNDING, "sweep_fragment", kind);
                let ours = drawn((&p, &d), *format, float4, blend, under);
                let theirs = drawn((&mp, &md), *format, float4, blend, under);
                if ours != theirs {
                    wrong.push(format!(
                        "{format:?} {blend:?} sweep {kind}: a float4 output"
                    ));
                }
            }
            if !relayed {
                let safe = (ROUNDING_SAFE, "sweep_half_fragment", 0);
                let site = crate::answer::dies_of(rt::Degradation::HalfOutputPrecision, || {
                    drawn((&p, &d), *format, safe, blend, under)
                });
                if site.file != "drawPrimitives:vertexStart:vertexCount:" {
                    wrong.push(format!("{format:?} {blend:?}: died at {}", site.file));
                }
                let before = blends_relayed();
                crate::answer::relaxed(|| drawn((&p, &d), *format, safe, blend, under));
                if blends_relayed() != before + 1 {
                    wrong.push(format!("{format:?} {blend:?}: not relayed by default"));
                }
                let fast = (ROUNDING, "sweep_half_fragment", 0);
                drawn((&p, &d), *format, fast, blend, under);
                continue;
            }
            let (mut most, mut differ) = (0f64, 0);
            for kind in [0, 4] {
                let half4 = (ROUNDING, "sweep_half_fragment", kind);
                let ours = drawn((&p, &d), *format, half4, blend, under);
                let theirs = drawn((&mp, &md), *format, half4, blend, under);
                let (far, count) = channels.apart(&ours, &theirs);
                (most, differ) = (most.max(far), differ + count);
            }
            eprintln!(
                "half output into {format:?}, blend {blend:?}: {most} apart at most, {differ} \
                 channels differ"
            );
            let bound = match channels {
                Channels::Float => 0.0,
                _ => 1.0,
            };
            if most > bound {
                wrong.push(format!(
                    "{format:?} {blend:?}: {most} apart, {differ} channels"
                ));
            }
        }
    }
    let relayed = crate::answer::RELAYED.lock().clone();
    if !relayed.contains(&rt::Degradation::HalfOutputFastMathBlend) {
        wrong.push(format!(
            "half-output-fast-math-blend not relayed: {relayed:?}"
        ));
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// A fragment function's `float4` output (`sweep_fragment`) and `half4` one
/// (`sweep_half_fragment`) drawn unblended into an attachment of each bfloat format
/// (`R16Bfloat`, `RG16Bfloat`, `RGBA16Bfloat`), at `textureWriteRoundingMode` 0 and 2, over
/// each of `rounding.metal`'s five sweeps (through unorm values, past 1 and past -1 by float
/// steps, past the largest half, and across -2 to 2), stores Metal's codes, byte for byte:
/// the runtime writes such an output as Metal's tile holds and stores it (a
/// `float` rounded to `half` toward zero, saturating at 65504, or to nearest even at mode 2,
/// then the nearest bfloat code, ties to even). A state
/// that blends into one, by one and zero, which Metal's validation refuses, dies as it is made,
/// naming the runtime's refusal (`Unsupported` of `Format`); Metal's is not made here. A draw
/// in a pass that loads one and stores it, which Metal makes, dies at the draw, naming the
/// attachment (`bfloat_loaded`: the tiles the runtime's pass leaves are not Metal's)
#[test]
fn bfloat_draws_store_metals_codes() {
    use reims_vgpu_core::raise::death::DeathKind;

    use crate::gpu::serialized::NEAREST_EVEN;

    const STATE: &str = "newRenderPipelineStateWithDescriptor:options:reflection:error:";
    const DRAW: &str = "drawPrimitives:vertexStart:vertexCount:";

    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new(Box::new(NoLauncher));
    let (p, d) = (platform(), dev());
    let half = Drawn {
        fragment: "sweep_half_fragment",
        ..PLAIN
    };
    let mut differ = Vec::new();
    for format in [(26, 2), (66, 4), (116, 8)] {
        for (output, how) in [("float4", PLAIN), ("half4", half)] {
            for mode in [0, NEAREST_EVEN] {
                for kind in [0, 1, 2, 3, 4] {
                    let case = format!("{format:?} {output} mode {mode} sweep {kind}");
                    let theirs = swept(&mp, &md, (false, mode), format, kind, how);
                    *crate::answer::LAST_LINE.lock() = None;
                    match catch_death(|| swept(&p, &d, (false, mode), format, kind, how)) {
                        Ok(ours) if ours == theirs => {}
                        Ok(ours) => {
                            let off = ours.iter().zip(&theirs).filter(|(a, b)| a != b);
                            differ.push(format!("{case}: {} bytes differ", off.count()));
                        }
                        Err(site) => {
                            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                            differ.push(format!("{case}: died at {site:?}: {line}"));
                        }
                    }
                }
            }
        }
        let replace: &[(u8, u32)] = &[(2, 1), (3, 1), (4, 0), (5, 0), (6, 1), (7, 0), (8, 0)];
        let blended = Drawn {
            colour: replace,
            ..PLAIN
        };
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(|| swept(&p, &d, (false, 0), format, 0, blended));
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        match site {
            Err(site)
                if (site.kind, site.file) == (DeathKind::Abort, STATE)
                    && line.contains("feature: Format") => {}
            Err(site) => differ.push(format!("{format:?} blended: died at {site:?}: {line}")),
            Ok(_) => differ.push(format!("{format:?} blended: drawn")),
        }
        let loaded = Drawn { load: 1, ..PLAIN };
        swept(&mp, &md, (false, 0), format, 0, loaded);
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(|| swept(&p, &d, (false, 0), format, 0, loaded));
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        match site {
            Err(site)
                if (site.kind, site.file) == (DeathKind::Abort, DRAW)
                    && line.contains("loads and stores the bfloat colour attachment 0") => {}
            Err(site) => differ.push(format!("{format:?} loaded: died at {site:?}: {line}")),
            Ok(_) => differ.push(format!("{format:?} loaded: drawn")),
        }
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
}

/// A render state of `textureWriteRoundingMode` 2 made of what no other test compares rounds
/// as Metal's, byte for byte, over the sweeps through unorm values, past 1 by float steps
/// and past the largest half, and so does one of mode 0 of it: `R16Float` and `RG16Float` at
/// colour 0 under alpha to coverage; `RGBA16Float` at colour 0 under alpha to coverage of a
/// write mask without alpha; and `R16Float`, `RG16Float` and `RGBA16Float` at colours 1 and
/// 3, beside an `RGBA8Unorm` colour 0
#[test]
fn mode_2_half_float_attachments_no_other_test_compares_round_as_metals() {
    use crate::gpu::serialized::NEAREST_EVEN;

    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new(Box::new(NoLauncher));
    let (p, d) = (platform(), dev());
    let coverage: &[&str] = &["05 04 01000000"];
    let covered = Drawn {
        render: coverage,
        ..PLAIN
    };
    let mut cases = vec![
        ((25, 2), covered),
        ((65, 4), covered),
        (
            (115, 8),
            Drawn {
                colour: &[(9, 0xe)],
                ..covered
            },
        ),
    ];
    for format in [(25, 2), (65, 4), (115, 8)] {
        for index in [1, 3] {
            let four = Drawn {
                fragment: "sweep_four_fragment",
                index,
                ..PLAIN
            };
            cases.push((format, four));
        }
    }
    let mut differ = Vec::new();
    for (format, how) in cases {
        for mode in [0, NEAREST_EVEN] {
            for kind in [0, 1, 3] {
                let ours = swept(&p, &d, (false, mode), format, kind, how);
                let theirs = swept(&mp, &md, (false, mode), format, kind, how);
                if ours != theirs {
                    let off = ours.iter().zip(&theirs).filter(|(a, b)| a != b).count();
                    differ.push(format!(
                        "{format:?} {how:?} mode {mode} sweep {kind}: {off} bytes"
                    ));
                }
            }
        }
    }
    assert!(differ.is_empty(), "{differ:#?}");
}

/// The bytes of a 256 by 256 texture of `format`, `bytes` a texel, after `write_fragment`
/// writes the sweep `kind` (`rounding.metal`) into it, bound at fragment texture 0, under a
/// [`texture_writing_state`] of `textureWriteRoundingMode` `mode`, drawn over an `RGBA8Unorm`
/// colour attachment.
fn fragment_written<P: Platform>(
    p: &P,
    d: &P::Device,
    mode: u32,
    (format, bytes): (u64, u64),
    kind: u32,
) -> Vec<u8> {
    let state = texture_writing_state::<P>(d, mode, "write_fragment", false)
        .unwrap()
        .unwrap()
        .unwrap();
    let t = texture(p, d, format, READ | 2, (256, 256), None);
    let target = texture(p, d, 70, READ | TARGET, (256, 256), None);
    let kinds = buffer::<P>(d, 4, 0, &kind.to_le_bytes());
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let pass = p.render_pass_descriptor();
    let attachment = ColorAttachmentSet {
        texture: Some(Some(&target)),
        load_action: Some(CLEAR),
        store_action: Some(STORE),
        ..ColorAttachmentSet::default()
    };
    pass.set_color_attachment(OnFault::Die, 0, &attachment)
        .unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    for cmd in [
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&state),
        },
        RenderCmd::SetFragmentBuffer {
            buffer: Some(&kinds),
            offset: 0,
            index: 0,
        },
        RenderCmd::SetFragmentTexture {
            texture: Some(&t),
            index: 0,
        },
        RenderCmd::DrawPrimitives {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 3,
        },
        RenderCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    let length = 256 * 256 * bytes;
    let out = buffer::<P>(d, length, 0, &[]);
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    blit.encode(
        OnFault::Die,
        BlitCmd::CopyTextureToBuffer {
            source_texture: &t,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(256, 256, 1),
            destination_buffer: &out,
            destination_offset: 0,
            destination_bytes_per_row: 256 * bytes,
            destination_bytes_per_image: length,
        },
    )
    .unwrap();
    blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    let n = usize::try_from(length).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the blit completed.
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// A render state of `textureWriteRoundingMode` 2 whose fragment function writes a texture
/// is made, as Metal's is, and its draws write what Metal's do, byte for byte over the sweeps
/// through unorm values, past 1 by float steps and past the largest half, into a texture of
/// a format Metal's mode 2 writes as its default does (`RGBA8Unorm`, `BGRA8Unorm`,
/// `R32Float`, `RGBA32Float`); a draw writing one of a half-float format (`R16Float`,
/// `RG16Float`, `RGBA16Float`), which Metal's mode 2 rounds to nearest even, or of
/// `RG11B10Float`, which it rounds otherwise than its default (in the sweep through unorm
/// values; past 1 and past the largest half as its default), dies at the draw's selector,
/// and so does one writing `RGBX16Float` or `RGB9E5Float`, which nothing measured at the mode
/// ([`crate::gpu::serialized::MODE_2_TEXTURE_FORMATS`]). At modes 0 and 1 every format's
/// writes that Metal's are compared with are Metal's
#[test]
fn mode_2_fragment_texture_writes_are_metals_or_die_at_the_draw() {
    use crate::gpu::serialized::NEAREST_EVEN;

    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new(Box::new(NoLauncher));
    let (p, d) = (platform(), dev());
    let draw = |format: (u64, u64), kind| {
        let site =
            catch_death(|| fragment_written(&p, &d, NEAREST_EVEN, format, kind)).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (
                reims_vgpu_core::raise::death::DeathKind::Abort,
                "drawPrimitives:vertexStart:vertexCount:"
            ),
            "{format:?} sweep {kind}"
        );
    };
    let half = [(25, 2), (65, 4), (115, 8)];
    let rg11b10 = (92, 4);
    let rounded = [half[0], half[1], half[2], rg11b10];
    let rest = [(70, 4), (80, 4), (55, 4), (125, 16)];
    let unmeasured = [(315, 8), (93, 4)];
    let mut rg11b10_rounded = Vec::new();
    for kind in [0, 1, 3] {
        for format in rest.into_iter().chain(rounded) {
            for mode in 0..=2 {
                let theirs = fragment_written(&mp, &md, mode, format, kind);
                if mode == NEAREST_EVEN && rounded.contains(&format) {
                    draw(format, kind);
                    let rounds = theirs != fragment_written(&mp, &md, 0, format, kind);
                    if format == rg11b10 {
                        rg11b10_rounded.extend(rounds.then_some(kind));
                    } else {
                        assert!(
                            rounds,
                            "Metal's mode 2 writes {format:?} sweep {kind} as its default"
                        );
                    }
                    continue;
                }
                let ours = fragment_written(&p, &d, mode, format, kind);
                assert!(ours == theirs, "{format:?} mode {mode} sweep {kind}");
            }
        }
        for format in unmeasured {
            draw(format, kind);
        }
    }
    assert_eq!(
        rg11b10_rounded,
        [0],
        "the sweeps Metal's mode 2 writes RG11B10Float otherwise"
    );
}

/// A render state of `textureWriteRoundingMode` 2 whose fragment function writes a texture
/// through an argument buffer is made, as Metal's is, and its draw dies, as which textures
/// it writes no draw tells; one of `supportIndirectCommandBuffers`, whose indirect commands'
/// draws tell nothing either, dies as it is made, where Metal makes it. At modes 0 and 1
/// both are made
#[test]
fn mode_2_fragment_writes_through_handles_die() {
    use crate::gpu::serialized::NEAREST_EVEN;
    use reims_vgpu_core::raise::death::DeathKind;

    type V = crate::platform::VulkanPlatform;
    let d = dev();
    let p = platform();
    if let Some(md) = MetalPlatform::system_default_device() {
        for mode in 0..=2 {
            for icb in [false, true] {
                let made =
                    texture_writing_state::<MetalPlatform>(&md, mode, "write_args_fragment", icb);
                assert!(matches!(made, Ok(Ok(Some(_)))), "Metal: {mode} {icb}");
            }
        }
    }
    for mode in [0, 1] {
        for icb in [false, true] {
            let made = texture_writing_state::<V>(&d, mode, "write_args_fragment", icb);
            assert!(matches!(made, Ok(Ok(Some(_)))), "{mode} {icb}");
        }
    }
    let site = catch_death(|| {
        texture_writing_state::<V>(&d, NEAREST_EVEN, "write_args_fragment", true).is_ok()
    })
    .unwrap_err();
    assert_eq!(
        (site.kind, site.file),
        (
            DeathKind::Abort,
            "newRenderPipelineStateWithDescriptor:options:reflection:error:"
        )
    );
    let state = texture_writing_state::<V>(&d, NEAREST_EVEN, "write_args_fragment", false)
        .unwrap()
        .unwrap()
        .unwrap();
    let target = texture(&p, &d, 70, READ | TARGET, (16, 16), None);
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let pass = p.render_pass_descriptor();
    let attachment = ColorAttachmentSet {
        texture: Some(Some(&target)),
        load_action: Some(CLEAR),
        store_action: Some(STORE),
        ..ColorAttachmentSet::default()
    };
    pass.set_color_attachment(OnFault::Die, 0, &attachment)
        .unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    e.encode(
        OnFault::Die,
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&state),
        },
    )
    .unwrap();
    let site = catch_death(|| {
        e.encode(
            OnFault::Die,
            RenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
            },
        )
        .is_ok()
    })
    .unwrap_err();
    assert_eq!(
        (site.kind, site.file),
        (DeathKind::Abort, "drawPrimitives:vertexStart:vertexCount:")
    );
}

/// The resolved bytes of a 256 by 256 `format` (`bytes` a texel) colour attachment of
/// `samples` samples, cleared to 0.3, after `sweep_fragment` draws the sweep `kind` into it
/// under a state of `textureWriteRoundingMode` `mode` over the viewport from `offset` (of a
/// pixel, both ways: past 0 the first row and column are partly covered, so their resolve
/// averages the draw's value and the clear's), resolved by the drawing pass where `apart` is
/// false, else by a later pass that loads the samples and draws nothing.
fn resolved<P: Platform>(
    p: &P,
    d: &P::Device,
    (mode, samples): (u32, u64),
    (format, bytes): (u64, u64),
    kind: u32,
    (offset, apart): (f64, bool),
) -> Vec<u8> {
    use reims_vgpu_core::platform::resource::Library as _;
    use reims_vgpu_core::platform::types::ClearColor;
    use reims_vgpu_core::rt::Hold;

    use crate::gpu::serialized::tests::rounded_render_of;

    /// `MTLTextureType2DMultisample`, `MTLLoadActionLoad`, `MTLStoreActionMultisampleResolve`
    /// and `MTLStoreActionStoreAndMultisampleResolve`.
    const MULTISAMPLE: u64 = 4;
    const LOAD: u64 = 1;
    const RESOLVE: u64 = 2;
    const STORE_AND_RESOLVE: u64 = 3;
    let library = d.new_library(ROUNDING).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    let functions = [function("sweep_vertex"), function("sweep_fragment")];
    let mut resolve = |_: Hold<'_>, r: u64| {
        let at = usize::try_from(r - 1).unwrap();
        Ok(Some(functions[at].clone_ref()))
    };
    let format32 = u32::try_from(format).unwrap();
    let colour: &[(u8, u32)] = &[(0, 0), (1, format32)];
    let count = format!("04 04 {samples:02x}000000");
    let blob = rounded_render_of(&["02 04 02000000", &count], &[colour], mode);
    let desc = d
        .deserialize_render_pipeline_descriptor(Hold::NONE, &blob, &mut resolve)
        .unwrap()
        .unwrap();
    let state = d
        .new_render_pipeline_state_from_descriptor(Some(&desc), 0)
        .unwrap()
        .unwrap()
        .unwrap();
    let samples = d
        .new_texture(
            OnFault::Die,
            &p.texture_descriptor(&TextureDesc {
                texture_type: MULTISAMPLE,
                pixel_format: format,
                width: 256,
                height: 256,
                sample_count: samples,
                usage: TARGET,
                resource_options: Some(0x20),
                ..TextureDesc::default()
            }),
        )
        .unwrap()
        .unwrap();
    let target = texture(p, d, format, READ | TARGET, (256, 256), None);
    let kinds = buffer::<P>(d, 4, 0, &kind.to_le_bytes());
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let pass = |load, store, resolves: bool| {
        let pass = p.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(&samples)),
            resolve_texture: resolves.then_some(Some(&target)),
            load_action: Some(load),
            store_action: Some(store),
            clear_color: Some(ClearColor {
                red: 0.3,
                green: 0.3,
                blue: 0.3,
                alpha: 0.3,
            }),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, 0, &attachment)
            .unwrap();
        pass
    };
    let first = match apart {
        false => pass(CLEAR, STORE_AND_RESOLVE, true),
        true => pass(CLEAR, STORE, false),
    };
    let e = cb.render_encoder(OnFault::Die, &first).unwrap().unwrap();
    for cmd in [
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&state),
        },
        RenderCmd::SetViewport {
            viewport: MtlViewport {
                origin_x: offset,
                origin_y: offset,
                width: 256.0 - offset,
                height: 256.0 - offset,
                znear: 0.0,
                zfar: 1.0,
            },
        },
        RenderCmd::SetFragmentBuffer {
            buffer: Some(&kinds),
            offset: 0,
            index: 0,
        },
        RenderCmd::DrawPrimitives {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 3,
        },
        RenderCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    if apart {
        let e = cb
            .render_encoder(OnFault::Die, &pass(LOAD, RESOLVE, true))
            .unwrap()
            .unwrap();
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    }
    let length = 256 * 256 * bytes;
    let out = buffer::<P>(d, length, 0, &[]);
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    blit.encode(
        OnFault::Die,
        BlitCmd::CopyTextureToBuffer {
            source_texture: &target,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(256, 256, 1),
            destination_buffer: &out,
            destination_offset: 0,
            destination_bytes_per_row: 256 * bytes,
            destination_bytes_per_image: length,
        },
    )
    .unwrap();
    blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    let n = usize::try_from(length).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the blit completed.
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// A render state of `textureWriteRoundingMode` 2 whose half-float colour attachment of two
/// samples, or of four, stores what the fragment function writes as written resolves as
/// Metal's, byte for byte, and so does one of mode 0 of it ([`resolves_as_metals`])
#[test]
fn mode_2_multisampled_half_float_attachments_resolve_as_metals() {
    resolves_as_metals(2);
    resolves_as_metals(4);
}

/// The resolved bytes of a 64 by 64 `format` (`bytes` a texel) colour attachment of
/// `samples` samples after `xr_loaded_fragment` draws two tables of `float4`s, a value a
/// pixel, into it ([`drawn_specials`]): the first over every sample, the second over the
/// viewport from half a pixel in, which covers part of the first row's and column's samples,
/// so their resolve averages a value of each table. Resolved by the drawing pass's store
/// action `store` (`MultisampleResolve` or `StoreAndMultisampleResolve`), or, where `store`
/// is `Store`, by a later pass that loads the samples and draws nothing.
fn drawn_resolved<P: Platform>(
    p: &P,
    d: &P::Device,
    (format, bytes, samples): (u64, u64, u64),
    store: u64,
) -> Vec<u8> {
    use reims_vgpu_core::platform::resource::Library as _;
    use reims_vgpu_core::rt::Hold;

    use crate::gpu::serialized::tests::rounded_render_of;

    /// `MTLTextureType2DMultisample`, `MTLLoadActionLoad` and
    /// `MTLStoreActionMultisampleResolve`.
    const MULTISAMPLE: u64 = 4;
    const LOAD: u64 = 1;
    const RESOLVE: u64 = 2;
    let library = d.new_library(ROUNDING).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    let functions = [function("sweep_vertex"), function("xr_loaded_fragment")];
    let mut resolve = |_: Hold<'_>, r: u64| {
        let at = usize::try_from(r - 1).unwrap();
        Ok(Some(functions[at].clone_ref()))
    };
    let format32 = u32::try_from(format).unwrap();
    let colour: &[(u8, u32)] = &[(0, 0), (1, format32)];
    let count = format!("04 04 {samples:02x}000000");
    let blob = rounded_render_of(&["02 04 02000000", &count], &[colour], 0);
    let desc = d
        .deserialize_render_pipeline_descriptor(Hold::NONE, &blob, &mut resolve)
        .unwrap()
        .unwrap();
    let state = d
        .new_render_pipeline_state_from_descriptor(Some(&desc), 0)
        .unwrap()
        .unwrap()
        .unwrap();
    let multisampled = d
        .new_texture(
            OnFault::Die,
            &p.texture_descriptor(&TextureDesc {
                texture_type: MULTISAMPLE,
                pixel_format: format,
                width: 64,
                height: 64,
                sample_count: samples,
                usage: TARGET,
                resource_options: Some(0x20),
                ..TextureDesc::default()
            }),
        )
        .unwrap()
        .unwrap();
    let target = texture(p, d, format, READ | TARGET, (64, 64), None);
    let words: Vec<u8> = (drawn_specials().iter().flatten().flatten())
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let values = buffer::<P>(d, 2 * 4096 * 16, 0, &words);
    let seeds = [1u32, 2].map(|seed| buffer::<P>(d, 4, 0, &seed.to_le_bytes()));
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let pass = |load, store, resolves: bool| {
        let pass = p.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(&multisampled)),
            resolve_texture: resolves.then_some(Some(&target)),
            load_action: Some(load),
            store_action: Some(store),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, 0, &attachment)
            .unwrap();
        pass
    };
    let first = pass(CLEAR, store, store != STORE);
    let e = cb.render_encoder(OnFault::Die, &first).unwrap().unwrap();
    e.encode(
        OnFault::Die,
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&state),
        },
    )
    .unwrap();
    e.encode(
        OnFault::Die,
        RenderCmd::SetFragmentBuffer {
            buffer: Some(&values),
            offset: 0,
            index: 1,
        },
    )
    .unwrap();
    for (seed, offset) in seeds.iter().zip([0.0, 0.5]) {
        for cmd in [
            RenderCmd::SetViewport {
                viewport: MtlViewport {
                    origin_x: offset,
                    origin_y: offset,
                    width: 64.0 - offset,
                    height: 64.0 - offset,
                    znear: 0.0,
                    zfar: 1.0,
                },
            },
            RenderCmd::SetFragmentBuffer {
                buffer: Some(seed),
                offset: 0,
                index: 0,
            },
            RenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
            },
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
    }
    e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    if store == STORE {
        let e = cb
            .render_encoder(OnFault::Die, &pass(LOAD, RESOLVE, true))
            .unwrap()
            .unwrap();
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    }
    let length = 64 * 64 * bytes;
    let out = buffer::<P>(d, length, 0, &[]);
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    blit.encode(
        OnFault::Die,
        BlitCmd::CopyTextureToBuffer {
            source_texture: &target,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(64, 64, 1),
            destination_buffer: &out,
            destination_offset: 0,
            destination_bytes_per_row: 64 * bytes,
            destination_bytes_per_image: length,
        },
    )
    .unwrap();
    blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    let n = usize::try_from(length).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the blit completed.
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// The two tables [`drawn_resolved`] draws, 4096 `float4`s each, a pixel's at `y * 64 + x`:
/// each lane one of a quiet NaN, a negative one, one of a payload, either infinity, a value
/// past the largest half, and two plain values, in another order in each lane and table, so
/// that over the first row, whose samples the resolve averages two values of, each lane
/// pairs each value with each.
fn drawn_specials() -> [Vec<[f32; 4]>; 2] {
    let specials = [
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7fc0_1234),
        f32::INFINITY,
        f32::NEG_INFINITY,
        131_008.0,
        0.5,
        -0.25,
    ];
    // The first table's lane `c` of pixel `i`, and the second's.
    let at = |i: usize, c: usize| [(i + c) % 8, (i + i / 8 + 2 * c) % 8];
    [0, 1].map(|table| {
        (0..4096)
            .map(|i| [0, 1, 2, 3].map(|c| specials[at(i, c)[table]]))
            .collect()
    })
}

/// A multisampled float colour attachment whose samples draws wrote as NaN (quiet, negative
/// and of a payload), as either infinity, past the largest half or as plain values, two
/// values in some pixels' samples, resolves as Metal's, byte for byte ([`drawn_resolved`]):
/// `R16Float`, `RG16Float`, `RGBA16Float` and `RG11B10Float`, which MoltenVK averages by Metal's
/// own resolve, announcing nothing (another device resolves them by the runtime's rules of
/// Metal's resolve), and `R32Float`, which the runtime resolves by its own rule, at 2 and 4
/// samples, resolved by
/// the drawing pass (`MultisampleResolve`, `StoreAndMultisampleResolve`) and by a later pass
/// that loads the samples. Each case whose bytes are not Metal's is named, with the texels off
#[test]
fn drawn_nan_and_infinite_samples_resolve_as_metals() {
    /// `MTLStoreActionMultisampleResolve` and `MTLStoreActionStoreAndMultisampleResolve`.
    const RESOLVE: u64 = 2;
    const STORE_AND_RESOLVE: u64 = 3;
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new(Box::new(NoLauncher));
    let (p, d) = (platform(), dev());
    let mut differ = Vec::new();
    for samples in [2, 4] {
        for store in [RESOLVE, STORE_AND_RESOLVE, STORE] {
            for (format, bytes) in [(25, 2), (65, 4), (115, 8), (55, 4), (92, 4)] {
                let case = (format, bytes, samples);
                let theirs = drawn_resolved(&mp, &md, case, store);
                let ours = drawn_resolved(&p, &d, case, store);
                let texel = usize::try_from(bytes).unwrap();
                let pairs = ours.chunks(texel).zip(theirs.chunks(texel));
                let off = pairs.filter(|(o, m)| o != m).count();
                if off != 0 {
                    differ.push(format!(
                        "{format} {samples}x store {store}: {off} of 4096 texels"
                    ));
                }
            }
        }
    }
    assert!(differ.is_empty(), "{differ:#?}");
}

/// How the resolve texture of [`resolved_into`] is made: a texture of its own of a storage
/// mode (`MTLResourceOptions`), which Metal may compress losslessly or not
/// (`allowGPUOptimizedContents`); one over a host IOSurface of a storage mode, presented as a
/// drawable (`setIsDrawable:` and `setFramebufferOnly:`, render target usage alone) or not; or a
/// Private shareable texture (`newSharedTextureWithDescriptor:`), or one opened from its handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Made {
    Own { options: u64, optimized: bool },
    Surface { options: u64, drawable: bool },
    Shareable { opened: bool },
}

/// A Shared texture of its own.
const SHARED: Made = Made::Own {
    options: 0,
    optimized: true,
};

/// A Private texture of its own, which Metal may compress.
const PRIVATE: Made = Made::Own {
    options: 0x20,
    optimized: true,
};

/// The resolve textures [`resolved_into`] measures plain Metal at: of its own, Shared, Managed
/// and Private (compressed where Metal may, and not), over an IOSurface, Shared and Managed,
/// each as a drawable too, and a shareable one and one opened from its handle.
const SHAPES: [Made; 10] = [
    SHARED,
    Made::Own {
        options: MANAGED,
        optimized: true,
    },
    PRIVATE,
    Made::Own {
        options: 0x20,
        optimized: false,
    },
    Made::Surface {
        options: 0,
        drawable: false,
    },
    Made::Surface {
        options: 0,
        drawable: true,
    },
    Made::Surface {
        options: MANAGED,
        drawable: false,
    },
    Made::Surface {
        options: MANAGED,
        drawable: true,
    },
    Made::Shareable { opened: false },
    Made::Shareable { opened: true },
];

/// The sizes [`resolved_into`] measures plain Metal at: 16 by 16, and 96 by 37, whose tiles
/// at its right and bottom edges it covers in part.
const SIZES: [(u64, u64); 2] = [(16, 16), (96, 37)];

/// `MTLLoadActionDontCare` and `MTLStoreActionDontCare`; `MTLLoadActionLoad`;
/// `MTLStoreActionMultisampleResolve` and `StoreAndMultisampleResolve`.
const DONT_CARE: u64 = 0;
const LOAD: u64 = 1;
const RESOLVE: u64 = 2;
const STORE_AND_RESOLVE: u64 = 3;

/// `MTLTextureUsageShaderWrite` and `PixelFormatView`.
const WRITE: u64 = 2;
const VIEW: u64 = 0x10;

/// `MTLPixelFormatDepth32Float` and `Depth32Float_Stencil8`.
const DEPTH32F: u64 = 252;
const DEPTH32F_S8: u64 = 260;

/// `MTLResourceStorageModePrivate` and `Memoryless`.
const PRIVATE_OPTIONS: u64 = 0x20;
const MEMORYLESS: u64 = 0x30;

/// A depth attachment of a pass [`resolved_into`] runs: a 4-sample texture of a format (of
/// `Depth32Float_Stencil8`, the stencil attachment too), of `MTLResourceOptions`, cleared to 1
/// or not loaded by its load action and stored by its store action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Depth {
    format: u64,
    options: u64,
    load: u64,
    store: u64,
}

/// Another colour attachment beside the resolving one in the pass [`resolved_into`] runs: a
/// 4-sample texture of a format at an index (the resolving attachment is at 1 where this is at
/// 0, else at 0), cleared, and resolved by the resolving store action into a Shared texture of
/// its format where `resolves`, else stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Beside {
    index: u32,
    format: u64,
    resolves: bool,
}

/// What the resolving pass of [`resolved_into`] draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Calls {
    /// A triangle over the area [`Resolve::load`] names.
    Triangle,
    /// No draw call, nor any other command.
    Nothing,
    /// No draw call, after the pipeline state, viewport and fragment buffer a draw takes.
    StateOnly,
    /// A draw call of no vertices.
    NoVertices,
    /// A draw call of two vertices of a triangle list, no primitive.
    TwoVertices,
    /// A draw call of no instances.
    NoInstances,
    /// An indirect draw call whose arguments name no vertices.
    IndirectNoVertices,
}

/// A resolve [`resolved_into`] runs.
#[derive(Clone, Copy, Debug)]
struct Resolve {
    /// The 4-sample attachment's format.
    from: u64,
    /// The resolve texture's format, and its bytes a texel.
    into: (u64, u64),
    /// How the resolve texture is made.
    made: Made,
    /// The width and height of the attachment and the resolve texture.
    size: (u64, u64),
    /// The resolving pass's load action, the pass after one that clears the samples to another
    /// colour and stores them: `Clear`, and its triangle ([`Calls::Triangle`]) draws over the
    /// top left 8 by 16 texels, `Load`, and it draws there too, or `DontCare`, and it draws over
    /// every texel.
    load: u64,
    /// The resolving store action.
    store: u64,
    /// Whether the resolving pass sets its store action again before it ends
    /// (`setColorStoreAction:atIndex:`).
    again: bool,
    /// The usage of the attachment's texture and of the resolve texture's (a drawable's is
    /// `RenderTarget` alone, whatever this says).
    usage: (u64, u64),
    /// Whether the attachment's texture is made with `allowGPUOptimizedContents`.
    optimized: bool,
    /// The pass's depth attachment, if any.
    depth: Option<Depth>,
    /// Another colour attachment beside it, if any.
    beside: Option<Beside>,
    /// Whether the resolving pass has a rasterization rate map, of the attachment's size and
    /// one sample of 1 on each axis.
    rate_map: bool,
    /// What the resolving pass draws.
    calls: Calls,
}

impl Resolve {
    /// The resolve of a `from` attachment into `into` made as `made`, 16 by 16, by `store` in a
    /// pass of `load`: the attachment of usage `RenderTarget`, compressed where Metal may, the
    /// resolve texture of `ShaderRead` and `RenderTarget`, in a pass of no other attachment and
    /// no rate map.
    fn of(from: u64, into: (u64, u64), made: Made, (load, store): (u64, u64)) -> Resolve {
        Resolve {
            from,
            into,
            made,
            size: (16, 16),
            load,
            store,
            again: false,
            usage: (TARGET, READ | TARGET),
            optimized: true,
            depth: None,
            beside: None,
            rate_map: false,
            calls: Calls::Triangle,
        }
    }

    /// The index of the resolving attachment.
    fn index(self) -> u32 {
        match self.beside {
            Some(Beside { index: 0, .. }) => 1,
            _ => 0,
        }
    }
}

/// What [`resolved_into`] reads: the resolve texture's bytes; the samples after the resolving
/// pass, resolved into a texture of their own format by a later pass that loads them; and the
/// resolve texture's bytes of the attachment beside it that resolves, if any.
#[derive(Debug, PartialEq, Eq)]
struct Resolved {
    resolved: Vec<u8>,
    samples: Vec<u8>,
    beside: Option<Vec<u8>>,
}

/// The states [`resolved_into`] draws with on a device, each made once ([`sweep_state`]), by
/// the colour attachments' indices and formats and the depth attachment's format.
struct States<'a, P: Platform> {
    d: &'a P::Device,
    made: std::cell::RefCell<StateMap<P>>,
}

/// The states of [`States`], by what they were made for.
type StateMap<P> =
    std::collections::HashMap<(Vec<(u32, u64)>, Option<u64>), <P as Platform>::RenderPipeline>;

impl<'a, P: Platform> States<'a, P> {
    fn new(d: &'a P::Device) -> States<'a, P> {
        States {
            d,
            made: std::cell::RefCell::new(StateMap::<P>::new()),
        }
    }

    /// The state that draws into the colour attachments of `c`'s resolving pass.
    fn of(&self, c: Resolve) -> P::RenderPipeline {
        let mut colours = vec![(c.index(), c.from)];
        colours.extend(c.beside.map(|b| (b.index, b.format)));
        colours.sort_unstable();
        let key = (colours, c.depth.map(|z| z.format));
        let mut made = self.made.borrow_mut();
        let state = made
            .entry(key.clone())
            .or_insert_with(|| sweep_state::<P>(self.d, &key.0, key.1));
        state.clone_ref()
    }
}

/// The state [`resolved_into`] draws with: `sweep_vertex`, and `sweep_fragment` into a 4-sample
/// colour attachment 0 of the one format of `colours`, else `sweep_four_fragment` into each of
/// `colours` (index and format); `depth` the depth attachment's format, a stencil one's the
/// stencil attachment's too.
fn sweep_state<P: Platform>(
    d: &P::Device,
    colours: &[(u32, u64)],
    depth: Option<u64>,
) -> P::RenderPipeline {
    use reims_vgpu_core::platform::resource::Library as _;
    use reims_vgpu_core::rt::Hold;

    use crate::gpu::serialized::tests::{le, rounded_render_of};

    let library = d.new_library(ROUNDING).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    let fragment = match colours {
        [(0, _)] => "sweep_fragment",
        _ => "sweep_four_fragment",
    };
    let functions = [function("sweep_vertex"), function(fragment)];
    let mut resolve = |_: Hold<'_>, r: u64| {
        let at = usize::try_from(r - 1).unwrap();
        Ok(Some(functions[at].clone_ref()))
    };
    let elements: Vec<[(u8, u32); 2]> = colours
        .iter()
        .map(|&(index, format)| [(0, index), (1, u32::try_from(format).unwrap())])
        .collect();
    let elements: Vec<&[(u8, u32)]> = elements.iter().map(|e| &e[..]).collect();
    let mut entries = vec!["02 04 02000000".to_owned(), "04 04 04000000".to_owned()];
    if let Some(format) = depth {
        let format = le(u32::try_from(format).unwrap());
        entries.push(format!("09 04 {format}"));
        if depth == Some(DEPTH32F_S8) {
            entries.push(format!("0a 04 {format}"));
        }
    }
    let entries: Vec<&str> = entries.iter().map(String::as_str).collect();
    let blob = rounded_render_of(&entries, &elements, 0);
    let desc = d
        .deserialize_render_pipeline_descriptor(Hold::NONE, &blob, &mut resolve)
        .unwrap()
        .unwrap();
    d.new_render_pipeline_state_from_descriptor(Some(&desc), 0)
        .unwrap()
        .unwrap()
        .unwrap()
}

/// `cmd` in a blit encoder of its own on `cb`.
fn blit_one<P: Platform>(cb: &P::CommandBuffer, cmd: BlitCmd<'_, P>) {
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    blit.encode(OnFault::Die, cmd).unwrap();
    blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
}

/// The copy of a `width` × `height` texture of `bytes` a texel into `out`, rows tight.
fn copy_out<'a, P: Platform>(
    texture: &'a P::Texture,
    (width, height, bytes): (u64, u64, u64),
    out: &'a P::Buffer,
) -> BlitCmd<'a, P> {
    BlitCmd::CopyTextureToBuffer {
        source_texture: texture,
        source_slice: 0,
        source_level: 0,
        source_origin: MtlOrigin::new(0, 0, 0),
        source_size: MtlSize::new(width, height, 1),
        destination_buffer: out,
        destination_offset: 0,
        destination_bytes_per_row: width * bytes,
        destination_bytes_per_image: width * height * bytes,
    }
}

/// A rate map of a `width` by `height` screen, with one layer of one sample of 1 on each axis.
fn even_rate_map<P: Platform>(p: &P, d: &P::Device, (width, height): (u64, u64)) -> P::RateMap {
    use reims_vgpu_core::platform::rate_map::{
        RateMapDescriptor as _, RateMapLayerDescriptor as _,
    };
    use reims_vgpu_core::platform::values::RateMapDesc;

    let m = p.rate_map_descriptor(&RateMapDesc {
        screen_size: MtlSize::new(width, height, 1),
        ..RateMapDesc::default()
    });
    let layer = p
        .new_rate_map_layer(OnFault::Die, MtlSize::new(1, 1, 0))
        .unwrap()
        .unwrap();
    for at in [
        layer.horizontal_sample_storage(),
        layer.vertical_sample_storage(),
    ] {
        // SAFETY: the layer's storage holds one sample on each axis.
        unsafe { *at.as_ptr::<f32>() = 1.0 };
    }
    m.set_layer(OnFault::Die, &layer, 0).unwrap();
    d.new_rate_map(&m).unwrap().unwrap()
}

/// What a resolve texture made as `c.made` says holds, prefilled with 0xa5 (a surface's
/// through its base address, another by a blit), after a 4-sample `c.from` colour attachment
/// of its size, cleared to (0.25, 0.5, 0.75, 1) and stored by an earlier pass, is drawn by
/// `states`' state ([`sweep_state`]) as `c.calls` says and resolved into it by `c.store` in a
/// pass of `c.load`, as [`Resolve`] says, and what its samples and the resolve texture of an
/// attachment beside it
/// hold after that pass ([`Resolved`]). The resolving pass clears to (0.75, 0.25, 0.5, 1), an
/// attachment beside to (0.125, 0.25, 0.5, 1). A surface's bytes are read through its base
/// address, as the guest reads them; another's by a blit. It runs in an autorelease pool of its
/// own: what Metal autoreleased (each command buffer and encoder, which a dev build cannot elide,
/// and through them the case's queue, textures and buffers) goes at its end, where the test
/// thread, which pops no pool, would keep every case's until the test ends.
fn resolved_into<P: Platform<Surface = MetalSurface>>(
    p: &P,
    d: &P::Device,
    states: &States<'_, P>,
    c: Resolve,
) -> Resolved {
    use reims_vgpu_core::platform::os::surface::{Surface as _, SurfaceSpec};
    use reims_vgpu_core::platform::values::{DepthAttachmentSet, PassSet, StencilAttachmentSet};

    /// `kIOSurfaceLockReadOnly`.
    const READ_ONLY: u32 = 1;
    let _pool = reims_vgpu_core::rt::Pool::here();
    let (into, bytes) = c.into;
    let (width, height) = c.size;
    let length = width * height * bytes;
    let n = usize::try_from(length).unwrap();
    let prefill = vec![0xa5u8; n];
    let multisample = |format, usage, options, optimized: bool| {
        let desc = p.texture_descriptor(&TextureDesc {
            texture_type: 4,
            pixel_format: format,
            width,
            height,
            sample_count: 4,
            usage,
            resource_options: Some(options),
            allow_gpu_optimized_contents: (!optimized).then_some(false),
            ..TextureDesc::default()
        });
        d.new_texture(OnFault::Die, &desc).unwrap().unwrap()
    };
    let samples = multisample(c.from, c.usage.0, PRIVATE_OPTIONS, c.optimized);
    let (options, usage, optimized, drawable) = match c.made {
        Made::Own { options, optimized } => (options, c.usage.1, optimized, false),
        Made::Surface {
            options,
            drawable: false,
        } => (options, c.usage.1, true, false),
        Made::Surface {
            options,
            drawable: true,
        } => (options, TARGET, true, true),
        Made::Shareable { .. } => (PRIVATE_OPTIONS, c.usage.1, true, false),
    };
    let desc = p.texture_descriptor(&TextureDesc {
        pixel_format: into,
        width,
        height,
        usage,
        resource_options: Some(options),
        allow_gpu_optimized_contents: (!optimized).then_some(false),
        framebuffer_only: drawable.then_some(true),
        is_drawable: drawable.then_some(true),
        ..TextureDesc::default()
    });
    let surface = matches!(c.made, Made::Surface { .. }).then(|| {
        let ostype = if bytes == 8 { *b"RGhA" } else { *b"w30r" };
        let s = p
            .surfaces()
            .create(&SurfaceSpec {
                pixel_format: u32::from_be_bytes(ostype),
                alloc_size: length,
                width,
                height,
                bytes_per_element: bytes,
                bytes_per_row: width * bytes,
                element_width: 1,
                element_height: 1,
                ..SurfaceSpec::default()
            })
            .unwrap();
        assert!(s.lock(0).is_success());
        let base = s.base_address().as_ptr::<u8>();
        // SAFETY: the surface's first `n` bytes, locked, which no texture uses yet.
        unsafe { std::ptr::copy_nonoverlapping(prefill.as_ptr(), base, n) };
        s.cpu_wrote();
        assert!(s.unlock(0).is_success());
        s
    });
    let shareable = matches!(c.made, Made::Shareable { .. })
        .then(|| d.new_shared_texture(&desc).unwrap().unwrap());
    let target = match (&surface, &shareable) {
        (Some(s), _) => d
            .new_texture_on_surface(&desc, Some(s), 0)
            .unwrap()
            .unwrap(),
        (None, Some(t)) if c.made == (Made::Shareable { opened: true }) => {
            let handle = t.new_shared_texture_handle().unwrap().unwrap();
            d.open_shared_texture(Some(&handle), 0).unwrap().unwrap()
        }
        (None, Some(t)) => t.clone_ref(),
        (None, None) => d.new_texture(OnFault::Die, &desc).unwrap().unwrap(),
    };
    let beside = c.beside.map(|b| {
        let other = multisample(b.format, TARGET, PRIVATE_OPTIONS, true);
        let resolve = b.resolves.then(|| {
            let texels = Some((
                &prefill[..usize::try_from(width * height * 4).unwrap()],
                width * 4,
            ));
            texture(p, d, b.format, READ | TARGET, c.size, texels)
        });
        (b, other, resolve)
    });
    let depth = c
        .depth
        .map(|z| (z, multisample(z.format, TARGET, z.options, true)));
    let map = c.rate_map.then(|| even_rate_map(p, d, c.size));
    let kinds = buffer::<P>(d, 4, 0, &0u32.to_le_bytes());
    let source = buffer::<P>(d, length, 0, &prefill);
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    if surface.is_none() {
        blit_one::<P>(
            &cb,
            BlitCmd::CopyBufferToTexture {
                source_buffer: &source,
                source_offset: 0,
                source_bytes_per_row: width * bytes,
                source_bytes_per_image: length,
                source_size: MtlSize::new(width, height, 1),
                destination_texture: &target,
                destination_slice: 0,
                destination_level: 0,
                destination_origin: MtlOrigin::new(0, 0, 0),
            },
        );
    }
    let colour = |red, green, blue| ClearColor {
        red,
        green,
        blue,
        alpha: 1.0,
    };
    // The samples, cleared and stored, by a pass of their own.
    let earlier = p.render_pass_descriptor();
    let attachment = ColorAttachmentSet {
        texture: Some(Some(&samples)),
        load_action: Some(CLEAR),
        store_action: Some(STORE),
        clear_color: Some(colour(0.25, 0.5, 0.75)),
        ..ColorAttachmentSet::default()
    };
    earlier
        .set_color_attachment(OnFault::Die, 0, &attachment)
        .unwrap();
    let e = cb.render_encoder(OnFault::Die, &earlier).unwrap().unwrap();
    e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    // The resolving pass.
    let index = c.index();
    let resolving = p.render_pass_descriptor();
    let attachment = ColorAttachmentSet {
        texture: Some(Some(&samples)),
        resolve_texture: Some(Some(&target)),
        load_action: Some(c.load),
        store_action: Some(c.store),
        clear_color: Some(colour(0.75, 0.25, 0.5)),
        ..ColorAttachmentSet::default()
    };
    resolving
        .set_color_attachment(OnFault::Die, u64::from(index), &attachment)
        .unwrap();
    if let Some((b, other, resolve)) = &beside {
        let attachment = ColorAttachmentSet {
            texture: Some(Some(other)),
            resolve_texture: resolve.as_ref().map(Some),
            load_action: Some(CLEAR),
            store_action: Some(if b.resolves { c.store } else { STORE }),
            clear_color: Some(colour(0.125, 0.25, 0.5)),
            ..ColorAttachmentSet::default()
        };
        resolving
            .set_color_attachment(OnFault::Die, u64::from(b.index), &attachment)
            .unwrap();
    }
    if let Some((z, texture)) = &depth {
        resolving.set_depth_attachment(&DepthAttachmentSet {
            texture: Some(Some(texture)),
            load_action: Some(z.load),
            store_action: Some(z.store),
            clear_depth: Some(1.0),
            ..DepthAttachmentSet::default()
        });
        if z.format == DEPTH32F_S8 {
            resolving.set_stencil_attachment(&StencilAttachmentSet {
                texture: Some(Some(texture)),
                load_action: Some(z.load),
                store_action: Some(z.store),
                clear_stencil: Some(0),
                ..StencilAttachmentSet::default()
            });
        }
    }
    if let Some(map) = &map {
        resolving.set_pass(&PassSet {
            rasterization_rate_map: Some(Some(map)),
            ..Default::default()
        });
    }
    let e = cb
        .render_encoder(OnFault::Die, &resolving)
        .unwrap()
        .unwrap();
    let state = states.of(c);
    let drawn = match c.load {
        DONT_CARE => (width, height),
        _ => (8, 16),
    };
    let again = c.again.then_some(RenderCmd::SetColorStoreAction {
        store_action: c.store,
        color_attachment_index: u64::from(index),
    });
    // `MTLDrawPrimitivesIndirectArguments` of no vertices and one instance.
    let arguments = buffer::<P>(d, 16, 0, &[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let call = match c.calls {
        Calls::Nothing | Calls::StateOnly => None,
        Calls::Triangle | Calls::NoVertices | Calls::TwoVertices => {
            Some(RenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: match c.calls {
                    Calls::NoVertices => 0,
                    Calls::TwoVertices => 2,
                    _ => 3,
                },
            })
        }
        Calls::NoInstances => Some(RenderCmd::DrawPrimitivesInstances {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 0,
        }),
        Calls::IndirectNoVertices => Some(RenderCmd::DrawPrimitivesIndirect {
            primitive_type: 3,
            indirect_buffer: &arguments,
            indirect_buffer_offset: 0,
        }),
    };
    let draw = [
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&state),
        },
        RenderCmd::SetViewport {
            viewport: MtlViewport {
                origin_x: 0.0,
                origin_y: 0.0,
                width: f64::from(u32::try_from(drawn.0).unwrap()),
                height: f64::from(u32::try_from(drawn.1).unwrap()),
                znear: 0.0,
                zfar: 1.0,
            },
        },
        RenderCmd::SetFragmentBuffer {
            buffer: Some(&kinds),
            offset: 0,
            index: 0,
        },
    ];
    let set = call.is_some() || c.calls == Calls::StateOnly;
    let draw = (set.then_some(draw).into_iter().flatten()).chain(call);
    for cmd in draw.chain(again).chain([RenderCmd::EndEncoding]) {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    // The samples, resolved into a texture of their format by a later pass that loads them.
    let check = texture(p, d, c.from, READ | TARGET, c.size, None);
    let later = p.render_pass_descriptor();
    let attachment = ColorAttachmentSet {
        texture: Some(Some(&samples)),
        resolve_texture: Some(Some(&check)),
        load_action: Some(LOAD),
        store_action: Some(RESOLVE),
        ..ColorAttachmentSet::default()
    };
    later
        .set_color_attachment(OnFault::Die, 0, &attachment)
        .unwrap();
    let e = cb.render_encoder(OnFault::Die, &later).unwrap().unwrap();
    e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    let texels = width * height * 4;
    let checked = buffer::<P>(d, texels, 0, &[]);
    blit_one::<P>(&cb, copy_out::<P>(&check, (width, height, 4), &checked));
    let besides = beside.as_ref().and_then(|(_, _, resolve)| {
        let resolve = resolve.as_ref()?;
        let out = buffer::<P>(d, texels, 0, &[]);
        blit_one::<P>(&cb, copy_out::<P>(resolve, (width, height, 4), &out));
        Some(out)
    });
    let out = buffer::<P>(d, length, 0, &[]);
    if surface.is_none() {
        blit_one::<P>(&cb, copy_out::<P>(&target, (width, height, bytes), &out));
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{c:?}: {:?}", cb.error());
    let read = |b: &P::Buffer, n: usize| {
        // SAFETY: a Shared buffer's first `n` bytes, read after the command buffer completed.
        unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>(), n) }.to_vec()
    };
    let resolved = match &surface {
        Some(s) => {
            assert!(s.lock(READ_ONLY).is_success());
            // SAFETY: the surface's first `n` bytes, locked, after the command buffer
            // completed.
            let bytes = unsafe { std::slice::from_raw_parts(s.base_address().as_ptr::<u8>(), n) };
            let bytes = bytes.to_vec();
            assert!(s.unlock(READ_ONLY).is_success());
            bytes
        }
        None => read(&out, n),
    };
    let texels = usize::try_from(texels).unwrap();
    Resolved {
        resolved,
        samples: read(&checked, texels),
        beside: besides.map(|b| read(&b, texels)),
    }
}

/// The death of `f`, which must abort at `sel`, and its line.
fn died_at<T>(sel: &str, f: impl FnOnce() -> T) -> String {
    use reims_vgpu_core::raise::death::DeathKind;
    *crate::answer::LAST_LINE.lock() = None;
    let Err(site) = catch_death(f) else {
        panic!("no death at {sel}");
    };
    let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
    assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
    line
}

/// A 16 by 16 4-sample 2D multisample texture of `format`, of `MTLResourceOptions` `options`.
fn multisampled<P: Platform>(p: &P, d: &P::Device, format: u64, options: u64) -> P::Texture {
    let desc = p.texture_descriptor(&TextureDesc {
        texture_type: 4,
        pixel_format: format,
        width: 16,
        height: 16,
        sample_count: 4,
        usage: TARGET,
        resource_options: Some(options),
        ..TextureDesc::default()
    });
    d.new_texture(OnFault::Die, &desc).unwrap().unwrap()
}

/// The pass of `samples` as colour attachment 0, cleared and stored by `store` into `resolve`
/// at its level and slice `at`.
fn resolve_pass<P: Platform>(
    p: &P,
    (samples, store): (&P::Texture, u64),
    resolve: &P::Texture,
    (level, slice): (u64, u64),
) -> P::RenderPassDescriptor {
    let pass = p.render_pass_descriptor();
    let attachment = ColorAttachmentSet {
        texture: Some(Some(samples)),
        resolve_texture: Some(Some(resolve)),
        resolve_level: Some(level),
        resolve_slice: Some(slice),
        load_action: Some(CLEAR),
        store_action: Some(store),
        ..ColorAttachmentSet::default()
    };
    pass.set_color_attachment(OnFault::Die, 0, &attachment)
        .unwrap();
    pass
}

/// The encoder of `pass`, open on the command buffer it returns.
fn encoder_on<P: Platform>(
    d: &P::Device,
    pass: &P::RenderPassDescriptor,
) -> (P::CommandBuffer, P::RenderEncoder) {
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, pass).unwrap().unwrap();
    (cb, e)
}

/// A 4-sample `from` colour attachment, Private and of its own, cleared and stored by `store`
/// into `resolve` at its level and slice `at`, and the encoder of its pass, open on the command
/// buffer it returns.
fn resolving_pass<P: Platform>(
    p: &P,
    d: &P::Device,
    (from, store): (u64, u64),
    resolve: &P::Texture,
    at: (u64, u64),
) -> (P::CommandBuffer, P::RenderEncoder) {
    let samples = multisampled(p, d, from, PRIVATE_OPTIONS);
    encoder_on::<P>(d, &resolve_pass(p, (&samples, store), resolve, at))
}

/// A 4-sample `BGRA8Unorm` or `BGRA8Unorm_sRGB` colour attachment, cleared, drawn and resolved
/// into a texture of another format ([`resolved_into`]) by `MultisampleResolve` and
/// `StoreAndMultisampleResolve`, by its pass and by a later one that loads its samples: into
/// `RGBA16Float`, `BGR10_XR` and `BGR10_XR_sRGB`, which plain Metal leaves untouched, of every
/// shape of [`SHAPES`] at each of [`SIZES`], the pass is made without the resolve and the
/// texture keeps its bytes, as Metal's, and the samples hold what Metal's hold:
/// `StoreAndMultisampleResolve` stores them as `Store` does, and `MultisampleResolve` leaves
/// them as they were before the pass, as `DontCare` does ([`untouched_resolves`]: macOS
/// 26.6.2's Maps); the resolving action set again before the pass ends encodes
/// nothing, as Metal's. So do they, at 16 by 16 into a Shared texture of its own (and a Private
/// one, for the resolve texture's usages), after `Clear`, `Load` and `DontCare`: of the resolve
/// texture's usage `RenderTarget`, or with `ShaderWrite`, `PixelFormatView` or both beside
/// `ShaderRead`; of the attachment's usage `ShaderRead` or `PixelFormatView` beside
/// `RenderTarget`, or without `allowGPUOptimizedContents`; beside a depth attachment
/// (`Depth32Float` memoryless and not stored, or Private and stored, and
/// `Depth32Float_Stencil8` as both depth and stencil attachment, memoryless, or Private, cleared
/// or not loaded, and not stored); beside another colour attachment resolving into its format
/// (`RGBA8Unorm` at 0 or 1, `BGRA8Unorm` at 2) or storing (`R32Float` at 2), whose resolve
/// texture holds what Metal's holds; and in a pass of a rate map of one sample of 1 on each axis.
/// So do they in a pass that draws nothing, after each load action, by both actions, the
/// attachment compressed where Metal may and not: with no draw call, into each format, 16 by 16
/// and 96 by 37 into a Shared texture of its own and 16 by 16 into a Private one; with only the
/// state a draw takes, or a draw call of no vertices, of no instances, of two vertices of a
/// triangle list or indirect of no vertices, into `RGBA16Float` at 16 by 16, but a `DontCare`
/// load stored by `StoreAndMultisampleResolve` under the last two, where the runtime's
/// `DontCareLoadTiles` keeps any pass that loads by `DontCare` and stores from Metal's samples.
/// And so do they at 1920 by 1080 into each format, drawn, after `Clear`, `Load` and
/// `DontCare`, and at 1023 by 771 into `RGBA16Float`, cleared with no draw call, and drawn after
/// `DontCare`.
/// Into `RGB10A2Unorm`, `BGR10A2Unorm`, `BGRA10_XR` and `BGRA10_XR_sRGB`, where Metal writes into
/// it, the pass dies at its creation naming both formats and the shape Metal was measured at
/// ([`refused_resolve_format`]); into the other attachment format of the pair, the twin, the
/// resolve writes, its bytes Metal's, in every shape of [`SHAPES`], which shows each read sees a
/// resolve's writes
#[test]
fn resolves_into_other_formats_are_metals_or_die_named() {
    let (p, d) = (platform(), dev());
    let metal = MetalPlatform::system_default_device()
        .map(|md| (MetalPlatform::new(Box::new(NoLauncher)), md));
    let untouched = [(115, 8), (554, 4), (555, 4)];
    let written = [(90, 4), (94, 4), (552, 8), (553, 8)];
    let kept = |bytes: &[u8]| bytes.iter().all(|&b| b == 0xa5);
    // Whether every texel of the samples is the same and not 0: the earlier pass's clear.
    let untouched_samples = |bytes: &[u8]| {
        let mut texels = bytes.chunks(4);
        let first = texels.next().unwrap_or_default();
        first != [0; 4] && texels.all(|t| t == first)
    };
    // The first bytes of each, and whether the resolve texture kept its 0xa5s.
    let shown = |r: &Resolved| {
        let head = |b: &[u8]| b.iter().take(8).copied().collect::<Vec<u8>>();
        format!(
            "kept {}, {:02x?}, samples {:02x?}, beside {:02x?}",
            kept(&r.resolved),
            head(&r.resolved),
            head(&r.samples),
            r.beside.as_deref().map(head)
        )
    };
    let ours = States::<VulkanPlatform>::new(&d);
    let theirs = metal
        .as_ref()
        .map(|(_, md)| States::<MetalPlatform>::new(md));
    let mut differ = Vec::new();
    let compared = std::cell::Cell::new(0);
    let metals = |c: Resolve| {
        let ((mp, md), states) = (metal.as_ref()?, theirs.as_ref()?);
        compared.set(compared.get() + 1);
        Some(resolved_into(mp, md, states, c))
    };
    let mut both = |c: Resolve, keeps: bool| {
        let resolved = resolved_into(&p, &d, &ours, c);
        if let Some(m) = metals(c).filter(|m| *m != resolved) {
            let (o, m) = (shown(&resolved), shown(&m));
            differ.push(format!("{c:?}: ours {o}; Metal {m}"));
        }
        if kept(&resolved.resolved) != keeps {
            differ.push(format!("{c:?}: ours {}", shown(&resolved)));
        }
        // Where the texture keeps its bytes, `MultisampleResolve` leaves the samples as the
        // earlier pass stored them, cleared alike, and `StoreAndMultisampleResolve` stores what
        // the pass drew: some texels cleared and some drawn, or, drawn over every texel, 0. A
        // pass that draws nothing leaves them alike either way, the earlier clear or its own.
        let drawn = c.calls == Calls::Triangle;
        if keeps && untouched_samples(&resolved.samples) != (c.store == RESOLVE || !drawn) {
            differ.push(format!("{c:?}: ours {}", shown(&resolved)));
        }
    };
    let actions = [CLEAR, LOAD]
        .into_iter()
        .flat_map(|load| [RESOLVE, STORE_AND_RESOLVE].map(|store| (load, store)));
    for from in [80, 81] {
        let twin = if from == 80 { 81 } else { 80 };
        for (load, store) in actions.clone() {
            let shapes = SHAPES.into_iter().flat_map(|m| untouched.map(|i| (m, i)));
            for ((made, into), size) in shapes.flat_map(|c| SIZES.map(|s| (c, s))) {
                let agains: &[bool] = if made == SHARED {
                    &[false, true]
                } else {
                    &[false]
                };
                for &again in agains {
                    let c = Resolve {
                        size,
                        again,
                        ..Resolve::of(from, into, made, (load, store))
                    };
                    both(c, true);
                }
            }
            if store == RESOLVE {
                for made in SHAPES {
                    both(Resolve::of(from, (twin, 4), made, (load, store)), false);
                }
            }
        }
    }
    // Each shape beside the base one, at 16 by 16, after each load action.
    let depth = |format, options, load, store| {
        Some(Depth {
            format,
            options,
            load,
            store,
        })
    };
    let beside = |index, format, resolves| {
        Some(Beside {
            index,
            format,
            resolves,
        })
    };
    let resolve_usages = [
        TARGET,
        READ | WRITE | TARGET,
        READ | TARGET | VIEW,
        READ | WRITE | TARGET | VIEW,
    ];
    let attachment_usages = [READ | TARGET, TARGET | VIEW];
    let depths = [
        depth(DEPTH32F, MEMORYLESS, CLEAR, DONT_CARE),
        depth(DEPTH32F, PRIVATE_OPTIONS, CLEAR, STORE),
        depth(DEPTH32F_S8, MEMORYLESS, CLEAR, DONT_CARE),
        depth(DEPTH32F_S8, PRIVATE_OPTIONS, CLEAR, DONT_CARE),
        depth(DEPTH32F_S8, PRIVATE_OPTIONS, DONT_CARE, DONT_CARE),
    ];
    let besides = [
        beside(1, RGBA8, true),
        beside(0, RGBA8, true),
        beside(2, BGRA8, true),
        beside(2, R32F, false),
    ];
    for from in [80, 81] {
        for (load, store) in [CLEAR, LOAD, DONT_CARE]
            .into_iter()
            .flat_map(|load| [RESOLVE, STORE_AND_RESOLVE].map(|store| (load, store)))
        {
            for into in untouched {
                let base = Resolve::of(from, into, SHARED, (load, store));
                let mut cases = vec![base];
                for made in [SHARED, PRIVATE] {
                    for usage in resolve_usages {
                        cases.push(Resolve {
                            made,
                            usage: (TARGET, usage),
                            ..base
                        });
                    }
                }
                for usage in attachment_usages {
                    cases.push(Resolve {
                        usage: (usage, READ | TARGET),
                        ..base
                    });
                }
                cases.push(Resolve {
                    optimized: false,
                    ..base
                });
                cases.extend(depths.map(|depth| Resolve { depth, ..base }));
                cases.extend(besides.map(|beside| Resolve { beside, ..base }));
                cases.push(Resolve {
                    rate_map: true,
                    ..base
                });
                for c in cases {
                    both(c, true);
                }
            }
        }
    }
    // Passes that draw nothing, after each load action, by both actions, the attachment
    // compressed where Metal may and not: with no draw call into each format, 16 by 16 and 96 by
    // 37 into a Shared texture of its own and 16 by 16 into a Private one; and into
    // `RGBA16Float`, 16 by 16, with only the state a draw takes or a draw call that draws
    // nothing.
    let nothing = [
        Calls::StateOnly,
        Calls::NoVertices,
        Calls::NoInstances,
        Calls::TwoVertices,
        Calls::IndirectNoVertices,
    ];
    for from in [80, 81] {
        for (load, store) in [CLEAR, LOAD, DONT_CARE]
            .into_iter()
            .flat_map(|load| [RESOLVE, STORE_AND_RESOLVE].map(|store| (load, store)))
        {
            for optimized in [true, false] {
                for into in untouched {
                    for (size, made) in
                        [((16, 16), SHARED), ((96, 37), SHARED), ((16, 16), PRIVATE)]
                    {
                        both(
                            Resolve {
                                size,
                                optimized,
                                calls: Calls::Nothing,
                                ..Resolve::of(from, into, made, (load, store))
                            },
                            true,
                        );
                    }
                }
                for calls in nothing {
                    // A `DontCare` load stored by `StoreAndMultisampleResolve` under a draw call
                    // that draws nothing on the GPU keeps the samples on Metal, where the
                    // runtime's `Store` clears their tiles to zero: its `DontCareLoadTiles`,
                    // announced and relayed for any pass that loads by `DontCare` and stores,
                    // the drop or not (`unmeasured_shape`).
                    let on_the_gpu =
                        matches!(calls, Calls::TwoVertices | Calls::IndirectNoVertices);
                    if (load, store) == (DONT_CARE, STORE_AND_RESOLVE) && on_the_gpu {
                        continue;
                    }
                    both(
                        Resolve {
                            optimized,
                            calls,
                            ..Resolve::of(from, (115, 8), SHARED, (load, store))
                        },
                        true,
                    );
                }
            }
        }
    }
    // Larger sizes, into a Shared texture of its own: 1920 by 1080 into each format, drawn,
    // after `Clear` by `MultisampleResolve`, `Load` by `StoreAndMultisampleResolve` and
    // `DontCare` by `MultisampleResolve`; 1023 by 771 into `RGBA16Float`, after `Clear` by
    // `MultisampleResolve` with no draw call, and after `DontCare` by
    // `StoreAndMultisampleResolve`, drawn.
    for from in [80, 81] {
        for into in untouched {
            for actions in [
                (CLEAR, RESOLVE),
                (LOAD, STORE_AND_RESOLVE),
                (DONT_CARE, RESOLVE),
            ] {
                let c = Resolve::of(from, into, SHARED, actions);
                both(
                    Resolve {
                        size: (1920, 1080),
                        ..c
                    },
                    true,
                );
            }
        }
        for (actions, calls) in [
            ((CLEAR, RESOLVE), Calls::Nothing),
            ((DONT_CARE, STORE_AND_RESOLVE), Calls::Triangle),
        ] {
            let c = Resolve::of(from, (115, 8), SHARED, actions);
            both(
                Resolve {
                    size: (1023, 771),
                    calls,
                    ..c
                },
                true,
            );
        }
    }
    for (from, (load, store), into) in [80, 81]
        .into_iter()
        .flat_map(|from| actions.clone().map(move |a| (from, a)))
        .flat_map(|(from, a)| written.map(|into| (from, a, into)))
    {
        let c = Resolve::of(from, into, SHARED, (load, store));
        if let Some(m) = metals(c).filter(|m| kept(&m.resolved)) {
            differ.push(format!("{c:?}: Metal {}", shown(&m)));
        }
        let line = died_at("renderCommandEncoderWithDescriptor:", || {
            resolved_into(&p, &d, &ours, c)
        });
        let format = PixelFormat::try_from(u32::try_from(into.0).unwrap()).unwrap();
        let named = format!("into a texture of {}", format.name());
        assert!(line.contains(&named), "{c:?}: {line}");
        let measured = "plain Metal writes into such a texture, measured from a Private 2D \
                        multisample texture of its own into a Shared single-level 2D texture of \
                        its own, both 16 by 16";
        assert!(line.contains(measured), "{c:?}: {line}");
    }
    assert!(metal.is_none() || compared.get() > 0);
    assert!(differ.is_empty(), "{differ:#?}");
}

/// A resolve into a texture of another format the runtime does not resolve into dies at the
/// encoder's creation, named with what plain Metal was measured to do there: a pair Metal is
/// not measured at (a 4-sample `RGBA8Unorm` or 2-sample `BGRA8Unorm` attachment into
/// `RGBA16Float`, a 4-sample `RGBA16Float` one into `RGBA32Float`); and a pair Metal leaves
/// untouched ([`measured_pair`]), a 4-sample `BGRA8Unorm_sRGB` attachment into `RGBA16Float`,
/// in a shape not measured ([`unmeasured_shape`]), whatever the runtime refuses of it, named
/// with what lies outside: a resolve texture of another size, level 1 of a texture whose level
/// 1 is the attachment's size, a 2D array's slice 1, a memoryless texture, a Private texture
/// over an IOSurface, a texture over a plane of a surface with planes, a view, a texture over a
/// buffer, and one in a heap; an attachment memoryless, in a heap, or with store action
/// options; a depth attachment that resolves; another colour attachment that resolves into a
/// texture of another format the runtime does not resolve into; and a rate map the runtime does
/// not take as the identity, a mutable one not yet reset and one whose rates vary. So does one
/// by a store action left `Unknown`, which the runtime refuses as the pass is made
#[test]
fn untouched_resolves_in_shapes_not_measured_die_named() {
    use reims_vgpu_core::platform::os::surface::{SurfacePlaneSpec, SurfaceSpec};
    use reims_vgpu_core::platform::rate_map::{
        RateMapDescriptor as _, RateMapLayerDescriptor as _,
    };
    use reims_vgpu_core::platform::resource::Heap as _;
    use reims_vgpu_core::platform::values::{
        DepthAttachmentSet, HeapDesc, OptionSet, PassSet, RateMapDesc,
    };

    /// `MTLStoreActionUnknown`; `MTLPixelFormatBGRA8Unorm_sRGB` and `RGBA32Float`.
    const UNKNOWN: u64 = 4;
    const BGRA8_SRGB: u64 = 81;
    const RGBA32F: u64 = 125;
    const SEL: &str = "renderCommandEncoderWithDescriptor:";
    let (p, d) = (platform(), dev());
    let desc = |texture_type, (width, height), levels, slices, options| {
        p.texture_descriptor(&TextureDesc {
            texture_type,
            pixel_format: RGBA16F,
            width,
            height,
            mipmap_level_count: levels,
            array_length: slices,
            usage: READ | TARGET,
            resource_options: Some(options),
            ..TextureDesc::default()
        })
    };
    let made = |texture_type, size, levels, slices, options| {
        let desc = desc(texture_type, size, levels, slices, options);
        d.new_texture(OnFault::Die, &desc).unwrap().unwrap()
    };
    let flat = desc(2, (16, 16), 1, 1, 0);
    let surface = |planes: Option<Vec<SurfacePlaneSpec>>| {
        let size = 16 * 16 * 8;
        p.surfaces()
            .create(&SurfaceSpec {
                planes,
                pixel_format: u32::from_be_bytes(*b"RGhA"),
                alloc_size: 2 * size,
                width: 16,
                height: 16,
                bytes_per_element: 8,
                bytes_per_row: 16 * 8,
                element_width: 1,
                element_height: 1,
                ..SurfaceSpec::default()
            })
            .unwrap()
    };
    let plane = |base: u64| SurfacePlaneSpec {
        base,
        offset: base,
        width: 16,
        height: 16,
        size: 16 * 16 * 8,
        bytes_per_row: 16 * 8,
        bytes_per_element: 8,
        element_width: 1,
        element_height: 1,
        ..SurfacePlaneSpec::default()
    };
    let flat_surface = surface(None);
    let planar = surface(Some(vec![plane(0), plane(16 * 16 * 8)]));
    let heap = d
        .new_heap(&HeapDesc {
            size: 1 << 20,
            heap_type: 1,
            options: vec![OptionSet::StorageMode(2)],
            address_ranges: None,
            pinned_gpu_address: None,
        })
        .unwrap()
        .heap
        .unwrap();
    let bytes = d.new_buffer_of_length(OnFault::Die, 16 * 16 * 8, 0);
    let bytes = bytes.unwrap().unwrap();
    let own = made(2, (16, 16), 1, 1, 0);
    let resolves: [(&str, VulkanTexture, (u64, u64)); 10] = [
        (
            "the resolve texture is 32 by 32, the attachment 16 by 16",
            made(2, (32, 32), 1, 1, 0),
            (0, 0),
        ),
        (
            "the resolve texture is of Type2D, 1 samples and 2 levels",
            made(2, (32, 32), 2, 1, 0),
            (1, 0),
        ),
        (
            "the resolve texture is of Type2DArray, 1 samples and 1 levels",
            made(3, (16, 16), 1, 2, 0),
            (0, 1),
        ),
        (
            "the resolve texture is a texture of its own of storage mode Memoryless",
            made(2, (16, 16), 1, 1, MEMORYLESS),
            (0, 0),
        ),
        (
            "the resolve texture is a texture over an IOSurface of storage mode Private",
            d.new_texture_on_surface(
                &desc(2, (16, 16), 1, 1, PRIVATE_OPTIONS),
                Some(&flat_surface),
                0,
            )
            .unwrap()
            .unwrap(),
            (0, 0),
        ),
        (
            "the resolve texture is a texture over a plane of an IOSurface with planes",
            d.new_texture_on_surface(&flat, Some(&planar), 0)
                .unwrap()
                .unwrap(),
            (0, 0),
        ),
        (
            "the resolve texture is a view of another texture",
            own.new_texture_view(RGBA16F).unwrap().unwrap(),
            (0, 0),
        ),
        (
            "the resolve texture is a texture over a buffer",
            bytes
                .new_texture(OnFault::Die, &flat, 0, 16 * 8)
                .unwrap()
                .unwrap(),
            (0, 0),
        ),
        (
            "the resolve texture is a texture in a heap",
            heap.new_texture(&desc(2, (16, 16), 1, 1, PRIVATE_OPTIONS), 0)
                .unwrap()
                .unwrap(),
            (0, 0),
        ),
        (
            "its store action left Unknown, whose late resolve plain Metal is not measured at",
            own.clone_ref(),
            (0, 0),
        ),
    ];
    for (i, (what, resolve, at)) in resolves.iter().enumerate() {
        let store = if i == resolves.len() - 1 {
            UNKNOWN
        } else {
            RESOLVE
        };
        let line = died_at(SEL, || {
            resolving_pass(&p, &d, (BGRA8_SRGB, store), resolve, *at)
        });
        assert!(
            line.contains("into a texture of MTLPixelFormatRGBA16Float"),
            "{what}: {line}"
        );
        assert!(line.contains(what), "{what}: {line}");
    }
    let heaped = heap
        .new_texture(
            &p.texture_descriptor(&TextureDesc {
                texture_type: 4,
                pixel_format: BGRA8_SRGB,
                width: 16,
                height: 16,
                sample_count: 4,
                usage: TARGET,
                resource_options: Some(PRIVATE_OPTIONS),
                ..TextureDesc::default()
            }),
            1 << 16,
        )
        .unwrap()
        .unwrap();
    let attachments = [
        (
            "the attachment is Memoryless",
            multisampled(&p, &d, BGRA8_SRGB, MEMORYLESS),
        ),
        ("the attachment is a texture in a heap", heaped),
    ];
    for (what, samples) in &attachments {
        let pass = resolve_pass(&p, (samples, RESOLVE), &own, (0, 0));
        let line = died_at(SEL, || encoder_on::<VulkanPlatform>(&d, &pass));
        assert!(line.contains(what), "{what}: {line}");
    }
    let samples = multisampled(&p, &d, BGRA8_SRGB, PRIVATE_OPTIONS);
    // Store action options, a depth attachment that resolves, and another colour attachment
    // that resolves into a texture of another format.
    let options = resolve_pass(&p, (&samples, RESOLVE), &own, (0, 0));
    let attachment = ColorAttachmentSet {
        store_action_options: Some(1),
        ..ColorAttachmentSet::default()
    };
    options
        .set_color_attachment(OnFault::Die, 0, &attachment)
        .unwrap();
    let depth = resolve_pass(&p, (&samples, RESOLVE), &own, (0, 0));
    let depth_samples = multisampled(&p, &d, DEPTH32F, PRIVATE_OPTIONS);
    let depth_resolve = texture(&p, &d, DEPTH32F, TARGET, (16, 16), None);
    depth.set_depth_attachment(&DepthAttachmentSet {
        texture: Some(Some(&depth_samples)),
        resolve_texture: Some(Some(&depth_resolve)),
        load_action: Some(CLEAR),
        store_action: Some(RESOLVE),
        ..DepthAttachmentSet::default()
    });
    let two = resolve_pass(&p, (&samples, RESOLVE), &own, (0, 0));
    let other = multisampled(&p, &d, BGRA8_SRGB, PRIVATE_OPTIONS);
    let into = made(2, (16, 16), 1, 1, 0);
    let attachment = ColorAttachmentSet {
        texture: Some(Some(&other)),
        resolve_texture: Some(Some(&into)),
        load_action: Some(CLEAR),
        store_action: Some(RESOLVE),
        ..ColorAttachmentSet::default()
    };
    two.set_color_attachment(OnFault::Die, 1, &attachment)
        .unwrap();
    // Rate maps the runtime does not take as the identity: a mutable one not yet reset, and
    // one whose rates vary, over a screen of 256 by 256 (whose rates vary between its tiles).
    let mapped = |mutability, horizontal: &[f32]| {
        let m = p.rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(256, 256, 1),
            mutability,
            ..RateMapDesc::default()
        });
        let count = u64::try_from(horizontal.len()).unwrap();
        let layer = p.new_rate_map_layer(OnFault::Die, MtlSize::new(count, 1, 0));
        let layer = layer.unwrap().unwrap();
        let rates = layer.horizontal_sample_storage().as_ptr::<f32>();
        for (i, &v) in horizontal.iter().enumerate() {
            // SAFETY: the layer's horizontal storage holds `count` samples.
            unsafe { *rates.add(i) = v };
        }
        // SAFETY: its vertical storage holds one.
        unsafe { *layer.vertical_sample_storage().as_ptr::<f32>() = 1.0 };
        m.set_layer(OnFault::Die, &layer, 0).unwrap();
        let map = d.new_rate_map(&m).unwrap().unwrap();
        let pass = resolve_pass(&p, (&samples, RESOLVE), &own, (0, 0));
        pass.set_pass(&PassSet {
            rasterization_rate_map: Some(Some(&map)),
            ..Default::default()
        });
        pass
    };
    let passes = [
        (
            "the attachment's store action options are StoreActionOptions(1)",
            options,
        ),
        ("the pass's depth or stencil attachment resolves", depth),
        (
            "colour attachment 1 resolves into a texture of another format too",
            two,
        ),
        (
            "the pass has a mutable rasterization rate map not yet reset",
            mapped(Some(1), &[1.0]),
        ),
        (
            "the pass has a rasterization rate map whose rates vary",
            mapped(None, &[1.0, 0.25]),
        ),
    ];
    for (what, pass) in &passes {
        let line = died_at(SEL, || encoder_on::<VulkanPlatform>(&d, pass));
        assert!(line.contains(what), "{what}: {line}");
    }
    let pairs = [(RGBA8, 4), (BGRA8, 2), (RGBA16F, 4)];
    for (from, count) in pairs {
        let desc = p.texture_descriptor(&TextureDesc {
            texture_type: 4,
            pixel_format: from,
            width: 16,
            height: 16,
            sample_count: count,
            usage: TARGET,
            resource_options: Some(PRIVATE_OPTIONS),
            ..TextureDesc::default()
        });
        let samples = d.new_texture(OnFault::Die, &desc).unwrap().unwrap();
        let into = match from {
            RGBA16F => texture(&p, &d, RGBA32F, READ | TARGET, (16, 16), None),
            _ => own.clone_ref(),
        };
        let pass = resolve_pass(&p, (&samples, RESOLVE), &into, (0, 0));
        let line = died_at(SEL, || encoder_on::<VulkanPlatform>(&d, &pass));
        assert!(
            line.contains("a pair plain Metal is not measured at"),
            "{from} at {count}: {line}"
        );
    }
}

/// A pass made without a resolve plain Metal leaves untouched ([`untouched_resolves`]): a
/// late resolve texture of that attachment dies, as nothing measured it; a late store action
/// is compared with the descriptor's own, so another dies and the resolving one again encodes
/// nothing, the pass completing. A late resolve texture of its depth or stencil attachment
/// whose store action is left `Unknown` dies too, as the pass's creation dies where the
/// descriptor names one ([`unmeasured_shape`]): `Depth32Float`'s, and `Depth32Float_Stencil8`'s
/// stencil's; a late nil one, or one of an attachment stored by `Store`, goes on. (One stored by
/// a resolving action with no resolve texture dies at the pass's creation, which the runtime
/// refuses.)
#[test]
fn late_setters_of_a_dropped_resolve_are_the_descriptors() {
    use reims_vgpu_core::platform::values::{DepthAttachmentSet, StencilAttachmentSet};

    /// `MTLPixelFormatBGRA8Unorm_sRGB`; `MTLStoreActionUnknown`.
    const BGRA8_SRGB: u64 = 81;
    const UNKNOWN: u64 = 4;
    /// A late depth resolve texture `t`.
    fn depth_late(t: Option<&VulkanTexture>) -> RenderCmd<'_, VulkanPlatform> {
        RenderCmd::SetDepthResolveTexture {
            texture: t,
            slice: 0,
            depth_plane: 0,
            level: 0,
            y_invert: false,
        }
    }
    /// A late stencil resolve texture `t`.
    fn stencil_late(t: Option<&VulkanTexture>) -> RenderCmd<'_, VulkanPlatform> {
        RenderCmd::SetStencilResolveTexture {
            texture: t,
            slice: 0,
            depth_plane: 0,
            level: 0,
            y_invert: false,
        }
    }
    let (p, d) = (platform(), dev());
    let target = texture(&p, &d, RGBA16F, READ | TARGET, (16, 16), None);
    let open = || resolving_pass(&p, &d, (BGRA8_SRGB, RESOLVE), &target, (0, 0));
    let (_cb, e) = open();
    let late = RenderCmd::SetColorResolveTexture {
        texture: Some(&target),
        slice: 0,
        depth_plane: 0,
        level: 0,
        y_invert: false,
        index: 0,
    };
    let sel = "setColorResolveTexture:slice:depthPlane:level:yInvert:atIndex:";
    let line = died_at(sel, || e.encode(OnFault::Die, late));
    assert!(
        line.contains("resolve by MultisampleResolve into MTLPixelFormatRGBA16Float"),
        "{line}"
    );
    let (_cb, e) = open();
    let store = RenderCmd::SetColorStoreAction {
        store_action: STORE,
        color_attachment_index: 0,
    };
    let line = died_at("setColorStoreAction:atIndex:", || {
        e.encode(OnFault::Die, store)
    });
    assert!(
        line.contains("a late store action Store over the known MultisampleResolve"),
        "{line}"
    );
    let (cb, e) = open();
    let again = RenderCmd::SetColorStoreAction {
        store_action: RESOLVE,
        color_attachment_index: 0,
    };
    for cmd in [again, RenderCmd::EndEncoding] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    // A depth or stencil attachment beside, of no resolve texture, given one late.
    let samples = multisampled(&p, &d, BGRA8_SRGB, PRIVATE_OPTIONS);
    let beside = |format, (depth, stencil): (u64, Option<u64>)| {
        let pass = resolve_pass(&p, (&samples, RESOLVE), &target, (0, 0));
        let z = multisampled(&p, &d, format, PRIVATE_OPTIONS);
        pass.set_depth_attachment(&DepthAttachmentSet {
            texture: Some(Some(&z)),
            load_action: Some(CLEAR),
            store_action: Some(depth),
            ..DepthAttachmentSet::default()
        });
        if let Some(store) = stencil {
            pass.set_stencil_attachment(&StencilAttachmentSet {
                texture: Some(Some(&z)),
                load_action: Some(CLEAR),
                store_action: Some(store),
                ..StencilAttachmentSet::default()
            });
        }
        let resolve = texture(&p, &d, format, TARGET, (16, 16), None);
        (encoder_on::<VulkanPlatform>(&d, &pass), resolve)
    };
    let ((_cb, e), resolve) = beside(DEPTH32F, (UNKNOWN, None));
    let late = depth_late(Some(&resolve));
    let sel = late.selector();
    let line = died_at(sel, || e.encode(OnFault::Die, late));
    let named = "a late depth resolve texture in a pass made without the resolve of colour \
                 attachment 0 by MultisampleResolve into MTLPixelFormatRGBA16Float";
    assert!(line.contains(named), "{line}");
    assert!(
        line.contains("depth or stencil attachment resolves"),
        "{line}"
    );
    let ((_cb, e), resolve) = beside(DEPTH32F_S8, (DONT_CARE, Some(UNKNOWN)));
    let late = stencil_late(Some(&resolve));
    let sel = late.selector();
    let line = died_at(sel, || e.encode(OnFault::Die, late));
    assert!(line.contains("a late stencil resolve texture"), "{line}");
    // A nil one, and one of an attachment stored, which resolves nowhere.
    let ((cb, e), resolve) = beside(DEPTH32F_S8, (STORE, Some(UNKNOWN)));
    for cmd in [
        depth_late(Some(&resolve)),
        stencil_late(None),
        RenderCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
}

/// A pass made without a colour resolve plain Metal leaves untouched ([`untouched_resolves`]),
/// with a depth or stencil attachment whose store action the descriptor left `Unknown` and no
/// resolve texture: once a late store action that does not resolve (`Store`) decides it, a late
/// resolve texture of it no longer dies as resolving beside the drop, as nothing resolves there.
/// It goes to the runtime, as in a pass whose colour resolve is the runtime's, which refuses a
/// resolve texture under an action that does not resolve (`ResolveStoreAction`): the setter
/// dies there, naming that refusal, in both passes alike. The attachment left undecided still
/// dies as resolving beside the drop: `Depth32Float_Stencil8`'s stencil, after its depth's
/// late `Store`
#[test]
fn late_resolves_beside_a_dropped_one_follow_late_store_actions() {
    use reims_vgpu_core::platform::values::{DepthAttachmentSet, StencilAttachmentSet};

    /// `MTLPixelFormatBGRA8Unorm_sRGB`; `MTLStoreActionUnknown`.
    const BGRA8_SRGB: u64 = 81;
    const UNKNOWN: u64 = 4;
    /// A late resolve texture `t` of the depth attachment, or else the stencil one.
    fn late(depth: bool, t: Option<&VulkanTexture>) -> RenderCmd<'_, VulkanPlatform> {
        match depth {
            true => RenderCmd::SetDepthResolveTexture {
                texture: t,
                slice: 0,
                depth_plane: 0,
                level: 0,
                y_invert: false,
            },
            false => RenderCmd::SetStencilResolveTexture {
                texture: t,
                slice: 0,
                depth_plane: 0,
                level: 0,
                y_invert: false,
            },
        }
    }
    let (p, d) = (platform(), dev());
    let samples = multisampled(&p, &d, BGRA8_SRGB, PRIVATE_OPTIONS);
    // Plain Metal leaves the `RGBA16Float` one untouched, a drop; the runtime resolves into the
    // attachment's own format.
    let dropped = texture(&p, &d, RGBA16F, READ | TARGET, (16, 16), None);
    let runtimes = texture(&p, &d, BGRA8_SRGB, READ | TARGET, (16, 16), None);
    let beside = |target: &VulkanTexture, format, stencil: bool| {
        let pass = resolve_pass(&p, (&samples, RESOLVE), target, (0, 0));
        let z = multisampled(&p, &d, format, PRIVATE_OPTIONS);
        pass.set_depth_attachment(&DepthAttachmentSet {
            texture: Some(Some(&z)),
            load_action: Some(CLEAR),
            store_action: Some(UNKNOWN),
            ..DepthAttachmentSet::default()
        });
        if stencil {
            pass.set_stencil_attachment(&StencilAttachmentSet {
                texture: Some(Some(&z)),
                load_action: Some(CLEAR),
                store_action: Some(UNKNOWN),
                ..StencilAttachmentSet::default()
            });
        }
        let resolve = texture(&p, &d, format, TARGET, (16, 16), None);
        (encoder_on::<VulkanPlatform>(&d, &pass), resolve)
    };
    let stored = || RenderCmd::SetDepthStoreAction {
        store_action: STORE,
    };
    let beside_drop = "depth or stencil attachment resolves";
    for (name, target) in [("dropped", &dropped), ("the runtime's", &runtimes)] {
        let ((_cb, e), resolve) = beside(target, DEPTH32F, false);
        e.encode(OnFault::Die, stored()).unwrap();
        let cmd = late(true, Some(&resolve));
        let sel = cmd.selector();
        let line = died_at(sel, || e.encode(OnFault::Die, cmd));
        assert!(line.contains("ResolveStoreAction"), "{name}: {line}");
        assert!(!line.contains(beside_drop), "{name}: {line}");
    }
    let ((_cb, e), resolve) = beside(&dropped, DEPTH32F_S8, true);
    e.encode(OnFault::Die, stored()).unwrap();
    let cmd = late(false, Some(&resolve));
    let sel = cmd.selector();
    let line = died_at(sel, || e.encode(OnFault::Die, cmd));
    assert!(line.contains("a late stencil resolve texture"), "{line}");
    assert!(line.contains(beside_drop), "{line}");
}

/// A resolve into the attachment's sRGB twin, or across the 8-bit four-channel family's
/// channel orders, is the runtime's to resolve or refuse ([`runtime_resolves`]): the pass of a
/// 4-sample `R8Unorm`, `R8Unorm_sRGB`, `RG8Unorm` or `RG8Unorm_sRGB` attachment resolving into
/// its twin is made and completes. Other pairs are not
#[test]
fn resolves_into_twins_are_the_runtimes() {
    use PixelFormat as P;
    let runtimes = [
        (P::R8Unorm, P::R8UnormSRGB),
        (P::RG8Unorm, P::RG8UnormSRGB),
        (P::RGBA8Unorm, P::RGBA8UnormSRGB),
        (P::BGRA8Unorm, P::BGRA8UnormSRGB),
        (P::BGR10XR, P::BGR10XRSRGB),
        (P::BGRA10XR, P::BGRA10XRSRGB),
        (P::RGBA8Unorm, P::BGRA8Unorm),
        (P::RGBA8UnormSRGB, P::BGRA8Unorm),
        (P::RGBA16Float, P::RGBA16Float),
    ];
    for (a, b) in runtimes {
        assert!(
            runtime_resolves(a, b) && runtime_resolves(b, a),
            "{a:?} {b:?}"
        );
    }
    let others = [
        (P::BGRA8UnormSRGB, P::RGBA16Float),
        (P::BGRA8Unorm, P::BGR10XR),
        (P::BGRA8Unorm, P::RGB10A2Unorm),
        (P::R8Unorm, P::RG8Unorm),
        (P::R8Unorm, P::RGBA8UnormSRGB),
        (P::RGBA16Float, P::RG16Float),
    ];
    for (a, b) in others {
        assert!(!runtime_resolves(a, b), "{a:?} {b:?}");
    }
    let (p, d) = (platform(), dev());
    for (from, into) in [(10, 11), (11, 10), (30, 31), (31, 30)] {
        let target = texture(&p, &d, into, READ | TARGET, (16, 16), None);
        let (cb, e) = resolving_pass(&p, &d, (from, RESOLVE), &target, (0, 0));
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{from} into {into}: {:?}", cb.error());
    }
}

/// Measures what [`mode_2_multisampled_half_float_attachments_resolve_as_metals`] checks at
/// 2 and 4 samples at 8, where Metal's device supports the count (the M3 Max's does not):
/// this platform's states of that count die as they are made, which the probe sets aside
/// ([`crate::gpu::serialized::RESOLVE_PROBED`]). A count it passes at is measured. One line,
/// what Metal's device supports and whether it passed.
#[test]
#[ignore = "a measurement, with the GPU free"]
fn metal_resolves_mode_2_at_other_sample_counts_here() {
    use reims_vgpu_core::platform::caps::SAMPLE_COUNTS;

    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let caps = md.caps();
    for samples in [8] {
        let at = SAMPLE_COUNTS.iter().position(|&n| n == samples).unwrap();
        if caps.sample_counts[at] != Ok(true) {
            println!("{samples} samples: Metal's device does not support them");
            continue;
        }
        let probed = u32::try_from(samples).unwrap();
        crate::gpu::serialized::RESOLVE_PROBED.store(probed, std::sync::atomic::Ordering::Relaxed);
        resolves_as_metals(samples);
        println!("{samples} samples: resolved as Metal's");
    }
}

/// Checks that a render state of `textureWriteRoundingMode` 2 whose half-float colour
/// attachment of `samples` samples stores what the fragment function writes as written
/// resolves as Metal's, byte for byte, and so does one of mode 0 of it, over the sweeps
/// through unorm values, past 1 by float steps and past the largest half: `R16Float`,
/// `RG16Float` and `RGBA16Float`, fully covered and a quarter pixel in, resolved by the
/// drawing pass, and a quarter pixel in by a later pass that loads the samples and draws
/// nothing ([`resolved`]). Metal's are measured first, and so are its facts the runtime's
/// rounding relies on: at mode 2, fully covered, each sample rounds as a write of one sample
/// does (the [`swept`] draw), and the later pass resolves as the drawing one, so the resolve
/// does not take the state's mode; the mode changes what it resolves.
fn resolves_as_metals(samples: u64) {
    use crate::gpu::serialized::NEAREST_EVEN;

    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new(Box::new(NoLauncher));
    let modes = [0, NEAREST_EVEN];
    // Fully covered, a quarter pixel in, and a quarter pixel in resolved apart.
    let hows = [(0.0, false), (0.25, false), (0.25, true)];
    let mut cases = Vec::new();
    for kind in [0, 1, 3] {
        for format in [(25, 2), (65, 4), (115, 8)] {
            let theirs = modes
                .map(|mode| hows.map(|how| resolved(&mp, &md, (mode, samples), format, kind, how)));
            let one = swept(&mp, &md, (false, NEAREST_EVEN), format, kind, PLAIN);
            let [m0, m2] = &theirs;
            let case = format!("{samples} samples {format:?} sweep {kind}");
            assert!(m2[0] == one, "Metal {case}: rounded as one sample");
            assert!(m2[1] == m2[2], "Metal {case}: resolved apart");
            assert!(m2[1] != m0[1], "Metal {case}: as at mode 0");
            cases.push((kind, format, theirs));
        }
    }
    let (p, d) = (platform(), dev());
    for (kind, format, theirs) in cases {
        for (&mode, theirs) in modes.iter().zip(&theirs) {
            for (&how, theirs) in hows.iter().zip(theirs) {
                let ours = resolved(&p, &d, (mode, samples), format, kind, how);
                assert!(
                    ours == *theirs,
                    "{samples} samples {format:?} sweep {kind} mode {mode} {how:?}"
                );
            }
        }
    }
}

/// A guest's state of the fixture `vertex` and `vertex_ids`' `plain_fragment` for a BGRA8
/// target, of the triangle input primitive topology class, which Metal asks of a vertex
/// function that writes a layer or viewport index, decoded from a blob as core decodes one.
fn routed_state<P: Platform>(d: &P::Device, vertex: &str) -> P::RenderPipeline {
    use reims_vgpu_core::rt::Hold;

    use crate::gpu::serialized::tests::hex;

    let functions = [
        fixture::<P>(d, vertex),
        fixture::<P>(d, "vertex_ids:plain_fragment"),
    ];
    let mut resolve = |_: Hold<'_>, r: u64| {
        let at = usize::try_from(r - 1).unwrap();
        Ok(Some(functions[at].clone_ref()))
    };
    let blob = hex(
        "04 08 04 1c000000 01 04 01000000 02 04 02000000 0b 04 03000000 000000 \
         01000000 08000000 02 00 04 00000000 01 04 50000000",
    );
    let desc = d
        .deserialize_render_pipeline_descriptor(Hold::NONE, &blob, &mut resolve)
        .unwrap()
        .unwrap();
    d.new_render_pipeline_state_from_descriptor(Some(&desc), 0)
        .unwrap()
        .unwrap()
        .unwrap()
}

/// The pixels of both slices of a 16 by 16 BGRA8 2D array of 2, cleared to black, after
/// `draw` in a pass of both, its render target array length 2, under a [`routed_state`] of
/// `vertex` over [`flat_vertices`] in buffer 0 and `indices`, each vertex's layer or
/// viewport, in buffer 1, at `setProvokingVertexMode:` of `mode`, with two viewports, the
/// target's left and right halves.
fn routed_pass<P: Platform>(
    p: &P,
    d: &P::Device,
    vertex: &str,
    (indices, mode): (&[u32], u64),
    draw: RenderCmd<'_, P>,
) -> Vec<[u8; 4]> {
    use reims_vgpu_core::platform::types::MtlViewport;
    use reims_vgpu_core::platform::values::PassSet;

    let state = routed_state::<P>(d, vertex);
    let (pos, _) = flat_vertices::<P>(d, 16);
    let mut words: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
    words.resize(64, 0);
    let routes = buffer::<P>(d, 64, 0, &words);
    let desc = p.texture_descriptor(&TextureDesc {
        texture_type: 3,
        pixel_format: BGRA8,
        width: 16,
        height: 16,
        array_length: 2,
        usage: TARGET | READ,
        ..TextureDesc::default()
    });
    let t = d.new_texture(OnFault::Die, &desc).unwrap().unwrap();
    let pass = p.render_pass_descriptor();
    pass.set_color_attachment(
        OnFault::Die,
        0,
        &ColorAttachmentSet {
            texture: Some(Some(&t)),
            load_action: Some(CLEAR),
            store_action: Some(STORE),
            ..ColorAttachmentSet::default()
        },
    )
    .unwrap();
    pass.set_pass(&PassSet {
        render_target_array_length: Some(2),
        ..PassSet::default()
    });
    let half = |x| MtlViewport {
        origin_x: x,
        origin_y: 0.0,
        width: 8.0,
        height: 16.0,
        znear: 0.0,
        zfar: 1.0,
    };
    let viewports = [half(0.0), half(8.0)];
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    for cmd in [
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&state),
        },
        RenderCmd::SetViewports {
            viewports: &viewports,
        },
        RenderCmd::SetVertexBuffer {
            buffer: Some(&pos),
            offset: 0,
            index: 0,
        },
        RenderCmd::SetVertexBuffer {
            buffer: Some(&routes),
            offset: 0,
            index: 1,
        },
        RenderCmd::SetProvokingVertexMode { mode },
        draw,
        RenderCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    let out = buffer::<P>(d, 2 * 1024, 0, &[]);
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    for slice in 0..2 {
        blit.encode(
            OnFault::Die,
            BlitCmd::CopyTextureToBuffer {
                source_texture: &t,
                source_slice: slice,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(16, 16, 1),
                destination_buffer: &out,
                destination_offset: slice * 1024,
                destination_bytes_per_row: 64,
                destination_bytes_per_image: 1024,
            },
        )
        .unwrap();
    }
    blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    // SAFETY: a Shared buffer's bytes, read after the blit completed.
    let texels = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), 2048) };
    texels.as_chunks::<4>().0.to_vec()
}

/// Under a vertex function that writes a layer or a viewport index, a list or strip of
/// triangles at the last vertex lands where Metal's does, each primitive in its last
/// vertex's layer or viewport, and at the first in its first's; what has no rotation dies
#[test]
fn the_last_vertex_routes_layers_and_viewports_as_metals() {
    let (p, d) = (platform(), dev());
    let metal = MetalPlatform::system_default_device()
        .map(|md| (MetalPlatform::new(Box::new(NoLauncher)), md));
    fn draw<'a, P: Platform>(primitive_type: u64, count: u64) -> RenderCmd<'a, P> {
        RenderCmd::DrawPrimitives {
            primitive_type,
            vertex_start: 0,
            vertex_count: count,
        }
    }
    let routes: [&[u32]; 3] = [
        &[0, 0, 1, 1, 1, 0],
        &[0, 1, 0, 1, 1, 0],
        &[1, 0, 0, 0, 1, 1],
    ];
    for vertex in ["vertex_ids:layer_vertex", "vertex_ids:viewport_vertex"] {
        for indices in routes {
            for mode in [0, 1] {
                for (primitive_type, count) in [(3, 6), (STRIP, 6)] {
                    let shape = (indices, mode);
                    let ours = routed_pass(&p, &d, vertex, shape, draw(primitive_type, count));
                    let lit = ours.iter().filter(|px| **px != [0, 0, 0, 0xff]).count();
                    assert!(lit > 0, "{vertex} {indices:?} {mode} {primitive_type}");
                    if let Some((mp, md)) = &metal {
                        let theirs =
                            routed_pass(mp, md, vertex, shape, draw(primitive_type, count));
                        assert!(
                            ours == theirs,
                            "{vertex} {indices:?} at mode {mode}, type {primitive_type}"
                        );
                    }
                }
            }
        }
        let (e, indices) = open_drawing(&routed_state::<VulkanPlatform>(&d, vertex));
        send_all(&e, vec![RenderCmd::SetProvokingVertexMode { mode: 1 }]);
        let sel = "drawPrimitives:vertexStart:vertexCount:";
        let indexed = "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:";
        assert_eq!(dies_at(&e, three(1, None)), sel, "{vertex}");
        assert_eq!(dies_at(&e, three(6, None)), sel, "{vertex}");
        assert_eq!(dies_at(&e, three(3, Some(&indices))), indexed, "{vertex}");
    }
}

/// After `setPrimitiveRestartEnabled:index:`, enabled or not, an indexed triangle strip of
/// 16- or 32-bit indices draws what Metal's does, directly and from indirect arguments
/// alike, each compared with Metal's own: it restarts at the setter's index alone, and draws
/// the all-ones index as a vertex; an indexed list draws as without the setter. 16-bit
/// indices of an odd and an even count are drawn from offsets 0 and 2, where the last of
/// them lies alone in its 4-byte word or past one (an odd count from 0, an even one from 2)
#[test]
fn indexed_strips_restart_at_the_set_index_as_metals_do() {
    let (p, d) = (platform(), dev());
    let metal = MetalPlatform::system_default_device()
        .map(|md| (MetalPlatform::new(Box::new(NoLauncher)), md));
    let odd = u16s(&[0, 1, 2, 3, 0xffff, 4, 5, 6, 7]);
    let even = u16s(&[0, 1, 2, 3, 0xffff, 4, 5, 6, 7, 8]);
    let wide = u32s(&[0, 1, 2, 3, 4, 5, 6, 7, 8]);
    for (index_type, indices, offset) in [
        (U16, &odd, 0),
        (U16, &even, 0),
        (U16, &odd, 2),
        (U16, &even, 2),
        (U32, &wide, 0),
    ] {
        for primitive_type in [STRIP, 3] {
            let drawn = (primitive_type, index_type, &indices[..]);
            let unset = restart_draw(&p, &d, drawn, None, (false, offset));
            for (enabled, k) in [(false, 3), (true, 3), (true, 0), (false, 15)] {
                let count = indices.len() / if index_type == U16 { 2 } else { 4 };
                let case = format!(
                    "type {primitive_type} of {count} of {index_type} from {offset} at \
                     {enabled} {k}"
                );
                let ours = restart_draw(&p, &d, drawn, Some((enabled, k)), (false, offset));
                let indirect = restart_draw(&p, &d, drawn, Some((enabled, k)), (true, offset));
                assert_eq!(indirect, ours, "{case}");
                if primitive_type != STRIP {
                    assert_eq!(ours, unset, "{case}");
                } else if k == 3 {
                    assert_ne!(ours, unset, "{case}");
                }
                if let Some((mp, md)) = &metal {
                    for indirectly in [false, true] {
                        let restart = Some((enabled, k));
                        let theirs = restart_draw(mp, md, drawn, restart, (indirectly, offset));
                        assert_eq!(ours, theirs, "{case}, Metal's indirect {indirectly}");
                    }
                }
            }
        }
    }
}

/// An indirect strip of 200,000 16- or 32-bit indices that restarts at the setter's index
/// ([`long_restarted_strips`]) draws what Metal's does, within the command buffer's index pool
/// (`a_long_restarted_strip_draws_within_the_index_pool`)
#[test]
fn a_long_restarted_strip_draws_within_the_index_pool_as_metals() {
    let (p, d) = (platform(), dev());
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new(Box::new(NoLauncher));
    for (index_type, indices) in long_restarted_strips() {
        let drawn = (STRIP, index_type, &indices[..]);
        let restart = Some((true, 3));
        let ours = restart_draw(&p, &d, drawn, restart, (true, 0));
        let theirs = restart_draw(&mp, &md, drawn, restart, (true, 0));
        assert_eq!(ours, theirs, "{index_type}");
    }
}

/// Metal's private types 5 to 8 of ten vertices draw after `setPrimitiveRestartEnabled:index:`
/// what Metal's do, and a line loop, a fan and quads what they draw without it: the setter's
/// index among their vertices is a vertex still
#[test]
fn private_types_draw_after_a_restart_set_as_metals_do() {
    let (p, d) = (platform(), dev());
    let metal = MetalPlatform::system_default_device()
        .map(|md| (MetalPlatform::new(Box::new(NoLauncher)), md));
    /// The pixels of ten vertices of `primitive_type`, after a setter of `k` where one is.
    fn drawn<P: Platform>(
        p: &P,
        d: &P::Device,
        primitive_type: u64,
        k: Option<u64>,
    ) -> Vec<[u8; 4]> {
        let draw = RenderCmd::DrawPrimitives {
            primitive_type,
            vertex_start: 0,
            vertex_count: 10,
        };
        match k {
            None => flat_draw(p, d, FLAT, 0, draw),
            Some(index) => {
                let set = RenderCmd::SetPrimitiveRestartEnabled {
                    enabled: true,
                    index,
                };
                set_draw(p, d, FLAT, set, draw)
            }
        }
    }
    for primitive_type in 5..=8 {
        let unset = drawn(&p, &d, primitive_type, None);
        for k in [0, 3] {
            let ours = drawn(&p, &d, primitive_type, Some(k));
            if primitive_type < 8 {
                assert_eq!(ours, unset, "type {primitive_type} at {k}");
            }
            if let Some((mp, md)) = &metal {
                let theirs = drawn(mp, md, primitive_type, Some(k));
                assert_eq!(ours, theirs, "type {primitive_type} at {k}");
            }
        }
    }
}

/// The pixels of [`restart_draw`]'s triangle strip, drawn directly.
fn restart_strip<P: Platform>(
    p: &P,
    d: &P::Device,
    (index_type, indices): (u64, &[u8]),
    restart: Option<(bool, u64)>,
) -> Vec<[u8; 4]> {
    restart_draw(p, d, (STRIP, index_type, indices), restart, (false, 0))
}

/// The pixels of [`restart_strip`]'s strip as an execution of an indirect command buffer
/// whose one command draws it, with its state and vertices, after `restart`'s
/// `setPrimitiveRestartEnabled:index:`.
fn executed_strip(
    p: &VulkanPlatform,
    d: &crate::gpu::device::VulkanDevice,
    (index_type, indices): (u64, &[u8]),
    (enabled, index): (bool, u64),
) -> Vec<[u8; 4]> {
    use metal2vulkan::metal::PixelFormat;
    use metal2vulkan::pipeline::{
        ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource,
    };
    use reims_vgpu_core::platform::icb::{
        IcbRenderCmd, IndirectCommandBuffer as _, IndirectRenderCommand as _,
    };
    use reims_vgpu_core::platform::values::IcbDesc;

    let function = |name: &str| {
        let path = format!(
            "{}/../../tests/suite/fixtures/shaders/{name}.metallib",
            env!("CARGO_MANIFEST_DIR")
        );
        d.raw
            .new_library_with_data(&std::fs::read(path).unwrap())
            .unwrap()
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut desc = RenderPipelineDescriptor::new(StageSource::Function(function(FLAT.0)));
    desc.fragment = Some(StageSource::Function(function(FLAT.1)));
    desc.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
    desc.support_indirect_command_buffers = true;
    let state = crate::gpu::resource::render_pipeline(
        d.raw
            .new_render_pipeline_state(&desc, &rt::PipelineOptions::default())
            .unwrap(),
    );
    let (pos, colours) = flat_vertices::<VulkanPlatform>(d, 0x10000);
    let length = u64::try_from(indices.len()).unwrap();
    let index_buffer = buffer::<VulkanPlatform>(d, length, 0, indices);
    let icb = d
        .new_indirect_command_buffer(
            &IcbDesc {
                command_types: 2,
                max_vertex_buffer_bind_count: 2,
                ..IcbDesc::default()
            },
            1,
            0,
        )
        .unwrap()
        .unwrap();
    let c = icb.render_command(0).unwrap().unwrap();
    for cmd in [
        IcbRenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&state),
        },
        IcbRenderCmd::SetVertexBuffer {
            buffer: Some(&pos),
            offset: 0,
            index: 0,
        },
        IcbRenderCmd::SetVertexBuffer {
            buffer: Some(&colours),
            offset: 0,
            index: 1,
        },
        IcbRenderCmd::DrawIndexedPrimitives {
            primitive_type: STRIP,
            index_count: length / if index_type == U16 { 2 } else { 4 },
            index_type,
            index_buffer: Some(&index_buffer),
            index_buffer_offset: 0,
            instance_count: 1,
            base_vertex: 0,
            base_instance: 0,
        },
    ] {
        c.encode(cmd).unwrap();
    }
    pass_pixels(
        p,
        d,
        vec![
            RenderCmd::UseResources {
                resources: &[
                    Some(ResourceRef::Icb(&icb)),
                    Some(ResourceRef::Buffer(&pos)),
                    Some(ResourceRef::Buffer(&colours)),
                    Some(ResourceRef::Buffer(&index_buffer)),
                ],
                usage: 1,
            },
            RenderCmd::SetPrimitiveRestartEnabled { enabled, index },
            RenderCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: &icb,
                execution_range: NsRange::new(0, 1),
            },
        ],
    )
}

/// An execution of an indirect command buffer's indexed triangle strip, 16- and 32-bit,
/// draws after `setPrimitiveRestartEnabled:index:`, enabled or not, what Metal's does: what
/// the strip draws directly without the setter, restarting at the all-ones index alone and
/// drawing the setter's index as a vertex. A direct draw after the setter draws otherwise
/// ([`indexed_strips_restart_at_the_set_index_as_metals_do`])
#[test]
fn an_execution_draws_its_strips_after_a_restart_set_as_metals_does() {
    let (p, d) = (platform(), dev());
    let metal = MetalPlatform::system_default_device()
        .map(|md| (MetalPlatform::new(Box::new(NoLauncher)), md));
    let k = 3;
    for (index_type, indices) in [
        (U16, u16s(&[0, 1, 2, 3, 0xffff, 4, 5, 6, 7])),
        (U32, u32s(&[0, 1, 2, 3, 4, 5, 6, 7, 8])),
    ] {
        let strip = (index_type, &indices[..]);
        let direct = restart_strip(&p, &d, strip, None);
        assert_ne!(direct, vec![[0, 0, 0, 0xff]; 256], "{index_type}");
        for enabled in [false, true] {
            let executed = executed_strip(&p, &d, strip, (enabled, k));
            assert_eq!(executed, direct, "{index_type} {enabled}");
        }
        let restarted = restart_strip(&p, &d, strip, Some((true, k)));
        assert_ne!(restarted, direct, "{index_type}");
        if let Some((mp, md)) = &metal {
            assert_eq!(restart_strip(mp, md, strip, None), direct, "{index_type}");
            let theirs = restart_strip(mp, md, strip, Some((true, k)));
            assert_eq!(theirs, restarted, "{index_type}");
        }
    }
}

/// An execution of an empty range, at the start or the end of an indirect command buffer
/// that inherits its pipeline state or not, runs nothing and the pass completes, as Metal's
/// does, under a pipeline state made without `supportIndirectCommandBuffers` too
/// (`render_draw_icb`)
#[test]
fn empty_executions_run_nothing_under_any_pipeline_state() {
    use reims_vgpu_core::platform::values::IcbDesc;

    fn pixels<P: Platform>(p: &P, d: &P::Device, inherit: bool) -> Vec<[u8; 4]> {
        let state = fixture_state::<P>(d, FLAT.0, FLAT.1);
        let icb = d
            .new_indirect_command_buffer(
                &IcbDesc {
                    command_types: 1,
                    inherit_pipeline_state: inherit,
                    inherit_buffers: inherit,
                    ..IcbDesc::default()
                },
                1,
                0,
            )
            .unwrap()
            .unwrap();
        let execute = |location| RenderCmd::ExecuteCommandsInBuffer {
            indirect_command_buffer: &icb,
            execution_range: NsRange::new(location, 0),
        };
        pass_pixels(
            p,
            d,
            vec![
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(&state),
                },
                execute(0),
                execute(1),
            ],
        )
    }
    let (p, d) = (platform(), dev());
    let metal = MetalPlatform::system_default_device()
        .map(|md| (MetalPlatform::new(Box::new(NoLauncher)), md));
    for inherit in [true, false] {
        assert_eq!(
            pixels(&p, &d, inherit),
            vec![[0, 0, 0, 0xff]; 256],
            "{inherit}"
        );
        if let Some((mp, md)) = &metal {
            assert_eq!(
                pixels(mp, md, inherit),
                vec![[0, 0, 0, 0xff]; 256],
                "{inherit}"
            );
        }
    }
}

/// A pass whose `renderTargetArrayLength` runs past its attachment's slices, slice 1 of a 2D
/// array of 2 at a length of 2, dies at the encoder's creation, where Metal clears the
/// missing slice 2 into the memory after the texture: for a 2D array over a buffer, made as
/// core makes one, the buffer's bytes at `0x100`, slice 1's being at `0x80`
#[test]
fn a_pass_past_its_attachments_slices_dies_where_metal_writes_past_the_texture() {
    use reims_vgpu_core::platform::values::PassSet;
    use reims_vgpu_core::raise::death::DeathKind;

    /// The pass on `d`, and the ranges of the 16 KiB buffer under the texture that its
    /// commit changed from `0x55`.
    fn pass<P: Platform>(p: &P, d: &P::Device) -> (P::RenderPassDescriptor, P::Buffer, P::Texture) {
        let desc = p.texture_descriptor(&TextureDesc {
            texture_type: 3,
            pixel_format: BGRA8,
            width: 4,
            height: 4,
            array_length: 2,
            usage: TARGET | READ,
            ..TextureDesc::default()
        });
        let backing = buffer::<P>(d, 0x4000, 0, &[0x55; 0x4000]);
        let t = backing
            .new_linear_texture(&desc, 0, 16, 0)
            .unwrap()
            .unwrap();
        let pass = p.render_pass_descriptor();
        pass.set_color_attachment(
            OnFault::Die,
            0,
            &ColorAttachmentSet {
                texture: Some(Some(&t)),
                slice: Some(1),
                load_action: Some(CLEAR),
                store_action: Some(STORE),
                ..ColorAttachmentSet::default()
            },
        )
        .unwrap();
        pass.set_pass(&PassSet {
            render_target_array_length: Some(2),
            ..PassSet::default()
        });
        (pass, backing, t)
    }
    let (p, d) = (platform(), dev());
    let (descriptor, _backing, _t) = pass(&p, &d);
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let site = catch_death(|| cb.render_encoder(OnFault::Return, &descriptor)).unwrap_err();
    assert_eq!(
        (site.kind, site.file),
        (DeathKind::Abort, "renderCommandEncoderWithDescriptor:")
    );
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new(Box::new(NoLauncher));
    let (descriptor, backing, _t) = pass(&mp, &md);
    let q = md.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb
        .render_encoder(OnFault::Die, &descriptor)
        .unwrap()
        .unwrap();
    e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    // SAFETY: the Shared buffer's bytes, read after the pass completed.
    let bytes = unsafe { std::slice::from_raw_parts(backing.contents().as_ptr::<u8>(), 0x4000) };
    let changed: Vec<usize> = (0..0x4000).filter(|&i| bytes[i] != 0x55).collect();
    let expected: Vec<usize> = (0x80..0xc0).chain(0x100..0x140).collect();
    assert_eq!(changed, expected);
}

/// The resolve texture's texels after a pass of a 4-sample colour and depth attachment, each
/// over a resolve texture, whose store actions stay `Unknown` through `endEncoding`.
fn unknown_stores_over_resolves<P: Platform>(p: &P, d: &P::Device) -> Vec<u8> {
    /// `MTLTextureType2DMultisample`, `MTLPixelFormatDepth32Float`, `MTLStoreActionUnknown`.
    const MULTISAMPLE: u64 = 4;
    const DEPTH32F: u64 = 252;
    const UNKNOWN: u64 = 4;
    let samples = |format| {
        let desc = p.texture_descriptor(&TextureDesc {
            texture_type: MULTISAMPLE,
            pixel_format: format,
            width: 4,
            height: 4,
            sample_count: 4,
            usage: TARGET,
            resource_options: Some(0x20),
            ..TextureDesc::default()
        });
        d.new_texture(OnFault::Die, &desc).unwrap().unwrap()
    };
    let before: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(37) | 1).collect();
    let resolve = texture(p, d, BGRA8, TARGET | READ, (4, 4), Some((&before, 16)));
    let depth_resolve = {
        let desc = p.texture_descriptor(&TextureDesc {
            pixel_format: DEPTH32F,
            width: 4,
            height: 4,
            usage: TARGET,
            resource_options: Some(0x20),
            ..TextureDesc::default()
        });
        d.new_texture(OnFault::Die, &desc).unwrap().unwrap()
    };
    let (color, depth) = (samples(BGRA8), samples(DEPTH32F));
    let pass = p.render_pass_descriptor();
    pass.set_color_attachment(
        OnFault::Die,
        0,
        &ColorAttachmentSet {
            texture: Some(Some(&color)),
            resolve_texture: Some(Some(&resolve)),
            load_action: Some(CLEAR),
            store_action: Some(UNKNOWN),
            clear_color: Some(ClearColor {
                red: 1.0,
                green: 0.0,
                blue: 0.0,
                alpha: 1.0,
            }),
            ..ColorAttachmentSet::default()
        },
    )
    .unwrap();
    pass.set_depth_attachment(&reims_vgpu_core::platform::values::DepthAttachmentSet {
        texture: Some(Some(&depth)),
        resolve_texture: Some(Some(&depth_resolve)),
        load_action: Some(CLEAR),
        store_action: Some(UNKNOWN),
        ..Default::default()
    });
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    let out = buffer::<P>(d, 64, 0, &[]);
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    blit.encode(
        OnFault::Die,
        BlitCmd::CopyTextureToBuffer {
            source_texture: &resolve,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(4, 4, 1),
            destination_buffer: &out,
            destination_offset: 0,
            destination_bytes_per_row: 16,
            destination_bytes_per_image: 64,
        },
    )
    .unwrap();
    blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    // SAFETY: a Shared buffer's bytes, read after the blit completed.
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), 64) }.to_vec()
}

/// A colour and a depth attachment whose store actions stay `Unknown` over a resolve texture
/// end their pass, and resolve nothing, as on Metal
#[test]
fn unknown_stores_over_resolves_resolve_nothing() {
    let before: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(37) | 1).collect();
    let (p, d) = (platform(), dev());
    assert_eq!(unknown_stores_over_resolves(&p, &d), before);
    if let Some(md) = MetalPlatform::system_default_device() {
        let mp = MetalPlatform::new(Box::new(NoLauncher));
        assert_eq!(unknown_stores_over_resolves(&mp, &md), before);
    }
}

/// A load action, store action or store action options the runtime has no value of is
/// taken, as Metal's setters take it, and read back as set; the encoder's creation dies at
/// it, where Metal asserts ("-[MTLRenderPassDescriptorInternal validate:width:height:]:2401:
/// failed assertion `Render Pass Descriptor Validation\nloadAction is not a valid
/// MTLLoadAction.'", and "storeAction is not a valid MTLStoreAction.", "storeActionOptions
/// contains unrecognized bits (0x8)" for 9 each, measured one case a process with
/// `MTL_DEBUG_LAYER` unset; under it, the store action options' assertion is
/// `_MTLDebugValidateRenderPassDescriptorAndTrackAttachments:370`'s, at the same send),
/// whether or not the attachment has a texture; so does a resolve filter the runtime has no
/// value of. A value the runtime has, set after it, makes the encoder
#[test]
fn render_pass_values_metal_refuses_die_at_the_encoder() {
    let p = platform();
    let d = dev();
    let mp = MetalPlatform::new(Box::new(NoLauncher));
    let md = MetalPlatform::system_default_device();
    // Each case's load action, store action and store action options.
    fn set_of<P: Platform>(
        (load_action, store_action, store_action_options): (Option<u64>, Option<u64>, Option<u64>),
    ) -> ColorAttachmentSet<'static, P> {
        ColorAttachmentSet {
            load_action,
            store_action,
            store_action_options,
            ..ColorAttachmentSet::default()
        }
    }
    let cases = [
        (Some(9), None, None),
        (None, Some(9), None),
        (None, None, Some(9)),
    ];
    for case in cases {
        let set = &set_of::<VulkanPlatform>(case);
        let (_dst, pass) = pass_of(&p, &d, (4, 4));
        pass.set_color_attachment(OnFault::Die, 0, set).unwrap();
        let state = pass.color_attachment_state(0).unwrap();
        let read = (state.load_action, state.store_action);
        let expected = (
            set.load_action.unwrap_or(CLEAR),
            set.store_action.unwrap_or(STORE),
        );
        assert_eq!(read, expected);
        if let Some(md) = &md {
            let (_dst, theirs) = pass_of(&mp, md, (4, 4));
            theirs
                .set_color_attachment(OnFault::Die, 0, &set_of::<MetalPlatform>(case))
                .unwrap();
            let state = theirs.color_attachment_state(0).unwrap();
            assert_eq!((state.load_action, state.store_action), read, "Metal");
        }
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(|| cb.render_encoder(OnFault::Die, &pass).is_ok()).unwrap_err();
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert_eq!(
            (site.kind, site.file),
            (
                reims_vgpu_core::raise::death::DeathKind::Abort,
                "renderCommandEncoderWithDescriptor:"
            ),
            "{line}"
        );
        assert!(line.contains("Metal aborts at"), "{line}");
        // A value the runtime has, set after it.
        let (_dst, pass) = pass_of(&p, &d, (4, 4));
        pass.set_color_attachment(OnFault::Die, 0, set).unwrap();
        let again = ColorAttachmentSet {
            load_action: set.load_action.map(|_| CLEAR),
            store_action: set.store_action.map(|_| STORE),
            store_action_options: set.store_action_options.map(|_| 0),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, 0, &again).unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    }
    // The depth and stencil attachments', with no texture: a load action of 9, a store
    // action of 9, a depth resolve filter of 9 and a stencil resolve filter of 2 or 9
    // ("resolveFilter is not a valid MTLMultisampleDepthResolveFilter.", and
    // "MTLMultisampleStencilResolveFilter."). A depth resolve filter of 2 makes the encoder.
    use reims_vgpu_core::platform::values::{DepthAttachmentSet, StencilAttachmentSet};
    let q = d.new_command_queue().unwrap();
    let encoder = |pass: &VulkanRenderPassDescriptor| {
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        *crate::answer::LAST_LINE.lock() = None;
        let made = catch_death(|| {
            let e = cb.render_encoder(OnFault::Die, pass).unwrap();
            e.map(|e| e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap())
                .is_some()
        });
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        (made, line)
    };
    let depth = |load_action, depth_resolve_filter| DepthAttachmentSet {
        load_action,
        depth_resolve_filter,
        ..DepthAttachmentSet::default()
    };
    let stencil = |store_action, stencil_resolve_filter| StencilAttachmentSet {
        store_action,
        stencil_resolve_filter,
        ..StencilAttachmentSet::default()
    };
    for (depth, stencil) in [
        (depth(Some(9), None), stencil(None, None)),
        (depth(None, Some(9)), stencil(None, None)),
        (depth(None, None), stencil(Some(9), None)),
        (depth(None, None), stencil(None, Some(2))),
        (depth(None, None), stencil(None, Some(9))),
    ] {
        let (_dst, pass) = pass_of(&p, &d, (4, 4));
        pass.set_depth_attachment(&depth);
        pass.set_stencil_attachment(&stencil);
        let (made, line) = encoder(&pass);
        let site = made.unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (
                reims_vgpu_core::raise::death::DeathKind::Abort,
                "renderCommandEncoderWithDescriptor:"
            ),
            "{line}"
        );
        assert!(line.contains("Metal aborts at"), "{line}");
    }
    let (_dst, pass) = pass_of(&p, &d, (4, 4));
    pass.set_depth_attachment(&depth(None, Some(2)));
    let (made, line) = encoder(&pass);
    assert!(matches!(made, Ok(true)), "{line}");
}

/// The display's functions cannot write an integer target: no pipeline, as on Metal
#[test]
fn an_integer_target_has_no_display_pipeline() {
    let p = platform();
    let d = dev();
    let cache = DisplayPipelineCache::<VulkanPlatform>::new(d.clone_ref()).unwrap();
    let key = PipelineKey {
        format: R32UINT,
        gamma: false,
    };
    assert!(cache.pipeline_for(key).is_none());
    drop(p);
    if let Some(md) = MetalPlatform::system_default_device() {
        let cache = DisplayPipelineCache::<MetalPlatform>::new(md).unwrap();
        assert!(cache.pipeline_for(key).is_none());
    }
}

/// The display's pipeline answers Metal's values for a plain state
#[test]
fn a_display_pipeline_answers_metals_values() {
    fn values<P: Platform>(d: &P::Device) -> (bool, u64, u64, bool, u64) {
        use reims_vgpu_core::platform::resource::RenderPipeline as _;
        let cache = DisplayPipelineCache::<P>::new(d.clone_ref()).unwrap();
        let state = cache
            .pipeline_for(PipelineKey {
                format: BGRA8,
                gamma: true,
            })
            .unwrap();
        (
            state.support_indirect_command_buffers(),
            state.resource_index(),
            state.imageblock_sample_length(),
            state.threadgroup_size_matches_tile_size(),
            state.max_total_threads_per_threadgroup(),
        )
    }
    let vulkan = values::<VulkanPlatform>(&dev());
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(values::<MetalPlatform>(&md), vulkan);
    }
}

/// A pass over a rasterization rate map of no layers, made from an immutable descriptor or
/// by a mutable map's reset, that only loads or clears its target and stores it stores what
/// Metal's does, in a process that has released no map of layers and after one was made,
/// drawn with and released, which makes Metal's draws in such a pass draw (its fresh side
/// first); a draw in such a pass dies at its selector, naming the map, and a draw of
/// primitive type 9 of no vertices and a tile dispatch there die by their own lines
#[test]
fn a_drawless_pass_over_a_map_of_no_layers_is_metals() {
    use reims_vgpu_core::platform::rate_map::{
        RateMap as _, RateMapDescriptor as _, RateMapLayerDescriptor as _,
    };
    use reims_vgpu_core::platform::values::{PassSet, RateMapDesc};
    use reims_vgpu_core::raise::death::DeathKind;

    /// `MTLLoadActionLoad`.
    const LOAD: u64 = 1;
    const SIDE: u64 = 16;

    /// A descriptor of a `SIDE` by `SIDE` screen of `mutability` (Immutable when unset), with
    /// no layers.
    fn none<P: Platform>(p: &P, mutability: Option<u64>) -> P::RateMapDescriptor {
        p.rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(SIDE, SIDE, 1),
            mutability,
            ..RateMapDesc::default()
        })
    }

    /// A descriptor of a `SIDE` by `SIDE` screen of `mutability`, with one layer of one
    /// sample of 1 on each axis.
    fn one_layer<P: Platform>(p: &P, mutability: Option<u64>) -> P::RateMapDescriptor {
        let desc = none(p, mutability);
        let layer = p
            .new_rate_map_layer(OnFault::Die, MtlSize::new(1, 1, 0))
            .unwrap()
            .unwrap();
        for at in [
            layer.horizontal_sample_storage(),
            layer.vertical_sample_storage(),
        ] {
            // SAFETY: the layer's storage holds one sample on each axis.
            unsafe { *at.as_ptr::<f32>() = 1.0 };
        }
        desc.set_layer(OnFault::Die, &layer, 0).unwrap();
        desc
    }

    /// A map of no layers of a `SIDE` by `SIDE` screen: made so, or a mutable map of one
    /// layer reset to none.
    fn no_layers<P: Platform>(p: &P, d: &P::Device, reset: bool) -> P::RateMap {
        if !reset {
            return d.new_rate_map(&none(p, None)).unwrap().unwrap();
        }
        let map = d.new_rate_map(&one_layer(p, Some(1))).unwrap().unwrap();
        map.reset(&none(p, None));
        map
    }

    /// Makes an immutable map of one layer, draws a triangle in a pass over it, and releases
    /// it once its command buffer completed: on Metal, a map of no layers made after it draws,
    /// which a pass that only loads or clears and stores must not change.
    fn draw_and_release<P: Platform>(p: &P, d: &P::Device) {
        let _pool = reims_vgpu_core::rt::Pool::here();
        let map = d.new_rate_map(&one_layer(p, None)).unwrap().unwrap();
        let (_dst, pass) = pass(p, d, &map, false);
        let state = fixture_state::<P>(d, "vert:tri_vert", "frag:tri_frag");
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        for cmd in [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            },
            RenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
            },
            RenderCmd::EndEncoding,
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
    }

    /// A `SIDE` by `SIDE` BGRA8 target of scattered texels and a pass of it over `map`,
    /// loading it or clearing it to (0.25, 0.5, 0.75, 1), and storing it.
    fn pass<P: Platform>(
        p: &P,
        d: &P::Device,
        map: &P::RateMap,
        load: bool,
    ) -> (P::Texture, P::RenderPassDescriptor) {
        let texels: Vec<u8> = (0..SIDE * SIDE * 4)
            .map(|i| u8::try_from(i * 7 % 251).unwrap())
            .collect();
        let dst = texture(
            p,
            d,
            BGRA8,
            TARGET | READ,
            (SIDE, SIDE),
            Some((&texels, SIDE * 4)),
        );
        let pass = p.render_pass_descriptor();
        pass.set_color_attachment(
            OnFault::Die,
            0,
            &ColorAttachmentSet {
                texture: Some(Some(&dst)),
                load_action: Some(if load { LOAD } else { CLEAR }),
                store_action: Some(STORE),
                clear_color: Some(ClearColor {
                    red: 0.25,
                    green: 0.5,
                    blue: 0.75,
                    alpha: 1.0,
                }),
                ..ColorAttachmentSet::default()
            },
        )
        .unwrap();
        pass.set_pass(&PassSet {
            rasterization_rate_map: Some(Some(map)),
            ..Default::default()
        });
        (dst, pass)
    }

    /// The target's bytes after a drawless pass over a map of no layers, made so or by a
    /// reset, loading or clearing; none where the pass makes no encoder.
    fn stored<P: Platform>(p: &P, d: &P::Device, reset: bool, load: bool) -> Option<Vec<u8>> {
        let map = no_layers(p, d, reset);
        let (dst, pass) = pass(p, d, &map, load);
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap()?;
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        let out = buffer::<P>(d, SIDE * SIDE * 4, 0, &[]);
        let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        for cmd in [
            BlitCmd::CopyTextureToBuffer {
                source_texture: &dst,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(SIDE, SIDE, 1),
                destination_buffer: &out,
                destination_offset: 0,
                destination_bytes_per_row: SIDE * 4,
                destination_bytes_per_image: SIDE * SIDE * 4,
            },
            BlitCmd::EndEncoding,
        ] {
            blit.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
        let n = usize::try_from(SIDE * SIDE * 4).unwrap();
        // SAFETY: a Shared buffer's bytes, read after the blit completed.
        Some(unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec())
    }

    let p = platform();
    let d = dev();
    let shapes = [(false, false), (false, true), (true, false), (true, true)];
    let ours = shapes.map(|(reset, load)| stored(&p, &d, reset, load));
    for (shape, bytes) in shapes.iter().zip(&ours) {
        assert!(bytes.is_some(), "reset, load {shape:?}: no encoder");
    }
    let state = fixture_state::<VulkanPlatform>(&d, "vert:tri_vert", "frag:tri_frag");
    let q = d.new_command_queue().unwrap();
    // A draw, named by the map; a draw of primitive type 9 of no vertices, by its own count;
    // and a tile dispatch, by its own line.
    let deaths = [
        (3, 3, "rate map of no layers"),
        (9, 0, "primitive type 9 of 0 vertices"),
        (
            0,
            0,
            "a tile dispatch in a pass of a rasterization rate map of no layers",
        ),
    ];
    for reset in [false, true] {
        let map = no_layers(&p, &d, reset);
        let (_dst, pass) = pass(&p, &d, &map, false);
        for (primitive_type, vertex_count, named) in deaths {
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
            e.encode(
                OnFault::Die,
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(&state),
                },
            )
            .unwrap();
            *crate::answer::LAST_LINE.lock() = None;
            let cmd = match primitive_type {
                0 => RenderCmd::DispatchThreadsPerTile {
                    threads_per_tile: MtlSize::new(1, 1, 1),
                },
                _ => RenderCmd::DrawPrimitives {
                    primitive_type,
                    vertex_start: 0,
                    vertex_count,
                },
            };
            let sel = cmd.selector();
            let site = catch_death(|| e.encode(OnFault::Die, cmd)).unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, sel),
                "reset {reset}: {line}"
            );
            assert!(line.contains(named), "{line}");
        }
    }
    let metal = MetalPlatform::system_default_device()
        .map(|md| (MetalPlatform::new(Box::new(NoLauncher)), md));
    if let Some((mp, md)) = &metal {
        let theirs = shapes.map(|(reset, load)| stored(mp, md, reset, load));
        for ((shape, o), t) in shapes.iter().zip(&ours).zip(&theirs) {
            assert!(o == t, "reset, load {shape:?}");
        }
    }
    // After a map of one layer was drawn with and released, each platform's own.
    draw_and_release(&p, &d);
    let ours = shapes.map(|(reset, load)| stored(&p, &d, reset, load));
    if let Some((mp, md)) = &metal {
        draw_and_release(mp, md);
        let theirs = shapes.map(|(reset, load)| stored(mp, md, reset, load));
        for ((shape, o), t) in shapes.iter().zip(&ours).zip(&theirs) {
            assert!(o == t, "after a released map, reset, load {shape:?}");
        }
    }
}

/// The tile size of a one-sample pass is Metal's
#[test]
fn the_tile_size_is_metals() {
    fn tile<P: Platform>(p: &P, d: &P::Device) -> (u64, u64) {
        let (_dst, pass) = pass_of(p, d, (64, 64));
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        let size = (e.tile_width(), e.tile_height());
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        size
    }
    let vulkan = tile(&platform(), &dev());
    assert_eq!(vulkan, (32, 32));
    if let Some(md) = MetalPlatform::system_default_device() {
        let mp = MetalPlatform::new(Box::new(NoLauncher));
        assert_eq!(tile(&mp, &md), vulkan);
    }
}

/// The tests' tile functions over a colour attachment's imageblock (`tile_edges.metal`).
const TILE_EDGES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tile_edges.metallib"));

/// The side of [`tile_edges_pass`]'s tiles.
const EDGE_TILE: u64 = 16;

/// The cells of [`tile_edges_pass`]'s tiles, one across and two down: cell `(x, y)` is
/// element `16 y + x` of `tile_edges.metal`'s buffers.
const EDGE_CELLS: u64 = EDGE_TILE * 2 * EDGE_TILE;

/// The width and height of [`tile_edges_pass`]'s attachment.
const EDGE_EXTENT: (u64, u64) = (10, 20);

/// The `RGBA8Unorm` codes of [`tile_edges_pass`]'s attachment's texel `(x, y)` before the
/// pass: red, green, blue and alpha.
fn edge_texel(x: u64, y: u64) -> [u8; 4] {
    let code = |v: u64| u8::try_from(v).unwrap();
    [code(16 * x + 1), code(8 * y + 2), 200, 255]
}

/// The `RGBA8Unorm` codes [`tile_edges_pass`]'s writes put in cell `i`.
fn edge_written(i: u64) -> [u8; 4] {
    [3 * i, 5 * i + 7, i / 2, 255 - i % 256].map(|v| v.to_le_bytes()[0])
}

/// The codes `(0.25, 0.5, 0.75, 1)` takes in `RGBA8Unorm`, [`tile_edges_pass`]'s clear colour.
const EDGE_CLEARED: [u8; 4] = [64, 128, 191, 255];

/// What a pass on `p`'s device `d` over an `RGBA8Unorm` attachment of [`EDGE_EXTENT`] in
/// tiles of [`EDGE_TILE`] leaves: one tile across and two down, so the cells past the
/// attachment's right edge, and the second tile's past its bottom edge, are cells of no
/// pixel. The attachment holds [`edge_texel`]s, and the pass loads it (`load` 1) or clears it
/// (2) to `(0.25, 0.5, 0.75, 1)`, stores it, and dispatches `tile_edges_read`, then
/// `tile_edges_write` of [`edge_written`], then `tile_edges_read` again, a thread a cell, each
/// state the suite's tile blob makes: what the two reads wrote, a cell's four floats after
/// another, and the attachment's bytes once stored.
fn tile_edges_pass<P: Platform>(p: &P, d: &P::Device, load: u64) -> [Vec<u8>; 3] {
    use reims_vgpu_core::platform::resource::Library as _;
    use reims_vgpu_core::platform::values::PassSet;
    use reims_vgpu_core::rt::Hold;
    let (w, h) = EDGE_EXTENT;
    let library = d.new_library(TILE_EDGES).unwrap().unwrap().unwrap();
    let state = |name: &str| {
        let f = library.new_function(OnFault::Die, name).unwrap().unwrap();
        let mut resolve = |_: Hold<'_>, r: u64| Ok((r == 0x31).then(|| f.clone_ref()));
        let blob = crate::gpu::serialized::tests::pipeline_blob("tile", 0, 0, 0x31);
        let desc = d
            .deserialize_tile_render_pipeline_descriptor(Hold::NONE, &blob, 0x100, &mut resolve)
            .unwrap()
            .unwrap();
        let made = d.new_tile_render_pipeline_state(Some(&desc), 0).unwrap();
        made.unwrap().unwrap()
    };
    let (read, write) = (state("tile_edges_read"), state("tile_edges_write"));
    let texels: Vec<u8> = (0..h)
        .flat_map(|y| (0..w).flat_map(move |x| edge_texel(x, y)))
        .collect();
    let target = texture(p, d, RGBA8, TARGET | READ, (w, h), Some((&texels, w * 4)));
    let pass = p.render_pass_descriptor();
    pass.set_color_attachment(
        OnFault::Die,
        0,
        &ColorAttachmentSet {
            texture: Some(Some(&target)),
            load_action: Some(load),
            store_action: Some(STORE),
            clear_color: Some(ClearColor {
                red: 0.25,
                green: 0.5,
                blue: 0.75,
                alpha: 1.0,
            }),
            ..ColorAttachmentSet::default()
        },
    )
    .unwrap();
    pass.set_pass(&PassSet {
        tile_width: Some(EDGE_TILE),
        tile_height: Some(EDGE_TILE),
        ..PassSet::default()
    });
    let cell_bytes = EDGE_CELLS * 16;
    let values: Vec<u8> = (0..EDGE_CELLS)
        .flat_map(|i| edge_written(i).map(|code| f32::from(code) / 255.0))
        .flat_map(f32::to_le_bytes)
        .collect();
    let written = buffer::<P>(d, cell_bytes, 0, &values);
    let unwritten = vec![0xab; usize::try_from(cell_bytes).unwrap()];
    let reads = [0, 1].map(|_| buffer::<P>(d, cell_bytes, 0, &unwritten));
    let stored = buffer::<P>(d, w * h * 4, 0, &[]);
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    for (state, buffer) in [(&read, &reads[0]), (&write, &written), (&read, &reads[1])] {
        for cmd in [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(state),
            },
            RenderCmd::SetTileBuffers {
                buffers: &[Some(buffer)],
                offsets: &[0],
                range: NsRange::new(0, 1),
            },
            RenderCmd::DispatchThreadsPerTile {
                threads_per_tile: MtlSize::new(EDGE_TILE, EDGE_TILE, 1),
            },
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
    }
    e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    for cmd in [
        BlitCmd::CopyTextureToBuffer {
            source_texture: &target,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(w, h, 1),
            destination_buffer: &stored,
            destination_offset: 0,
            destination_bytes_per_row: w * 4,
            destination_bytes_per_image: w * h * 4,
        },
        BlitCmd::EndEncoding,
    ] {
        blit.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    let bytes = |b: &P::Buffer, n: u64| {
        let n = usize::try_from(n).unwrap();
        // SAFETY: a Shared buffer of at least `n` bytes, read after its work completed.
        unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>(), n) }.to_vec()
    };
    [
        bytes(&reads[0], cell_bytes),
        bytes(&reads[1], cell_bytes),
        bytes(&stored, w * h * 4),
    ]
}

/// A tile dispatch's cells past its pass's attachment, in the partial tiles at its right and
/// bottom edges, hold Metal's values when the pass begins, and a tile function's writes there
/// are what the pass's next dispatch reads: [`tile_edges_pass`], loaded and cleared, reads
/// and stores what Metal's does, byte for byte. Metal's own answers here are the runtime's
/// measured rule, checked too: past a loaded attachment the texel nearest each cell, clamped to
/// its edge, past a cleared one the clear colour, and after the writes every cell's written
/// value, the attachment storing those inside it
#[test]
fn tile_dispatch_cells_past_the_attachment_are_metals() {
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new(Box::new(NoLauncher));
    let (p, d) = (platform(), dev());
    let (w, h) = EDGE_EXTENT;
    let floats = |codes: [u8; 4]| codes.map(|c| (f32::from(c) / 255.0).to_bits());
    let mut differ = Vec::new();
    for load in [1, CLEAR] {
        let theirs = tile_edges_pass(&mp, &md, load);
        let ours = catch_death(|| tile_edges_pass(&p, &d, load)).unwrap_or_else(|site| {
            let line = crate::answer::LAST_LINE.lock().clone();
            panic!(
                "load {load}: died at {} ({:?}): {line:?}",
                site.file, site.kind
            )
        });
        let cells = |bytes: &[u8]| -> Vec<[u32; 4]> {
            (bytes.as_chunks::<16>().0.iter())
                .map(|c| {
                    std::array::from_fn(|k| u32::from_le_bytes(c[4 * k..][..4].try_into().unwrap()))
                })
                .collect()
        };
        let (first, after) = (cells(&theirs[0]), cells(&theirs[1]));
        for i in 0..EDGE_CELLS {
            let (x, y) = (i % EDGE_TILE, i / EDGE_TILE);
            let at = usize::try_from(i).unwrap();
            if x >= w || y >= h {
                let want = match load {
                    CLEAR => EDGE_CLEARED,
                    _ => edge_texel(x.min(w - 1), y.min(h - 1)),
                };
                if first[at] != floats(want) {
                    differ.push(format!(
                        "load {load}: Metal's cell ({x}, {y}) began {:08x?}, not {want:?}",
                        first[at]
                    ));
                }
            }
            if after[at] != floats(edge_written(i)) {
                differ.push(format!(
                    "load {load}: Metal's cell ({x}, {y}) read {:08x?} after the writes",
                    after[at]
                ));
            }
        }
        for (k, what) in ["the first read", "the read after the writes"]
            .into_iter()
            .enumerate()
        {
            let (o, m) = (cells(&ours[k]), cells(&theirs[k]));
            let wrong: Vec<_> = (0..EDGE_CELLS)
                .zip(o.iter().zip(&m))
                .filter(|(_, (a, b))| a != b)
                .collect();
            if let Some(&(i, (a, b))) = wrong.first() {
                differ.push(format!(
                    "load {load}, {what}: {} cells differ, the first ({}, {}): ours {a:08x?}, \
                     Metal {b:08x?}",
                    wrong.len(),
                    i % EDGE_TILE,
                    i / EDGE_TILE
                ));
            }
        }
        let texels = |bytes: &[u8]| bytes.as_chunks::<4>().0.to_vec();
        let (o, m) = (texels(&ours[2]), texels(&theirs[2]));
        for (i, (a, b)) in (0..w * h).zip(o.iter().zip(&m)) {
            let (x, y) = (i % w, i / w);
            if *b != edge_written(y * EDGE_TILE + x) {
                differ.push(format!("load {load}: Metal stored {b:?} at ({x}, {y})"));
            }
            if a != b {
                differ.push(format!(
                    "load {load}: ours stored {a:?} at ({x}, {y}), Metal {b:?}"
                ));
            }
        }
    }
    assert!(differ.is_empty(), "{differ:#?}");
}

/// A guest's render state on `d` of `vertex` and `value_fragment` of `simd_stages.metal`, made
/// by `newRenderPipelineStateWithDescriptor:options:reflection:error:` of a descriptor decoded
/// as core decodes one.
fn simd_state<P: Platform>(
    d: &P::Device,
    vertex: &str,
) -> RaiseResult<Result<Option<P::RenderPipeline>, reims_vgpu_core::platform::types::NsErrorText>> {
    use crate::gpu::serialized::tests::pipeline_blob;
    use reims_vgpu_core::rt::Hold;
    let v = fixture::<P>(d, &format!("simd_stages:{vertex}"));
    let f = fixture::<P>(d, "simd_stages:value_fragment");
    let mut resolve = |_: Hold<'_>, r: u64| {
        Ok(match r {
            0x21 => Some(v.clone_ref()),
            0x22 => Some(f.clone_ref()),
            _ => None,
        })
    };
    let desc = d
        .deserialize_render_pipeline_descriptor(
            Hold::NONE,
            &pipeline_blob("basic", 0x21, 0x22, 0),
            &mut resolve,
        )
        .unwrap()
        .unwrap();
    d.new_render_pipeline_state_from_descriptor(Some(&desc), 0)
}

/// The pixels `vertex` and `value_fragment` of `simd_stages.metal` draw over [`pass_pixels`]'
/// target: three triangles of 9 vertices, each covering it, so the last, of vertices 6 to 8,
/// colours every pixel with the value its provoking vertex, 6, passes.
fn simd_vertex_pixels<P: Platform>(p: &P, d: &P::Device, vertex: &str) -> Vec<[u8; 4]> {
    let state = fixture_state::<P>(
        d,
        &format!("simd_stages:{vertex}"),
        "simd_stages:value_fragment",
    );
    pass_pixels(
        p,
        d,
        vec![
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            },
            RenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 9,
            },
        ],
    )
}

/// A vertex function using `simd_sum` or an explicit-width `simd_shuffle_and_fill_down`,
/// which the runtime runs in the vertex stage on this device as Metal does
/// (`DriverDeviation::MetalVertexSimdgroups`), makes its state by the guest's creator and
/// draws what Metal's draws: vertex 6's ID, the sum of the 9 vertices' IDs, and the filling
/// value of vertex 1 (1 ^ 0xff), as each vertex runs in the lane of its ID
#[test]
fn simd_operations_in_a_vertex_function_draw_as_metals() {
    let (p, d) = (platform(), dev());
    let md = MetalPlatform::system_default_device();
    for (vertex, red) in [
        ("plain_vertex", 6),
        ("sum_vertex", 36),
        ("fill_vertex", 0xfe),
    ] {
        let made = simd_state::<VulkanPlatform>(&d, vertex);
        assert!(matches!(made, Ok(Ok(Some(_)))), "{vertex}: {made:?}");
        let ours = simd_vertex_pixels(&p, &d, vertex);
        assert!(
            ours.iter().all(|&px| px == [0, 0, red, 255]),
            "{vertex}: {ours:?}"
        );
        if let Some(md) = &md {
            let mp = MetalPlatform::new(Box::new(NoLauncher));
            assert_eq!(ours, simd_vertex_pixels(&mp, md, vertex), "{vertex}");
        }
    }
}

/// The words `simd_kernel` of `simd_stages.metal` writes on `d`, two a thread, over 64 threads
/// in one threadgroup: each thread's `simd_sum` and explicit-width fill.
fn simd_kernel_words<P: Platform>(d: &P::Device) -> Vec<[u32; 2]> {
    use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    let f = fixture::<P>(d, "simd_stages:simd_kernel");
    let state = d
        .new_compute_pipeline_state_for_function(Some(&f))
        .unwrap()
        .unwrap();
    let out = buffer::<P>(d, 64 * 8, 0, &[]);
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
    let buffers = [Some(&out)];
    for cmd in [
        ComputeCmd::SetComputePipelineState {
            state: Some(&state),
        },
        ComputeCmd::SetBuffers {
            buffers: &buffers,
            offsets: &[0],
            range: NsRange::new(0, 1),
        },
        ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(64, 1, 1),
            threads_per_threadgroup: MtlSize::new(64, 1, 1),
        },
        ComputeCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    // SAFETY: a Shared buffer's 512 bytes, read after the dispatch completed.
    let words = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u32>(), 128) };
    words.as_chunks::<2>().0.to_vec()
}

/// The pixels `plain_vertex` and `fragment` of `simd_stages.metal` draw over [`pass_pixels`]'
/// target, one triangle covering it.
fn simd_fragment_pixels<P: Platform>(p: &P, d: &P::Device, fragment: &str) -> Vec<[u8; 4]> {
    let state = fixture_state::<P>(
        d,
        "simd_stages:plain_vertex",
        &format!("simd_stages:{fragment}"),
    );
    pass_pixels(
        p,
        d,
        vec![
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            },
            RenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
            },
        ],
    )
}

/// A kernel and a fragment function using `simd_sum` and an explicit-width
/// `simd_shuffle_and_fill_down`, the classes of device info key 33's SIMD bits, write what
/// Metal's write: these stages take them on this device, though key 33 answers no for them
/// (`caps`)
#[test]
fn simd_operations_in_kernels_and_fragments_run_as_metals() {
    let (p, d) = (platform(), dev());
    let words = simd_kernel_words::<VulkanPlatform>(&d);
    // Each simdgroup's sum of 7i + 1 over its 32 threads, and the fill of thread 5, the
    // filling value of thread 0 in its segment of 8 (1 ^ 0xff), and of thread 4, thread 7's.
    assert_eq!(
        (words[0][0], words[32][0], words[5][1], words[4][1]),
        (3504, 10672, 0xfe, 50),
        "{words:?}"
    );
    let pixels = ["sum_fragment", "fill_fragment"].map(|f| simd_fragment_pixels(&p, &d, f));
    assert!(pixels[0].iter().all(|px| px[3] == 255), "{:?}", pixels[0]);
    if let Some(md) = MetalPlatform::system_default_device() {
        let mp = MetalPlatform::new(Box::new(NoLauncher));
        assert_eq!(words, simd_kernel_words::<MetalPlatform>(&md));
        for (fragment, ours) in ["sum_fragment", "fill_fragment"].into_iter().zip(&pixels) {
            assert_eq!(
                ours,
                &simd_fragment_pixels(&mp, &md, fragment),
                "{fragment}"
            );
        }
    }
}

/// The identity colour matrix, as rows.
const IDENTITY: [[f32; 4]; 3] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];

/// Three points from vertex 1 under `state`, of [`SIZED`]'s positions.
fn points<P: Platform>(state: &P::RenderPipeline) -> Vec<RenderCmd<'_, P>> {
    vec![
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(state),
        },
        RenderCmd::DrawPrimitives {
            primitive_type: 0,
            vertex_start: 1,
            vertex_count: 3,
        },
    ]
}

/// The bytes of `seen`, a Private 64 by 64 `RGBA8Unorm` attachment, after a pass on platform
/// `P` whose colour 0 is a memoryless 64 by 64 `RGBA8Unorm` attachment loaded and stored
/// `DontCare`, and colour 1 `seen`, cleared to grey: it draws `fetch_seen_fragment` over both,
/// so `seen` holds what each fragment read of colour 0 by framebuffer fetch. With
/// `tiles_written`, a pass before it in its command buffer clears two other Private
/// attachments of that shape to (1, 0.5, 0.25, 1) and draws the same function over them, so
/// the tiles held other bytes before the memoryless attachment's pass.
pub(crate) fn memoryless_fetched<P: Platform>(
    p: &P,
    d: &P::Device,
    tiles_written: bool,
) -> Vec<u8> {
    use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
    use reims_vgpu_core::platform::resource::Library as _;
    use reims_vgpu_core::rt::Hold;

    use crate::gpu::serialized::tests::rounded_render_of;

    /// `MTLLoadActionDontCare` and `MTLStoreActionDontCare`.
    const DONT_CARE: u64 = 0;
    /// `MTLResourceStorageModePrivate` and `MTLResourceStorageModeMemoryless`.
    const PRIVATE: u64 = 0x20;
    const MEMORYLESS: u64 = 0x30;
    let library = d.new_library(ROUNDING).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    let functions = [function("sweep_vertex"), function("fetch_seen_fragment")];
    let mut resolve = |_: Hold<'_>, r: u64| {
        let at = usize::try_from(r - 1).unwrap();
        Ok(Some(functions[at].clone_ref()))
    };
    let colours: [&[(u8, u32)]; 2] = [&[(0, 0), (1, 70)], &[(0, 1), (1, 70)]];
    let blob = rounded_render_of(&["02 04 02000000"], &colours, 0);
    let desc = d
        .deserialize_render_pipeline_descriptor(Hold::NONE, &blob, &mut resolve)
        .unwrap()
        .unwrap();
    let state = d
        .new_render_pipeline_state_from_descriptor(Some(&desc), 0)
        .unwrap()
        .unwrap()
        .unwrap();
    let texture = |usage, options| {
        let desc = p.texture_descriptor(&TextureDesc {
            pixel_format: 70,
            width: 64,
            height: 64,
            usage,
            resource_options: Some(options),
            ..TextureDesc::default()
        });
        d.new_texture(OnFault::Die, &desc).unwrap().unwrap()
    };
    let memoryless = texture(TARGET, MEMORYLESS);
    let seen = texture(TARGET, PRIVATE);
    let others = [texture(TARGET, PRIVATE), texture(TARGET, PRIVATE)];
    let length = 64 * 64 * 4;
    let down = buffer::<P>(d, length, 0, &[]);
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let clear = |red, green, blue, alpha| ClearColor {
        red,
        green,
        blue,
        alpha,
    };
    let written = [
        (&others[0], CLEAR, STORE, clear(1.0, 0.5, 0.25, 1.0)),
        (&others[1], CLEAR, STORE, clear(1.0, 0.5, 0.25, 1.0)),
    ];
    let fetched = [
        (&memoryless, DONT_CARE, DONT_CARE, clear(0.0, 0.0, 0.0, 0.0)),
        (&seen, CLEAR, STORE, clear(0.5, 0.5, 0.5, 0.5)),
    ];
    let passes = match tiles_written {
        true => vec![written, fetched],
        false => vec![fetched],
    };
    for attachments in passes {
        let pass = p.render_pass_descriptor();
        for (index, (texture, load, store, clear)) in (0..).zip(attachments) {
            let attachment = ColorAttachmentSet {
                texture: Some(Some(texture)),
                load_action: Some(load),
                store_action: Some(store),
                clear_color: Some(clear),
                ..ColorAttachmentSet::default()
            };
            pass.set_color_attachment(OnFault::Die, index, &attachment)
                .unwrap();
        }
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        for cmd in [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            },
            RenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
            },
            RenderCmd::EndEncoding,
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
    }
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    for cmd in [
        BlitCmd::CopyTextureToBuffer {
            source_texture: &seen,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(64, 64, 1),
            destination_buffer: &down,
            destination_offset: 0,
            destination_bytes_per_row: 64 * 4,
            destination_bytes_per_image: length,
        },
        BlitCmd::EndEncoding,
    ] {
        blit.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    let n = usize::try_from(length).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the command buffer completed.
    unsafe { std::slice::from_raw_parts(down.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// How the first command buffer of [`after_a_failure`] fails.
#[derive(Debug, Clone, Copy)]
enum Failure {
    /// Its render pass, over a 4 by 4 target, draws primitive type 9: Metal's GPU hangs at it.
    Hang,
    /// Its blit encoder is made under these protection options.
    Protected(u64),
}

/// A command buffer of `q` that fails as `failure` says, on `p`'s device `d`, under `state`,
/// keeping its target in `keep`.
fn failing<P: Platform>(
    p: &P,
    d: &P::Device,
    q: &P::CommandQueue,
    (failure, state): (Failure, &P::RenderPipeline),
    keep: &mut Vec<P::Texture>,
) -> P::CommandBuffer {
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    match failure {
        Failure::Hang => {
            let (dst, pass) = pass_of(p, d, (4, 4));
            let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
            for cmd in [
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(state),
                },
                RenderCmd::DrawPrimitives {
                    primitive_type: 9,
                    vertex_start: 0,
                    vertex_count: 3,
                },
                RenderCmd::EndEncoding,
            ] {
                e.encode(OnFault::Die, cmd).unwrap();
            }
            keep.push(dst);
        }
        Failure::Protected(options) => {
            cb.set_protection_options(OnFault::Die, options).unwrap();
            let b = buffer::<P>(d, 16, 0, &[0; 16]);
            let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
            for cmd in [
                BlitCmd::FillBuffer {
                    buffer: &b,
                    range: NsRange::new(0, 16),
                    value: 0x77,
                },
                BlitCmd::EndEncoding,
            ] {
                blit.encode(OnFault::Die, cmd).unwrap();
            }
        }
    }
    cb
}

/// What [`filling`]'s completed handlers saw: each one's fill value, status and error code.
type Seen = Arc<reims_vgpu_core::sync::Mutex<Vec<(u8, u64, Option<i64>)>>>;

/// A command buffer of `q` that waits for `wait` where given, fills a new 16-byte buffer of `d`'s
/// with `value` in a blit, then signals a new event of its own to 1, with a completed handler that
/// keeps what it saw in `seen`; the buffer, and the event.
fn filling<P: Platform>(
    d: &P::Device,
    q: &P::CommandQueue,
    (value, wait): (u8, Option<&P::SharedEvent>),
    seen: &Seen,
) -> (P::CommandBuffer, P::Buffer, P::SharedEvent) {
    use reims_vgpu_core::platform::event::EventRef;
    let b = buffer::<P>(d, 16, 0, &[0; 16]);
    let event = d.new_shared_event().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    if let Some(wait) = wait {
        cb.encode_wait(Some(EventRef::Shared(wait)), 1).unwrap();
    }
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    for cmd in [
        BlitCmd::FillBuffer {
            buffer: &b,
            range: NsRange::new(0, 16),
            value,
        },
        BlitCmd::EndEncoding,
    ] {
        blit.encode(OnFault::Die, cmd).unwrap();
    }
    cb.encode_signal(Some(EventRef::Shared(&event)), 1).unwrap();
    let seen = Arc::clone(seen);
    cb.add_completed_handler(
        OnFault::Die,
        Box::new(move |info| {
            let code = info.error.map(|e| e.code);
            seen.lock().push((value, info.status.0, code));
        }),
    )
    .unwrap();
    (cb, b, event)
}

/// When [`after_a_failure`] commits its later command buffers on the hang's queue.
#[derive(Debug, Clone, Copy)]
enum Later {
    /// At once after the hang's commit.
    AtOnce,
    /// Once the hang's `waitUntilScheduled` returned.
    Scheduled,
    /// This long after the hang's commit.
    After(std::time::Duration),
}

/// What two command buffers committed on the queue of one whose draw of primitive type 9 hangs
/// Metal's GPU, `later`, show on `p`'s device `d`, and one committed on another queue then: each
/// one's status and error, the hang's status as it was committed, the first byte of the buffer
/// its blit fills, whether its event signal fired, and what its completed handler saw, in the
/// order the handlers ran. One line each.
fn after_a_failure<P: Platform>(
    p: &P,
    d: &P::Device,
    failure: Failure,
    later: Later,
) -> Vec<String> {
    use reims_vgpu_core::platform::event::SharedEvent as _;
    use reims_vgpu_core::platform::os::Clock as _;
    let state = fixture_state::<P>(d, "vert:tri_vert", "frag:tri_frag");
    let (q1, q2) = (
        d.new_command_queue().unwrap(),
        d.new_command_queue().unwrap(),
    );
    let seen = Seen::default();
    let mut keep = Vec::new();
    let hang = failing(p, d, &q1, (failure, &state), &mut keep);
    let clock = crate::os::OsClock::new();
    let since = |start| clock.now_mono().saturating_duration_since(start);
    let started = clock.now_mono();
    hang.commit(OnFault::Die).unwrap();
    match later {
        Later::AtOnce => {}
        Later::Scheduled => hang.wait_until_scheduled().unwrap(),
        Later::After(wait) => std::thread::sleep(wait),
    }
    let mut cases = Vec::new();
    let mut commit = |what, q, value| {
        let (cb, b, event) = filling::<P>(d, q, (value, None), &seen);
        let status = hang.status().0;
        cb.commit(OnFault::Die).unwrap();
        cases.push((what, cb, b, event, value, since(started), status));
    };
    commit("same queue", &q1, 0x11u8);
    commit("same queue, a second", &q1, 0x12);
    commit("other queue", &q2, 0x21);
    hang.wait_until_completed(OnFault::Die).unwrap();
    let hung = since(started);
    let mut lines = vec![format!(
        "{failure:?} {later:?}: status {} error {:?} after {hung:?}",
        hang.status().0,
        hang.error().map(|e| e.code)
    )];
    for (what, cb, b, event, value, at, status) in &cases {
        cb.wait_until_completed(OnFault::Die).unwrap();
        // SAFETY: a Shared buffer's bytes, read after its command buffer completed.
        let first = unsafe { *b.contents().as_ptr::<u8>() };
        let signalled = event.wait_until_signaled_value(1, 1);
        lines.push(format!(
            "  {what} (committed at {at:?}, the hang's status {status}): status {} error {:?} \
             filled {first:#x} of {value:#x} signal {signalled}",
            cb.status().0,
            cb.error().map(|e| (e.code, e.description)),
        ));
    }
    lines.push(format!("  handlers saw {:x?}", seen.lock()));
    lines
}

/// Measures what Metal's GPU does to command buffers committed on the queue of one whose draw of
/// primitive type 9 hangs it, and on another, at once, once it is scheduled and after 1 to 8 ms
/// ([`after_a_failure`]). One line a command buffer.
#[test]
#[ignore = "a measurement, with the GPU free: it hangs Metal's GPU"]
fn metal_runs_after_a_type_9_hang_here() {
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new(Box::new(NoLauncher));
    let ms = std::time::Duration::from_millis;
    for later in [
        Later::AtOnce,
        Later::Scheduled,
        Later::After(ms(1)),
        Later::After(ms(2)),
        Later::After(ms(4)),
        Later::After(ms(8)),
        Later::AtOnce,
    ] {
        for line in after_a_failure(&mp, &md, Failure::Hang, later) {
            println!("{line}");
        }
    }
    for options in [2, 0x10] {
        for later in [Later::AtOnce, Later::Scheduled] {
            for line in after_a_failure(&mp, &md, Failure::Protected(options), later) {
                println!("{line}");
            }
        }
    }
}
