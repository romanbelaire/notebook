use crate::app::App;
use crate::gfx::text_layout::{
    brush_bits_from_rgba_f32, build_cached_paragraph, measure_default_brush,
    paragraph_wrapped_flow, vello_draw_paragraph_layout, CachedParagraph, ParagraphCacheKey,
    ParagraphWrappedFlow, MAX_PARAGRAPH_CACHE_ENTRIES, PARAGRAPH_MEASURE_BRUSH_BITS,
};
use crate::gfx::types::Vertex;
use crate::gfx::icons::IconCache;
use crate::ui::core::Rect;
use crate::ui::components::Renderable;
use crate::ui::tab_bar::Tab;
use glam::{Mat4, Vec2, Vec4};
use wgpu::util::DeviceExt;
use winit::window::Window;
use pulldown_cmark::{Parser, Event, Tag};
use std::sync::Arc;
use vello::Scene;
use vello::peniko::{Color as VelloColor, Brush, Fill};
use parley::{FontContext, LayoutContext};

/// GPU compositing order (discriminant = draw order). `end_frame` draws quads then Vello for each layer in this order so chrome occludes layers below without hiding MainContent text above background.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum CompositeLayer {
    Background = 0,
    MainContent = 1,
    ConstellationText = 2,
    /// Vello (icons, small labels) for constellation shards: after markdown blits, still under sidebar/composer/HUD chrome.
    ConstellationOverlay = 3,
    SidebarChrome = 4,
    ComposerChrome = 5,
    HudChrome = 6,
    Modal = 7,
}

#[derive(Clone)]
struct TextDrawCommand {
    text: String,
    position: Vec2,
    color: Vec4,
    size: f32,
    /// Layout width for Parley (`break_all_lines`); None = unbounded single-line layout.
    max_width: Option<f32>,
    scissor: Option<ScissorRect>,  // Track which scissor was active when text was queued
    layer: CompositeLayer,
}

#[derive(Clone)]
struct IconDrawCommand {
    icon_name: String,
    position: Vec2,
    size: f32,
    color: Vec4,
    scissor: Option<ScissorRect>,  // Track which scissor was active when icon was queued
    layer: CompositeLayer,
}

/// A batch of vertices to render with a specific scissor rect
#[derive(Debug, Clone)]
struct RenderBatch {
    vertices: Vec<Vertex>,
    scissor: Option<ScissorRect>,
    layer: CompositeLayer,
}

/// Scissor rect for clipping rendering to a rectangular region
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScissorRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl ScissorRect {
    pub fn from_rect(rect: &crate::ui::core::Rect, _viewport_height: f32) -> Self {
        // IMPORTANT: Despite WGPU documentation stating scissor uses bottom-left origin,
        // with our projection matrix (orthographic_rh with top-left origin), scissor rects
        // work correctly when using UI coordinates directly - no conversion needed!
        // See WGPU_IDIOSYNCRASIES.md for details.
        //
        // Intersect `rect` with the framebuffer's top-left quadrant (x>=0, y>=0) before
        // converting to integer pixel extents. Clamping only min.y to 0 while keeping the
        // original height is wrong when the rect extends above y=0 (e.g. scrollable message
        // bubbles): the GPU scissor would cover too many rows and no longer match the bubble.
        let ix0 = rect.x.max(0.0);
        let iy0 = rect.y.max(0.0);
        let ix1 = rect.right().max(ix0);
        let iy1 = rect.bottom().max(iy0);
        let w = ix1 - ix0;
        let h = iy1 - iy0;
        Self {
            x: ix0 as u32,
            y: iy0 as u32,
            width: w.max(0.0) as u32,
            height: h.max(0.0) as u32,
        }
    }
    
    /// Intersect two scissor rects (for nested clipping)
    pub fn intersect(&self, other: &ScissorRect) -> ScissorRect {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);
        
        ScissorRect {
            x: x1,
            y: y1,
            width: x2.saturating_sub(x1),
            height: y2.saturating_sub(y1),
        }
    }
}

/// Trim amounts from `dest_rect` edges so the visible quad is `dest_rect ∩ [bx0,bx1)×[by0,by1)`.
/// Returns `(clip_left, clip_top, clip_right, clip_bottom)` or `None` if there is no intersection.
fn dest_rect_clip_against_bounds(
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    bx0: f32,
    by0: f32,
    bx1: f32,
    by1: f32,
) -> Option<(f32, f32, f32, f32)> {
    if w <= 0.0 || h <= 0.0 || bx1 <= bx0 || by1 <= by0 {
        return None;
    }
    let left_visible = x.max(bx0);
    let top_visible = y.max(by0);
    let right_visible = (x + w).min(bx1);
    let bottom_visible = (y + h).min(by1);
    if right_visible <= left_visible || bottom_visible <= top_visible {
        return None;
    }
    Some((
        left_visible - x,
        top_visible - y,
        (x + w) - right_visible,
        (y + h) - bottom_visible,
    ))
}

const MSAA_SAMPLE_COUNT: u32 = 4;

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    start_time: std::time::Instant,
    staging_belt: wgpu::util::StagingBelt,
    // Vello for text + 2D rendering
    vello_renderer: vello::Renderer,
    vello_target_texture: wgpu::Texture,
    vello_target_view: wgpu::TextureView,
    // 4× MSAA intermediate target; resolved to the swapchain at the end of each frame.
    msaa_texture: wgpu::Texture,
    msaa_view: wgpu::TextureView,
    // Full-screen blit (legacy); queued Vello text/icons use `blit_rect` only.
    #[allow(dead_code)]
    blit_pipeline: wgpu::RenderPipeline,
    /// Same bindings as `blit_pipeline`; kept for potential full-screen debug.
    #[allow(dead_code)]
    blit_tint_pipeline: wgpu::RenderPipeline,
    blit_bind_group: wgpu::BindGroup,
    blit_bind_group_layout: wgpu::BindGroupLayout,
    vello_sampler: wgpu::Sampler,
    font_context: FontContext,
    layout_context: LayoutContext<Brush>,
    text_queue: Vec<TextDrawCommand>,
    icon_cache: IconCache,
    icon_queue: Vec<IconDrawCommand>,
    icon_scenes: Vec<(Option<ScissorRect>, Scene)>,
    scissor_stack: Vec<ScissorRect>,
    viewport_height: f32,
    render_batches: Vec<RenderBatch>,
    current_batch_vertices: Vec<Vertex>,
    current_batch_clip_rect: Option<ScissorRect>,
    current_composite_layer: CompositeLayer,
    text_scenes: Vec<(Option<ScissorRect>, Scene)>,
    // Texture pools for text/icon rendering (reuse across frames)
    // Store only textures and views; bind groups are recreated as needed
    text_texture_pool: Vec<(wgpu::Texture, wgpu::TextureView)>,
    icon_texture_pool: Vec<(wgpu::Texture, wgpu::TextureView)>,
    max_texture_pool_size: usize,
    /// One Parley layout per key (text, size, max width, brush); measurement, wrapping, and draw reuse it.
    paragraph_cache: std::collections::HashMap<ParagraphCacheKey, CachedParagraph>,
    // Component validation tracking
    component_validation_enabled: bool,
    rendered_components: std::collections::HashSet<String>,
    component_hierarchy: std::collections::HashMap<String, Option<String>>, // component_id -> parent_id
    current_parent_stack: Vec<String>, // Stack of current parent IDs for nested rendering
    skipped_components: std::collections::HashSet<String>, // Components that should be skipped (orphaned/duplicate)
    duplicate_warnings_shown: std::collections::HashSet<String>, // Track which duplicate warnings we've already shown
    orphaned_warnings_shown: std::collections::HashSet<String>, // Track which orphaned warnings we've already shown
    /// First frame after creation: must full clear.
    first_frame: bool,
    /// Skip constellation update_node_sizes when (content_version, scale_bucket) unchanged.
    last_constellation_node_sizes_key: Option<(u64, u32, u32)>,
    /// Set by render_constellation when debug_text_stats; used for instrumentation logging.
    debug_constellation_visible_nodes: Option<usize>,
    /// Per-node text layer cache for constellation. Key: (node_id, is_user, content_hash, scale_bucket, width_bucket).
    text_layer_cache: TextLayerCache,
    /// Pipeline for drawing textured rects (text layers) with UV offset for scroll.
    blit_rect_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for blit_rect (texture, sampler, uv uniform).
    blit_rect_bind_group_layout: wgpu::BindGroupLayout,
    /// Draw commands for text layers; processed after main quads (before main Vello text when reordered).
    text_layer_draws: Vec<TextLayerDraw>,
    /// Cached textures for document previews (PDF page bitmaps).
    image_texture_cache: std::collections::HashMap<String, CachedImageEntry>,
    /// Draw commands for cached image textures.
    image_draws: Vec<ImageDrawCommand>,
}

/// Cached offscreen texture for a constellation bubble's text content.
struct TextLayerEntry {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

/// Cache of text layer textures keyed by (node_id, is_user, content_hash, scale_bucket, width_bucket).
struct TextLayerCache {
    entries: std::collections::HashMap<(String, bool, u64, u32, u32), TextLayerEntry>,
    /// Max entries; evict oldest when exceeded (simple: clear all for now).
    max_entries: usize,
}

/// Queued draw of a text layer at a screen rect with scroll-based UV.
#[derive(Clone)]
struct TextLayerDraw {
    key: (String, bool, u64, u32, u32),
    dest_rect: (f32, f32, f32, f32),  // x, y, w, h in screen space
    /// Vertical scroll offset in texture pixel space (same units as layer_h).
    scroll_offset: f32,
    scissor: Option<ScissorRect>,
    layer: CompositeLayer,
}

struct CachedImageEntry {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

#[derive(Clone)]
struct ImageDrawCommand {
    cache_key: String,
    dest_rect: (f32, f32, f32, f32),
    scissor: Option<ScissorRect>,
    layer: CompositeLayer,
}

impl Renderer {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .unwrap();
        let surface_caps = surface.get_capabilities(&adapter);
        
        // Vello requires Rgba8Unorm (non-sRGB) for storage texture compatibility
        let format = surface_caps.formats.iter()
            .find(|f| matches!(f, wgpu::TextureFormat::Rgba8Unorm))
            .copied()
            .or_else(|| {
                // Fallback to Bgra8Unorm if Rgba8Unorm not available
                surface_caps.formats.iter()
                    .find(|f| matches!(f, wgpu::TextureFormat::Bgra8Unorm))
                    .copied()
            })
            .or_else(|| {
                // Last resort: use first available format
                surface_caps.formats.first().copied()
            })
            .expect("No suitable surface format found");
        
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ui shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ui_shader.wgsl").into()),
        });

        // Uniform block: projection (64) + fragment: time, scroll_velocity, cursor(2), slider_velocity, tab_bar_x, tab_width, from_index, to_index (9*4=36) = 100 bytes; pad to 112
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniform buffer"),
            contents: bytemuck::cast_slice(&[0.0f32; 28]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform bind group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Cw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: MSAA_SAMPLE_COUNT,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            cache: None,
            multiview: None,
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vertex buffer"),
            size: 1024 * 1024,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_belt = wgpu::util::StagingBelt::new(1024);

        // Initialize vello renderer
        // Don't specify surface_format since we'll render to an intermediate texture
        let vello_renderer = vello::Renderer::new(
            &device,
            vello::RendererOptions {
                surface_format: None,
                use_cpu: false,
                antialiasing_support: vello::AaSupport::all(),
                num_init_threads: None,
            },
        ).unwrap();

        // Create intermediate texture for vello rendering
        // Use Rgba8Unorm which vello supports for storage bindings
        let vello_target_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vello_target"),
            size: wgpu::Extent3d {
                width: size.width.max(1),
                height: size.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING 
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        
        let vello_target_view = vello_target_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create 4× MSAA render target (same format as the swapchain surface).
        let msaa_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("msaa_texture"),
            size: wgpu::Extent3d {
                width: size.width.max(1),
                height: size.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLE_COUNT,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let msaa_view = msaa_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create sampler for vello texture
        let vello_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("vello_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create blit pipeline to copy vello texture to surface
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blit.wgsl").into()),
        });

        let blit_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit_bind_group"),
            layout: &blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&vello_target_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&vello_sampler),
                },
            ],
        });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit_pipeline_layout"),
            bind_group_layouts: &[&blit_bind_group_layout],
            push_constant_ranges: &[],
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit_pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: MSAA_SAMPLE_COUNT,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let blit_tint_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit_tint shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blit_tint.wgsl").into()),
        });
        let blit_tint_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit_tint_pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_tint_shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_tint_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: MSAA_SAMPLE_COUNT,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Blit rect pipeline for text layers (textured quad at rect with UV offset)
        let blit_rect_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit_rect shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blit_rect.wgsl").into()),
        });
        let blit_rect_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit_rect_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(16).unwrap()),
                    },
                    count: None,
                },
            ],
        });
        let blit_rect_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit_rect_pipeline_layout"),
            bind_group_layouts: &[&blit_rect_bind_group_layout],
            push_constant_ranges: &[],
        });
        let blit_rect_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit_rect_pipeline"),
            layout: Some(&blit_rect_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_rect_shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_rect_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    // Vello outputs straight RGBA; shader premultiplies. Blend is "over" for premultiplied src.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: MSAA_SAMPLE_COUNT,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Initialize parley font and layout contexts
        let font_context = FontContext::default();
        let layout_context = LayoutContext::new();

        let mut renderer = Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            vertex_buffer,
            vertex_count: 0,
            uniform_buffer,
            uniform_bind_group,
            start_time: std::time::Instant::now(),
            staging_belt,
            vello_renderer,
            vello_target_texture,
            vello_target_view,
            msaa_texture,
            msaa_view,
            blit_pipeline,
            blit_tint_pipeline,
            blit_bind_group,
            blit_bind_group_layout,
            vello_sampler,
            font_context,
            layout_context,
            text_queue: Vec::new(),
            icon_cache: IconCache::new(),
            icon_queue: Vec::new(),
            icon_scenes: Vec::new(),
            scissor_stack: Vec::new(),
            viewport_height: size.height as f32,
            render_batches: Vec::new(),
            current_batch_vertices: Vec::new(),
            current_batch_clip_rect: None,
            current_composite_layer: CompositeLayer::MainContent,
            text_scenes: Vec::new(),
            text_texture_pool: Vec::new(),
            icon_texture_pool: Vec::new(),
            max_texture_pool_size: 8,
            paragraph_cache: std::collections::HashMap::new(),
            // Hardcoded off: duplicate/orphan checks were blocking Text queue; re-enable after fixing IDs / hierarchy.
            component_validation_enabled: false,
            rendered_components: std::collections::HashSet::new(),
            component_hierarchy: std::collections::HashMap::new(),
            skipped_components: std::collections::HashSet::new(),
            duplicate_warnings_shown: std::collections::HashSet::new(),
            orphaned_warnings_shown: std::collections::HashSet::new(),
            current_parent_stack: Vec::new(),
            first_frame: true,
            last_constellation_node_sizes_key: None,
            debug_constellation_visible_nodes: None,
            text_layer_cache: TextLayerCache {
                entries: std::collections::HashMap::new(),
                max_entries: 64,
            },
            blit_rect_pipeline,
            blit_rect_bind_group_layout,
            text_layer_draws: Vec::new(),
            image_texture_cache: std::collections::HashMap::new(),
            image_draws: Vec::new(),
        };

        renderer.update_projection_matrix(size.width, size.height);
        
        renderer
    }

    pub(crate) fn paragraph_cache_get_or_insert(
        &mut self,
        key: ParagraphCacheKey,
        text: &str,
        font_size: f32,
        max_width: Option<f32>,
        brush: Brush,
    ) -> &CachedParagraph {
        if !self.paragraph_cache.contains_key(&key) {
            while self.paragraph_cache.len() >= MAX_PARAGRAPH_CACHE_ENTRIES {
                if let Some(k) = self.paragraph_cache.keys().next().cloned() {
                    self.paragraph_cache.remove(&k);
                } else {
                    break;
                }
            }
            let entry = build_cached_paragraph(
                &mut self.layout_context,
                &mut self.font_context,
                text,
                font_size,
                max_width,
                brush,
                crate::ui::style::font_size::LINE_HEIGHT_RATIO,
                key.bold(),
                key.italic(),
            );
            self.paragraph_cache.insert(key.clone(), entry);
        }
        self.paragraph_cache.get(&key).unwrap()
    }

    /// After [`Self::render_text_to_scene`] for an inline segment, Parley line geometry for continuation.
    fn segment_inline_continuation_after_scene_draw(
        &mut self,
        text: &str,
        scale: f32,
        rem: f32,
        color: [f32; 4],
        bold: bool,
        italic: bool,
    ) -> ParagraphWrappedFlow {
        let bits = brush_bits_from_rgba_f32(color);
        let brush = Brush::Solid(VelloColor::rgba(
            color[0] as f64,
            color[1] as f64,
            color[2] as f64,
            color[3] as f64,
        ));
        let key = ParagraphCacheKey::new(text, scale, Some(rem), bits, bold, italic);
        let entry = self.paragraph_cache_get_or_insert(key, text, scale, Some(rem), brush);
        let mut flow = paragraph_wrapped_flow(&entry.layout);
        if flow.content_height == 0.0 {
            flow.content_height = entry.content_height;
        }
        if flow.last_line_height == 0.0 && flow.content_height > 0.0 {
            flow.last_line_height = flow.content_height;
        }
        flow
    }

    pub fn measure_markdown_segment_flow(
        &mut self,
        text: &str,
        size: f32,
        rem: f32,
        bold: bool,
        italic: bool,
    ) -> ParagraphWrappedFlow {
        let key =
            ParagraphCacheKey::new(text, size, Some(rem), PARAGRAPH_MEASURE_BRUSH_BITS, bold, italic);
        let brush = measure_default_brush();
        let entry = self.paragraph_cache_get_or_insert(key, text, size, Some(rem), brush);
        let mut flow = paragraph_wrapped_flow(&entry.layout);
        if flow.content_height == 0.0 {
            flow.content_height = entry.content_height;
        }
        if flow.last_line_height == 0.0 && flow.content_height > 0.0 {
            flow.last_line_height = flow.content_height;
        }
        flow
    }

    pub(crate) fn segment_width_unbounded_queue(&mut self, text: &str, scale: f32, color: Vec4) -> f32 {
        let c = [color.x, color.y, color.z, color.w];
        let bits = brush_bits_from_rgba_f32(c);
        let brush = Brush::Solid(VelloColor::rgba(
            c[0] as f64,
            c[1] as f64,
            c[2] as f64,
            c[3] as f64,
        ));
        let key = ParagraphCacheKey::new(text, scale, None, bits, false, false);
        self.paragraph_cache_get_or_insert(key, text, scale, None, brush)
            .first_width
    }

    /// Render text to vello scene using parley. Returns the height of the laid-out content.
    /// Line height ratio accommodates bold/italic glyphs to prevent overlap.
    fn render_text_to_scene(
        &mut self,
        scene: &mut Scene,
        text: &str,
        x: f32,
        y: f32,
        font_size: f32,
        max_width: Option<f32>,
        color: [f32; 4],
        bold: bool,
        italic: bool,
    ) -> f32 {
        let brush = Brush::Solid(VelloColor::rgba(
            color[0] as f64,
            color[1] as f64,
            color[2] as f64,
            color[3] as f64,
        ));
        let bits = brush_bits_from_rgba_f32(color);
        let key = ParagraphCacheKey::new(text, font_size, max_width, bits, bold, italic);
        let entry = self.paragraph_cache_get_or_insert(
            key.clone(),
            text,
            font_size,
            max_width,
            brush,
        );
        let line_h = crate::ui::style::font_size::LINE_HEIGHT_RATIO;
        vello_draw_paragraph_layout(
            scene,
            &entry.layout,
            x,
            y,
            font_size,
            line_h,
        )
    }

    /// Full-line flush for markdown block layout (paragraph / list item / soft break).
    fn markdown_scene_end_line(
        &mut self,
        scene: &mut Scene,
        current_text: &mut String,
        seg_flush_x: f32,
        current_y: f32,
        start_x: f32,
        max_width: f32,
        max_w: &mut f32,
        font_size: f32,
        line_height: f32,
        base_color: Vec4,
        bold: bool,
        italic: bool,
        code: bool,
    ) -> f32 {
        let wrap_remaining = |cx: f32| (max_width - (cx - start_x)).max(1.0);
        let effective_font_size = |_b: bool, _i: bool, _c: bool| -> f32 { font_size };
        let code_fg = crate::ui::style::markdown::CODE_FOREGROUND();
        let segment_color = |c: bool| -> [f32; 4] {
            if c {
                [code_fg.x, code_fg.y, code_fg.z, code_fg.w]
            } else {
                [base_color.x, base_color.y, base_color.z, base_color.w]
            }
        };
        if !current_text.is_empty() {
            let scale = effective_font_size(bold, italic, code);
            let color = segment_color(code);
            let rem = wrap_remaining(seg_flush_x);
            let seg_h = self.render_text_to_scene(
                scene, current_text, seg_flush_x, current_y, scale, Some(rem), color, bold, italic,
            );
            let flow =
                self.segment_inline_continuation_after_scene_draw(current_text, scale, rem, color, bold, italic);
            *max_w = (*max_w).max(seg_flush_x - start_x + flow.layout_width);
            current_text.clear();
            current_y + seg_h
        } else {
            current_y + line_height
        }
    }

    /// Build a Vello scene from markdown text. Used for text layer caching.
    /// Returns (width, height) of the laid-out content.
    fn build_markdown_scene(
        &mut self,
        scene: &mut Scene,
        markdown: &str,
        start_x: f32,
        start_y: f32,
        max_width: f32,
        font_size: f32,
        base_color: Vec4,
    ) -> (f32, f32) {
        use pulldown_cmark::{Parser, Event, Tag, TagEnd};
        let mut current_x = start_x;
        let mut current_y = start_y;
        // Must stay aligned with GraphState markdown measurement.
        let line_height = font_size * crate::ui::style::font_size::LINE_HEIGHT_RATIO;
        let mut max_w = 0.0f32;
        let mut bold = false;
        let mut italic = false;
        let mut code = false;
        let mut current_text = String::new();
        #[derive(Clone, Copy)]
        struct ListFrame {
            ordered: bool,
            next_n: u64,
        }
        let mut list_stack: Vec<ListFrame> = Vec::new();

        let effective_font_size = |_b: bool, _i: bool, _c: bool| -> f32 {
            font_size
        };
        let code_fg = crate::ui::style::markdown::CODE_FOREGROUND();
        let segment_color = |c: bool| -> [f32; 4] {
            if c {
                [code_fg.x, code_fg.y, code_fg.z, code_fg.w]
            } else {
                [base_color.x, base_color.y, base_color.z, base_color.w]
            }
        };
        let wrap_remaining = |cx: f32| (max_width - (cx - start_x)).max(1.0);

        for event in Parser::new(markdown) {
            match event {
                Event::Start(Tag::List(first)) => {
                    list_stack.push(ListFrame {
                        ordered: first.is_some(),
                        next_n: first.unwrap_or(1),
                    });
                }
                Event::End(TagEnd::List(_)) => {
                    list_stack.pop();
                }
                Event::Start(Tag::Item) => {
                    let prefix = list_stack.last().map(|f| {
                        if f.ordered {
                            format!("{}. ", f.next_n)
                        } else {
                            "• ".to_string()
                        }
                    });
                    if let Some(p) = prefix {
                        current_text.push_str(&p);
                    }
                }
                Event::End(TagEnd::Item) => {
                    if let Some(f) = list_stack.last_mut() {
                        if f.ordered {
                            f.next_n += 1;
                        }
                    }
                    if !current_text.is_empty() {
                        current_y = self.markdown_scene_end_line(
                            scene,
                            &mut current_text,
                            current_x,
                            current_y,
                            start_x,
                            max_width,
                            &mut max_w,
                            font_size,
                            line_height,
                            base_color,
                            bold,
                            italic,
                            code,
                        );
                    }
                    current_x = start_x;
                }
                Event::End(TagEnd::Paragraph) => {
                    current_y = self.markdown_scene_end_line(
                        scene,
                        &mut current_text,
                        current_x,
                        current_y,
                        start_x,
                        max_width,
                        &mut max_w,
                        font_size,
                        line_height,
                        base_color,
                            bold,
                            italic,
                            code,
                        );
                    current_x = start_x;
                }
                Event::Start(Tag::Strong) => {
                    if !current_text.is_empty() {
                        let scale = effective_font_size(bold, italic, code);
                        let color = segment_color(code);
                        let seg_x = current_x;
                        let rem = wrap_remaining(current_x);
                        let seg_h = self.render_text_to_scene(
                            scene, &current_text, seg_x, current_y, scale, Some(rem), color, bold, italic,
                        );
                        let flow = self.segment_inline_continuation_after_scene_draw(
                            &current_text, scale, rem, color, bold, italic,
                        );
                        current_y += seg_h - flow.last_line_height;
                        current_x = seg_x + flow.last_line_advance;
                        max_w = max_w.max(seg_x - start_x + flow.layout_width);
                        current_text.clear();
                    }
                    bold = true;
                }
                Event::End(pulldown_cmark::TagEnd::Strong) => {
                    if !current_text.is_empty() {
                        let scale = effective_font_size(bold, italic, code);
                        let color = segment_color(code);
                        let seg_x = current_x;
                        let rem = wrap_remaining(current_x);
                        let seg_h = self.render_text_to_scene(
                            scene, &current_text, seg_x, current_y, scale, Some(rem), color, bold, italic,
                        );
                        let flow = self.segment_inline_continuation_after_scene_draw(
                            &current_text, scale, rem, color, bold, italic,
                        );
                        current_y += seg_h - flow.last_line_height;
                        current_x = seg_x + flow.last_line_advance;
                        max_w = max_w.max(seg_x - start_x + flow.layout_width);
                        current_text.clear();
                    }
                    bold = false;
                }
                Event::Start(Tag::Emphasis) => {
                    if !current_text.is_empty() {
                        let scale = effective_font_size(bold, italic, code);
                        let color = segment_color(code);
                        let seg_x = current_x;
                        let rem = wrap_remaining(current_x);
                        let seg_h = self.render_text_to_scene(
                            scene, &current_text, seg_x, current_y, scale, Some(rem), color, bold, italic,
                        );
                        let flow = self.segment_inline_continuation_after_scene_draw(
                            &current_text, scale, rem, color, bold, italic,
                        );
                        current_y += seg_h - flow.last_line_height;
                        current_x = seg_x + flow.last_line_advance;
                        max_w = max_w.max(seg_x - start_x + flow.layout_width);
                        current_text.clear();
                    }
                    italic = true;
                }
                Event::End(pulldown_cmark::TagEnd::Emphasis) => {
                    if !current_text.is_empty() {
                        let scale = effective_font_size(bold, italic, code);
                        let color = segment_color(code);
                        let seg_x = current_x;
                        let rem = wrap_remaining(current_x);
                        let seg_h = self.render_text_to_scene(
                            scene, &current_text, seg_x, current_y, scale, Some(rem), color, bold, italic,
                        );
                        let flow = self.segment_inline_continuation_after_scene_draw(
                            &current_text, scale, rem, color, bold, italic,
                        );
                        current_y += seg_h - flow.last_line_height;
                        current_x = seg_x + flow.last_line_advance;
                        max_w = max_w.max(seg_x - start_x + flow.layout_width);
                        current_text.clear();
                    }
                    italic = false;
                }
                Event::Start(Tag::CodeBlock(_)) => {
                    if !current_text.is_empty() {
                        let scale = effective_font_size(bold, italic, code);
                        let color = segment_color(code);
                        let seg_x = current_x;
                        let rem = wrap_remaining(current_x);
                        let seg_h = self.render_text_to_scene(
                            scene, &current_text, seg_x, current_y, scale, Some(rem), color, bold, italic,
                        );
                        let flow = self.segment_inline_continuation_after_scene_draw(
                            &current_text, scale, rem, color, bold, italic,
                        );
                        current_y += seg_h - flow.last_line_height;
                        current_x = seg_x + flow.last_line_advance;
                        max_w = max_w.max(seg_x - start_x + flow.layout_width);
                        current_text.clear();
                    }
                    code = true;
                }
                Event::End(pulldown_cmark::TagEnd::CodeBlock) => {
                    if !current_text.is_empty() {
                        let scale = effective_font_size(bold, italic, code);
                        let color = segment_color(code);
                        let seg_x = current_x;
                        let rem = wrap_remaining(current_x);
                        let seg_h = self.render_text_to_scene(
                            scene, &current_text, seg_x, current_y, scale, Some(rem), color, bold, italic,
                        );
                        let flow = self.segment_inline_continuation_after_scene_draw(
                            &current_text, scale, rem, color, bold, italic,
                        );
                        current_y += seg_h - flow.last_line_height;
                        current_x = seg_x + flow.last_line_advance;
                        max_w = max_w.max(seg_x - start_x + flow.layout_width);
                        current_text.clear();
                    }
                    code = false;
                }
                Event::Text(text) => {
                    for word in text.split_whitespace() {
                        let word_with_space = if current_text.is_empty() {
                            word.to_string()
                        } else {
                            format!(" {}", word)
                        };
                        let scale = effective_font_size(bold, italic, code);
                        let test_text = format!("{}{}", current_text, word_with_space);
                        let line_used = current_x - start_x;
                        let rem = (max_width - line_used).max(1.0);
                        let cand_key = ParagraphCacheKey::new(
                            &test_text,
                            scale,
                            Some(rem),
                            PARAGRAPH_MEASURE_BRUSH_BITS,
                            bold,
                            italic,
                        );
                        let n_lines = self
                            .paragraph_cache_get_or_insert(
                                cand_key,
                                &test_text,
                                scale,
                                Some(rem),
                                measure_default_brush(),
                            )
                            .layout
                            .len();
                        if n_lines > 1 && !current_text.is_empty() {
                            let color = segment_color(code);
                            let seg_flush_x = current_x;
                            let rem_flush = wrap_remaining(current_x);
                            let seg_h = self.render_text_to_scene(
                                scene,
                                &current_text,
                                seg_flush_x,
                                current_y,
                                scale,
                                Some(rem_flush),
                                color,
                                bold,
                                italic,
                            );
                            let flow = self.segment_inline_continuation_after_scene_draw(
                                &current_text,
                                scale,
                                rem_flush,
                                color,
                                bold,
                                italic,
                            );
                            max_w = max_w.max(seg_flush_x - start_x + flow.layout_width);
                            current_y += seg_h;
                            current_x = start_x;
                            current_text = word.to_string();
                        } else {
                            current_text = test_text;
                        }
                    }
                }
                Event::SoftBreak => {
                    current_y = self.markdown_scene_end_line(
                        scene,
                        &mut current_text,
                        current_x,
                        current_y,
                        start_x,
                        max_width,
                        &mut max_w,
                        font_size,
                        line_height,
                        base_color,
                            bold,
                            italic,
                            code,
                        );
                    current_x = start_x;
                }
                Event::HardBreak => {
                    current_y = self.markdown_scene_end_line(
                        scene,
                        &mut current_text,
                        current_x,
                        current_y,
                        start_x,
                        max_width,
                        &mut max_w,
                        font_size,
                        line_height,
                        base_color,
                            bold,
                            italic,
                            code,
                        );
                    current_x = start_x;
                }
                _ => {}
            }
        }
        let height = if !current_text.is_empty() {
            let scale = effective_font_size(bold, italic, code);
            let color = segment_color(code);
            let seg_x = current_x;
            let rem = wrap_remaining(current_x);
            let seg_h = self.render_text_to_scene(
                scene, &current_text, seg_x, current_y, scale, Some(rem), color, bold, italic,
            );
            let flow = self.segment_inline_continuation_after_scene_draw(&current_text, scale, rem, color, bold, italic);
            max_w = max_w.max(seg_x - start_x + flow.layout_width);
            current_y - start_y + seg_h
        } else {
            current_y - start_y
        };
        let out = (max_w.max(1.0).min(max_width), height.max(1.0));
        out
    }

    // Helper: Render icon to vello scene
    fn render_icon_to_scene(
        &mut self,
        scene: &mut Scene,
        icon_name: &str,
        position: Vec2,
        size: f32,
        color: Vec4,
    ) {
        use vello::peniko::kurbo::{Affine, Cap, Join, Stroke};
        
        // Get cached paths for this icon
        let paths = match self.icon_cache.get_paths(icon_name) {
            Some(paths) => paths,
            None => return, // Icon not found or failed to parse
        };
        
        // Create brush from color
        let brush = Brush::Solid(VelloColor::rgba(
            color.x as f64,
            color.y as f64,
            color.z as f64,
            color.w as f64,
        ));
        
        // Scale from SVG viewBox space to requested pixel size
        let viewbox_size = paths.viewbox_size;
        let scale = size as f64 / viewbox_size;
        
        // Create transform: translate to position, then scale
        let transform = Affine::translate((position.x as f64, position.y as f64))
            * Affine::scale(scale);
        
        // Render fill paths
        for path in &paths.fill_paths {
            scene.fill(Fill::NonZero, transform, &brush, None, path);
        }
        
        // Stroke-only SVGs (pencil, trash, etc.) must use GPU stroke; filling their outlines draws nothing.
        for (path, stroke_width) in &paths.stroke_paths {
            let w = stroke_width * scale;
            let style = Stroke::new(w)
                .with_start_cap(Cap::Round)
                .with_end_cap(Cap::Round)
                .with_join(Join::Round);
            scene.stroke(&style, transform, &brush, None, path);
        }
    }

    fn handle_text_input_mouse_interactions(&mut self, app: &mut App) {
        // Handle mouse move for selection during drag using accurate glyph positions
        match app.focused_input {
            Some(0) => {
                if let Some(ref mut chat) = app.chat_window {
                    if chat.input_field.is_selecting && chat.input_field.contains(app.mouse_pos) {
                    let positions = if chat.input_field.glyph_positions.is_empty() {
                        self.compute_glyph_positions(&chat.input_field.text, 14.0, 0.0)
                        } else {
                        chat.input_field.glyph_positions.clone()
                        };
                        let new_pos = chat.input_field.get_cursor_position_from_positions(app.mouse_pos, &positions);
                        chat.input_field.cursor_position = new_pos;
                        if let Some(anchor) = chat.input_field.selection_anchor {
                            chat.input_field.selection_start = Some(anchor.min(new_pos));
                            chat.input_field.selection_end = Some(anchor.max(new_pos));
                        }
                    }
                }
            }
            Some(1) => {
                if let Some(ref mut library) = app.library_window {
                    if library.search_input.is_selecting && library.search_input.contains(app.mouse_pos) {
                        let positions = if library.search_input.glyph_positions.is_empty() {
                            self.compute_glyph_positions(&library.search_input.text, 14.0, 0.0)
                        } else {
                            library.search_input.glyph_positions.clone()
                        };
                        let new_pos = library.search_input.get_cursor_position_from_positions(app.mouse_pos, &positions);
                        library.search_input.cursor_position = new_pos;
                        if let Some(anchor) = library.search_input.selection_anchor {
                            library.search_input.selection_start = Some(anchor.min(new_pos));
                            library.search_input.selection_end = Some(anchor.max(new_pos));
                        }
                    }
                }
            }
            Some(2) => {
                if let Some(ref mut ingest) = app.ingest_window {
                    if ingest.pdf_dir_input.is_selecting && ingest.pdf_dir_input.contains(app.mouse_pos) {
                        let positions = if ingest.pdf_dir_input.glyph_positions.is_empty() {
                            self.compute_glyph_positions(&ingest.pdf_dir_input.text, 14.0, 0.0)
                        } else {
                            ingest.pdf_dir_input.glyph_positions.clone()
                        };
                        let new_pos = ingest.pdf_dir_input.get_cursor_position_from_positions(app.mouse_pos, &positions);
                        ingest.pdf_dir_input.cursor_position = new_pos;
                        if let Some(anchor) = ingest.pdf_dir_input.selection_anchor {
                            ingest.pdf_dir_input.selection_start = Some(anchor.min(new_pos));
                            ingest.pdf_dir_input.selection_end = Some(anchor.max(new_pos));
                        }
                    }
                }
            }
            Some(3) => {
                if let Some(settings) = app.settings_window.borrow_mut().as_mut() {
                    if settings.hf_token_input.is_selecting && settings.hf_token_input.contains(app.mouse_pos) {
                        let positions = if settings.hf_token_input.glyph_positions.is_empty() {
                            self.compute_glyph_positions(&settings.hf_token_input.text, 14.0, 0.0)
                        } else {
                            settings.hf_token_input.glyph_positions.clone()
                        };
                        let new_pos = settings.hf_token_input.get_cursor_position_from_positions(app.mouse_pos, &positions);
                        settings.hf_token_input.cursor_position = new_pos;
                        if let Some(anchor) = settings.hf_token_input.selection_anchor {
                            settings.hf_token_input.selection_start = Some(anchor.min(new_pos));
                            settings.hf_token_input.selection_end = Some(anchor.max(new_pos));
                        }
                    }
                }
            }
            Some(4) => {
                if let Some(settings) = app.settings_window.borrow_mut().as_mut() {
                    if settings.model_id_input.is_selecting && settings.model_id_input.contains(app.mouse_pos) {
                        let positions = if settings.model_id_input.glyph_positions.is_empty() {
                            self.compute_glyph_positions(&settings.model_id_input.text, 14.0, 0.0)
                        } else {
                            settings.model_id_input.glyph_positions.clone()
                        };
                        let new_pos = settings.model_id_input.get_cursor_position_from_positions(app.mouse_pos, &positions);
                        settings.model_id_input.cursor_position = new_pos;
                        if let Some(anchor) = settings.model_id_input.selection_anchor {
                            settings.model_id_input.selection_start = Some(anchor.min(new_pos));
                            settings.model_id_input.selection_end = Some(anchor.max(new_pos));
                        }
                    }
                }
            }
            Some(6) => {
                if app.insight_modal.title_input.is_selecting && app.insight_modal.title_input.contains(app.mouse_pos) {
                    let positions = if app.insight_modal.title_input.glyph_positions.is_empty() {
                        self.compute_glyph_positions(&app.insight_modal.title_input.text, 14.0, 0.0)
                    } else {
                        app.insight_modal.title_input.glyph_positions.clone()
                    };
                    let new_pos = app.insight_modal.title_input.get_cursor_position_from_positions(app.mouse_pos, &positions);
                    app.insight_modal.title_input.cursor_position = new_pos;
                    if let Some(anchor) = app.insight_modal.title_input.selection_anchor {
                        app.insight_modal.title_input.selection_start = Some(anchor.min(new_pos));
                        app.insight_modal.title_input.selection_end = Some(anchor.max(new_pos));
                    }
                }
            }
            Some(7) => {
                if app.insight_modal.text_input.is_selecting && app.insight_modal.text_input.contains(app.mouse_pos) {
                    let positions = if app.insight_modal.text_input.glyph_positions.is_empty() {
                        self.compute_glyph_positions(&app.insight_modal.text_input.text, 14.0, 0.0)
                    } else {
                        app.insight_modal.text_input.glyph_positions.clone()
                    };
                    let new_pos = app.insight_modal.text_input.get_cursor_position_from_positions(app.mouse_pos, &positions);
                    app.insight_modal.text_input.cursor_position = new_pos;
                    if let Some(anchor) = app.insight_modal.text_input.selection_anchor {
                        app.insight_modal.text_input.selection_start = Some(anchor.min(new_pos));
                        app.insight_modal.text_input.selection_end = Some(anchor.max(new_pos));
                    }
                }
            }
            Some(8) => {
                if app.chat_info_dialog.title_input.is_selecting && app.chat_info_dialog.title_input.contains(app.mouse_pos) {
                    let positions = if app.chat_info_dialog.title_input.glyph_positions.is_empty() {
                        self.compute_glyph_positions(&app.chat_info_dialog.title_input.text, 14.0, 0.0)
                    } else {
                        app.chat_info_dialog.title_input.glyph_positions.clone()
                    };
                    let new_pos = app.chat_info_dialog.title_input.get_cursor_position_from_positions(app.mouse_pos, &positions);
                    app.chat_info_dialog.title_input.cursor_position = new_pos;
                    if let Some(anchor) = app.chat_info_dialog.title_input.selection_anchor {
                        app.chat_info_dialog.title_input.selection_start = Some(anchor.min(new_pos));
                        app.chat_info_dialog.title_input.selection_end = Some(anchor.max(new_pos));
                    }
                }
            }
            _ => {}
        }
    }
    
    pub fn set_text_input_cursor_from_mouse(&mut self, input: &mut crate::ui::TextInput, mouse_pos: Vec2) {
        if input.contains(mouse_pos) {
            // Compute glyph positions for accurate cursor positioning based on actual text
            let positions = self.compute_glyph_positions(&input.text, 14.0, 0.0);
            input.set_glyph_positions(positions.clone());
            
            input.cursor_position = input.get_cursor_position_from_positions(mouse_pos, &positions);
            input.cursor_position = input.cursor_position.min(input.text.chars().count());
            input.selection_anchor = Some(input.cursor_position);
            input.is_selecting = true;
            input.clear_selection();
        }
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        if size.width > 0 && size.height > 0 {
            self.config.width = size.width;
            self.config.height = size.height;
            self.viewport_height = size.height as f32;
            self.surface.configure(&self.device, &self.config);
            self.update_projection_matrix(size.width, size.height);
            
            // Clear texture pools on resize (textures have fixed size, need to be recreated)
            self.text_texture_pool.clear();
            self.icon_texture_pool.clear();
            
            // Recreate vello target texture with new size
            self.vello_target_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("vello_target"),
                size: wgpu::Extent3d {
                    width: size.width,
                    height: size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::STORAGE_BINDING 
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            self.vello_target_view = self.vello_target_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Recreate MSAA texture with new size.
            self.msaa_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("msaa_texture"),
                size: wgpu::Extent3d {
                    width: size.width,
                    height: size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: MSAA_SAMPLE_COUNT,
                dimension: wgpu::TextureDimension::D2,
                format: self.config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.msaa_view = self.msaa_texture.create_view(&wgpu::TextureViewDescriptor::default());
            
            // Recreate blit bind group with new texture view
            self.blit_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blit_bind_group"),
                layout: &self.blit_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.vello_target_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.vello_sampler),
                    },
                ],
            });

        }
    }

    /// Blit a Vello texture region to the matching surface rect using `blit_rect.wgsl` (UV window +
    /// viewport), same approach as constellation `text_layer_blit`. Always use this path for queued
    /// text/icons — the full-screen `blit.wgsl` triangle + scissor is not equivalent on all backends
    /// when combined with explicit `set_viewport`.
    fn vello_queue_scene_blit(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        scissor_opt: Option<ScissorRect>,
        texture_view: &wgpu::TextureView,
        dirty_scissor_opt: Option<&ScissorRect>,
    ) {
        let (x, y, width, height) = if let Some(ds) = dirty_scissor_opt {
            let eff = scissor_opt.map(|s| s.intersect(ds)).unwrap_or(*ds);
            if eff.width == 0 || eff.height == 0 {
                return;
            }
            (eff.x, eff.y, eff.width, eff.height)
        } else if let Some(scissor) = scissor_opt {
            (scissor.x, scissor.y, scissor.width, scissor.height)
        } else {
            (0u32, 0u32, self.config.width, self.config.height)
        };
        let x = x.min(self.config.width);
        let y = y.min(self.config.height);
        let width = width.min(self.config.width.saturating_sub(x));
        let height = height.min(self.config.height.saturating_sub(y));
        if width == 0 || height == 0 {
            return;
        }

        let sw = self.config.width as f32;
        let sh = self.config.height as f32;
        let u_min = x as f32 / sw;
        let v_min = y as f32 / sh;
        let u_max = (x + width) as f32 / sw;
        let v_max = (y + height) as f32 / sh;

        let uv_data: [u8; 16] = bytemuck::cast([u_min, v_min, u_max, v_max]);
        let uv_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vello_queue_uv"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&uv_buffer, 0, &uv_data);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vello_queue_rect_blit"),
            layout: &self.blit_rect_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.vello_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uv_buffer.as_entire_binding(),
                },
            ],
        });
        pass.set_pipeline(&self.blit_rect_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_viewport(x as f32, y as f32, width as f32, height as f32, 0.0, 1.0);
        pass.set_scissor_rect(x, y, width, height);
        pass.draw(0..3, 0..1);
    }

    /// Layout size (width × height) for a queued text draw; matches `render_text_to_scene` / Parley cache keys.
    fn text_draw_command_layout_size(&mut self, cmd: &TextDrawCommand) -> Vec2 {
        let c = [cmd.color.x, cmd.color.y, cmd.color.z, cmd.color.w];
        let brush = Brush::Solid(VelloColor::rgba(
            c[0] as f64,
            c[1] as f64,
            c[2] as f64,
            c[3] as f64,
        ));
        let bits = brush_bits_from_rgba_f32(c);
        let key = ParagraphCacheKey::new(&cmd.text, cmd.size, cmd.max_width, bits, false, false);
        let entry = self.paragraph_cache_get_or_insert(key, &cmd.text, cmd.size, cmd.max_width, brush);
        if cmd.max_width.is_some() {
            Vec2::new(entry.layout.width(), entry.content_height)
        } else {
            Vec2::new(entry.first_width, entry.first_height)
        }
    }

    fn log_text_pipeline_after_queue(
        &mut self,
        app: &App,
        main_text: &[TextDrawCommand],
        main_icons: &[IconDrawCommand],
    ) {
        if !app.debug_text_pipeline {
            return;
        }
        let graph_active = app.graph_state.constellation_view_active();
        let tab = app.ui_state.active_tab;
        crate::gfx::debug_text_pipeline_log::append_line(&format!(
            "[debug_text_pipeline] tab={tab:?} constellation_graph_active={graph_active} main_text={} main_icons={} viewport={}x{}",
            main_text.len(),
            main_icons.len(),
            self.config.width,
            self.config.height,
        ));
        if app.debug_text_pipeline_force_full_main_blit {
            crate::gfx::debug_text_pipeline_log::append_line(
                "[debug_text_pipeline] NOTEBOOK_DEBUG_TEXT_FORCE_FULL_MAIN_BLIT active: MainContent text blits use full-frame scissor",
            );
        }
        use std::collections::HashSet;
        let layers = [
            CompositeLayer::MainContent,
            CompositeLayer::ConstellationOverlay,
            CompositeLayer::SidebarChrome,
            CompositeLayer::ComposerChrome,
            CompositeLayer::HudChrome,
            CompositeLayer::Modal,
        ];
        for layer in layers {
            let cmds: Vec<&TextDrawCommand> = main_text.iter().filter(|c| c.layer == layer).collect();
            if cmds.is_empty() {
                crate::gfx::debug_text_pipeline_log::append_line(&format!("  text {layer:?}: 0 commands"));
                continue;
            }
            let none_sc = cmds.iter().filter(|c| c.scissor.is_none()).count();
            let some_sc = cmds.len() - none_sc;
            let distinct: HashSet<ScissorRect> = cmds.iter().filter_map(|c| c.scissor).collect();
            crate::gfx::debug_text_pipeline_log::append_line(&format!(
                "  text {layer:?}: total={} scissor_none={} scissor_some={} distinct_some_rects={}",
                cmds.len(),
                none_sc,
                some_sc,
                distinct.len(),
            ));
        }

        let mut violations = 0usize;
        const MAX_LOG: usize = 40;
        for cmd in main_text.iter() {
            let sr = match cmd.scissor {
                Some(s) => s,
                None => continue,
            };
            let sz = self.text_draw_command_layout_size(cmd);
            let px = cmd.position.x;
            let py = cmd.position.y;
            let pr = px + sz.x;
            let pb = py + sz.y;
            let sx = sr.x as f32;
            let sy = sr.y as f32;
            let srgt = sx + sr.width as f32;
            let sbt = sy + sr.height as f32;
            const EPS: f32 = 0.5;
            if px + EPS < sx
                || py + EPS < sy
                || pr > srgt + EPS
                || pb > sbt + EPS
            {
                if violations < MAX_LOG {
                    let snippet: String = cmd.text.chars().take(48).collect();
                    crate::gfx::debug_text_pipeline_log::append_line(&format!(
                        "[debug_text_pipeline] scissor/bbox MISMATCH layer={:?} pos=({px:.1},{py:.1}) layout=({:.1}x{:.1}) scissor={sr:?} text_prefix={snippet:?}",
                        cmd.layer,
                        sz.x,
                        sz.y,
                    ));
                }
                violations += 1;
            }
        }
        if violations > MAX_LOG {
            crate::gfx::debug_text_pipeline_log::append_line(&format!(
                "[debug_text_pipeline] scissor/bbox mismatches: {} shown, {} total",
                MAX_LOG,
                violations
            ));
        } else if violations > 0 {
            crate::gfx::debug_text_pipeline_log::append_line(&format!(
                "[debug_text_pipeline] scissor/bbox mismatches total={}",
                violations
            ));
        }

        for layer in layers {
            let n = main_icons.iter().filter(|c| c.layer == layer).count();
            if n > 0 {
                let none_sc = main_icons
                    .iter()
                    .filter(|c| c.layer == layer && c.scissor.is_none())
                    .count();
                crate::gfx::debug_text_pipeline_log::append_line(&format!(
                    "  icons {layer:?}: total={n} scissor_none={none_sc}"
                ));
            }
        }

        if main_icons.is_empty() {
            crate::gfx::debug_text_pipeline_log::append_line(
                "  icons: 0 commands in main pass (before modals)",
            );
        }

        let mc_cmds: Vec<&TextDrawCommand> = main_text
            .iter()
            .filter(|c| c.layer == CompositeLayer::MainContent)
            .collect();
        for (i, c) in mc_cmds.iter().take(16).enumerate() {
            let snippet: String = c.text.chars().take(56).collect();
            crate::gfx::debug_text_pipeline_log::append_line(&format!(
                "  MainContent_text[{i}] pos=({:.1},{:.1}) font={} max_w={:?} snippet={snippet:?}",
                c.position.x,
                c.position.y,
                c.size,
                c.max_width,
            ));
        }
    }

    fn log_text_pipeline_batches(&self, app: &App, batches: &[RenderBatch]) {
        if !app.debug_text_pipeline {
            return;
        }
        use std::collections::HashMap;
        let mut verts: HashMap<CompositeLayer, usize> = HashMap::new();
        for b in batches {
            *verts.entry(b.layer).or_insert(0) += b.vertices.len();
        }
        crate::gfx::debug_text_pipeline_log::append_line(&format!(
            "[debug_text_pipeline] quad_vertex_count_by_layer {verts:?}"
        ));

        let fw = self.config.width;
        let fh = self.config.height;
        for b in batches {
            if b.layer == CompositeLayer::Background || b.layer == CompositeLayer::MainContent {
                continue;
            }
            if let Some(s) = b.scissor {
                if s.width >= fw.saturating_sub(2)
                    && s.height >= fh.saturating_sub(2)
                    && s.x <= 1
                    && s.y <= 1
                {
                    crate::gfx::debug_text_pipeline_log::append_line(&format!(
                        "[debug_text_pipeline] overdraw_check: near_fullscreen_quads layer={:?} vertices={}",
                        b.layer,
                        b.vertices.len()
                    ));
                }
            }
        }

        crate::gfx::debug_text_pipeline_log::append_line(&format!(
            "[debug_text_pipeline] surface_format={:?} (intermediate Vello atlases use Rgba8Unorm)",
            self.config.format
        ));
    }

    fn log_text_pipeline_main_vello_groups(&self, app: &App, groups: &[(Option<ScissorRect>, Scene)]) {
        if !app.debug_text_pipeline {
            return;
        }
        crate::gfx::debug_text_pipeline_log::append_line(&format!(
            "[debug_text_pipeline] MainContent Vello scene groups count={}",
            groups.len()
        ));
        for (i, (sc, _)) in groups.iter().enumerate() {
            crate::gfx::debug_text_pipeline_log::append_line(&format!(
                "  vello_group[{i}] scissor={sc:?}"
            ));
        }
    }

    /// Set compositing layer for subsequent quads and queued text/icons. Flushes the current vertex batch when the layer changes.
    pub fn set_composite_layer(&mut self, layer: CompositeLayer) {
        if layer == self.current_composite_layer {
            return;
        }
        self.flush_current_batch();
        self.current_composite_layer = layer;
    }

    fn update_projection_matrix(&mut self, width: u32, height: u32) {
        let width = width as f32;
        let height = height as f32;
        
        let projection = Mat4::orthographic_rh(0.0, width, height, 0.0, -1.0, 1.0);
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&projection.to_cols_array()));
    }

    /// Queue text, capturing the current active scissor (for legacy code)
    pub fn queue_text(&mut self, text: &str, position: Vec2, color: Vec4, size: f32, max_width: Option<f32>) {
        // Capture the current active scissor (if any) when text is queued
        let active_scissor = self.scissor_stack.last().copied();
        self.queue_text_with_scissor(text, position, color, size, active_scissor, max_width);
    }
    
    /// Queue text with an explicit scissor rect (for containers that manage their own clipping)
    /// scissor_rect should be a ScissorRect (already converted to WGPU coordinates)
    pub fn queue_text_with_scissor(
        &mut self,
        text: &str,
        position: Vec2,
        color: Vec4,
        size: f32,
        scissor_rect: Option<ScissorRect>,
        max_width: Option<f32>,
    ) {
        self.text_queue.push(TextDrawCommand {
            text: text.to_string(),
            position,
            color,
            size,
            max_width,
            scissor: scissor_rect,
            layer: self.current_composite_layer,
        });
    }
    
    /// Queue text with a UI Rect scissor (containers can pass their scissor rect directly)
    pub fn queue_text_with_ui_scissor(
        &mut self,
        text: &str,
        position: Vec2,
        color: Vec4,
        size: f32,
        scissor_rect: Option<&Rect>,
        max_width: Option<f32>,
    ) {
        let scissor = scissor_rect.map(|r| ScissorRect::from_rect(r, self.viewport_height));
        self.queue_text_with_scissor(text, position, color, size, scissor, max_width);
    }

    /// Queue plain text with word wrapping to max_width. Returns the y offset of the last line (total height).
    pub fn queue_plain_text_wrapped(&mut self, text: &str, position: Vec2, color: Vec4, size: f32, max_width: f32) -> f32 {
        let key = ParagraphCacheKey::new(text, size, Some(max_width), PARAGRAPH_MEASURE_BRUSH_BITS, false, false);
        let h = self
            .paragraph_cache_get_or_insert(key, text, size, Some(max_width), measure_default_brush())
            .content_height;
        self.queue_text(text, position, color, size, Some(max_width));
        h
    }

    /// Queue icon, capturing the current active scissor
    pub fn queue_icon(&mut self, icon_name: &str, position: Vec2, size: f32, color: Vec4) {
        let active_scissor = self.scissor_stack.last().copied();
        self.queue_icon_with_scissor(icon_name, position, size, color, active_scissor);
    }
    
    /// Queue icon with an explicit scissor rect
    pub fn queue_icon_with_scissor(&mut self, icon_name: &str, position: Vec2, size: f32, color: Vec4, scissor_rect: Option<ScissorRect>) {
        self.icon_queue.push(IconDrawCommand {
            icon_name: icon_name.to_string(),
            position,
            size,
            color,
            scissor: scissor_rect,
            layer: self.current_composite_layer,
        });
    }
    
    /// Queue icon with a UI Rect scissor
    pub fn queue_icon_with_ui_scissor(&mut self, icon_name: &str, position: Vec2, size: f32, color: Vec4, scissor_rect: Option<&Rect>) {
        let scissor = scissor_rect.map(|r| ScissorRect::from_rect(r, self.viewport_height));
        self.queue_icon_with_scissor(icon_name, position, size, color, scissor);
    }

    /// Minimum text layer raster size in pixels to avoid flicker from tiny textures.
    /// Applied uniformly to preserve aspect ratio (no glyph squashing).
    const MIN_TEXT_LAYER_SIZE: u32 = 32;

    /// Get or create a cached text layer for constellation. Returns (width, height) of the layer.
    /// Only creates when cache miss; on hit, returns stored size.
    pub fn get_or_create_text_layer(
        &mut self,
        key: (String, bool, u64, u32, u32),
        markdown: &str,
        max_width: f32,
        font_size: f32,
        text_color: Vec4,
    ) -> (u32, u32) {
        if let Some(entry) = self.text_layer_cache.entries.get(&key) {
            return (entry.width, entry.height);
        }
        if self.text_layer_cache.entries.len() >= self.text_layer_cache.max_entries {
            self.text_layer_cache.entries.clear();
        }
        let mut scene = Scene::new();
        let (w, h) = self.build_markdown_scene(&mut scene, markdown, 0.0, 0.0, max_width, font_size, text_color);
        let desired_w = w.ceil().max(1.0) as u32;
        let desired_h = h.ceil().max(1.0) as u32;

        // Enforce a minimum raster size while keeping aspect ratio so glyphs are not stretched.
        let mut layer_w = desired_w;
        let mut layer_h = desired_h;
        if layer_w < Self::MIN_TEXT_LAYER_SIZE || layer_h < Self::MIN_TEXT_LAYER_SIZE {
            let scale_w = Self::MIN_TEXT_LAYER_SIZE as f32 / layer_w as f32;
            let scale_h = Self::MIN_TEXT_LAYER_SIZE as f32 / layer_h as f32;
            let scale = scale_w.max(scale_h);
            layer_w = ((layer_w as f32) * scale).ceil() as u32;
            layer_h = ((layer_h as f32) * scale).ceil() as u32;
        }
        layer_w = layer_w.min(4096);
        layer_h = layer_h.min(4096);
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("text_layer"),
            size: wgpu::Extent3d { width: layer_w, height: layer_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.vello_renderer
            .render_to_texture(
                &self.device,
                &self.queue,
                &scene,
                &view,
                &vello::RenderParams {
                    base_color: VelloColor::TRANSPARENT,
                    width: layer_w,
                    height: layer_h,
                    antialiasing_method: vello::AaConfig::Area,
                },
            )
            .expect("Failed to render text layer");
        self.text_layer_cache.entries.insert(key, TextLayerEntry {
            texture,
            view,
            width: layer_w,
            height: layer_h,
        });
        (layer_w, layer_h)
    }

    /// Queue drawing a text layer at dest_rect with scroll offset. Call after get_or_create_text_layer.
    /// When `clip_rect` is Some, it is intersected with the top of the scissor stack (nested clips, e.g.
    /// constellation viewport ∩ shard card). When clip_rect is None, the stack top alone is used.
    pub fn draw_text_layer(
        &mut self,
        key: (String, bool, u64, u32, u32),
        dest_rect: (f32, f32, f32, f32),
        scroll_offset: f32,
        clip_rect: Option<&crate::ui::core::Rect>,
    ) {
        if !self.text_layer_cache.entries.contains_key(&key) {
            return;
        }
        let stack_scissor = self.scissor_stack.last().copied();
        let scissor = match (clip_rect, stack_scissor) {
            (Some(clip_r), Some(stack_s)) => {
                let stack_r = Rect::new(
                    stack_s.x as f32,
                    stack_s.y as f32,
                    stack_s.width as f32,
                    stack_s.height as f32,
                );
                clip_r
                    .intersection(&stack_r)
                    .map(|r| ScissorRect::from_rect(&r, self.viewport_height))
            }
            (Some(clip_r), None) => Some(ScissorRect::from_rect(clip_r, self.viewport_height)),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        let scissor = scissor.filter(|s| s.width > 0 && s.height > 0);
        let Some(scissor) = scissor else {
            return;
        };
        self.text_layer_draws.push(TextLayerDraw {
            key,
            dest_rect,
            scroll_offset,
            scissor: Some(scissor),
            layer: self.current_composite_layer,
        });
    }

    pub fn cache_rgba_image(&mut self, cache_key: &str, rgba: &[u8], width: u32, height: u32) {
        if self.image_texture_cache.contains_key(cache_key) {
            return;
        }
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pdf_page_texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * width.max(1)),
                rows_per_image: Some(height.max(1)),
            },
            wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.image_texture_cache.insert(
            cache_key.to_string(),
            CachedImageEntry { texture, view },
        );
    }

    pub fn draw_cached_image(
        &mut self,
        cache_key: &str,
        dest_rect: (f32, f32, f32, f32),
        clip_rect: Option<&crate::ui::core::Rect>,
    ) {
        if !self.image_texture_cache.contains_key(cache_key) {
            return;
        }
        let stack_scissor = self.scissor_stack.last().copied();
        let scissor = match (clip_rect, stack_scissor) {
            (Some(clip_r), Some(stack_s)) => {
                let stack_r = Rect::new(
                    stack_s.x as f32,
                    stack_s.y as f32,
                    stack_s.width as f32,
                    stack_s.height as f32,
                );
                clip_r
                    .intersection(&stack_r)
                    .map(|r| ScissorRect::from_rect(&r, self.viewport_height))
            }
            (Some(clip_r), None) => Some(ScissorRect::from_rect(clip_r, self.viewport_height)),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        self.image_draws.push(ImageDrawCommand {
            cache_key: cache_key.to_string(),
            dest_rect,
            scissor,
            layer: self.current_composite_layer,
        });
    }
    
    /// Push a scissor rect onto the stack for clipping
    /// All subsequent rendering will be clipped to this rectangle (and any parent scissors)
    pub fn push_scissor(&mut self, rect: &crate::ui::core::Rect) {
        // Flush current batch before changing scissor state
        self.flush_current_batch();
        
        let scissor = ScissorRect::from_rect(rect, self.viewport_height);
        
        // If there's already a scissor active, intersect with it (for nested clipping)
        let final_scissor = if let Some(parent) = self.scissor_stack.last() {
            parent.intersect(&scissor)
        } else {
            scissor
        };
        
        self.scissor_stack.push(final_scissor);
    }
    
    /// Pop the most recent scissor rect from the stack
    pub fn pop_scissor(&mut self) {
        // Flush current batch before changing scissor state
        self.flush_current_batch();
        
        self.scissor_stack.pop();
    }
    
    /// Flush the current batch of vertices into render_batches
    fn flush_current_batch(&mut self) {
        if !self.current_batch_vertices.is_empty() {
            self.render_batches.push(RenderBatch {
                vertices: std::mem::take(&mut self.current_batch_vertices),
                scissor: self.current_batch_clip_rect,
                layer: self.current_composite_layer,
            });
            // Don't clear current_batch_clip_rect - keep it so we can merge with next batch if same scissor
            // It will be overwritten when a different scissor is set
        }
    }
    
    /// Merge batches with identical scissor rects to reduce draw calls
    /// Preserves order by processing batches sequentially and merging adjacent compatible batches
    fn merge_compatible_batches(&mut self) {
        if self.render_batches.len() <= 1 {
            return; // Nothing to merge
        }
        
        // Merge adjacent batches with the same scissor rect (preserves rendering order)
        let mut merged: Vec<RenderBatch> = Vec::new();
        
        for batch in self.render_batches.drain(..) {
            if let Some(last_batch) = merged.last_mut() {
                // If this batch has the same scissor and layer as the last one, merge them
                if last_batch.scissor == batch.scissor && last_batch.layer == batch.layer {
                    last_batch.vertices.extend(batch.vertices);
                    continue;
                }
            }
            // Otherwise, add as a new batch
            merged.push(batch);
        }
        
        self.render_batches = merged;
    }
    
    /// Get or create a texture from the text texture pool
    fn get_text_texture(&mut self) -> (wgpu::Texture, wgpu::TextureView, wgpu::BindGroup) {
        let (texture, view) = if let Some((texture, view)) = self.text_texture_pool.pop() {
            // Reuse existing texture and view
            (texture, view)
        } else {
            // Create new texture if pool is exhausted
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("vello_texture_pooled"),
                size: wgpu::Extent3d {
                    width: self.config.width.max(1),
                    height: self.config.height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::STORAGE_BINDING 
                    | wgpu::TextureUsages::TEXTURE_BINDING 
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        
        // Always recreate bind group (they're lightweight and tied to specific views)
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit_bind_group_pooled"),
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.vello_sampler),
                },
            ],
        });
        (texture, view, bind_group)
    }
    
    /// Return a texture to the text texture pool
    fn return_text_texture(&mut self, texture: wgpu::Texture, view: wgpu::TextureView, _bind_group: wgpu::BindGroup) {
        if self.text_texture_pool.len() < self.max_texture_pool_size {
            self.text_texture_pool.push((texture, view));
        }
        // If pool is full, texture is dropped (freed)
    }
    
    /// Get or create a texture from the icon texture pool
    fn get_icon_texture(&mut self) -> (wgpu::Texture, wgpu::TextureView, wgpu::BindGroup) {
        let (texture, view) = if let Some((texture, view)) = self.icon_texture_pool.pop() {
            // Reuse existing texture and view
            (texture, view)
        } else {
            // Create new texture if pool is exhausted
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("vello_icon_texture_pooled"),
                size: wgpu::Extent3d {
                    width: self.config.width.max(1),
                    height: self.config.height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::STORAGE_BINDING 
                    | wgpu::TextureUsages::TEXTURE_BINDING 
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        
        // Always recreate bind group (they're lightweight and tied to specific views)
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit_icon_bind_group_pooled"),
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.vello_sampler),
                },
            ],
        });
        (texture, view, bind_group)
    }
    
    /// Return a texture to the icon texture pool
    fn return_icon_texture(&mut self, texture: wgpu::Texture, view: wgpu::TextureView, _bind_group: wgpu::BindGroup) {
        if self.icon_texture_pool.len() < self.max_texture_pool_size {
            self.icon_texture_pool.push((texture, view));
        }
        // If pool is full, texture is dropped (freed)
    }
    
    /// Add vertices to the current batch with optional clipping
    /// This is used by components to add their geometry to be rendered
    /// If clip_rect is provided and differs from the current batch's clip_rect, the current batch is flushed first
    pub fn add_vertices(&mut self, vertices: &[Vertex], clip_rect: Option<&Rect>) {
        let scissor = clip_rect.map(|r| ScissorRect::from_rect(r, self.viewport_height));
        
        // If clip_rect changed, flush current batch
        if scissor != self.current_batch_clip_rect {
            self.flush_current_batch();
            self.current_batch_clip_rect = scissor;
        }
        
        self.current_batch_vertices.extend_from_slice(vertices);
    }
    
    /// Add a quad directly without creating a temporary vertex array
    /// This is more efficient than add_vertices(&quad.to_vertices(), ...)
    pub fn add_quad(&mut self, quad: &crate::gfx::types::Quad, clip_rect: Option<&Rect>) {
        // Use explicit clip_rect if provided, otherwise use active scissor from stack
        let scissor = if let Some(rect) = clip_rect {
            Some(ScissorRect::from_rect(rect, self.viewport_height))
        } else {
            // If no explicit clip_rect, use active scissor from stack (if any)
            self.scissor_stack.last().copied()
        };
        
        // If scissor changed, flush current batch
        if scissor != self.current_batch_clip_rect {
            self.flush_current_batch();
            self.current_batch_clip_rect = scissor;
        }
        
        // Push vertices directly to avoid temporary allocation
        quad.push_vertices_to(&mut self.current_batch_vertices);
    }

    /// End the current quad batch so the next geometry starts a new batch (e.g. backdrop, then modal shell).
    pub fn end_batch(&mut self) {
        self.flush_current_batch();
    }

    /// Queue an analytical drop shadow for a rounded rectangle.
    ///
    /// Emits a single quad inflated by `3 * sigma + spread` around `rect + spec.offset`, with
    /// the `bubble == 3.0` sentinel so [`ui_shader.wgsl`](../shaders/ui_shader.wgsl) evaluates
    /// a gaussian-blurred rounded box (erf of SDF) per pixel. One quad, O(1) per pixel; no extra
    /// render passes or textures. Routed through the existing scissor / composite-layer batching
    /// so the shadow inherits the caller's active clip and layer.
    ///
    /// Call this *before* the component renders its own geometry so the shadow lands behind it.
    pub fn queue_shadow(
        &mut self,
        rect: &crate::ui::core::Rect,
        corner_radius: f32,
        spec: &crate::ui::shadow::ShadowSpec,
    ) {
        if rect.width <= 0.0 || rect.height <= 0.0 || spec.sigma <= 0.0 || spec.color.w <= 0.0 {
            return;
        }

        // Source shape (dilated by spread, offset to shadow position).
        let src_x = rect.x + spec.offset.x - spec.spread;
        let src_y = rect.y + spec.offset.y - spec.spread;
        let src_w = rect.width + spec.spread * 2.0;
        let src_h = rect.height + spec.spread * 2.0;
        let src_r = (corner_radius + spec.spread).max(0.0);

        // Visible falloff extends ~3 sigma beyond the source shape.
        let pad = spec.sigma * 3.0;
        let qx = src_x - pad;
        let qy = src_y - pad;
        let qw = src_w + pad * 2.0;
        let qh = src_h + pad * 2.0;

        let scissor = self.scissor_stack.last().copied();
        if scissor != self.current_batch_clip_rect {
            self.flush_current_batch();
            self.current_batch_clip_rect = scissor;
        }

        let c = [spec.color.x, spec.color.y, spec.color.z, spec.color.w];
        // Source rect (quad_pos / quad_size) is the *shape*; the rendered quad just covers its falloff.
        let quad_pos = [src_x, src_y];
        let quad_size = [src_w, src_h];
        let sigma = spec.sigma;

        let vtx = |px: f32, py: f32| Vertex {
            position: [px, py],
            color: c,
            quad_pos,
            quad_size,
            corner_radius: src_r,
            bubble: 3.0,
            slider: 0.0,
            shadow_sigma: sigma,
        };

        let v = &mut self.current_batch_vertices;
        v.push(vtx(qx, qy));
        v.push(vtx(qx + qw, qy));
        v.push(vtx(qx, qy + qh));
        v.push(vtx(qx + qw, qy));
        v.push(vtx(qx + qw, qy + qh));
        v.push(vtx(qx, qy + qh));
    }

    /// Shared helper for the three *interior* lighting primitives (inner shadow, border
    /// highlight, surface highlight). All paint **inside** the component via an SDF
    /// `inside_mask` in the shader, so the quad covers the component bounds exactly — no
    /// inflation. Picks up the active scissor and composite layer like [`queue_shadow`].
    ///
    /// - `bubble` — shader sentinel selecting the branch (4.0 / 5.0 / 6.0).
    /// - `sigma` — fed into `shadow_sigma`; meaning depends on the branch (blur, feather, AA).
    /// - `scalar` — fed into `slider`; meaning depends on the branch (offset size, border
    ///   width, curve exponent).
    fn queue_interior_effect(
        &mut self,
        rect: &crate::ui::core::Rect,
        corner_radius: f32,
        color: glam::Vec4,
        bubble: f32,
        sigma: f32,
        scalar: f32,
    ) {
        if rect.width <= 0.0 || rect.height <= 0.0 || color.w <= 0.0 {
            return;
        }

        let scissor = self.scissor_stack.last().copied();
        if scissor != self.current_batch_clip_rect {
            self.flush_current_batch();
            self.current_batch_clip_rect = scissor;
        }

        let c = [color.x, color.y, color.z, color.w];
        let quad_pos = [rect.x, rect.y];
        let quad_size = [rect.width, rect.height];
        let r = corner_radius.max(0.0);

        let vtx = |px: f32, py: f32| Vertex {
            position: [px, py],
            color: c,
            quad_pos,
            quad_size,
            corner_radius: r,
            bubble,
            slider: scalar,
            shadow_sigma: sigma,
        };

        let v = &mut self.current_batch_vertices;
        v.push(vtx(rect.x, rect.y));
        v.push(vtx(rect.x + rect.width, rect.y));
        v.push(vtx(rect.x, rect.y + rect.height));
        v.push(vtx(rect.x + rect.width, rect.y));
        v.push(vtx(rect.x + rect.width, rect.y + rect.height));
        v.push(vtx(rect.x, rect.y + rect.height));
    }

    /// Queue an **inner drop shadow** on the top-left interior of the component
    /// (see [`crate::ui::shadow::InnerShadowSpec`]). Call this *after* the component's fill
    /// so the shadow renders on top — the shader clips it to the component's rounded shape.
    /// Uses the `bubble == 4.0` branch of `ui_shader.wgsl`.
    pub fn queue_inner_shadow(
        &mut self,
        rect: &crate::ui::core::Rect,
        corner_radius: f32,
        spec: &crate::ui::shadow::InnerShadowSpec,
    ) {
        self.queue_interior_effect(
            rect,
            corner_radius,
            spec.color,
            4.0,
            spec.sigma,
            spec.offset_size,
        );
    }

    /// Queue a **specular border highlight** along the top-left inner edge of the component
    /// (see [`crate::ui::shadow::BorderHighlightSpec`]). Call this *after* the component's
    /// fill so the rim renders on top. Uses the `bubble == 5.0` branch of `ui_shader.wgsl`.
    pub fn queue_border_highlight(
        &mut self,
        rect: &crate::ui::core::Rect,
        corner_radius: f32,
        spec: &crate::ui::shadow::BorderHighlightSpec,
    ) {
        self.queue_interior_effect(
            rect,
            corner_radius,
            spec.color,
            5.0,
            spec.sigma,
            spec.width,
        );
    }

    /// Queue a **specular surface highlight** — a diagonal sheen across the interior, brightest
    /// at the top-left (see [`crate::ui::shadow::SurfaceHighlightSpec`]). Call this *after*
    /// the component's fill so the sheen renders on top. Uses the `bubble == 6.0` branch of
    /// `ui_shader.wgsl`.
    pub fn queue_surface_highlight(
        &mut self,
        rect: &crate::ui::core::Rect,
        corner_radius: f32,
        spec: &crate::ui::shadow::SurfaceHighlightSpec,
    ) {
        self.queue_interior_effect(
            rect,
            corner_radius,
            spec.color,
            6.0,
            spec.sigma,
            spec.curve,
        );
    }

    /// Set visible constellation node count for debug_text_stats instrumentation.
    pub fn set_debug_constellation_visible_nodes(&mut self, n: usize) {
        self.debug_constellation_visible_nodes = Some(n);
    }

    /// Begin a new render frame - clears all batches
    fn begin_frame(&mut self) {
        self.render_batches.clear();
        self.current_batch_vertices.clear();
        self.current_batch_clip_rect = None;
        self.scissor_stack.clear();
        self.text_queue.clear();
        self.icon_queue.clear();
        self.text_layer_draws.clear();
        self.image_draws.clear();
        self.current_composite_layer = CompositeLayer::MainContent;
        self.debug_constellation_visible_nodes = None;
        
        // Clear component validation tracking for new frame
        if self.component_validation_enabled {
            self.rendered_components.clear();
            self.component_hierarchy.clear();
            self.skipped_components.clear();
            // Keep warning tracking across frames to avoid spam
            // Only clear if we have too many (prevent memory leak)
            if self.duplicate_warnings_shown.len() > 100 {
                self.duplicate_warnings_shown.clear();
            }
            if self.orphaned_warnings_shown.len() > 100 {
                self.orphaned_warnings_shown.clear();
            }
            self.current_parent_stack.clear();
        }
    }
    
    /// Validate that a component is a Renderable and part of the hierarchy
    /// This is called when components are rendered to ensure architectural compliance
    /// Returns false if the component should be skipped (orphaned or duplicate)
    #[cfg(debug_assertions)]
    pub fn validate_component(&mut self, component_id: &str, parent_id: Option<&str>, component_type: &str) -> bool {
        if !self.component_validation_enabled {
            return true; // Allow rendering if validation is disabled
        }
        
        // Use current parent from stack if parent_id is None
        let actual_parent = parent_id.or_else(|| self.current_parent_stack.last().map(|s| s.as_str()));
        
        // Check if component is already rendered (potential duplicate rendering)
        let is_duplicate = self.rendered_components.contains(component_id);
        if is_duplicate {
            // Only show warning once per component ID to avoid spam
            if !self.duplicate_warnings_shown.contains(component_id) {
                eprintln!("⚠️  SKIPPING: Component '{}' ({}) rendered multiple times in the same frame. Skipping duplicate render.", component_id, component_type);
                self.duplicate_warnings_shown.insert(component_id.to_string());
            }
            self.skipped_components.insert(component_id.to_string());
            return false; // Skip rendering
        }
        
        // Track component in hierarchy
        self.rendered_components.insert(component_id.to_string());
        if let Some(parent) = actual_parent {
            // Check if parent exists in hierarchy or is in the current parent stack
            // (parents in the stack will be validated, so they're valid even if not yet in hierarchy)
            let parent_exists = parent == "root" 
                || self.component_hierarchy.contains_key(parent)
                || self.current_parent_stack.contains(&parent.to_string());
            
            if !parent_exists {
                // Only show warning once per component ID to avoid spam
                if !self.orphaned_warnings_shown.contains(component_id) {
                    eprintln!("⚠️  SKIPPING: Component '{}' ({}) has parent '{}' that doesn't exist in hierarchy. Skipping orphaned component.", component_id, component_type, parent);
                    eprintln!("   DEBUG: Hierarchy keys: {:?}", self.component_hierarchy.keys().take(20).collect::<Vec<_>>());
                    eprintln!("   DEBUG: Parent stack: {:?}", self.current_parent_stack);
                    // For Text components, try to identify which text it is from the component ID
                    if component_id.starts_with("text_") {
                        eprintln!("   DEBUG: This is a Text component. Component ID format: text_<hash>_<x>_<y>");
                    }
                    self.orphaned_warnings_shown.insert(component_id.to_string());
                }
                self.skipped_components.insert(component_id.to_string());
                self.component_hierarchy.insert(component_id.to_string(), Some(parent.to_string()));
                return false; // Skip rendering
            }
            self.component_hierarchy.insert(component_id.to_string(), Some(parent.to_string()));
        } else {
            // Component has no parent - check if it's the root
            // Only "root" can exist without a parent
            if component_id != "root" {
                // Only show warning once per component ID to avoid spam
                if !self.orphaned_warnings_shown.contains(component_id) {
                    eprintln!("⚠️  SKIPPING: Component '{}' ({}) has no parent and is not the root. Skipping orphaned component.", component_id, component_type);
                    self.orphaned_warnings_shown.insert(component_id.to_string());
                }
                self.skipped_components.insert(component_id.to_string());
                self.component_hierarchy.insert(component_id.to_string(), None);
                return false; // Skip rendering
            }
            self.component_hierarchy.insert(component_id.to_string(), None);
        }
        
        true // Allow rendering
    }
    
    /// Validate that a component is a Renderable and part of the hierarchy (release build - no-op)
    #[cfg(not(debug_assertions))]
    #[inline(always)]
    pub fn validate_component(&mut self, _component_id: &str, _parent_id: Option<&str>, _component_type: &str) -> bool {
        true // Always allow rendering in release builds
    }
    
    /// Check if a component should be skipped (orphaned or duplicate)
    #[cfg(debug_assertions)]
    pub fn should_skip_component(&self, component_id: &str) -> bool {
        if !self.component_validation_enabled {
            return false;
        }
        self.skipped_components.contains(component_id)
    }
    
    /// Check if a component should be skipped (release build - no-op)
    #[cfg(not(debug_assertions))]
    #[inline(always)]
    pub fn should_skip_component(&self, _component_id: &str) -> bool {
        false // Never skip in release builds
    }
    
    /// Check if a component is in the hierarchy (for debugging)
    pub fn is_component_in_hierarchy(&self, component_id: &str) -> bool {
        self.component_hierarchy.contains_key(component_id)
    }
    
    /// Push a parent component onto the stack (for nested rendering)
    pub fn push_parent(&mut self, parent_id: String) {
        if self.component_validation_enabled {
            self.current_parent_stack.push(parent_id);
        }
    }
    
    /// Pop a parent component from the stack
    pub fn pop_parent(&mut self) {
        if self.component_validation_enabled {
            self.current_parent_stack.pop();
        }
    }
    
    /// Validate component hierarchy at end of frame
    /// Ensures all rendered components are connected to the root
    #[cfg(debug_assertions)]
    pub fn validate_hierarchy(&mut self) {
        if !self.component_validation_enabled {
            return;
        }
        
        // Find root components (those with no parent or parent is "app"/"root")
        let root_components: Vec<String> = self.component_hierarchy.iter()
            .filter_map(|(id, parent)| {
                match parent {
                    None => Some(id.clone()),
                    Some(p) if p == "root" => Some(id.clone()),
                    _ => None,
                }
            })
            .collect();
        
        // Check for orphaned components (not connected to root)
        let mut orphaned = Vec::new();
        for (component_id, parent_id) in &self.component_hierarchy {
            if let Some(parent) = parent_id {
                // Check if parent exists in hierarchy
                // Only "root" can exist without being in hierarchy (it's the base)
                if !self.component_hierarchy.contains_key(parent) && parent != "root" {
                    orphaned.push(component_id.clone());
                }
            } else if !root_components.contains(component_id) && component_id != "root" {
                // Component has no parent and is not a root component
                orphaned.push(component_id.clone());
            }
        }
        
        if !orphaned.is_empty() {
            // Only warn about orphaned components we haven't warned about before
            let new_orphaned: Vec<String> = orphaned.iter()
                .filter(|id| !self.orphaned_warnings_shown.contains(*id))
                .cloned()
                .collect();
            
            if !new_orphaned.is_empty() {
                eprintln!("⚠️  SKIPPING: Found {} orphaned component(s) not connected to the root hierarchy: {:?}", new_orphaned.len(), new_orphaned);
                eprintln!("   These components will be skipped in future renders.");
                // Mark these as warned about
                for orphan_id in &new_orphaned {
                    self.orphaned_warnings_shown.insert(orphan_id.clone());
                }
            }
            
            // Mark orphaned components to be skipped (even if we've already warned)
            for orphan_id in &orphaned {
                self.skipped_components.insert(orphan_id.clone());
            }
        }
    }
    
    /// Validate component hierarchy at end of frame (release build - no-op)
    #[cfg(not(debug_assertions))]
    #[inline(always)]
    pub fn validate_hierarchy(&mut self) {
        // No-op in release builds
    }
    
    /// Get the current active scissor rect (if any)
    pub fn current_scissor(&self) -> Option<&ScissorRect> {
        self.scissor_stack.last()
    }
    
    /// Clear all scissor rects
    pub fn clear_scissors(&mut self) {
        self.scissor_stack.clear();
    }

    /// Compute glyph positions for text using parley
    /// Returns a vector of x positions, one for each character boundary (including start and end)
    /// positions[0] = start position, positions[i] = position after character i-1
    /// Results are cached to avoid redundant Parley calculations
    pub fn compute_glyph_positions(&mut self, text: &str, size: f32, start_x: f32) -> Vec<f32> {
        let key = ParagraphCacheKey::new(text, size, None, PARAGRAPH_MEASURE_BRUSH_BITS, false, false);
        let entry = self.paragraph_cache_get_or_insert(
            key,
            text,
            size,
            None,
            measure_default_brush(),
        );
        let mut positions: Vec<f32> = entry
            .glyph_x_unbounded
            .iter()
            .map(|p| p + start_x)
            .collect();
        while positions.len() < text.chars().count() + 1 {
            positions.push(*positions.last().unwrap_or(&start_x));
        }
        positions
    }

    /// Compute glyph positions with line wrapping. Returns (line_height, per-line x positions).
    /// Each inner vec has length = num_chars_in_line + 1 (positions at character boundaries).
    /// Phase 4: Layout cached by (text, size, max_width); only content changes trigger Parley. Scroll reuses cache.
    pub fn compute_glyph_positions_wrapped(
        &mut self,
        text: &str,
        size: f32,
        start_x: f32,
        max_width: f32,
    ) -> (f32, Vec<Vec<f32>>) {
        let line_height = size * crate::ui::style::font_size::LINE_HEIGHT_RATIO;
        if text.is_empty() {
            return (line_height, vec![vec![start_x]]);
        }
        let key = ParagraphCacheKey::new(text, size, Some(max_width), PARAGRAPH_MEASURE_BRUSH_BITS, false, false);
        let entry = self.paragraph_cache_get_or_insert(
            key,
            text,
            size,
            Some(max_width),
            measure_default_brush(),
        );
        let (lh, rel_lines) = entry
            .wrapped_glyph_lines
            .clone()
            .expect("wrapped_glyph_lines populated when max_width is set");
        let lines: Vec<Vec<f32>> = rel_lines
            .iter()
            .map(|line| line.iter().map(|&p| p + start_x).collect())
            .collect();
        (lh, lines)
    }

    /// Top-left of the cursor before character index `cursor_char_index` in wrapped layout (`start_y` = first line top).
    pub fn cursor_xy_for_wrapped_cursor_index(
        &mut self,
        text: &str,
        size: f32,
        start_x: f32,
        start_y: f32,
        max_width: f32,
        cursor_char_index: usize,
    ) -> (f32, f32) {
        let (line_height, lines) = self.compute_glyph_positions_wrapped(text, size, start_x, max_width);
        if text.is_empty() {
            return (start_x, start_y);
        }
        let text_len = text.chars().count();
        let idx = cursor_char_index.min(text_len);
        let mut offset = 0usize;
        for (line_idx, line_positions) in lines.iter().enumerate() {
            let n_chars = line_positions.len().saturating_sub(1);
            let line_start = offset;
            if idx <= line_start + n_chars {
                let local = idx - line_start;
                let local = local.min(line_positions.len().saturating_sub(1));
                let x = line_positions[local];
                let y = start_y + line_idx as f32 * line_height;
                return (x, y);
            }
            offset += n_chars;
        }
        let last_line = lines.len().saturating_sub(1);
        let line_positions = &lines[last_line];
        let x = *line_positions.last().unwrap_or(&start_x);
        let y = start_y + last_line as f32 * line_height;
        (x, y)
    }

    /// Map a screen position to (block_id, character_index) for the notepad editor using glyph-based layout.
    pub fn compute_notepad_cursor_from_pos(
        &mut self,
        editor: &crate::stylus::StylusEditor,
        pos: Vec2,
    ) -> Option<(String, usize)> {
        use crate::stylus::renderer::StylusRenderer;
        const PADDING: f32 = 10.0;
        const BLOCK_SPACING: f32 = 8.0;
        let mut y_offset = editor.position.y + PADDING - editor.scroll_offset;
        let max_width = (editor.size.x - PADDING * 2.0).max(1.0);
        for block in &editor.document.blocks {
            let block_height = StylusRenderer::get_block_height_static(block, max_width);
            if pos.y >= y_offset && pos.y < y_offset + block_height {
                if let Some(text) = block.content.get_text() {
                    let font_size = StylusRenderer::get_font_size_static(&block.block_type);
                    let block_x = editor.position.x + PADDING;
                    let (line_height, lines) = self.compute_glyph_positions_wrapped(
                        text,
                        font_size,
                        block_x,
                        max_width,
                    );
                    let line_idx = ((pos.y - y_offset) / line_height).floor() as usize;
                    let line_idx = line_idx.min(lines.len().saturating_sub(1));
                    let positions = &lines[line_idx];
                    let mut char_in_line = 0usize;
                    for (i, &px) in positions.iter().enumerate() {
                        if i == 0 {
                            continue;
                        }
                        let mid = (positions[i - 1] + px) / 2.0;
                        if pos.x <= mid {
                            char_in_line = i - 1;
                            break;
                        }
                        char_in_line = i;
                    }
                    if positions.len() > 1 && pos.x > *positions.last().unwrap() {
                        char_in_line = positions.len() - 1;
                    }
                    let total_char: usize = lines.iter().take(line_idx).map(|p| p.len().saturating_sub(1)).sum();
                    let cursor_pos = (total_char + char_in_line).min(text.len());
                    return Some((block.id.clone(), cursor_pos));
                }
                return Some((block.id.clone(), 0));
            }
            y_offset += block_height + BLOCK_SPACING;
        }
        None
    }

    pub fn measure_text(&mut self, text: &str, size: f32) -> Vec2 {
        if text.is_empty() {
            return Vec2::new(0.0, size * crate::ui::style::font_size::LINE_HEIGHT_RATIO);
        }
        let key = ParagraphCacheKey::new(text, size, None, PARAGRAPH_MEASURE_BRUSH_BITS, false, false);
        let entry = self.paragraph_cache_get_or_insert(
            key,
            text,
            size,
            None,
            measure_default_brush(),
        );
        Vec2::new(entry.first_width, entry.first_height)
    }

    /// Get accurate text metrics using Parley layout
    /// Returns (width, height, baseline_from_top)
    /// Height includes ascent + descent, baseline is distance from top to baseline
    pub fn measure_text_accurate(&mut self, text: &str, size: f32) -> (f32, f32, f32) {
        if text.is_empty() {
            return (
                0.0,
                size * crate::ui::style::font_size::LINE_HEIGHT_RATIO,
                size * 0.75,
            );
        }
        let key = ParagraphCacheKey::new(text, size, None, PARAGRAPH_MEASURE_BRUSH_BITS, false, false);
        let entry = self.paragraph_cache_get_or_insert(
            key,
            text,
            size,
            None,
            measure_default_brush(),
        );
        (entry.first_width, entry.first_height, entry.first_baseline)
    }

    fn text_line_height(&self, size: f32) -> f32 {
        size * crate::ui::style::font_size::LINE_HEIGHT_RATIO
    }

    fn queue_markdown_text(&mut self, markdown: &str, position: Vec2, base_color: Vec4, size: f32, max_width: f32) -> f32 {
        let code_color = crate::ui::style::markdown::CODE_FOREGROUND();
        // Parse markdown and render with appropriate styling
        let parser = Parser::new(markdown);
        let mut current_x = position.x;
        let mut current_y = position.y;
        let line_height = self.text_line_height(size);
        let mut current_line_width = 0.0;
        let mut bold = false;
        let mut italic = false;
        let mut code = false;
        let mut current_text = String::new();

        for event in parser {
            match event {
                Event::Start(Tag::Strong) => {
                    // Flush current text before changing style
                    if !current_text.is_empty() {
                        let color = if code { code_color } else { base_color };
                        let scale = if code { size * 0.9 } else { size };
                        self.queue_text(&current_text, Vec2::new(current_x, current_y), color, scale, None);
                        let text_width = self.segment_width_unbounded_queue(&current_text, scale, color);
                        current_x += text_width;
                        current_line_width += text_width;
                        current_text.clear();
                    }
                    bold = true;
                }
                Event::End(pulldown_cmark::TagEnd::Strong) => {
                    // Render bold text
                    if !current_text.is_empty() {
                        let color = base_color;
                        let scale = size * 1.1; // Slightly larger for bold effect
                        self.queue_text(&current_text, Vec2::new(current_x, current_y), color, scale, None);
                        let text_width = self.segment_width_unbounded_queue(&current_text, scale, color);
                        current_x += text_width;
                        current_line_width += text_width;
                        current_text.clear();
                    }
                    bold = false;
                }
                Event::Start(Tag::Emphasis) => {
                    if !current_text.is_empty() {
                        let color = if code { code_color } else { base_color };
                        let scale = if code { size * 0.9 } else { size };
                        self.queue_text(&current_text, Vec2::new(current_x, current_y), color, scale, None);
                        let text_width = self.segment_width_unbounded_queue(&current_text, scale, color);
                        current_x += text_width;
                        current_line_width += text_width;
                        current_text.clear();
                    }
                    italic = true;
                }
                Event::End(pulldown_cmark::TagEnd::Emphasis) => {
                    if !current_text.is_empty() {
                        let color = base_color;
                        let scale = size * 0.95; // Slightly smaller for italic
                        self.queue_text(&current_text, Vec2::new(current_x, current_y), color, scale, None);
                        let text_width = self.segment_width_unbounded_queue(&current_text, scale, color);
                        current_x += text_width;
                        current_line_width += text_width;
                        current_text.clear();
                    }
                    italic = false;
                }
                Event::Start(Tag::CodeBlock(_)) => {
                    if !current_text.is_empty() {
                        let color = base_color;
                        let scale = size;
                        self.queue_text(&current_text, Vec2::new(current_x, current_y), color, scale, None);
                        let text_width = self.segment_width_unbounded_queue(&current_text, scale, color);
                        current_x += text_width;
                        current_line_width += text_width;
                        current_text.clear();
                    }
                    code = true;
                }
                Event::End(pulldown_cmark::TagEnd::CodeBlock) => {
                    if !current_text.is_empty() {
                        let color = code_color;
                        let scale = size * 0.9;
                        self.queue_text(&current_text, Vec2::new(current_x, current_y), color, scale, None);
                        let text_width = self.segment_width_unbounded_queue(&current_text, scale, color);
                        current_x += text_width;
                        current_line_width += text_width;
                        current_text.clear();
                    }
                    code = false;
                }
                // Inline code is handled differently - it's just text with backticks
                // We'll handle it by detecting backticks in the text itself
                Event::Text(text) => {
                    // Handle word wrapping for text
                    let words: Vec<&str> = text.split_whitespace().collect();
                    for word in words {
                        let word_with_space = if current_text.is_empty() {
                            word.to_string()
                        } else {
                            format!(" {}", word)
                        };
                        
                        let color = if code { code_color } else { base_color };
                        let scale = if code { size * 0.9 } else if bold { size * 1.1 } else if italic { size * 0.95 } else { size };
                        
                        let test_text = format!("{}{}", current_text, word_with_space);
                        let line_used = current_x - position.x;
                        let rem = (max_width - line_used).max(1.0);
                        let cand_key = ParagraphCacheKey::new(
                            &test_text,
                            scale,
                            Some(rem),
                            PARAGRAPH_MEASURE_BRUSH_BITS,
                            bold,
                            italic,
                        );
                        let n_lines = self
                            .paragraph_cache_get_or_insert(
                                cand_key,
                                &test_text,
                                scale,
                                Some(rem),
                                measure_default_brush(),
                            )
                            .layout
                            .len();

                        if n_lines > 1 && !current_text.is_empty() {
                            // Render current line and wrap
                            let flush_color = if code { code_color } else { base_color };
                            let flush_scale = if code { size * 0.9 } else if bold { size * 1.1 } else if italic { size * 0.95 } else { size };
                            self.queue_text(&current_text, Vec2::new(position.x, current_y), flush_color, flush_scale, None);
                            current_y += line_height + 2.0;
                            current_x = position.x;
                            current_line_width = 0.0;
                            current_text = word.to_string();
                        } else {
                            current_text = test_text;
                        }
                    }
                }
                Event::SoftBreak | Event::HardBreak => {
                    // Render current text and move to next line
                    if !current_text.is_empty() {
                        let color = if code { code_color } else { base_color };
                        let scale = if code { size * 0.9 } else if bold { size * 1.1 } else if italic { size * 0.95 } else { size };
                        self.queue_text(&current_text, Vec2::new(position.x, current_y), color, scale, None);
                        current_text.clear();
                    }
                    current_y += line_height + 2.0;
                    current_x = position.x;
                    current_line_width = 0.0;
                }
                _ => {}
            }
        }

        // Render any remaining text
        if !current_text.is_empty() {
            let color = if code { code_color } else { base_color };
            let scale = if code { size * 0.9 } else if bold { size * 1.1 } else if italic { size * 0.95 } else { size };
            self.queue_text(&current_text, Vec2::new(current_x, current_y), color, scale, None);
        }

        current_y + line_height - position.y
    }

    pub fn render(&mut self, app: &mut App) -> anyhow::Result<()> {
        crate::ui::style::install_theme_for_id(&app.settings_state.theme);

        // Process deferred notepad click for glyph-based cursor placement
        if let Some(pos) = app.pending_notepad_click.take() {
            if let Some(ref mut notepad) = app.notepad_window {
                if let Some((block_id, char_idx)) = self.compute_notepad_cursor_from_pos(&notepad.editor, pos) {
                    notepad.editor.set_cursor_from_character_index(&block_id, char_idx);
                }
            }
        }

        // Begin new frame - clear batches and state
        self.begin_frame();

        // Write fragment uniforms (time, scroll_velocity, cursor, slider_velocity) at offset 64
        let time = self.start_time.elapsed().as_secs_f32();
        let scroll_velocity = app.chat_window.as_ref()
            .filter(|_| app.ui_state.active_tab == crate::ui::tab_bar::Tab::Chat)
            .map(|c| c.message_list.scroll_velocity)
            .unwrap_or(0.0);
        let tab_width = app.header.tab_bar.size.x / app.header.tab_bar.tabs.len() as f32;
        let slider_velocity = app.header.tab_bar.slider_trailing_animation.velocity * tab_width;
        let leading = app.header.tab_bar.slider_animation.value;
        let following = app.header.tab_bar.slider_trailing_animation.value;
        let tab_bar_x = app.header.position.x + app.header.tab_bar.position.x;
        let from_index = leading.min(following).floor() as i32;
        let to_index = leading.max(following).ceil() as i32;
        let fragment_uniforms: [f32; 9] = [
            time,
            scroll_velocity,
            app.mouse_pos.x,
            app.mouse_pos.y,
            slider_velocity,
            tab_bar_x,
            tab_width,
            from_index as f32,
            to_index as f32,
        ];
        self.queue.write_buffer(&self.uniform_buffer, 64, bytemuck::cast_slice(&fragment_uniforms));
        
        let frame = self
            .surface
            .get_current_texture()
            .map_err(|e| anyhow::anyhow!("get_current_texture: {:?}", e))?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut vertices = Vec::new();

        // Ensure "root" is in hierarchy before validating any components
        // Root is the only component that can exist without a parent
        // We add it to hierarchy here so children can reference it
        // But we DON'T add it to rendered_components - let root.render() do that
        // This prevents duplicate warnings
        if self.component_validation_enabled && !self.component_hierarchy.contains_key("root") {
            self.component_hierarchy.insert("root".to_string(), None);
        }
        
        // Background, glow, and toasts live in Root. Modals use a separate pass below.
        use crate::gfx::components::modals;
        
        // Update Root layout only when layout_generation changed (viewport, sidebar, tab, etc.).
        let root_rect = crate::ui::core::Rect::new(0.0, 0.0, app.viewport_size.x, app.viewport_size.y);
        if app.root.last_layout_generation != app.layout_generation {
            app.root.last_layout_generation = app.layout_generation;
            app.root.update_layout(root_rect, None, None);
        }

        // Constellation: update node sizes only when graph content, scale, or viewport width changed.
        let constellation_key = app.graph_state.graph_id.as_ref().and_then(|_| {
            app.chat_window.as_ref().map(|chat| {
                let scale = chat.constellation_view.scale_animated;
                let scale_bucket = (scale * 4.0).round() as u32;
                let viewport_w_bucket = app.viewport_size.x.round() as u32;
                (app.graph_state.content_version, scale_bucket, viewport_w_bucket)
            })
        });
        let run_update_node_sizes = constellation_key != self.last_constellation_node_sizes_key;
        if run_update_node_sizes {
            self.last_constellation_node_sizes_key = constellation_key;
        }
        if app.graph_state.constellation_view_active() && run_update_node_sizes {
            let editing_override = app.chat_window.as_ref().and_then(|chat| {
                chat.editing_message_idx.and_then(|idx| {
                    chat.messages
                        .get(idx)
                        .and_then(|m| m.shard_id.as_deref())
                        .map(|id| (id, chat.edit_textarea.size))
                })
            });
            let visible_rect = app.chat_window.as_ref().map(|chat| {
                let v = &chat.constellation_view;
                let center = v.position + v.size * 0.5;
                let tl = (v.position - center) / v.scale_animated + v.camera_position_animated;
                let br = (v.position + v.size - center) / v.scale_animated + v.camera_position_animated;
                let margin = v.size / v.scale_animated;
                (tl - margin, br + margin)
            });
            app.graph_state.update_node_sizes(
                |text, size, rem, bold, italic| {
                    self.measure_markdown_segment_flow(text, size, rem, bold, italic)
                },
                editing_override,
                visible_rect,
                app.viewport_size.x,
            );
        }

        if app.ui_state.active_tab == Tab::Chat && !app.graph_state.constellation_view_active() {
            if let Some(ref mut chat) = app.chat_window {
                chat.refresh_linear_message_layout(self);
            }
        }

        // Render Root component tree (no dirty rect culling).
        let app_ref: &App = &*app;
        vertices.clear();
        app.root.render(self, app_ref, &mut vertices, None);

        assert!(
            self.scissor_stack.is_empty(),
            "scissor_stack not empty after root.render (len={})",
            self.scissor_stack.len()
        );

        // Flush and merge MAIN batches only (so modals can be drawn on top later)
        self.flush_current_batch();
        self.merge_compatible_batches();

        // Full clear every frame (no dirty rect / partial redraw).
        const BG_R: f32 = 0.1;
        const BG_G: f32 = 0.1;
        const BG_B: f32 = 0.12;
        let dirty_scissor_opt: Option<ScissorRect> = None;

        // Capture main text/icon before modal render (so we can blit main first, then modal on top)
        let main_text_commands: Vec<_> = self.text_queue.drain(..).collect();
        let main_icon_commands: Vec<_> = self.icon_queue.drain(..).collect();

        self.log_text_pipeline_after_queue(app, &main_text_commands, &main_icon_commands);

        // Phase 1 instrumentation: log when text command count exceeds threshold
        const DEBUG_TEXT_THRESHOLD: usize = 10_000;
        if app.debug_text_stats && main_text_commands.len() > DEBUG_TEXT_THRESHOLD {
            let visible = self.debug_constellation_visible_nodes.unwrap_or(0);
            eprintln!(
                "[debug_text_stats] text_commands={} visible_constellation_nodes={} (threshold={})",
                main_text_commands.len(),
                visible,
                DEBUG_TEXT_THRESHOLD,
            );
        }

        // Render modals (z-order 50) — geometry and queues captured separately so we draw them after main text/icon
        vertices.clear();
        modals::render_modals(self, app, &mut vertices);
        if !vertices.is_empty() {
            self.add_vertices(&vertices, None);
        }
        self.flush_current_batch();
        self.merge_compatible_batches();

        // Stable sort by compositing layer (then submission order) after modal geometry is batched.
        {
            let mut indexed: Vec<(usize, RenderBatch)> = self.render_batches.drain(..).enumerate().collect();
            indexed.sort_by_key(|(i, b)| (b.layer, *i));
            self.render_batches = indexed.into_iter().map(|(_, b)| b).collect();
        }

        self.log_text_pipeline_batches(app, &self.render_batches);

        let total_vertex_count: usize = self.render_batches.iter().map(|b| b.vertices.len()).sum();
        self.vertex_count = total_vertex_count as u32;

        let modal_text_commands: Vec<_> = self.text_queue.drain(..).collect();
        let modal_icon_commands: Vec<_> = self.icon_queue.drain(..).collect();
        
        // Clear previous text and icon scenes
        self.text_scenes.clear();
        self.icon_scenes.clear();
        
        fn build_text_scenes_for_layer(
            renderer: &mut Renderer,
            commands: &[TextDrawCommand],
            layer: CompositeLayer,
        ) -> Vec<(Option<ScissorRect>, Scene)> {
            use std::collections::HashMap;
            let mut text_groups: HashMap<Option<ScissorRect>, Vec<TextDrawCommand>> = HashMap::new();
            for cmd in commands.iter().filter(|c| c.layer == layer) {
                text_groups.entry(cmd.scissor).or_insert_with(Vec::new).push(cmd.clone());
            }
            let mut sorted_groups: Vec<_> = text_groups.into_iter().collect();
            sorted_groups.sort_by_key(|(scissor, _)| match scissor {
                None => (0, 0, 0, 0),
                Some(s) => (1, s.y, s.x, s.width),
            });
            let mut out = Vec::new();
            for (scissor_opt, commands) in sorted_groups {
                let mut scene = Scene::new();
                for cmd in commands {
                    renderer.render_text_to_scene(
                        &mut scene,
                        &cmd.text,
                        cmd.position.x,
                        cmd.position.y,
                        cmd.size,
                        cmd.max_width,
                        [cmd.color.x, cmd.color.y, cmd.color.z, cmd.color.w],
                        false,
                        false,
                    );
                }
                out.push((scissor_opt, scene));
            }
            out
        }
        fn build_icon_scenes_for_layer(
            renderer: &mut Renderer,
            commands: &[IconDrawCommand],
            layer: CompositeLayer,
        ) -> Vec<(Option<ScissorRect>, Scene)> {
            use std::collections::HashMap;
            let mut icon_groups: HashMap<Option<ScissorRect>, Vec<IconDrawCommand>> = HashMap::new();
            for cmd in commands.iter().filter(|c| c.layer == layer) {
                icon_groups.entry(cmd.scissor).or_insert_with(Vec::new).push(cmd.clone());
            }
            let mut sorted_groups: Vec<_> = icon_groups.into_iter().collect();
            sorted_groups.sort_by_key(|(scissor, _)| match scissor {
                None => (0, 0, 0, 0),
                Some(s) => (1, s.y, s.x, s.width),
            });
            let mut out = Vec::new();
            for (scissor_opt, commands) in sorted_groups {
                let mut scene = Scene::new();
                for cmd in commands {
                    renderer.render_icon_to_scene(
                        &mut scene,
                        &cmd.icon_name,
                        cmd.position,
                        cmd.size,
                        cmd.color,
                    );
                }
                out.push((scissor_opt, scene));
            }
            out
        }

        let all_text_commands: Vec<TextDrawCommand> = main_text_commands
            .iter()
            .chain(modal_text_commands.iter())
            .cloned()
            .collect();
        let all_icon_commands: Vec<IconDrawCommand> = main_icon_commands
            .iter()
            .chain(modal_icon_commands.iter())
            .cloned()
            .collect();
        if app.debug_text_pipeline {
            crate::gfx::debug_text_pipeline_log::append_line(&format!(
                "[debug_text_pipeline] all_text_commands len={} (main+modal); modal_text_only len={}",
                all_text_commands.len(),
                modal_text_commands.len(),
            ));
        }
        self.text_scenes.clear();
        self.icon_scenes.clear();
        
        let batch_vertex_offsets: Vec<u32> = {
            let mut starts = Vec::with_capacity(self.render_batches.len());
            let mut o = 0u32;
            for b in &self.render_batches {
                starts.push(o);
                o += b.vertices.len() as u32;
            }
            starts
        };

        // Upload all vertex data BEFORE starting the render pass (batched using staging belt)
        if self.vertex_count > 0 {
            let vertex_size = std::mem::size_of::<Vertex>();
            let total_vertex_bytes = (self.vertex_count as usize) * vertex_size;
            
            // Use staging belt for batched upload
            // Wrap in a block to ensure the guard is dropped before render pass
            {
                use std::num::NonZero;
                let mut staging_buffer = self.staging_belt.write_buffer(
                    &mut encoder,
                    &self.vertex_buffer,
                    0,
                    NonZero::new(total_vertex_bytes as u64).unwrap(),
                    &self.device,
                );
                
                // Copy all batches into staging buffer
                let mut vertex_offset = 0u32;
                for batch in self.render_batches.iter() {
                    let batch_vertex_count = batch.vertices.len() as u32;
                    
                    if batch_vertex_count == 0 {
                        continue;
                    }
                    
                    let byte_offset = (vertex_offset * vertex_size as u32) as usize;
                    let batch_bytes = bytemuck::cast_slice(&batch.vertices);
                    // Write to staging buffer - BufferWriteGuard implements DerefMut to [u8]
                    staging_buffer[byte_offset..byte_offset + batch_bytes.len()].copy_from_slice(batch_bytes);
                    
                    vertex_offset += batch_vertex_count;
                }
                // Explicitly drop the guard
                drop(staging_buffer);
            }
            
            // Finish staging belt to unmap buffers before submission
            self.staging_belt.finish();
        }
        
        let text_layer_draws = std::mem::take(&mut self.text_layer_draws);
        if app.debug_text_pipeline {
            crate::gfx::debug_text_pipeline_log::append_line(&format!(
                "[debug_text_pipeline] text_layer_draws (ConstellationText queued) count={}",
                text_layer_draws.len()
            ));
        }

        let main_text_scenes =
            build_text_scenes_for_layer(self, &all_text_commands, CompositeLayer::MainContent);
        self.log_text_pipeline_main_vello_groups(app, &main_text_scenes);
        let main_icon_scenes =
            build_icon_scenes_for_layer(self, &all_icon_commands, CompositeLayer::MainContent);

        const COMPOSITE_DRAW_ORDER: [CompositeLayer; 8] = [
            CompositeLayer::Background,
            CompositeLayer::MainContent,
            CompositeLayer::ConstellationText,
            CompositeLayer::ConstellationOverlay,
            CompositeLayer::SidebarChrome,
            CompositeLayer::ComposerChrome,
            CompositeLayer::HudChrome,
            CompositeLayer::Modal,
        ];

        for layer in COMPOSITE_DRAW_ORDER {
            let has_quads = self
                .render_batches
                .iter()
                .any(|b| b.layer == layer && !b.vertices.is_empty());
            let need_quads_pass = layer == CompositeLayer::Background || has_quads;
            if need_quads_pass {
                let load_op = if layer == CompositeLayer::Background {
                    wgpu::LoadOp::Clear(wgpu::Color {
                        r: BG_R as f64,
                        g: BG_G as f64,
                        b: BG_B as f64,
                        a: 1.0,
                    })
                } else {
                    wgpu::LoadOp::Load
                };

                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("quads_layer"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.msaa_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: load_op,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                if self.vertex_count > 0 {
                    render_pass.set_pipeline(&self.pipeline);
                    render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

                    for (batch_idx, batch) in self.render_batches.iter().enumerate() {
                        if batch.layer != layer {
                            continue;
                        }
                        let batch_vertex_count = batch.vertices.len() as u32;
                        if batch_vertex_count == 0 {
                            continue;
                        }
                        let vertex_offset = batch_vertex_offsets[batch_idx];
                        let scissor = if let Some(ref ds) = dirty_scissor_opt {
                            let effective = batch.scissor
                                .map(|s| s.intersect(ds))
                                .unwrap_or(*ds);
                            if effective.width == 0 || effective.height == 0 {
                                continue;
                            }
                            effective
                        } else if let Some(scissor) = &batch.scissor {
                            *scissor
                        } else {
                            ScissorRect {
                                x: 0,
                                y: 0,
                                width: self.config.width,
                                height: self.config.height,
                            }
                        };
                        let x = scissor.x.min(self.config.width);
                        let y = scissor.y.min(self.config.height);
                        let width = scissor.width.min(self.config.width.saturating_sub(x));
                        let height = scissor.height.min(self.config.height.saturating_sub(y));
                        render_pass.set_scissor_rect(x, y, width, height);
                        render_pass.draw(vertex_offset..vertex_offset + batch_vertex_count, 0..1);
                    }
                }
            }

            let layer_image_draws: Vec<ImageDrawCommand> = self
                .image_draws
                .iter()
                .filter(|draw| draw.layer == layer)
                .cloned()
                .collect();
            if !layer_image_draws.is_empty() {
                let mut image_bind_groups: Vec<(ImageDrawCommand, wgpu::Buffer, wgpu::BindGroup)> =
                    Vec::new();
                for draw in &layer_image_draws {
                    let Some(entry) = self.image_texture_cache.get(&draw.cache_key) else {
                        continue;
                    };
                    let uv_data: [u8; 16] = bytemuck::cast([0.0f32, 0.0f32, 1.0f32, 1.0f32]);
                    let uv_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("pdf_blit_uv"),
                        size: 16,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    self.queue.write_buffer(&uv_buffer, 0, &uv_data);
                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("pdf_image_bind"),
                        layout: &self.blit_rect_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&entry.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&self.vello_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: uv_buffer.as_entire_binding(),
                            },
                        ],
                    });
                    image_bind_groups.push((draw.clone(), uv_buffer, bind_group));
                }
                if !image_bind_groups.is_empty() {
                    let mut image_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("pdf_image_blit"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.msaa_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    image_pass.set_pipeline(&self.blit_rect_pipeline);
                    for (draw, _uv_buffer, bind_group) in &image_bind_groups {
                        let (x, y, w, h) = draw.dest_rect;
                        if w <= 0.0 || h <= 0.0 {
                            continue;
                        }
                        let vp_x = x.max(0.0).round() as u32;
                        let vp_y = y.max(0.0).round() as u32;
                        let vp_w = w.round() as u32;
                        let vp_h = h.round() as u32;
                        if vp_w == 0 || vp_h == 0 {
                            continue;
                        }
                        image_pass.set_bind_group(0, bind_group, &[]);
                        image_pass.set_viewport(
                            vp_x as f32,
                            vp_y as f32,
                            vp_w as f32,
                            vp_h as f32,
                            0.0,
                            1.0,
                        );
                        if let Some(scissor) = draw.scissor {
                            image_pass.set_scissor_rect(
                                scissor.x,
                                scissor.y,
                                scissor.width,
                                scissor.height,
                            );
                        } else {
                            image_pass.set_scissor_rect(0, 0, self.config.width, self.config.height);
                        }
                        image_pass.draw(0..6, 0..1);
                    }
                }
            }

            match layer {
                CompositeLayer::Background => {}
                CompositeLayer::MainContent => {
        // Blit main text (MainContent: tooltips, linear chat, library, etc.)
        if !main_text_scenes.is_empty() {
            let mut texture_bind_groups: Vec<(Option<ScissorRect>, wgpu::Texture, wgpu::TextureView, wgpu::BindGroup)> = Vec::new();
            for (scissor_opt, scene) in main_text_scenes.iter() {
                // Get texture from pool (reuse or create new)
                let (group_texture, group_texture_view, group_bind_group) = self.get_text_texture();
                
                // Render this group's scene to its texture
                self.vello_renderer
                    .render_to_texture(
                        &self.device,
                        &self.queue,
                        scene,
                        &group_texture_view,
                        &vello::RenderParams {
                            base_color: VelloColor::TRANSPARENT,
                            width: self.config.width,
                            height: self.config.height,
                            antialiasing_method: vello::AaConfig::Area,
                        },
                    )
                    .expect("Failed to render vello scene");
                
                texture_bind_groups.push((*scissor_opt, group_texture, group_texture_view, group_bind_group));
            }
            if !texture_bind_groups.is_empty() {
                let mut text_render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("main_text_blit"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.msaa_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                
                for (scissor_opt, _texture, texture_view, _bind_group) in texture_bind_groups.iter() {
                    let blit_scissor = if app.debug_text_pipeline_force_full_main_blit {
                        None
                    } else {
                        *scissor_opt
                    };
                    self.vello_queue_scene_blit(
                        &mut text_render_pass,
                        blit_scissor,
                        texture_view,
                        dirty_scissor_opt.as_ref(),
                    );
                }
                // Render pass ends here (dropped)
            }
            
            // Return textures to pool after use
            for (_, texture, texture_view, bind_group) in texture_bind_groups {
                self.return_text_texture(texture, texture_view, bind_group);
            }
        }

        // Blit main icon scenes
        if !main_icon_scenes.is_empty() {
            let mut icon_texture_bind_groups: Vec<(Option<ScissorRect>, wgpu::Texture, wgpu::TextureView, wgpu::BindGroup)> = Vec::new();
            for (scissor_opt, scene) in main_icon_scenes.iter() {
                // Get texture from pool (reuse or create new)
                let (icon_texture, icon_texture_view, icon_bind_group) = self.get_icon_texture();
                
                // Render icon scene to texture
                self.vello_renderer
                    .render_to_texture(
                        &self.device,
                        &self.queue,
                        scene,
                        &icon_texture_view,
                        &vello::RenderParams {
                            base_color: VelloColor::TRANSPARENT,
                            width: self.config.width,
                            height: self.config.height,
                            antialiasing_method: vello::AaConfig::Area,
                        },
                    )
                    .expect("Failed to render vello icon scene");
                
                icon_texture_bind_groups.push((*scissor_opt, icon_texture, icon_texture_view, icon_bind_group));
            }
            if !icon_texture_bind_groups.is_empty() {
                let mut icon_render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("main_icon_blit"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.msaa_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                
                for (scissor_opt, _texture, texture_view, _bind_group) in icon_texture_bind_groups.iter() {
                    self.vello_queue_scene_blit(
                        &mut icon_render_pass,
                        *scissor_opt,
                        texture_view,
                        dirty_scissor_opt.as_ref(),
                    );
                }
                // Render pass ends here (dropped)
            }
            
            for (_, texture, texture_view, bind_group) in icon_texture_bind_groups {
                self.return_icon_texture(texture, texture_view, bind_group);
            }
        }
                }
                CompositeLayer::ConstellationText => {
        if !text_layer_draws.is_empty() {
            let mut text_layer_bind_groups: Vec<(TextLayerDraw, wgpu::Buffer, wgpu::BindGroup)> = Vec::new();
            for draw in &text_layer_draws {
                if draw.layer != CompositeLayer::ConstellationText {
                    continue;
                }
                if let Some(entry) = self.text_layer_cache.entries.get(&draw.key) {
                    let (x, y, w, h) = draw.dest_rect;
                    if w <= 0.0 || h <= 0.0 {
                        continue;
                    }

                    // Logical full-rect UV range before screen clipping. Scroll is implemented
                    // by translating the destination quad; UVs always map the full layer.
                    let base_u_min = 0.0f32;
                    let base_u_max = 1.0f32;
                    let base_v_min = 0.0f32;
                    let base_v_max = 1.0f32;

                    // Clip dest_rect to framebuffer ∩ draw.scissor so UVs match the visible graph/card
                    // region (using only y=0 mis-maps when dest.y is above the graph top).
                    let screen_w = self.config.width as f32;
                    let screen_h = self.config.height as f32;
                    let (bx0, by0, bx1, by1) = if let Some(s) = draw.scissor {
                        let bx0 = s.x as f32;
                        let by0 = s.y as f32;
                        let bx1 = (s.x + s.width) as f32;
                        let by1 = (s.y + s.height) as f32;
                        (
                            bx0.max(0.0),
                            by0.max(0.0),
                            bx1.min(screen_w),
                            by1.min(screen_h),
                        )
                    } else {
                        (0.0, 0.0, screen_w, screen_h)
                    };
                    let Some((clip_left, clip_top, clip_right, clip_bottom)) =
                        dest_rect_clip_against_bounds(x, y, w, h, bx0, by0, bx1, by1)
                    else {
                        continue;
                    };

                    let visible_w = (w - clip_left - clip_right).max(0.0);
                    let visible_h = (h - clip_top - clip_bottom).max(0.0);
                    if visible_w <= 0.0 || visible_h <= 0.0 {
                        continue;
                    }

                    // Horizontal UV window after clipping.
                    let u_min = base_u_min + (base_u_max - base_u_min) * (clip_left / w).clamp(0.0, 1.0);
                    let u_max = base_u_max - (base_u_max - base_u_min) * (clip_right / w).clamp(0.0, 1.0);

                    // Vertical UV window after clipping.
                    let v_min = base_v_min + (base_v_max - base_v_min) * (clip_top / h).clamp(0.0, 1.0);
                    let v_max = base_v_max - (base_v_max - base_v_min) * (clip_bottom / h).clamp(0.0, 1.0);

                    let uv_data: [u8; 16] = bytemuck::cast([u_min, v_min, u_max, v_max]);
                    let uv_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("blit_rect_uv"),
                        size: 16,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    self.queue.write_buffer(&uv_buffer, 0, &uv_data);
                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("text_layer_bind"),
                        layout: &self.blit_rect_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&entry.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&self.vello_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: uv_buffer.as_entire_binding(),
                            },
                        ],
                    });
                    text_layer_bind_groups.push((draw.clone(), uv_buffer, bind_group));
                }
            }
            if !text_layer_bind_groups.is_empty() {
                let mut text_layer_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("text_layer_blit"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.msaa_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                text_layer_pass.set_pipeline(&self.blit_rect_pipeline);
                for (draw, _uv_buffer, bind_group) in &text_layer_bind_groups {
                    let (x, y, w, h) = draw.dest_rect;
                    if w <= 0.0 || h <= 0.0 {
                        continue;
                    }

                    let screen_w = self.config.width as f32;
                    let screen_h = self.config.height as f32;
                    let (bx0, by0, bx1, by1) = if let Some(s) = draw.scissor {
                        let bx0 = s.x as f32;
                        let by0 = s.y as f32;
                        let bx1 = (s.x + s.width) as f32;
                        let by1 = (s.y + s.height) as f32;
                        (
                            bx0.max(0.0),
                            by0.max(0.0),
                            bx1.min(screen_w),
                            by1.min(screen_h),
                        )
                    } else {
                        (0.0, 0.0, screen_w, screen_h)
                    };
                    let Some((clip_left, clip_top, clip_right, clip_bottom)) =
                        dest_rect_clip_against_bounds(x, y, w, h, bx0, by0, bx1, by1)
                    else {
                        continue;
                    };

                    let visible_w = (w - clip_left - clip_right).max(0.0);
                    let visible_h = (h - clip_top - clip_bottom).max(0.0);
                    if visible_w <= 0.0 || visible_h <= 0.0 {
                        continue;
                    }

                    // `as u32` truncates; sums like (y + clip_top) can be 71.999 for a 72px edge,
                    // which disagrees with integer `ScissorRect` and clips an extra row.
                    let vp_x = (x + clip_left).max(0.0).round() as u32;
                    let vp_y = (y + clip_top).max(0.0).round() as u32;
                    let vp_w = visible_w.round() as u32;
                    let vp_h = visible_h.round() as u32;

                    if vp_w > 0 && vp_h > 0 {
                        let merged_scissor = draw.scissor;

                        text_layer_pass.set_bind_group(0, bind_group, &[]);
                        text_layer_pass.set_viewport(
                            vp_x as f32,
                            vp_y as f32,
                            vp_w as f32,
                            vp_h as f32,
                            0.0,
                            1.0,
                        );
                        if let Some(s) = merged_scissor {
                            // Intersect with viewport quad and content scissor in absolute coordinates.
                            let sx = s.x.max(vp_x);
                            let sy = s.y.max(vp_y);
                            let s_right = s.x + s.width;
                            let s_bottom = s.y + s.height;
                            let vp_right = vp_x + vp_w;
                            let vp_bottom = vp_y + vp_h;
                            let sw = s_right.min(vp_right).saturating_sub(sx);
                            let sh = s_bottom.min(vp_bottom).saturating_sub(sy);
                            if sw > 0 && sh > 0 {
                                text_layer_pass.set_scissor_rect(sx, sy, sw, sh);
                                text_layer_pass.draw(0..3, 0..1);
                            }
                        } else {
                            text_layer_pass.set_scissor_rect(vp_x, vp_y, vp_w, vp_h);
                            text_layer_pass.draw(0..3, 0..1);
                        }
                    }
                }
            }
        }
                }
                CompositeLayer::ConstellationOverlay
                | CompositeLayer::SidebarChrome
                | CompositeLayer::ComposerChrome
                | CompositeLayer::HudChrome
                | CompositeLayer::Modal => {
            let chrome_text = build_text_scenes_for_layer(self, &all_text_commands, layer);
            if !chrome_text.is_empty() {
                let mut texture_bind_groups: Vec<(Option<ScissorRect>, wgpu::Texture, wgpu::TextureView, wgpu::BindGroup)> = Vec::new();
                for (scissor_opt, scene) in chrome_text.iter() {
                    let (group_texture, group_texture_view, group_bind_group) = self.get_text_texture();
                    self.vello_renderer
                        .render_to_texture(
                            &self.device,
                            &self.queue,
                            scene,
                            &group_texture_view,
                            &vello::RenderParams {
                                base_color: VelloColor::TRANSPARENT,
                                width: self.config.width,
                                height: self.config.height,
                                antialiasing_method: vello::AaConfig::Area,
                            },
                        )
                        .expect("Failed to render vello scene");
                    texture_bind_groups.push((*scissor_opt, group_texture, group_texture_view, group_bind_group));
                }
                if !texture_bind_groups.is_empty() {
                    let mut text_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("layer_text_blit"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.msaa_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    for (scissor_opt, _texture, texture_view, _bind_group) in texture_bind_groups.iter() {
                        self.vello_queue_scene_blit(
                            &mut text_pass,
                            *scissor_opt,
                            texture_view,
                            dirty_scissor_opt.as_ref(),
                        );
                    }
                }
                for (_, texture, texture_view, bind_group) in texture_bind_groups {
                    self.return_text_texture(texture, texture_view, bind_group);
                }
            }

            let chrome_icons = build_icon_scenes_for_layer(self, &all_icon_commands, layer);
            if !chrome_icons.is_empty() {
                let mut icon_texture_bind_groups: Vec<(Option<ScissorRect>, wgpu::Texture, wgpu::TextureView, wgpu::BindGroup)> = Vec::new();
                for (scissor_opt, scene) in chrome_icons.iter() {
                    let (icon_texture, icon_texture_view, icon_bind_group) = self.get_icon_texture();
                    self.vello_renderer
                        .render_to_texture(
                            &self.device,
                            &self.queue,
                            scene,
                            &icon_texture_view,
                            &vello::RenderParams {
                                base_color: VelloColor::TRANSPARENT,
                                width: self.config.width,
                                height: self.config.height,
                                antialiasing_method: vello::AaConfig::Area,
                            },
                        )
                        .expect("Failed to render vello icon scene");
                    icon_texture_bind_groups.push((*scissor_opt, icon_texture, icon_texture_view, icon_bind_group));
                }
                if !icon_texture_bind_groups.is_empty() {
                    let mut icon_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("layer_icon_blit"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.msaa_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    for (scissor_opt, _texture, texture_view, _bind_group) in icon_texture_bind_groups.iter() {
                        self.vello_queue_scene_blit(
                            &mut icon_pass,
                            *scissor_opt,
                            texture_view,
                            dirty_scissor_opt.as_ref(),
                        );
                    }
                }
                for (_, texture, texture_view, bind_group) in icon_texture_bind_groups {
                    self.return_icon_texture(texture, texture_view, bind_group);
                }
            }
                }
            }
        }

        // Resolve MSAA → swapchain.  An empty render pass is enough: wgpu performs
        // the multisample resolve automatically when the pass ends with a resolve_target set.
        {
            let _resolve_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("msaa_resolve"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa_view,
                    resolve_target: Some(&view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }

        self.queue.submit([encoder.finish()]);

        
        frame.present();
        
        // Recall staging belt after frame is presented (reclaim memory)
        self.staging_belt.recall();
        
        // Validate component hierarchy at end of frame
        if self.component_validation_enabled {
            self.validate_hierarchy();
        }

        self.first_frame = false;
        
        Ok(())
    }
}
