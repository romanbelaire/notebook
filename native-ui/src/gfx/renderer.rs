use crate::app::App;
use crate::gfx::types::Vertex;
use crate::gfx::icons::IconCache;
use crate::ui::core::Rect;
use crate::ui::components::Renderable;
use glam::{Mat4, Vec2, Vec4};
use wgpu::util::DeviceExt;
use winit::window::Window;
use pulldown_cmark::{Parser, Event, Tag};
use std::sync::Arc;
use vello::Scene;
use vello::peniko::{Color as VelloColor, Brush, Fill};
use parley::{FontContext, LayoutContext};
use parley::layout::{Alignment, PositionedLayoutItem};

#[derive(Clone)]
struct TextDrawCommand {
    text: String,
    position: Vec2,
    color: Vec4,
    size: f32,
    scissor: Option<ScissorRect>,  // Track which scissor was active when text was queued
}

#[derive(Clone)]
struct IconDrawCommand {
    icon_name: String,
    position: Vec2,
    size: f32,
    color: Vec4,
    scissor: Option<ScissorRect>,  // Track which scissor was active when icon was queued
}

/// A batch of vertices to render with a specific scissor rect
#[derive(Debug, Clone)]
struct RenderBatch {
    vertices: Vec<Vertex>,
    scissor: Option<ScissorRect>,
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
        // Use rect.y directly - no coordinate conversion needed with our projection matrix!
        let y_flipped = rect.y;
        
        let result = Self {
            x: rect.x.max(0.0) as u32,
            y: y_flipped.max(0.0) as u32,
            width: rect.width.max(0.0) as u32,
            height: rect.height.max(0.0) as u32,
        };
        
        result
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
    // Blit pipeline to copy vello texture to surface
    blit_pipeline: wgpu::RenderPipeline,
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
    text_scenes: Vec<(Option<ScissorRect>, Scene)>,
    // Texture pools for text/icon rendering (reuse across frames)
    // Store only textures and views; bind groups are recreated as needed
    text_texture_pool: Vec<(wgpu::Texture, wgpu::TextureView)>,
    icon_texture_pool: Vec<(wgpu::Texture, wgpu::TextureView)>,
    max_texture_pool_size: usize,
    // Text measurement caches (avoid redundant Parley calculations)
    // Use u32 for f32 values (scaled by 1000) since f32 doesn't implement Hash
    glyph_position_cache: std::collections::HashMap<(String, u32, u32), Vec<f32>>,
    text_measurement_cache: std::collections::HashMap<(String, u32), Vec2>,
    /// Phase 4: Cached wrapped layout for stylus/notepad. Key: (text, size*1000, max_width*1000). Value: (line_height, lines with positions relative to 0).
    glyph_position_wrapped_cache: std::collections::HashMap<(String, u32, u32), (f32, Vec<Vec<f32>>)>,
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
    last_constellation_node_sizes_key: Option<(u64, u32)>,
    /// Set by render_constellation when debug_text_stats; used for instrumentation logging.
    debug_constellation_visible_nodes: Option<usize>,
    /// Per-node text layer cache for constellation. Key: (node_id, is_user, content_hash, scale_bucket, width_bucket).
    text_layer_cache: TextLayerCache,
    /// Pipeline for drawing textured rects (text layers) with UV offset for scroll.
    blit_rect_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for blit_rect (texture, sampler, uv uniform).
    blit_rect_bind_group_layout: wgpu::BindGroupLayout,
    /// Draw commands for text layers; processed after main quads.
    text_layer_draws: Vec<TextLayerDraw>,
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
                count: 1,
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
                count: 1,
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
                count: 1,
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
            blit_pipeline,
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
            text_scenes: Vec::new(),
            text_texture_pool: Vec::new(),
            icon_texture_pool: Vec::new(),
            max_texture_pool_size: 8,
            glyph_position_cache: std::collections::HashMap::new(),
            text_measurement_cache: std::collections::HashMap::new(),
            glyph_position_wrapped_cache: std::collections::HashMap::new(),
            component_validation_enabled: true, // Enable validation in debug builds
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
        };

        renderer.update_projection_matrix(size.width, size.height);
        
        renderer
    }

    // Helper: Render text directly to vello scene using parley
    fn render_text_to_scene(
        &mut self,
        scene: &mut Scene,
        text: &str,
        x: f32,
        y: f32,
        font_size: f32,
        max_width: Option<f32>,
        color: [f32; 4],
    ) {
        use parley::style::StyleProperty;
        use vello::peniko::kurbo::Affine;
        use vello::Glyph;
        
        let brush = Brush::Solid(VelloColor::rgba(
            color[0] as f64,
            color[1] as f64,
            color[2] as f64,
            color[3] as f64,
        ));
        
        // Create layout
        let mut builder = self.layout_context.ranged_builder(&mut self.font_context, text, 1.0);
        builder.push_default(StyleProperty::FontSize(font_size));
        builder.push_default(StyleProperty::Brush(brush.clone()));
        
        let mut layout = builder.build(text);
        layout.break_all_lines(max_width);
        layout.align(max_width, Alignment::Start);
        
        // Render glyphs
        let transform = Affine::translate((x as f64, y as f64));
        for line in layout.lines() {
            for item in line.items() {
                if let PositionedLayoutItem::GlyphRun(glyph_run) = item {
                    let mut glyph_x = glyph_run.offset();
                    let glyph_y = line.metrics().baseline;
                    
                    let run = glyph_run.run();
                    let font = run.font();
                    let font_size_val = run.font_size();
                    let style = glyph_run.style();
                    
                    scene
                        .draw_glyphs(font)
                        .brush(&style.brush)
                        .transform(transform)
                        .font_size(font_size_val)
                        .draw(
                            Fill::NonZero,
                            glyph_run.glyphs().map(|g| {
                                let gx = glyph_x + g.x;
                                let gy = glyph_y - g.y;
                                glyph_x += g.advance;
                                Glyph {
                                    id: g.id as u32,
                                    x: gx,
                                    y: gy,
                                }
                            }),
                        );
                }
            }
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
        use pulldown_cmark::{Parser, Event, Tag};
        let mut current_x = start_x;
        let mut current_y = start_y;
        let line_height = font_size * 1.2;
        let mut max_w = 0.0f32;
        let mut bold = false;
        let mut italic = false;
        let mut code = false;
        let mut current_text = String::new();

        let effective_font_size = |b: bool, i: bool, c: bool| -> f32 {
            if c { font_size * 0.9 }
            else if b && i { font_size * 1.05 }
            else if b { font_size * 1.1 }
            else if i { font_size * 0.95 }
            else { font_size }
        };
        let segment_color = |c: bool| -> [f32; 4] {
            if c { [0.8, 0.8, 0.9, 1.0] }
            else { [base_color.x, base_color.y, base_color.z, base_color.w] }
        };

        for event in Parser::new(markdown) {
            match event {
                Event::Start(Tag::Strong) => {
                    if !current_text.is_empty() {
                        let scale = effective_font_size(bold, italic, code);
                        let color = segment_color(code);
                        self.render_text_to_scene(scene, &current_text, current_x, current_y, scale, Some(max_width), color);
                        let w = self.measure_text(&current_text, scale).x;
                        current_x += w;
                        max_w = max_w.max(current_x - start_x);
                        current_text.clear();
                    }
                    bold = true;
                }
                Event::End(pulldown_cmark::TagEnd::Strong) => {
                    if !current_text.is_empty() {
                        let scale = effective_font_size(bold, italic, code);
                        let color = segment_color(code);
                        self.render_text_to_scene(scene, &current_text, current_x, current_y, scale, Some(max_width), color);
                        let w = self.measure_text(&current_text, scale).x;
                        current_x += w;
                        max_w = max_w.max(current_x - start_x);
                        current_text.clear();
                    }
                    bold = false;
                }
                Event::Start(Tag::Emphasis) => {
                    if !current_text.is_empty() {
                        let scale = effective_font_size(bold, italic, code);
                        let color = segment_color(code);
                        self.render_text_to_scene(scene, &current_text, current_x, current_y, scale, Some(max_width), color);
                        let w = self.measure_text(&current_text, scale).x;
                        current_x += w;
                        max_w = max_w.max(current_x - start_x);
                        current_text.clear();
                    }
                    italic = true;
                }
                Event::End(pulldown_cmark::TagEnd::Emphasis) => {
                    if !current_text.is_empty() {
                        let scale = effective_font_size(bold, italic, code);
                        let color = segment_color(code);
                        self.render_text_to_scene(scene, &current_text, current_x, current_y, scale, Some(max_width), color);
                        let w = self.measure_text(&current_text, scale).x;
                        current_x += w;
                        max_w = max_w.max(current_x - start_x);
                        current_text.clear();
                    }
                    italic = false;
                }
                Event::Start(Tag::CodeBlock(_)) => {
                    if !current_text.is_empty() {
                        let scale = effective_font_size(bold, italic, code);
                        let color = segment_color(code);
                        self.render_text_to_scene(scene, &current_text, current_x, current_y, scale, Some(max_width), color);
                        let w = self.measure_text(&current_text, scale).x;
                        current_x += w;
                        max_w = max_w.max(current_x - start_x);
                        current_text.clear();
                    }
                    code = true;
                }
                Event::End(pulldown_cmark::TagEnd::CodeBlock) => {
                    if !current_text.is_empty() {
                        let scale = effective_font_size(bold, italic, code);
                        let color = segment_color(code);
                        self.render_text_to_scene(scene, &current_text, current_x, current_y, scale, Some(max_width), color);
                        let w = self.measure_text(&current_text, scale).x;
                        current_x += w;
                        max_w = max_w.max(current_x - start_x);
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
                        let test_width = self.measure_text(&test_text, scale).x;
                        if test_width > max_width && !current_text.is_empty() {
                            let color = segment_color(code);
                            self.render_text_to_scene(scene, &current_text, start_x, current_y, scale, Some(max_width), color);
                            let w = self.measure_text(&current_text, scale).x;
                            max_w = max_w.max(w);
                            current_y += line_height;
                            current_x = start_x;
                            current_text = word.to_string();
                        } else {
                            current_text = test_text;
                        }
                    }
                }
                Event::SoftBreak | Event::HardBreak => {
                    if !current_text.is_empty() {
                        let scale = effective_font_size(bold, italic, code);
                        let color = segment_color(code);
                        self.render_text_to_scene(scene, &current_text, start_x, current_y, scale, Some(max_width), color);
                        let w = self.measure_text(&current_text, scale).x;
                        max_w = max_w.max(w);
                        current_text.clear();
                    }
                    current_y += line_height;
                    current_x = start_x;
                }
                _ => {}
            }
        }
        if !current_text.is_empty() {
            let scale = effective_font_size(bold, italic, code);
            let color = segment_color(code);
            self.render_text_to_scene(scene, &current_text, current_x, current_y, scale, Some(max_width), color);
            let w = self.measure_text(&current_text, scale).x;
            max_w = max_w.max(current_x - start_x + w);
        }
        let height = current_y - start_y + line_height;
        (max_w.max(1.0), height.max(1.0))
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
        use vello::peniko::kurbo::Affine;
        
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
        
        // Render stroke paths as filled paths (most icons use fill)
        // TODO: Add proper stroke support when vello API is clarified
        for (path, _stroke_width) in &paths.stroke_paths {
            // Render stroke as fill for now (most icons are fill-based)
            scene.fill(Fill::NonZero, transform, &brush, None, path);
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
                        let draw_text = if library.search_input.text.is_empty() {
                            library.search_input.placeholder.clone()
                        } else {
                            library.search_input.text.clone()
                        };
                        let positions = self.compute_glyph_positions(&draw_text, 14.0, 0.0);
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
                        let draw_text = if ingest.pdf_dir_input.text.is_empty() {
                            ingest.pdf_dir_input.placeholder.clone()
                        } else {
                            ingest.pdf_dir_input.text.clone()
                        };
                        let positions = self.compute_glyph_positions(&draw_text, 14.0, 0.0);
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
                if let Some(ref mut settings) = app.settings_window {
                    if settings.hf_token_input.is_selecting && settings.hf_token_input.contains(app.mouse_pos) {
                        let draw_text = if settings.hf_token_input.text.is_empty() {
                            settings.hf_token_input.placeholder.clone()
                        } else {
                            settings.hf_token_input.text.clone()
                        };
                        let positions = self.compute_glyph_positions(&draw_text, 14.0, 0.0);
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
                if let Some(ref mut settings) = app.settings_window {
                    if settings.model_id_input.is_selecting && settings.model_id_input.contains(app.mouse_pos) {
                        let draw_text = if settings.model_id_input.text.is_empty() {
                            settings.model_id_input.placeholder.clone()
                        } else {
                            settings.model_id_input.text.clone()
                        };
                        let positions = self.compute_glyph_positions(&draw_text, 14.0, 0.0);
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

    fn update_projection_matrix(&mut self, width: u32, height: u32) {
        let width = width as f32;
        let height = height as f32;
        
        let projection = Mat4::orthographic_rh(0.0, width, height, 0.0, -1.0, 1.0);
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&projection.to_cols_array()));
    }

    /// Queue text, capturing the current active scissor (for legacy code)
    pub fn queue_text(&mut self, text: &str, position: Vec2, color: Vec4, size: f32) {
        // Capture the current active scissor (if any) when text is queued
        let active_scissor = self.scissor_stack.last().copied();
        self.queue_text_with_scissor(text, position, color, size, active_scissor);
    }
    
    /// Queue text with an explicit scissor rect (for containers that manage their own clipping)
    /// scissor_rect should be a ScissorRect (already converted to WGPU coordinates)
    pub fn queue_text_with_scissor(&mut self, text: &str, position: Vec2, color: Vec4, size: f32, scissor_rect: Option<ScissorRect>) {
        self.text_queue.push(TextDrawCommand {
            text: text.to_string(),
            position,
            color,
            size,
            scissor: scissor_rect,
        });
    }
    
    /// Queue text with a UI Rect scissor (containers can pass their scissor rect directly)
    pub fn queue_text_with_ui_scissor(&mut self, text: &str, position: Vec2, color: Vec4, size: f32, scissor_rect: Option<&Rect>) {
        let scissor = scissor_rect.map(|r| ScissorRect::from_rect(r, self.viewport_height));
        self.queue_text_with_scissor(text, position, color, size, scissor);
    }

    /// Queue plain text with word wrapping to max_width. Returns the y offset of the last line (total height).
    pub fn queue_plain_text_wrapped(&mut self, text: &str, position: Vec2, color: Vec4, size: f32, max_width: f32) -> f32 {
        let line_height = size * 1.2;
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut current_line = String::new();
        let mut current_y = position.y;
        for word in words {
            let test_line = if current_line.is_empty() {
                word.to_string()
            } else {
                format!("{} {}", current_line, word)
            };
            let test_w = self.measure_text(&test_line, size).x;
            if test_w > max_width && !current_line.is_empty() {
                self.queue_text(&current_line, Vec2::new(position.x, current_y), color, size);
                current_y += line_height;
                current_line = word.to_string();
            } else {
                current_line = test_line;
            }
        }
        if !current_line.is_empty() {
            self.queue_text(&current_line, Vec2::new(position.x, current_y), color, size);
            current_y += line_height;
        }
        current_y - position.y
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
    pub fn draw_text_layer(
        &mut self,
        key: (String, bool, u64, u32, u32),
        dest_rect: (f32, f32, f32, f32),
        scroll_offset: f32,
    ) {
        if !self.text_layer_cache.entries.contains_key(&key) {
            return;
        }
        let scissor = self.scissor_stack.last().copied();
        self.text_layer_draws.push(TextLayerDraw {
            key,
            dest_rect,
            scroll_offset,
            scissor,
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
                // If this batch has the same scissor as the last one, merge them
                if last_batch.scissor == batch.scissor {
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
        // Check cache first
        // Convert f32 to u32 for HashMap key (scale by 1000 for precision)
        let cache_key = (text.to_string(), (size * 1000.0) as u32, (start_x * 1000.0) as u32);
        if let Some(cached) = self.glyph_position_cache.get(&cache_key) {
            return cached.clone();
        }
        
        use parley::style::StyleProperty;
        
        // Create a parley layout
        let mut builder = self.layout_context.ranged_builder(&mut self.font_context, text, 1.0);
        builder.push_default(StyleProperty::FontSize(size));
        
        let mut layout = builder.build(text);
        layout.break_all_lines(None);
        layout.align(None, Alignment::Start);
        
        // Extract glyph positions
        let mut positions = Vec::with_capacity(text.chars().count() + 1);
        positions.push(start_x);
            
        for line in layout.lines() {
            for item in line.items() {
                if let PositionedLayoutItem::GlyphRun(glyph_run) = item {
                    let mut x = start_x + glyph_run.offset();
                    for glyph in glyph_run.glyphs() {
                        x += glyph.advance;
                        positions.push(x);
                    }
                }
            }
        }
        
        // Ensure we have the right number of positions
        while positions.len() < text.chars().count() + 1 {
            positions.push(*positions.last().unwrap_or(&start_x));
        }
        
        // Limit cache size to prevent memory growth (evict oldest entries when >1000)
        if self.glyph_position_cache.len() > 1000 {
            // Remove a random entry (simple eviction strategy)
            if let Some(key) = self.glyph_position_cache.keys().next().cloned() {
                self.glyph_position_cache.remove(&key);
            }
        }
        
        // Store in cache
        self.glyph_position_cache.insert(cache_key, positions.clone());
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
        let line_height = size * 1.2;
        if text.is_empty() {
            return (line_height, vec![vec![start_x]]);
        }
        let size_u = (size * 1000.0).round() as u32;
        let max_u = (max_width * 1000.0).round() as u32;
        let cache_key = (text.to_string(), size_u, max_u);
        if let Some((lh, cached_lines)) = self.glyph_position_wrapped_cache.get(&cache_key) {
            let lines: Vec<Vec<f32>> = cached_lines
                .iter()
                .map(|line| line.iter().map(|&p| p + start_x).collect())
                .collect();
            return (*lh, lines);
        }
        use parley::style::StyleProperty;
        let mut builder = self.layout_context.ranged_builder(&mut self.font_context, text, 1.0);
        builder.push_default(StyleProperty::FontSize(size));
        let mut layout = builder.build(text);
        layout.break_all_lines(Some(max_width));
        layout.align(Some(max_width), Alignment::Start);
        let mut lines = Vec::new();
        for line in layout.lines() {
            let mut positions = Vec::new();
            positions.push(0.0);
            let mut x = 0.0f32;
            for item in line.items() {
                if let PositionedLayoutItem::GlyphRun(glyph_run) = item {
                    x = glyph_run.offset();
                    for glyph in glyph_run.glyphs() {
                        x += glyph.advance;
                        positions.push(x);
                    }
                }
            }
            lines.push(positions);
        }
        if lines.is_empty() {
            lines.push(vec![0.0]);
        }
        const MAX_WRAPPED_CACHE: usize = 256;
        if self.glyph_position_wrapped_cache.len() >= MAX_WRAPPED_CACHE {
            self.glyph_position_wrapped_cache.clear();
        }
        self.glyph_position_wrapped_cache.insert(cache_key, (line_height, lines.clone()));
        let lines: Vec<Vec<f32>> = lines
            .iter()
            .map(|line| line.iter().map(|&p| p + start_x).collect())
            .collect();
        (line_height, lines)
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
            return Vec2::new(0.0, size * 1.2);
        }
        
        // Check cache first
        // Convert f32 to u32 for HashMap key (scale by 1000 for precision)
        let cache_key = (text.to_string(), (size * 1000.0) as u32);
        if let Some(cached) = self.text_measurement_cache.get(&cache_key) {
            return *cached;
        }
        
        // Use approximation for quick measurements during rendering
        // For accurate cursor positioning, use compute_glyph_positions
        let char_count = text.chars().count();
        let avg_char_width = size * 0.66;  // Slightly wider average for more open text
        let width = char_count as f32 * avg_char_width;
        let result = Vec2::new(width, size * 1.2);
        
        // Limit cache size to prevent memory growth (evict oldest entries when >1000)
        if self.text_measurement_cache.len() > 1000 {
            // Remove a random entry (simple eviction strategy)
            if let Some(key) = self.text_measurement_cache.keys().next().cloned() {
                self.text_measurement_cache.remove(&key);
            }
        }
        
        // Store in cache
        self.text_measurement_cache.insert(cache_key, result);
        result
    }

    /// Get accurate text metrics using Parley layout
    /// Returns (width, height, baseline_from_top)
    /// Height includes ascent + descent, baseline is distance from top to baseline
    pub fn measure_text_accurate(&mut self, text: &str, size: f32) -> (f32, f32, f32) {
        use parley::style::StyleProperty;
        
        if text.is_empty() {
            return (0.0, size * 1.2, size * 0.75);
        }
        
        // Create a parley layout
        let mut builder = self.layout_context.ranged_builder(&mut self.font_context, text, 1.0);
        builder.push_default(StyleProperty::FontSize(size));
        
        let mut layout = builder.build(text);
        layout.break_all_lines(None);
        layout.align(None, Alignment::Start);
        
        // Get metrics from first line (single-line text)
        let mut width = 0.0;
        let mut height = size * 1.2;
        let mut baseline = size * 0.75;
        
        for line in layout.lines() {
            let metrics = line.metrics();
            width = metrics.advance;
            height = metrics.size();
            baseline = metrics.baseline;
            break; // Only need first line for single-line text
        }
        
        (width, height, baseline)
    }

    fn text_line_height(&self, size: f32) -> f32 {
        size * 1.2
    }

    fn queue_markdown_text(&mut self, markdown: &str, position: Vec2, base_color: Vec4, size: f32, max_width: f32) -> f32 {
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
                        let color = if code {
                            Vec4::new(0.8, 0.8, 0.9, 1.0) // Light gray for code
                        } else {
                            base_color
                        };
                        let scale = if code { size * 0.9 } else { size };
                        self.queue_text(&current_text, Vec2::new(current_x, current_y), color, scale);
                        let text_width = self.measure_text(&current_text, scale).x;
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
                        self.queue_text(&current_text, Vec2::new(current_x, current_y), color, scale);
                        let text_width = self.measure_text(&current_text, scale).x;
                        current_x += text_width;
                        current_line_width += text_width;
                        current_text.clear();
                    }
                    bold = false;
                }
                Event::Start(Tag::Emphasis) => {
                    if !current_text.is_empty() {
                        let color = if code {
                            Vec4::new(0.8, 0.8, 0.9, 1.0)
                        } else {
                            base_color
                        };
                        let scale = if code { size * 0.9 } else { size };
                        self.queue_text(&current_text, Vec2::new(current_x, current_y), color, scale);
                        let text_width = self.measure_text(&current_text, scale).x;
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
                        self.queue_text(&current_text, Vec2::new(current_x, current_y), color, scale);
                        let text_width = self.measure_text(&current_text, scale).x;
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
                        self.queue_text(&current_text, Vec2::new(current_x, current_y), color, scale);
                        let text_width = self.measure_text(&current_text, scale).x;
                        current_x += text_width;
                        current_line_width += text_width;
                        current_text.clear();
                    }
                    code = true;
                }
                Event::End(pulldown_cmark::TagEnd::CodeBlock) => {
                    if !current_text.is_empty() {
                        let color = Vec4::new(0.8, 0.8, 0.9, 1.0); // Light gray for code
                        let scale = size * 0.9;
                        self.queue_text(&current_text, Vec2::new(current_x, current_y), color, scale);
                        let text_width = self.measure_text(&current_text, scale).x;
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
                        
                        let color = if code {
                            Vec4::new(0.8, 0.8, 0.9, 1.0)
                        } else {
                            base_color
                        };
                        let scale = if code { size * 0.9 } else if bold { size * 1.1 } else if italic { size * 0.95 } else { size };
                        
                        let test_text = format!("{}{}", current_text, word_with_space);
                        let test_width = self.measure_text(&test_text, scale).x;
                        
                        if test_width > max_width && !current_text.is_empty() {
                            // Render current line and wrap
                            self.queue_text(&current_text, Vec2::new(position.x, current_y), color, scale);
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
                        let color = if code {
                            Vec4::new(0.8, 0.8, 0.9, 1.0)
                        } else {
                            base_color
                        };
                        let scale = if code { size * 0.9 } else if bold { size * 1.1 } else if italic { size * 0.95 } else { size };
                        self.queue_text(&current_text, Vec2::new(position.x, current_y), color, scale);
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
            let color = if code {
                Vec4::new(0.8, 0.8, 0.9, 1.0)
            } else {
                base_color
            };
            let scale = if code { size * 0.9 } else if bold { size * 1.1 } else if italic { size * 0.95 } else { size };
            self.queue_text(&current_text, Vec2::new(current_x, current_y), color, scale);
        }

        current_y + line_height - position.y
    }

    pub fn render(&mut self, app: &mut App) -> anyhow::Result<()> {
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
        
        // Render special components that aren't in Root tree (background, glow, modals, toasts)
        // These are kept separate for now but could be added to Root later
        use crate::gfx::components::{background, glow, modals, toasts};
        
        // Render background (z-order 0)
        vertices.clear();
        background::render_background(self, app, &mut vertices);
        if !vertices.is_empty() {
            self.add_vertices(&vertices, None);
        }
        
        // Render glow effects (z-order 5)
        vertices.clear();
        glow::render_sidebar_glow(self, app, &mut vertices);
        if !vertices.is_empty() {
            self.add_vertices(&vertices, None);
        }
        
        // Update Root layout only when layout_generation changed (viewport, sidebar, tab, etc.).
        let root_rect = crate::ui::core::Rect::new(0.0, 0.0, app.viewport_size.x, app.viewport_size.y);
        if app.root.last_layout_generation != app.layout_generation {
            app.root.last_layout_generation = app.layout_generation;
            app.root.update_layout(root_rect, None, None);
        }

        // Constellation: update node sizes only when graph content or scale changed.
        let constellation_key = app.graph_state.graph_id.as_ref().and_then(|_| {
            app.chat_window.as_ref().map(|chat| {
                let scale = chat.constellation_view.scale_animated;
                let scale_bucket = (scale * 4.0).round() as u32;
                (app.graph_state.content_version, scale_bucket)
            })
        });
        let run_update_node_sizes = constellation_key != self.last_constellation_node_sizes_key;
        if run_update_node_sizes {
            self.last_constellation_node_sizes_key = constellation_key;
        }
        if app.graph_state.graph_id.is_some() && run_update_node_sizes {
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
                app.viewport_size,
                |text, size| self.measure_text(text, size),
                editing_override,
                visible_rect,
            );
        }

        // Render Root component tree (no dirty rect culling).
        let app_ref: &App = &*app;
        vertices.clear();
        app.root.render(self, app_ref, &mut vertices, None);
        
        // Render toasts (z-order 90)
        vertices.clear();
        toasts::render_toasts(self, app, &mut vertices);
        if !vertices.is_empty() {
            self.add_vertices(&vertices, None);
        }
        
        // Flush and merge MAIN batches only (so modals can be drawn on top later)
        self.flush_current_batch();
        self.merge_compatible_batches();

        // Full clear every frame (no dirty rect / partial redraw).
        const BG_R: f32 = 0.1;
        const BG_G: f32 = 0.1;
        const BG_B: f32 = 0.12;
        let dirty_scissor_opt: Option<ScissorRect> = None;

        let main_vertex_count: usize = self.render_batches.iter().map(|b| b.vertices.len()).sum();

        // Capture main text/icon before modal render (so we can blit main first, then modal on top)
        let main_text_commands: Vec<_> = self.text_queue.drain(..).collect();
        let main_icon_commands: Vec<_> = self.icon_queue.drain(..).collect();

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
        // Do not merge modal batches with main — we draw modal geometry after main text/icon blit

        let total_vertex_count: usize = self.render_batches.iter().map(|b| b.vertices.len()).sum();
        self.vertex_count = total_vertex_count as u32;

        let modal_text_commands: Vec<_> = self.text_queue.drain(..).collect();
        let modal_icon_commands: Vec<_> = self.icon_queue.drain(..).collect();
        
        // Clear previous text and icon scenes
        self.text_scenes.clear();
        self.icon_scenes.clear();
        
        // Store text scenes for sequential render/blit after main render pass
        let mut text_scenes: Vec<(Option<ScissorRect>, Scene)> = Vec::new();
        let mut icon_scenes: Vec<(Option<ScissorRect>, Scene)> = Vec::new();
        
        fn build_text_scenes(renderer: &mut Renderer, commands: &[TextDrawCommand]) -> Vec<(Option<ScissorRect>, Scene)> {
            use std::collections::HashMap;
            let mut text_groups: HashMap<Option<ScissorRect>, Vec<TextDrawCommand>> = HashMap::new();
            for cmd in commands {
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
                        None,
                        [cmd.color.x, cmd.color.y, cmd.color.z, cmd.color.w],
                    );
                }
                out.push((scissor_opt, scene));
            }
            out
        }
        fn build_icon_scenes(renderer: &mut Renderer, commands: &[IconDrawCommand]) -> Vec<(Option<ScissorRect>, Scene)> {
            use std::collections::HashMap;
            let mut icon_groups: HashMap<Option<ScissorRect>, Vec<IconDrawCommand>> = HashMap::new();
            for cmd in commands {
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

        let main_text_scenes = build_text_scenes(self, &main_text_commands);
        let main_icon_scenes = build_icon_scenes(self, &main_icon_commands);
        let modal_text_scenes = build_text_scenes(self, &modal_text_commands);
        let modal_icon_scenes = build_icon_scenes(self, &modal_icon_commands);
        self.text_scenes.clear();
        self.icon_scenes.clear();
        
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
        
        // Create main render pass for quads (always full clear).
        {
            let use_partial = false;
            let load_op = if use_partial {
                wgpu::LoadOp::Load
            } else {
                wgpu::LoadOp::Clear(wgpu::Color {
                    r: BG_R as f64,
                    g: BG_G as f64,
                    b: BG_B as f64,
                    a: 1.0,
                })
            };
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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

            // Draw UI quads in batches (each batch may have different scissor rect)
            if self.vertex_count > 0 {
                render_pass.set_pipeline(&self.pipeline);
                render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                
                // Set vertex buffer once before the loop
                render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                
                let mut vertex_offset = 0u32;
                let main_vertex_count_u32 = main_vertex_count as u32;
                for batch in self.render_batches.iter() {
                    let batch_vertex_count = batch.vertices.len() as u32;
                    if batch_vertex_count == 0 {
                        continue;
                    }
                    if vertex_offset >= main_vertex_count_u32 {
                        break;
                    }
                    let draw_end = (vertex_offset + batch_vertex_count).min(main_vertex_count_u32);
                    if draw_end <= vertex_offset {
                        vertex_offset += batch_vertex_count;
                        continue;
                    }
                    let scissor = if let Some(ref ds) = dirty_scissor_opt {
                        let effective = batch.scissor
                            .map(|s| s.intersect(ds))
                            .unwrap_or(*ds);
                        if effective.width == 0 || effective.height == 0 {
                            vertex_offset += batch_vertex_count;
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
                    render_pass.draw(vertex_offset..draw_end, 0..1);
                    vertex_offset += batch_vertex_count;
                }
            }
        }

        // Blit main text (so modal geometry can be drawn on top of it)
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
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load, // Load existing content (quads)
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                
                // Set pipeline once for all draws
                text_render_pass.set_pipeline(&self.blit_pipeline);
                
                // Draw each texture group with its scissor (intersect with dirty when partial redraw)
                for (scissor_opt, _texture, _texture_view, bind_group) in texture_bind_groups.iter() {
                    let (x, y, width, height) = if let Some(ref ds) = dirty_scissor_opt {
                        let eff = scissor_opt.map(|s| s.intersect(ds)).unwrap_or(*ds);
                        if eff.width == 0 || eff.height == 0 {
                            continue;
                        }
                        (eff.x, eff.y, eff.width, eff.height)
                    } else if let Some(scissor) = scissor_opt {
                        (scissor.x, scissor.y, scissor.width, scissor.height)
                    } else {
                        (0u32, 0u32, self.config.width, self.config.height)
                    };
                    text_render_pass.set_bind_group(0, bind_group, &[]);
                    let x = x.min(self.config.width);
                    let y = y.min(self.config.height);
                    let width = width.min(self.config.width.saturating_sub(x));
                    let height = height.min(self.config.height.saturating_sub(y));
                    text_render_pass.set_scissor_rect(x, y, width, height);
                    text_render_pass.draw(0..3, 0..1);
                }
                // Render pass ends here (dropped)
            }
            
            // Return textures to pool after use
            for (_, texture, texture_view, bind_group) in texture_bind_groups {
                self.return_text_texture(texture, texture_view, bind_group);
            }
        }

        // Blit text layers (constellation cached text)
        let text_layer_draws = std::mem::take(&mut self.text_layer_draws);
        if !text_layer_draws.is_empty() {
            let mut text_layer_bind_groups: Vec<(TextLayerDraw, wgpu::Buffer, wgpu::BindGroup)> = Vec::new();
            for draw in &text_layer_draws {
                if let Some(entry) = self.text_layer_cache.entries.get(&draw.key) {
                    let (x, y, w, h) = draw.dest_rect;
                    if w <= 0.0 || h <= 0.0 {
                        continue;
                    }

                    let layer_w = entry.width as f32;
                    let layer_h = entry.height as f32;

                    // Logical full-rect UV range before screen clipping. Scroll is implemented
                    // by translating the destination quad; UVs always map the full layer.
                    let base_u_min = 0.0f32;
                    let base_u_max = 1.0f32;
                    let base_v_min = 0.0f32;
                    let base_v_max = 1.0f32;

                    // Compute how much of dest_rect is clipped by the screen rectangle.
                    let screen_w = self.config.width as f32;
                    let screen_h = self.config.height as f32;
                    let clip_left = (0.0 - x).max(0.0);
                    let clip_top = (0.0 - y).max(0.0);
                    let clip_right = (x + w - screen_w).max(0.0);
                    let clip_bottom = (y + h - screen_h).max(0.0);

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
                        view: &view,
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
                    let clip_left = (0.0 - x).max(0.0);
                    let clip_top = (0.0 - y).max(0.0);
                    let clip_right = (x + w - screen_w).max(0.0);
                    let clip_bottom = (y + h - screen_h).max(0.0);

                    let visible_w = (w - clip_left - clip_right).max(0.0);
                    let visible_h = (h - clip_top - clip_bottom).max(0.0);
                    if visible_w <= 0.0 || visible_h <= 0.0 {
                        continue;
                    }

                    let vp_x = (x + clip_left).max(0.0) as u32;
                    let vp_y = (y + clip_top).max(0.0) as u32;
                    let vp_w = visible_w as u32;
                    let vp_h = visible_h as u32;

                    if vp_w > 0 && vp_h > 0 {
                        text_layer_pass.set_bind_group(0, bind_group, &[]);
                        text_layer_pass.set_viewport(
                            vp_x as f32,
                            vp_y as f32,
                            vp_w as f32,
                            vp_h as f32,
                            0.0,
                            1.0,
                        );
                        if let Some(s) = draw.scissor {
                            // Intersect text-layer viewport with the content scissor in absolute coordinates.
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
                            }
                        }
                        text_layer_pass.draw(0..3, 0..1);
                    }
                }
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
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load, // Load existing content
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                
                // Set pipeline once for all draws
                icon_render_pass.set_pipeline(&self.blit_pipeline);
                
                // Draw each icon texture group with its scissor (intersect with dirty when partial redraw)
                for (scissor_opt, _texture, _texture_view, bind_group) in icon_texture_bind_groups.iter() {
                    let (x, y, width, height) = if let Some(ref ds) = dirty_scissor_opt {
                        let eff = scissor_opt.map(|s| s.intersect(ds)).unwrap_or(*ds);
                        if eff.width == 0 || eff.height == 0 {
                            continue;
                        }
                        (eff.x, eff.y, eff.width, eff.height)
                    } else if let Some(scissor) = scissor_opt {
                        (scissor.x, scissor.y, scissor.width, scissor.height)
                    } else {
                        (0u32, 0u32, self.config.width, self.config.height)
                    };
                    icon_render_pass.set_bind_group(0, bind_group, &[]);
                    let x = x.min(self.config.width);
                    let y = y.min(self.config.height);
                    let width = width.min(self.config.width.saturating_sub(x));
                    let height = height.min(self.config.height.saturating_sub(y));
                    icon_render_pass.set_scissor_rect(x, y, width, height);
                    icon_render_pass.draw(0..3, 0..1);
                }
                // Render pass ends here (dropped)
            }
            
            for (_, texture, texture_view, bind_group) in icon_texture_bind_groups {
                self.return_icon_texture(texture, texture_view, bind_group);
            }
        }

        // Draw modal quads on top of main content (so modals are above chat text)
        if main_vertex_count < total_vertex_count && self.vertex_count > 0 {
            let mut modal_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("modal_quads"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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
            modal_pass.set_pipeline(&self.pipeline);
            modal_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            modal_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            let mut vertex_offset = 0u32;
            let main_v = main_vertex_count as u32;
            let total_v = total_vertex_count as u32;
            for batch in self.render_batches.iter() {
                let batch_vertex_count = batch.vertices.len() as u32;
                if batch_vertex_count == 0 {
                    continue;
                }
                if vertex_offset + batch_vertex_count <= main_v {
                    vertex_offset += batch_vertex_count;
                    continue;
                }
                let draw_start = vertex_offset.max(main_v);
                let draw_end = (vertex_offset + batch_vertex_count).min(total_v);
                if draw_end <= draw_start {
                    vertex_offset += batch_vertex_count;
                    continue;
                }
                if let Some(scissor) = &batch.scissor {
                    let x = scissor.x.min(self.config.width);
                    let y = scissor.y.min(self.config.height);
                    let width = scissor.width.min(self.config.width.saturating_sub(x));
                    let height = scissor.height.min(self.config.height.saturating_sub(y));
                    modal_pass.set_scissor_rect(x, y, width, height);
                } else {
                    modal_pass.set_scissor_rect(0, 0, self.config.width, self.config.height);
                }
                modal_pass.draw(draw_start..draw_end, 0..1);
                vertex_offset += batch_vertex_count;
            }
        }

        // Blit modal text and icons on top of modal quads
        if !modal_text_scenes.is_empty() {
            let mut texture_bind_groups: Vec<(Option<ScissorRect>, wgpu::Texture, wgpu::TextureView, wgpu::BindGroup)> = Vec::new();
            for (scissor_opt, scene) in modal_text_scenes.iter() {
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
                    .expect("Failed to render modal vello scene");
                texture_bind_groups.push((*scissor_opt, group_texture, group_texture_view, group_bind_group));
            }
            if !texture_bind_groups.is_empty() {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("modal_text_blit"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&self.blit_pipeline);
                for (scissor_opt, _t, _tv, bind_group) in texture_bind_groups.iter() {
                    pass.set_bind_group(0, bind_group, &[]);
                    if let Some(s) = scissor_opt {
                        pass.set_scissor_rect(s.x.min(self.config.width), s.y.min(self.config.height),
                            s.width.min(self.config.width.saturating_sub(s.x.min(self.config.width))),
                            s.height.min(self.config.height.saturating_sub(s.y.min(self.config.height))));
                    } else {
                        pass.set_scissor_rect(0, 0, self.config.width, self.config.height);
                    }
                    pass.draw(0..3, 0..1);
                }
            }
            for (_, texture, texture_view, bind_group) in texture_bind_groups {
                self.return_text_texture(texture, texture_view, bind_group);
            }
        }
        if !modal_icon_scenes.is_empty() {
            let mut icon_bind_groups: Vec<(Option<ScissorRect>, wgpu::Texture, wgpu::TextureView, wgpu::BindGroup)> = Vec::new();
            for (scissor_opt, scene) in modal_icon_scenes.iter() {
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
                    .expect("Failed to render modal icon scene");
                icon_bind_groups.push((*scissor_opt, icon_texture, icon_texture_view, icon_bind_group));
            }
            if !icon_bind_groups.is_empty() {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("modal_icon_blit"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&self.blit_pipeline);
                for (scissor_opt, _t, _tv, bind_group) in icon_bind_groups.iter() {
                    pass.set_bind_group(0, bind_group, &[]);
                    if let Some(s) = scissor_opt {
                        pass.set_scissor_rect(s.x.min(self.config.width), s.y.min(self.config.height),
                            s.width.min(self.config.width.saturating_sub(s.x.min(self.config.width))),
                            s.height.min(self.config.height.saturating_sub(s.y.min(self.config.height))));
                    } else {
                        pass.set_scissor_rect(0, 0, self.config.width, self.config.height);
                    }
                    pass.draw(0..3, 0..1);
                }
            }
            for (_, texture, texture_view, bind_group) in icon_bind_groups {
                self.return_icon_texture(texture, texture_view, bind_group);
            }
        }

        // Submit encoder (staging belt must be finished before this)
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
