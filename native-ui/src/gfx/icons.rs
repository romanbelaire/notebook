use std::collections::HashMap;
use usvg::{Tree, Options, fontdb};
use vello::kurbo::{BezPath, PathEl, Point};

/// Cached SVG paths for an icon
#[derive(Clone)]
pub struct IconPaths {
    pub fill_paths: Vec<BezPath>,
    pub stroke_paths: Vec<(BezPath, f64)>, // path and stroke width
    /// ViewBox width from the SVG (used to scale icon to requested pixel size)
    pub viewbox_size: f64,
}

/// Cache for parsed SVG icons
pub struct IconCache {
    cache: HashMap<String, IconPaths>,
    registry: IconRegistry,
}

impl IconCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            registry: IconRegistry::new(),
        }
    }

    /// Get cached paths for an icon, parsing if necessary
    pub fn get_paths(&mut self, icon_name: &str) -> Option<&IconPaths> {
        if !self.cache.contains_key(icon_name) {
            if let Some(svg_content) = self.registry.get_svg(icon_name) {
                if let Ok(paths) = parse_svg_to_paths(svg_content) {
                    self.cache.insert(icon_name.to_string(), paths);
                } else {
                    eprintln!("Failed to parse SVG for icon: {}", icon_name);
                    return None;
                }
            } else {
                eprintln!("Icon not found: {}", icon_name);
                return None;
            }
        }
        self.cache.get(icon_name)
    }
}

/// Registry of embedded SVG icons
struct IconRegistry {
    icons: HashMap<String, &'static str>,
}

impl IconRegistry {
    fn new() -> Self {
        let mut icons = HashMap::new();
        
        // Embed all SVG icons at compile time
        icons.insert("close".to_string(), include_str!("../../assets/icons/close.svg"));
        icons.insert("plus".to_string(), include_str!("../../assets/icons/plus.svg"));
        icons.insert("trash".to_string(), include_str!("../../assets/icons/trash.svg"));
        icons.insert("pencil".to_string(), include_str!("../../assets/icons/pencil.svg"));
        icons.insert("magnify".to_string(), include_str!("../../assets/icons/magnify.svg"));
        icons.insert("pin".to_string(), include_str!("../../assets/icons/pin.svg"));
        icons.insert("pin-red".to_string(), include_str!("../../assets/icons/pin-red.svg"));
        icons.insert("book".to_string(), include_str!("../../assets/icons/book.svg"));
        icons.insert("gear".to_string(), include_str!("../../assets/icons/gear.svg"));
        icons.insert("chevron-left".to_string(), include_str!("../../assets/icons/chevron-left.svg"));
        icons.insert("chevron-right".to_string(), include_str!("../../assets/icons/chevron-right.svg"));
        icons.insert("chevron-up".to_string(), include_str!("../../assets/icons/chevron-up.svg"));
        icons.insert("chevron-down".to_string(), include_str!("../../assets/icons/chevron-down.svg"));
        icons.insert("eye-open".to_string(), include_str!("../../assets/icons/eye-open.svg"));
        icons.insert("eye-closed".to_string(), include_str!("../../assets/icons/eye-closed.svg"));
        icons.insert("folder".to_string(), include_str!("../../assets/icons/folder.svg"));
        icons.insert("folder-yellow".to_string(), include_str!("../../assets/icons/folder-yellow.svg"));
        icons.insert("question".to_string(), include_str!("../../assets/icons/question.svg"));
        icons.insert("dots-6-vertical".to_string(), include_str!("../../assets/icons/dots-6-vertical.svg"));
        icons.insert("minimize".to_string(), include_str!("../../assets/icons/minimize.svg"));
        icons.insert("maximize".to_string(), include_str!("../../assets/icons/maximize.svg"));
        icons.insert("windowed".to_string(), include_str!("../../assets/icons/windowed.svg"));
        icons.insert("zap".to_string(), include_str!("../../assets/icons/zap.svg"));
        icons.insert("zap-yellow".to_string(), include_str!("../../assets/icons/zap-yellow.svg"));
        icons.insert("save".to_string(), include_str!("../../assets/icons/save.svg"));
        
        Self { icons }
    }

    fn get_svg(&self, name: &str) -> Option<&'static str> {
        self.icons.get(name).copied()
    }
}

/// Parse SVG string to kurbo BezPath objects
fn parse_svg_to_paths(svg_content: &str) -> Result<IconPaths, String> {
    let opt = Options::default();
    let mut fontdb = fontdb::Database::new();
    fontdb.load_system_fonts();
    let tree = Tree::from_str(svg_content, &opt, &fontdb)
        .map_err(|e| format!("Failed to parse SVG: {}", e))?;
    
    let mut fill_paths = Vec::new();
    let mut stroke_paths = Vec::new();
    
    // Traverse the tree and extract paths
    // tree.root() returns &Group, iterate over its children
    for child in tree.root().children() {
        extract_paths_from_node(child, &mut fill_paths, &mut stroke_paths);
    }
    
    // Use viewBox width for scaling; path coordinates are in viewBox space
    let viewbox_size = tree.view_box().rect.width() as f64;
    
    Ok(IconPaths {
        fill_paths,
        stroke_paths,
        viewbox_size,
    })
}

/// Recursively extract paths from SVG nodes
fn extract_paths_from_node(
    node: &usvg::Node,
    fill_paths: &mut Vec<BezPath>,
    stroke_paths: &mut Vec<(BezPath, f64)>,
) {
    match node {
        usvg::Node::Path(path_node) => {
            // Convert usvg path data to kurbo BezPath
            // usvg stores paths as tiny-skia paths internally
            let path_data = path_node.data();
            
            // Convert tiny-skia path to kurbo BezPath
            let mut bez_path = BezPath::new();
            
            // Iterate through path segments
            for segment in path_data.segments() {
                match segment {
                    tiny_skia_path::PathSegment::MoveTo(p) => {
                        bez_path.move_to(Point::new(p.x as f64, p.y as f64));
                    }
                    tiny_skia_path::PathSegment::LineTo(p) => {
                        bez_path.line_to(Point::new(p.x as f64, p.y as f64));
                    }
                    tiny_skia_path::PathSegment::QuadTo(p1, p2) => {
                        // Get current point for quadratic to cubic conversion
                        let p0 = bez_path.elements().last()
                            .and_then(|el| match el {
                                PathEl::MoveTo(p) | PathEl::LineTo(p) | PathEl::CurveTo(_, _, p) | PathEl::QuadTo(_, p) => Some(*p),
                                PathEl::ClosePath => None,
                            })
                            .unwrap_or(Point::new(0.0, 0.0));
                        let p1_pt = Point::new(p1.x as f64, p1.y as f64);
                        let p2_pt = Point::new(p2.x as f64, p2.y as f64);
                        // Quadratic to cubic conversion
                        let cp1 = Point::new(
                            p0.x + 2.0 / 3.0 * (p1_pt.x - p0.x),
                            p0.y + 2.0 / 3.0 * (p1_pt.y - p0.y),
                        );
                        let cp2 = Point::new(
                            p2_pt.x + 2.0 / 3.0 * (p1_pt.x - p2_pt.x),
                            p2_pt.y + 2.0 / 3.0 * (p1_pt.y - p2_pt.y),
                        );
                        bez_path.curve_to(cp1, cp2, p2_pt);
                    }
                    tiny_skia_path::PathSegment::CubicTo(p1, p2, p3) => {
                        bez_path.curve_to(
                            Point::new(p1.x as f64, p1.y as f64),
                            Point::new(p2.x as f64, p2.y as f64),
                            Point::new(p3.x as f64, p3.y as f64),
                        );
                    }
                    tiny_skia_path::PathSegment::Close => {
                        bez_path.close_path();
                    }
                }
            }
            
            // Check if this path has fill or stroke
            if path_node.fill().is_some() {
                fill_paths.push(bez_path.clone());
            }
            
            if let Some(ref stroke) = path_node.stroke() {
                let stroke_width = stroke.width().get() as f64;
                stroke_paths.push((bez_path, stroke_width));
            }
        }
        usvg::Node::Group(group) => {
            // Recursively process children
            for child in group.children() {
                extract_paths_from_node(child, fill_paths, stroke_paths);
            }
        }
        _ => {}
    }
}

