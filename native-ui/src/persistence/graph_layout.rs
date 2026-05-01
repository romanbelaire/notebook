//! Persist constellation node positions per graph. Stored in data/graph_layouts/{graph_id}.json

use crate::persistence::get_data_dir;
use glam::Vec2;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(serde::Serialize, serde::Deserialize)]
struct LayoutFile {
    positions: HashMap<String, [f32; 2]>,
    #[serde(default)]
    sizes: HashMap<String, [f32; 2]>,
}

pub struct GraphLayoutPersistence;

impl GraphLayoutPersistence {
    fn get_layouts_dir() -> Result<PathBuf, std::io::Error> {
        let data_dir = get_data_dir()?;
        let layouts_dir = data_dir.join("graph_layouts");
        if !layouts_dir.exists() {
            fs::create_dir_all(&layouts_dir)?;
        }
        Ok(layouts_dir)
    }

    fn get_layout_path(graph_id: &str) -> Result<PathBuf, std::io::Error> {
        let layouts_dir = Self::get_layouts_dir()?;
        Ok(layouts_dir.join(format!("{}.json", graph_id)))
    }

    /// Save node positions and sizes for a graph. Throttle calls (e.g. every 2–3 seconds).
    pub fn save_positions(
        graph_id: &str,
        positions: &HashMap<String, Vec2>,
        sizes: &HashMap<String, Vec2>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path = Self::get_layout_path(graph_id)?;
        let positions_ser: HashMap<String, [f32; 2]> = positions
            .iter()
            .map(|(k, v)| (k.clone(), [v.x, v.y]))
            .collect();
        let sizes_ser: HashMap<String, [f32; 2]> = sizes
            .iter()
            .map(|(k, v)| (k.clone(), [v.x, v.y]))
            .collect();
        let file = LayoutFile {
            positions: positions_ser,
            sizes: sizes_ser,
        };
        let json = serde_json::to_string_pretty(&file)?;
        fs::write(&path, json)?;
        Ok(())
    }

    /// Load stored positions and sizes for a graph. Returns empty maps if no file or error.
    pub fn load_positions(graph_id: &str) -> (HashMap<String, Vec2>, HashMap<String, Vec2>) {
        let path = match Self::get_layout_path(graph_id) {
            Ok(p) => p,
            Err(_) => return (HashMap::new(), HashMap::new()),
        };
        if !path.exists() {
            return (HashMap::new(), HashMap::new());
        }
        let json = match fs::read_to_string(&path) {
            Ok(j) => j,
            Err(_) => return (HashMap::new(), HashMap::new()),
        };
        let file: LayoutFile = match serde_json::from_str(&json) {
            Ok(f) => f,
            Err(_) => return (HashMap::new(), HashMap::new()),
        };
        let positions = file
            .positions
            .into_iter()
            .map(|(k, v)| (k, Vec2::new(v[0], v[1])))
            .collect();
        let sizes = file
            .sizes
            .into_iter()
            .map(|(k, v)| (k, Vec2::new(v[0], v[1])))
            .collect();
        (positions, sizes)
    }
}
