use glam::Vec2;
use crate::api::models::Insight;
use crate::ui::SectionList;

const INSIGHTS_ITEM_HEIGHT: f32 = 35.0;

pub struct InsightsPanel {
    pub position: Vec2,
    pub size: Vec2,
    pub insights_list: SectionList,
    pub selected_insight_id: Option<String>,
}

impl InsightsPanel {
    pub fn new(position: Vec2, size: Vec2) -> Self {
        let padding = 10.0;
        let list_height = size.y - padding * 2.0;
        let list_pos = Vec2::new(position.x + padding, position.y + padding);
        let list_size = Vec2::new(size.x - padding * 2.0, list_height);
        Self {
            position,
            size,
            insights_list: SectionList::new(list_pos, list_size, INSIGHTS_ITEM_HEIGHT),
            selected_insight_id: None,
        }
    }

    pub fn update_layout(&mut self, position: Vec2, size: Vec2) {
        self.position = position;
        self.size = size;
        let padding = 10.0;
        let list_height = size.y - padding * 2.0;
        let list_pos = Vec2::new(position.x + padding, position.y + padding);
        let list_size = Vec2::new(size.x - padding * 2.0, list_height);
        self.insights_list.set_position_size(list_pos, list_size);
    }

    pub fn get_insight_at(&self, pos: Vec2, insights: &[Insight]) -> Option<usize> {
        self.insights_list.get_item_at(pos, insights.len())
    }

    pub fn contains(&self, pos: Vec2) -> bool {
        let rel = pos - self.position;
        rel.x >= 0.0 && rel.x <= self.size.x && rel.y >= 0.0 && rel.y <= self.size.y
    }
}
