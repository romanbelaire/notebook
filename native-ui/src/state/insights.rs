use serde::{Serialize, Deserialize};
use crate::api::models::Insight;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InsightsState {
    pub insights: Vec<Insight>,
    pub modal_insight: Option<Insight>,
}

impl InsightsState {
    pub fn new() -> Self {
        Self {
            insights: Vec::new(),
            modal_insight: None,
        }
    }

    pub fn set_insights(&mut self, insights: Vec<Insight>) {
        self.insights = insights;
    }

    pub fn add_insight(&mut self, insight: Insight) {
        self.insights.insert(0, insight);
    }

    pub fn remove_insight(&mut self, id: &str) {
        self.insights.retain(|i| i.id != id);
        if let Some(ref modal) = self.modal_insight {
            if modal.id == id {
                self.modal_insight = None;
            }
        }
    }

    pub fn update_insight_title(&mut self, id: &str, title: String) {
        for insight in &mut self.insights {
            if insight.id == id {
                insight.title = title.clone();
            }
        }
        if let Some(ref mut modal) = self.modal_insight {
            if modal.id == id {
                modal.title = title;
            }
        }
    }

    pub fn update_insight_text(&mut self, id: &str, text: String) {
        for insight in &mut self.insights {
            if insight.id == id {
                insight.text = text.clone();
            }
        }
        if let Some(ref mut modal) = self.modal_insight {
            if modal.id == id {
                modal.text = text;
            }
        }
    }

    pub fn set_modal_insight(&mut self, insight: Option<Insight>) {
        self.modal_insight = insight;
    }
}

impl Default for InsightsState {
    fn default() -> Self {
        Self::new()
    }
}

