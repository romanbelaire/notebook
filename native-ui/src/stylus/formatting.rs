use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextFormat {
    Bold,
    Italic,
    Underline,
    Strikethrough,
    Code,
    Link { url: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatSpan {
    pub start: usize,
    pub end: usize,
    pub format: TextFormat,
}

impl FormatSpan {
    pub fn new(start: usize, end: usize, format: TextFormat) -> Self {
        Self { start, end, format }
    }

    pub fn contains(&self, pos: usize) -> bool {
        pos >= self.start && pos < self.end
    }

    pub fn overlaps(&self, start: usize, end: usize) -> bool {
        !(self.end <= start || self.start >= end)
    }
}

pub fn apply_formatting(text: &str, formats: &[FormatSpan], pos: usize) -> Vec<TextFormat> {
    formats
        .iter()
        .filter(|span| span.contains(pos))
        .map(|span| span.format.clone())
        .collect()
}

pub fn has_format(formats: &[FormatSpan], pos: usize, format: TextFormat) -> bool {
    formats
        .iter()
        .any(|span| span.contains(pos) && span.format == format)
}

