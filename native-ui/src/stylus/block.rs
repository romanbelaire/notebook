use serde::{Serialize, Deserialize};
use crate::stylus::formatting::FormatSpan;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum BlockType {
    Paragraph,
    Heading1,
    Heading2,
    Heading3,
    Heading4,
    Heading5,
    Code,
    List,
    NumberedList,
    Quote,
    Divider,
    Checkbox,
    Image,
    Table,
    Embed,
    Latex,
    Mermaid,
    Chart,
    Canvas,
    Video,
    Callout,
    Accordion,
    Columns,
    Database,
}

impl BlockType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "text-element" | "paragraph" => Some(BlockType::Paragraph),
            "heading1-element" | "h1" => Some(BlockType::Heading1),
            "heading2-element" | "h2" => Some(BlockType::Heading2),
            "heading3-element" | "h3" => Some(BlockType::Heading3),
            "heading4-element" | "h4" => Some(BlockType::Heading4),
            "heading5-element" | "h5" => Some(BlockType::Heading5),
            "code-element" | "code" => Some(BlockType::Code),
            "list-element" | "list" => Some(BlockType::List),
            "numbered-list-element" | "numbered-list" => Some(BlockType::NumberedList),
            "quote-element" | "quote" => Some(BlockType::Quote),
            "divider-element" | "divider" => Some(BlockType::Divider),
            "checkbox-element" | "checkbox" => Some(BlockType::Checkbox),
            "image-element" | "image" => Some(BlockType::Image),
            "table-element" | "table" => Some(BlockType::Table),
            "embed-element" | "embed" => Some(BlockType::Embed),
            "latex-element" | "latex" => Some(BlockType::Latex),
            "mermaid-element" | "mermaid" => Some(BlockType::Mermaid),
            "chart-element" | "chart" => Some(BlockType::Chart),
            "canvas-element" | "canvas" => Some(BlockType::Canvas),
            "video-element" | "video" => Some(BlockType::Video),
            "callout-element" | "callout" => Some(BlockType::Callout),
            "accordion-element" | "accordion" => Some(BlockType::Accordion),
            "columns-element" | "columns" => Some(BlockType::Columns),
            "database-element" | "database" => Some(BlockType::Database),
            _ => None,
        }
    }

    pub fn to_str(&self) -> &'static str {
        match self {
            BlockType::Paragraph => "text-element",
            BlockType::Heading1 => "heading1-element",
            BlockType::Heading2 => "heading2-element",
            BlockType::Heading3 => "heading3-element",
            BlockType::Heading4 => "heading4-element",
            BlockType::Heading5 => "heading5-element",
            BlockType::Code => "code-element",
            BlockType::List => "list-element",
            BlockType::NumberedList => "numbered-list-element",
            BlockType::Quote => "quote-element",
            BlockType::Divider => "divider-element",
            BlockType::Checkbox => "checkbox-element",
            BlockType::Image => "image-element",
            BlockType::Table => "table-element",
            BlockType::Embed => "embed-element",
            BlockType::Latex => "latex-element",
            BlockType::Mermaid => "mermaid-element",
            BlockType::Chart => "chart-element",
            BlockType::Canvas => "canvas-element",
            BlockType::Video => "video-element",
            BlockType::Callout => "callout-element",
            BlockType::Accordion => "accordion-element",
            BlockType::Columns => "columns-element",
            BlockType::Database => "database-element",
        }
    }

    pub fn default_text(&self) -> &'static str {
        match self {
            BlockType::Paragraph => "",
            BlockType::Heading1 => "Heading 1",
            BlockType::Heading2 => "Heading 2",
            BlockType::Heading3 => "Heading 3",
            BlockType::Heading4 => "Heading 4",
            BlockType::Heading5 => "Heading 5",
            BlockType::Code => "",
            BlockType::List => "",
            BlockType::NumberedList => "",
            BlockType::Quote => "",
            BlockType::Divider => "",
            BlockType::Checkbox => "",
            BlockType::Image => "",
            BlockType::Table => "",
            BlockType::Embed => "",
            BlockType::Latex => "",
            BlockType::Mermaid => "",
            BlockType::Chart => "",
            BlockType::Canvas => "",
            BlockType::Video => "",
            BlockType::Callout => "",
            BlockType::Accordion => "",
            BlockType::Columns => "",
            BlockType::Database => "",
        }
    }

    pub fn is_text_block(&self) -> bool {
        matches!(
            self,
            BlockType::Paragraph
                | BlockType::Heading1
                | BlockType::Heading2
                | BlockType::Heading3
                | BlockType::Heading4
                | BlockType::Heading5
                | BlockType::Code
                | BlockType::List
                | BlockType::NumberedList
                | BlockType::Quote
                | BlockType::Checkbox
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlockContent {
    Text {
        text: String,
        formats: Vec<FormatSpan>,
    },
    Image {
        url: String,
        alt: Option<String>,
        width: Option<f32>,
        height: Option<f32>,
    },
    Code {
        code: String,
        language: Option<String>,
    },
    Table {
        rows: Vec<Vec<String>>,
    },
    Embed {
        url: String,
        embed_type: String,
    },
    Latex {
        formula: String,
    },
    Mermaid {
        diagram: String,
    },
    Chart {
        chart_type: String,
        data: serde_json::Value,
    },
    Video {
        url: String,
        embed_type: String,
    },
    Checkbox {
        checked: bool,
        text: String,
    },
    Divider,
    Callout {
        callout_type: String,
        text: String,
    },
    Accordion {
        title: String,
        content: Vec<Block>,
    },
    Columns {
        columns: Vec<Vec<Block>>,
    },
    Database {
        schema: serde_json::Value,
    },
    Canvas {
        data: serde_json::Value,
    },
}

impl BlockContent {
    pub fn text(text: String) -> Self {
        BlockContent::Text {
            text,
            formats: Vec::new(),
        }
    }

    pub fn get_text(&self) -> Option<&str> {
        match self {
            BlockContent::Text { text, .. } => Some(text),
            BlockContent::Code { code, .. } => Some(code),
            BlockContent::Checkbox { text, .. } => Some(text),
            _ => None,
        }
    }

    pub fn get_text_mut(&mut self) -> Option<&mut String> {
        match self {
            BlockContent::Text { text, .. } => Some(text),
            BlockContent::Code { code, .. } => Some(code),
            BlockContent::Checkbox { text, .. } => Some(text),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub id: String,
    pub block_type: BlockType,
    pub content: BlockContent,
    pub collapsed: bool,
    pub metadata: serde_json::Value,
}

impl Block {
    pub fn new(block_type: BlockType) -> Self {
        let id = format!("block_{}", uuid::Uuid::new_v4().to_string().replace("-", ""));
        let default_text = block_type.default_text().to_string();
        
        let content = match block_type {
            BlockType::Paragraph
            | BlockType::Heading1
            | BlockType::Heading2
            | BlockType::Heading3
            | BlockType::Heading4
            | BlockType::Heading5
            | BlockType::Quote => BlockContent::text(default_text),
            BlockType::Code => BlockContent::Code {
                code: String::new(),
                language: None,
            },
            BlockType::List | BlockType::NumberedList => BlockContent::text(String::new()),
            BlockType::Checkbox => BlockContent::Checkbox {
                checked: false,
                text: String::new(),
            },
            BlockType::Divider => BlockContent::Divider,
            BlockType::Image => BlockContent::Image {
                url: String::new(),
                alt: None,
                width: None,
                height: None,
            },
            BlockType::Table => BlockContent::Table {
                rows: vec![vec![String::new(); 2]; 2],
            },
            BlockType::Embed => BlockContent::Embed {
                url: String::new(),
                embed_type: String::new(),
            },
            BlockType::Latex => BlockContent::Latex {
                formula: String::new(),
            },
            BlockType::Mermaid => BlockContent::Mermaid {
                diagram: String::new(),
            },
            BlockType::Chart => BlockContent::Chart {
                chart_type: "bar".to_string(),
                data: serde_json::json!({}),
            },
            BlockType::Video => BlockContent::Video {
                url: String::new(),
                embed_type: "youtube".to_string(),
            },
            BlockType::Callout => BlockContent::Callout {
                callout_type: "info".to_string(),
                text: String::new(),
            },
            BlockType::Accordion => BlockContent::Accordion {
                title: String::new(),
                content: Vec::new(),
            },
            BlockType::Columns => BlockContent::Columns {
                columns: vec![Vec::new(); 2],
            },
            BlockType::Database => BlockContent::Database {
                schema: serde_json::json!({}),
            },
            BlockType::Canvas => BlockContent::Canvas {
                data: serde_json::json!({}),
            },
        };

        Self {
            id,
            block_type,
            content,
            collapsed: false,
            metadata: serde_json::json!({}),
        }
    }

    pub fn is_empty(&self) -> bool {
        match &self.content {
            BlockContent::Text { text, .. } => text.trim().is_empty(),
            BlockContent::Code { code, .. } => code.trim().is_empty(),
            BlockContent::Checkbox { text, .. } => text.trim().is_empty(),
            BlockContent::Image { url, .. } => url.is_empty(),
            BlockContent::Table { rows } => rows.is_empty() || rows.iter().all(|r| r.iter().all(|c| c.trim().is_empty())),
            BlockContent::Embed { url, .. } => url.is_empty(),
            BlockContent::Latex { formula } => formula.trim().is_empty(),
            BlockContent::Mermaid { diagram } => diagram.trim().is_empty(),
            BlockContent::Video { url, .. } => url.is_empty(),
            BlockContent::Divider => false,
            BlockContent::Callout { text, .. } => text.trim().is_empty(),
            BlockContent::Accordion { title, content } => title.trim().is_empty() && content.is_empty(),
            BlockContent::Columns { columns } => columns.is_empty() || columns.iter().all(|c| c.is_empty()),
            BlockContent::Database { .. } => false,
            BlockContent::Chart { .. } => false,
            BlockContent::Canvas { .. } => false,
        }
    }
}

