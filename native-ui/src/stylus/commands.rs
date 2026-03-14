use crate::stylus::block::BlockType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlashCommand {
    Heading1,
    Heading2,
    Heading3,
    Heading4,
    Heading5,
    Paragraph,
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
    Video,
    Callout,
    Accordion,
    Columns,
    Database,
}

impl SlashCommand {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "h1" | "heading1" | "heading 1" => Some(SlashCommand::Heading1),
            "h2" | "heading2" | "heading 2" => Some(SlashCommand::Heading2),
            "h3" | "heading3" | "heading 3" => Some(SlashCommand::Heading3),
            "h4" | "heading4" | "heading 4" => Some(SlashCommand::Heading4),
            "h5" | "heading5" | "heading 5" => Some(SlashCommand::Heading5),
            "p" | "paragraph" | "text" => Some(SlashCommand::Paragraph),
            "code" | "code block" => Some(SlashCommand::Code),
            "list" | "bullet" | "bullets" => Some(SlashCommand::List),
            "numbered" | "numbered list" | "ol" => Some(SlashCommand::NumberedList),
            "quote" | "blockquote" => Some(SlashCommand::Quote),
            "divider" | "hr" | "horizontal rule" => Some(SlashCommand::Divider),
            "checkbox" | "todo" | "task" => Some(SlashCommand::Checkbox),
            "image" | "img" => Some(SlashCommand::Image),
            "table" => Some(SlashCommand::Table),
            "embed" => Some(SlashCommand::Embed),
            "latex" | "math" => Some(SlashCommand::Latex),
            "mermaid" => Some(SlashCommand::Mermaid),
            "chart" => Some(SlashCommand::Chart),
            "video" => Some(SlashCommand::Video),
            "callout" => Some(SlashCommand::Callout),
            "accordion" => Some(SlashCommand::Accordion),
            "columns" => Some(SlashCommand::Columns),
            "database" => Some(SlashCommand::Database),
            _ => None,
        }
    }

    pub fn to_block_type(&self) -> BlockType {
        match self {
            SlashCommand::Heading1 => BlockType::Heading1,
            SlashCommand::Heading2 => BlockType::Heading2,
            SlashCommand::Heading3 => BlockType::Heading3,
            SlashCommand::Heading4 => BlockType::Heading4,
            SlashCommand::Heading5 => BlockType::Heading5,
            SlashCommand::Paragraph => BlockType::Paragraph,
            SlashCommand::Code => BlockType::Code,
            SlashCommand::List => BlockType::List,
            SlashCommand::NumberedList => BlockType::NumberedList,
            SlashCommand::Quote => BlockType::Quote,
            SlashCommand::Divider => BlockType::Divider,
            SlashCommand::Checkbox => BlockType::Checkbox,
            SlashCommand::Image => BlockType::Image,
            SlashCommand::Table => BlockType::Table,
            SlashCommand::Embed => BlockType::Embed,
            SlashCommand::Latex => BlockType::Latex,
            SlashCommand::Mermaid => BlockType::Mermaid,
            SlashCommand::Chart => BlockType::Chart,
            SlashCommand::Video => BlockType::Video,
            SlashCommand::Callout => BlockType::Callout,
            SlashCommand::Accordion => BlockType::Accordion,
            SlashCommand::Columns => BlockType::Columns,
            SlashCommand::Database => BlockType::Database,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            SlashCommand::Heading1 => "Heading 1",
            SlashCommand::Heading2 => "Heading 2",
            SlashCommand::Heading3 => "Heading 3",
            SlashCommand::Heading4 => "Heading 4",
            SlashCommand::Heading5 => "Heading 5",
            SlashCommand::Paragraph => "Paragraph",
            SlashCommand::Code => "Code Block",
            SlashCommand::List => "Bullet List",
            SlashCommand::NumberedList => "Numbered List",
            SlashCommand::Quote => "Quote",
            SlashCommand::Divider => "Divider",
            SlashCommand::Checkbox => "Checkbox",
            SlashCommand::Image => "Image",
            SlashCommand::Table => "Table",
            SlashCommand::Embed => "Embed",
            SlashCommand::Latex => "LaTeX",
            SlashCommand::Mermaid => "Mermaid Diagram",
            SlashCommand::Chart => "Chart",
            SlashCommand::Video => "Video",
            SlashCommand::Callout => "Callout",
            SlashCommand::Accordion => "Accordion",
            SlashCommand::Columns => "Columns",
            SlashCommand::Database => "Database",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            SlashCommand::Heading1 => "Large heading",
            SlashCommand::Heading2 => "Medium heading",
            SlashCommand::Heading3 => "Small heading",
            SlashCommand::Heading4 => "Smaller heading",
            SlashCommand::Heading5 => "Smallest heading",
            SlashCommand::Paragraph => "Plain text",
            SlashCommand::Code => "Code block with syntax highlighting",
            SlashCommand::List => "Bullet list",
            SlashCommand::NumberedList => "Numbered list",
            SlashCommand::Quote => "Quote block",
            SlashCommand::Divider => "Horizontal divider",
            SlashCommand::Checkbox => "Checkbox item",
            SlashCommand::Image => "Insert image",
            SlashCommand::Table => "Insert table",
            SlashCommand::Embed => "Embed content",
            SlashCommand::Latex => "LaTeX math formula",
            SlashCommand::Mermaid => "Mermaid diagram",
            SlashCommand::Chart => "Chart or graph",
            SlashCommand::Video => "Video embed",
            SlashCommand::Callout => "Callout box",
            SlashCommand::Accordion => "Collapsible accordion",
            SlashCommand::Columns => "Multi-column layout",
            SlashCommand::Database => "Database view",
        }
    }

    pub fn all_commands() -> Vec<SlashCommand> {
        vec![
            SlashCommand::Heading1,
            SlashCommand::Heading2,
            SlashCommand::Heading3,
            SlashCommand::Heading4,
            SlashCommand::Heading5,
            SlashCommand::Paragraph,
            SlashCommand::Code,
            SlashCommand::List,
            SlashCommand::NumberedList,
            SlashCommand::Quote,
            SlashCommand::Divider,
            SlashCommand::Checkbox,
            SlashCommand::Image,
            SlashCommand::Table,
            SlashCommand::Embed,
            SlashCommand::Latex,
            SlashCommand::Mermaid,
            SlashCommand::Chart,
            SlashCommand::Video,
            SlashCommand::Callout,
            SlashCommand::Accordion,
            SlashCommand::Columns,
            SlashCommand::Database,
        ]
    }

    pub fn search(query: &str) -> Vec<SlashCommand> {
        let query_lower = query.to_lowercase();
        Self::all_commands()
            .into_iter()
            .filter(|cmd| {
                cmd.name().to_lowercase().contains(&query_lower)
                    || cmd.description().to_lowercase().contains(&query_lower)
            })
            .collect()
    }
}

