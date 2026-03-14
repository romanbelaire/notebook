use crate::ui::{SubWindow, HeaderWindow, SidebarWindow, ChatWindow, LibraryWindow, IngestWindow, SettingsWindow, NotepadWindow, InsightModal, PdfModal, ChatInfoDialog, ToastManager, NotepadModal, TextEditor, GlobalSearchModal, ShardModal};
use crate::ui::tab_bar::Tab;
use crate::ui::chat_window::ChatMessage;
use crate::state::{ChatState, GraphState, UIState, SettingsState, InsightsState};
use crate::api::ApiClient;
use crate::api::models::Collection;
use crate::persistence::{ConversationPersistence, SettingsPersistence};
use glam::Vec2;
use winit::event::{ElementState, MouseButton};
use winit::event_loop::EventLoopProxy;
use winit::keyboard::{KeyCode, PhysicalKey, ModifiersState};
use std::sync::mpsc;

#[derive(Debug, Clone)]
pub enum WindowControlEvent {
    Minimize,
    ToggleMaximize,
    Close,
    DragWindow,
}

/// Pending click inside the constellation viewport. Used to distinguish click vs drag.
#[derive(Debug, Clone)]
pub struct PendingConstellationClick {
    pub node_id: Option<String>,
    pub start_pos: Vec2,
}

pub struct App {
    pub windows: Vec<SubWindow>,
    pub header: HeaderWindow,
    pub sidebar: SidebarWindow,
    pub chat_window: Option<ChatWindow>,
    pub library_window: Option<LibraryWindow>,
    pub ingest_window: Option<IngestWindow>,
    pub settings_window: Option<SettingsWindow>,
    pub notepad_window: Option<NotepadWindow>,
    pub insight_modal: InsightModal,
    pub shard_modal: ShardModal,
    pub pdf_modal: PdfModal,
    pub chat_info_dialog: ChatInfoDialog,
    pub global_search_modal: GlobalSearchModal,
    pub toast_manager: ToastManager,
    pub notepad_modal: NotepadModal,
    pub chat_state: ChatState,
    pub graph_state: GraphState,
    pub api_client: ApiClient,
    pub mouse_pos: Vec2,
    dragging_id: Option<usize>,
    drag_offset: Vec2,
    pub viewport_size: Vec2,
    pub ui_state: UIState,
    pub settings_state: SettingsState,
    pub insights_state: InsightsState,
    pub focused_input: Option<usize>,  // Index to identify which input is focused
    pub is_sending_message: bool,
    pub modifiers: ModifiersState,
    pub api_response_receiver: mpsc::Receiver<Result<crate::api::models::ChatResponse, String>>,
    api_response_sender: mpsc::Sender<Result<crate::api::models::ChatResponse, String>>,
    pub collections_receiver: mpsc::Receiver<Result<Vec<Collection>, String>>,
    collections_sender: mpsc::Sender<Result<Vec<Collection>, String>>,
    pub context_pool_response_receiver: mpsc::Receiver<Result<(), String>>,
    context_pool_response_sender: mpsc::Sender<Result<(), String>>,
    pub papers_receiver: mpsc::Receiver<Result<Vec<crate::api::models::ApiPaper>, String>>,
    papers_sender: mpsc::Sender<Result<Vec<crate::api::models::ApiPaper>, String>>,
    pub ingest_response_receiver: mpsc::Receiver<Result<String, String>>,
    ingest_response_sender: mpsc::Sender<Result<String, String>>,
    pub task_status_receiver: mpsc::Receiver<Result<serde_json::Value, String>>,
    task_status_sender: mpsc::Sender<Result<serde_json::Value, String>>,
    pub insights_receiver: mpsc::Receiver<Result<Vec<crate::api::models::Insight>, String>>,
    insights_sender: mpsc::Sender<Result<Vec<crate::api::models::Insight>, String>>,
    pub pdf_bytes_receiver: mpsc::Receiver<Result<Vec<u8>, String>>,
    pdf_bytes_sender: mpsc::Sender<Result<Vec<u8>, String>>,
    pub note_content_receiver: mpsc::Receiver<Result<String, String>>,
    note_content_sender: mpsc::Sender<Result<String, String>>,
    pub graph_loaded_receiver: mpsc::Receiver<Result<(String, crate::api::models::GetGraphResponse), String>>,
    graph_loaded_sender: mpsc::Sender<Result<(String, crate::api::models::GetGraphResponse), String>>,
    /// On startup, if current conversation has graph_id, load it once.
    pending_initial_graph_load: Option<String>,
    graph_send_receiver: mpsc::Receiver<Result<(String, crate::api::models::GraphSendResponse), String>>,
    graph_send_sender: mpsc::Sender<Result<(String, crate::api::models::GraphSendResponse), String>>,
    collections_loaded: bool,
    papers_loaded: bool,
    insights_loaded: bool,
    /// One-shot: ensure we have a conversation and graph on startup when none exist.
    conversation_ensured: bool,
    current_ingest_task_id: Option<String>,
    window_proxy: Option<EventLoopProxy<WindowControlEvent>>,
    pub sidebar_edge_glow_position: Option<f32>,  // Y-position of glow effect
    pub sidebar_edge_glow_intensity: f32,  // 0.0 to 1.0
    pub sidebar_edge_glow_target_intensity: f32,  // Target intensity for smooth transitions
    pub last_glow_debug_state: (bool, f32),  // (was_hovering, last_intensity) for throttling debug
    pub is_dragging_window: bool,
    pub window_drag_start: Vec2,
    pub cursor_blink_timer: f32,
    pub cursor_visible: bool,
    pub cursor_target_position: usize,  // For smooth interpolation
    pub cursor_position_animation: crate::utils::animation::SpringAnimation,  // Smooth cursor position animation
    // Click tracking for double/triple click detection
    pub last_click_time: std::time::Instant,
    pub last_click_position: Vec2,
    pub click_count: u32,  // 1 = single, 2 = double, 3 = triple
    // Clipboard state
    pub clipboard_text: String,
    pub undo_history: Vec<String>,  // History of text states for undo
    pub redo_history: Vec<String>,  // History for redo
    // Extended mouse event state
    pub drag_state: crate::ui::events::DragState,
    pub hover_state: crate::ui::events::HoverState,
    pub is_dragging: bool,  // Track if currently dragging (not just window)
    pub drag_start_pos: Vec2,  // Position where drag started
    pub drag_button: Option<MouseButton>,  // Button used for drag
    // Window event state
    pub window_focused: bool,  // Track if window has focus
    pub window_position: Vec2,  // Window position on screen
    pub scale_factor: f64,  // Current DPI scale factor
    // File drag and drop state
    pub file_drag_active: bool,  // Files are being dragged over window
    pub file_drag_paths: Vec<std::path::PathBuf>,  // Paths of files being dragged
    pub file_drag_position: Vec2,  // Position of file drag
    // Keyboard shortcuts
    pub shortcut_registry: crate::ui::shortcuts::ShortcutRegistry,
    pub pressed_keys: std::collections::HashSet<KeyCode>,  // Track currently pressed keys
    // Touch event state
    pub active_touches: std::collections::HashMap<u64, Vec2>,  // Track active touches (touch_id -> position)
    // Focus management
    pub focus_state: crate::ui::events::FocusState,
    // Accessibility state
    pub accessibility_focused_component: Option<String>,  // Component with screen reader focus
    /// Deferred notepad click for glyph-based cursor placement (processed in render when renderer is available).
    pub pending_notepad_click: Option<Vec2>,
    /// Pending constellation click (evaluated on mouse release for click vs drag).
    pub pending_constellation_click: Option<PendingConstellationClick>,
    // Component hierarchy root
    pub root: crate::ui::components::Root,
    /// Throttle for saving graph layout (persist node positions every 2s when graph active).
    layout_save_timer: f32,
    /// Frames to keep running constellation physics after graph load/change; when 0, physics runs only when total velocity > threshold.
    physics_settle_frames: u32,
    /// Accumulator for fixed-rate physics stepping (e.g. 30 Hz).
    physics_dt_accumulator: f32,
    /// Seconds since last constellation interaction; when high and velocity low, physics is turned off.
    physics_idle_timer: f32,
    /// Bumped when viewport, sidebar, or active tab changes; used to skip root.update_layout when unchanged.
    pub layout_generation: u64,
    /// When true, log per-frame text command count and visible constellation nodes when threshold exceeded.
    pub debug_text_stats: bool,
}

impl App {
    pub fn new(viewport_size: (u32, u32)) -> Self {
        let viewport = Vec2::new(viewport_size.0 as f32, viewport_size.1 as f32);

        // Load UI state and settings state first
        let ui_state = UIState::new();
        let settings_state = SettingsPersistence::load_settings().unwrap_or_else(|_| SettingsState::new());
        let insights_state = InsightsState::new();
        
        let header_height = 60.0;
        let header = HeaderWindow::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(viewport.x, header_height),
        );

        let sidebar_height = viewport.y - header_height;
        let mut sidebar = SidebarWindow::new(
            Vec2::new(0.0, header_height),
            sidebar_height,
        );
        sidebar.is_open = ui_state.sidebar_open;
        if !ui_state.sidebar_open {
            sidebar.toggle();
        }

        let mut chat_window = if viewport.x > 0.0 && viewport.y > 0.0 {
            let chat_height = viewport.y - header_height;
            let open_content_width = viewport.x - SidebarWindow::OPEN_WIDTH;
            let (chat_x, chat_width) = if sidebar.is_open {
                (sidebar.current_width, viewport.x - sidebar.current_width)
            } else {
                ((viewport.x - open_content_width) / 2.0, open_content_width)
            };
            Some(ChatWindow::new(
                Vec2::new(chat_x, header_height),
                Vec2::new(chat_width, chat_height),
            ))
        } else {
            None
        };

        let library_window = if viewport.x > 0.0 && viewport.y > 0.0 {
            let _library_y = header_height + sidebar.current_width;
            let library_width = viewport.x - sidebar.current_width;
            let library_height = viewport.y - header_height;
            Some(LibraryWindow::new(
                Vec2::new(sidebar.current_width, header_height),
                Vec2::new(library_width, library_height),
            ))
        } else {
            None
        };

        let ingest_window = if viewport.x > 0.0 && viewport.y > 0.0 {
            let _ingest_y = header_height + sidebar.current_width;
            let ingest_width = viewport.x - sidebar.current_width;
            let ingest_height = viewport.y - header_height;
            Some(IngestWindow::new(
                Vec2::new(sidebar.current_width, header_height),
                Vec2::new(ingest_width, ingest_height),
            ))
        } else {
            None
        };

        let settings_window = if viewport.x > 0.0 && viewport.y > 0.0 {
            let _settings_y = header_height + sidebar.current_width;
            let settings_width = viewport.x - sidebar.current_width;
            let settings_height = viewport.y - header_height;
            Some(SettingsWindow::new(
                Vec2::new(sidebar.current_width, header_height),
                Vec2::new(settings_width, settings_height),
            ))
        } else {
            None
        };

        let notepad_window = if viewport.x > 0.0 && viewport.y > 0.0 {
            let _notepad_y = header_height + sidebar.current_width;
            let notepad_width = viewport.x - sidebar.current_width;
            let notepad_height = viewport.y - header_height;
            Some(NotepadWindow::new(
                Vec2::new(sidebar.current_width, header_height),
                Vec2::new(notepad_width, notepad_height),
            ))
        } else {
            None
        };

        // Load chat state and sync messages to chat window if there's a current conversation
        let chat_state = ConversationPersistence::load_chat_state()
            .unwrap_or_else(|_| ChatState::new());
        
        // Sync messages from chat_state to chat window if there's a current conversation.
        // When conv has graph_id, leave messages empty; check_graph_loaded will rebuild from graph.
        if let Some(ref mut chat) = chat_window {
            if let Some(ref conv_id) = chat_state.current_conversation_id {
                if let Some(conv) = chat_state.conversations.iter().find(|c| c.id == *conv_id) {
                    if conv.graph_id.is_none() {
                        chat.messages = conv.shards.iter().map(|s| crate::ui::chat_window::ChatMessage::from_shard(s)).collect();
                    }
                }
            }
        }

        let (api_response_sender, api_response_receiver) = mpsc::channel();
        let (collections_sender, collections_receiver) = mpsc::channel();
        let (context_pool_response_sender, context_pool_response_receiver) = mpsc::channel();
        let (papers_sender, papers_receiver) = mpsc::channel();
        let (ingest_response_sender, ingest_response_receiver) = mpsc::channel();
        let (task_status_sender, task_status_receiver) = mpsc::channel();
        let (insights_sender, insights_receiver) = mpsc::channel();
        let (pdf_bytes_sender, pdf_bytes_receiver) = mpsc::channel();
        let (note_content_sender, note_content_receiver) = mpsc::channel();
        let (graph_loaded_sender, graph_loaded_receiver) = mpsc::channel();
        let (graph_send_sender, graph_send_receiver) = mpsc::channel();
        let pending_initial_graph_load = chat_state.current_conversation_id.as_ref().and_then(|cid| {
            chat_state.conversations.iter().find(|c| c.id == *cid).and_then(|c| c.graph_id.clone())
        });

        Self {
            windows: Vec::new(),  // No demo windows
            header,
            sidebar,
            chat_window,
            library_window,
            ingest_window,
            settings_window,
            notepad_window,
            insight_modal: InsightModal::new(),
            shard_modal: ShardModal::new(),
            pdf_modal: PdfModal::new(),
            chat_info_dialog: ChatInfoDialog::new(),
            global_search_modal: GlobalSearchModal::new(),
            toast_manager: ToastManager::new(),
            notepad_modal: NotepadModal::new(),
            chat_state,
            graph_state: GraphState::new(),
            insights_state,
            api_client: ApiClient::new(settings_state.api_base_url.clone()),
            mouse_pos: Vec2::ZERO,
            dragging_id: None,
            drag_offset: Vec2::ZERO,
            viewport_size: viewport,
            ui_state,
            settings_state,
            focused_input: None,
            is_sending_message: false,
            modifiers: ModifiersState::empty(),
            api_response_receiver,
            api_response_sender,
            collections_receiver,
            collections_sender,
            context_pool_response_receiver,
            context_pool_response_sender,
            papers_receiver,
            papers_sender,
            ingest_response_receiver,
            ingest_response_sender,
            task_status_receiver,
            task_status_sender,
            insights_receiver,
            insights_sender,
            pdf_bytes_receiver,
            pdf_bytes_sender,
            note_content_receiver,
            note_content_sender,
            graph_loaded_receiver,
            graph_loaded_sender,
            pending_initial_graph_load,
            graph_send_receiver,
            graph_send_sender,
            collections_loaded: false,
            papers_loaded: false,
            insights_loaded: false,
            conversation_ensured: false,
            current_ingest_task_id: None,
            window_proxy: None,
            sidebar_edge_glow_position: None,
            sidebar_edge_glow_intensity: 0.0,
            sidebar_edge_glow_target_intensity: 0.0,
            last_glow_debug_state: (false, 0.0),
            is_dragging_window: false,
            window_drag_start: Vec2::ZERO,
            cursor_blink_timer: 0.0,
            cursor_visible: true,
            cursor_target_position: 0,
            cursor_position_animation: crate::utils::animation::SpringAnimation::with_preset(
                0.0,
                crate::utils::animation::AnimationPreset::TightBounce,
            ),
            last_click_time: std::time::Instant::now(),
            last_click_position: Vec2::ZERO,
            click_count: 0,
            clipboard_text: String::new(),
            undo_history: Vec::new(),
            redo_history: Vec::new(),
            drag_state: crate::ui::events::DragState::None,
            hover_state: crate::ui::events::HoverState::default(),
            is_dragging: false,
            drag_start_pos: Vec2::ZERO,
            drag_button: None,
            window_focused: true,  // Assume focused on startup
            window_position: Vec2::ZERO,
            scale_factor: 1.0,
            file_drag_active: false,
            file_drag_paths: Vec::new(),
            file_drag_position: Vec2::ZERO,
            shortcut_registry: {
                let mut registry = crate::ui::shortcuts::ShortcutRegistry::new();
                use winit::keyboard::{KeyCode, ModifiersState};
                
                // Register keyboard shortcuts
                // Note: On macOS, SUPER is Command; on Windows/Linux, CONTROL is Ctrl
                // We register both variants for cross-platform support
                
                // Cmd/Ctrl+N: New conversation
                registry.register(ModifiersState::SUPER, KeyCode::KeyN, "New conversation (Mac)".to_string());
                registry.register(ModifiersState::CONTROL, KeyCode::KeyN, "New conversation (Win/Linux)".to_string());
                
                // Cmd/Ctrl+K: Global search
                registry.register(ModifiersState::SUPER, KeyCode::KeyK, "Global search (Mac)".to_string());
                registry.register(ModifiersState::CONTROL, KeyCode::KeyK, "Global search (Win/Linux)".to_string());
                
                // Cmd/Ctrl+,: Settings
                registry.register(ModifiersState::SUPER, KeyCode::Comma, "Open settings (Mac)".to_string());
                registry.register(ModifiersState::CONTROL, KeyCode::Comma, "Open settings (Win/Linux)".to_string());
                
                // Cmd/Ctrl+B: Toggle sidebar
                registry.register(ModifiersState::SUPER, KeyCode::KeyB, "Toggle sidebar (Mac)".to_string());
                registry.register(ModifiersState::CONTROL, KeyCode::KeyB, "Toggle sidebar (Win/Linux)".to_string());
                
                // Cmd/Ctrl+Enter: Send message
                registry.register(ModifiersState::SUPER, KeyCode::Enter, "Send message (Mac)".to_string());
                registry.register(ModifiersState::CONTROL, KeyCode::Enter, "Send message (Win/Linux)".to_string());
                
                registry
            },
            pressed_keys: std::collections::HashSet::new(),
            active_touches: std::collections::HashMap::new(),
            focus_state: crate::ui::events::FocusState::default(),
            accessibility_focused_component: None,
            pending_notepad_click: None,
            pending_constellation_click: None,
            layout_save_timer: 0.0,
            physics_settle_frames: 0,
            physics_dt_accumulator: 0.0,
            physics_idle_timer: 0.0,
            layout_generation: 0,
            debug_text_stats: false,
            // Build component tree
            root: {
                use crate::ui::components::{Root, HeaderComponent, SidebarComponent, SidebarContentComponent, ChatComponent, LibraryComponent, DataComponent, SettingsComponent, NotepadComponent};
                let mut root = Root::new(viewport);
                
                // Add window components in z-order (lower z-order renders first/behind)
                // Order: content (chat, library, etc.) first, then sidebar above main, then header above all
                root.add_child(Box::new(ChatComponent::new()));
                root.add_child(Box::new(LibraryComponent::new()));
                root.add_child(Box::new(DataComponent::new()));
                root.add_child(Box::new(SettingsComponent::new()));
                root.add_child(Box::new(NotepadComponent::new()));
                root.add_child(Box::new(SidebarComponent::new()));
                root.add_child(Box::new(SidebarContentComponent::new()));
                root.add_child(Box::new(HeaderComponent::new())); // Header last (highest z-order)
                
                root
            },
        }
    }
    
    pub fn set_window_proxy(&mut self, proxy: EventLoopProxy<WindowControlEvent>) {
        self.window_proxy = Some(proxy);
    }

    /// Call when viewport, sidebar open/width, or active tab has changed so root layout is re-run next frame.
    fn bump_layout_generation(&mut self) {
        self.layout_generation = self.layout_generation.wrapping_add(1);
    }

    pub fn resize(&mut self, size: (u32, u32)) {
        self.viewport_size = Vec2::new(size.0 as f32, size.1 as f32);
        self.layout_generation = self.layout_generation.wrapping_add(1);
        self.header.update_layout(self.viewport_size);
        let header_height = self.header.size.y;
        self.sidebar.height = self.viewport_size.y - header_height;
        
        let content_height = self.viewport_size.y - header_height;
        let content_y = header_height;
        let open_content_width = self.viewport_size.x - SidebarWindow::OPEN_WIDTH;
        let (chat_x, chat_width) = if self.sidebar.is_open {
            (self.sidebar.current_width, self.viewport_size.x - self.sidebar.current_width)
        } else {
            ((self.viewport_size.x - open_content_width) / 2.0, open_content_width)
        };
        if let Some(ref mut chat) = self.chat_window {
            chat.position = Vec2::new(chat_x, content_y);
            chat.size = Vec2::new(chat_width, content_height);
            chat.update_layout();
        }
        let sidebar_width = self.sidebar.current_width;
        let content_width = self.viewport_size.x - sidebar_width;
        
        if let Some(ref mut library) = self.library_window {
            library.position = Vec2::new(sidebar_width, content_y);
            library.size = Vec2::new(content_width, content_height);
            library.update_layout();
        }
        
        if let Some(ref mut ingest) = self.ingest_window {
            ingest.position = Vec2::new(sidebar_width, content_y);
            ingest.size = Vec2::new(content_width, content_height);
            ingest.update_layout();
        }
        
        if let Some(ref mut settings) = self.settings_window {
            settings.position = Vec2::new(sidebar_width, content_y);
            settings.size = Vec2::new(content_width, content_height);
            settings.update_layout(&self.settings_state.provider);
        }
        
        if let Some(ref mut notepad) = self.notepad_window {
            notepad.position = Vec2::new(sidebar_width, content_y);
            notepad.size = Vec2::new(content_width, content_height);
            notepad.update_layout();
        }
    }

    pub fn on_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        match state {
            ElementState::Pressed => {
                // Handle right-click for context menus
                if button == MouseButton::Right {
                    self.on_mouse_right_click();
                    return;
                }
                
                // Handle middle-click
                if button == MouseButton::Middle {
                    self.on_mouse_middle_click();
                    return;
                }
                
                // Only process left-click for the main interaction logic
                if button != MouseButton::Left {
                    return;
                }
                
                // Start drag operation tracking
                self.is_dragging = true;
                self.drag_start_pos = self.mouse_pos;
                self.drag_button = Some(button);
                self.drag_state = crate::ui::events::DragState::Starting {
                    button: button.into(),
                    start_pos: self.mouse_pos,
                };
                // Track click count for double/triple click detection
                let now = std::time::Instant::now();
                let click_interval = now.duration_since(self.last_click_time);
                let click_distance = (self.mouse_pos - self.last_click_position).length();
                
                // Double/triple click threshold: 500ms and 5px movement
                if click_interval.as_millis() < 500 && click_distance < 5.0 {
                    self.click_count = (self.click_count + 1).min(3);
                } else {
                    self.click_count = 1;
                }
                
                self.last_click_time = now;
                self.last_click_position = self.mouse_pos;
                // Z-index ordered hit testing (highest to lowest)
                // Order matches render z-order from renderer.rs:
                // 100: Header (always on top)
                // 90: Toasts  
                // 50: Modals, dialogs
                // 20: Chat window
                // 11: Sidebar content
                // 10: Sidebar background
                // 5: Sidebar toggle/glow
                
                // ===== Z-INDEX 100: Header (tabs, window controls) =====
                if let Some(click) = self.header.on_mouse_click(self.mouse_pos) {
                    match click {
                        crate::ui::header::HeaderClick::Tab(index) => {
                            let new_tab = self.header.tab_bar.tabs[index];
                            let old_tab = self.ui_state.active_tab;
                            
                            // Clear focus when switching tabs
                            if old_tab != new_tab {
                                // Blur all inputs when switching away from a tab
                                if let Some(ref mut library) = self.library_window {
                                    library.search_input.on_blur();
                                    library.new_collection_input.on_blur();
                                }
                                if let Some(ref mut chat) = self.chat_window {
                                    chat.input_field.on_blur();
                                }
                                if let Some(ref mut ingest) = self.ingest_window {
                                    ingest.pdf_dir_input.on_blur();
                                }
                                if let Some(ref mut settings) = self.settings_window {
                                    settings.model_id_input.on_blur();
                                    settings.hf_token_input.on_blur();
                                    settings.openai_model_input.on_blur();
                                }
                                
                                // Clear focused_input state
                                self.focused_input = None;
                            }
                            
                            let old_tab = self.ui_state.active_tab;
                            self.ui_state.set_active_tab(new_tab);
                            self.bump_layout_generation();
                            
                            // Auto-focus appropriate input when switching tabs
                            if old_tab != new_tab {
                                match new_tab {
                                    Tab::Notepad => {
                                        if let Some(ref mut notepad) = self.notepad_window {
                                            notepad.editor.focus();
                                            // Focus the first block if no block is focused
                                            if notepad.editor.cursor.is_none() && !notepad.editor.document.blocks.is_empty() {
                                                let first_block_id = notepad.editor.document.blocks[0].id.clone();
                                                notepad.editor.focus_block(&first_block_id);
                                            }
                                            self.focused_input = Some(5);
                                        }
                                    }
                                    Tab::Chat => {
                                        if let Some(ref mut chat) = self.chat_window {
                                            chat.input_field.on_focus();
                                            self.focused_input = Some(0);
                                        }
                                    }
                                    Tab::Library => {
                                        if let Some(ref mut library) = self.library_window {
                                            library.search_input.on_focus();
                                            self.focused_input = Some(1);
                                        }
                                    }
                                    Tab::Data => {
                                        if let Some(ref mut ingest) = self.ingest_window {
                                            ingest.pdf_dir_input.on_focus();
                                            self.focused_input = Some(2);
                                        }
                                    }
                                    Tab::Settings => {
                                        // Don't auto-focus settings inputs
                                        self.focused_input = None;
                                    }
                                }
                            }
                        }
                        crate::ui::header::HeaderClick::Minimize => {
                            if let Some(ref proxy) = self.window_proxy {
                                let _ = proxy.send_event(WindowControlEvent::Minimize);
                            }
                        }
                        crate::ui::header::HeaderClick::Maximize => {
                            if let Some(ref proxy) = self.window_proxy {
                                let _ = proxy.send_event(WindowControlEvent::ToggleMaximize);
                            }
                        }
                        crate::ui::header::HeaderClick::Close => {
                            if let Some(ref proxy) = self.window_proxy {
                                let _ = proxy.send_event(WindowControlEvent::Close);
                            }
                        }
                    }
                    return;
                }
                
                // Check if clicking in header area for window dragging (but not on buttons)
                let header_height = self.header.size.y;
                if self.mouse_pos.y < header_height {
                    let control_buttons_start = self.viewport_size.x - 120.0;
                    if self.mouse_pos.x < control_buttons_start {
                        self.is_dragging_window = true;
                        self.window_drag_start = self.mouse_pos;
                        return;
                    }
                }
                
                // ===== Z-INDEX 90: Toasts =====
                // (Toasts typically don't need click handling, they auto-dismiss)
                
                // ===== Z-INDEX 50: Modals, dialogs =====
                // Test modals before other content so they capture clicks
                if self.pdf_modal.is_open {
                    if !self.pdf_modal.contains(self.mouse_pos) {
                        self.pdf_modal.close();
                        return;
                    }
                    if self.pdf_modal.close_button.contains(self.mouse_pos) {
                        self.pdf_modal.close();
                        return;
                    }
                    if self.pdf_modal.prev_page_button.contains(self.mouse_pos) && self.pdf_modal.current_page > 1 {
                        self.pdf_modal.prev_page();
                        return;
                    }
                    if self.pdf_modal.next_page_button.contains(self.mouse_pos) {
                        if let Some(total) = self.pdf_modal.total_pages {
                            if self.pdf_modal.current_page < total {
                                self.pdf_modal.next_page();
                                return;
                            }
                        } else {
                            self.pdf_modal.next_page();
                            return;
                        }
                    }
                }
                
                if self.chat_info_dialog.is_open {
                    if !self.chat_info_dialog.contains(self.mouse_pos) {
                        self.chat_info_dialog.close();
                        return;
                    }
                    if self.chat_info_dialog.close_button.contains(self.mouse_pos) {
                        self.chat_info_dialog.close();
                        return;
                    }
                    // Handle mode toggle button
                    if self.chat_info_dialog.mode_toggle_button.contains(self.mouse_pos) {
                        self.chat_info_dialog.citation_mode = match self.chat_info_dialog.citation_mode {
                            crate::ui::CitationMode::All => crate::ui::CitationMode::Unique,
                            crate::ui::CitationMode::Unique => crate::ui::CitationMode::All,
                        };
                        return;
                    }
                    
                    // Handle citation magnify icon clicks (open PDF)
                    if let Some(conv_id) = &self.chat_info_dialog.conversation_id {
                        if let Some(conv) = self.chat_state.conversations.iter().find(|c| c.id == *conv_id) {
                            // Get citations from shards (convert to messages first)
                            let messages: Vec<crate::ui::chat_window::ChatMessage> = conv.shards.iter()
                                .map(|s| crate::ui::chat_window::ChatMessage::from_shard(s))
                                .collect();
                            let all_citations: Vec<&crate::ui::chat_window::Citation> = messages
                                .iter()
                                .filter_map(|m| {
                                    if matches!(m.role, crate::ui::chat_window::MessageRole::Assistant) {
                                        Some(m.citations.iter())
                                    } else {
                                        None
                                    }
                                })
                                .flatten()
                                .collect();
                            
                            let citations: Vec<&crate::ui::chat_window::Citation> = match self.chat_info_dialog.citation_mode {
                                crate::ui::CitationMode::All => all_citations.clone(),
                                crate::ui::CitationMode::Unique => {
                                    let mut seen = std::collections::HashSet::new();
                                    all_citations.into_iter().filter(|cit| {
                                        let key = format!("{}:{}", cit.source, cit.title.as_ref().unwrap_or(&String::new()));
                                        seen.insert(key)
                                    }).collect()
                                }
                            };
                            
                            let citations_rect = crate::ui::core::Rect::new(
                                self.chat_info_dialog.citations_list.position.x,
                                self.chat_info_dialog.citations_list.position.y,
                                self.chat_info_dialog.citations_list.size.x,
                                self.chat_info_dialog.citations_list.size.y,
                            );
                            
                            if citations_rect.contains_point(self.mouse_pos) {
                                let item_height = 25.0;
                                let scroll_offset = self.chat_info_dialog.citations_list.scroll_offset;
                                let rel_y = self.mouse_pos.y - citations_rect.y + scroll_offset;
                                let item_index = (rel_y / item_height) as usize;
                                
                                if item_index < citations.len() {
                                    let icon_x = citations_rect.x + citations_rect.width - 30.0;
                                    let icon_y = citations_rect.y + (item_index as f32 * item_height) - scroll_offset + item_height / 2.0 - 7.0;
                                    let icon_rect = crate::ui::core::Rect::new(icon_x, icon_y, 14.0, 14.0);
                                    
                                    if icon_rect.contains_point(self.mouse_pos) {
                                        let citation = citations[item_index];
                                        // Open PDF modal with the citation source
                                        if citation.source.to_lowercase().ends_with(".pdf") {
                                            let page = citation.page.unwrap_or(1);
                                            let filename = format!("{}#page={}", citation.source, page);
                                            self.pdf_modal.open(filename, Some(page));
                                            self.pdf_modal.loading = true;
                                            
                                            // Load PDF from backend
                                            let base_url = self.api_client.base_url.clone();
                                            let client = self.api_client.client.clone();
                                            let source = citation.source.clone();
                                            tokio::spawn(async move {
                                                let url = format!("{}/papers/{}", base_url, source);
                                                match client.get(&url).send().await {
                                                    Ok(resp) => {
                                                        if resp.status().is_success() {
                                                            if let Ok(_bytes) = resp.bytes().await {
                                                                // PDF bytes received
                                                            }
                                                        }
                                                    }
                                                    Err(e) => eprintln!("Failed to fetch PDF: {}", e),
                                                }
                                            });
                                        }
                                        return;
                                    }
                                }
                            }
                        }
                    }
                    
                    if self.chat_info_dialog.delete_button.contains(self.mouse_pos) {
                        if let Some(ref conv_id) = self.chat_info_dialog.conversation_id {
                            let conv_id_clone = conv_id.clone();
                            self.chat_state.conversations.retain(|c| c.id != conv_id_clone);
                            self.save_chat_state();
                            use crate::persistence::ConversationPersistence;
                            if let Err(e) = ConversationPersistence::delete_conversation(&conv_id_clone) {
                                eprintln!("Failed to delete conversation file: {}", e);
                            }
                            self.chat_info_dialog.close();
                            self.show_success_toast("Conversation deleted".to_string());
                        }
                        return;
                    }
                    // Click on title to start editing
                    let title_rect = crate::ui::core::Rect::from_pos_size(
                        self.chat_info_dialog.title_input.position,
                        self.chat_info_dialog.title_input.size,
                    );
                    if title_rect.contains_point(self.mouse_pos) && !self.chat_info_dialog.is_editing_title {
                        self.chat_info_dialog.is_editing_title = true;
                        self.chat_info_dialog.title_input.text = self.chat_info_dialog.draft_title.clone();
                        self.chat_info_dialog.title_input.on_focus();
                        self.chat_info_dialog.title_input.cursor_position = self.chat_info_dialog.title_input.text.chars().count();
                        self.focused_input = Some(8);
                        return;
                    }
                }
                
                if self.shard_modal.is_open {
                    if !self.shard_modal.contains(self.mouse_pos) {
                        self.shard_modal.close();
                        return;
                    }
                    if self.shard_modal.close_button.contains(self.mouse_pos) {
                        self.shard_modal.close();
                        return;
                    }
                    if self.shard_modal.remove_from_graph_button.contains(self.mouse_pos) {
                        let graph_id_opt = self.graph_state.graph_id.clone();
                        let shard_id_opt = self.shard_modal.shard_id.clone();
                        if let (Some(graph_id), Some(shard_id)) = (graph_id_opt, shard_id_opt) {
                            let base_url = self.api_client.base_url.clone();
                            let shard_id_clone = shard_id.clone();
                            tokio::spawn(async move {
                                let api_client = crate::api::ApiClient::new(Some(base_url));
                                match api_client.remove_shard_from_graph(&graph_id, &shard_id_clone).await {
                                    Ok(()) => {}
                                    Err(e) => eprintln!("Failed to remove shard from graph: {}", e),
                                }
                            });
                            self.graph_state.nodes.remove(&shard_id);
                            self.graph_state.bump_content_version();
                            if let Some(ref mut chat) = self.chat_window {
                                chat.messages = self.graph_state.node_ids_bfs_order()
                                    .into_iter()
                                    .filter_map(|id| self.graph_state.get_node(&id))
                                    .flat_map(|node| {
                                        let id = node.shard.id.clone();
                                        let contexts = node.shard.contexts.clone();
                                        let mut msgs = Vec::new();
                                        if let Some(ref u) = node.shard.user_content {
                                            if !u.is_empty() {
                                                msgs.push(crate::ui::chat_window::ChatMessage {
                                                    shard_id: Some(id.clone()),
                                                    role: crate::ui::chat_window::MessageRole::User,
                                                    content: u.clone(),
                                                    contexts: contexts.clone(),
                                                    citations: Vec::new(),
                                                    notes: Vec::new(),
                                                });
                                            }
                                        }
                                        if let Some(ref a) = node.shard.assistant_content {
                                            if !a.is_empty() {
                                                msgs.push(crate::ui::chat_window::ChatMessage {
                                                    shard_id: Some(id.clone()),
                                                    role: crate::ui::chat_window::MessageRole::Assistant,
                                                    content: a.clone(),
                                                    contexts: contexts.clone(),
                                                    citations: Vec::new(),
                                                    notes: node.shard.notes.clone(),
                                                });
                                            }
                                        }
                                        msgs
                                    })
                                    .collect();
                            }
                            self.chat_state.set_current_messages(self.chat_window.as_ref().unwrap().messages.clone());
                            self.save_chat_state();
                            self.shard_modal.close();
                            self.show_success_toast("Removed from graph".to_string());
                        }
                        return;
                    }
                    if self.shard_modal.save_button.contains(self.mouse_pos) {
                        if let Some(ref shard_id) = self.shard_modal.shard_id {
                            let user_content = self.shard_modal.user_input.text.clone();
                            let assistant_content = self.shard_modal.assistant_input.text.clone();
                            if let Some(node) = self.graph_state.nodes.get_mut(shard_id) {
                                node.shard.user_content = Some(user_content.clone());
                                node.shard.assistant_content = Some(assistant_content.clone());
                                self.graph_state.bump_content_version();
                            }
                            if let Some(ref mut chat) = self.chat_window {
                                if let Some(msg_idx) = chat.messages.iter().position(|m| m.shard_id.as_deref() == Some(shard_id.as_str())) {
                                    chat.messages[msg_idx].content = assistant_content.clone();
                                    if msg_idx > 0 {
                                        chat.messages[msg_idx - 1].content = user_content;
                                    }
                                }
                            }
                            self.chat_state.set_current_messages(self.chat_window.as_ref().unwrap().messages.clone());
                            self.save_chat_state();
                        }
                        self.shard_modal.close();
                        return;
                    }
                    if self.shard_modal.user_input.contains(self.mouse_pos) {
                        self.shard_modal.user_input.on_focus();
                        self.shard_modal.user_input.on_mouse_down(self.mouse_pos, |text, size| {
                            let char_count = text.chars().count();
                            glam::Vec2::new(char_count as f32 * size * 0.66, size)
                        }, self.click_count);
                        self.focused_input = Some(20);
                        return;
                    }
                    if self.shard_modal.assistant_input.contains(self.mouse_pos) {
                        self.shard_modal.assistant_input.on_focus();
                        self.shard_modal.assistant_input.on_mouse_down(self.mouse_pos, |text, size| {
                            let char_count = text.chars().count();
                            glam::Vec2::new(char_count as f32 * size * 0.66, size)
                        }, self.click_count);
                        self.focused_input = Some(21);
                        return;
                    }
                }
                
                if self.insight_modal.is_open {
                    if !self.insight_modal.contains(self.mouse_pos) {
                        self.insight_modal.close();
                        self.insights_state.set_modal_insight(None);
                        return;
                    }
                    if self.insight_modal.close_button.contains(self.mouse_pos) {
                        self.insight_modal.close();
                        self.insights_state.set_modal_insight(None);
                        return;
                    }
                    if self.insight_modal.save_button.contains(self.mouse_pos) {
                        if let Some(ref insight) = self.insight_modal.insight {
                            let insight_id = insight.id.clone();
                            let new_text = self.insight_modal.draft_text.clone();
                            let new_title = if self.insight_modal.draft_title.trim().is_empty() {
                                None
                            } else {
                                Some(self.insight_modal.draft_title.clone())
                            };
                            if !new_text.trim().is_empty() {
                                self.insights_state.update_insight_text(&insight_id, new_text.clone());
                                if let Some(ref title) = new_title {
                                    self.insights_state.update_insight_title(&insight_id, title.clone());
                                }
                                let base_url = self.api_client.base_url.clone();
                                let insight_id_clone = insight_id.clone();
                                tokio::spawn(async move {
                                    let api_client = crate::api::ApiClient::new(Some(base_url));
                                    if let Err(e) = api_client.update_shard(&insight_id_clone, Some(new_text), None, new_title, None).await {
                                        eprintln!("Failed to update shard: {}", e);
                                    }
                                });
                            }
                            self.insight_modal.is_editing_text = false;
                            self.insight_modal.is_editing_title = false;
                        }
                        return;
                    }
                    if self.insight_modal.delete_button.contains(self.mouse_pos) {
                        if let Some(ref insight) = self.insight_modal.insight {
                            let insight_id = insight.id.clone();
                            self.insights_state.remove_insight(&insight_id);
                            let base_url = self.api_client.base_url.clone();
                            let insight_id_clone = insight_id.clone();
                            tokio::spawn(async move {
                                let api_client = crate::api::ApiClient::new(Some(base_url));
                                if let Err(e) = api_client.delete_shard(&insight_id_clone).await {
                                    eprintln!("Failed to delete shard: {}", e);
                                }
                            });
                            self.insight_modal.close();
                            self.insights_state.set_modal_insight(None);
                        }
                        return;
                    }
                    if !self.insight_modal.is_editing_title {
                        let title_rect = (
                            self.insight_modal.position.x + 20.0,
                            self.insight_modal.position.y + 60.0,
                            self.insight_modal.position.x + self.insight_modal.size.x - 40.0,
                            self.insight_modal.position.y + 90.0,
                        );
                        if self.mouse_pos.x >= title_rect.0 && self.mouse_pos.x <= title_rect.2 &&
                           self.mouse_pos.y >= title_rect.1 && self.mouse_pos.y <= title_rect.3 {
                            self.insight_modal.is_editing_title = true;
                            self.insight_modal.title_input.on_focus();
                            self.focused_input = Some(6);
                            return;
                        }
                    }
                    if !self.insight_modal.is_editing_text {
                        let text_rect = (
                            self.insight_modal.position.x + 20.0,
                            self.insight_modal.position.y + 100.0,
                            self.insight_modal.position.x + self.insight_modal.size.x - 40.0,
                            self.insight_modal.position.y + self.insight_modal.size.y - 100.0,
                        );
                        if self.mouse_pos.x >= text_rect.0 && self.mouse_pos.x <= text_rect.2 &&
                           self.mouse_pos.y >= text_rect.1 && self.mouse_pos.y <= text_rect.3 {
                            self.insight_modal.is_editing_text = true;
                            self.insight_modal.text_input.on_focus();
                            self.focused_input = Some(7);
                            return;
                        }
                    }
                }
                
                // Old notepad_modal handling removed - now using NotepadWindow's modal
                
                // ===== Z-INDEX 20: Chat window =====
                // Check chat window BEFORE sidebar to prioritize chat bar clicks
                // Only consume click if something clickable was hit (not background)
                let send_provider = self.settings_state.provider.clone();
                let send_model_id = self.settings_state.model_id_for_send();
                let send_openai_model = self.settings_state.openai_model_for_send();
                let mut send_pending: Option<(String, crate::api::models::GraphSendRequest, String)> = None;
                if let Some(ref mut chat) = self.chat_window {
                    if self.ui_state.active_tab == Tab::Chat {
                        // When constellation (graph) view is active, skip linear list hit tests
                        // and go straight to hit_test which handles constellation nodes
                        let use_linear_list = self.graph_state.graph_id.is_none();
                        let bubbles = if use_linear_list {
                            Some(chat.get_message_bubbles(|text, size| {
                                let char_width = size * 0.6;
                                let chars = text.chars().count();
                                Vec2::new(chars as f32 * char_width, size * 1.2)
                            }))
                        } else {
                            None
                        };
                        
                        if use_linear_list {
                        if let Some((msg_idx, citation_idx)) = chat.get_citation_at(self.mouse_pos, bubbles.as_ref().unwrap()) {
                            if msg_idx < chat.messages.len() {
                                let msg = &chat.messages[msg_idx];
                                if citation_idx < msg.citations.len() {
                                    let citation = &msg.citations[citation_idx];
                                    let source = citation.source.clone();
                                    self.pdf_modal.open(source.clone(), None);
                                    let base_url = self.api_client.base_url.clone();
                                    let client = self.api_client.client.clone();
                                    self.pdf_modal.loading = true;
                                    tokio::spawn(async move {
                                        let url = format!("{}/papers/{}", base_url, source);
                                        match client.get(&url).send().await {
                                            Ok(resp) => {
                                                let status = resp.status();
                                                if status.is_success() {
                                                    if let Err(e) = resp.bytes().await {
                                                        eprintln!("Failed to read PDF bytes: {}", e);
                                                    }
                                                } else {
                                                    eprintln!("Failed to load PDF: HTTP {}", status);
                                                }
                                            }
                                            Err(e) => eprintln!("Failed to fetch PDF: {}", e),
                                        }
                                    });
                                    return;
                                }
                            }
                        }
                        
                        if let Some(msg_idx) = chat.get_pin_button_at(self.mouse_pos, bubbles.as_ref().unwrap()) {
                            if msg_idx < chat.messages.len() {
                                let msg = &chat.messages[msg_idx];
                                let shard_id = msg.shard_id.clone().unwrap_or_else(|| {
                                    format!("shard_{}", std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap()
                                        .as_nanos())
                                });
                                let existing = self.insights_state.insights.iter().find(|i| i.id == shard_id);
                                if let Some(insight) = existing {
                                    let id = insight.id.clone();
                                    self.insights_state.remove_insight(&id);
                                    let base_url = self.api_client.base_url.clone();
                                    tokio::spawn(async move {
                                        let api_client = crate::api::ApiClient::new(Some(base_url));
                                        if let Err(e) = api_client.delete_shard(&id).await {
                                            eprintln!("Failed to delete shard: {}", e);
                                        }
                                    });
                                } else {
                                    let contexts = msg.contexts.clone();
                                    let content = msg.content.clone();
                                    let notes = if msg.notes.is_empty() { None } else { Some(msg.notes.clone()) };
                                    let title = if content.len() > 60 {
                                        format!("{}...", &content[..60])
                                    } else {
                                        content.clone()
                                    };
                                    let base_url = self.api_client.base_url.clone();
                                    let conv_id = self.chat_state.current_conversation_id.clone();
                                    let content_clone = content.clone();
                                    let contexts_clone = contexts.clone();
                                    let title_clone = title.clone();
                                    let shard_id_clone = shard_id.clone();
                                    tokio::spawn(async move {
                                        let api_client = crate::api::ApiClient::new(Some(base_url));
                                        match api_client.create_or_update_shard(
                                            &shard_id_clone,
                                            &content_clone,
                                            contexts_clone,
                                            Some(title_clone),
                                            conv_id,
                                            None,
                                            notes,
                                        ).await {
                                            Ok(_) => {}
                                            Err(e) => eprintln!("Failed to pin shard: {}", e),
                                        }
                                    });
                                    self.insights_loaded = false;
                                    self.load_insights();
                                }
                                return;
                            }
                        }
                        } // end use_linear_list
                        
                        let hit = chat.hit_test(self.mouse_pos, &self.graph_state);
                        match hit {
                            crate::ui::chat_window::ChatHit::EditButton(msg_idx) => {
                                chat.start_editing_message(msg_idx);
                                self.focused_input = Some(10); // Use index 10 for edit textarea
                                return;
                            }
                            crate::ui::chat_window::ChatHit::DeleteButton(msg_idx) => {
                                if chat.delete_confirm_idx == Some(msg_idx) {
                                    // Confirm deletion
                                    chat.delete_message(msg_idx);
                                    // Also remove from chat_state
                                    self.chat_state.delete_message(msg_idx);
                                    self.save_chat_state();
                                    self.show_success_toast("Message deleted".to_string());
                                } else {
                                    // Start delete confirmation
                                    chat.delete_confirm_idx = Some(msg_idx);
                                }
                                return;
                            }
                            crate::ui::chat_window::ChatHit::MuteButton(msg_idx) => {
                                chat.toggle_mute_message(msg_idx);
                                return;
                            }
                            crate::ui::chat_window::ChatHit::AddNoteButton(msg_idx) => {
                                chat.start_adding_note(msg_idx);
                                self.focused_input = Some(13); // Note input
                                return;
                            }
                            crate::ui::chat_window::ChatHit::RemoveNote(msg_idx, note_idx) => {
                                chat.remove_note(msg_idx, note_idx);
                                self.chat_state.set_current_messages(chat.messages.clone());
                                self.save_chat_state();
                                return;
                            }
                            crate::ui::chat_window::ChatHit::EditNote(msg_idx, note_idx) => {
                                chat.start_editing_note(msg_idx, note_idx);
                                self.focused_input = Some(13);
                                return;
                            }
                            crate::ui::chat_window::ChatHit::ContextPoolButton => {
                                chat.context_pool_dropdown.toggle();
                                return; // Consume click - something clickable was hit
                            }
                            crate::ui::chat_window::ChatHit::ContextPoolItem(index) => {
                                chat.context_pool_dropdown.select(index);
                                let collection_id = chat.context_pool_dropdown.get_selected_id();
                                chat.selected_collection_id = collection_id;
                                self.set_context_pool(collection_id);
                                return; // Consume click - something clickable was hit
                            }
                            crate::ui::chat_window::ChatHit::ContextPoolMenu => {
                                return; // Consume click - menu area is clickable
                            }
                            crate::ui::chat_window::ChatHit::ContextPoolCreate => {
                                self.ui_state.set_active_tab(Tab::Library);
                                chat.context_pool_dropdown.close();
                                if let Some(ref mut library) = self.library_window {
                                    library.is_creating_collection = true;
                                    library.new_collection_input.on_focus();
                                    self.focused_input = Some(9);
                                }
                                return;
                            }
                            crate::ui::chat_window::ChatHit::Input => {
                                chat.input_field.on_focus();
                                chat.context_pool_dropdown.close();
                                self.focused_input = Some(0);
                                chat.input_field.on_mouse_down(self.mouse_pos, |text, size| {
                                    let char_count = text.chars().count();
                                    let approx_width = char_count as f32 * size * 0.6;
                                    glam::Vec2::new(approx_width, size * 1.2)
                                }, self.click_count);
                                return; // Consume click - input field is clickable
                            }
                            crate::ui::chat_window::ChatHit::MessageList => {
                                // Check if clicking on edit textarea
                                if let Some(msg_idx) = chat.editing_message_idx {
                                    if chat.edit_textarea.contains(self.mouse_pos) {
                                        chat.edit_textarea.on_focus();
                                        self.focused_input = Some(10);
                                        chat.edit_textarea.on_mouse_down(self.mouse_pos, |text, size| {
                                            let char_count = text.chars().count();
                                            let approx_width = char_count as f32 * size * 0.6;
                                            glam::Vec2::new(approx_width, size * 1.2)
                                        }, self.click_count);
                                        return;
                                    }
                                }
                                // Check if clicking on note input (add/edit note)
                                if chat.adding_note_msg_idx.is_some() || chat.editing_note.is_some() {
                                    if chat.note_input.contains(self.mouse_pos) {
                                        chat.note_input.on_focus();
                                        self.focused_input = Some(13);
                                        chat.note_input.on_mouse_down(self.mouse_pos, |text, size| {
                                            let char_count = text.chars().count();
                                            let approx_width = char_count as f32 * size * 0.6;
                                            glam::Vec2::new(approx_width, size * 1.2)
                                        }, self.click_count);
                                        return;
                                    }
                                }
                                // Message list itself isn't directly clickable (citations/pin buttons handled above)
                                // Don't consume click, let it fall through to sidebar
                                chat.input_field.on_blur();
                                chat.context_pool_dropdown.close();
                                self.focused_input = None;
                            }
                            crate::ui::chat_window::ChatHit::SendButton => {
                                if let Some(text) = chat.send_message() {
                                    if self.graph_state.graph_id.is_some() {
                                        let graph_id = self.graph_state.graph_id.clone().unwrap();
                                        let leaf_id = self.graph_state.current_leaf_id.clone().unwrap_or_default();
                                        let request = crate::api::models::GraphSendRequest {
                                            current_leaf_id: leaf_id,
                                            user_draft: text.clone(),
                                            provider: send_provider.clone(),
                                            model_id: send_model_id.clone(),
                                            openai_model: send_openai_model.clone(),
                                            temperature: None,
                                            max_tokens: None,
                                            model_token_limit: None,
                                        };
                                        let user_msg = crate::ui::chat_window::ChatMessage::from_legacy(
                                            crate::ui::chat_window::MessageRole::User,
                                            text.clone(),
                                            Vec::new(),
                                            Vec::new(),
                                        );
                                        chat.add_message(user_msg.clone());
                                        self.chat_state.add_message_to_current(user_msg);
                                        send_pending = Some((graph_id, request, text));
                                    } else {
                                        log::info!("chat send skipped: no graph_id (conversation not ready)");
                                        self.show_error_toast("Conversation not ready. Please wait for it to load.".to_string());
                                    }
                                }
                            }
                            crate::ui::chat_window::ChatHit::ConstellationEditTextarea => {
                                chat.edit_textarea.on_focus();
                                self.focused_input = Some(10);
                                chat.edit_textarea.on_mouse_down(self.mouse_pos, |text, size| {
                                    let char_count = text.chars().count();
                                    glam::Vec2::new(char_count as f32 * size * 0.66, size)
                                }, self.click_count);
                                return;
                            }
                            crate::ui::chat_window::ChatHit::ConstellationNoteInput => {
                                chat.note_input.on_focus();
                                self.focused_input = Some(13);
                                chat.note_input.on_mouse_down(self.mouse_pos, |text, size| {
                                    let char_count = text.chars().count();
                                    glam::Vec2::new(char_count as f32 * size * 0.66, size)
                                }, self.click_count);
                                return;
                            }
                            crate::ui::chat_window::ChatHit::ConstellationPinButton(node_id) => {
                                let existing = self.insights_state.insights.iter().find(|i| i.id == node_id);
                                if let Some(insight) = existing {
                                    let id = insight.id.clone();
                                    self.insights_state.remove_insight(&id);
                                    let base_url = self.api_client.base_url.clone();
                                    tokio::spawn(async move {
                                        let api_client = crate::api::ApiClient::new(Some(base_url));
                                        if let Err(e) = api_client.delete_shard(&id).await {
                                            eprintln!("Failed to delete shard: {}", e);
                                        }
                                    });
                                } else if let Some(node) = self.graph_state.get_node(&node_id) {
                                    let content = node.shard.assistant_content.clone().unwrap_or_default();
                                    let title = if content.len() > 60 { format!("{}...", &content[..60]) } else { content.clone() };
                                    let base_url = self.api_client.base_url.clone();
                                    let conv_id = self.chat_state.current_conversation_id.clone();
                                    let node_id_clone = node_id.clone();
                                    tokio::spawn(async move {
                                        let api_client = crate::api::ApiClient::new(Some(base_url));
                                        if let Err(e) = api_client.create_or_update_shard(
                                            &node_id_clone,
                                            &content,
                                            vec![],
                                            Some(title),
                                            conv_id,
                                            None,
                                            None,
                                        ).await {
                                            eprintln!("Failed to pin shard: {}", e);
                                        }
                                    });
                                    self.insights_loaded = false;
                                    self.load_insights();
                                }
                                return;
                            }
                            crate::ui::chat_window::ChatHit::ConstellationHideButton(node_id) => {
                                if let Some(node) = self.graph_state.get_node_mut(&node_id) {
                                    node.shard.user_visible = false;
                                    node.shard.assistant_visible = false;
                                }
                                return;
                            }
                            crate::ui::chat_window::ChatHit::ConstellationNoteButton(node_id) => {
                                if let Some(msg_idx) = chat.messages.iter().position(|m| m.shard_id.as_ref() == Some(&node_id)) {
                                    chat.start_adding_note(msg_idx);
                                    self.focused_input = Some(13);
                                }
                                return;
                            }
                            crate::ui::chat_window::ChatHit::ConstellationAddContextButton(node_id) => {
                                if let Some(msg_idx) = chat.messages.iter().position(|m| m.shard_id.as_ref() == Some(&node_id)) {
                                    let msg = &chat.messages[msg_idx];
                                    let content = msg.content.clone();
                                    let title = if content.len() > 60 { format!("{}...", &content[..60]) } else { content.clone() };
                                    let shard_id = msg.shard_id.clone().unwrap_or_else(|| node_id.clone());
                                    let base_url = self.api_client.base_url.clone();
                                    let conv_id = self.chat_state.current_conversation_id.clone();
                                    let contexts = self.graph_state.get_node(&node_id).map(|n| n.shard.contexts.clone()).unwrap_or_default();
                                    tokio::spawn(async move {
                                        let api_client = crate::api::ApiClient::new(Some(base_url));
                                        let _ = api_client.create_or_update_shard(
                                            &shard_id,
                                            &content,
                                            contexts,
                                            Some(title),
                                            conv_id,
                                            None,
                                            None,
                                        ).await;
                                    });
                                    self.insights_loaded = false;
                                    self.load_insights();
                                }
                                return;
                            }
                            crate::ui::chat_window::ChatHit::ConstellationMoreButton(node_id) => {
                                if let Some(node) = self.graph_state.get_node(&node_id) {
                                    self.shard_modal.open(node_id.clone(), node.shard.clone());
                                }
                                return;
                            }
                            crate::ui::chat_window::ChatHit::ConstellationMessageEditButton(node_id, part) => {
                                let role = match part {
                                    crate::ui::chat_window::MessagePart::User => crate::ui::chat_window::MessageRole::User,
                                    crate::ui::chat_window::MessagePart::Assistant => crate::ui::chat_window::MessageRole::Assistant,
                                };
                                if let Some(msg_idx) = chat.messages.iter().position(|m| {
                                    m.shard_id.as_deref() == Some(node_id.as_str()) && m.role == role
                                }) {
                                    chat.start_editing_message(msg_idx);
                                    self.focused_input = Some(10);
                                }
                                return;
                            }
                            crate::ui::chat_window::ChatHit::ConstellationMessageHideButton(node_id, part) => {
                                if let Some(node) = self.graph_state.get_node_mut(&node_id) {
                                    match part {
                                        crate::ui::chat_window::MessagePart::User => {
                                            node.shard.user_visible = !node.shard.user_visible;
                                        }
                                        crate::ui::chat_window::MessagePart::Assistant => {
                                            node.shard.assistant_visible = !node.shard.assistant_visible;
                                        }
                                    }
                                }
                                return;
                            }
                            crate::ui::chat_window::ChatHit::ConstellationCitation(node_id, citation_idx) => {
                                if let Some(node) = self.graph_state.get_node(&node_id) {
                                    if let Some(citation) = node.shard.citations.get(citation_idx) {
                                        let source = citation.get("source").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                        if !source.is_empty() && source.to_lowercase().ends_with(".pdf") {
                                            let page = citation.get("page").and_then(|v| v.as_u64()).map(|p| p as u32).unwrap_or(1);
                                            let filename = format!("{}#page={}", source, page);
                                            self.pdf_modal.open(filename, Some(page));
                                            self.pdf_modal.loading = true;
                                            let base_url = self.api_client.base_url.clone();
                                            let client = self.api_client.client.clone();
                                            tokio::spawn(async move {
                                                let url = format!("{}/papers/{}", base_url, source);
                                                match client.get(&url).send().await {
                                                    Ok(resp) => {
                                                        if resp.status().is_success() {
                                                            if let Err(e) = resp.bytes().await {
                                                                eprintln!("Failed to read PDF bytes: {}", e);
                                                            }
                                                        } else {
                                                            eprintln!("Failed to load PDF: HTTP {}", resp.status());
                                                        }
                                                    }
                                                    Err(e) => eprintln!("Failed to fetch PDF: {}", e),
                                                }
                                            });
                                        }
                                    }
                                }
                                return;
                            }
                            crate::ui::chat_window::ChatHit::ConstellationEditNote(node_id, note_idx) => {
                                if let Some(msg_idx) = chat.messages.iter().position(|m| m.shard_id.as_deref() == Some(node_id.as_str())) {
                                    chat.start_editing_note(msg_idx, note_idx);
                                    self.focused_input = Some(13);
                                }
                                return;
                            }
                            crate::ui::chat_window::ChatHit::ConstellationRemoveNote(node_id, note_idx) => {
                                if let Some(msg_idx) = chat.messages.iter().position(|m| m.shard_id.as_deref() == Some(node_id.as_str())) {
                                    chat.remove_note(msg_idx, note_idx);
                                    if let Some(node) = self.graph_state.get_node_mut(&node_id) {
                                        node.shard.notes = chat.messages[msg_idx].notes.clone();
                                    }
                                    self.chat_state.set_current_messages(chat.messages.clone());
                                    self.save_chat_state();
                                    let base_url = self.api_client.base_url.clone();
                                    let node_id_clone = node_id.clone();
                                    let notes = self.graph_state.get_node(&node_id).map(|n| n.shard.notes.clone()).unwrap_or_default();
                                    tokio::spawn(async move {
                                        let api_client = crate::api::ApiClient::new(Some(base_url));
                                        let _ = api_client.update_shard(&node_id_clone, None, None, None, Some(notes)).await;
                                    });
                                }
                                return;
                            }
                            crate::ui::chat_window::ChatHit::ConstellationNode(node_id) => {
                                if self.click_count == 2 {
                                    if let Some(node) = self.graph_state.get_node(&node_id) {
                                        self.shard_modal.open(node_id.clone(), node.shard.clone());
                                    }
                                } else {
                                    self.graph_state.current_leaf_id = Some(node_id.clone());
                                    if let Some(node) = self.graph_state.get_node(&node_id) {
                                        let center = node.position + node.size * 0.5;
                                        chat.constellation_view.center_camera_on(center);
                                        chat.constellation_view.reset_zoom_to_normal();
                                    }
                                    chat.input_field.on_blur();
                                    chat.context_pool_dropdown.close();
                                    self.focused_input = None;
                                }
                                return;
                            }
                            crate::ui::chat_window::ChatHit::ConstellationBackground => {
                                chat.constellation_view.pan_drag_start = Some(self.mouse_pos);
                                self.physics_idle_timer = 0.0;
                                chat.input_field.on_blur();
                                chat.context_pool_dropdown.close();
                                self.focused_input = None;
                                return;
                            }
                            crate::ui::chat_window::ChatHit::MessageList => {
                                // Message list itself isn't directly clickable (citations/pin buttons handled above)
                                // Don't consume click, let it fall through to sidebar
                                chat.input_field.on_blur();
                                chat.context_pool_dropdown.close();
                                self.focused_input = None;
                            }
                            crate::ui::chat_window::ChatHit::Background => {
                                // Background is not clickable, don't consume click
                                chat.input_field.on_blur();
                                chat.context_pool_dropdown.close();
                                if chat.adding_note_msg_idx.is_some() || chat.editing_note.is_some() {
                                    chat.cancel_note();
                                }
                                self.focused_input = None;
                            }
                            _ => {
                                // Other hits (Citation, PinButton) are already handled above
                                // Don't consume click here
                                chat.input_field.on_blur();
                                chat.context_pool_dropdown.close();
                                if chat.adding_note_msg_idx.is_some() || chat.editing_note.is_some() {
                                    chat.cancel_note();
                                }
                                self.focused_input = None;
                            }
                        }
                        // Only reach here if hit was Background or MessageList (not clickable)
                        // Or SendButton (send_pending set, handled below)
                    }
                }
                if let Some((graph_id, request, text)) = send_pending {
                    log::info!("chat send: POST /graph/{}/send", graph_id);
                    self.save_chat_state();
                    let client = self.api_client.client.clone();
                    let base_url = self.api_client.base_url.clone();
                    let sender = self.graph_send_sender.clone();
                    tokio::spawn(async move {
                        let url = format!("{}/graph/{}/send", base_url, graph_id);
                        match client.post(&url).json(&request).send().await {
                            Ok(r) if r.status().is_success() => {
                                match r.json::<crate::api::models::GraphSendResponse>().await {
                                    Ok(body) => { let _ = sender.send(Ok((text, body))); }
                                    Err(e) => { let _ = sender.send(Err(format!("Parse: {:?}", e))); }
                                }
                            }
                            Ok(r) => {
                                let err = r.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                                let _ = sender.send(Err(err));
                            }
                            Err(e) => { let _ = sender.send(Err(format!("Request: {:?}", e))); }
                        }
                    });
                    self.is_sending_message = true;
                    if let Some(ref mut chat) = self.chat_window {
                        chat.is_sending = true;
                    }
                }
                
                // ===== Z-INDEX 10: Notepad (before sidebar when on Notepad tab) =====
                let mut notepad_toast_after = None::<(String, bool)>; // (message, is_success)
                let mut notepad_handled_with_toast = false;
                if self.ui_state.active_tab == Tab::Notepad {
                    'notepad_hits: loop {
                        if let Some(ref mut notepad) = self.notepad_window {
                            let hit = notepad.hit_test(self.mouse_pos);
                            match hit {
                                crate::ui::notepad_window::NotepadHit::TitleInput => {
                                    notepad.title_input.on_focus();
                                    self.focused_input = Some(12); // New index for title input
                                    notepad.title_input.on_mouse_down(self.mouse_pos, |text, size| {
                                        let char_count = text.chars().count();
                                        let approx_width = char_count as f32 * size * 0.6;
                                        glam::Vec2::new(approx_width, size * 1.2)
                                    }, self.click_count);
                                    return;
                                }
                                crate::ui::notepad_window::NotepadHit::NewButton => {
                                    notepad.create_new_note();
                                    self.show_success_toast("New note created".to_string());
                                    return;
                                }
                                crate::ui::notepad_window::NotepadHit::SaveButton => {
                                    match notepad.save_note() {
                                        Ok(_) => {
                                            self.show_success_toast("Note saved".to_string());
                                        }
                                        Err(e) => {
                                            self.show_error_toast(format!("Failed to save: {}", e));
                                        }
                                    }
                                    return;
                                }
                                crate::ui::notepad_window::NotepadHit::OpenButton => {
                                    notepad.open_modal();
                                    return;
                                }
                                crate::ui::notepad_window::NotepadHit::DeleteButton => {
                                    match notepad.delete_note() {
                                        Ok(_) => {
                                            self.show_success_toast("Note deleted".to_string());
                                        }
                                        Err(e) => {
                                            self.show_error_toast(format!("Failed to delete: {}", e));
                                        }
                                    }
                                    return;
                                }
                                crate::ui::notepad_window::NotepadHit::ToolbarButton(button) => {
                                    use crate::ui::ToolbarButton;
                                    use crate::stylus::formatting::TextFormat;
                                    match button {
                                        ToolbarButton::Bold => {
                                            notepad.editor.apply_format(TextFormat::Bold);
                                        }
                                        ToolbarButton::Italic => {
                                            notepad.editor.apply_format(TextFormat::Italic);
                                        }
                                        ToolbarButton::Underline => {
                                            notepad.editor.apply_format(TextFormat::Underline);
                                        }
                                        ToolbarButton::Strikethrough => {
                                            notepad.editor.apply_format(TextFormat::Strikethrough);
                                        }
                                        ToolbarButton::Code => {
                                            notepad.editor.apply_format(TextFormat::Code);
                                        }
                                        ToolbarButton::Link => {
                                            // TODO: Open dialog for URL input
                                            // For now, just apply link format with empty URL
                                            notepad.editor.apply_format(TextFormat::Link { url: String::new() });
                                        }
                                    }
                                    return;
                                }
                                crate::ui::notepad_window::NotepadHit::ModalClose => {
                                    notepad.notepad_modal.close();
                                    return;
                                }
                                crate::ui::notepad_window::NotepadHit::ModalPaper(index) => {
                                    if index < notepad.notepad_modal.filtered_papers.len() {
                                        notepad.notepad_modal.selected_paper_index = Some(index);
                                        if self.click_count >= 2 {
                                            let doc_id = notepad.notepad_modal.filtered_papers.get(index).and_then(|p| p.filename.strip_suffix(".json").map(|s| s.to_string()));
                                            if let Some(ref doc_id) = doc_id {
                                                match notepad.load_note(doc_id) {
                                                    Err(e) => notepad_toast_after = Some((format!("Failed to load note: {}", e), false)),
                                                    Ok(()) => {
                                                        notepad.notepad_modal.close();
                                                        notepad_toast_after = Some(("Note loaded".to_string(), true));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    notepad_handled_with_toast = true;
                                    break 'notepad_hits;
                                }
                                crate::ui::notepad_window::NotepadHit::ModalDelete => {
                                    if let Some(index) = notepad.notepad_modal.selected_paper_index {
                                        if index < notepad.notepad_modal.filtered_papers.len() {
                                            let doc_id = notepad.notepad_modal.filtered_papers.get(index).and_then(|p| p.filename.strip_suffix(".json").map(|s| s.to_string()));
                                            if let Some(doc_id) = doc_id {
                                                use crate::persistence::DocumentPersistence;
                                                match DocumentPersistence::delete_document(&doc_id) {
                                                    Err(e) => notepad_toast_after = Some((format!("Failed to delete note: {}", e), false)),
                                                    Ok(()) => {
                                                        notepad.notepad_modal.selected_paper_index = None;
                                                        notepad.refresh_modal_documents();
                                                        notepad_toast_after = Some(("Note deleted".to_string(), true));
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        notepad_toast_after = Some(("Please select a note to delete".to_string(), false));
                                    }
                                    notepad_handled_with_toast = true;
                                    break 'notepad_hits;
                                }
                            crate::ui::notepad_window::NotepadHit::ModalImport => {
                                // Import note from file
                                use rfd::FileDialog;
                                if let Some(path) = FileDialog::new()
                                    .add_filter("Markdown", &["md", "txt"])
                                    .add_filter("Text", &["txt"])
                                    .add_filter("All Files", &["*"])
                                    .pick_file()
                                {
                                    if let Ok(contents) = std::fs::read_to_string(&path) {
                                        notepad.editor.load_from_markdown(&contents);
                                        // Set title from filename
                                        if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                                            let title = filename.strip_suffix(".md")
                                                .or_else(|| filename.strip_suffix(".txt"))
                                                .unwrap_or(filename);
                                            notepad.document_title = title.to_string();
                                            notepad.title_input.text = title.to_string();
                                            notepad.editor.document.set_title(title.to_string());
                                        }
                                        self.show_success_toast("Note imported".to_string());
                                    } else {
                                        self.show_error_toast("Failed to read file".to_string());
                                    }
                                }
                                return;
                            }
                            crate::ui::notepad_window::NotepadHit::Editor => {
                                self.pending_notepad_click = Some(self.mouse_pos);
                                // Start text selection on mouse down
                                use crate::ui::TextEditor;
                                TextEditor::on_mouse_down(&mut notepad.editor, self.mouse_pos, self.click_count);
                                // Focus the first block if no block is focused
                                if notepad.editor.cursor.is_none() && !notepad.editor.document.blocks.is_empty() {
                                    let first_block_id = notepad.editor.document.blocks[0].id.clone();
                                    notepad.editor.focus_block(&first_block_id);
                                }
                                self.focused_input = Some(5);
                                return; // Consume click - notepad takes precedence
                            }
                            _ => {
                                // Background or other - blur inputs
                                notepad.title_input.on_blur();
                                use crate::ui::TextEditor;
                                TextEditor::on_mouse_up(&mut notepad.editor);
                                self.focused_input = None;
                            }
                        }
                        break;
                    }
                }
                }
                if let Some((msg, success)) = notepad_toast_after {
                    if success { self.show_success_toast(msg); } else { self.show_error_toast(msg); }
                }
                if notepad_handled_with_toast { return; }
                
                // ===== Z-INDEX 11: Sidebar content =====
                if self.sidebar.hit_test(self.mouse_pos) {
                    use crate::persistence::DocumentPersistence;
                    
                    // Check buttons first
                    if self.sidebar.get_new_conversation_button_at(self.mouse_pos) {
                        let conv_id = self.chat_state.create_conversation();
                        self.sidebar.selected_conversation_id = Some(conv_id);
                        self.request_graph_for_new_conversation();
                        self.save_chat_state();
                        return;
                    }
                    
                    if self.sidebar.get_new_document_button_at(self.mouse_pos) {
                        if let Some(ref mut notepad) = self.notepad_window {
                            let doc_id = notepad.editor.create_new_document();
                            self.sidebar.selected_document_id = Some(doc_id);
                        }
                        return;
                    }
                    
                    if self.sidebar.get_delete_conversation_button_at(self.mouse_pos) {
                        if let Some(ref conv_id) = self.sidebar.selected_conversation_id {
                            self.chat_state.delete_conversation(conv_id);
                            ConversationPersistence::delete_conversation(conv_id).ok();
                            self.sidebar.selected_conversation_id = None;
                            self.sidebar.conversations_list.clear_selection_border();
                            self.save_chat_state();
                        }
                        return;
                    }
                    
                    if self.sidebar.get_delete_document_button_at(self.mouse_pos) {
                        if let Some(ref doc_id) = self.sidebar.selected_document_id {
                            DocumentPersistence::delete_document(doc_id).ok();
                            self.sidebar.selected_document_id = None;
                            // Create new document
                            if let Some(ref mut notepad) = self.notepad_window {
                                let new_doc_id = notepad.editor.create_new_document();
                                self.sidebar.selected_document_id = Some(new_doc_id);
                            }
                        }
                        return;
                    }
                    
                    // Check if clicked on conversations list
                    let conv_index = self.sidebar.get_conversation_at(self.mouse_pos, &self.chat_state.conversations);
                    if let Some(index) = conv_index {
                        if index < self.chat_state.conversations.len() {
                            // Calculate handle/button positions
                            // Match layout from gfx/components/sidebar_content (item_rect.right() - handle_size - padding, same button_size)
                            let item_height = 40.0;
                            let start_y = self.sidebar.conversations_list.scroll_view.position.y;
                            let scroll_offset = self.sidebar.conversations_list.scroll_view.scroll_offset;
                            let y = start_y + (index as f32 * item_height) - scroll_offset;
                            let padding = crate::ui::style::padding::MEDIUM;
                            let handle_size = 24.0;
                            let handle_x = self.sidebar.position.x + self.sidebar.current_width - padding - handle_size - padding;
                            let handle_y = y + (item_height - handle_size) / 2.0;
                            let button_size = crate::ui::style::font_size::SMALL + crate::ui::style::padding::TINY * 2.0;
                            let button_spacing = crate::ui::style::padding::TINY;

                            let is_expanded = self.sidebar.conversations_list.expanded_index == Some(index);

                            if is_expanded {
                                let info_x = handle_x - button_size - button_spacing;
                                let delete_x = info_x - button_size - button_spacing;
                                if self.mouse_pos.x >= info_x && self.mouse_pos.x <= info_x + button_size &&
                                   self.mouse_pos.y >= handle_y && self.mouse_pos.y <= handle_y + button_size {
                                    let conv_id = self.chat_state.conversations[index].id.clone();
                                    let conv_title = self.chat_state.conversations[index].title.clone();
                                    self.chat_info_dialog.open(conv_id, conv_title);
                                    return;
                                }
                                if self.mouse_pos.x >= delete_x && self.mouse_pos.x <= delete_x + button_size &&
                                   self.mouse_pos.y >= handle_y && self.mouse_pos.y <= handle_y + button_size {
                                    let conv_id = self.chat_state.conversations[index].id.clone();
                                    self.chat_state.delete_conversation(&conv_id);
                                    ConversationPersistence::delete_conversation(&conv_id).ok();
                                    self.sidebar.conversations_list.expanded_index = None;
                                    self.save_chat_state();
                                    return;
                                }
                            }

                            if self.mouse_pos.x >= handle_x && self.mouse_pos.x <= handle_x + handle_size &&
                               self.mouse_pos.y >= handle_y && self.mouse_pos.y <= handle_y + handle_size {
                                // Toggle expanded state
                                if self.sidebar.conversations_list.expanded_index == Some(index) {
                                    self.sidebar.conversations_list.expanded_index = None;
                                } else {
                                    self.sidebar.conversations_list.expanded_index = Some(index);
                                }
                                return;
                            }
                            
                            // Otherwise, load the conversation (don't open modal)
                            let conv_id = self.chat_state.conversations[index].id.clone();
                            let graph_id = self.chat_state.conversations[index].graph_id.clone();
                            
                            self.chat_state.switch_conversation(&conv_id);
                            self.sidebar.selected_conversation_id = Some(conv_id.clone());
                            if let Some(gid) = graph_id {
                                self.request_graph_load(gid);
                            } else {
                                // No graph yet (legacy or new): clear then create one so constellation view is used
                                self.graph_state.clear();
                                self.request_graph_for_new_conversation();
                            }
                            
                            // Update selection border animation target
                            
                            // Sync messages to chat window. When conv has graph_id, messages are rebuilt when graph loads.
                            if let Some(ref mut chat) = self.chat_window {
                                let use_shards = self.chat_state.current_conversation_id.as_ref()
                                    .and_then(|cid| self.chat_state.conversations.iter().find(|c| &c.id == cid))
                                    .map(|c| c.graph_id.is_none())
                                    .unwrap_or(true);
                                if use_shards {
                                    chat.messages = self.chat_state.get_current_messages();
                                }
                            }
                        }
                    }
                    
                    // Check if clicked on documents list
                    if let Ok(document_ids) = DocumentPersistence::list_documents() {
                        if let Some(index) = self.sidebar.get_document_at(self.mouse_pos, &document_ids) {
                            if index < document_ids.len() {
                                let item_height = 40.0;
                                let docs_start_y = self.sidebar.documents_list.scroll_view.position.y;
                                let docs_scroll_offset = self.sidebar.documents_list.scroll_view.scroll_offset;
                                let y = docs_start_y + (index as f32 * item_height) - docs_scroll_offset;
                                let padding = crate::ui::style::padding::MEDIUM;
                                let handle_size = 24.0;
                                let handle_x = self.sidebar.position.x + self.sidebar.current_width - padding - handle_size - padding;
                                let handle_y = y + (item_height - handle_size) / 2.0;
                                let button_size = crate::ui::style::font_size::SMALL + crate::ui::style::padding::TINY * 2.0;
                                let button_spacing = crate::ui::style::padding::TINY;

                                let is_expanded = self.sidebar.documents_list.expanded_index == Some(index);

                                if is_expanded {
                                    let info_x = handle_x - button_size - button_spacing;
                                    let delete_x = info_x - button_size - button_spacing;
                                    if self.mouse_pos.x >= info_x && self.mouse_pos.x <= info_x + button_size &&
                                       self.mouse_pos.y >= handle_y && self.mouse_pos.y <= handle_y + button_size {
                                        // TODO: Open document info modal
                                        self.sidebar.documents_list.expanded_index = None;
                                        return;
                                    }
                                    if self.mouse_pos.x >= delete_x && self.mouse_pos.x <= delete_x + button_size &&
                                       self.mouse_pos.y >= handle_y && self.mouse_pos.y <= handle_y + button_size {
                                        let doc_id = &document_ids[index];
                                        DocumentPersistence::delete_document(doc_id).ok();
                                        self.sidebar.documents_list.expanded_index = None;
                                        self.sidebar.selected_document_id = None;
                                        if let Some(ref mut notepad) = self.notepad_window {
                                            let new_doc_id = notepad.editor.create_new_document();
                                            self.sidebar.selected_document_id = Some(new_doc_id);
                                        }
                                        return;
                                    }
                                }

                                if self.mouse_pos.x >= handle_x && self.mouse_pos.x <= handle_x + handle_size &&
                                   self.mouse_pos.y >= handle_y && self.mouse_pos.y <= handle_y + handle_size {
                                    // Toggle expanded state
                                    if self.sidebar.documents_list.expanded_index == Some(index) {
                                        self.sidebar.documents_list.expanded_index = None;
                                    } else {
                                        self.sidebar.documents_list.expanded_index = Some(index);
                                    }
                                    return;
                                }
                                
                                // Otherwise, load the document
                                let doc_id = document_ids[index].clone();
                                self.sidebar.selected_document_id = Some(doc_id.clone());
                                
                                // Load document into notepad
                                if let Some(ref mut notepad) = self.notepad_window {
                                    if let Err(e) = notepad.editor.load_document(&doc_id) {
                                        eprintln!("Failed to load document {}: {}", doc_id, e);
                                    }
                                }
                            }
                        }
                    }
                    
                    // Check if clicked on insights list
                    if let Some(index) = self.sidebar.get_insight_at(self.mouse_pos, &self.insights_state.insights) {
                        if index < self.insights_state.insights.len() {
                            const INSIGHTS_ITEM_H: f32 = 35.0;
                            let pos = self.sidebar.insights_panel.insights_list.scroll_view.position;
                            let list_width = self.sidebar.insights_panel.insights_list.scroll_view.size.x;
                            let scroll = self.sidebar.insights_panel.insights_list.scroll_view.scroll_offset;
                            let padding = 10.0;
                            let handle_size = 24.0;
                            let item_y = pos.y + (index as f32 * INSIGHTS_ITEM_H) - scroll;
                            let handle_x = pos.x + list_width - padding - handle_size;
                            let handle_y = item_y + (INSIGHTS_ITEM_H - handle_size) / 2.0;
                            let on_handle = self.mouse_pos.x >= handle_x && self.mouse_pos.x <= handle_x + handle_size
                                && self.mouse_pos.y >= handle_y && self.mouse_pos.y <= handle_y + handle_size;
                            if on_handle {
                                let expanded = self.sidebar.insights_panel.insights_list.expanded_index;
                                self.sidebar.insights_panel.insights_list.expanded_index = if expanded == Some(index) { None } else { Some(index) };
                                return;
                            }
                            let insight = self.insights_state.insights[index].clone();
                            self.sidebar.selected_insight_id = Some(insight.id.clone());
                            self.insights_state.set_modal_insight(Some(insight));
                            self.insight_modal.open(self.insights_state.modal_insight.as_ref().unwrap().clone());
                        }
                        return;
                    }
                    
                    // If we're in the sidebar area but didn't click on any specific item,
                    // only close if clicking in the ~10px wide area on the inner edge (right side)
                    let close_area_width = 10.0;
                    let sidebar_right_edge = self.sidebar.current_width;
                    if self.mouse_pos.x >= sidebar_right_edge - close_area_width && 
                       self.mouse_pos.x <= sidebar_right_edge {
                        self.sidebar.toggle();
                        self.ui_state.sidebar_open = self.sidebar.is_open;
                        self.bump_layout_generation();
                        return;
                    }
                }
                
                // ===== Z-INDEX 10: Sidebar background =====
                // (Sidebar background doesn't need click handling, content is handled above)
                
                // ===== Z-INDEX 5: Sidebar toggle/glow =====
                // Check for clicks in 100px margin area to collapse/expand sidebar
                let header_height = self.header.size.y;
                let sidebar_margin = 100.0;
                
                if self.sidebar.is_open {
                    // When open: check if click is in the 100px margin to the right of sidebar
                    let sidebar_right_edge = self.sidebar.current_width;
                    if self.mouse_pos.x >= sidebar_right_edge && 
                       self.mouse_pos.x < sidebar_right_edge + sidebar_margin &&
                       self.mouse_pos.y >= header_height {
                        self.sidebar.toggle();
                        self.ui_state.sidebar_open = self.sidebar.is_open;
                        self.bump_layout_generation();
                        return;
                    }
                } else {
                    // When closed: check if click is within 100px from left edge
                    if self.mouse_pos.x < sidebar_margin && 
                       self.mouse_pos.y >= header_height {
                        self.sidebar.toggle();
                        self.ui_state.sidebar_open = self.sidebar.is_open;
                        self.bump_layout_generation();
                        return;
                    }
                }
                
                let toggle_button_size = 40.0;
                let toggle_x = if self.sidebar.is_open { self.sidebar.current_width - toggle_button_size } else { 8.0 };
                let toggle_rect = Vec2::new(toggle_x, header_height + 8.0);
                let toggle_size = Vec2::new(toggle_button_size, toggle_button_size);
                
                if self.mouse_pos.x >= toggle_rect.x && self.mouse_pos.x <= toggle_rect.x + toggle_size.x &&
                   self.mouse_pos.y >= toggle_rect.y && self.mouse_pos.y <= toggle_rect.y + toggle_size.y {
                    self.sidebar.toggle();
                    self.bump_layout_generation();
                    return;
                }
                
                // Handle other window interactions (library, ingest, settings, notepad)
                // These are at z-index 20 (same as chat) but only active when their tab is selected
                
                // Handle library window interactions
                if let Some(ref mut library) = self.library_window {
                    if self.ui_state.active_tab == Tab::Library {
                        // New collection button
                        if library.new_collection_button.contains(self.mouse_pos) {
                            library.is_creating_collection = !library.is_creating_collection;
                            if library.is_creating_collection {
                                library.new_collection_input.on_focus();
                                self.focused_input = Some(9);
                            } else {
                                library.new_collection_input.on_blur();
                                self.focused_input = None;
                            }
                            return;
                        }
                        
                        // New collection input
                        if library.is_creating_collection && library.new_collection_input.contains(self.mouse_pos) {
                            library.new_collection_input.on_focus();
                            self.focused_input = Some(9);
                            library.new_collection_input.on_mouse_down(self.mouse_pos, |text, size| {
                                let char_count = text.chars().count();
                                let approx_width = char_count as f32 * size * 0.6;
                                glam::Vec2::new(approx_width, size * 1.2)
                            }, self.click_count);
                            return;
                        }
                        
                        if library.search_input.contains(self.mouse_pos) {
                            library.search_input.on_focus();
                            self.focused_input = Some(1);
                            library.search_input.on_mouse_down(self.mouse_pos, |text, size| {
                                let char_count = text.chars().count();
                                let approx_width = char_count as f32 * size * 0.6;
                                glam::Vec2::new(approx_width, size * 1.2)
                            }, self.click_count);
                            return;
                        } else if library.collections_list.contains(self.mouse_pos - library.position) {
                            // Clicked on collections list
                            if let Some(index) = library.get_collection_at(self.mouse_pos) {
                                if index == usize::MAX {
                                    // "All papers" option
                                    library.select_collection(None);
                                } else if index < library.collections.len() {
                                    let collection = &library.collections[index];
                                    library.select_collection(Some(collection.id));
                                }
                            }
                            library.search_input.on_blur();
                            library.new_collection_input.on_blur();
                            self.focused_input = None;
                            return;
                        } else if library.papers_list.contains(self.mouse_pos - library.position) {
                            // Check if clicking on delete button
                            if library.delete_button.contains(self.mouse_pos) {
                                if library.delete_confirm {
                                    // Confirm deletion
                                    let selected_ids = library.get_selected_paper_ids();
                                    if !selected_ids.is_empty() {
                                        let base_url = self.api_client.base_url.clone();
                                        let selected_ids_clone = selected_ids.clone();
                                        let sender = self.papers_sender.clone();
                                        tokio::spawn(async move {
                                            let api_client = crate::api::ApiClient::new(Some(base_url.clone()));
                                            let mut errors = Vec::new();
                                            for paper_id in selected_ids_clone {
                                                if let Err(e) = api_client.delete_note(paper_id).await {
                                                    errors.push(format!("Failed to delete paper {}: {}", paper_id, e));
                                                }
                                            }
                                            
                                            // Reload papers after deletion
                                            let client = reqwest::Client::new();
                                            let url = format!("{}/papers", base_url);
                                            if let Ok(resp) = client.get(&url).send().await {
                                                if resp.status().is_success() {
                                                    if let Ok(papers) = resp.json::<Vec<crate::api::models::ApiPaper>>().await {
                                                        let _ = sender.send(Ok(papers));
                                                    }
                                                }
                                            }
                                            
                                            if !errors.is_empty() {
                                                eprintln!("Deletion errors: {:?}", errors);
                                            }
                                        });
                                        
                                        library.clear_selection();
                                        library.delete_confirm = false;
                                        self.show_success_toast(format!("Deleting {} paper(s)...", selected_ids.len()));
                                    }
                                } else {
                                    // Start delete confirmation
                                    library.delete_confirm = true;
                                }
                                return;
                            } else if library.delete_confirm {
                                // Clicked elsewhere while in confirm mode - cancel confirmation
                                library.delete_confirm = false;
                                return;
                            }
                            
                            // Check if clicking on checkbox
                            if let Some(index) = library.get_paper_at(self.mouse_pos) {
                                if index < library.filtered_papers.len() {
                                    let paper = &library.filtered_papers[index];
                                    let checkbox_pos = library.get_checkbox_position(index);
                                    const CHECKBOX_SIZE: f32 = 16.0;
                                    let checkbox_rect = glam::Vec2::new(checkbox_pos.x, checkbox_pos.y);
                                    let checkbox_end = checkbox_rect + glam::Vec2::new(CHECKBOX_SIZE, CHECKBOX_SIZE);
                                    
                                    if self.mouse_pos.x >= checkbox_rect.x && self.mouse_pos.x <= checkbox_end.x &&
                                       self.mouse_pos.y >= checkbox_rect.y && self.mouse_pos.y <= checkbox_end.y {
                                        // Clicked on checkbox
                                        library.toggle_paper_selection(paper.id);
                                        library.delete_confirm = false;  // Reset confirmation when selection changes
                                        return;
                                    }
                                    
                                    // Open PDF modal for PDF files (if not clicking checkbox)
                                    if paper.filename.to_lowercase().ends_with(".pdf") {
                                        self.pdf_modal.open(paper.filename.clone(), None);
                                        self.pdf_modal.loading = true;
                                        let base_url = self.api_client.base_url.clone();
                                        let client = self.api_client.client.clone();
                                        let filename = paper.filename.clone();
                                        tokio::spawn(async move {
                                            let url = format!("{}/papers/{}", base_url, filename);
                                            match client.get(&url).send().await {
                                                Ok(resp) => {
                                                    if resp.status().is_success() {
                                                        if let Ok(_bytes) = resp.bytes().await {
                                                            // PDF bytes received
                                                        }
                                                    }
                                                }
                                                Err(e) => eprintln!("Failed to fetch PDF: {}", e),
                                            }
                                        });
                                    }
                                }
                            }
                            library.search_input.on_blur();
                            library.new_collection_input.on_blur();
                            self.focused_input = None;
                            return;
                        } else {
                            library.search_input.on_blur();
                            library.new_collection_input.on_blur();
                            self.focused_input = None;
                        }
                    }
                }

                // Handle ingest window interactions
                if let Some(ref mut ingest) = self.ingest_window {
                    if self.ui_state.active_tab == Tab::Data {
                        if ingest.pdf_dir_input.contains(self.mouse_pos) {
                            ingest.pdf_dir_input.on_focus();
                            self.focused_input = Some(2);
                            ingest.pdf_dir_input.on_mouse_down(self.mouse_pos, |text, size| {
                                let char_count = text.chars().count();
                                let approx_width = char_count as f32 * size * 0.6;
                                glam::Vec2::new(approx_width, size * 1.2)
                            }, self.click_count);
                        } else if ingest.is_browse_button_clicked(self.mouse_pos) {
                            ingest.browse_button.on_press();
                            self.open_file_picker();
                        } else if ingest.ingest_button.contains(self.mouse_pos) {
                            ingest.ingest_button.on_press();
                            self.start_ingestion();
                        } else {
                            ingest.pdf_dir_input.on_blur();
                            self.focused_input = None;
                        }
                    }
                }

                // Handle settings window interactions
                if let Some(ref mut settings) = self.settings_window {
                    if self.ui_state.active_tab == Tab::Settings {
                        // Theme selection area (click to cycle through themes)
                        let theme_area = crate::ui::core::Rect::new(
                            settings.position.x + 20.0,
                            settings.position.y + 400.0, // Approximate position based on rendering
                            300.0,
                            30.0,
                        );
                        if theme_area.contains_point(self.mouse_pos) {
                            // Cycle to next theme
                            settings.selected_theme = (settings.selected_theme + 1) % 8;
                            let theme_names = [
                                "standard",
                                "sakura-light",
                                "springtime-light",
                                "forest-dark",
                                "toadstool-light",
                                "acorn-dark",
                                "light",
                                "dark",
                            ];
                            if settings.selected_theme < theme_names.len() {
                                self.settings_state.theme = theme_names[settings.selected_theme].to_string();
                                use crate::persistence::SettingsPersistence;
                                if let Err(e) = SettingsPersistence::save_settings(&self.settings_state) {
                                    eprintln!("Failed to save settings: {}", e);
                                }
                            }
                            return;
                        } else if settings.provider_selector_rect.contains_point(self.mouse_pos) {
                            self.settings_state.provider = if self.settings_state.provider == "openai" {
                                "local".to_string()
                            } else {
                                "openai".to_string()
                            };
                            use crate::persistence::SettingsPersistence;
                            if let Err(e) = SettingsPersistence::save_settings(&self.settings_state) {
                                eprintln!("Failed to save settings: {}", e);
                            }
                        } else if settings.hf_token_input.contains(self.mouse_pos) && self.settings_state.provider == "local" {
                            settings.hf_token_input.on_focus();
                            self.focused_input = Some(3);
                            settings.hf_token_input.on_mouse_down(self.mouse_pos, |text, size| {
                                let char_count = text.chars().count();
                                let approx_width = char_count as f32 * size * 0.6;
                                glam::Vec2::new(approx_width, size * 1.2)
                            }, self.click_count);
                        } else if settings.model_id_input.contains(self.mouse_pos) && self.settings_state.provider == "local" {
                            settings.model_id_input.on_focus();
                            self.focused_input = Some(4);
                            settings.model_id_input.on_mouse_down(self.mouse_pos, |text, size| {
                                let char_count = text.chars().count();
                                let approx_width = char_count as f32 * size * 0.6;
                                glam::Vec2::new(approx_width, size * 1.2)
                            }, self.click_count);
                        } else if settings.openai_model_input.contains(self.mouse_pos) && self.settings_state.provider == "openai" {
                            settings.openai_model_input.on_focus();
                            self.focused_input = Some(6);
                            settings.openai_model_input.on_mouse_down(self.mouse_pos, |text, size| {
                                let char_count = text.chars().count();
                                let approx_width = char_count as f32 * size * 0.6;
                                glam::Vec2::new(approx_width, size * 1.2)
                            }, self.click_count);
                        } else {
                            if settings.hf_token_input.focused {
                                settings.hf_token_input.on_blur();
                                self.settings_state.hf_token = settings.hf_token_input.text.clone();
                                use crate::persistence::SettingsPersistence;
                                if let Err(e) = SettingsPersistence::save_settings(&self.settings_state) {
                                    eprintln!("Failed to save settings: {}", e);
                                }
                            }
                            if settings.model_id_input.focused {
                                settings.model_id_input.on_blur();
                                self.settings_state.model_id = settings.model_id_input.text.clone();
                                use crate::persistence::SettingsPersistence;
                                if let Err(e) = SettingsPersistence::save_settings(&self.settings_state) {
                                    eprintln!("Failed to save settings: {}", e);
                                }
                            }
                            if settings.openai_model_input.focused {
                                settings.openai_model_input.on_blur();
                                self.settings_state.openai_model = settings.openai_model_input.text.clone();
                                use crate::persistence::SettingsPersistence;
                                if let Err(e) = SettingsPersistence::save_settings(&self.settings_state) {
                                    eprintln!("Failed to save settings: {}", e);
                                }
                            }
                            self.focused_input = None;
                        }
                    }
                }
                
                if let Some(ref mut settings) = self.settings_window {
                    settings.hf_token_input.text = self.settings_state.hf_token.clone();
                    settings.model_id_input.text = self.settings_state.model_id.clone();
                    settings.openai_model_input.text = self.settings_state.openai_model.clone();
                }

                // (Modal handling moved to z-index 50 section above)
                // (Notepad handling moved to z-index 10 section above, before sidebar)

                for (id, win) in self.windows.iter_mut().enumerate().rev() {
                    if win.hit_title_bar(self.mouse_pos) {
                        self.dragging_id = Some(id);
                        self.drag_offset = self.mouse_pos - win.position;
                        let w = self.windows.remove(id);
                        self.windows.push(w);
                        self.dragging_id = Some(self.windows.len() - 1);
                        return;
                    }
                }
            }
            ElementState::Released => {
                // Handle drag end
                if self.is_dragging {
                    self.on_mouse_drag_end(self.mouse_pos);
                }
                
                // Stop window dragging
                self.is_dragging_window = false;
                
                if let Some(id) = self.dragging_id.take() {
                    if id < self.windows.len() {
                        self.windows[id].velocity = Vec2::ZERO;
                    }
                }
            }
        }
    }
    
    /// Handle right mouse button click (context menu)
    pub fn on_mouse_right_click(&mut self) {
        // Route to component under cursor for context menu
        use crate::ui::tab_bar::Tab;
        
        // Check z-index order (highest to lowest)
        // Modals first
        if self.insight_modal.is_open && self.insight_modal.contains(self.mouse_pos) {
            self.on_context_menu(self.mouse_pos, "insight_modal");
            return;
        }
        if self.pdf_modal.is_open && self.pdf_modal.contains(self.mouse_pos) {
            self.on_context_menu(self.mouse_pos, "pdf_modal");
            return;
        }
        if self.chat_info_dialog.is_open && self.chat_info_dialog.contains(self.mouse_pos) {
            self.on_context_menu(self.mouse_pos, "chat_info_dialog");
            return;
        }
        
        // Windows based on active tab
        match self.ui_state.active_tab {
            Tab::Chat => {
                if let Some(ref chat) = self.chat_window {
                    if self.mouse_pos.x >= chat.position.x && self.mouse_pos.x <= chat.position.x + chat.size.x &&
                       self.mouse_pos.y >= chat.position.y && self.mouse_pos.y <= chat.position.y + chat.size.y {
                        self.on_context_menu(self.mouse_pos, "chat_window");
                        return;
                    }
                }
            }
            Tab::Library => {
                if let Some(ref library) = self.library_window {
                    if library.contains(self.mouse_pos) {
                        self.on_context_menu(self.mouse_pos, "library_window");
                        return;
                    }
                }
            }
            Tab::Notepad => {
                if let Some(ref notepad) = self.notepad_window {
                    if notepad.editor.contains(self.mouse_pos) {
                        self.on_context_menu(self.mouse_pos, "notepad_editor");
                        return;
                    }
                }
            }
            _ => {}
        }
        
        // Sidebar
        if self.sidebar.hit_test(self.mouse_pos) {
            self.on_context_menu(self.mouse_pos, "sidebar");
            return;
        }
        
        // Default: show window context menu
        self.on_context_menu(self.mouse_pos, "window");
    }
    
    /// Show context menu at position for component
    fn on_context_menu(&mut self, _position: Vec2, component_id: &str) {
        // TODO: Implement context menu rendering
        match component_id {
            "notepad_editor" => {
                // Context menu for text editor: Cut, Copy, Paste, etc.
            }
            "chat_window" => {
                // Context menu for chat: Copy message, etc.
            }
            "sidebar" => {
                // Context menu for sidebar items: Delete, Rename, etc.
            }
            _ => {
                // Default context menu
            }
        }
    }
    
    /// Handle accessibility focus (screen reader focus)
    pub fn on_accessibility_focus(&mut self, component_id: String) {
        self.accessibility_focused_component = Some(component_id);
        // TODO: Announce component to screen reader and update ARIA attributes
    }
    
    /// Handle accessibility action
    pub fn on_accessibility_action(&mut self, action: crate::ui::events::AccessibilityAction) {
        // Route accessibility action to focused component
        if let Some(ref component_id) = self.accessibility_focused_component {
            match component_id.as_str() {
                "chat_input" => {
                    if let Some(ref mut chat) = self.chat_window {
                        // TODO: Implement Accessible trait for TextInput
                        match action {
                            crate::ui::events::AccessibilityAction::Activate => {
                                chat.input_field.on_focus();
                            }
                            _ => {}
                        }
                    }
                }
                "notepad_editor" => {
                    if let Some(ref mut notepad) = self.notepad_window {
                        match action {
                            crate::ui::events::AccessibilityAction::Activate => {
                                notepad.editor.focus();
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
    }
    
    /// Handle middle mouse button click
    pub fn on_mouse_middle_click(&mut self) {
        // TODO: Implement middle-click actions (e.g., paste, close tab)
    }
    
    /// Handle drag start
    pub fn on_mouse_drag_start(&mut self, position: Vec2, button: MouseButton) {
        self.drag_start_pos = position;
        self.drag_button = Some(button);
        self.is_dragging = true;
        self.drag_state = crate::ui::events::DragState::Starting {
            button: button.into(),
            start_pos: position,
        };
    }
    
    /// Handle drag end
    pub fn on_mouse_drag_end(&mut self, position: Vec2) {
        if let Some(_button) = self.drag_button {
            // Route drag end to appropriate component
            // For now, handle text selection in editors
            if let Some(ref mut notepad) = self.notepad_window {
                if self.ui_state.active_tab == Tab::Notepad && notepad.editor.contains(position) {
                    // on_mouse_up takes no args per TextEditor trait
                    use crate::ui::text_editor::TextEditor;
                    TextEditor::on_mouse_up(&mut notepad.editor);
                }
            }
        }

        // Clear constellation pan
        if let Some(ref mut chat) = self.chat_window {
            chat.constellation_view.pan_drag_start = None;
        }
        
        self.is_dragging = false;
        self.drag_button = None;
        self.drag_state = crate::ui::events::DragState::None;
    }

    pub fn on_cursor_moved(&mut self, pos: winit::dpi::PhysicalPosition<f64>) {
        let old_pos = self.mouse_pos;
        self.mouse_pos = Vec2::new(pos.x as f32, pos.y as f32);
        
        // Handle drag operations
        if self.is_dragging {
            // Update drag state
            if let crate::ui::events::DragState::Starting { button, start_pos } = self.drag_state {
                self.drag_state = crate::ui::events::DragState::Dragging {
                    button,
                    start_pos,
                };
            }
            // Route drag to appropriate component
            self.on_mouse_drag(self.mouse_pos);
        }
        
        // Handle window dragging
        if self.is_dragging_window {
            if let Some(ref proxy) = self.window_proxy {
                let _ = proxy.send_event(WindowControlEvent::DragWindow);
            }
        }

        // Update hover state (mouse enter/leave tracking)
        self.update_hover_state(old_pos);

        self.header.on_mouse_move(self.mouse_pos);
        
        // Collapse expanded handles when mouse leaves sidebar
        if !self.sidebar.hit_test(self.mouse_pos) {
            self.sidebar.conversations_list.expanded_index = None;
            self.sidebar.documents_list.expanded_index = None;
            self.sidebar.insights_panel.insights_list.expanded_index = None;
        }
        
        // Update sidebar edge glow effect
        let edge_hover_width_left = 20.0;  // Width of hover detection zone on left
        let edge_hover_width_right = 150.0; // Width of hover detection zone on right
        let header_height = self.header.size.y;
        
        let mut is_hovering = false;
        
        if self.sidebar.is_open {
            // Sidebar open: glow on RIGHT edge of sidebar (2x wider detection)
            let edge_x = self.sidebar.current_width;
            if self.mouse_pos.x >= edge_x - edge_hover_width_left && 
               self.mouse_pos.x <= edge_x + edge_hover_width_right &&
               self.mouse_pos.y >= header_height {
                self.sidebar_edge_glow_position = Some(self.mouse_pos.y);
                self.sidebar_edge_glow_target_intensity = 1.0;
                is_hovering = true;
            } else {
                self.sidebar_edge_glow_target_intensity = 0.0;
                if self.sidebar_edge_glow_intensity < 0.05 {
                    self.sidebar_edge_glow_position = None;
                }
            }
        } else {
            // Sidebar closed: glow on LEFT screen edge
            if self.mouse_pos.x <= edge_hover_width_left && self.mouse_pos.y >= header_height {
                self.sidebar_edge_glow_position = Some(self.mouse_pos.y);
                self.sidebar_edge_glow_target_intensity = 1.0;
                is_hovering = true;
            } else {
                self.sidebar_edge_glow_target_intensity = 0.0;
                if self.sidebar_edge_glow_intensity < 0.05 {
                    self.sidebar_edge_glow_position = None;
                }
            }
        }
        
        // Throttled debug output - only print when state changes
        let _intensity_changed = (self.sidebar_edge_glow_intensity - self.last_glow_debug_state.1).abs() > 0.05;
        let _hover_changed = is_hovering != self.last_glow_debug_state.0;
        

        
        // Update button and list item hover states
        self.update_button_hover_states();

        if let Some(id) = self.dragging_id {
            if id < self.windows.len() {
                let target = self.mouse_pos - self.drag_offset;
                self.windows[id].position = target;
            }
        }
        
        // Handle text selection and block dragging in notepad
        if let Some(ref mut notepad) = self.notepad_window {
            if self.ui_state.active_tab == Tab::Notepad {
                // Handle if selecting text or dragging a block
                if notepad.editor.is_selecting || notepad.editor.dragging_block_id.is_some() {
                    // Use TextEditor trait for consistency
                    use crate::ui::TextEditor;
                    TextEditor::on_mouse_move(&mut notepad.editor, self.mouse_pos);
                }
            }
        }
    }

    pub fn on_mouse_wheel(&mut self, delta: winit::event::MouseScrollDelta) {
        use winit::event::MouseScrollDelta;
        use crate::ui::tab_bar::Tab;

        let scroll_amount = match delta {
            MouseScrollDelta::LineDelta(_, y) => y * 20.0,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
        };

        // Handle scrolling for sidebar conversations and documents
        if self.sidebar.hit_test(self.mouse_pos) {
            if self.sidebar.conversations_list.contains(self.mouse_pos - self.sidebar.position) {
                self.sidebar.conversations_list.scroll(-scroll_amount);
                // Update highlight bar position after scrolling
                use crate::persistence::DocumentPersistence;
                let document_ids = DocumentPersistence::list_documents().unwrap_or_default();
                self.sidebar.update_hover_state(
                    self.mouse_pos,
                    &self.chat_state.conversations,
                    &document_ids,
                    &self.insights_state.insights,
                );
            } else if self.sidebar.documents_list.contains(self.mouse_pos - self.sidebar.position) {
                self.sidebar.documents_list.scroll(-scroll_amount);
                // Update highlight bar position after scrolling
                use crate::persistence::DocumentPersistence;
                let document_ids = DocumentPersistence::list_documents().unwrap_or_default();
                self.sidebar.update_hover_state(
                    self.mouse_pos,
                    &self.chat_state.conversations,
                    &document_ids,
                    &self.insights_state.insights,
                );
            } else if self.sidebar.insights_panel.insights_list.contains(self.mouse_pos) {
                self.sidebar.insights_panel.insights_list.scroll(-scroll_amount);
                use crate::persistence::DocumentPersistence;
                let document_ids = DocumentPersistence::list_documents().unwrap_or_default();
                self.sidebar.update_hover_state(
                    self.mouse_pos,
                    &self.chat_state.conversations,
                    &document_ids,
                    &self.insights_state.insights,
                );
            }
        }

        // Handle scrolling for Library tab
        if let Some(ref mut library) = self.library_window {
            if self.ui_state.active_tab == Tab::Library {
                // Check if mouse is over collections list
                if library.collections_list.contains(self.mouse_pos - library.position) {
                    library.collections_list.scroll(-scroll_amount);
                }
                // Check if mouse is over papers list
                else if library.papers_list.contains(self.mouse_pos - library.position) {
                    library.papers_list.scroll(-scroll_amount);
                }
            }
        }

        // Handle scrolling for Settings tab
        if let Some(ref mut settings) = self.settings_window {
            if self.ui_state.active_tab == Tab::Settings && settings.scroll_view.contains(self.mouse_pos - settings.position) {
                settings.scroll_view.scroll(-scroll_amount);
            }
        }

        // Handle scrolling for Chat tab
        if let Some(ref mut chat) = self.chat_window {
            if self.ui_state.active_tab == Tab::Chat {
                if self.graph_state.graph_id.is_some() && chat.constellation_view.contains_screen(self.mouse_pos) {
                    use crate::ui::chat_window::ChatHit;
                    let hit = chat.hit_test(self.mouse_pos, &self.graph_state);
                    let scale = chat.constellation_view.scale_animated;
                    const PAD: f32 = 8.0;
                    const ACTION_ROW: f32 = 28.0;
                    const BUBBLE_SPACING: f32 = 6.0;
                    const MSG_BUTTON_RESERVE: f32 = 22.0;
                    let visible_text_height = |node: &crate::state::ConstellationNode| {
                        let content_h = (node.size.y - PAD * 2.0 - ACTION_ROW - BUBBLE_SPACING).max(0.0) * 0.5;
                        (content_h - PAD * 2.0 - MSG_BUTTON_RESERVE).max(0.0)
                    };
                    match hit {
                        ChatHit::ConstellationUserBubbleContent(node_id) => {
                            let node = self.graph_state.get_node(&node_id).expect("node");
                            let visible = visible_text_height(node);
                            let overflow_screen = ((node.user_text_height - visible).max(0.0)) * scale;
                            if overflow_screen > 0.0 {
                                let mut targets = chat.constellation_scroll_targets.borrow_mut();
                                let (u, a) = targets.get(&node_id).copied().unwrap_or((0.0, 0.0));
                                let new_u = (u + scroll_amount * 20.0).clamp(0.0, overflow_screen);
                                targets.insert(node_id, (new_u, a));
                            } else {
                                let factor = if scroll_amount > 0.0 { 1.1 } else { 1.0 / 1.1 };
                                chat.constellation_view.zoom(factor);
                            }
                        }
                        ChatHit::ConstellationAssistantBubbleContent(node_id) => {
                            let node = self.graph_state.get_node(&node_id).expect("node");
                            let visible = visible_text_height(node);
                            let overflow_screen = ((node.assistant_text_height - visible).max(0.0)) * scale;
                            if overflow_screen > 0.0 {
                                let mut targets = chat.constellation_scroll_targets.borrow_mut();
                                let (u, a) = targets.get(&node_id).copied().unwrap_or((0.0, 0.0));
                                let new_a = (a + scroll_amount * 20.0).clamp(0.0, overflow_screen);
                                targets.insert(node_id, (u, new_a));
                            } else {
                                let factor = if scroll_amount > 0.0 { 1.1 } else { 1.0 / 1.1 };
                                chat.constellation_view.zoom(factor);
                            }
                        }
                        ChatHit::ConstellationNode(_) | _ => {
                            let factor = if scroll_amount > 0.0 { 1.1 } else { 1.0 / 1.1 };
                            chat.constellation_view.zoom(factor);
                        }
                    }
                    self.physics_idle_timer = 0.0;
                } else if chat.message_list.contains(self.mouse_pos - chat.position) {
                    chat.message_list.scroll(-scroll_amount);
                    chat.message_list.scroll_velocity = scroll_amount;
                }
            }
        }
    }

    pub fn on_keyboard(&mut self, event: &winit::event::KeyEvent) {
        if event.state == winit::event::ElementState::Pressed {
            // Check for shortcuts first
            use winit::keyboard::KeyCode;
            if let PhysicalKey::Code(key_code) = event.physical_key {
                if let Some(shortcut_id) = self.shortcut_registry.find(self.modifiers, key_code) {
                    self.on_shortcut_triggered(shortcut_id);
                    return; // Consume the key event
                }
            }
            
            // Handle special keys - winit 0.30 uses KeyCode
            if let PhysicalKey::Code(key_code) = event.physical_key {
                match key_code {
                    KeyCode::Tab => {
                        // Handle Tab/Shift+Tab for focus traversal
                        let shift_pressed = self.modifiers.contains(ModifiersState::SHIFT);
                        let direction = if shift_pressed {
                            crate::ui::events::FocusDirection::Backward
                        } else {
                            crate::ui::events::FocusDirection::Forward
                        };
                        self.focus_traverse(direction);
                        return; // Consume Tab key
                    }
                    KeyCode::Enter => {
                    // Handle insight modal save
                    if self.insight_modal.is_open && (self.insight_modal.is_editing_title || self.insight_modal.is_editing_text) {
                        if let Some(ref insight) = self.insight_modal.insight {
                            let insight_id = insight.id.clone();
                            let new_text = self.insight_modal.draft_text.clone();
                            let new_title = if self.insight_modal.draft_title.trim().is_empty() {
                                None
                            } else {
                                Some(self.insight_modal.draft_title.clone())
                            };
                            
                            if !new_text.trim().is_empty() {
                                self.insights_state.update_insight_text(&insight_id, new_text.clone());
                                if let Some(ref title) = new_title {
                                    self.insights_state.update_insight_title(&insight_id, title.clone());
                                }
                                
                                let base_url = self.api_client.base_url.clone();
                                let insight_id_clone = insight_id.clone();
                                let new_text_clone = new_text.clone();
                                let new_title_clone = new_title.clone();
                                tokio::spawn(async move {
                                    let api_client = crate::api::ApiClient::new(Some(base_url));
                                    if let Err(e) = api_client
                                        .update_shard(&insight_id_clone, Some(new_text_clone), None, new_title_clone, None)
                                        .await
                                    {
                                        eprintln!("Failed to update shard: {}", e);
                                    }
                                });
                            }
                            
                            self.insight_modal.is_editing_text = false;
                            self.insight_modal.is_editing_title = false;
                        }
                        return;
                    }
                    
                    // Handle message editing save
                    let mut send_note_sync_after: Option<(String, Vec<String>)> = None;
                    let mut send_messages_after: Option<Vec<crate::ui::chat_window::ChatMessage>> = None;
                    if let Some(ref mut chat) = self.chat_window {
                        if let Some(msg_idx) = chat.editing_message_idx {
                            let shard_id = chat.messages.get(msg_idx).and_then(|m| m.shard_id.clone());
                            if chat.save_edited_message() {
                                let new_content = chat.messages[msg_idx].content.clone();
                                let role = chat.messages[msg_idx].role.clone();
                                if let Some(ref sid) = shard_id {
                                    if let Some(node) = self.graph_state.get_node_mut(sid) {
                                        match role {
                                            crate::ui::chat_window::MessageRole::User => {
                                                node.shard.user_content =
                                                    Some(new_content.clone());
                                            }
                                            crate::ui::chat_window::MessageRole::Assistant => {
                                                node.shard.assistant_content =
                                                    Some(new_content.clone());
                                            }
                                        }
                                        self.graph_state.bump_content_version();
                                    }
                                }
                                self.chat_state.update_message(msg_idx, |msg| {
                                    msg.content = new_content;
                                });
                                self.save_chat_state();
                                self.show_success_toast("Message updated".to_string());
                                self.focused_input = None;
                            }
                            return;
                        }
                        // Handle note save (add or edit note)
                        if chat.adding_note_msg_idx.is_some() || chat.editing_note.is_some() {
                            let msg_idx = chat.adding_note_msg_idx.or(chat.editing_note.map(|(m, _)| m)).unwrap();
                            let shard_id = chat.messages.get(msg_idx).and_then(|m| m.shard_id.clone());
                            if chat.save_note() {
                                let msgs = chat.messages.clone();
                                send_messages_after = Some(msgs.clone());
                                send_note_sync_after = shard_id.as_ref().map(|sid| (sid.clone(), msgs[msg_idx].notes.clone()));
                                return;
                            }
                            return;
                        }
                    }
                    let had_note_save = send_messages_after.is_some();
                    if let Some(msgs) = send_messages_after.take() {
                        self.chat_state.set_current_messages(msgs);
                        self.save_chat_state();
                        self.focused_input = None;
                    }
                    if let Some((sid, notes)) = send_note_sync_after.take() {
                        if self.graph_state.graph_id.is_some() {
                            if let Some(node) = self.graph_state.get_node_mut(&sid) {
                                node.shard.notes = notes.clone();
                                self.graph_state.bump_content_version();
                            }
                            let base_url = self.api_client.base_url.clone();
                            tokio::spawn(async move {
                                let api_client = crate::api::ApiClient::new(Some(base_url));
                                let _ = api_client.update_shard(&sid, None, None, None, Some(notes)).await;
                            });
                        }
                    }
                    if had_note_save {
                        return;
                    }
                    
                    // Handle new collection creation
                    if let Some(ref mut library) = self.library_window {
                        if self.ui_state.active_tab == Tab::Library && library.is_creating_collection {
                            let collection_name = library.new_collection_input.text.trim().to_string();
                            if !collection_name.is_empty() {
                                let _base_url = self.api_client.base_url.clone();
                                let _client = self.api_client.client.clone();
                                let collection_name_clone = collection_name.clone();
                                let base_url_clone = self.api_client.base_url.clone();
                                let sender = self.collections_sender.clone();
                                tokio::spawn(async move {
                                    let api_client = crate::api::ApiClient::new(Some(base_url_clone.clone()));
                                    match api_client.create_collection(&collection_name_clone).await {
                                        Ok(_) => {
                                            // Reload collections after creation
                                            let client = reqwest::Client::new();
                                            let url = format!("{}/collections", base_url_clone);
                                            if let Ok(resp) = client.get(&url).send().await {
                                                if resp.status().is_success() {
                                                    if let Ok(collections) = resp.json::<Vec<Collection>>().await {
                                                        let _ = sender.send(Ok(collections));
                                                    }
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            eprintln!("Failed to create collection: {}", e);
                                            let _ = sender.send(Err(format!("Failed to create collection: {}", e)));
                                        }
                                    }
                                });
                                library.new_collection_input.text.clear();
                                library.is_creating_collection = false;
                                library.new_collection_input.on_blur();
                                self.focused_input = None;
                                self.show_success_toast(format!("Creating collection: {}", collection_name));
                            }
                            return;
                        }
                    }
                    
                    // Handle notepad modal - Enter to load selected note
                    let mut enter_toast_after = None::<(String, bool)>;
                    if let Some(ref mut notepad) = self.notepad_window {
                        if notepad.notepad_modal.is_open {
                            if let Some(index) = notepad.notepad_modal.selected_paper_index {
                                if index < notepad.notepad_modal.filtered_papers.len() {
                                    let doc_id = notepad.notepad_modal.filtered_papers.get(index).and_then(|p| p.filename.strip_suffix(".json").map(|s| s.to_string()));
                                    if let Some(ref doc_id) = doc_id {
                                        match notepad.load_note(doc_id) {
                                            Err(e) => enter_toast_after = Some((format!("Failed to load note: {}", e), false)),
                                            Ok(()) => {
                                                notepad.notepad_modal.close();
                                                enter_toast_after = Some(("Note loaded".to_string(), true));
                                            }
                                        }
                                    }
                                }
                            }
                            if enter_toast_after.is_some() {
                                // Defer toast until after we release the notepad borrow
                            } else {
                                return;
                            }
                        }
                    }
                    if let Some((msg, success)) = enter_toast_after {
                        if success { self.show_success_toast(msg); } else { self.show_error_toast(msg); }
                        return;
                    }
                    
                    // Handle conversation title save
                    if self.chat_info_dialog.is_open && self.chat_info_dialog.is_editing_title {
                        if let Some(conv_id) = &self.chat_info_dialog.conversation_id {
                            let new_title = self.chat_info_dialog.title_input.text.trim().to_string();
                            if !new_title.is_empty() {
                                // Update conversation title in state
                                if let Some(conv) = self.chat_state.conversations.iter_mut().find(|c| c.id == *conv_id) {
                                    conv.title = new_title.clone();
                                }
                                
                                // Save to disk
                                use crate::persistence::ConversationPersistence;
                                if let Some(conv) = self.chat_state.conversations.iter().find(|c| c.id == *conv_id) {
                                    if let Err(e) = ConversationPersistence::save_conversation(conv) {
                                        eprintln!("Failed to save conversation: {}", e);
                                        self.show_error_toast(format!("Failed to save: {}", e));
                                    } else {
                                        self.show_success_toast("Conversation title saved".to_string());
                                    }
                                }
                                
                                self.chat_info_dialog.draft_title = new_title;
                                self.chat_info_dialog.is_editing_title = false;
                                self.chat_info_dialog.title_input.on_blur();
                                self.focused_input = None;
                            }
                        }
                        return;
                    }
                    
                    let enter_send_provider = self.settings_state.provider.clone();
                    let enter_send_model_id = self.settings_state.model_id_for_send();
                    let enter_send_openai_model = self.settings_state.openai_model_for_send();
                    let mut enter_send_pending: Option<(String, crate::api::models::GraphSendRequest, String)> = None;
                    if let Some(ref mut chat) = self.chat_window {
                        if self.ui_state.active_tab == Tab::Chat && chat.input_field.focused {
                            if let Some(text) = chat.send_message() {
                                if self.graph_state.graph_id.is_some() {
                                    let graph_id = self.graph_state.graph_id.clone().unwrap();
                                    let leaf_id = self.graph_state.current_leaf_id.clone().unwrap_or_default();
                                    let request = crate::api::models::GraphSendRequest {
                                        current_leaf_id: leaf_id,
                                        user_draft: text.clone(),
                                        provider: enter_send_provider.clone(),
                                        model_id: enter_send_model_id.clone(),
                                        openai_model: enter_send_openai_model.clone(),
                                        temperature: None,
                                        max_tokens: None,
                                        model_token_limit: None,
                                    };
                                    let user_msg = crate::ui::chat_window::ChatMessage::from_legacy(
                                        crate::ui::chat_window::MessageRole::User,
                                        text.clone(),
                                        Vec::new(),
                                        Vec::new(),
                                    );
                                    chat.add_message(user_msg.clone());
                                    self.chat_state.add_message_to_current(user_msg);
                                    enter_send_pending = Some((graph_id, request, text));
                                } else {
                                    self.show_error_toast("Conversation not ready. Please wait for it to load.".to_string());
                                }
                            }
                        }
                    }
                    if let Some((graph_id, request, text)) = enter_send_pending {
                        self.save_chat_state();
                        let client = self.api_client.client.clone();
                        let base_url = self.api_client.base_url.clone();
                        let sender = self.graph_send_sender.clone();
                        tokio::spawn(async move {
                            let url = format!("{}/graph/{}/send", base_url, graph_id);
                            match client.post(&url).json(&request).send().await {
                                Ok(r) if r.status().is_success() => {
                                    match r.json::<crate::api::models::GraphSendResponse>().await {
                                        Ok(body) => { let _ = sender.send(Ok((text, body))); }
                                        Err(e) => { let _ = sender.send(Err(format!("Parse: {:?}", e))); }
                                    }
                                }
                                Ok(r) => {
                                    let err = r.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                                    let _ = sender.send(Err(err));
                                }
                                Err(e) => { let _ = sender.send(Err(format!("Request: {:?}", e))); }
                            }
                        });
                        self.is_sending_message = true;
                        if let Some(ref mut chat) = self.chat_window {
                            chat.is_sending = true;
                        }
                    }
                }
                KeyCode::KeyN => {
                    let ctrl_pressed = self.modifiers.contains(ModifiersState::CONTROL);
                    
                    // Ctrl+N: Create new conversation and autofocus chat input
                    if ctrl_pressed {
                        let conv_id = self.chat_state.create_conversation();
                        self.sidebar.selected_conversation_id = Some(conv_id.clone());
                        self.chat_state.current_conversation_id = Some(conv_id);
                        self.request_graph_for_new_conversation();
                        self.save_chat_state();
                        
                        // Autofocus the chat input
                        if let Some(ref mut chat) = self.chat_window {
                            chat.input_field.on_focus();
                            self.focused_input = Some(0);
                        }
                    }
                }
                KeyCode::Escape => {
                    // Handle escape key for all modals
                    if self.shard_modal.is_open {
                        self.shard_modal.close();
                        self.focused_input = None;
                        return;
                    }
                    if self.insight_modal.is_open {
                        if self.insight_modal.is_editing_title {
                            if let Some(ref insight) = self.insight_modal.insight {
                                self.insight_modal.draft_title = insight.title.clone();
                            }
                            self.insight_modal.is_editing_title = false;
                            self.insight_modal.title_input.on_blur();
                            self.focused_input = None;
                        } else if self.insight_modal.is_editing_text {
                            if let Some(ref insight) = self.insight_modal.insight {
                                self.insight_modal.draft_text = insight.text.clone();
                            }
                            self.insight_modal.is_editing_text = false;
                            self.insight_modal.text_input.on_blur();
                            self.focused_input = None;
                        } else {
                            // Close modal
                            self.insight_modal.close();
                            self.insights_state.set_modal_insight(None);
                        }
                        return;
                    }
                    
                    if self.pdf_modal.is_open {
                        self.pdf_modal.close();
                        return;
                    }
                    
                    // Handle message editing cancel
                    if let Some(ref mut chat) = self.chat_window {
                        if chat.editing_note.is_some() || chat.adding_note_msg_idx.is_some() {
                            chat.cancel_note();
                            self.focused_input = None;
                            return;
                        }
                        if chat.editing_message_idx.is_some() {
                            chat.cancel_editing_message();
                            self.focused_input = None;
                            return;
                        }
                    }
                    
                    if self.chat_info_dialog.is_open {
                        if self.chat_info_dialog.is_editing_title {
                            if let Some(conv_id) = &self.chat_info_dialog.conversation_id {
                                if let Some(conv) = self.chat_state.conversations.iter().find(|c| c.id == *conv_id) {
                                    self.chat_info_dialog.draft_title = conv.title.clone();
                                }
                            }
                            self.chat_info_dialog.is_editing_title = false;
                            self.chat_info_dialog.title_input.on_blur();
                            self.focused_input = None;
                        } else {
                            self.chat_info_dialog.close();
                        }
                        return;
                    }
                }
                    KeyCode::Backspace => {
                    let ctrl_pressed = self.modifiers.contains(ModifiersState::CONTROL);
                    
                    // Special handling for chat input (undo history)
                    if self.focused_input == Some(0) {
                        if let Some(ref mut chat) = self.chat_window {
                            let current_text = chat.input_field.text.clone();
                            // Save state for undo before deletion
                            self.undo_history.push(current_text);
                            self.redo_history.clear();
                            if self.undo_history.len() > 50 {
                                self.undo_history.remove(0);
                            }
                        }
                    }
                    
                    // Use router for all backspace operations (router updates cursor animation)
                    if ctrl_pressed {
                        self.route_to_focused_editor(|editor| {
                            editor.on_backspace_word();
                        });
                    } else {
                        self.route_to_focused_editor(|editor| {
                            editor.on_backspace();
                        });
                    }
                    
                    // Post-processing after backspace
                    match self.focused_input {
                        Some(3) | Some(4) | Some(6) => {
                            // Auto-save settings
                            use crate::persistence::SettingsPersistence;
                            if let Err(e) = SettingsPersistence::save_settings(&self.settings_state) {
                                eprintln!("Failed to save settings: {}", e);
                            }
                        }
                        _ => {}
                    }
                }
                    KeyCode::ArrowLeft => {
                    let shift_pressed = self.modifiers.contains(ModifiersState::SHIFT);
                    let ctrl_pressed = self.modifiers.contains(ModifiersState::CONTROL);
                    let alt_pressed = self.modifiers.contains(ModifiersState::ALT);

                    // Alt+Arrow: traverse graph (focus hop + center camera)
                    if alt_pressed && self.ui_state.active_tab == Tab::Chat && self.graph_state.graph_id.is_some() && self.focused_input.is_none() {
                        if let Some(ref leaf) = self.graph_state.current_leaf_id {
                            let parents = self.graph_state.parent_ids(leaf);
                            if let Some(first_parent) = parents.first() {
                                self.graph_state.current_leaf_id = Some(first_parent.clone());
                                if let Some(chat) = self.chat_window.as_mut() {
                                    if let Some(node) = self.graph_state.get_node(first_parent) {
                                        let center = node.position + node.size * 0.5;
                                        chat.constellation_view.center_camera_on(center);
                                        chat.constellation_view.reset_zoom_to_normal();
                                    }
                                }
                            }
                        }
                        return;
                    }
                    
                    // Ctrl+Left: Close sidebar (if no input is focused)
                    if ctrl_pressed && self.focused_input.is_none() && !shift_pressed {
                        if self.sidebar.is_open {
                            self.sidebar.toggle();
                            self.ui_state.sidebar_open = self.sidebar.is_open;
                            self.bump_layout_generation();
                        }
                        return;
                    }
                    
                    // Use router for arrow key navigation
                    self.route_to_focused_editor(|editor| {
                        editor.on_arrow_left(shift_pressed, ctrl_pressed);
                    });
                }
                    KeyCode::ArrowRight => {
                    let shift_pressed = self.modifiers.contains(ModifiersState::SHIFT);
                    let ctrl_pressed = self.modifiers.contains(ModifiersState::CONTROL);
                    let alt_pressed = self.modifiers.contains(ModifiersState::ALT);

                    // Alt+Arrow: traverse graph
                    if alt_pressed && self.ui_state.active_tab == Tab::Chat && self.graph_state.graph_id.is_some() && self.focused_input.is_none() {
                        if let Some(ref leaf) = self.graph_state.current_leaf_id {
                            let children = self.graph_state.children_ids(leaf);
                            if let Some(first_child) = children.first() {
                                self.graph_state.current_leaf_id = Some(first_child.clone());
                                if let Some(chat) = self.chat_window.as_mut() {
                                    if let Some(node) = self.graph_state.get_node(first_child) {
                                        let center = node.position + node.size * 0.5;
                                        chat.constellation_view.center_camera_on(center);
                                        chat.constellation_view.reset_zoom_to_normal();
                                    }
                                }
                            }
                        }
                        return;
                    }
                    
                    // Ctrl+Right: Open sidebar (if no input is focused)
                    if ctrl_pressed && self.focused_input.is_none() && !shift_pressed {
                        if !self.sidebar.is_open {
                            self.sidebar.toggle();
                            self.ui_state.sidebar_open = self.sidebar.is_open;
                            self.bump_layout_generation();
                        }
                        return;
                    }
                    
                    // Use router for arrow key navigation
                    self.route_to_focused_editor(|editor| {
                        editor.on_arrow_right(shift_pressed, ctrl_pressed);
                    });
                }
                    KeyCode::Home => {
                    let shift_pressed = self.modifiers.contains(ModifiersState::SHIFT);
                    let alt_pressed = self.modifiers.contains(ModifiersState::ALT);
                    // Alt+Home: center camera on active node (constellation)
                    if alt_pressed && self.ui_state.active_tab == Tab::Chat && self.graph_state.graph_id.is_some() {
                        if let Some(ref leaf) = self.graph_state.current_leaf_id {
                            if let Some(chat) = self.chat_window.as_mut() {
                                if let Some(node) = self.graph_state.get_node(leaf) {
                                    let center = node.position + node.size * 0.5;
                                    chat.constellation_view.center_camera_on(center);
                                    chat.constellation_view.reset_zoom_to_normal();
                                }
                            }
                        }
                        return;
                    }
                    // Use router for Home key
                    self.route_to_focused_editor(|editor| {
                        editor.on_home(shift_pressed);
                    });
                }
                    KeyCode::KeyF => {
                    let alt_pressed = self.modifiers.contains(ModifiersState::ALT);
                    // Alt+F: fit graph in view (constellation)
                    if alt_pressed && self.ui_state.active_tab == Tab::Chat && self.graph_state.graph_id.is_some() {
                        if let Some(chat) = self.chat_window.as_mut() {
                            if let Some((min, max)) = self.graph_state.compute_bbox() {
                                chat.constellation_view.fit_in_view(min, max, 40.0);
                            }
                        }
                        return;
                    }
                }
                    KeyCode::End => {
                    let shift_pressed = self.modifiers.contains(ModifiersState::SHIFT);
                    // Use router for End key
                    self.route_to_focused_editor(|editor| {
                        editor.on_end(shift_pressed);
                    });
                }
                KeyCode::KeyA => {
                    // Ctrl+A for select all
                    if self.modifiers.contains(ModifiersState::CONTROL) {
                        match self.focused_input {
                            Some(0) => {
                                if let Some(ref mut chat) = self.chat_window {
                                    chat.input_field.select_all();
                                }
                            }
                            Some(1) => {
                                if let Some(ref mut library) = self.library_window {
                                    library.search_input.select_all();
                                }
                            }
                            Some(2) => {
                                if let Some(ref mut ingest) = self.ingest_window {
                                    ingest.pdf_dir_input.select_all();
                                }
                            }
                            Some(6) => {
                                if self.insight_modal.is_editing_title {
                                    self.insight_modal.title_input.select_all();
                                }
                            }
                            Some(7) => {
                                if self.insight_modal.is_editing_text {
                                    self.insight_modal.text_input.select_all();
                                }
                            }
                            Some(8) => {
                                if self.chat_info_dialog.is_editing_title {
                                    self.chat_info_dialog.title_input.select_all();
                                }
                            }
                            Some(5) => {
                                // Notepad - select all
                                if let Some(ref mut notepad) = self.notepad_window {
                                    notepad.editor.select_all();
                                }
                            }
                            _ => {}
                        }
                    }
                }
                KeyCode::KeyZ => {
                    // Ctrl+Z for undo
                    if self.modifiers.contains(ModifiersState::CONTROL) {
                        match self.focused_input {
                            Some(0) => {
                                if let Some(ref mut chat) = self.chat_window {
                                    let current_text = chat.input_field.text.clone();
                                    if let Some(previous_text) = self.undo_history.pop() {
                                        self.redo_history.push(current_text);
                                        chat.input_field.text = previous_text;
                                        chat.input_field.cursor_position = chat.input_field.text.chars().count();
                                        chat.input_field.clear_selection();
                                        chat.input_field.ensure_cursor_valid();
                                    }
                                }
                            }
                            Some(1) => {
                                if let Some(ref mut library) = self.library_window {
                                    let current_text = library.search_input.text.clone();
                                    if let Some(previous_text) = self.undo_history.pop() {
                                        self.redo_history.push(current_text);
                                        library.search_input.text = previous_text;
                                        library.search_input.cursor_position = library.search_input.text.chars().count();
                                        library.search_input.clear_selection();
                                        library.search_input.ensure_cursor_valid();
                                    }
                                }
                            }
                            Some(2) => {
                                if let Some(ref mut ingest) = self.ingest_window {
                                    let current_text = ingest.pdf_dir_input.text.clone();
                                    if let Some(previous_text) = self.undo_history.pop() {
                                        self.redo_history.push(current_text);
                                        ingest.pdf_dir_input.text = previous_text;
                                        ingest.pdf_dir_input.cursor_position = ingest.pdf_dir_input.text.chars().count();
                                        ingest.pdf_dir_input.clear_selection();
                                        ingest.pdf_dir_input.ensure_cursor_valid();
                                    }
                                }
                            }
                            Some(6) => {
                                if self.insight_modal.is_editing_title {
                                    let current_text = self.insight_modal.title_input.text.clone();
                                    if let Some(previous_text) = self.undo_history.pop() {
                                        self.redo_history.push(current_text);
                                        self.insight_modal.title_input.text = previous_text;
                                        self.insight_modal.title_input.cursor_position = self.insight_modal.title_input.text.chars().count();
                                        self.insight_modal.title_input.clear_selection();
                                        self.insight_modal.title_input.ensure_cursor_valid();
                                        self.insight_modal.draft_title = self.insight_modal.title_input.text.clone();
                                    }
                                }
                            }
                            Some(7) => {
                                if self.insight_modal.is_editing_text {
                                    let current_text = self.insight_modal.text_input.text.clone();
                                    if let Some(previous_text) = self.undo_history.pop() {
                                        self.redo_history.push(current_text);
                                        self.insight_modal.text_input.text = previous_text;
                                        self.insight_modal.text_input.cursor_position = self.insight_modal.text_input.text.chars().count();
                                        self.insight_modal.text_input.clear_selection();
                                        self.insight_modal.text_input.ensure_cursor_valid();
                                        self.insight_modal.draft_text = self.insight_modal.text_input.text.clone();
                                    }
                                }
                            }
                            Some(8) => {
                                if self.chat_info_dialog.is_editing_title {
                                    let current_text = self.chat_info_dialog.title_input.text.clone();
                                    if let Some(previous_text) = self.undo_history.pop() {
                                        self.redo_history.push(current_text);
                                        self.chat_info_dialog.title_input.text = previous_text;
                                        self.chat_info_dialog.title_input.cursor_position = self.chat_info_dialog.title_input.text.chars().count();
                                        self.chat_info_dialog.title_input.clear_selection();
                                        self.chat_info_dialog.title_input.ensure_cursor_valid();
                                        self.chat_info_dialog.draft_title = self.chat_info_dialog.title_input.text.clone();
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                KeyCode::KeyY => {
                    // Ctrl+Y for redo
                    if self.modifiers.contains(ModifiersState::CONTROL) {
                        match self.focused_input {
                            Some(0) => {
                                if let Some(ref mut chat) = self.chat_window {
                                    let current_text = chat.input_field.text.clone();
                                    if let Some(next_text) = self.redo_history.pop() {
                                        self.undo_history.push(current_text);
                                        chat.input_field.text = next_text;
                                        chat.input_field.cursor_position = chat.input_field.text.chars().count();
                                        chat.input_field.clear_selection();
                                        chat.input_field.ensure_cursor_valid();
                                    }
                                }
                            }
                            Some(1) => {
                                if let Some(ref mut library) = self.library_window {
                                    let current_text = library.search_input.text.clone();
                                    if let Some(next_text) = self.redo_history.pop() {
                                        self.undo_history.push(current_text);
                                        library.search_input.text = next_text;
                                        library.search_input.cursor_position = library.search_input.text.chars().count();
                                        library.search_input.clear_selection();
                                        library.search_input.ensure_cursor_valid();
                                    }
                                }
                            }
                            Some(2) => {
                                if let Some(ref mut ingest) = self.ingest_window {
                                    let current_text = ingest.pdf_dir_input.text.clone();
                                    if let Some(next_text) = self.redo_history.pop() {
                                        self.undo_history.push(current_text);
                                        ingest.pdf_dir_input.text = next_text;
                                        ingest.pdf_dir_input.cursor_position = ingest.pdf_dir_input.text.chars().count();
                                        ingest.pdf_dir_input.clear_selection();
                                        ingest.pdf_dir_input.ensure_cursor_valid();
                                    }
                                }
                            }
                            Some(6) => {
                                if self.insight_modal.is_editing_title {
                                    let current_text = self.insight_modal.title_input.text.clone();
                                    if let Some(next_text) = self.redo_history.pop() {
                                        self.undo_history.push(current_text);
                                        self.insight_modal.title_input.text = next_text;
                                        self.insight_modal.title_input.cursor_position = self.insight_modal.title_input.text.chars().count();
                                        self.insight_modal.title_input.clear_selection();
                                        self.insight_modal.title_input.ensure_cursor_valid();
                                        self.insight_modal.draft_title = self.insight_modal.title_input.text.clone();
                                    }
                                }
                            }
                            Some(7) => {
                                if self.insight_modal.is_editing_text {
                                    let current_text = self.insight_modal.text_input.text.clone();
                                    if let Some(next_text) = self.redo_history.pop() {
                                        self.undo_history.push(current_text);
                                        self.insight_modal.text_input.text = next_text;
                                        self.insight_modal.text_input.cursor_position = self.insight_modal.text_input.text.chars().count();
                                        self.insight_modal.text_input.clear_selection();
                                        self.insight_modal.text_input.ensure_cursor_valid();
                                        self.insight_modal.draft_text = self.insight_modal.text_input.text.clone();
                                    }
                                }
                            }
                            Some(8) => {
                                if self.chat_info_dialog.is_editing_title {
                                    let current_text = self.chat_info_dialog.title_input.text.clone();
                                    if let Some(next_text) = self.redo_history.pop() {
                                        self.undo_history.push(current_text);
                                        self.chat_info_dialog.title_input.text = next_text;
                                        self.chat_info_dialog.title_input.cursor_position = self.chat_info_dialog.title_input.text.chars().count();
                                        self.chat_info_dialog.title_input.clear_selection();
                                        self.chat_info_dialog.title_input.ensure_cursor_valid();
                                        self.chat_info_dialog.draft_title = self.chat_info_dialog.title_input.text.clone();
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                KeyCode::KeyX => {
                    // Ctrl+X for cut
                    if self.modifiers.contains(ModifiersState::CONTROL) {
                        match self.focused_input {
                            Some(0) => {
                                if let Some(ref mut chat) = self.chat_window {
                                    let selected = chat.input_field.get_selected_text();
                                    let current_text = chat.input_field.text.clone();
                                    if !selected.is_empty() {
                                        self.clipboard_text = selected;
                                        chat.input_field.delete_selection();
                                        // Save state after deletion
                                        self.undo_history.push(current_text);
                                        self.redo_history.clear();
                                        if self.undo_history.len() > 50 {
                                            self.undo_history.remove(0);
                                        }
                                    }
                                }
                            }
                            Some(1) => {
                                if let Some(ref mut library) = self.library_window {
                                    let selected = library.search_input.get_selected_text();
                                    let current_text = library.search_input.text.clone();
                                    if !selected.is_empty() {
                                        self.clipboard_text = selected;
                                        library.search_input.delete_selection();
                                        // Save state for undo
                                        self.undo_history.push(current_text);
                                        self.redo_history.clear();
                                        if self.undo_history.len() > 50 {
                                            self.undo_history.remove(0);
                                        }
                                    }
                                }
                            }
                            Some(2) => {
                                if let Some(ref mut ingest) = self.ingest_window {
                                    let selected = ingest.pdf_dir_input.get_selected_text();
                                    let current_text = ingest.pdf_dir_input.text.clone();
                                    if !selected.is_empty() {
                                        self.clipboard_text = selected;
                                        ingest.pdf_dir_input.delete_selection();
                                        // Save state for undo
                                        self.undo_history.push(current_text);
                                        self.redo_history.clear();
                                        if self.undo_history.len() > 50 {
                                            self.undo_history.remove(0);
                                        }
                                    }
                                }
                            }
                            Some(6) => {
                                if self.insight_modal.is_editing_title {
                                    let selected = self.insight_modal.title_input.get_selected_text();
                                    let current_text = self.insight_modal.title_input.text.clone();
                                    if !selected.is_empty() {
                                        self.clipboard_text = selected;
                                        self.insight_modal.title_input.delete_selection();
                                        // Save state for undo
                                        self.undo_history.push(current_text);
                                        self.redo_history.clear();
                                        if self.undo_history.len() > 50 {
                                            self.undo_history.remove(0);
                                        }
                                        self.insight_modal.draft_title = self.insight_modal.title_input.text.clone();
                                    }
                                }
                            }
                            Some(7) => {
                                if self.insight_modal.is_editing_text {
                                    let selected = self.insight_modal.text_input.get_selected_text();
                                    let current_text = self.insight_modal.text_input.text.clone();
                                    if !selected.is_empty() {
                                        self.clipboard_text = selected;
                                        self.insight_modal.text_input.delete_selection();
                                        // Save state for undo
                                        self.undo_history.push(current_text);
                                        self.redo_history.clear();
                                        if self.undo_history.len() > 50 {
                                            self.undo_history.remove(0);
                                        }
                                        self.insight_modal.draft_text = self.insight_modal.text_input.text.clone();
                                    }
                                }
                            }
                            Some(8) => {
                                if self.chat_info_dialog.is_editing_title {
                                    let selected = self.chat_info_dialog.title_input.get_selected_text();
                                    let current_text = self.chat_info_dialog.title_input.text.clone();
                                    if !selected.is_empty() {
                                        self.clipboard_text = selected;
                                        self.chat_info_dialog.title_input.delete_selection();
                                        // Save state for undo
                                        self.undo_history.push(current_text);
                                        self.redo_history.clear();
                                        if self.undo_history.len() > 50 {
                                            self.undo_history.remove(0);
                                        }
                                        self.chat_info_dialog.draft_title = self.chat_info_dialog.title_input.text.clone();
                                    }
                                }
                            }
                            Some(5) => {
                                // Notepad - cut
                                if let Some(ref mut notepad) = self.notepad_window {
                                    let selected = notepad.editor.get_selected_text();
                                    if !selected.is_empty() {
                                        self.clipboard_text = selected;
                                        // Delete selection
                                        if let Some(ref selection) = notepad.editor.selection {
                                            if let Some(ref cursor) = notepad.editor.cursor {
                                                if selection.start.block_id == cursor.block_id && selection.end.block_id == cursor.block_id {
                                                    let start = selection.start.position.min(selection.end.position);
                                                    let end = selection.start.position.max(selection.end.position);
                                                    
                                                    if let Some(block) = notepad.editor.document.get_block_mut(&cursor.block_id) {
                                                        if let Some(block_text) = block.content.get_text_mut() {
                                                            if end <= block_text.len() {
                                                                block_text.drain(start..end);
                                                                // Remove format spans that are now invalid
                                                                use crate::stylus::block::BlockContent;
                                                                if let BlockContent::Text { ref mut formats, .. } = block.content {
                                                                    formats.retain(|span| {
                                                                        !(span.start >= start && span.end <= end) && 
                                                                        !(span.start < end && span.end > start)
                                                                    });
                                                                    // Adjust format span positions
                                                                    for span in formats.iter_mut() {
                                                                        if span.start > end {
                                                                            span.start -= end - start;
                                                                            span.end -= end - start;
                                                                        } else if span.start > start {
                                                                            span.start = start;
                                                                            if span.end > end {
                                                                                span.end = start;
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                                if let Some(ref mut c) = notepad.editor.cursor {
                                                                    c.position = start;
                                                                }
                                                                notepad.editor.selection = None;
                                                                notepad.editor.mark_changed();
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                KeyCode::KeyC => {
                    // Ctrl+C for copy
                    if self.modifiers.contains(ModifiersState::CONTROL) {
                        match self.focused_input {
                            Some(0) => {
                                if let Some(ref chat) = self.chat_window {
                                    let selected = chat.input_field.get_selected_text();
                                    if !selected.is_empty() {
                                        self.clipboard_text = selected;
                                    }
                                }
                            }
                            Some(1) => {
                                if let Some(ref library) = self.library_window {
                                    let selected = library.search_input.get_selected_text();
                                    if !selected.is_empty() {
                                        self.clipboard_text = selected;
                                    }
                                }
                            }
                            Some(2) => {
                                if let Some(ref ingest) = self.ingest_window {
                                    let selected = ingest.pdf_dir_input.get_selected_text();
                                    if !selected.is_empty() {
                                        self.clipboard_text = selected;
                                    }
                                }
                            }
                            Some(6) => {
                                if self.insight_modal.is_editing_title {
                                    let selected = self.insight_modal.title_input.get_selected_text();
                                    if !selected.is_empty() {
                                        self.clipboard_text = selected;
                                    }
                                }
                            }
                            Some(7) => {
                                if self.insight_modal.is_editing_text {
                                    let selected = self.insight_modal.text_input.get_selected_text();
                                    if !selected.is_empty() {
                                        self.clipboard_text = selected;
                                    }
                                }
                            }
                            Some(8) => {
                                if self.chat_info_dialog.is_editing_title {
                                    let selected = self.chat_info_dialog.title_input.get_selected_text();
                                    if !selected.is_empty() {
                                        self.clipboard_text = selected;
                                    }
                                }
                            }
                            Some(5) => {
                                // Notepad - copy
                                if let Some(ref notepad) = self.notepad_window {
                                    let selected = notepad.editor.get_selected_text();
                                    if !selected.is_empty() {
                                        self.clipboard_text = selected;
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                KeyCode::KeyV => {
                    // Ctrl+V for paste
                    if self.modifiers.contains(ModifiersState::CONTROL) {
                        match self.focused_input {
                            Some(0) => {
                                if let Some(ref mut chat) = self.chat_window {
                                    let clipboard = self.clipboard_text.clone();
                                    let current_text = chat.input_field.text.clone();
                                    if !clipboard.is_empty() {
                                        chat.input_field.paste(&clipboard);
                                        // Save state after paste
                                        self.undo_history.push(current_text);
                                        self.redo_history.clear();
                                        if self.undo_history.len() > 50 {
                                            self.undo_history.remove(0);
                                        }
                                        self.cursor_target_position = chat.input_field.cursor_position;
                                        self.cursor_blink_timer = 0.0;
                                        self.cursor_visible = true;
                                    }
                                }
                            }
                            Some(1) => {
                                if let Some(ref mut library) = self.library_window {
                                    let clipboard = self.clipboard_text.clone();
                                    let current_text = library.search_input.text.clone();
                                    if !clipboard.is_empty() {
                                        library.search_input.paste(&clipboard);
                                        // Save state for undo
                                        self.undo_history.push(current_text);
                                        self.redo_history.clear();
                                        if self.undo_history.len() > 50 {
                                            self.undo_history.remove(0);
                                        }
                                    }
                                }
                            }
                            Some(2) => {
                                if let Some(ref mut ingest) = self.ingest_window {
                                    let clipboard = self.clipboard_text.clone();
                                    let current_text = ingest.pdf_dir_input.text.clone();
                                    if !clipboard.is_empty() {
                                        ingest.pdf_dir_input.paste(&clipboard);
                                        // Save state for undo
                                        self.undo_history.push(current_text);
                                        self.redo_history.clear();
                                        if self.undo_history.len() > 50 {
                                            self.undo_history.remove(0);
                                        }
                                    }
                                }
                            }
                            Some(6) => {
                                if self.insight_modal.is_editing_title {
                                    let clipboard = self.clipboard_text.clone();
                                    let current_text = self.insight_modal.title_input.text.clone();
                                    if !clipboard.is_empty() {
                                        self.insight_modal.title_input.paste(&clipboard);
                                        // Save state for undo
                                        self.undo_history.push(current_text);
                                        self.redo_history.clear();
                                        if self.undo_history.len() > 50 {
                                            self.undo_history.remove(0);
                                        }
                                        self.insight_modal.draft_title = self.insight_modal.title_input.text.clone();
                                    }
                                }
                            }
                            Some(7) => {
                                if self.insight_modal.is_editing_text {
                                    let clipboard = self.clipboard_text.clone();
                                    let current_text = self.insight_modal.text_input.text.clone();
                                    if !clipboard.is_empty() {
                                        self.insight_modal.text_input.paste(&clipboard);
                                        // Save state for undo
                                        self.undo_history.push(current_text);
                                        self.redo_history.clear();
                                        if self.undo_history.len() > 50 {
                                            self.undo_history.remove(0);
                                        }
                                        self.insight_modal.draft_text = self.insight_modal.text_input.text.clone();
                                    }
                                }
                            }
                            Some(8) => {
                                if self.chat_info_dialog.is_editing_title {
                                    let clipboard = self.clipboard_text.clone();
                                    let current_text = self.chat_info_dialog.title_input.text.clone();
                                    if !clipboard.is_empty() {
                                        self.chat_info_dialog.title_input.paste(&clipboard);
                                        // Save state for undo
                                        self.undo_history.push(current_text);
                                        self.redo_history.clear();
                                        if self.undo_history.len() > 50 {
                                            self.undo_history.remove(0);
                                        }
                                        self.chat_info_dialog.draft_title = self.chat_info_dialog.title_input.text.clone();
                                    }
                                }
                            }
                            Some(5) => {
                                // Notepad - paste
                                if let Some(ref mut notepad) = self.notepad_window {
                                    let clipboard = self.clipboard_text.clone();
                                    if !clipboard.is_empty() {
                                        notepad.editor.paste(&clipboard);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                KeyCode::KeyB => {
                    // Ctrl+B for bold (only when notepad editor is focused)
                    if self.modifiers.contains(ModifiersState::CONTROL) && self.focused_input == Some(5) {
                        if let Some(ref mut notepad) = self.notepad_window {
                            use crate::stylus::formatting::TextFormat;
                            notepad.editor.apply_format(TextFormat::Bold);
                        }
                        return; // Consume the key event
                    }
                }
                KeyCode::KeyI => {
                    // Ctrl+I for italic (only when notepad editor is focused)
                    if self.modifiers.contains(ModifiersState::CONTROL) && self.focused_input == Some(5) {
                        if let Some(ref mut notepad) = self.notepad_window {
                            use crate::stylus::formatting::TextFormat;
                            notepad.editor.apply_format(TextFormat::Italic);
                        }
                        return; // Consume the key event
                    }
                }
                KeyCode::KeyU => {
                    // Ctrl+U for underline (only when notepad editor is focused)
                    if self.modifiers.contains(ModifiersState::CONTROL) && self.focused_input == Some(5) {
                        if let Some(ref mut notepad) = self.notepad_window {
                            use crate::stylus::formatting::TextFormat;
                            notepad.editor.apply_format(TextFormat::Underline);
                        }
                        return; // Consume the key event
                    }
                }
                KeyCode::Delete => {
                    // Handle Delete key (or Ctrl+Delete for smaller keyboards)
                    let _ctrl_pressed = self.modifiers.contains(ModifiersState::CONTROL);
                    // If Ctrl is pressed with Delete, treat it the same as Delete
                    // This helps users with smaller keyboards that might not have a dedicated Delete key
                    
                    // Special handling for settings (auto-save)
                    if self.focused_input == Some(3) || self.focused_input == Some(4) || self.focused_input == Some(6) {
                        self.route_to_focused_editor(|editor| {
                            editor.on_delete();
                        });
                        // Auto-save settings
                        use crate::persistence::SettingsPersistence;
                        if let Err(e) = SettingsPersistence::save_settings(&self.settings_state) {
                            eprintln!("Failed to save settings: {}", e);
                        }
                        return;
                    }
                    
                    // Use router for Delete key
                    self.route_to_focused_editor(|editor| {
                        editor.on_delete();
                    });
                }
                KeyCode::ArrowUp => {
                    let shift_pressed = self.modifiers.contains(ModifiersState::SHIFT);
                    let alt_pressed = self.modifiers.contains(ModifiersState::ALT);
                    if alt_pressed && self.ui_state.active_tab == Tab::Chat && self.graph_state.graph_id.is_some() && self.focused_input.is_none() {
                        if let Some(ref leaf) = self.graph_state.current_leaf_id {
                            let parents = self.graph_state.parent_ids(leaf);
                            if let Some(first_parent) = parents.first() {
                                self.graph_state.current_leaf_id = Some(first_parent.clone());
                                if let Some(chat) = self.chat_window.as_mut() {
                                    if let Some(node) = self.graph_state.get_node(first_parent) {
                                        let center = node.position + node.size * 0.5;
                                        chat.constellation_view.center_camera_on(center);
                                        chat.constellation_view.reset_zoom_to_normal();
                                    }
                                }
                            }
                        }
                        return;
                    }
                    self.route_to_focused_editor(|editor| {
                        editor.on_arrow_up(shift_pressed);
                    });
                }
                KeyCode::ArrowDown => {
                    let shift_pressed = self.modifiers.contains(ModifiersState::SHIFT);
                    let alt_pressed = self.modifiers.contains(ModifiersState::ALT);
                    if alt_pressed && self.ui_state.active_tab == Tab::Chat && self.graph_state.graph_id.is_some() && self.focused_input.is_none() {
                        if let Some(ref leaf) = self.graph_state.current_leaf_id {
                            let children = self.graph_state.children_ids(leaf);
                            if let Some(first_child) = children.first() {
                                self.graph_state.current_leaf_id = Some(first_child.clone());
                                if let Some(chat) = self.chat_window.as_mut() {
                                    if let Some(node) = self.graph_state.get_node(first_child) {
                                        let center = node.position + node.size * 0.5;
                                        chat.constellation_view.center_camera_on(center);
                                        chat.constellation_view.reset_zoom_to_normal();
                                    }
                                }
                            }
                        }
                        return;
                    }
                    self.route_to_focused_editor(|editor| {
                        editor.on_arrow_down(shift_pressed);
                    });
                }
                _ => {}
                } // Close match key_code
            } // Close if let PhysicalKey::Code(key_code)
            
            // Sync cursor position for smooth interpolation after keyboard events
            if self.focused_input == Some(0) {
                if let Some(ref chat) = self.chat_window {
                    self.cursor_target_position = chat.input_field.cursor_position;
                    self.cursor_blink_timer = 0.0;
                    self.cursor_visible = true;
                }
            }
        } // Close if event.state == Pressed
    }

    pub fn on_char_received(&mut self, ch: char) {
        // Special handling for chat input (undo history)
        if self.focused_input == Some(0) {
            if let Some(ref mut chat) = self.chat_window {
                if !chat.input_field.has_selection() {
                    let current_text = chat.input_field.text.clone();
                    // Save state for undo
                    self.undo_history.push(current_text);
                    self.redo_history.clear();
                    if self.undo_history.len() > 50 {
                        self.undo_history.remove(0);
                    }
                }
            }
        }
        
        // Route character to focused editor (router updates cursor animation)
        self.route_to_focused_editor(|editor| {
            editor.on_char_received(ch);
        });
        
        // Post-processing after character is received
        match self.focused_input {
            Some(1) => {
                // Update search after character is received
                if let Some(ref mut library) = self.library_window {
                    library.update_search();
                }
            }
            Some(11) => {
                // Update global search results
                let query = self.global_search_modal.search_input.text.clone();
                let papers: Vec<_> = if let Some(ref library) = self.library_window {
                    library.papers.iter().map(|p| crate::api::models::ApiPaper {
                        id: p.id,
                        filename: p.filename.clone(),
                        title: p.title.clone(),
                        authors: None,
                        year: p.year,
                    }).collect()
                } else {
                    Vec::new()
                };
                self.global_search_modal.search(&query, &self.chat_state.conversations, &self.insights_state.insights, &papers);
            }
            Some(12) => {
                // Update notepad document title when title input changes
                if let Some(ref mut notepad) = self.notepad_window {
                    notepad.document_title = notepad.title_input.text.clone();
                }
            }
            Some(3) | Some(4) | Some(6) => {
                // Auto-save settings
                use crate::persistence::SettingsPersistence;
                if let Err(e) = SettingsPersistence::save_settings(&self.settings_state) {
                    eprintln!("Failed to save settings: {}", e);
                }
            }
            _ => {}
        }
    }

    /// Centralized router for text editor operations
    /// Routes operations to the focused editor using the TextEditor trait
    fn route_to_focused_editor<F>(&mut self, f: F)
    where
        F: FnOnce(&mut dyn TextEditor),
    {
        match self.focused_input {
            Some(0) => {
                if let Some(ref mut chat) = self.chat_window {
                    f(&mut chat.input_field as &mut dyn TextEditor);
                    // Update cursor animation for chat
                    self.cursor_target_position = chat.input_field.get_cursor_position();
                    self.cursor_blink_timer = 0.0;
                    self.cursor_visible = true;
                }
            }
            Some(1) => {
                if let Some(ref mut library) = self.library_window {
                    f(&mut library.search_input as &mut dyn TextEditor);
                    // Update cursor animation for library search
                    self.cursor_target_position = library.search_input.get_cursor_position();
                    self.cursor_blink_timer = 0.0;
                    self.cursor_visible = true;
                }
            }
            Some(2) => {
                if let Some(ref mut ingest) = self.ingest_window {
                    f(&mut ingest.pdf_dir_input as &mut dyn TextEditor);
                    // Update cursor animation for data tab
                    self.cursor_target_position = ingest.pdf_dir_input.get_cursor_position();
                    self.cursor_blink_timer = 0.0;
                    self.cursor_visible = true;
                }
            }
            Some(3) => {
                if let Some(ref mut settings) = self.settings_window {
                    f(&mut settings.hf_token_input as &mut dyn TextEditor);
                    self.settings_state.hf_token = settings.hf_token_input.text.clone();
                    // Update cursor animation for settings HF token input
                    self.cursor_target_position = settings.hf_token_input.get_cursor_position();
                    self.cursor_blink_timer = 0.0;
                    self.cursor_visible = true;
                }
            }
            Some(4) => {
                if let Some(ref mut settings) = self.settings_window {
                    f(&mut settings.model_id_input as &mut dyn TextEditor);
                    self.settings_state.model_id = settings.model_id_input.text.clone();
                    // Update cursor animation for settings model ID input
                    self.cursor_target_position = settings.model_id_input.get_cursor_position();
                    self.cursor_blink_timer = 0.0;
                    self.cursor_visible = true;
                }
            }
            Some(6) => {
                if let Some(ref mut settings) = self.settings_window {
                    f(&mut settings.openai_model_input as &mut dyn TextEditor);
                    self.settings_state.openai_model = settings.openai_model_input.text.clone();
                    self.cursor_target_position = settings.openai_model_input.get_cursor_position();
                    self.cursor_blink_timer = 0.0;
                    self.cursor_visible = true;
                }
            }
            Some(5) => {
                if let Some(ref mut notepad) = self.notepad_window {
                    f(&mut notepad.editor as &mut dyn TextEditor);
                    // Update cursor animation for notepad
                    self.cursor_target_position = notepad.editor.get_cursor_position();
                    self.cursor_blink_timer = 0.0;
                    self.cursor_visible = true;
                }
            }
            Some(6) => {
                if self.insight_modal.is_editing_title {
                    f(&mut self.insight_modal.title_input as &mut dyn TextEditor);
                    self.insight_modal.draft_title = self.insight_modal.title_input.text.clone();
                    // Update cursor animation for insight modal title
                    self.cursor_target_position = self.insight_modal.title_input.get_cursor_position();
                    self.cursor_blink_timer = 0.0;
                    self.cursor_visible = true;
                }
            }
            Some(7) => {
                if self.insight_modal.is_editing_text {
                    f(&mut self.insight_modal.text_input as &mut dyn TextEditor);
                    self.insight_modal.draft_text = self.insight_modal.text_input.text.clone();
                    // Update cursor animation for insight modal text
                    self.cursor_target_position = self.insight_modal.text_input.get_cursor_position();
                    self.cursor_blink_timer = 0.0;
                    self.cursor_visible = true;
                }
            }
            Some(8) => {
                if self.chat_info_dialog.is_editing_title {
                    f(&mut self.chat_info_dialog.title_input as &mut dyn TextEditor);
                    self.chat_info_dialog.draft_title = self.chat_info_dialog.title_input.text.clone();
                    // Update cursor animation for chat info dialog title
                    self.cursor_target_position = self.chat_info_dialog.title_input.get_cursor_position();
                    self.cursor_blink_timer = 0.0;
                    self.cursor_visible = true;
                }
            }
            Some(9) => {
                if let Some(ref mut library) = self.library_window {
                    if library.is_creating_collection {
                        f(&mut library.new_collection_input as &mut dyn TextEditor);
                        // Update cursor animation for library new collection input
                        self.cursor_target_position = library.new_collection_input.get_cursor_position();
                        self.cursor_blink_timer = 0.0;
                        self.cursor_visible = true;
                    }
                }
            }
            Some(10) => {
                if let Some(ref mut chat) = self.chat_window {
                    if chat.editing_message_idx.is_some() {
                        f(&mut chat.edit_textarea as &mut dyn TextEditor);
                        // Update cursor animation for edit textarea
                        self.cursor_target_position = chat.edit_textarea.get_cursor_position();
                        self.cursor_blink_timer = 0.0;
                        self.cursor_visible = true;
                    }
                }
            }
            Some(20) => {
                if self.shard_modal.is_open {
                    f(&mut self.shard_modal.user_input as &mut dyn TextEditor);
                    self.cursor_target_position = self.shard_modal.user_input.get_cursor_position();
                    self.cursor_blink_timer = 0.0;
                    self.cursor_visible = true;
                }
            }
            Some(21) => {
                if self.shard_modal.is_open {
                    f(&mut self.shard_modal.assistant_input as &mut dyn TextEditor);
                    self.cursor_target_position = self.shard_modal.assistant_input.get_cursor_position();
                    self.cursor_blink_timer = 0.0;
                    self.cursor_visible = true;
                }
            }
            Some(11) => { // Global search input
                f(&mut self.global_search_modal.search_input as &mut dyn TextEditor);
                self.cursor_target_position = self.global_search_modal.search_input.get_cursor_position();
                self.cursor_blink_timer = 0.0;
                self.cursor_visible = true;
                
                // Update search results as user types
                let query = self.global_search_modal.search_input.text.clone();
                let papers: Vec<_> = if let Some(ref library) = self.library_window {
                    library.papers.iter().map(|p| crate::api::models::ApiPaper {
                        id: p.id,
                        filename: p.filename.clone(),
                        title: p.title.clone(),
                        authors: None,
                        year: p.year,
                    }).collect()
                } else {
                    Vec::new()
                };
                self.global_search_modal.search(&query, &self.chat_state.conversations, &self.insights_state.insights, &papers);
            }
            Some(13) => { // Chat note input (add/edit note)
                if let Some(ref mut chat) = self.chat_window {
                    if chat.adding_note_msg_idx.is_some() || chat.editing_note.is_some() {
                        f(&mut chat.note_input as &mut dyn TextEditor);
                        self.cursor_target_position = chat.note_input.get_cursor_position();
                        self.cursor_blink_timer = 0.0;
                        self.cursor_visible = true;
                    }
                }
            }
            Some(12) => { // Notepad title input
                if let Some(ref mut notepad) = self.notepad_window {
                    f(&mut notepad.title_input as &mut dyn TextEditor);
                    // Update document title when title input changes
                    notepad.document_title = notepad.title_input.text.clone();
                    self.cursor_target_position = notepad.title_input.get_cursor_position();
                    self.cursor_blink_timer = 0.0;
                    self.cursor_visible = true;
                }
            }
            _ => {}
        }
    }

    pub fn update(&mut self, dt: f32) {
        // Ensure we have a current conversation and graph when none exist (e.g. first run or empty state)
        if !self.conversation_ensured && (self.chat_state.current_conversation_id.is_none() || self.chat_state.conversations.is_empty()) {
            let _ = self.chat_state.create_conversation();
            self.request_graph_for_new_conversation();
            self.conversation_ensured = true;
            self.save_chat_state();
        }

        // Smooth glow intensity transitions (50% faster)
        let glow_transition_speed = 4.5; // Units per second
        if self.sidebar_edge_glow_intensity < self.sidebar_edge_glow_target_intensity {
            self.sidebar_edge_glow_intensity = (self.sidebar_edge_glow_intensity + glow_transition_speed * dt)
                .min(self.sidebar_edge_glow_target_intensity);
        } else if self.sidebar_edge_glow_intensity > self.sidebar_edge_glow_target_intensity {
            self.sidebar_edge_glow_intensity = (self.sidebar_edge_glow_intensity - glow_transition_speed * dt)
                .max(self.sidebar_edge_glow_target_intensity);
        }
        
        // Cursor blinking (0.5 second cycle)
        self.cursor_blink_timer += dt;
        if self.cursor_blink_timer >= 0.5 {
            self.cursor_blink_timer = 0.0;
            self.cursor_visible = !self.cursor_visible;
        }
        
        // Smooth cursor position interpolation using SpringAnimation
        self.cursor_position_animation.target = self.cursor_target_position as f32;
        self.cursor_position_animation.update(dt);
        
        self.header.update(dt);
        // Update document list
        use crate::persistence::DocumentPersistence;
        let documents = DocumentPersistence::list_documents().unwrap_or_default();
        
        self.sidebar.update(
            dt,
            self.chat_state.conversations.len(),
            documents.len(),
            self.insights_state.insights.len(),
            &self.chat_state.conversations,
            &documents,
            &self.insights_state.insights,
        );
        self.sidebar.update_layout(
            self.header.size.y,
            &self.chat_state.conversations,
            &documents,
            &self.insights_state.insights,
        );
        let conversations_height = 10.0 + (self.chat_state.conversations.len() as f32 * 40.0) + 10.0;
        self.sidebar.conversations_list.set_content_height(conversations_height);
        let documents_height = 10.0 + (documents.len() as f32 * 40.0) + 10.0;
        self.sidebar.documents_list.set_content_height(documents_height);
        let insights_height = 10.0 + (self.insights_state.insights.len() as f32 * 35.0) + 10.0;
        self.sidebar.insights_panel.insights_list.set_content_height(insights_height);
        
        // Update chat window layout if sidebar width changed (keep same width and center when sidebar closed)
        if let Some(ref mut chat) = self.chat_window {
            let chat_y = self.header.size.y;
            let chat_height = self.viewport_size.y - self.header.size.y;
            let open_content_width = self.viewport_size.x - SidebarWindow::OPEN_WIDTH;
            let (chat_x, chat_width) = if self.sidebar.is_open {
                (self.sidebar.current_width, self.viewport_size.x - self.sidebar.current_width)
            } else {
                ((self.viewport_size.x - open_content_width) / 2.0, open_content_width)
            };
            chat.position = Vec2::new(chat_x, chat_y);
            chat.size = Vec2::new(chat_width, chat_height);
            chat.update_layout();
            if chat.editing_message_idx.is_some() {
                let measure = |text: &str, size: f32| -> Vec2 {
                    Vec2::new(text.len() as f32 * size * 0.66, size)
                };
                if self.graph_state.graph_id.is_some() {
                    chat.update_edit_textarea_rect_constellation(measure, &self.graph_state);
                } else {
                    chat.update_edit_textarea_rect(measure);
                }
            }
            if chat.adding_note_msg_idx.is_some() || chat.editing_note.is_some() {
                if self.graph_state.graph_id.is_some() {
                    let measure = |text: &str, size: f32| -> Vec2 {
                        Vec2::new(text.len() as f32 * size * 0.66, size)
                    };
                    let viewport_bottom = chat.constellation_view.position.y + chat.constellation_view.size.y;
                    chat.update_note_input_rect_constellation(measure, &self.graph_state, viewport_bottom);
                }
            }
            chat.message_list.update(dt);
            chat.input_field.update(dt);
            chat.context_pool_dropdown.update(dt);
            chat.constellation_view.update(dt);

            // Constellation physics: fixed-step integration with idle cutoff to keep shards from overlapping.
            if self.graph_state.graph_id.is_some() {
                // Run physics primarily while in Chat tab; keep a short settling window after graph changes.
                self.physics_idle_timer += dt;
                const PHYSICS_FPS: f32 = 30.0;
                const PHYSICS_DT: f32 = 1.0 / PHYSICS_FPS;
                const MAX_STEPS_PER_FRAME: u32 = 4;
                const IDLE_TIMEOUT: f32 = 3.0;
                const VELOCITY_EPS: f32 = 0.01;

                let velocity_sum = self.graph_state.total_velocity_magnitude();
                let active_tab_is_chat = self.ui_state.active_tab == crate::ui::tab_bar::Tab::Chat;
                let should_run_physics = active_tab_is_chat
                    && (self.physics_settle_frames > 0
                        || self.physics_idle_timer < IDLE_TIMEOUT
                        || velocity_sum > VELOCITY_EPS);

                if should_run_physics {
                    self.physics_dt_accumulator += dt;
                    let mut steps: u32 = 0;
                    while self.physics_dt_accumulator >= PHYSICS_DT && steps < MAX_STEPS_PER_FRAME {
                        self.graph_state.step_physics(PHYSICS_DT);
                        self.physics_dt_accumulator -= PHYSICS_DT;
                        steps += 1;
                    }
                    if self.physics_settle_frames > 0 {
                        self.physics_settle_frames -= 1;
                    }
                } else {
                    self.physics_dt_accumulator = 0.0;
                    self.graph_state.zero_velocities_if_settled(VELOCITY_EPS);
                }
            }

            // Smooth scroll: lerp constellation scroll offsets toward targets
            {
                const LERP_SPEED: f32 = 12.0;
                let t = (LERP_SPEED * dt).min(1.0);
                let targets = chat.constellation_scroll_targets.borrow();
                let mut offsets = chat.constellation_scroll_offsets.borrow_mut();
                for (id, &(ut, at)) in targets.iter() {
                    let (uo, ao) = offsets.get(id).copied().unwrap_or((0.0, 0.0));
                    offsets.insert(id.clone(), (uo + (ut - uo) * t, ao + (at - ao) * t));
                }
            }

            // Throttled layout persistence (every 2s) when graph is loaded, regardless of active tab
            if self.graph_state.graph_id.is_some() {
                self.layout_save_timer += dt;
                if self.layout_save_timer >= 2.0 {
                    self.layout_save_timer = 0.0;
                    if let Some(ref graph_id) = self.graph_state.graph_id {
                        let positions: std::collections::HashMap<String, glam::Vec2> = self
                            .graph_state
                            .nodes
                            .iter()
                            .map(|(id, n)| (id.clone(), n.position))
                            .collect();
                        let _ = crate::persistence::GraphLayoutPersistence::save_positions(graph_id, &positions);
                    }
                }
            } else {
                self.layout_save_timer = 0.0;
            }

            // Load collections when chat window is first shown
            if !self.collections_loaded {
                self.load_collections();
            }
        }

        // Update library window layout
        if let Some(ref mut library) = self.library_window {
            let library_y = self.header.size.y;
            let library_width = self.viewport_size.x - self.sidebar.current_width;
            let library_height = self.viewport_size.y - self.header.size.y;
            library.position = Vec2::new(self.sidebar.current_width, library_y);
            library.size = Vec2::new(library_width, library_height);
            library.update_layout();
            library.search_input.update(dt);
            library.papers_list.update(dt);
            library.collections_list.update(dt);
        }

        // Update notepad window layout
        if let Some(ref mut notepad) = self.notepad_window {
            let notepad_y = self.header.size.y;
            let notepad_width = self.viewport_size.x - self.sidebar.current_width;
            let notepad_height = self.viewport_size.y - self.header.size.y;
            notepad.position = Vec2::new(self.sidebar.current_width, notepad_y);
            notepad.size = Vec2::new(notepad_width, notepad_height);
            notepad.update_layout();
            notepad.editor.update(dt);
            notepad.title_input.update(dt);
        }
        
        // Update ingest window layout (Data tab)
        if let Some(ref mut ingest) = self.ingest_window {
            let ingest_y = self.header.size.y;
            let ingest_width = self.viewport_size.x - self.sidebar.current_width;
            let ingest_height = self.viewport_size.y - self.header.size.y;
            ingest.position = Vec2::new(self.sidebar.current_width, ingest_y);
            ingest.size = Vec2::new(ingest_width, ingest_height);
            ingest.update_layout();
            ingest.pdf_dir_input.update(dt);
        }
        
        // Update settings window layout
        if let Some(ref mut settings) = self.settings_window {
            let settings_y = self.header.size.y;
            let settings_width = self.viewport_size.x - self.sidebar.current_width;
            let settings_height = self.viewport_size.y - self.header.size.y;
            settings.position = Vec2::new(self.sidebar.current_width, settings_y);
            settings.size = Vec2::new(settings_width, settings_height);
            settings.update_layout(&self.settings_state.provider);
        }
        
        for win in &mut self.windows {
            win.update(dt, self.viewport_size);
        }
        
        // Update settings window inputs
        if let Some(ref mut settings) = self.settings_window {
            settings.hf_token_input.update(dt);
            settings.model_id_input.update(dt);
            settings.openai_model_input.update(dt);
        }
        
        // Update modal layouts and inputs
        if self.shard_modal.is_open {
            self.shard_modal.update_layout(self.viewport_size);
            self.shard_modal.user_input.update(dt);
            self.shard_modal.assistant_input.update(dt);
        }
        if self.insight_modal.is_open {
            self.insight_modal.update_layout(self.viewport_size);
            self.insight_modal.title_input.update(dt);
            self.insight_modal.text_input.update(dt);
        }
        if self.pdf_modal.is_open {
            self.pdf_modal.update_layout(self.viewport_size);
        }
        if self.chat_info_dialog.is_open {
            self.chat_info_dialog.update_layout(self.viewport_size);
            self.chat_info_dialog.title_input.update(dt);
        }
        if let Some(ref mut notepad) = self.notepad_window {
            if notepad.notepad_modal.is_open {
                notepad.notepad_modal.update_layout(self.viewport_size);
            }
        }
        
        // Update toast manager
        self.toast_manager.update(dt, self.viewport_size);
    }

    /// True if any animation or physics is active and we need to keep redrawing every frame.
    /// When false, the event loop can stop requesting redraws (saves CPU/GPU when graph is idle).
    pub fn needs_continuous_redraw(&self) -> bool {
        const LERP_EPS: f32 = 0.5;
        const SCALE_EPS: f32 = 0.001;

        // Cursor position spring
        if !self.cursor_position_animation.is_at_target() {
            return true;
        }

        // Header tab bar slider springs
        if !self.header.tab_bar.slider_animation.is_at_target()
            || !self.header.tab_bar.slider_trailing_animation.is_at_target()
        {
            return true;
        }

        // Sidebar edge glow
        if (self.sidebar_edge_glow_intensity - self.sidebar_edge_glow_target_intensity).abs() > 0.001 {
            return true;
        }

        // Sidebar width or list scroll/expand animations
        if self.sidebar.has_active_animation() {
            return true;
        }

        // Constellation view: camera and scale lerp
        if let Some(chat) = &self.chat_window {
            if self.graph_state.graph_id.is_some() {
                let v = &chat.constellation_view;
                if (v.camera_position_animated - v.camera_position).length() > LERP_EPS {
                    return true;
                }
                if (v.scale_animated - v.scale).abs() > SCALE_EPS {
                    return true;
                }
                // Constellation scroll offsets lerping toward targets
                let targets = chat.constellation_scroll_targets.borrow();
                let offsets = chat.constellation_scroll_offsets.borrow();
                for (id, &(ut, at)) in targets.iter() {
                    let (uo, ao) = offsets.get(id).copied().unwrap_or((0.0, 0.0));
                    if (uo - ut).abs() > LERP_EPS || (ao - at).abs() > LERP_EPS {
                        return true;
                    }
                }
            }
        }

        // Constellation physics: keep redrawing while nodes are moving or within settle window.
        if self.graph_state.graph_id.is_some() {
            if self.physics_settle_frames > 0 {
                return true;
            }
            if self.graph_state.total_velocity_magnitude() > 0.01 {
                return true;
            }
        }

        false
    }

    /// Returns the union of screen regions that are currently animating, for partial redraw.
    /// Disabled for baseline: always return None so we always full redraw (no dirty rect logic).
    pub fn get_dirty_rects(&self) -> Option<crate::ui::core::Rect> {
        None
    }

    #[allow(dead_code)]
    fn get_dirty_rects_impl(&self) -> Option<crate::ui::core::Rect> {
        use crate::ui::core::Rect;
        const MAX_DIRTY_AREA_RATIO: f32 = 0.8;

        if self.shard_modal.is_open
            || self.insight_modal.is_open
            || self.pdf_modal.is_open
            || self.chat_info_dialog.is_open
        {
            return None;
        }
        if !self.toast_manager.toasts.is_empty() {
            return None;
        }
        if let Some(ref notepad) = self.notepad_window {
            if notepad.notepad_modal.is_open {
                return None;
            }
        }

        let viewport = self.viewport_size;
        let total_area = viewport.x * viewport.y;

        fn union(a: Rect, b: Rect) -> Rect {
            let left = a.x.min(b.x);
            let top = a.y.min(b.y);
            let right = (a.x + a.width).max(b.x + b.width);
            let bottom = (a.y + a.height).max(b.y + b.height);
            Rect::new(left, top, (right - left).max(0.0), (bottom - top).max(0.0))
        }

        let mut dirty: Option<Rect> = None;

        if !self.cursor_position_animation.is_at_target() {
            if let Some(chat) = &self.chat_window {
                let r = Rect::new(
                    chat.position.x,
                    chat.position.y,
                    chat.size.x,
                    chat.size.y,
                );
                dirty = Some(dirty.map(|d| union(d, r)).unwrap_or(r));
            }
        }

        if !self.header.tab_bar.slider_animation.is_at_target()
            || !self.header.tab_bar.slider_trailing_animation.is_at_target()
        {
            let r = Rect::new(
                self.header.position.x,
                self.header.position.y,
                self.header.size.x,
                self.header.size.y,
            );
            dirty = Some(dirty.map(|d| union(d, r)).unwrap_or(r));
        }

        if (self.sidebar_edge_glow_intensity - self.sidebar_edge_glow_target_intensity).abs() > 0.001
            || self.sidebar.has_active_animation()
        {
            let r = Rect::new(
                0.0,
                0.0,
                self.sidebar.current_width + 24.0,
                viewport.y,
            );
            dirty = Some(dirty.map(|d| union(d, r)).unwrap_or(r));
        }

        if let Some(chat) = &self.chat_window {
            if self.graph_state.graph_id.is_some() {
                let v = &chat.constellation_view;
                let camera_moving = (v.camera_position_animated - v.camera_position).length() > 0.5;
                let scale_moving = (v.scale_animated - v.scale).abs() > 0.001;
                let scroll_moving = {
                    let targets = chat.constellation_scroll_targets.borrow();
                    let offsets = chat.constellation_scroll_offsets.borrow();
                    let mut any = false;
                    for (id, &(ut, at)) in targets.iter() {
                        let (uo, ao) = offsets.get(id).copied().unwrap_or((0.0, 0.0));
                        if (uo - ut).abs() > 0.5 || (ao - at).abs() > 0.5 {
                            any = true;
                            break;
                        }
                    }
                    any
                };
                if camera_moving || scale_moving || scroll_moving
                    || self.physics_settle_frames > 0
                    || self.graph_state.total_velocity_magnitude() > 0.03
                {
                    let r = Rect::new(
                        chat.constellation_view.position.x,
                        chat.constellation_view.position.y,
                        chat.constellation_view.size.x,
                        chat.constellation_view.size.y,
                    );
                    dirty = Some(dirty.map(|d| union(d, r)).unwrap_or(r));
                }
            }
        }

        let d = dirty?;
        let dirty_area = d.width * d.height;
        if dirty_area >= total_area * MAX_DIRTY_AREA_RATIO {
            return None;
        }
        Some(d)
    }

    fn update_button_hover_states(&mut self) {
        use crate::ui::tab_bar::Tab;
        use crate::persistence::DocumentPersistence;
        
        // Update sidebar buttons
        self.sidebar.new_conversation_button.on_hover(self.mouse_pos);
        self.sidebar.delete_conversation_button.on_hover(self.mouse_pos);
        self.sidebar.new_document_button.on_hover(self.mouse_pos);
        self.sidebar.delete_document_button.on_hover(self.mouse_pos);
        
        // Update sidebar hover states for list items
        let document_ids = DocumentPersistence::list_documents().unwrap_or_default();
        self.sidebar.update_hover_state(
            self.mouse_pos,
            &self.chat_state.conversations,
            &document_ids,
            &self.insights_state.insights,
        );
        
        // Update chat window buttons
        if let Some(ref mut chat) = self.chat_window {
            if self.ui_state.active_tab == Tab::Chat {
                // Send button (using position/size, not Button struct)
                // Context pool button (using position/size, not Button struct)
            }
        }
        
        // Update library window buttons
        if let Some(ref mut library) = self.library_window {
            if self.ui_state.active_tab == Tab::Library {
                // Library buttons if any
            }
        }
        
        // Update ingest window buttons
        if let Some(ref mut ingest) = self.ingest_window {
            if self.ui_state.active_tab == Tab::Data {
                ingest.ingest_button.on_hover(self.mouse_pos);
        }
    }
    }
    
    /// Update hover state (mouse enter/leave tracking)
    fn update_hover_state(&mut self, old_pos: Vec2) {
        // Simple hover tracking: detect which component is under cursor
        // For now, track major components - can be extended with component IDs
        let current_component = self.get_component_at_position(self.mouse_pos);
        let old_component = self.get_component_at_position(old_pos);
        
        // If component changed, trigger leave/enter events
        if current_component != old_component {
            if let Some(old_id) = old_component {
                self.hover_state.last_hovered_component_id = Some(old_id.clone());
                // TODO: Route on_mouse_leave to component
            }
            if let Some(new_id) = current_component {
                self.hover_state.hovered_component_id = Some(new_id.clone());
                // TODO: Route on_mouse_enter to component
            }
        } else {
            // Same component, update hovered state
            if let Some(id) = current_component {
                self.hover_state.hovered_component_id = Some(id);
            }
        }
    }
    
    /// Get component ID at position (simplified - returns string identifier)
    fn get_component_at_position(&self, pos: Vec2) -> Option<String> {
        use crate::ui::tab_bar::Tab;
        
        // Check z-index order (highest to lowest)
        // Header
        if pos.y < self.header.size.y {
            return Some("header".to_string());
        }
        
        // Modals
        if self.shard_modal.is_open && self.shard_modal.contains(pos) {
            return Some("shard_modal".to_string());
        }
        if self.insight_modal.is_open && self.insight_modal.contains(pos) {
            return Some("insight_modal".to_string());
        }
        if self.pdf_modal.is_open && self.pdf_modal.contains(pos) {
            return Some("pdf_modal".to_string());
        }
        if self.chat_info_dialog.is_open && self.chat_info_dialog.contains(pos) {
            return Some("chat_info_dialog".to_string());
        }
        
        // Windows based on active tab
        match self.ui_state.active_tab {
            Tab::Chat => {
                if let Some(ref chat) = self.chat_window {
                    if pos.x >= chat.position.x && pos.x <= chat.position.x + chat.size.x &&
                       pos.y >= chat.position.y && pos.y <= chat.position.y + chat.size.y {
                        return Some("chat_window".to_string());
                    }
                }
            }
            Tab::Library => {
                if let Some(ref library) = self.library_window {
                    if library.contains(pos) {
                        return Some("library_window".to_string());
                    }
                }
            }
            Tab::Data => {
                if let Some(ref ingest) = self.ingest_window {
                    if ingest.contains(pos) {
                        return Some("ingest_window".to_string());
                    }
                }
            }
            Tab::Settings => {
                if let Some(ref settings) = self.settings_window {
                    if settings.contains(pos) {
                        return Some("settings_window".to_string());
                    }
                }
            }
            Tab::Notepad => {
                if let Some(ref notepad) = self.notepad_window {
                    if notepad.editor.contains(pos) {
                        return Some("notepad_window".to_string());
                    }
                }
            }
        }
        
        // Sidebar
        if self.sidebar.hit_test(pos) {
            return Some("sidebar".to_string());
        }
        
        None
    }
    
    /// Handle mouse drag (during drag operation)
    fn on_mouse_drag(&mut self, position: Vec2) {
        // Constellation pan: drag in viewport moves camera
        if self.ui_state.active_tab == Tab::Chat
            && self.graph_state.graph_id.is_some()
            && self.chat_window.is_some()
        {
            let chat = self.chat_window.as_mut().unwrap();
            if let Some(start) = chat.constellation_view.pan_drag_start {
                let delta = position - start;
                chat.constellation_view.pan(delta);
                chat.constellation_view.pan_drag_start = Some(position);
                self.physics_idle_timer = 0.0;
                return;
            }
        }

        // Route drag to appropriate component
        // For now, handle text selection in editors
        if let Some(ref mut notepad) = self.notepad_window {
            if self.ui_state.active_tab == Tab::Notepad && notepad.editor.contains(position) {
                // Use TextEditor trait for consistency
                use crate::ui::TextEditor;
                TextEditor::on_mouse_move(&mut notepad.editor, position);
            }
        }
        
        // Update drag state
        if let crate::ui::events::DragState::Starting { button, start_pos } = self.drag_state {
            self.drag_state = crate::ui::events::DragState::Dragging {
                button,
                start_pos,
            };
        }
    }
    
    /// Handle window focus (window gains focus)
    pub fn on_window_focus(&mut self) {
        self.window_focused = true;
        // Resume cursor blinking
        self.cursor_visible = true;
        self.cursor_blink_timer = 0.0;
        // TODO: Resume any paused animations
    }
    
    /// Handle window blur (window loses focus)
    pub fn on_window_blur(&mut self) {
        self.window_focused = false;
        // Stop cursor blinking when unfocused
        self.cursor_visible = false;
        // TODO: Pause non-essential animations
    }
    
    /// Handle window moved
    pub fn on_window_moved(&mut self, position: winit::dpi::PhysicalPosition<i32>) {
        self.window_position = Vec2::new(position.x as f32, position.y as f32);
        // TODO: Save window position for persistence
    }
    
    /// Handle scale factor changed (DPI change)
    pub fn on_scale_factor_changed(&mut self, scale_factor: f64) {
        self.scale_factor = scale_factor;
        // Recalculate font sizes and layout based on new scale
        // The renderer should handle this, but we may need to update UI element sizes
        // For now, just store the scale factor - layout will be recalculated on next resize
    }
    
    /// Handle key release
    pub fn on_key_released(&mut self, key_code: KeyCode) {
        // Check for shortcut matches on key release
        if let Some(shortcut_id) = self.shortcut_registry.find(self.modifiers, key_code) {
            self.on_shortcut_triggered(shortcut_id);
        }
        // TODO: Handle key release for toggle states and key combinations
    }
    
    /// Handle shortcut triggered
    fn on_shortcut_triggered(&mut self, shortcut_id: crate::ui::shortcuts::ShortcutId) {
        use winit::keyboard::{KeyCode, ModifiersState};
        
        // Find the shortcut to get its key
        let shortcut = self.shortcut_registry.all()
            .iter()
            .find(|s| s.id == shortcut_id);
        
        if let Some(shortcut) = shortcut {
            match shortcut.key {
                KeyCode::KeyN => {
                    // New conversation (works with both SUPER and CONTROL)
                    if shortcut.modifiers.contains(ModifiersState::SUPER) || shortcut.modifiers.contains(ModifiersState::CONTROL) {
                        let conv_id = self.chat_state.create_conversation();
                        self.sidebar.selected_conversation_id = Some(conv_id.clone());
                        self.request_graph_for_new_conversation();
                        self.save_chat_state();
                        self.show_success_toast("New conversation created".to_string());
                    }
                }
                KeyCode::KeyK => {
                    // Global search
                    if shortcut.modifiers.contains(ModifiersState::SUPER) || shortcut.modifiers.contains(ModifiersState::CONTROL) {
                        self.global_search_modal.open();
                        self.focused_input = Some(11); // Use index 11 for global search input
                    }
                }
                KeyCode::Comma => {
                    // Open settings
                    if shortcut.modifiers.contains(ModifiersState::SUPER) || shortcut.modifiers.contains(ModifiersState::CONTROL) {
                        self.ui_state.active_tab = crate::ui::tab_bar::Tab::Settings;
                    }
                }
                KeyCode::KeyB => {
                    // Toggle sidebar
                    if shortcut.modifiers.contains(ModifiersState::SUPER) || shortcut.modifiers.contains(ModifiersState::CONTROL) {
                        self.ui_state.toggle_sidebar();
                    }
                }
                KeyCode::Enter => {
                    // Send message (works with both SUPER and CONTROL) — graph send only
                    if shortcut.modifiers.contains(ModifiersState::SUPER) || shortcut.modifiers.contains(ModifiersState::CONTROL) {
                        if self.focused_input == Some(0) {
                            let shortcut_send_provider = self.settings_state.provider.clone();
                            let shortcut_send_model_id = self.settings_state.model_id_for_send();
                            let shortcut_send_openai_model = self.settings_state.openai_model_for_send();
                            let mut shortcut_send_pending: Option<(String, crate::api::models::GraphSendRequest, String)> = None;
                            if let Some(ref mut chat) = self.chat_window {
                                let text = chat.input_field.text.trim().to_string();
                                if !text.is_empty() && self.graph_state.graph_id.is_some() {
                                    let graph_id = self.graph_state.graph_id.clone().unwrap();
                                    let leaf_id = self.graph_state.current_leaf_id.clone().unwrap_or_default();
                                    let request = crate::api::models::GraphSendRequest {
                                        current_leaf_id: leaf_id,
                                        user_draft: text.clone(),
                                        provider: shortcut_send_provider.clone(),
                                        model_id: shortcut_send_model_id.clone(),
                                        openai_model: shortcut_send_openai_model.clone(),
                                        temperature: None,
                                        max_tokens: None,
                                        model_token_limit: None,
                                    };
                                    let user_msg = crate::ui::chat_window::ChatMessage::from_legacy(
                                        crate::ui::chat_window::MessageRole::User,
                                        text.clone(),
                                        Vec::new(),
                                        Vec::new(),
                                    );
                                    chat.add_message(user_msg.clone());
                                    self.chat_state.add_message_to_current(user_msg);
                                    shortcut_send_pending = Some((graph_id, request, text));
                                    chat.input_field.text.clear();
                                    chat.input_field.cursor_position = 0;
                                } else if !text.is_empty() {
                                    self.show_error_toast("Conversation not ready. Please wait for it to load.".to_string());
                                }
                            }
                            if let Some((graph_id, request, text)) = shortcut_send_pending {
                                self.save_chat_state();
                                let client = self.api_client.client.clone();
                                let base_url = self.api_client.base_url.clone();
                                let sender = self.graph_send_sender.clone();
                                self.is_sending_message = true;
                                if let Some(ref mut chat) = self.chat_window {
                                    chat.is_sending = true;
                                }
                                tokio::spawn(async move {
                                    let url = format!("{}/graph/{}/send", base_url, graph_id);
                                    match client.post(&url).json(&request).send().await {
                                        Ok(r) if r.status().is_success() => {
                                            match r.json::<crate::api::models::GraphSendResponse>().await {
                                                Ok(body) => { let _ = sender.send(Ok((text, body))); }
                                                Err(e) => { let _ = sender.send(Err(format!("Parse: {:?}", e))); }
                                            }
                                        }
                                        Ok(r) => {
                                            let err = r.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                                            let _ = sender.send(Err(err));
                                        }
                                        Err(e) => { let _ = sender.send(Err(format!("Request: {:?}", e))); }
                                    }
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    
    /// Handle touch event
    pub fn on_touch(&mut self, touch: &winit::event::Touch) {
        use winit::event::TouchPhase;
        let touch_id = touch.id;
        let position = Vec2::new(touch.location.x as f32, touch.location.y as f32);
        
        match touch.phase {
            TouchPhase::Started => {
                self.on_touch_start(touch_id, position);
            }
            TouchPhase::Moved => {
                self.on_touch_move(touch_id, position);
            }
            TouchPhase::Ended => {
                self.on_touch_end(touch_id, position);
            }
            TouchPhase::Cancelled => {
                self.on_touch_cancel(touch_id);
            }
        }
    }
    
    /// Handle touch start
    fn on_touch_start(&mut self, touch_id: u64, position: Vec2) {
        self.active_touches.insert(touch_id, position);
        // Map to mouse down for compatibility
        // Use left mouse button for single touch
        if self.active_touches.len() == 1 {
            self.mouse_pos = position;
            self.on_mouse_button(MouseButton::Left, ElementState::Pressed);
        }
    }
    
    /// Handle touch move
    fn on_touch_move(&mut self, touch_id: u64, position: Vec2) {
        if let Some(old_pos) = self.active_touches.get_mut(&touch_id) {
            *old_pos = position;
            // Map to mouse move for compatibility
            if self.active_touches.len() == 1 {
                self.on_cursor_moved(winit::dpi::PhysicalPosition::new(position.x as f64, position.y as f64));
            }
        }
    }
    
    /// Handle touch end
    fn on_touch_end(&mut self, touch_id: u64, position: Vec2) {
        self.active_touches.remove(&touch_id);
        // Map to mouse up for compatibility
        if self.active_touches.is_empty() {
            self.mouse_pos = position;
            self.on_mouse_button(MouseButton::Left, ElementState::Released);
        }
    }
    
    /// Handle touch cancel
    fn on_touch_cancel(&mut self, touch_id: u64) {
        self.active_touches.remove(&touch_id);
        // Clear any drag operations
        if self.active_touches.is_empty() {
            self.is_dragging = false;
            self.drag_button = None;
            self.drag_state = crate::ui::events::DragState::None;
        }
    }
    
    /// Handle focus traversal (Tab/Shift+Tab)
    fn focus_traverse(&mut self, direction: crate::ui::events::FocusDirection) {
        // Build list of focusable components based on current tab
        use crate::ui::tab_bar::Tab;
        let mut focusable = Vec::new();
        
        match self.ui_state.active_tab {
            Tab::Chat => {
                if self.chat_window.is_some() {
                    focusable.push("chat_input".to_string());
                }
            }
            Tab::Library => {
                if self.library_window.is_some() {
                    focusable.push("library_search".to_string());
                }
            }
            Tab::Data => {
                if self.ingest_window.is_some() {
                    focusable.push("pdf_dir_input".to_string());
                }
            }
            Tab::Settings => {
                if self.settings_window.is_some() {
                    focusable.push("hf_token_input".to_string());
                    focusable.push("model_id_input".to_string());
                    focusable.push("openai_model_input".to_string());
                }
            }
            Tab::Notepad => {
                if self.notepad_window.is_some() {
                    focusable.push("notepad_editor".to_string());
                }
            }
        }
        
        self.focus_state.focusable_components = focusable;
        
        // Find current focus index
        let current_id = self.focus_state.focused_component_id.as_ref();
        let current_index = current_id.and_then(|id| {
            self.focus_state.focusable_components.iter().position(|c| c == id)
        });
        
        // Calculate next index
        let next_index = if self.focus_state.focusable_components.is_empty() {
            None
        } else {
            match direction {
                crate::ui::events::FocusDirection::Forward => {
                    if let Some(idx) = current_index {
                        Some((idx + 1) % self.focus_state.focusable_components.len())
                    } else {
                        Some(0)
                    }
                }
                crate::ui::events::FocusDirection::Backward => {
                    if let Some(idx) = current_index {
                        if idx == 0 {
                            Some(self.focus_state.focusable_components.len() - 1)
                        } else {
                            Some(idx - 1)
                        }
                    } else {
                        Some(self.focus_state.focusable_components.len() - 1)
                    }
                }
            }
        };
        
        // Blur current component
        if let Some(ref old_id) = self.focus_state.focused_component_id {
            self.on_component_blurred(old_id.clone());
        }
        
        // Focus next component
        if let Some(idx) = next_index {
            if let Some(component_id) = self.focus_state.focusable_components.get(idx) {
                self.focus_state.focused_component_id = Some(component_id.clone());
                self.focus_state.focus_index = Some(idx);
                self.on_component_focused(component_id.clone());
            }
        }
    }
    
    /// Handle component focused
    fn on_component_focused(&mut self, component_id: String) {
        // Route focus to appropriate component
        match component_id.as_str() {
            "chat_input" => {
                if let Some(ref mut chat) = self.chat_window {
                    chat.input_field.on_focus();
                    self.focused_input = Some(0);
                }
            }
            "library_search" => {
                if let Some(ref mut library) = self.library_window {
                    library.search_input.on_focus();
                    self.focused_input = Some(1);
                }
            }
            "pdf_dir_input" => {
                if let Some(ref mut ingest) = self.ingest_window {
                    ingest.pdf_dir_input.on_focus();
                    self.focused_input = Some(2);
                }
            }
            "hf_token_input" => {
                if let Some(ref mut settings) = self.settings_window {
                    settings.hf_token_input.on_focus();
                    self.focused_input = Some(3);
                }
            }
            "model_id_input" => {
                if let Some(ref mut settings) = self.settings_window {
                    settings.model_id_input.on_focus();
                    self.focused_input = Some(4);
                }
            }
            "openai_model_input" => {
                if let Some(ref mut settings) = self.settings_window {
                    settings.openai_model_input.on_focus();
                    self.focused_input = Some(6);
                }
            }
            "notepad_editor" => {
                if let Some(ref mut notepad) = self.notepad_window {
                    notepad.editor.focus();
                    self.focused_input = Some(5);
                }
            }
            _ => {}
        }
    }
    
    /// Handle component blurred
    fn on_component_blurred(&mut self, component_id: String) {
        // Route blur to appropriate component
        match component_id.as_str() {
            "chat_input" => {
                if let Some(ref mut chat) = self.chat_window {
                    chat.input_field.on_blur();
                }
            }
            "library_search" => {
                if let Some(ref mut library) = self.library_window {
                    library.search_input.on_blur();
                }
            }
            "pdf_dir_input" => {
                if let Some(ref mut ingest) = self.ingest_window {
                    ingest.pdf_dir_input.on_blur();
                }
            }
            "hf_token_input" => {
                if let Some(ref mut settings) = self.settings_window {
                    settings.hf_token_input.on_blur();
                }
            }
            "model_id_input" => {
                if let Some(ref mut settings) = self.settings_window {
                    settings.model_id_input.on_blur();
                }
            }
            "openai_model_input" => {
                if let Some(ref mut settings) = self.settings_window {
                    settings.openai_model_input.on_blur();
                }
            }
            "notepad_editor" => {
                if let Some(ref mut notepad) = self.notepad_window {
                    notepad.editor.blur();
                }
            }
            _ => {}
        }
    }
    
    /// Handle file drop (files dropped on window)
    pub fn on_file_drop(&mut self, paths: Vec<std::path::PathBuf>, position: Vec2) {
        self.file_drag_active = false;
        self.file_drag_position = position;
        
        use crate::ui::tab_bar::Tab;
        
        // Filter for PDF files and directories
        let mut pdf_files: Vec<std::path::PathBuf> = Vec::new();
        let mut directories: Vec<std::path::PathBuf> = Vec::new();
        
        for path in &paths {
            if path.is_dir() {
                directories.push(path.clone());
            } else if path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("pdf"))
                .unwrap_or(false) {
                pdf_files.push(path.clone());
            }
        }
        
        // Determine the directory to ingest
        let ingest_dir = if !directories.is_empty() {
            // If directories are dropped, use the first one
            Some(directories[0].clone())
        } else if !pdf_files.is_empty() {
            // If PDF files are dropped, use their common parent directory
            if pdf_files.len() == 1 {
                pdf_files[0].parent().map(|p| p.to_path_buf())
            } else {
                // Find common parent directory
                let mut common_parent = pdf_files[0].parent();
                for pdf_file in &pdf_files[1..] {
                    if let Some(parent) = pdf_file.parent() {
                        if let Some(ref common) = common_parent {
                            if common.as_os_str() != parent.as_os_str() {
                                // Find the actual common ancestor
                                let mut current = Some(*common);
                                while let Some(cp) = current {
                                    if parent.starts_with(cp) {
                                        common_parent = Some(cp);
                                        break;
                                    }
                                    current = cp.parent();
                                }
                            }
                        }
                    }
                }
                common_parent.map(|p| p.to_path_buf())
            }
        } else {
            None
        };
        
        if let Some(dir) = ingest_dir {
            // Switch to Data tab
            self.ui_state.set_active_tab(Tab::Data);
            self.bump_layout_generation();
            
            // Set the directory in the ingest window and start ingestion
            if let Some(ref mut ingest) = self.ingest_window {
                let dir_str = dir.to_string_lossy().to_string();
                ingest.pdf_dir_input.text = dir_str.clone();
                
                // Start ingestion automatically
                if !ingest.is_ingesting {
                    self.start_ingestion();
                }
                
                self.show_success_toast(format!(
                    "Ingesting {} from {}",
                    if !pdf_files.is_empty() {
                        format!("{} PDF file(s)", pdf_files.len())
                    } else {
                        "directory".to_string()
                    },
                    dir.to_string_lossy()
                ));
            }
        } else {
            // No valid PDF files or directories
            self.show_info_toast("Only PDF files and directories are supported".to_string());
        }
        
        self.file_drag_paths.clear();
    }
    
    /// Handle file hover (files dragged over window)
    pub fn on_file_hover(&mut self, paths: Vec<std::path::PathBuf>, position: Vec2) {
        self.file_drag_active = true;
        self.file_drag_paths = paths;
        self.file_drag_position = position;
        // Visual feedback will be handled in renderer
    }
    
    /// Handle file hover cancelled (drag cancelled)
    pub fn on_file_hover_cancelled(&mut self) {
        self.file_drag_active = false;
        self.file_drag_paths.clear();
    }

    pub fn check_graph_send_responses(&mut self) {
        while let Ok(result) = self.graph_send_receiver.try_recv() {
            self.is_sending_message = false;
            if let Some(ref mut chat) = self.chat_window {
                chat.is_sending = false;
            }
            match result {
                Ok((user_draft, response)) => {
                    let new_leaf_id = response.new_leaf_id.clone();
                    let parent_id = self.graph_state.current_leaf_id.clone();
                    let new_shard = crate::state::GraphShard {
                        id: new_leaf_id.clone(),
                        parent_ids: parent_id.as_ref().map(|p| vec![p.clone()]).unwrap_or_default(),
                        visible: true,
                        user_visible: true,
                        assistant_visible: true,
                        user_content: Some(user_draft),
                        assistant_content: Some(response.response.clone()),
                        contexts: Vec::new(),
                        citations: Vec::new(),
                        notes: Vec::new(),
                        content: None,
                        role: None,
                    };
                    // Place new shard so its top is below the parent's bottom (vertical stack under parent).
                    const SPAWN_GAP: f32 = 24.0;
                    let new_pos = self.graph_state.current_leaf_id
                        .as_ref()
                        .and_then(|id| self.graph_state.get_node(id))
                        .map(|n| glam::Vec2::new(n.position.x, n.position.y + n.size.y + SPAWN_GAP))
                        .unwrap_or(glam::Vec2::ZERO);
                    self.graph_state.add_node(new_shard, new_pos);
                    // New shard added: keep physics running briefly so layout can relax and avoid overlaps.
                    self.physics_settle_frames = 60;
                    self.physics_idle_timer = 0.0;
                    self.graph_state.current_leaf_id = Some(new_leaf_id.clone());
                    if let Some(ref mut chat) = self.chat_window {
                        chat.messages = self.graph_state.node_ids_bfs_order()
                            .into_iter()
                            .filter_map(|id| self.graph_state.get_node(&id))
                            .flat_map(|node| {
                                let id = node.shard.id.clone();
                                let contexts = node.shard.contexts.clone();
                                let mut msgs = Vec::new();
                                if let Some(ref u) = node.shard.user_content {
                                    if !u.is_empty() {
                                        msgs.push(crate::ui::chat_window::ChatMessage {
                                            shard_id: Some(id.clone()),
                                            role: crate::ui::chat_window::MessageRole::User,
                                            content: u.clone(),
                                            contexts: contexts.clone(),
                                            citations: Vec::new(),
                                            notes: Vec::new(),
                                        });
                                    }
                                }
                                if let Some(ref a) = node.shard.assistant_content {
                                    if !a.is_empty() {
                                        msgs.push(crate::ui::chat_window::ChatMessage {
                                            shard_id: Some(id.clone()),
                                            role: crate::ui::chat_window::MessageRole::Assistant,
                                            content: a.clone(),
                                            contexts: contexts.clone(),
                                            citations: Vec::new(),
                                            notes: node.shard.notes.clone(),
                                        });
                                    }
                                }
                                msgs
                            })
                            .collect();
                        self.chat_state.set_current_messages(chat.messages.clone());
                    }
                    self.save_chat_state();
                    let center_on = self.graph_state.get_node(&new_leaf_id).map(|n| n.position + n.size * 0.5);
                    if let (Some(center_pos), Some(ref mut chat)) = (center_on, self.chat_window.as_mut()) {
                        chat.constellation_view.center_camera_on(center_pos);
                        chat.constellation_view.reset_zoom_to_normal();
                    }
                }
                Err(e) => {
                    self.show_error_toast(format!("Send failed: {}", e));
                }
            }
        }
    }

    pub fn check_graph_loaded(&mut self) {
        if let Some(gid) = self.pending_initial_graph_load.take() {
            self.request_graph_load(gid);
        }
        while let Ok(result) = self.graph_loaded_receiver.try_recv() {
            match result {
                Ok((graph_id, response)) => {
                    let shards: std::collections::HashMap<String, crate::state::GraphShard> = response
                        .shards
                        .iter()
                        .map(|(k, v)| (k.clone(), v.into()))
                        .collect();
                    self.graph_state.set_graph(
                        graph_id.clone(),
                        response.root_id,
                        response.current_leaf_id,
                        shards,
                    );
                    // New graph loaded: run physics for a short settling window so branches spread out.
                    self.physics_settle_frames = 90;
                    self.physics_idle_timer = 0.0;
                    if let Some(ref conv_id) = self.chat_state.current_conversation_id {
                        if let Some(conv) = self.chat_state.conversations.iter_mut().find(|c| c.id == *conv_id) {
                            if conv.graph_id.is_none() {
                                conv.graph_id = Some(graph_id);
                            }
                        }
                    }
                    // Rebuild chat.messages from graph so shard_id and order match constellation nodes.
                    if let Some(ref mut chat) = self.chat_window {
                        chat.messages = self.graph_state.node_ids_bfs_order()
                            .into_iter()
                            .filter_map(|id| self.graph_state.get_node(&id))
                            .flat_map(|node| {
                                let id = node.shard.id.clone();
                                let contexts = node.shard.contexts.clone();
                                let mut msgs = Vec::new();
                                if let Some(ref u) = node.shard.user_content {
                                    if !u.is_empty() {
                                        msgs.push(crate::ui::chat_window::ChatMessage {
                                            shard_id: Some(id.clone()),
                                            role: crate::ui::chat_window::MessageRole::User,
                                            content: u.clone(),
                                            contexts: contexts.clone(),
                                            citations: Vec::new(),
                                            notes: Vec::new(),
                                        });
                                    }
                                }
                                if let Some(ref a) = node.shard.assistant_content {
                                    if !a.is_empty() {
                                        msgs.push(crate::ui::chat_window::ChatMessage {
                                            shard_id: Some(id.clone()),
                                            role: crate::ui::chat_window::MessageRole::Assistant,
                                            content: a.clone(),
                                            contexts: contexts.clone(),
                                            citations: Vec::new(),
                                            notes: node.shard.notes.clone(),
                                        });
                                    }
                                }
                                msgs
                            })
                            .collect();
                        self.chat_state.set_current_messages(chat.messages.clone());
                    }
                }
                Err(e) => {
                    self.show_error_toast(format!("Graph load failed: {}", e));
                    // Recover: create a new graph for current conversation so send can work after server restart
                    if let Some(ref conv_id) = self.chat_state.current_conversation_id {
                        if let Some(conv) = self.chat_state.conversations.iter_mut().find(|c| c.id == *conv_id) {
                            conv.graph_id = None;
                        }
                        self.graph_state.clear();
                        self.request_graph_for_new_conversation();
                    }
                }
            }
        }
    }

    pub fn check_api_responses(&mut self) {
        // Check for completed API responses (non-blocking)
        while let Ok(result) = self.api_response_receiver.try_recv() {
            self.is_sending_message = false;
            
            if let Some(ref mut chat) = self.chat_window {
                chat.is_sending = false;
                match result {
                    Ok(chat_response) => {
                        // Add assistant response
                        use crate::ui::chat_window::{ChatMessage, MessageRole};
                        let assistant_msg = ChatMessage::from_legacy(
                            MessageRole::Assistant,
                            chat_response.answer,
                            chat_response.contexts,
                            chat_response.citations.iter().map(|c| {
                                crate::ui::chat_window::Citation {
                                    text: c.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                                    source: c.get("source").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                                    title: c.get("title").and_then(|v| v.as_str()).map(|s| s.to_string()),
                                    year: c.get("year").and_then(|v| v.as_str()).map(|s| s.to_string()),
                                    section: c.get("section").and_then(|v| v.as_str()).map(|s| s.to_string()),
                                    page: c.get("page").and_then(|v| v.as_u64()).map(|p| p as u32),
                                }
                            }).collect(),
                        );
                        chat.add_message(assistant_msg.clone());
                        // Sync with chat_state and auto-save
                        self.chat_state.add_message_to_current(assistant_msg);
                        self.save_chat_state();
                    }
                    Err(error_msg) => {
                        // Add error message
                        use crate::ui::chat_window::{ChatMessage, MessageRole};
                        let error_msg_obj = ChatMessage::from_legacy(
                            MessageRole::Assistant,
                            format!("Error: {}", error_msg),
                            Vec::new(),
                            Vec::new(),
                        );
                        chat.add_message(error_msg_obj.clone());
                        // Sync with chat_state and auto-save
                        self.chat_state.add_message_to_current(error_msg_obj);
                        self.save_chat_state();
                        // Show error toast
                        self.show_error_toast(format!("Chat API error: {}", error_msg));
                    }
                }
            }
        }
    }

    /// Spawn create_graph then get_graph; result is applied in check_graph_loaded. Call after create_conversation.
    pub fn request_graph_for_new_conversation(&self) {
        let base_url = self.api_client.base_url.clone();
        let client = self.api_client.client.clone();
        let sender = self.graph_loaded_sender.clone();
        tokio::spawn(async move {
            let create_url = format!("{}/graph", base_url);
            let create_resp = match client.post(&create_url).send().await {
                Ok(r) => r,
                Err(e) => {
                    let _ = sender.send(Err(format!("Create graph failed: {:?}", e)));
                    return;
                }
            };
            if !create_resp.status().is_success() {
                let err = create_resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                let _ = sender.send(Err(format!("Create graph: {}", err)));
                return;
            }
            let create_body: crate::api::models::CreateGraphResponse = match create_resp.json().await {
                Ok(b) => b,
                Err(e) => {
                    let _ = sender.send(Err(format!("Parse create response: {:?}", e)));
                    return;
                }
            };
            let get_url = format!("{}/graph/{}", base_url, create_body.graph_id);
            let get_resp = match client.get(&get_url).send().await {
                Ok(r) => r,
                Err(e) => {
                    let _ = sender.send(Err(format!("Get graph failed: {:?}", e)));
                    return;
                }
            };
            if !get_resp.status().is_success() {
                let err = get_resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                let _ = sender.send(Err(format!("Get graph: {}", err)));
                return;
            }
            let get_body: crate::api::models::GetGraphResponse = match get_resp.json().await {
                Ok(b) => b,
                Err(e) => {
                    let _ = sender.send(Err(format!("Parse get response: {:?}", e)));
                    return;
                }
            };
            let _ = sender.send(Ok((create_body.graph_id, get_body)));
        });
    }

    /// Spawn get_graph(graph_id); result applied in check_graph_loaded. Call when switching to a conversation that has graph_id.
    pub fn request_graph_load(&self, graph_id: String) {
        let base_url = self.api_client.base_url.clone();
        let client = self.api_client.client.clone();
        let sender = self.graph_loaded_sender.clone();
        tokio::spawn(async move {
            let url = format!("{}/graph/{}", base_url, graph_id);
            let resp = match client.get(&url).send().await {
                Ok(r) => r,
                Err(e) => {
                    let _ = sender.send(Err(format!("Get graph failed: {:?}", e)));
                    return;
                }
            };
            if !resp.status().is_success() {
                let err = resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                let _ = sender.send(Err(format!("Get graph: {}", err)));
                return;
            }
            match resp.json::<crate::api::models::GetGraphResponse>().await {
                Ok(body) => { let _ = sender.send(Ok((graph_id, body))); }
                Err(e) => { let _ = sender.send(Err(format!("Parse: {:?}", e))); }
            }
        });
    }

    /// Save chat state to disk
    fn save_chat_state(&self) {
        if let Err(e) = ConversationPersistence::save_chat_state(&self.chat_state) {
            eprintln!("Failed to save chat state: {}", e);
        }
    }

    /// Show an error toast notification
    fn show_error_toast(&mut self, message: String) {
        self.toast_manager.show(message, crate::ui::toast::ToastType::Error, self.viewport_size);
    }

    /// Show a success toast notification
    fn show_success_toast(&mut self, message: String) {
        self.toast_manager.show(message, crate::ui::toast::ToastType::Success, self.viewport_size);
    }

    /// Show an info toast notification
    fn show_info_toast(&mut self, message: String) {
        self.toast_manager.show(message, crate::ui::toast::ToastType::Info, self.viewport_size);
    }

    pub fn load_collections(&mut self) {
        if self.collections_loaded {
            return; // Already loaded or loading
        }

        let base_url = self.api_client.base_url.clone();
        let client = self.api_client.client.clone();
        let sender = self.collections_sender.clone();

        // Spawn async task to load collections
        tokio::spawn(async move {
            let url = format!("{}/collections", base_url);
            let response = client
                .get(&url)
                .send()
                .await;

            match response {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        match resp.json::<Vec<Collection>>().await {
                            Ok(collections) => {
                                let _ = sender.send(Ok(collections));
                            }
                            Err(e) => {
                                let _ = sender.send(Err(format!("Failed to parse collections: {:?}", e)));
                            }
                        }
                    } else {
                        let error_text = resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                        let _ = sender.send(Err(format!("API error {}: {}", status, error_text)));
                    }
                }
                Err(e) => {
                    let _ = sender.send(Err(format!("API request failed: {:?}", e)));
                }
            }
        });

        self.collections_loaded = true;
    }

    pub fn set_context_pool(&mut self, collection_id: Option<i32>) {
        let base_url = self.api_client.base_url.clone();
        let client = self.api_client.client.clone();
        let sender = self.context_pool_response_sender.clone();

        // Spawn async task to set context pool
        tokio::spawn(async move {
            let request = crate::api::models::ContextPoolRequest {
                collection_id,
                model_id: None,
            };

            let url = format!("{}/context_pool", base_url);
            let response = client
                .post(&url)
                .json(&request)
                .send()
                .await;

            match response {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        let _ = sender.send(Ok(()));
                    } else {
                        let error_text = resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                        let _ = sender.send(Err(format!("API error {}: {}", status, error_text)));
                    }
                }
                Err(e) => {
                    let _ = sender.send(Err(format!("API request failed: {:?}", e)));
                }
            }
        });
    }

    pub fn check_collections_responses(&mut self) {
        // Check for completed collection loading responses (non-blocking)
        while let Ok(result) = self.collections_receiver.try_recv() {
            match result {
                Ok(collections) => {
                    // Update chat window context pool dropdown
                    if let Some(ref mut chat) = self.chat_window {
                        chat.context_pool_dropdown.items.clear();
                        // Add "All papers" option first
                        chat.context_pool_dropdown.items.push(crate::ui::DropdownItem {
                            id: None,
                            label: "All papers".to_string(),
                        });

                        for collection in &collections {
                            chat.context_pool_dropdown.items.push(crate::ui::DropdownItem {
                                id: Some(collection.id),
                                label: collection.name.clone(),
                            });
                        }
                        // Sync selected_index with selected_collection_id after all items are added
                        chat.context_pool_dropdown.set_selected_by_id(chat.selected_collection_id);
                    }
                    
                    // Update library window collections list
                    if let Some(ref mut library) = self.library_window {
                        let library_collections: Vec<crate::ui::library_window::LibraryCollection> = collections.iter().map(|c| {
                            crate::ui::library_window::LibraryCollection {
                                id: c.id,
                                name: c.name.clone(),
                                paper_count: 0, // TODO: Get actual paper count from API
                                papers: Vec::new(),
                            }
                        }).collect();
                        library.set_collections(library_collections);
                    }
                }
                Err(e) => {
                    // Log error but don't show toast (collections are optional, API might not be running)
                    eprintln!("Failed to load collections (this is normal on first startup or if API is not running): {}", e);
                }
            }
        }
    }

    pub fn check_context_pool_responses(&mut self) {
        // Check for completed context pool setting responses (non-blocking)
        while let Ok(result) = self.context_pool_response_receiver.try_recv() {
            match result {
                Ok(_) => {
                    // Context pool set successfully
                    self.show_success_toast("Context pool updated".to_string());
                }
                Err(e) => {
                    eprintln!("Failed to set context pool: {}", e);
                    self.show_error_toast(format!("Failed to set context pool: {}", e));
                }
            }
        }
    }

    pub fn open_file_picker(&mut self) {
        // Open native file picker for PDF files
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("PDF Files", &["pdf"])
            .add_filter("All Files", &["*"])
            .set_title("Select PDF file to ingest")
            .pick_file()
        {
            if let Some(ref mut ingest) = self.ingest_window {
                ingest.pdf_dir_input.text = path.to_string_lossy().to_string();
                // Update cursor position to end of text
                ingest.pdf_dir_input.cursor_position = ingest.pdf_dir_input.text.chars().count();
            }
        }
    }

    pub fn start_ingestion(&mut self) {
        if let Some(ref mut ingest) = self.ingest_window {
            if ingest.is_ingesting {
                return; // Already ingesting
            }

            let pdf_dir = if ingest.pdf_dir_input.text.is_empty() {
                "data/papers".to_string()
            } else {
                ingest.pdf_dir_input.text.clone()
            };

            let base_url = self.api_client.base_url.clone();
            let sender = self.ingest_response_sender.clone();

            ingest.is_ingesting = true;
            ingest.status_text = format!("Starting ingestion from: {}", pdf_dir);
            ingest.progress = 0.0;

            tokio::spawn(async move {
                match crate::api::ApiClient::new(Some(base_url.clone())).ingest_pdfs(&pdf_dir).await {
                    Ok(task_id) => {
                        let _ = sender.send(Ok(task_id));
                    }
                    Err(e) => {
                        let _ = sender.send(Err(format!("Failed to start ingestion: {:?}", e)));
                    }
                }
            });
        }
    }

    pub fn check_ingest_responses(&mut self) {
        while let Ok(result) = self.ingest_response_receiver.try_recv() {
            if let Some(ref mut ingest) = self.ingest_window {
                match result {
                    Ok(task_id) => {
                        self.current_ingest_task_id = Some(task_id.clone());
                        ingest.status_text = format!("Ingestion started. Task ID: {}", task_id);
                        // Start polling task status
                        self.poll_task_status(&task_id);
                    }
                    Err(e) => {
                        ingest.is_ingesting = false;
                        ingest.status_text = format!("Error: {}", e);
                    }
                }
            }
        }
    }

    pub fn poll_task_status(&self, task_id: &str) {
        let base_url = self.api_client.base_url.clone();
        let sender = self.task_status_sender.clone();
        let task_id = task_id.to_string();

        tokio::spawn(async move {
            match crate::api::ApiClient::new(Some(base_url.clone())).get_task_status(&task_id).await {
                Ok(status) => {
                    let _ = sender.send(Ok(status));
                }
                Err(e) => {
                    let _ = sender.send(Err(format!("Failed to get task status: {:?}", e)));
                }
            }
        });
    }

    pub fn check_task_status_responses(&mut self) {
        while let Ok(result) = self.task_status_receiver.try_recv() {
            if let Some(ref mut ingest) = self.ingest_window {
                match result {
                    Ok(status) => {
                        // The API returns "status" not "state"
                        if let Some(state) = status.get("status").and_then(|v| v.as_str()) {
                            match state {
                                "done" | "completed" => {
                                    ingest.is_ingesting = false;
                                    ingest.progress = 1.0;
                                    ingest.status_text = "✅ Ingestion completed successfully!".to_string();
                                    self.current_ingest_task_id = None;
                                    // Reload papers and collections
                                    self.papers_loaded = false;
                                    self.collections_loaded = false;
                                    // Show success toast
                                    self.show_success_toast("PDF ingestion completed!".to_string());
                                }
                                "error" | "failed" => {
                                    ingest.is_ingesting = false;
                                    ingest.progress = 0.0;
                                    // Extract error message from response
                                    let error_msg = status.get("error")
                                        .and_then(|v| v.as_str())
                                        .map(|s| s.to_string())
                                        .unwrap_or_else(|| "Unknown error".to_string());
                                    ingest.status_text = format!("❌ Ingestion failed: {}", error_msg);
                                    self.current_ingest_task_id = None;
                                    // Show error toast
                                    self.show_error_toast(format!("Ingestion failed: {}", error_msg));
                                }
                                "running" => {
                                    // Update progress if available
                                    if let Some(progress_dict) = status.get("progress") {
                                        if let Some(progress_val) = progress_dict.get("progress").and_then(|v| v.as_f64()) {
                                            ingest.progress = progress_val as f32;
                                            ingest.status_text = format!("Ingesting... {:.0}%", progress_val * 100.0);
                                        } else if let Some(current) = progress_dict.get("current").and_then(|v| v.as_u64()) {
                                            if let Some(total) = progress_dict.get("total").and_then(|v| v.as_u64()) {
                                                if total > 0 {
                                                    ingest.progress = (current as f32) / (total as f32);
                                                    ingest.status_text = format!("Ingesting... {} of {} files ({:.0}%)", current, total, ingest.progress * 100.0);
                                                }
                                            }
                                        }
                                    } else {
                                        ingest.status_text = "Ingesting...".to_string();
                                    }
                                    // Continue polling with delay
                                    if let Some(ref task_id) = self.current_ingest_task_id {
                                        let task_id_clone = task_id.clone();
                                        let base_url = self.api_client.base_url.clone();
                                        let sender = self.task_status_sender.clone();
                                        tokio::spawn(async move {
                                            tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
                                            match crate::api::ApiClient::new(Some(base_url.clone())).get_task_status(&task_id_clone).await {
                                                Ok(status) => {
                                                    let _ = sender.send(Ok(status));
                                                }
                                                Err(e) => {
                                                    let _ = sender.send(Err(format!("Failed to get task status: {:?}", e)));
                                                }
                                            }
                                        });
                                    }
                                }
                                "pending" => {
                                    ingest.status_text = "Waiting to start ingestion...".to_string();
                                    // Continue polling with delay
                                    if let Some(ref task_id) = self.current_ingest_task_id {
                                        let task_id_clone = task_id.clone();
                                        let base_url = self.api_client.base_url.clone();
                                        let sender = self.task_status_sender.clone();
                                        tokio::spawn(async move {
                                            tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
                                            match crate::api::ApiClient::new(Some(base_url.clone())).get_task_status(&task_id_clone).await {
                                                Ok(status) => {
                                                    let _ = sender.send(Ok(status));
                                                }
                                                Err(e) => {
                                                    let _ = sender.send(Err(format!("Failed to get task status: {:?}", e)));
                                                }
                                            }
                                        });
                                    }
                                }
                                _ => {
                                    // Unknown status, continue polling
                                    if let Some(ref task_id) = self.current_ingest_task_id {
                                        self.poll_task_status(&task_id);
                                    }
                                }
                            }
                        } else {
                            // No status field, continue polling
                            if let Some(ref task_id) = self.current_ingest_task_id {
                                self.poll_task_status(&task_id);
                            }
                        }
                    }
                    Err(e) => {
                        ingest.is_ingesting = false;
                        ingest.status_text = format!("❌ Error checking status: {}", e);
                        self.current_ingest_task_id = None;
                    }
                }
            }
        }
    }

    pub fn load_insights(&mut self) {
        if self.insights_loaded {
            return;
        }

        let base_url = self.api_client.base_url.clone();
        let client = self.api_client.client.clone();
        let sender = self.insights_sender.clone();

        tokio::spawn(async move {
            match client.get(&format!("{}/shards", base_url)).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        match resp.json::<Vec<crate::api::models::Insight>>().await {
                            Ok(insights) => {
                                let _ = sender.send(Ok(insights));
                            }
                            Err(e) => {
                                let _ = sender.send(Err(format!("Failed to parse insights: {}", e)));
                            }
                        }
                    } else {
                        let error_text = resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                        let _ = sender.send(Err(format!("API error {}: {}", status, error_text)));
                    }
                }
                Err(e) => {
                    let _ = sender.send(Err(format!("Failed to load insights: {}", e)));
                }
            }
        });

        self.insights_loaded = true;
    }

    pub fn check_insights_responses(&mut self) {
        while let Ok(result) = self.insights_receiver.try_recv() {
            match result {
                Ok(insights) => {
                    self.insights_state.set_insights(insights);
                }
                Err(e) => {
                    // Log error but don't show toast (insights are optional, API might not be running)
                    eprintln!("Failed to load insights (this is normal on first startup or if API is not running): {}", e);
                }
            }
        }
    }
    
    pub fn load_pdf(&mut self, filename: &str) {
        if self.pdf_modal.loading {
            return; // Already loading
        }
        
        self.pdf_modal.loading = true;
        let base_url = self.api_client.base_url.clone();
        let client = self.api_client.client.clone();
        let filename_clone = filename.to_string();
        let sender = self.pdf_bytes_sender.clone();
        
        // Spawn async task to load PDF
        tokio::spawn(async move {
            let url = format!("{}/papers/{}", base_url, filename_clone);
            match client.get(&url).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        match resp.bytes().await {
                            Ok(bytes) => {
                                // Send PDF bytes through channel
                                let _ = sender.send(Ok(bytes.to_vec()));
                            }
                            Err(e) => {
                                let _ = sender.send(Err(format!("Failed to read PDF bytes: {}", e)));
                            }
                        }
                    } else {
                        let _ = sender.send(Err(format!("Failed to load PDF: HTTP {}", status)));
                    }
                }
                Err(e) => {
                    let _ = sender.send(Err(format!("Failed to fetch PDF: {}", e)));
                }
            }
        });
    }

    pub fn check_pdf_responses(&mut self) {
        while let Ok(result) = self.pdf_bytes_receiver.try_recv() {
            match result {
                Ok(bytes) => {
                    if let Err(e) = self.pdf_modal.load_pdf(bytes) {
                        eprintln!("Failed to load PDF into renderer: {}", e);
                        self.pdf_modal.set_error(format!("Failed to load PDF: {}", e));
                        self.show_error_toast(format!("Failed to load PDF: {}", e));
                    } else {
                        self.show_success_toast("PDF loaded successfully".to_string());
                    }
                }
                Err(e) => {
                    eprintln!("Failed to load PDF: {}", e);
                    self.pdf_modal.set_error(e.clone());
                    self.show_error_toast(format!("Failed to load PDF: {}", e));
                }
            }
        }
    }

    pub fn check_note_content_responses(&mut self) {
        while let Ok(result) = self.note_content_receiver.try_recv() {
            match result {
                Ok(content) => {
                    if let Some(ref mut notepad) = self.notepad_window {
                        notepad.editor.load_from_markdown(&content);
                        self.show_success_toast("Note loaded into editor".to_string());
                    }
                }
                Err(e) => {
                    self.show_error_toast(format!("Failed to load note: {}", e));
                }
            }
        }
    }

    pub fn load_papers(&mut self) {
        if self.papers_loaded {
            return;
        }

        let base_url = self.api_client.base_url.clone();
        let client = self.api_client.client.clone();
        let sender = self.papers_sender.clone();

        tokio::spawn(async move {
            let url = format!("{}/papers", base_url);
            let response = client
                .get(&url)
                .send()
                .await;

            match response {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        match resp.json::<Vec<crate::api::models::ApiPaper>>().await {
                            Ok(papers) => {
                                let _ = sender.send(Ok(papers));
                            }
                            Err(e) => {
                                let _ = sender.send(Err(format!("Failed to parse papers: {:?}", e)));
                            }
                        }
                    } else {
                        let error_text = resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                        let _ = sender.send(Err(format!("API error {}: {}", status, error_text)));
                    }
                }
                Err(e) => {
                    let _ = sender.send(Err(format!("API request failed: {:?}", e)));
                }
            }
        });

        self.papers_loaded = true;
    }

    pub fn check_papers_responses(&mut self) {
        while let Ok(result) = self.papers_receiver.try_recv() {
            if let Some(ref mut library) = self.library_window {
                match result {
                    Ok(papers) => {
                        let library_papers: Vec<crate::ui::library_window::Paper> = papers.iter().map(|p| {
                            crate::ui::library_window::Paper {
                                id: p.id,
                                filename: p.filename.clone(),
                                title: p.title.clone(),
                                authors: None,
                                year: p.year,
                            }
                        }).collect();
                        library.set_papers(library_papers);
                    }
                    Err(e) => {
                        eprintln!("Failed to load papers: {}", e);
                    }
                }
            }
        }
    }
    
    // Clipboard helper methods
    fn save_text_state_for_undo(&mut self, current_text: &str) {
        // Save current state to undo history
        self.undo_history.push(current_text.to_string());
        // Clear redo history when new action is performed
        self.redo_history.clear();
        // Limit undo history size
        if self.undo_history.len() > 50 {
            self.undo_history.remove(0);
        }
    }
    
    fn undo_text(&mut self, input: &mut crate::ui::text_input::TextInput) {
        let current_text = input.text.clone();
        let previous_text = self.undo_history.pop();
        if let Some(previous_text) = previous_text {
            // Save current state to redo history
            self.redo_history.push(current_text);
            // Restore previous state
            input.text = previous_text;
            input.cursor_position = input.text.chars().count();
            input.clear_selection();
            input.ensure_cursor_valid();
        }
    }
    
    fn redo_text(&mut self, input: &mut crate::ui::text_input::TextInput) {
        let current_text = input.text.clone();
        let next_text = self.redo_history.pop();
        if let Some(next_text) = next_text {
            // Save current state to undo history
            self.undo_history.push(current_text);
            // Restore next state
            input.text = next_text;
            input.cursor_position = input.text.chars().count();
            input.clear_selection();
            input.ensure_cursor_valid();
        }
    }
    
    fn copy_text(&mut self, input: &crate::ui::text_input::TextInput) {
        let selected = input.get_selected_text();
        if !selected.is_empty() {
            self.clipboard_text = selected;
        }
    }
    
    fn cut_text(&mut self, input: &mut crate::ui::text_input::TextInput) {
        let selected = input.get_selected_text();
        let current_text = input.text.clone();
        if !selected.is_empty() {
            self.clipboard_text = selected;
            input.delete_selection();
            // Save state after deletion
            self.save_text_state_for_undo(&current_text);
        }
    }
    
    fn paste_text(&mut self, input: &mut crate::ui::text_input::TextInput) {
        let clipboard = self.clipboard_text.clone();
        let current_text = input.text.clone();
        if !clipboard.is_empty() {
            input.paste(&clipboard);
            // Save state after paste
            self.save_text_state_for_undo(&current_text);
        }
    }

}


