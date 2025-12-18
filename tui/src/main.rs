use anyhow::Result;
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
        MouseEventKind,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::{
    io,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::Mutex;

mod logo;
mod stream;
mod types;
mod widgets;

use logo::ELEPHANT;
use stream::MemoryStream;
use types::{AppState, SearchMode, ViewMode};
use widgets::{render_footer, render_header, render_main};

struct UserSelector {
    users: Vec<String>,
    selected: usize,
    loading: bool,
    error: Option<String>,
}

impl UserSelector {
    fn new() -> Self {
        Self {
            users: vec![],
            selected: 0,
            loading: true,
            error: None,
        }
    }
    fn select_next(&mut self) {
        if !self.users.is_empty() {
            self.selected = (self.selected + 1) % self.users.len();
        }
    }
    fn select_prev(&mut self) {
        if !self.users.is_empty() {
            self.selected = self.selected.checked_sub(1).unwrap_or(self.users.len() - 1);
        }
    }
    fn selected_user(&self) -> Option<&String> {
        self.users.get(self.selected)
    }
}

async fn fetch_users(base_url: &str, api_key: &str) -> Result<Vec<String>, String> {
    let client = reqwest::Client::new();
    let url = format!("{}/api/users", base_url);
    match client.get(&url).header("X-API-Key", api_key).send().await {
        Ok(resp) => resp
            .json::<Vec<String>>()
            .await
            .map_err(|e| format!("Parse: {}", e)),
        Err(e) => Err(format!("Connection: {}", e)),
    }
}

fn render_user_selector(f: &mut Frame, selector: &UserSelector) {
    let area = f.area();
    f.render_widget(
        Block::default().style(Style::default().bg(Color::Black)),
        area,
    );
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(3),
        ])
        .margin(2)
        .split(area);

    let logo_lines: Vec<Line> = ELEPHANT
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let g = [
                (255, 180, 50),
                (255, 160, 40),
                (255, 140, 30),
                (255, 120, 20),
                (255, 100, 10),
                (255, 80, 0),
            ];
            Line::from(Span::styled(
                *l,
                Style::default().fg(Color::Rgb(g[i % 6].0, g[i % 6].1, g[i % 6].2)),
            ))
        })
        .collect();
    f.render_widget(
        Paragraph::new(logo_lines).alignment(Alignment::Center),
        chunks[0],
    );

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "SELECT USER",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )))
        .alignment(Alignment::Center),
        chunks[1],
    );

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(" Users ", Style::default().fg(Color::Cyan)));
    if selector.loading {
        f.render_widget(
            Paragraph::new("Loading...")
                .fg(Color::Yellow)
                .alignment(Alignment::Center)
                .block(block),
            chunks[2],
        );
    } else if let Some(ref e) = selector.error {
        f.render_widget(
            Paragraph::new(e.as_str())
                .fg(Color::Red)
                .alignment(Alignment::Center)
                .block(block),
            chunks[2],
        );
    } else if selector.users.is_empty() {
        f.render_widget(
            Paragraph::new("No users found.")
                .fg(Color::DarkGray)
                .alignment(Alignment::Center)
                .block(block),
            chunks[2],
        );
    } else {
        let items: Vec<ListItem> = selector
            .users
            .iter()
            .enumerate()
            .map(|(i, u)| {
                let s = i == selector.selected;
                let st = if s {
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::White)
                };
                ListItem::new(Line::from(vec![
                    Span::styled(if s { "> " } else { "  " }, st),
                    Span::styled(u, st),
                ]))
            })
            .collect();
        f.render_widget(List::new(items).block(block), chunks[2]);
    }

    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                " j/k ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("select ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                " Enter ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("confirm ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                " q ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("quit", Style::default().fg(Color::DarkGray)),
        ]))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        ),
        chunks[3],
    );
}

async fn run_user_selector(base_url: &str, api_key: &str) -> Result<Option<String>> {
    enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    let mut selector = UserSelector::new();
    match fetch_users(base_url, api_key).await {
        Ok(u) => {
            selector.users = u;
            selector.loading = false;
        }
        Err(e) => {
            selector.error = Some(e);
            selector.loading = false;
        }
    }
    let result = loop {
        terminal.draw(|f| render_user_selector(f, &selector))?;
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break None,
                        KeyCode::Up | KeyCode::Char('k') => selector.select_prev(),
                        KeyCode::Down | KeyCode::Char('j') => selector.select_next(),
                        KeyCode::Enter => {
                            if let Some(u) = selector.selected_user() {
                                break Some(u.clone());
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    };
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<()> {
    let base_url = std::env::var("SHODH_SERVER_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:3030".to_string())
        .trim_end_matches("/api/events")
        .to_string();
    let api_key = std::env::var("SHODH_API_KEY")
        .unwrap_or_else(|_| "sk-shodh-dev-local-testing-key".to_string());

    let user = match run_user_selector(&base_url, &api_key).await? {
        Some(u) => u,
        None => return Ok(()),
    };

    let state = Arc::new(Mutex::new(AppState::new()));
    {
        let mut s = state.lock().await;
        s.current_user = user.clone();
    }

    let stream = MemoryStream::new(
        &format!("{}/api/events", base_url),
        &api_key,
        &user,
        Arc::clone(&state),
    );
    let h = tokio::spawn(async move {
        stream.run().await;
    });
    let r = run_tui(state).await;
    h.abort();
    r
}

async fn run_tui(state: Arc<Mutex<AppState>>) -> Result<()> {
    enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    let tick_rate = Duration::from_millis(100);
    let mut last_tick = Instant::now();

    // Clone state for search API calls
    let search_state = Arc::clone(&state);
    let base_url = std::env::var("SHODH_SERVER_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:3030".to_string())
        .trim_end_matches("/api/events")
        .to_string();
    let api_key = std::env::var("SHODH_API_KEY")
        .unwrap_or_else(|_| "sk-shodh-dev-local-testing-key".to_string());

    loop {
        {
            let g = state.lock().await;
            terminal.draw(|f| ui(f, &g))?;
        }
        if crossterm::event::poll(tick_rate.saturating_sub(last_tick.elapsed()))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    let mut g = state.lock().await;

                    // Handle search mode input
                    if g.search_active {
                        match key.code {
                            KeyCode::Esc => {
                                g.cancel_search();
                            }
                            KeyCode::Enter => {
                                if !g.search_query.is_empty() {
                                    let query = g.search_query.clone();
                                    let mode = g.search_mode;
                                    let user_id = g.current_user.clone();
                                    g.search_loading = true;
                                    drop(g);

                                    // Execute search API call
                                    let results =
                                        execute_search(&base_url, &api_key, &user_id, &query, mode)
                                            .await;

                                    let mut g = search_state.lock().await;
                                    match results {
                                        Ok(r) => g.set_search_results(r),
                                        Err(e) => {
                                            g.set_error(format!("Search failed: {}", e));
                                            g.search_loading = false;
                                        }
                                    }
                                }
                            }
                            KeyCode::Tab => {
                                g.cycle_search_mode();
                            }
                            KeyCode::Backspace => {
                                g.search_query.pop();
                                g.schedule_search();
                            }
                            KeyCode::Up => {
                                if g.search_results_visible {
                                    g.search_select_prev();
                                }
                            }
                            KeyCode::Down => {
                                if g.search_results_visible {
                                    g.search_select_next();
                                }
                            }
                            KeyCode::Char(c) => {
                                if g.search_query.len() < 100 {
                                    g.search_query.push(c);
                                    g.schedule_search();
                                }
                            }
                            _ => {}
                        }
                        continue;
                    }

                    // Handle search detail view
                    if g.search_detail_visible {
                        match key.code {
                            KeyCode::Esc | KeyCode::Backspace => {
                                g.search_detail_visible = false;
                            }
                            _ => {}
                        }
                        continue;
                    }

                    // Handle search results navigation
                    if g.search_results_visible {
                        match key.code {
                            KeyCode::Esc => {
                                g.search_results_visible = false;
                                g.search_results.clear();
                                g.search_active = false;
                            }
                            KeyCode::Enter => {
                                if !g.search_results.is_empty() {
                                    g.search_detail_visible = true;
                                }
                            }
                            KeyCode::Up | KeyCode::Char('k') => {
                                g.search_select_prev();
                            }
                            KeyCode::Down | KeyCode::Char('j') => {
                                g.search_select_next();
                            }
                            KeyCode::Char('/') => {
                                g.start_search();
                            }
                            _ => {}
                        }
                        continue;
                    }

                    // Normal mode keybindings
                    match key.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Esc => {
                            if g.selected_event.is_some() {
                                g.clear_event_selection();
                            } else {
                                break;
                            }
                        }
                        KeyCode::Char('/') => {
                            g.start_search();
                        }
                        KeyCode::Char('c') => g.events.clear(),
                        KeyCode::Char('d') => g.set_view(ViewMode::Dashboard),
                        KeyCode::Char('a') => g.set_view(ViewMode::ActivityLogs),
                        KeyCode::Char('g') => g.set_view(ViewMode::GraphList),
                        KeyCode::Char('m') => g.set_view(ViewMode::GraphMap),
                        KeyCode::Char('+') | KeyCode::Char('=') => {
                            g.zoom_in();
                        }
                        KeyCode::Char('-') => {
                            g.zoom_out();
                        }
                        KeyCode::Tab => g.cycle_view(),
                        KeyCode::Up | KeyCode::Char('k') => match g.view_mode {
                            ViewMode::Dashboard | ViewMode::ActivityLogs => g.select_event_prev(),
                            _ => g.scroll_up(),
                        },
                        KeyCode::Down | KeyCode::Char('j') => match g.view_mode {
                            ViewMode::Dashboard | ViewMode::ActivityLogs => g.select_event_next(),
                            _ => g.scroll_down(),
                        },
                        KeyCode::Enter => {
                            if g.selected_event.is_none() && !g.events.is_empty() {
                                g.selected_event = Some(0);
                            }
                        }
                        KeyCode::Backspace => g.clear_event_selection(),
                        KeyCode::PageUp => {
                            for _ in 0..5 {
                                match g.view_mode {
                                    ViewMode::Dashboard | ViewMode::ActivityLogs => {
                                        g.select_event_prev()
                                    }
                                    _ => g.scroll_up(),
                                }
                            }
                        }
                        KeyCode::PageDown => {
                            for _ in 0..5 {
                                match g.view_mode {
                                    ViewMode::Dashboard | ViewMode::ActivityLogs => {
                                        g.select_event_next()
                                    }
                                    _ => g.scroll_down(),
                                }
                            }
                        }
                        KeyCode::Home => {
                            g.scroll_offset = 0;
                            if matches!(g.view_mode, ViewMode::Dashboard | ViewMode::ActivityLogs)
                                && !g.events.is_empty()
                            {
                                g.selected_event = Some(0);
                            }
                        }
                        _ => {}
                    }
                }
                Event::Mouse(m) => {
                    let mut g = state.lock().await;
                    match m.kind {
                        MouseEventKind::ScrollUp => {
                            if m.modifiers.contains(KeyModifiers::CONTROL) {
                                g.zoom_in();
                            } else {
                                g.scroll_up();
                            }
                        }
                        MouseEventKind::ScrollDown => {
                            if m.modifiers.contains(KeyModifiers::CONTROL) {
                                g.zoom_out();
                            } else {
                                g.scroll_down();
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        if last_tick.elapsed() >= tick_rate {
            let mut g = state.lock().await;
            g.tick();

            // Check if debounced search should execute
            if g.should_execute_search() {
                let query = g.search_query.clone();
                let mode = g.search_mode;
                let user_id = g.current_user.clone();
                g.mark_search_started();
                drop(g);

                // Execute search in background
                let results = execute_search(&base_url, &api_key, &user_id, &query, mode).await;

                let mut g = search_state.lock().await;
                match results {
                    Ok(r) => {
                        g.set_search_results(r);
                        g.search_results_visible = true;
                    }
                    Err(e) => {
                        g.set_error(format!("Search: {}", e));
                        g.search_loading = false;
                    }
                }
            }

            last_tick = Instant::now();
        }
    }

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    Ok(())
}

async fn execute_search(
    base_url: &str,
    api_key: &str,
    user_id: &str,
    query: &str,
    mode: SearchMode,
) -> Result<Vec<types::SearchResult>, String> {
    let client = reqwest::Client::new();

    let url = match mode {
        SearchMode::Keyword => format!("{}/api/list/{}?query={}", base_url, user_id, query),
        SearchMode::Semantic => format!("{}/api/recall", base_url),
        SearchMode::Date => format!("{}/api/recall/date", base_url),
    };

    match mode {
        SearchMode::Keyword => {
            // GET request for keyword search
            let resp = client
                .get(&url)
                .header("X-API-Key", api_key)
                .send()
                .await
                .map_err(|e| e.to_string())?;

            let data: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
            parse_memory_list_response(data)
        }
        SearchMode::Semantic => {
            // POST request for semantic search
            let body = serde_json::json!({
                "user_id": user_id,
                "query": query,
                "mode": "semantic",
                "limit": 20
            });

            let resp = client
                .post(&url)
                .header("X-API-Key", api_key)
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| e.to_string())?;

            let data: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
            parse_recall_response(data)
        }
        SearchMode::Date => {
            // POST request for date range search
            let body = serde_json::json!({
                "user_id": user_id,
                "start": query,
                "limit": 20
            });

            let resp = client
                .post(&url)
                .header("X-API-Key", api_key)
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| e.to_string())?;

            let data: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
            parse_recall_response(data)
        }
    }
}

fn parse_memory_list_response(data: serde_json::Value) -> Result<Vec<types::SearchResult>, String> {
    let memories = data
        .get("memories")
        .and_then(|m| m.as_array())
        .ok_or("Invalid response format")?;

    let mut results = Vec::new();
    for mem in memories {
        let id = mem
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let content = mem
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let memory_type = mem
            .get("memory_type")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown")
            .to_string();
        let created_at = mem
            .get("created_at")
            .and_then(|v| v.as_str())
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(chrono::Utc::now);
        let tags = mem
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        results.push(types::SearchResult {
            id,
            content,
            memory_type,
            score: 1.0,
            created_at,
            tags,
        });
    }
    Ok(results)
}

fn parse_recall_response(data: serde_json::Value) -> Result<Vec<types::SearchResult>, String> {
    let memories = data
        .get("memories")
        .and_then(|m| m.as_array())
        .ok_or("Invalid response format")?;

    let mut results = Vec::new();
    for mem in memories {
        let id = mem
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let content = mem
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let memory_type = mem
            .get("memory_type")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown")
            .to_string();
        let score = mem
            .get("score")
            .and_then(|v| v.as_f64())
            .map(|f| f as f32)
            .unwrap_or(0.0);
        let created_at = mem
            .get("created_at")
            .and_then(|v| v.as_str())
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(chrono::Utc::now);
        let tags = mem
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        results.push(types::SearchResult {
            id,
            content,
            memory_type,
            score,
            created_at,
            tags,
        });
    }
    Ok(results)
}

fn ui(f: &mut Frame, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(9),
            Constraint::Min(10),
            Constraint::Length(3),
        ])
        .split(f.area());
    render_header(f, chunks[0], state);
    render_main(f, chunks[1], state);
    render_footer(f, chunks[2], state);
}
