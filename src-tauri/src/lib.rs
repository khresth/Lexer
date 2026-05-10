mod indexer;
use indexer::{index_workspace, search_index, clear_index};

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[tauri::command]
async fn ai_chat(
    url: String,
    api_key: String,
    model: String,
    messages: Vec<serde_json::Value>,
) -> Result<String, String> {
    let client = reqwest::Client::new();
    let mut request = client.post(&url)
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": false
        }));

    if !api_key.is_empty() {
        request = request.header("Authorization", format!("Bearer {}", api_key));
    }

    let response = request.send().await.map_err(|e| e.to_string())?;
    let text = response.text().await.map_err(|e| e.to_string())?;
    Ok(text)
}

use std::collections::HashMap;
use std::sync::Arc;
use tauri::State;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::Mutex;

pub struct TerminalSession {
    stdin: tokio::process::ChildStdin,
}

#[derive(Clone)]
pub enum TerminalEvent {
    Stdout(String),
    Stderr(String),
}

pub struct TerminalManager {
    sessions: Mutex<HashMap<String, TerminalSession>>,
    outputs: Mutex<HashMap<String, Vec<TerminalEvent>>>,
}

impl TerminalManager {
    fn new() -> Self {
        TerminalManager {
            sessions: Mutex::new(HashMap::new()),
            outputs: Mutex::new(HashMap::new()),
        }
    }
}

async fn store_output(terminal_manager: &TerminalManager, id: String, event: TerminalEvent) {
    let mut outputs = terminal_manager.outputs.lock().await;
    if let Some(vec) = outputs.get_mut(&id) {
        vec.push(event);
    }
}

#[tauri::command]
async fn spawn_terminal(
    cwd: String,
    terminal_manager: State<'_, Arc<TerminalManager>>,
) -> Result<String, String> {
    let working_dir = if cwd.is_empty() {
        dirs::home_dir()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_else(|| ".".to_string())
    } else {
        cwd
    };

    // Spawn PowerShell
    let mut child = Command::new("powershell.exe")
        .arg("-NoExit")
        .arg("-Command")
        .arg("$Host.UI.RawUI.WindowTitle = 'Lexer Terminal'")
        .current_dir(&working_dir)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn terminal: {}", e))?;

    let stdin = child
        .stdin
        .take()
        .ok_or("Failed to get stdin")?;
    let stdout = child
        .stdout
        .take()
        .ok_or("Failed to get stdout")?;
    let stderr = child
        .stderr
        .take()
        .ok_or("Failed to get stderr")?;

    let terminal_id = uuid::Uuid::new_v4().to_string();

    let session = TerminalSession { stdin };

    {
        let mut sessions = terminal_manager.sessions.lock().await;
        sessions.insert(terminal_id.clone(), session);
    }

    {
        let mut outputs = terminal_manager.outputs.lock().await;
        outputs.insert(terminal_id.clone(), Vec::new());
    }

    // Spawn output readers
    let tm_stdout = Arc::clone(terminal_manager.inner());
    let tm_stderr = Arc::clone(terminal_manager.inner());
    let id_stdout = terminal_id.clone();
    let id_stderr = terminal_id.clone();

    tokio::spawn(async move {
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            store_output(&tm_stdout, id_stdout.clone(), TerminalEvent::Stdout(line)).await;
        }
    });

    tokio::spawn(async move {
        let reader = BufReader::new(stderr);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            store_output(&tm_stderr, id_stderr.clone(), TerminalEvent::Stderr(line)).await;
        }
    });

    Ok(terminal_id)
}

#[tauri::command]
async fn write_terminal_input(
    terminal_id: String,
    input: String,
    terminal_manager: State<'_, Arc<TerminalManager>>,
) -> Result<(), String> {
    let mut sessions = terminal_manager.sessions.lock().await;
    if let Some(session) = sessions.get_mut(&terminal_id) {
        let input_with_newline = format!("{}\r\n", input);
        session
            .stdin
            .write_all(input_with_newline.as_bytes())
            .await
            .map_err(|e| format!("Failed to write to terminal: {}", e))?;
        session
            .stdin
            .flush()
            .await
            .map_err(|e| format!("Failed to flush: {}", e))?;
    }
    Ok(())
}

#[tauri::command]
async fn read_terminal_output(
    terminal_id: String,
    terminal_manager: State<'_, Arc<TerminalManager>>,
) -> Result<Vec<(String, String)>, String> {
    let mut outputs = terminal_manager.outputs.lock().await;
    if let Some(events) = outputs.get_mut(&terminal_id) {
        let result: Vec<(String, String)> = events
            .drain(..)
            .map(|e| match e {
                TerminalEvent::Stdout(s) => ("stdout".to_string(), s),
                TerminalEvent::Stderr(s) => ("stderr".to_string(), s),
            })
            .collect();
        Ok(result)
    } else {
        Ok(Vec::new())
    }
}

#[tauri::command]
async fn kill_terminal(
    terminal_id: String,
    terminal_manager: State<'_, Arc<TerminalManager>>,
) -> Result<(), String> {
    let mut sessions = terminal_manager.sessions.lock().await;
    if let Some(_session) = sessions.remove(&terminal_id) {
        // Session dropped, process will be killed
    }
    let mut outputs = terminal_manager.outputs.lock().await;
    outputs.remove(&terminal_id);
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_shell::init())
        .manage(Arc::new(TerminalManager::new()))
        .invoke_handler(tauri::generate_handler![
            greet, ai_chat, 
            spawn_terminal, write_terminal_input, read_terminal_output, kill_terminal,
            index_workspace, search_index, clear_index
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
