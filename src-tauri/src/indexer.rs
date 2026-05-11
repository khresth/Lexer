use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::path::Path;

const CHUNK_SIZE: usize = 500;
const CHUNK_OVERLAP: usize = 100;
const MAX_FILE_SIZE: u64 = 100 * 1024; // 100KB

#[derive(Serialize)]
pub struct IndexProgress {
    pub indexed: usize,
    pub total: usize,
}

#[derive(Serialize)]
pub struct SearchResult {
    pub file_path: String,
    pub content: String,
    pub similarity_score: f32,
}

#[derive(Clone, Debug)]
struct EmbeddingConfig {
    provider: String,
    api_key: String,
    base_url: String,
    model: String,
}

impl EmbeddingConfig {
    fn normalize(
        provider: Option<String>,
        api_key: String,
        base_url: Option<String>,
        model: Option<String>,
    ) -> Self {
        let provider = provider.unwrap_or_else(|| "nim".to_string()).to_lowercase();
        match provider.as_str() {
            "ollama" => Self {
                provider,
                api_key: String::new(),
                base_url: base_url.unwrap_or_else(|| "http://localhost:11434".to_string()),
                model: model.unwrap_or_else(|| "nomic-embed-text".to_string()),
            },
            "groq" => Self {
                provider,
                api_key,
                base_url: base_url.unwrap_or_else(|| "https://api.groq.com/openai/v1".to_string()),
                model: model.unwrap_or_else(|| "llama-3.1-8b-instant".to_string()),
            },
            "custom" => Self {
                provider,
                api_key,
                base_url: base_url.unwrap_or_default(),
                model: model.unwrap_or_default(),
            },
            _ => Self {
                provider: "nim".to_string(),
                api_key,
                base_url: base_url.unwrap_or_else(|| "https://integrate.api.nvidia.com/v1".to_string()),
                model: model.unwrap_or_else(|| "nvidia/nv-embedqa-e5-v5".to_string()),
            },
        }
    }
}

fn get_db_path(folder_path: &str) -> std::path::PathBuf {
    let folder_name = Path::new(folder_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("project");
    let db_name = format!("{}-lexer.db", folder_name);
    Path::new(folder_path).join(db_name)
}

fn init_db(conn: &mut Connection) -> Result<(), String> {
    // Create chunks table with embedding column
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            embedding TEXT,
            UNIQUE(file_path, chunk_index)
        )",
        [],
    ).map_err(|e| e.to_string())?;

    // Add embedding column if it doesn't exist (for backward compatibility)
    // SQLite will return error if column already exists, which we ignore
    let _ = conn.execute("ALTER TABLE chunks ADD COLUMN embedding TEXT", []);
    conn.execute(
        "CREATE TABLE IF NOT EXISTS index_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )",
        [],
    ).map_err(|e| e.to_string())?;

    Ok(())
}

fn save_embedding_metadata(conn: &Connection, config: &EmbeddingConfig) -> Result<(), String> {
    conn.execute(
        "INSERT INTO index_metadata (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params!["embedding_provider", config.provider],
    ).map_err(|e| e.to_string())?;
    conn.execute(
        "INSERT INTO index_metadata (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params!["embedding_base_url", config.base_url],
    ).map_err(|e| e.to_string())?;
    conn.execute(
        "INSERT INTO index_metadata (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params!["embedding_model", config.model],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

fn load_embedding_metadata(conn: &Connection) -> EmbeddingConfig {
    let provider = conn.query_row(
        "SELECT value FROM index_metadata WHERE key = 'embedding_provider'",
        [],
        |row| row.get::<_, String>(0),
    ).ok();
    let base_url = conn.query_row(
        "SELECT value FROM index_metadata WHERE key = 'embedding_base_url'",
        [],
        |row| row.get::<_, String>(0),
    ).ok();
    let model = conn.query_row(
        "SELECT value FROM index_metadata WHERE key = 'embedding_model'",
        [],
        |row| row.get::<_, String>(0),
    ).ok();
    EmbeddingConfig::normalize(provider, String::new(), base_url, model)
}

fn is_binary_file(path: &Path) -> bool {
    if let Ok(contents) = fs::read(path) {
        // Check first 1024 bytes for null bytes
        let sample = &contents[..contents.len().min(1024)];
        return sample.contains(&0);
    }
    true
}

fn should_skip_folder(name: &str) -> bool {
    let skip_folders: HashSet<&str> = [
        "node_modules", ".git", "target", "dist", 
        ".next", "__pycache__"
    ].iter().cloned().collect();
    skip_folders.contains(name)
}

fn chunk_file(content: &str) -> Vec<(String, usize, usize)> {
    let lines: Vec<&str> = content.lines().collect();
    let mut chunks = Vec::new();
    let mut start_line = 0;

    while start_line < lines.len() {
        let mut chunk_lines = Vec::new();
        let mut char_count = 0;
        let mut end_line = start_line;

        while end_line < lines.len() && char_count < CHUNK_SIZE {
            let line = lines[end_line];
            char_count += line.len() + 1; // +1 for newline
            chunk_lines.push(line);
            end_line += 1;
        }

        if !chunk_lines.is_empty() {
            let chunk_content = chunk_lines.join("\n");
            chunks.push((chunk_content, start_line + 1, end_line));
            
            // Move forward with overlap
            let advance = (end_line - start_line).saturating_sub(CHUNK_OVERLAP / 50);
            start_line += advance.max(1);
        } else {
            break;
        }
    }

    chunks
}

async fn generate_embedding(text: &str, config: &EmbeddingConfig, input_type: &str) -> Result<Vec<f32>, String> {
    let client = reqwest::Client::new();
    let base = config.base_url.trim_end_matches('/');

    match config.provider.as_str() {
        "ollama" => {
            let response = client
                .post(format!("{}/api/embeddings", base))
                .header("Content-Type", "application/json")
                .json(&serde_json::json!({
                    "model": config.model,
                    "prompt": text,
                }))
                .send()
                .await
                .map_err(|e| format!("Ollama embedding request failed: {}", e))?;

            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                return Err(format!("Ollama embedding API error {}: {}", status, text));
            }

            let result: Value = response
                .json()
                .await
                .map_err(|e| format!("Failed to parse Ollama embedding response: {}", e))?;

            let embedding = result
                .get("embedding")
                .and_then(|v| v.as_array())
                .ok_or_else(|| "Ollama embedding missing 'embedding' array".to_string())?
                .iter()
                .map(|v| v.as_f64().ok_or_else(|| "Invalid Ollama embedding value".to_string()).map(|x| x as f32))
                .collect::<Result<Vec<f32>, String>>()?;
            Ok(embedding)
        }
        "groq" => {
            let prompt = format!(
                "Generate a JSON array of 128 floating point numbers between -1 and 1 that semantically represents this text. Return ONLY the JSON array, no explanation:\n{}",
                text
            );
            let mut request = client
                .post(format!("{}/chat/completions", base))
                .header("Content-Type", "application/json")
                .json(&serde_json::json!({
                    "model": config.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "stream": false
                }));
            if !config.api_key.is_empty() {
                request = request.header("Authorization", format!("Bearer {}", config.api_key));
            }
            let response = request
                .send()
                .await
                .map_err(|e| format!("Groq pseudo-embedding request failed: {}", e))?;

            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                return Err(format!("Groq pseudo-embedding API error {}: {}", status, text));
            }

            let result: Value = response
                .json()
                .await
                .map_err(|e| format!("Failed to parse Groq response: {}", e))?;
            let content = result
                .get("choices")
                .and_then(|c| c.as_array())
                .and_then(|arr| arr.first())
                .and_then(|c| c.get("message"))
                .and_then(|m| m.get("content"))
                .and_then(|s| s.as_str())
                .ok_or_else(|| "Missing Groq completion content".to_string())?;

            let start = content.find('[').ok_or_else(|| "Groq response missing JSON array".to_string())?;
            let end = content.rfind(']').ok_or_else(|| "Groq response missing array end".to_string())?;
            let json_slice = &content[start..=end];
            let raw: Vec<f64> = serde_json::from_str(json_slice)
                .map_err(|e| format!("Failed to parse Groq pseudo-embedding array: {}", e))?;
            Ok(raw.into_iter().map(|v| v as f32).collect())
        }
        "custom" => {
            let mut request = client
                .post(format!("{}/embeddings", base))
                .header("Content-Type", "application/json")
                .json(&serde_json::json!({
                    "input": text,
                    "model": config.model,
                    "encoding_format": "float"
                }));
            if !config.api_key.is_empty() {
                request = request.header("Authorization", format!("Bearer {}", config.api_key));
            }
            parse_openai_embedding(request.send().await.map_err(|e| format!("Custom embedding request failed: {}", e))?).await
        }
        _ => {
            let mut request = client
                .post(format!("{}/embeddings", base))
                .header("Content-Type", "application/json")
                .json(&serde_json::json!({
                    "input": text,
                    "model": config.model,
                    "encoding_format": "float",
                    "input_type": input_type
                }));
            if !config.api_key.is_empty() {
                request = request.header("Authorization", format!("Bearer {}", config.api_key));
            }
            parse_openai_embedding(request.send().await.map_err(|e| format!("NIM embedding request failed: {}", e))?).await
        }
    }
}

async fn parse_openai_embedding(response: reqwest::Response) -> Result<Vec<f32>, String> {
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(format!("Embedding API error {}: {}", status, text));
    }

    let result: Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse embedding response: {}", e))?;
    let embedding = result
        .get("data")
        .and_then(|d| d.as_array())
        .and_then(|arr| arr.first())
        .and_then(|x| x.get("embedding"))
        .and_then(|v| v.as_array())
        .ok_or_else(|| "No embedding returned".to_string())?
        .iter()
        .map(|v| v.as_f64().ok_or_else(|| "Invalid embedding value".to_string()).map(|x| x as f32))
        .collect::<Result<Vec<f32>, String>>()?;
    Ok(embedding)
}

#[tauri::command]
pub async fn index_workspace(
    folder_path: String,
    api_key: String,
    provider: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
) -> Result<IndexProgress, String> {
    let db_path = get_db_path(&folder_path);

    let mut conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    init_db(&mut conn)?;
    let config = EmbeddingConfig::normalize(provider, api_key, base_url, model);
    save_embedding_metadata(&conn, &config)?;

    // Collect all files to index
    let mut files_to_index = Vec::new();
    
    fn collect_files(dir: &Path, files: &mut Vec<std::path::PathBuf>) -> Result<(), String> {
        for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();

            if path.is_dir() {
                if !should_skip_folder(&name) {
                    collect_files(&path, files)?;
                }
            } else {
                // Check file size
                if let Ok(metadata) = entry.metadata() {
                    if metadata.len() <= MAX_FILE_SIZE && !is_binary_file(&path) {
                        files.push(path);
                    }
                }
            }
        }
        Ok(())
    }

    collect_files(Path::new(&folder_path), &mut files_to_index)?;

    let total = files_to_index.len();
    let mut indexed = 0;

    // Clear existing index for this workspace
    conn.execute("DELETE FROM chunks", []).map_err(|e| e.to_string())?;

    for file_path in files_to_index {
        let relative_path = file_path
            .strip_prefix(&folder_path)
            .unwrap_or(&file_path)
            .to_string_lossy()
            .to_string();

        eprintln!("Indexing file {}/{}: {}", indexed + 1, total, relative_path);

        if let Ok(content) = fs::read_to_string(&file_path) {
            let chunks = chunk_file(&content);

            for (chunk_index, (chunk_content, start_line, end_line)) in chunks.iter().enumerate() {
                // Generate embedding
                match generate_embedding(chunk_content, &config, "passage").await {
                    Ok(embedding) => {
                        // Insert chunk with embedding
                        let embedding_json = serde_json::to_string(&embedding).map_err(|e| e.to_string())?;
                        conn.execute(
                            "INSERT INTO chunks (file_path, chunk_index, content, start_line, end_line, embedding)
                             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                             ON CONFLICT(file_path, chunk_index) DO UPDATE SET
                             content = excluded.content, start_line = excluded.start_line, end_line = excluded.end_line, embedding = excluded.embedding",
                            params![relative_path, chunk_index, chunk_content, start_line, end_line, embedding_json],
                        ).map_err(|e| e.to_string())?;
                    }
                    Err(e) => {
                        eprintln!("Failed to generate embedding for {}: {}", relative_path, e);
                    }
                }
            }
        } else {
            eprintln!("Failed to read file: {}", relative_path);
        }

        indexed += 1;
    }

    Ok(IndexProgress { indexed, total })
}

#[tauri::command]
pub async fn search_index(
    query: String,
    api_key: String,
    folder_path: String,
    provider: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
) -> Result<Vec<SearchResult>, String> {
    let db_path = get_db_path(&folder_path);
    println!("search_index called with folder_path: {}", folder_path);
    println!("db_path: {}", db_path.display());
    println!("query: {}", query);
    
    if !db_path.exists() {
        return Ok(Vec::new());
    }

    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    let db_config = load_embedding_metadata(&conn);
    let config = EmbeddingConfig::normalize(
        provider.or_else(|| Some(db_config.provider.clone())),
        api_key,
        base_url.or_else(|| Some(db_config.base_url.clone())),
        model.or_else(|| Some(db_config.model.clone())),
    );

    // Generate query embedding
    let query_embedding = generate_embedding(&query, &config, "query").await?;

    // Get all chunks with embeddings
    let mut stmt = conn.prepare(
        "SELECT file_path, content, embedding FROM chunks WHERE embedding IS NOT NULL"
    ).map_err(|e| e.to_string())?;

    let rows = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?))
        })
        .map_err(|e| e.to_string())?;

    let mut chunks = Vec::new();
    for row in rows {
        let (file_path, content, embedding_json) = row.map_err(|e| e.to_string())?;
        let embedding: Vec<f32> = serde_json::from_str(&embedding_json)
            .map_err(|e| format!("Failed to parse embedding: {}", e))?;
        chunks.push((file_path, content, embedding));
    }

    // Calculate cosine similarity for each chunk
    let mut results: Vec<SearchResult> = chunks
        .into_iter()
        .map(|(file_path, content, embedding)| {
            let similarity = cosine_similarity(&query_embedding, &embedding);
            SearchResult {
                file_path,
                content,
                similarity_score: similarity,
            }
        })
        .collect();

    // Sort by similarity and return top 10
    results.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap());
    results.truncate(10);

    Ok(results)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

#[tauri::command]
pub async fn clear_index(folder_path: String) -> Result<(), String> {
    let db_path = get_db_path(&folder_path);
    
    if db_path.exists() {
        let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
        conn.execute("DELETE FROM chunks", []).map_err(|e| e.to_string())?;
    }

    Ok(())
}
