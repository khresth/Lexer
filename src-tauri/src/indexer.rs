use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
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

    Ok(())
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

async fn generate_embedding(text: &str, api_key: &str) -> Result<Vec<f32>, String> {
    let client = reqwest::Client::new();
    
    #[derive(Serialize)]
    struct EmbeddingRequest {
        input: String,
        model: String,
        encoding_format: String,
        input_type: String,
    }

    #[derive(Deserialize)]
    struct EmbeddingData {
        embedding: Vec<f32>,
    }

    #[derive(Deserialize)]
    struct EmbeddingResponse {
        data: Vec<EmbeddingData>,
    }

    let request = EmbeddingRequest {
        input: text.to_string(),
        model: "nvidia/nv-embedqa-e5-v5".to_string(),
        encoding_format: "float".to_string(),
        input_type: "passage".to_string(),
    };

    let response = client
        .post("https://integrate.api.nvidia.com/v1/embeddings")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("Embedding request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(format!("Embedding API error {}: {}", status, text));
    }

    let result: EmbeddingResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse embedding response: {}", e))?;

    result
        .data
        .into_iter()
        .next()
        .map(|d| d.embedding)
        .ok_or_else(|| "No embedding returned".to_string())
}

async fn generate_query_embedding(text: &str, api_key: &str) -> Result<Vec<f32>, String> {
    let client = reqwest::Client::new();
    
    #[derive(Serialize)]
    struct EmbeddingRequest {
        input: String,
        model: String,
        encoding_format: String,
        input_type: String,
    }

    #[derive(Deserialize)]
    struct EmbeddingData {
        embedding: Vec<f32>,
    }

    #[derive(Deserialize)]
    struct EmbeddingResponse {
        data: Vec<EmbeddingData>,
    }

    let request = EmbeddingRequest {
        input: text.to_string(),
        model: "nvidia/nv-embedqa-e5-v5".to_string(),
        encoding_format: "float".to_string(),
        input_type: "query".to_string(),
    };

    let response = client
        .post("https://integrate.api.nvidia.com/v1/embeddings")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("Embedding request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(format!("Embedding API error {}: {}", status, text));
    }

    let result: EmbeddingResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse embedding response: {}", e))?;

    result
        .data
        .into_iter()
        .next()
        .map(|d| d.embedding)
        .ok_or_else(|| "No embedding returned".to_string())
}

#[tauri::command]
pub async fn index_workspace(
    folder_path: String,
    api_key: String,
) -> Result<IndexProgress, String> {
    let db_path = get_db_path(&folder_path);

    let mut conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    init_db(&mut conn)?;

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
                match generate_embedding(chunk_content, &api_key).await {
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
) -> Result<Vec<SearchResult>, String> {
    let db_path = get_db_path(&folder_path);
    println!("search_index called with folder_path: {}", folder_path);
    println!("db_path: {}", db_path.display());
    println!("query: {}", query);
    
    if !db_path.exists() {
        return Ok(Vec::new());
    }

    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;

    // Generate query embedding
    let query_embedding = generate_query_embedding(&query, &api_key).await?;

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
