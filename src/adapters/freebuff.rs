use std::collections::HashMap;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::config;
use crate::model::{RawAdapterStats, Session, file_mtime_seconds, file_timestamp, truncate_title};

use super::shared::{
    IncrementalParse, SessionFileScan, build_resume_command, incremental_parse_from_option,
    incremental_scan, parse_datetime, string_at,
};
use super::{Adapter, IncrementalScan, KnownSessions, SessionCallback};

const CHAT_MESSAGES_FILE: &str = "chat-messages.json";
const RUN_STATE_FILE: &str = "run-state.json";
// `run-state.json` embeds the project file tree after its identity fields, so
// the project root is near the start. Reading a bounded prefix avoids building
// a multi-megabyte JSON tree just to learn the working directory.
const RUN_STATE_PREFIX_BYTES: u64 = 64 * 1024;

#[derive(Debug, Clone)]
pub struct FreebuffAdapter {
    projects_dir: PathBuf,
}

impl Default for FreebuffAdapter {
    fn default() -> Self {
        Self {
            projects_dir: config::freebuff_projects_dir(),
        }
    }
}

impl Adapter for FreebuffAdapter {
    fn name(&self) -> &'static str {
        "freebuff"
    }

    fn find_sessions(&self) -> Vec<Session> {
        self.chat_dirs()
            .0
            .iter()
            .filter_map(|chat_dir| self.parse_session(chat_dir))
            .collect()
    }

    fn find_sessions_incremental(&self, known: &KnownSessions) -> IncrementalScan {
        self.incremental(known, None)
    }

    fn find_sessions_incremental_streaming(
        &self,
        known: &KnownSessions,
        on_session: &mut SessionCallback<'_>,
    ) -> IncrementalScan {
        self.incremental(known, Some(on_session))
    }

    fn resume_command(&self, session: &Session, yolo: bool) -> Vec<String> {
        build_resume_command("freebuff", &[], yolo, &["--continue"], &session.id)
    }

    fn raw_stats(&self) -> RawAdapterStats {
        let (chat_dirs, _) = self.chat_dirs();
        RawAdapterStats {
            agent: self.name(),
            data_dir: self.projects_dir.display().to_string(),
            available: self.projects_dir.is_dir(),
            file_count: chat_dirs.len(),
            total_bytes: chat_dirs
                .iter()
                .filter_map(|chat_dir| {
                    fs::metadata(chat_dir.join(CHAT_MESSAGES_FILE))
                        .ok()
                        .map(|metadata| metadata.len())
                })
                .sum(),
        }
    }
}

impl FreebuffAdapter {
    fn incremental(
        &self,
        known: &KnownSessions,
        on_session: Option<&mut SessionCallback<'_>>,
    ) -> IncrementalScan {
        incremental_scan(
            self.name(),
            known,
            self.scan_session_files(),
            |chat_dir| self.parse_session_incremental(chat_dir),
            on_session,
        )
    }

    fn scan_session_files(&self) -> Option<SessionFileScan> {
        let (chat_dirs, complete) = self.chat_dirs();
        let mut current_files = HashMap::with_capacity(chat_dirs.len());
        for chat_dir in chat_dirs {
            let Some(chat_id) = chat_dir.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let mtime = freebuff_chat_mtime(&chat_dir);
            current_files.insert(chat_id.to_string(), (chat_dir, mtime));
        }
        Some((current_files, complete))
    }

    /// Chat directories that contain a transcript, plus whether every directory
    /// on the way could be read. An incomplete scan must never delete indexed
    /// sessions: a permission error or a vanished mount looks like a deletion
    /// otherwise.
    fn chat_dirs(&self) -> (Vec<PathBuf>, bool) {
        let mut chat_dirs = Vec::new();
        let mut complete = true;
        match fs::metadata(&self.projects_dir) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => return (chat_dirs, complete),
            Err(_) => return (chat_dirs, false),
            Ok(metadata) if !metadata.is_dir() => return (chat_dirs, false),
            Ok(_) => {}
        }

        let Ok(projects) = fs::read_dir(&self.projects_dir) else {
            return (chat_dirs, false);
        };
        for project in projects {
            let Ok(project) = project else {
                complete = false;
                continue;
            };
            let project_dir = project.path();
            match directory_exists(&project_dir) {
                Ok(true) => {}
                Ok(false) => continue,
                Err(()) => {
                    complete = false;
                    continue;
                }
            }

            let chats_dir = project_dir.join("chats");
            match directory_exists(&chats_dir) {
                Ok(true) => {}
                Ok(false) => continue,
                Err(()) => {
                    complete = false;
                    continue;
                }
            }
            let Ok(chats) = fs::read_dir(&chats_dir) else {
                complete = false;
                continue;
            };
            for chat in chats {
                let Ok(chat) = chat else {
                    complete = false;
                    continue;
                };
                let chat_dir = chat.path();
                match directory_exists(&chat_dir) {
                    Ok(true) => {}
                    Ok(false) => continue,
                    Err(()) => {
                        complete = false;
                        continue;
                    }
                }
                match fs::metadata(chat_dir.join(CHAT_MESSAGES_FILE)) {
                    Ok(metadata) if metadata.is_file() => chat_dirs.push(chat_dir),
                    Ok(_) => {}
                    Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                    Err(_) => complete = false,
                }
            }
        }
        (chat_dirs, complete)
    }

    fn parse_session(&self, chat_dir: &Path) -> Option<Session> {
        let data = fs::read(chat_dir.join(CHAT_MESSAGES_FILE)).ok()?;
        self.parse_session_bytes(chat_dir, &data)
    }

    fn parse_session_bytes(&self, chat_dir: &Path, data: &[u8]) -> Option<Session> {
        let messages: Value = serde_json::from_slice(data).ok()?;
        self.parse_messages(chat_dir, messages.as_array()?)
    }

    fn parse_messages(&self, chat_dir: &Path, messages: &[Value]) -> Option<Session> {
        if messages.is_empty() {
            return None;
        }
        let mut entries = Vec::new();
        let mut first_user = String::new();
        let mut first_timestamp = None;
        for message in messages {
            if first_timestamp.is_none() {
                first_timestamp = parse_datetime(&string_at(message, &["timestamp"]));
            }
            let variant = string_at(message, &["variant"]);
            let role_prefix = match variant.as_str() {
                "user" => "» ",
                "ai" => "  ",
                _ => continue,
            };
            for text in message_texts(message) {
                if variant == "user" && first_user.is_empty() {
                    first_user = text.clone();
                }
                entries.push(format!("{role_prefix}{text}"));
            }
        }
        if entries.is_empty() {
            return None;
        }

        let chat_id = chat_dir.file_name()?.to_str()?.to_string();
        let title = if first_user.is_empty() {
            "Freebuff session".to_string()
        } else {
            truncate_title(&first_user, 80, false)
        };
        let directory = project_root(chat_dir);
        let timestamp =
            first_timestamp.unwrap_or_else(|| file_timestamp(&chat_dir.join(CHAT_MESSAGES_FILE)));
        let mut session = Session::new(
            chat_id,
            self.name(),
            title,
            directory,
            timestamp,
            entries.join("\n\n"),
            entries.len(),
        );
        session.mtime = freebuff_chat_mtime(chat_dir);
        Some(session)
    }

    fn parse_session_incremental(&self, chat_dir: &Path) -> IncrementalParse {
        let Ok(data) = fs::read(chat_dir.join(CHAT_MESSAGES_FILE)) else {
            return IncrementalParse::Retain;
        };
        let parsed = match serde_json::from_slice::<Value>(&data) {
            Ok(value) => value
                .as_array()
                .and_then(|messages| self.parse_messages(chat_dir, messages)),
            // A torn or mid-write transcript keeps its indexed data until it
            // parses again.
            Err(_) => return IncrementalParse::Retain,
        };
        incremental_parse_from_option(parsed)
    }
}

/// `Ok(true)` for a directory, `Ok(false)` when the path is absent or is not a
/// directory, and `Err(())` when it exists but cannot be inspected.
fn directory_exists(path: &Path) -> Result<bool, ()> {
    match fs::metadata(path) {
        Ok(metadata) => Ok(metadata.is_dir()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(_) => Err(()),
    }
}

fn freebuff_chat_mtime(chat_dir: &Path) -> f64 {
    file_mtime_seconds(&chat_dir.join(CHAT_MESSAGES_FILE))
        .max(file_mtime_seconds(&chat_dir.join(RUN_STATE_FILE)))
}

/// Index user prompts and assistant prose. Tool output, subagent transcripts,
/// images, and reasoning blocks are UI state rather than conversation text.
fn message_texts(message: &Value) -> Vec<String> {
    let mut texts = Vec::new();
    if let Some(blocks) = message.get("blocks").and_then(Value::as_array) {
        for block in blocks {
            if string_at(block, &["type"]) != "text"
                || string_at(block, &["textType"]) == "reasoning"
            {
                continue;
            }
            let content = string_at(block, &["content"]);
            if !content.trim().is_empty() {
                texts.push(content);
            }
        }
    }
    // Older messages and system notices keep their prose in `content` instead
    // of a text block.
    if texts.is_empty() {
        let content = string_at(message, &["content"]);
        if !content.trim().is_empty() {
            texts.push(content);
        }
    }
    texts
}

/// Recover the full project path from `run-state.json`. The project data
/// directory is keyed by basename only, so this is the only reliable way to
/// resume from the original working directory.
fn project_root(chat_dir: &Path) -> String {
    let Ok(file) = fs::File::open(chat_dir.join(RUN_STATE_FILE)) else {
        return String::new();
    };
    let mut data = Vec::new();
    if file
        .take(RUN_STATE_PREFIX_BYTES)
        .read_to_end(&mut data)
        .is_err()
    {
        return String::new();
    }
    json_string_field(&data, "projectRoot")
        .or_else(|| json_string_field(&data, "cwd"))
        .unwrap_or_default()
}

/// Read a top-level-shaped JSON string value without parsing the surrounding
/// document. Used for bounded prefixes of large run-state files.
fn json_string_field(data: &[u8], key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let start = data
        .windows(needle.len())
        .position(|window| window == needle.as_bytes())?
        + needle.len();
    let mut index = skip_ascii_whitespace(data, start);
    if data.get(index) != Some(&b':') {
        return None;
    }
    index = skip_ascii_whitespace(data, index + 1);
    if data.get(index) != Some(&b'"') {
        return None;
    }

    let value_start = index;
    index += 1;
    let mut escaped = false;
    while let Some(&byte) = data.get(index) {
        if escaped {
            escaped = false;
        } else if byte == b'\\' {
            escaped = true;
        } else if byte == b'"' {
            let literal = std::str::from_utf8(&data[value_start..=index]).ok()?;
            return serde_json::from_str(literal).ok();
        }
        index += 1;
    }
    None
}

fn skip_ascii_whitespace(data: &[u8], mut index: usize) -> usize {
    while data.get(index).is_some_and(u8::is_ascii_whitespace) {
        index += 1;
    }
    index
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::thread;
    use std::time::Duration;

    use serde_json::{Value, json};
    use tempfile::tempdir;

    use crate::adapters::{Adapter, KnownSessions};

    use super::*;

    fn write_chat(
        projects_dir: &Path,
        project: &str,
        chat_id: &str,
        messages: &Value,
        run_state: Option<&Value>,
    ) -> PathBuf {
        let chat_dir = projects_dir.join(project).join("chats").join(chat_id);
        fs::create_dir_all(&chat_dir).unwrap();
        fs::write(
            chat_dir.join(CHAT_MESSAGES_FILE),
            serde_json::to_vec(messages).unwrap(),
        )
        .unwrap();
        if let Some(run_state) = run_state {
            fs::write(
                chat_dir.join(RUN_STATE_FILE),
                serde_json::to_vec(run_state).unwrap(),
            )
            .unwrap();
        }
        chat_dir
    }

    fn run_state(directory: &str) -> Value {
        json!({
            "traceSessionId": "trace-1",
            "sessionState": {
                "fileContext": {
                    "projectRoot": directory,
                    "cwd": directory,
                    "fileTree": [{"path": "src/main.rs", "type": "file"}]
                },
                "mainAgentState": {"agentId": "main-agent"}
            },
            "output": {"type": "error", "message": "No output yet"}
        })
    }

    fn sample_messages() -> Value {
        json!([
            {
                "id": "user-1",
                "variant": "user",
                "content": "Add the Freebuff adapter",
                "timestamp": "2026-08-01T10:00:00.000Z",
                "blocks": []
            },
            {
                "id": "ai-1",
                "variant": "ai",
                "content": "",
                "timestamp": "2026-08-01T10:00:01.000Z",
                "blocks": [
                    {"type": "text", "content": "Implemented the parser", "textType": "text"},
                    {"type": "text", "content": "private chain of thought", "textType": "reasoning"},
                    {
                        "type": "tool",
                        "toolName": "run_terminal_command",
                        "toolCallId": "tool-1",
                        "input": {"command": "cargo test"},
                        "output": "test output that must not be indexed"
                    }
                ]
            },
            {
                "id": "agent-1",
                "variant": "agent",
                "content": "",
                "timestamp": "2026-08-01T10:00:02.000Z",
                "blocks": [
                    {
                        "type": "agent",
                        "agentId": "basher",
                        "agentName": "Basher",
                        "agentType": "worker",
                        "content": "subagent transcript",
                        "status": "complete",
                        "blocks": [{"type": "text", "content": "subagent answer"}]
                    }
                ]
            }
        ])
    }

    #[test]
    fn parses_chat_and_builds_resume_command() {
        let temp = tempdir().unwrap();
        let projects_dir = temp.path().join("projects");
        let chat_id = "2026-08-01T10-00-00.000Z";
        write_chat(
            &projects_dir,
            "fast-resume",
            chat_id,
            &sample_messages(),
            Some(&run_state("/repo/fast-resume")),
        );

        let adapter = FreebuffAdapter { projects_dir };
        let sessions = adapter.find_sessions();

        assert_eq!(sessions.len(), 1);
        let session = &sessions[0];
        assert_eq!(session.id, chat_id);
        assert_eq!(session.agent, "freebuff");
        assert_eq!(session.title, "Add the Freebuff adapter");
        assert_eq!(session.directory, "/repo/fast-resume");
        assert_eq!(
            session.timestamp,
            parse_datetime("2026-08-01T10:00:00.000Z").unwrap()
        );
        assert_eq!(session.message_count, 2);
        assert!(session.content.contains("» Add the Freebuff adapter"));
        assert!(session.content.contains("Implemented the parser"));
        assert!(!session.content.contains("private chain of thought"));
        assert!(!session.content.contains("test output"));
        assert!(!session.content.contains("subagent answer"));
        assert_eq!(
            adapter.resume_command(session, false),
            vec!["freebuff", "--continue", chat_id]
        );
        assert_eq!(
            adapter.resume_command(session, true),
            vec!["freebuff", "--continue", chat_id]
        );
    }

    #[test]
    fn excludes_reasoning_only_blocks_from_content() {
        let temp = tempdir().unwrap();
        let projects_dir = temp.path().join("projects");
        write_chat(
            &projects_dir,
            "repo",
            "reasoning-only",
            &json!([
                {
                    "id": "ai-1",
                    "variant": "ai",
                    "blocks": [{"type": "text", "content": "thinking", "textType": "reasoning"}]
                }
            ]),
            None,
        );

        let adapter = FreebuffAdapter { projects_dir };

        assert!(adapter.find_sessions().is_empty());
    }

    #[test]
    fn incremental_refresh_uses_transcript_and_run_state_mtime() {
        let temp = tempdir().unwrap();
        let projects_dir = temp.path().join("projects");
        let chat_dir = write_chat(
            &projects_dir,
            "repo",
            "chat-1",
            &json!([{
                "id": "user-1",
                "variant": "user",
                "content": "First prompt",
                "timestamp": "2026-08-01T10:00:00.000Z"
            }]),
            Some(&run_state("/repo")),
        );
        let transcript_mtime = file_mtime_seconds(&chat_dir.join(CHAT_MESSAGES_FILE));
        thread::sleep(Duration::from_millis(20));
        fs::write(
            chat_dir.join(RUN_STATE_FILE),
            serde_json::to_vec(&run_state("/repo")).unwrap(),
        )
        .unwrap();

        let adapter = FreebuffAdapter { projects_dir };
        let mut known = KnownSessions::new();
        known.insert(
            ("freebuff".to_string(), "chat-1".to_string()),
            transcript_mtime,
        );

        let scan = adapter.find_sessions_incremental(&known);

        assert_eq!(scan.new_or_modified.len(), 1);
        assert!(scan.new_or_modified[0].mtime > transcript_mtime);
        assert!(scan.deleted_ids.is_empty());
    }

    #[test]
    fn updated_transcript_replaces_indexed_content() {
        let temp = tempdir().unwrap();
        let projects_dir = temp.path().join("projects");
        write_chat(
            &projects_dir,
            "repo",
            "chat-1",
            &json!([{
                "id": "user-1",
                "variant": "user",
                "content": "First prompt",
                "timestamp": "2026-08-01T10:00:00.000Z"
            }]),
            None,
        );

        let adapter = FreebuffAdapter { projects_dir };
        let mut known = KnownSessions::new();
        known.insert(("freebuff".to_string(), "chat-1".to_string()), 0.0);

        let scan = adapter.find_sessions_incremental(&known);

        assert_eq!(scan.new_or_modified.len(), 1);
        assert!(scan.new_or_modified[0].content.contains("First prompt"));
    }

    #[test]
    fn malformed_transcript_retains_and_missing_transcript_deletes() {
        let temp = tempdir().unwrap();
        let projects_dir = temp.path().join("projects");
        let malformed_dir = projects_dir.join("repo/chats/malformed");
        fs::create_dir_all(&malformed_dir).unwrap();
        fs::write(malformed_dir.join(CHAT_MESSAGES_FILE), "[{\"id\":").unwrap();
        let empty_dir = projects_dir.join("repo/chats/empty");
        fs::create_dir_all(&empty_dir).unwrap();
        fs::write(empty_dir.join(CHAT_MESSAGES_FILE), "[]").unwrap();

        let adapter = FreebuffAdapter {
            projects_dir: projects_dir.clone(),
        };
        let mut known = KnownSessions::new();
        known.insert(("freebuff".to_string(), "malformed".to_string()), 1.0);
        known.insert(("freebuff".to_string(), "empty".to_string()), 1.0);
        known.insert(("freebuff".to_string(), "removed".to_string()), 1.0);

        let mut scan = adapter.find_sessions_incremental(&known);

        assert!(scan.new_or_modified.is_empty());
        scan.deleted_ids.sort();
        assert_eq!(scan.deleted_ids, vec!["empty", "removed"]);
    }

    #[test]
    fn unreadable_project_tree_does_not_delete_known_sessions() {
        let temp = tempdir().unwrap();
        let projects_dir = temp.path().join("projects");
        fs::write(&projects_dir, "not a directory").unwrap();

        let adapter = FreebuffAdapter {
            projects_dir: projects_dir.clone(),
        };
        let mut known = KnownSessions::new();
        known.insert(("freebuff".to_string(), "chat-1".to_string()), 1.0);

        let scan = adapter.find_sessions_incremental(&known);

        assert!(scan.new_or_modified.is_empty());
        assert!(scan.deleted_ids.is_empty());

        let missing = FreebuffAdapter {
            projects_dir: temp.path().join("missing-projects"),
        };
        let scan = missing.find_sessions_incremental(&known);
        assert_eq!(scan.deleted_ids, vec!["chat-1"]);
    }

    #[test]
    fn missing_run_state_leaves_directory_empty() {
        let temp = tempdir().unwrap();
        let projects_dir = temp.path().join("projects");
        write_chat(
            &projects_dir,
            "repo",
            "chat-1",
            &json!([{
                "id": "user-1",
                "variant": "user",
                "content": "No run state",
                "timestamp": "2026-08-01T10:00:00.000Z"
            }]),
            None,
        );

        let adapter = FreebuffAdapter { projects_dir };
        let sessions = adapter.find_sessions();

        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].directory, "");
    }

    #[test]
    fn json_string_field_reads_escaped_values() {
        let data = br#"{"traceSessionId":"t","sessionState":{"fileContext":{"projectRoot":"C:\\repo\\\"quoted\"","cwd":"/ignored"}}}"#;

        assert_eq!(
            json_string_field(data, "projectRoot").as_deref(),
            Some("C:\\repo\\\"quoted\"")
        );
        assert_eq!(json_string_field(data, "missing"), None);
    }
}
