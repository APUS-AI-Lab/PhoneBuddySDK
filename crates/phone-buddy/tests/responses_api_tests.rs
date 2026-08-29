//! Integration tests for Responses API features:
//! - Multimodal inputs: input_text, input_image, input_audio
//! - Hosted tools: XSearch with toggle & options, WebSearch
//! - Tool calls: LocalShellCall (sandboxed pure-Rust busybox) and CustomToolCall

use phone_buddy::config::{EngineConfigBuilder, XSearchOptions};
use phone_buddy::conversation::{AudioMimeType, UserContentPart, UserItem};
use phone_buddy::engine::PhoneBuddyEngine;

#[tokio::test]
async fn test_x_search_toggle_and_options() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg_disabled = EngineConfigBuilder::new()
        .api_key("test_key")
        .root_dir(tmp.path())
        .enable_x_search(false)
        .build()
        .unwrap();
    assert!(!cfg_disabled.enable_x_search);
    assert!(!cfg_disabled.backend_x_search_active());

    let cfg_enabled = EngineConfigBuilder::new()
        .api_key("test_key")
        .root_dir(tmp.path())
        .api_backend(phone_buddy::config::ApiBackend::Responses)
        .enable_x_search(true)
        .x_search_options(XSearchOptions {
            date_bound: None,
            from_date: Some("2026-01-01".into()),
            to_date: Some("2026-08-01".into()),
        })
        .build()
        .unwrap();
    assert!(cfg_enabled.enable_x_search);
    assert!(cfg_enabled.backend_x_search_active());
    assert!(cfg_enabled.x_search_options.is_some());
}

#[tokio::test]
async fn test_multimodal_audio_validation_and_materialization() {
    let tmp = tempfile::tempdir().unwrap();
    let audio_path = tmp.path().join("test_sample.wav");

    // Standard WAV header (RIFF ... WAVE)
    let mut wav_bytes = Vec::new();
    wav_bytes.extend_from_slice(b"RIFF");
    wav_bytes.extend_from_slice(&(36u32.to_le_bytes()));
    wav_bytes.extend_from_slice(b"WAVEfmt ");
    wav_bytes.extend_from_slice(&(16u32.to_le_bytes()));
    wav_bytes.extend_from_slice(&(1u16.to_le_bytes())); // PCM
    wav_bytes.extend_from_slice(&(1u16.to_le_bytes())); // 1 channel
    wav_bytes.extend_from_slice(&(16000u32.to_le_bytes())); // sample rate
    wav_bytes.extend_from_slice(&(32000u32.to_le_bytes())); // byte rate
    wav_bytes.extend_from_slice(&(2u16.to_le_bytes())); // block align
    wav_bytes.extend_from_slice(&(16u16.to_le_bytes())); // bits per sample
    wav_bytes.extend_from_slice(b"data");
    wav_bytes.extend_from_slice(&(0u32.to_le_bytes()));

    std::fs::write(&audio_path, &wav_bytes).unwrap();

    let user_turn = UserItem {
        parts: vec![
            UserContentPart::Text { text: "analyze audio".into() },
            UserContentPart::Audio {
                attachment_id: "audio_1".into(),
                local_path: audio_path.to_str().unwrap().into(),
                mime_type: AudioMimeType::Wav,
                byte_size: wav_bytes.len() as u64,
                format: Some("wav".into()),
            },
        ],
    };

    assert!(user_turn.has_audio());
    assert!(user_turn.has_media());
    assert_eq!(user_turn.audio_count(), 1);
    assert!(user_turn.validate_shape().is_ok());

    let store = phone_buddy::llm::image::AudioBytesStore::default();
    let mat_res = phone_buddy::llm::image::materialize_audio_user_item(&user_turn, tmp.path(), &store);
    assert!(mat_res.is_ok());

    let retrieved = store.get("audio_1").unwrap();
    assert_eq!(retrieved.mime_type, AudioMimeType::Wav);
    assert!(retrieved.data_url().starts_with("data:audio/wav;base64,"));
}

#[tokio::test]
async fn test_local_shell_busybox_sandboxed_execution() {
    let tmp = tempfile::tempdir().unwrap();
    let test_file = tmp.path().join("greeting.txt");
    std::fs::write(&test_file, "Hello from mobile sandbox!\n").unwrap();

    let cfg = EngineConfigBuilder::new()
        .api_key("test_key")
        .root_dir(tmp.path())
        .build()
        .unwrap();
    let engine = PhoneBuddyEngine::new(cfg).unwrap();

    // Directly test sandboxed command execution via busybox
    let ctx = phone_buddy::tools::ToolCtx {
        sandbox: engine.sandbox().clone(),
        cancel: tokio_util::sync::CancellationToken::new(),
    };

    // 1. cat greeting.txt
    let out_cat = phone_buddy::tools::busybox::execute_command_argv(
        &["cat".into(), "greeting.txt".into()],
        &ctx,
    ).unwrap();
    assert_eq!(out_cat.text.trim(), "Hello from mobile sandbox!");

    // 2. sh -c "echo 'pure rust'"
    let out_sh = phone_buddy::tools::busybox::execute_command_line(
        "echo pure rust",
        &ctx,
    ).unwrap();
    assert_eq!(out_sh.text.trim(), "pure rust");

    // 3. ls
    let out_ls = phone_buddy::tools::busybox::execute_command_argv(
        &["ls".into()],
        &ctx,
    ).unwrap();
    assert!(out_ls.text.contains("greeting.txt"));
}

#[tokio::test]
async fn test_codex_profile_protocol_conformance() {
    use phone_buddy::config::ClientProfile;
    use phone_buddy::llm::profiles::{build_profile_headers, render_user_agent};
    use phone_buddy::llm::wire::responses::build_responses_payload;
    use phone_buddy::llm::types::{
        ConversationRequest, FunctionDefinitionWire, ToolDefinitionWire,
    };
    use phone_buddy::conversation::ConversationItem;

    // 1. Check Codex User-Agent & Headers
    let ua = render_user_agent(ClientProfile::Codex, Some("0.1.0"));
    assert!(ua.starts_with("codex_cli_rs/0.1.0"));
    assert!(ua.ends_with("codex_cli"));

    let headers = build_profile_headers(
        ClientProfile::Codex,
        "sk-test-key",
        Some("session-xyz-123"),
        Some("0.1.0"),
        false,
    );
    assert_eq!(headers.get("originator").unwrap(), "codex_cli_rs");
    assert_eq!(headers.get("session-id").unwrap(), "session-xyz-123");
    assert_eq!(headers.get("thread-id").unwrap(), "session-xyz-123");
    assert_eq!(headers.get("x-client-request-id").unwrap(), "session-xyz-123");
    assert_eq!(headers.get("x-codex-installation-id").unwrap(), "session-xyz-123");
    assert_eq!(headers.get("x-codex-window-id").unwrap(), "session-xyz-123");
    assert_eq!(headers.get("openai-beta").unwrap(), "responses=true");
    assert_eq!(headers.get("authorization").unwrap(), "Bearer sk-test-key");

    // 2. Check Responses API payload structure alignment with codex-rs
    let req = ConversationRequest {
        model: "gpt-5".into(),
        items: vec![
            ConversationItem::system("You are an expert coding assistant."),
            ConversationItem::user("Inspect the files in workspace"),
        ],
        stream: Some(true),
        tools: Some(vec![ToolDefinitionWire {
            kind: "function".into(),
            function: FunctionDefinitionWire {
                name: "read_file".into(),
                description: Some("Read file content".into()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" }
                    },
                    "required": ["path"]
                }),
            },
        }]),
        tool_choice: None,
        temperature: None,
        max_tokens: None,
        reasoning_effort: Some(phone_buddy::config::ReasoningEffort::Medium),
        search_parameters: None,
        hosted_tools: vec![],
        previous_response_id: None,
        response_format: None,
        image_bytes: phone_buddy::llm::image::ImageBytesStore::default(),
        audio_bytes: phone_buddy::llm::image::AudioBytesStore::default(),
    };

    let payload = build_responses_payload(&req).unwrap();
    assert_eq!(payload["model"], "gpt-5");
    assert_eq!(payload["parallel_tool_calls"], true);
    assert_eq!(payload["tool_choice"], "auto");
    assert_eq!(payload["stream"], true);
    assert_eq!(payload["store"], false);
    assert_eq!(
        payload["stream_options"]["reasoning_summary_delivery"],
        "sequential_cutoff"
    );
    assert_eq!(payload["text"]["verbosity"], "medium");
    assert_eq!(payload["reasoning"]["summary"], "concise");
    assert_eq!(payload["reasoning"]["effort"], "medium");
    assert_eq!(payload["tools"].as_array().unwrap().len(), 1);
    assert_eq!(payload["tools"][0]["name"], "read_file");
    assert_eq!(payload["tools"][0]["type"], "function");
}

#[tokio::test]
async fn test_x_thread_fetch_and_x_search_alignment() {
    use phone_buddy::conversation::{BackendToolCallItem, ConversationItem};
    use phone_buddy::llm::types::{ConversationRequest, HostedTool, XSearchOptions};
    use phone_buddy::llm::wire::responses::{build_responses_payload, parse_responses_chunk};

    // 1. Check hosted tool entry generation with options (fromDate / toDate)
    let hosted = HostedTool::for_request_with_options(
        false,
        None,
        true,
        Some(XSearchOptions {
            date_bound: None,
            from_date: Some("2026-01-01".into()),
            to_date: Some("2026-08-28".into()),
        }),
        phone_buddy::config::ApiBackend::Responses,
    );
    assert_eq!(hosted.len(), 1);
    let entry = hosted[0].to_tool_entry();
    assert_eq!(entry["type"], "x_search");
    assert_eq!(entry["from_date"], "2026-01-01");
    assert_eq!(entry["to_date"], "2026-08-28");

    // 2. Request payload includes { "type": "x_search", "from_date": ..., "to_date": ... }
    let req = ConversationRequest {
        model: "grok-4.6".into(),
        items: vec![ConversationItem::user("Summarize this thread")],
        stream: Some(true),
        tools: None,
        tool_choice: None,
        temperature: None,
        max_tokens: None,
        reasoning_effort: None,
        search_parameters: None,
        hosted_tools: hosted,
        previous_response_id: None,
        response_format: None,
        image_bytes: phone_buddy::llm::image::ImageBytesStore::default(),
        audio_bytes: phone_buddy::llm::image::AudioBytesStore::default(),
    };
    let payload = build_responses_payload(&req).unwrap();
    let tools = payload["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["type"], "x_search");
    assert_eq!(tools[0]["from_date"], "2026-01-01");
    assert_eq!(tools[0]["to_date"], "2026-08-28");

    // 3. Streaming chunk with custom_tool_call (e.g. x_thread_fetch)
    let chunk_json = serde_json::json!({
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "type": "custom_tool_call",
            "id": "ct_12345",
            "call_id": "call_12345",
            "name": "x_thread_fetch",
            "input": "{\"query\": \"status/123456789\"}"
        }
    })
    .to_string();

    let chunk = parse_responses_chunk("response.output_item.added", &chunk_json)
        .unwrap()
        .expect("chunk parsed");
    let delta = &chunk.choices[0].delta;
    assert_eq!(delta.tool_calls.len(), 1);
    let tc = &delta.tool_calls[0];
    assert_eq!(tc.kind.as_deref(), Some("server"));
    assert_eq!(tc.function.as_ref().unwrap().name.as_deref(), Some("x_thread_fetch"));

    // 4. BackendToolCallItem display_name formatting
    let backend_item = BackendToolCallItem {
        item_type: "custom_tool_call".into(),
        id: "call_12345".into(),
        payload: serde_json::json!({
            "type": "custom_tool_call",
            "name": "x_thread_fetch",
            "call_id": "call_12345",
            "input": {"query": "status/123456789"}
        }),
    };
    assert_eq!(backend_item.display_name(), "x_thread_fetch");
}

