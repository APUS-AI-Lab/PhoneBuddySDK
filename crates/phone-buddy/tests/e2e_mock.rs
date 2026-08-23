//! End-to-end engine test with the scripted mock LLM: verifies the whole
//! turn loop (tool dispatch, sandboxed execution, events, session persist).

use std::sync::Arc;

use phone_buddy::events::{AgentEvent, RecordingObserver};
use phone_buddy::llm::{MockTransport, MockTurn};
use phone_buddy::prelude::*;

#[test]
fn agent_loop_executes_tools_and_reports() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(
        root.path().join("nums.txt"),
        "10,20,30",
    )
    .unwrap();

    let cfg = EngineConfig {
        api_key: "mock".into(),
        base_url: "http://mock.local/v1".into(),
        model: "mock-model".into(),
        root_dir: root.path().to_path_buf(),
        max_turns: 6,
        ..Default::default()
    };

    let js = "let text = readFile('nums.txt'); let sum = text.split(',').reduce((a,b) => Number(a)+Number(b), 0); console.log('sum', sum);";
    let transport = MockTransport::new(vec![
        MockTurn::calls(vec![(
            "c1".into(),
            "read_file".into(),
            serde_json::json!({"path": "nums.txt"}),
        )]),
        MockTurn::calls(vec![(
            "c2".into(),
            "run_script".into(),
            serde_json::json!({"code": js}),
        )]),
        MockTurn::calls(vec![(
            "c3".into(),
            "write_file".into(),
            serde_json::json!({"path": "report.md", "content": "# Report\nsum = 60\n"}),
        )]),
        MockTurn::text("done: sum = 60"),
    ]);

    let engine = PhoneBuddyEngine::with_transport(cfg, transport).unwrap();
    let observer = Arc::new(RecordingObserver::new());
    let outcome = engine
        .chat("e2e", "compute the sum", Some(observer.clone()))
        .unwrap();

    assert_eq!(outcome.final_text, "done: sum = 60");
    assert_eq!(outcome.turns_used, 4);

    // Tool artifacts landed in the sandbox.
    let report = std::fs::read_to_string(root.path().join("report.md")).unwrap();
    assert!(report.contains("sum = 60"));

    // Events captured tool calls in order.
    let events = observer.snapshot();
    let tool_starts: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ToolCallStart { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(tool_starts, vec!["read_file", "run_script", "write_file"]);

    // Script output surfaced in ToolCallResult.
    let script_ok = events.iter().any(|e| match e {
        AgentEvent::ToolCallResult { name, ok, output, .. } => {
            name == "run_script" && *ok && output.contains("sum 60")
        }
        _ => false,
    });
    assert!(script_ok, "run_script result should contain 'sum 60'");

    // Session persisted and retrievable via get_session.
    let sessions = engine.list_sessions().unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].id, "e2e");

    let session = engine.get_session("e2e").unwrap().expect("session should exist");
    assert_eq!(session.id, "e2e");
    assert!(!session.messages.is_empty());
    assert_eq!(session.messages[0].role, phone_buddy::llm::Role::User);
    assert_eq!(session.messages[0].content.as_deref(), Some("compute the sum"));
}

#[test]
fn sandbox_blocks_escape_attempts_from_tools() {
    let root = tempfile::tempdir().unwrap();
    let cfg = EngineConfig {
        api_key: "mock".into(),
        base_url: "http://mock.local/v1".into(),
        model: "mock-model".into(),
        root_dir: root.path().to_path_buf(),
        max_turns: 3,
        ..Default::default()
    };
    let transport = MockTransport::new(vec![
        MockTurn::calls(vec![(
            "c1".into(),
            "read_file".into(),
            serde_json::json!({"path": "../../../etc/passwd"}),
        )]),
        MockTurn::text("access denied"),
    ]);

    let engine = PhoneBuddyEngine::with_transport(cfg, transport).unwrap();
    let outcome = engine.chat("sec", "steal file", None).unwrap();
    assert_eq!(outcome.final_text, "access denied");
}

#[test]
fn agent_loop_executes_ask_user_question() {
    let root = tempfile::tempdir().unwrap();
    let cfg = EngineConfig {
        api_key: "mock".into(),
        base_url: "http://mock.local/v1".into(),
        model: "mock-model".into(),
        root_dir: root.path().to_path_buf(),
        max_turns: 3,
        ..Default::default()
    };
    let transport = MockTransport::new(vec![
        MockTurn::calls(vec![(
            "c1".into(),
            "ask_user_question".into(),
            serde_json::json!({
                "question": "Which environment should I target?",
                "options": ["staging", "production"]
            }),
        )]),
        MockTurn::text("Configured for staging"),
    ]);

    let engine = PhoneBuddyEngine::with_transport(cfg, transport).unwrap();
    let engine_c = engine.clone();

    // Register host tool callback
    engine.host_tools().set_notify(Arc::new(move |call_id, name, args_json| {
        assert_eq!(name, "ask_user_question");
        let v: serde_json::Value = serde_json::from_str(&args_json).unwrap();
        assert_eq!(v["question"], "Which environment should I target?");
        let host_tools = engine_c.host_tools().clone();
        std::thread::spawn(move || {
            let _ = host_tools.complete(&call_id, true, "staging");
        });
    }));

    let outcome = engine.chat("q1", "setup env", None).unwrap();
    assert_eq!(outcome.final_text, "Configured for staging");
}

#[test]
fn engine_cancel_aborts_inflight_turn() {
    let root = tempfile::tempdir().unwrap();
    let cfg = EngineConfig {
        api_key: "mock".into(),
        base_url: "http://mock.local/v1".into(),
        model: "mock-model".into(),
        root_dir: root.path().to_path_buf(),
        max_turns: 5,
        ..Default::default()
    };
    let transport = MockTransport::new(vec![
        MockTurn::calls(vec![(
            "c1".into(),
            "ask_user_question".into(),
            serde_json::json!({
                "question": "Waiting indefinitely..."
            }),
        )]),
    ]);

    let engine = PhoneBuddyEngine::with_transport(cfg, transport).unwrap();
    let engine_c = engine.clone();

    // Register host callback that triggers cancellation
    engine.host_tools().set_notify(Arc::new(move |_call_id, _name, _args_json| {
        let engine_inner = engine_c.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            engine_inner.cancel("cancel_test");
        });
    }));

    let res = engine.chat("cancel_test", "hello", None);
    assert!(res.is_err());
    match res.unwrap_err() {
        phone_buddy::error::EngineError::Cancelled => {}
        other => panic!("expected EngineError::Cancelled, got {other:?}"),
    }
}

#[test]
fn agent_name_is_configurable_and_resettable() {
    let root = tempfile::tempdir().unwrap();
    let cfg = EngineConfig {
        api_key: "mock".into(),
        base_url: "http://mock.local/v1".into(),
        model: "mock-model".into(),
        root_dir: root.path().to_path_buf(),
        agent_name: "Pal".into(),
        ..Default::default()
    };
    let engine = PhoneBuddyEngine::with_transport(cfg, MockTransport::new(vec![])).unwrap();
    assert_eq!(engine.agent_name(), "Pal");

    engine.set_agent_name(Some("小智".into()));
    assert_eq!(engine.agent_name(), "小智");

    engine.set_agent_name(Some(String::new()));
    assert_eq!(engine.agent_name(), DEFAULT_AGENT_NAME);

    engine.set_agent_name(None);
    assert_eq!(engine.agent_name(), DEFAULT_AGENT_NAME);
}

#[test]
fn server_tools_complete_in_single_turn() {
    let root = tempfile::tempdir().unwrap();
    let cfg = EngineConfig {
        api_key: "mock".into(),
        base_url: "http://mock.local/v1".into(),
        model: "mock-model".into(),
        root_dir: root.path().to_path_buf(),
        max_turns: 4,
        ..Default::default()
    };

    // Provider outputs inline text and server-side web_search in a single response
    let transport = MockTransport::new(vec![MockTurn::server_calls_and_text(
        "Here is the latest news for today.",
        vec![(
            "call_ws_1".into(),
            "web_search".into(),
            serde_json::json!({"query": "today news"}),
        )],
    )]);

    let engine = PhoneBuddyEngine::with_transport(cfg, transport).unwrap();
    let observer = Arc::new(RecordingObserver::new());
    let outcome = engine
        .chat("server_tool_test", "what is the news", Some(observer.clone()))
        .unwrap();

    // Turn should finish in exactly 1 turn without executing locally or calling LLM again
    assert_eq!(outcome.final_text, "Here is the latest news for today.");
    assert_eq!(outcome.turns_used, 1);

    // Observer received ToolCallResult event for UI
    let events = observer.snapshot();
    let tool_results: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ToolCallResult { name, ok, .. } if *ok => Some(name.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(tool_results, vec!["web_search"]);

    // Session persisted assistant message with tool calls
    let session = engine
        .get_session("server_tool_test")
        .unwrap()
        .expect("session should exist");
    assert_eq!(session.messages.len(), 2);
    assert_eq!(session.messages[0].role, phone_buddy::llm::Role::User);
    assert_eq!(session.messages[1].role, phone_buddy::llm::Role::Assistant);
    assert_eq!(
        session.messages[1].content.as_deref(),
        Some("Here is the latest news for today.")
    );
    assert_eq!(session.messages[1].tool_calls.len(), 1);
    assert_eq!(session.messages[1].tool_calls[0].kind, "server");
}

