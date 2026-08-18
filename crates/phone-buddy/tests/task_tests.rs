use std::sync::Arc;
use tempfile::tempdir;

use phone_buddy::agent::task_manager::{TaskInput, TaskManager};
use phone_buddy::config::EngineConfig;
use phone_buddy::llm::{MockTransport, MockTurn};
use phone_buddy::llm::client::LlmClient;
use phone_buddy::tools::fs::Sandbox;
use phone_buddy::tools::ToolRegistry;

#[tokio::test]
async fn test_task_manager_sync_and_async_spawn() {
    let dir = tempdir().unwrap();
    let cfg = EngineConfig {
        api_key: "mock".into(),
        base_url: "http://mock.local/v1".into(),
        model: "mock-model".into(),
        root_dir: dir.path().to_path_buf(),
        ..Default::default()
    };

    let turns = vec![
        MockTurn::text("Subagent finished subtask successfully."),
        MockTurn::text("Second subagent finished."),
    ];
    let transport = MockTransport::new(turns);
    let client = Arc::new(LlmClient::new(transport, 0));
    let sandbox = Arc::new(Sandbox::new(dir.path()).unwrap());
    let subagent_tools = Arc::new(ToolRegistry::new());

    let manager = Arc::new(TaskManager::new(cfg, client, sandbox, subagent_tools));

    // 1. Sync spawn
    let sync_input = TaskInput {
        prompt: "Run quick subtask".into(),
        description: "Quick subtask".into(),
        subagent_type: "general-purpose".into(),
        run_in_background: false,
        resume_from: None,
        model: None,
    };

    let sync_res = manager.spawn_task(sync_input).await.unwrap();
    assert!(sync_res.contains("Subagent finished subtask successfully."));
    assert!(sync_res.contains("<subagent_result>"));

    // 2. Async background spawn
    let bg_input = TaskInput {
        prompt: "Run background task".into(),
        description: "Bg task".into(),
        subagent_type: "general-purpose".into(),
        run_in_background: true,
        resume_from: None,
        model: None,
    };

    let bg_res = manager.spawn_task(bg_input).await.unwrap();
    assert!(bg_res.contains("Subagent started in background"));
    assert!(bg_res.contains("subagent_id: task-2"));

    // Wait for background task to complete
    let output = manager.get_task_output(&["task-2".to_string()], Some(2000)).await.unwrap();
    assert!(output.contains("completed") || output.contains("task-2"));
}

#[tokio::test]
async fn test_task_manager_kill_and_wait() {
    let dir = tempdir().unwrap();
    let cfg = EngineConfig {
        api_key: "mock".into(),
        base_url: "http://mock.local/v1".into(),
        model: "mock-model".into(),
        root_dir: dir.path().to_path_buf(),
        ..Default::default()
    };

    let turns = vec![MockTurn::text("Task done.")];
    let transport = MockTransport::new(turns);
    let client = Arc::new(LlmClient::new(transport, 0));
    let sandbox = Arc::new(Sandbox::new(dir.path()).unwrap());
    let subagent_tools = Arc::new(ToolRegistry::new());

    let manager = Arc::new(TaskManager::new(cfg, client, sandbox, subagent_tools));

    let bg_input = TaskInput {
        prompt: "Long running task".into(),
        description: "Long task".into(),
        subagent_type: "general-purpose".into(),
        run_in_background: true,
        resume_from: None,
        model: None,
    };

    let _res = manager.spawn_task(bg_input).await.unwrap();

    // Test wait_tasks
    let wait_res = manager.wait_tasks(&["task-1".to_string()], "wait_all", Some(2000)).await.unwrap();
    assert!(wait_res.contains("task-1"));

    // Test kill_task on completed or existing task
    let kill_res = manager.kill_task("task-1").unwrap();
    assert!(kill_res.contains("task-1"));
}
