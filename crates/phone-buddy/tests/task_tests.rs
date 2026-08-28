use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use tempfile::tempdir;

use phone_buddy::agent::task_manager::{TaskInput, TaskManager};
use phone_buddy::config::EngineConfig;
use phone_buddy::error::EngineError;
use phone_buddy::llm::client::{LlmClient, LlmTransportObj};
use phone_buddy::llm::router::{
    ExhaustionPolicy, LlmRouter, LlmRoutingConfig, PoolMember, ProviderPool, ProviderTarget,
    RetryPolicy, MAIN_POOL_ID, SUBAGENT_POOL_ID,
};
use phone_buddy::llm::transport::LlmTransport;
use phone_buddy::llm::types::{
    ChatChunkChoice, ChatChunkDelta, ChatCompletionChunk, ConversationRequest, Role,
};
use phone_buddy::llm::{MockTransport, MockTurn};
use phone_buddy::tools::fs::Sandbox;
use phone_buddy::tools::task::TaskTool;
use phone_buddy::tools::{Tool, ToolRegistry};

struct CountingTransport {
    name: String,
    remaining_fails: AtomicU32,
    error: String,
    hits: Arc<AtomicU32>,
    models: Arc<Mutex<Vec<String>>>,
}

impl CountingTransport {
    fn ok(name: &str, hits: Arc<AtomicU32>, models: Arc<Mutex<Vec<String>>>) -> Arc<Self> {
        Arc::new(Self {
            name: name.into(),
            remaining_fails: AtomicU32::new(0),
            error: String::new(),
            hits,
            models,
        })
    }

    fn failing(
        name: &str,
        fails: u32,
        error: &str,
        hits: Arc<AtomicU32>,
        models: Arc<Mutex<Vec<String>>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            name: name.into(),
            remaining_fails: AtomicU32::new(fails),
            error: error.into(),
            hits,
            models,
        })
    }
}

impl LlmTransport for CountingTransport {
    async fn request_stream(
        &self,
        req: &ConversationRequest,
    ) -> phone_buddy::error::EngineResult<phone_buddy::llm::transport::ChunkStream> {
        self.hits.fetch_add(1, Ordering::SeqCst);
        self.models.lock().unwrap().push(req.model.clone());
        let left = self.remaining_fails.load(Ordering::SeqCst);
        if left > 0 {
            self.remaining_fails.fetch_sub(1, Ordering::SeqCst);
            return Err(EngineError::Llm(self.error.clone()));
        }
        let stream = async_stream::stream! {
            let mut d = ChatChunkDelta::default();
            d.role = Some(Role::Assistant);
            d.content = Some("subagent done".into());
            yield Ok(ChatCompletionChunk {
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: d,
                    finish_reason: Some("stop".into()),
                }],
                ..Default::default()
            });
        };
        Ok(Box::pin(stream))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

fn engine_cfg(root: &std::path::Path) -> EngineConfig {
    EngineConfig {
        api_key: "mock".into(),
        base_url: "http://mock.local/v1".into(),
        model: "mock-model".into(),
        root_dir: root.to_path_buf(),
        ..Default::default()
    }
}

fn target(id: &str, url: &str, model: &str) -> ProviderTarget {
    ProviderTarget {
        provider_id: id.into(),
        base_url: url.into(),
        api_key: "k".into(),
        model: model.into(),
        api_backend: Default::default(),
        client_profile: Default::default(),
        client_version: None,
        client_session_id: None,
        reasoning_compatibility_key: None,
        capabilities: Default::default(),
        extra_headers: Default::default(),
        extra_body: Default::default(),
        enable_web_search: false,
        web_search_options: None,
        enable_x_search: false,
        x_search_options: None,
        reasoning_effort: None,
    }
}

fn member(id: &str, order: u32) -> PoolMember {
    PoolMember {
        provider_id: id.into(),
        routing_group: "default".into(),
        base_score: 10,
        order,
        enabled: true,
    }
}

fn pool(members: Vec<PoolMember>) -> ProviderPool {
    ProviderPool {
        members,
        retry: RetryPolicy {
            failover_max_attempts: 3,
            max_retries: 1,
        },
        when_exhausted: ExhaustionPolicy::ProbeEarliest,
    }
}

fn task_input(prompt: &str) -> TaskInput {
    TaskInput {
        prompt: prompt.into(),
        description: "subtask".into(),
        subagent_type: "general-purpose".into(),
        run_in_background: false,
        resume_from: None,
    }
}

#[tokio::test]
async fn test_task_manager_sync_and_async_spawn() {
    let dir = tempdir().unwrap();
    let cfg = engine_cfg(dir.path());

    let turns = vec![
        MockTurn::text("Subagent finished subtask successfully."),
        MockTurn::text("Second subagent finished."),
    ];
    let transport = MockTransport::new(turns);
    let client = Arc::new(LlmClient::new(transport, 0));
    let sandbox = Arc::new(Sandbox::new(dir.path()).unwrap());
    let subagent_tools = Arc::new(ToolRegistry::new());

    let manager = Arc::new(TaskManager::new(cfg, client, sandbox, subagent_tools));

    let sync_input = task_input("Run quick subtask");
    let mut sync_input = sync_input;
    sync_input.description = "Quick subtask".into();

    let sync_res = manager.spawn_task(sync_input).await.unwrap();
    assert!(sync_res.contains("Subagent finished subtask successfully."));
    assert!(sync_res.contains("<subagent_result>"));

    let mut bg_input = task_input("Run background task");
    bg_input.description = "Bg task".into();
    bg_input.run_in_background = true;

    let bg_res = manager.spawn_task(bg_input).await.unwrap();
    assert!(bg_res.contains("Subagent started in background"));
    assert!(bg_res.contains("subagent_id: task-2"));

    let output = manager
        .get_task_output(&["task-2".to_string()], Some(2000))
        .await
        .unwrap();
    assert!(output.contains("completed") || output.contains("task-2"));
}

#[tokio::test]
async fn test_task_manager_kill_and_wait() {
    let dir = tempdir().unwrap();
    let cfg = engine_cfg(dir.path());

    let turns = vec![MockTurn::text("Task done.")];
    let transport = MockTransport::new(turns);
    let client = Arc::new(LlmClient::new(transport, 0));
    let sandbox = Arc::new(Sandbox::new(dir.path()).unwrap());
    let subagent_tools = Arc::new(ToolRegistry::new());

    let manager = Arc::new(TaskManager::new(cfg, client, sandbox, subagent_tools));

    let mut bg_input = task_input("Long running task");
    bg_input.description = "Long task".into();
    bg_input.run_in_background = true;

    let _res = manager.spawn_task(bg_input).await.unwrap();

    let wait_res = manager
        .wait_tasks(&["task-1".to_string()], "wait_all", Some(2000))
        .await
        .unwrap();
    assert!(wait_res.contains("task-1"));

    let kill_res = manager.kill_task("task-1").unwrap();
    assert!(kill_res.contains("task-1"));
}

#[tokio::test]
async fn subagent_spawn_never_hits_main_pool_transport() {
    let dir = tempdir().unwrap();
    let main_hits = Arc::new(AtomicU32::new(0));
    let sub_hits = Arc::new(AtomicU32::new(0));
    let main_models = Arc::new(Mutex::new(Vec::new()));
    let sub_models = Arc::new(Mutex::new(Vec::new()));
    let main_t = CountingTransport::ok("main", main_hits.clone(), main_models);
    let sub_t = CountingTransport::ok("sub", sub_hits.clone(), sub_models.clone());

    let mut pools = BTreeMap::new();
    pools.insert(MAIN_POOL_ID.into(), pool(vec![member("p-main", 0)]));
    pools.insert(SUBAGENT_POOL_ID.into(), pool(vec![member("p-sub", 0)]));
    let routing = LlmRoutingConfig {
        providers: vec![
            target("p-main", "https://main.example/v1", "main-model"),
            target("p-sub", "https://cheap.example/v1", "cheap-model"),
        ],
        pools,
        health: Default::default(),
    };
    let router = LlmRouter::in_memory(routing).unwrap();
    let _main = LlmClient::from_router_with_transports(
        router.clone(),
        MAIN_POOL_ID,
        HashMap::from([("p-main".into(), main_t as Arc<dyn LlmTransportObj>)]),
    )
    .unwrap();
    let sub = Arc::new(
        LlmClient::from_router_with_transports(
            router,
            SUBAGENT_POOL_ID,
            HashMap::from([("p-sub".into(), sub_t as Arc<dyn LlmTransportObj>)]),
        )
        .unwrap(),
    );

    let manager = Arc::new(TaskManager::new(
        engine_cfg(dir.path()),
        sub,
        Arc::new(Sandbox::new(dir.path()).unwrap()),
        Arc::new(ToolRegistry::new()),
    ));
    assert_eq!(manager.pool_id(), SUBAGENT_POOL_ID);

    manager.spawn_task(task_input("cheap work")).await.unwrap();
    assert_eq!(main_hits.load(Ordering::SeqCst), 0);
    assert_eq!(sub_hits.load(Ordering::SeqCst), 1);
    assert_eq!(*sub_models.lock().unwrap(), vec!["cheap-model".to_string()]);
}

#[tokio::test]
async fn task_json_model_cannot_target_a_main_only_model() {
    let dir = tempdir().unwrap();
    let hits = Arc::new(AtomicU32::new(0));
    let models = Arc::new(Mutex::new(Vec::new()));
    let sub_t = CountingTransport::ok("sub", hits.clone(), models.clone());

    let mut pools = BTreeMap::new();
    pools.insert(MAIN_POOL_ID.into(), pool(vec![member("p-main", 0)]));
    pools.insert(SUBAGENT_POOL_ID.into(), pool(vec![member("p-sub", 0)]));
    let routing = LlmRoutingConfig {
        providers: vec![
            target("p-main", "https://main.example/v1", "main-only-model"),
            target("p-sub", "https://cheap.example/v1", "cheap-model"),
        ],
        pools,
        health: Default::default(),
    };
    let router = LlmRouter::in_memory(routing).unwrap();
    let sub = Arc::new(
        LlmClient::from_router_with_transports(
            router,
            SUBAGENT_POOL_ID,
            HashMap::from([("p-sub".into(), sub_t as Arc<dyn LlmTransportObj>)]),
        )
        .unwrap(),
    );
    let manager = Arc::new(TaskManager::new(
        engine_cfg(dir.path()),
        sub,
        Arc::new(Sandbox::new(dir.path()).unwrap()),
        Arc::new(ToolRegistry::new()),
    ));

    let args = serde_json::json!({
        "prompt": "use the expensive model",
        "description": "bypass attempt",
        "run_in_background": false,
        "model": "main-only-model",
    });
    let input: TaskInput = serde_json::from_value(args).unwrap();
    manager.spawn_task(input).await.unwrap();
    assert_eq!(*models.lock().unwrap(), vec!["cheap-model".to_string()]);
}

#[test]
fn task_tool_schema_has_no_model_override() {
    let dir = tempdir().unwrap();
    let client = Arc::new(LlmClient::new(
        MockTransport::new(vec![MockTurn::text("ok")]),
        0,
    ));
    let manager = Arc::new(TaskManager::new(
        engine_cfg(dir.path()),
        client,
        Arc::new(Sandbox::new(dir.path()).unwrap()),
        Arc::new(ToolRegistry::new()),
    ));
    let spec = TaskTool::new(manager).spec();
    let props = spec.parameters["properties"].as_object().unwrap();
    assert!(
        !props.contains_key("model"),
        "task tool must not expose a free-form model override: {props:?}"
    );
}

#[tokio::test]
async fn subagent_trip_of_shared_id_affects_main_selection() {
    tokio::time::pause();
    let dir = tempdir().unwrap();
    let shared_hits = Arc::new(AtomicU32::new(0));
    let backup_hits = Arc::new(AtomicU32::new(0));
    let cheap_hits = Arc::new(AtomicU32::new(0));
    let unused = Arc::new(Mutex::new(Vec::new()));
    let shared = CountingTransport::failing(
        "shared",
        u32::MAX,
        "status=503 busy",
        shared_hits.clone(),
        unused.clone(),
    );
    let backup = CountingTransport::ok("backup", backup_hits.clone(), unused.clone());
    let cheap = CountingTransport::ok("cheap", cheap_hits.clone(), unused);

    let mut pools = BTreeMap::new();
    pools.insert(
        MAIN_POOL_ID.into(),
        pool(vec![member("p-shared", 0), member("p-backup", 1)]),
    );
    pools.insert(
        SUBAGENT_POOL_ID.into(),
        pool(vec![member("p-shared", 0), member("p-cheap", 1)]),
    );
    let routing = LlmRoutingConfig {
        providers: vec![
            target("p-shared", "https://shared.example/v1", "m"),
            target("p-backup", "https://backup.example/v1", "m"),
            target("p-cheap", "https://cheap.example/v1", "m"),
        ],
        pools,
        health: Default::default(),
    };
    let router = LlmRouter::in_memory(routing).unwrap();
    let main = LlmClient::from_router_with_transports(
        router.clone(),
        MAIN_POOL_ID,
        HashMap::from([
            (
                "p-shared".into(),
                shared.clone() as Arc<dyn LlmTransportObj>,
            ),
            ("p-backup".into(), backup as Arc<dyn LlmTransportObj>),
        ]),
    )
    .unwrap();
    let sub = Arc::new(
        LlmClient::from_router_with_transports(
            router,
            SUBAGENT_POOL_ID,
            HashMap::from([
                ("p-shared".into(), shared as Arc<dyn LlmTransportObj>),
                ("p-cheap".into(), cheap as Arc<dyn LlmTransportObj>),
            ]),
        )
        .unwrap(),
    );
    let manager = Arc::new(TaskManager::new(
        engine_cfg(dir.path()),
        sub,
        Arc::new(Sandbox::new(dir.path()).unwrap()),
        Arc::new(ToolRegistry::new()),
    ));

    let spawned = manager.spawn_task(task_input("trip shared")).await.unwrap();
    assert!(spawned.contains("subagent done"));
    let after_sub = shared_hits.load(Ordering::SeqCst);
    assert!(after_sub >= 1);
    assert_eq!(cheap_hits.load(Ordering::SeqCst), 1);

    let observer = phone_buddy::events::RecordingObserver::new();
    let req = ConversationRequest {
        model: "m".into(),
        items: vec![phone_buddy::conversation::ConversationItem::user("hi")],
        stream: Some(true),
        tools: None,
        tool_choice: None,
        temperature: None,
        max_tokens: None,
        reasoning_effort: None,
        search_parameters: None,
        hosted_tools: Vec::new(),
        previous_response_id: None,
        image_bytes: Default::default(),
        audio_bytes: Default::default(),
    };
    main.complete(&req, &observer).await.unwrap();
    assert_eq!(shared_hits.load(Ordering::SeqCst), after_sub);
    assert_eq!(backup_hits.load(Ordering::SeqCst), 1);
}
