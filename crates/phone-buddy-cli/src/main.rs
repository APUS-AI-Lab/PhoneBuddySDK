//! PhoneBuddy demo CLI.
//!
//! Subcommands:
//! - `mock`      — fully offline end-to-end demo with a scripted LLM:
//!                 Excel analysis via the agent loop (no API key needed).
//! - `self-test` — exercise the built-in tools directly (no LLM at all).
//! - `chat`      — real LLM mode; reads PHONEBUDDY_API_KEY,
//!                 PHONEBUDDY_BASE_URL (default https://api.x.ai/v1),
//!                 PHONEBUDDY_MODEL (default grok-3).

use std::path::PathBuf;
use std::sync::Arc;

use phone_buddy::engine::PhoneBuddyEngine;
use phone_buddy::events::{AgentEvent, AgentObserver};
use phone_buddy::llm::{MockTransport, MockTurn};
use phone_buddy::prelude::*;

struct PrintObserver;

impl AgentObserver for PrintObserver {
    fn on_event(&self, event: AgentEvent) {
        match event {
            AgentEvent::TextDelta { text } => {
                print!("{text}");
                use std::io::Write as _;
                std::io::stdout().flush().ok();
            }
            AgentEvent::ReasoningDelta { .. } => {}
            AgentEvent::ToolCallStart {
                name,
                arguments_json,
                ..
            } => {
                println!("\n\x1b[2m  ▶ tool call: {name}({arguments_json:.160})\x1b[0m");
            }
            AgentEvent::ToolCallResult {
                name, ok, output, ..
            } => {
                let mark = if ok { "✓" } else { "✗" };
                let out: String = output.lines().take(6).collect::<Vec<_>>().join("\n");
                println!("\x1b[2m  {mark} {name} → {out:.400}\x1b[0m");
            }
            AgentEvent::PlanUpdated { items_json } => {
                println!("\x1b[36m  📋 plan: {items_json}\x1b[0m");
            }
            AgentEvent::Completed { .. } => println!(),
            AgentEvent::Failed { message } => eprintln!("\x1b[31m  ✗ failed: {message}\x1b[0m"),
        }
    }
}

fn demo_root() -> PathBuf {
    std::env::temp_dir().join("phone-buddy-demo")
}

fn write_sample_data(root: &std::path::Path) -> anyhow::Result<()> {
    let sales_csv = "\
month,region,product,units,unit_price
2025-01,East,PhoneCase,1200,19.9
2025-01,West,PhoneCase,900,19.9
2025-01,East,Charger,2000,12.5
2025-02,East,PhoneCase,1400,19.9
2025-02,West,PhoneCase,1100,19.9
2025-02,West,Charger,1800,12.5
2025-03,East,PhoneCase,1600,19.9
2025-03,West,Charger,2200,12.5
2025-03,East,Charger,2100,12.5
";
    let path = root.join("data").join("sales.csv");
    std::fs::create_dir_all(path.parent().unwrap())?;
    std::fs::write(&path, sales_csv)?;
    println!("sample data: {}", path.display());
    Ok(())
}

fn run_mock() -> anyhow::Result<()> {
    println!("=== PhoneBuddy offline demo (mock LLM) ===");
    let cfg = EngineConfig {
        api_key: "mock".into(),
        base_url: "http://mock.local/v1".into(),
        model: "mock-model".into(),
        root_dir: demo_root(),
        ..Default::default()
    };
    write_sample_data(&cfg.root_dir)?;

    // Scripted model: read CSV, spawn subagent task, then report.
    let turns = vec![
        MockTurn::calls(vec![(
            "call_1".into(),
            "read_file".into(),
            serde_json::json!({"path": "data/sales.csv"}),
        )]),
        MockTurn::calls(vec![(
            "call_2".into(),
            "task".into(),
            serde_json::json!({
                "prompt": "Analyze sales CSV data",
                "description": "Analyze sales CSV",
                "run_in_background": false
            }),
        )]),
        MockTurn::text(
            "## Sales Data Analysis Report\n\n\
             - Total revenue: **$224,630.00**\n\
             - East region: $121,130.00; West region: $103,500.00\n\
             - Data source: `data/sales.csv`, analyzed via subagent task.\n",
        ),
    ];
    let transport = MockTransport::new(turns);
    let engine = PhoneBuddyEngine::with_transport(cfg, transport)?;

    let outcome = engine.chat("demo-session", "Analyze the sales data in data/sales.csv", Some(Arc::new(PrintObserver)))?;
    println!("\n=== final report ===\n{}", outcome.final_text);
    println!("turns used: {}", outcome.turns_used);
    Ok(())
}

fn run_self_test() -> anyhow::Result<()> {
    println!("=== PhoneBuddy tool self-test (no LLM) ===");
    let cfg = EngineConfig {
        api_key: "unused".into(),
        base_url: "http://unused.local/v1".into(),
        model: "unused".into(),
        root_dir: demo_root(),
        ..Default::default()
    };
    let engine = PhoneBuddyEngine::new(cfg)?;
    write_sample_data(&engine.sandbox().root())?;

    // Drive the JS tool directly through the script tool implementation.
    use phone_buddy::tools::ToolCtx;
    let ctx = ToolCtx {
        sandbox: engine.sandbox().clone(),
        cancel: tokio_util::sync::CancellationToken::new(),
    };
    let rt = tokio::runtime::Runtime::new()?;

    let script = phone_buddy::tools::script::arc();
    let out = rt.block_on(script.execute(
        serde_json::json!({
            "code": "console.log('hello from script');"
        }),
        &ctx,
    ))?;
    println!("run_script output:\n{}", out.text);

    let grep = phone_buddy::tools::grep::arc();
    let out = rt.block_on(grep.execute(
        serde_json::json!({"pattern": "Charger", "path": "data", "output_mode": "count"}),
        &ctx,
    ))?;
    println!("grep Charger (count): {}", out.text);

    let busybox = phone_buddy::tools::busybox::arc();
    let out = rt.block_on(busybox.execute(
        serde_json::json!({"applet": "head", "args": ["-n", "3", "data/sales.csv"]}),
        &ctx,
    ))?;
    println!("busybox head -n 3:\n{}", out.text);

    println!("\nself-test OK");
    Ok(())
}

fn run_chat(input: &str) -> anyhow::Result<()> {
    let api_key = std::env::var("PHONEBUDDY_API_KEY")
        .ok()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow::anyhow!("set PHONEBUDDY_API_KEY to use chat mode"))?;
    let base_url = std::env::var("PHONEBUDDY_BASE_URL")
        .unwrap_or_else(|_| "https://api.x.ai/v1".into());
    let model = std::env::var("PHONEBUDDY_MODEL").unwrap_or_else(|_| "grok-3".into());

    let api_backend = match std::env::var("PHONEBUDDY_API_BACKEND").as_deref() {
        Ok("responses") => ApiBackend::Responses,
        Ok("messages") => ApiBackend::Messages,
        _ => ApiBackend::ChatCompletions,
    };

    let cfg = EngineConfig {
        api_key,
        base_url,
        model,
        root_dir: demo_root(),
        enable_web_search: std::env::var("PHONEBUDDY_WEB_SEARCH").as_deref() == Ok("1"),
        api_backend,
        ..Default::default()
    };
    let engine = PhoneBuddyEngine::new(cfg)?;
    let outcome = engine.chat("cli-session", input, Some(Arc::new(PrintObserver)))?;
    println!("\n=== final report ===\n{}", outcome.final_text);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        None | Some("mock") => run_mock(),
        Some("self-test") => run_self_test(),
        Some("chat") => {
            let input = args.collect::<Vec<_>>().join(" ");
            if input.is_empty() {
                anyhow::bail!("usage: phone-buddy-demo chat \"<your task>\"");
            }
            run_chat(&input)
        }
        Some(other) => anyhow::bail!("unknown subcommand '{other}' (expected mock|self-test|chat)"),
    }
}
