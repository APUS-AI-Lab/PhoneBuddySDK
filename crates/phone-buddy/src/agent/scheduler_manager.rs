//! In-memory and persisted Scheduled Task Manager.
//!
//! Manages scheduled background agent tasks. Tasks are stored in memory and
//! persisted to `scheduler.json` inside the sandbox directory.
//! When a task is created or deleted, event notifications are dispatched to
//! host OS native schedulers (iOS BGTaskScheduler/Notifications, Android WorkManager/AlarmManager).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::error::{EngineError, EngineResult};
use crate::tools::fs::Sandbox;
use crate::tools::host::HostToolHub;

/// Status of a scheduled task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduledStatus {
    Scheduled,
    Completed,
    Cancelled,
}

/// Record for a scheduled task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledTaskItem {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub cron_or_time: String,
    #[serde(default)]
    pub recurring: bool,
    pub created_at: String,
    pub status: ScheduledStatus,
}

pub struct SchedulerManager {
    sandbox: Arc<Sandbox>,
    tasks: Arc<Mutex<HashMap<String, ScheduledTaskItem>>>,
    counter: Arc<Mutex<u64>>,
}

impl SchedulerManager {
    pub fn new(sandbox: Arc<Sandbox>) -> Self {
        let manager = Self {
            sandbox,
            tasks: Arc::new(Mutex::new(HashMap::new())),
            counter: Arc::new(Mutex::new(1)),
        };
        let _ = manager.load_from_disk();
        manager
    }

    fn generate_id(&self) -> String {
        let mut c = self.counter.lock().unwrap();
        let id = format!("sched-{}", *c);
        *c += 1;
        id
    }

    fn save_to_disk(&self) -> EngineResult<()> {
        let tasks = self.tasks.lock().unwrap();
        let items: Vec<ScheduledTaskItem> = tasks.values().cloned().collect();
        let json = serde_json::to_string_pretty(&items).unwrap_or_else(|_| "[]".into());
        let path = self.sandbox.resolve("scheduler.json")?;
        std::fs::write(path, json)?;
        Ok(())
    }

    fn load_from_disk(&self) -> EngineResult<()> {
        if let Ok(path) = self.sandbox.resolve("scheduler.json") {
            if let Ok(content) = std::fs::read_to_string(path) {
                if let Ok(items) = serde_json::from_str::<Vec<ScheduledTaskItem>>(&content) {
                    let mut map = self.tasks.lock().unwrap();
                    let mut max_id = 0u64;
                    for item in items {
                        if let Some(num_str) = item.id.strip_prefix("sched-") {
                            if let Ok(num) = num_str.parse::<u64>() {
                                if num > max_id {
                                    max_id = num;
                                }
                            }
                        }
                        map.insert(item.id.clone(), item);
                    }
                    *self.counter.lock().unwrap() = max_id + 1;
                }
            }
        }
        Ok(())
    }

    pub fn create_task(
        &self,
        title: Option<String>,
        prompt: String,
        cron_or_time: Option<String>,
        recurring: bool,
        host_tools: &HostToolHub,
    ) -> EngineResult<ScheduledTaskItem> {
        let id = self.generate_id();
        let cron_or_time = cron_or_time.unwrap_or_else(|| "in 5m".to_string());
        let item = ScheduledTaskItem {
            id: id.clone(),
            title,
            prompt,
            cron_or_time,
            recurring,
            created_at: Utc::now().to_rfc3339(),
            status: ScheduledStatus::Scheduled,
        };

        {
            let mut map = self.tasks.lock().unwrap();
            map.insert(id.clone(), item.clone());
        }
        let _ = self.save_to_disk();

        let event_payload = serde_json::json!({
            "event": "registered",
            "task": item
        });
        host_tools.notify_event("scheduler_registered", &event_payload.to_string());

        Ok(item)
    }

    pub fn list_tasks(&self) -> Vec<ScheduledTaskItem> {
        let map = self.tasks.lock().unwrap();
        let mut list: Vec<ScheduledTaskItem> = map.values().cloned().collect();
        list.sort_by(|a, b| a.id.cmp(&b.id));
        list
    }

    pub fn delete_task(
        &self,
        task_id: &str,
        host_tools: &HostToolHub,
    ) -> EngineResult<ScheduledTaskItem> {
        let mut map = self.tasks.lock().unwrap();
        if let Some(mut item) = map.remove(task_id) {
            item.status = ScheduledStatus::Cancelled;
            let event_payload = serde_json::json!({
                "event": "cancelled",
                "task_id": task_id
            });
            host_tools.notify_event("scheduler_cancelled", &event_payload.to_string());
            drop(map);
            let _ = self.save_to_disk();
            Ok(item)
        } else {
            Err(EngineError::ToolArgs {
                name: "scheduler".into(),
                message: format!("scheduled task '{task_id}' not found"),
            })
        }
    }
}
