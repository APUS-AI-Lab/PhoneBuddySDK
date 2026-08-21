//! System prompt assembly.
//!
//! Lean identity, environment, safety, and working rules. Tool HOWTO lives
//! on tool descriptions. Product copy goes in `system_prompt_extra`.
//! Hosts set the identity via [`EngineConfig::agent_name`].

use crate::config::EngineConfig;

/// Runtime overrides applied on top of the engine config when rendering.
#[derive(Debug, Clone)]
pub struct PromptRuntime {
    pub agent_name: String,
    pub extra: Option<String>,
}

impl PromptRuntime {
    pub fn from_config(cfg: &EngineConfig) -> Self {
        Self {
            agent_name: cfg.resolved_agent_name(),
            extra: cfg
                .system_prompt_extra
                .as_ref()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
        }
    }

    pub fn apply_to(&self, cfg: &mut EngineConfig) {
        cfg.agent_name = self.agent_name.clone();
        cfg.system_prompt_extra = self.extra.clone();
    }
}

#[derive(Clone, Copy)]
enum Audience {
    Primary,
    Subagent,
}

/// System prompt for the top-level interactive session.
pub fn build_system_prompt(cfg: &EngineConfig) -> String {
    assemble(cfg, Audience::Primary)
}

/// Shorter prompt for in-memory subagents. No nested-task guidance.
pub fn build_subagent_prompt(cfg: &EngineConfig) -> String {
    assemble(cfg, Audience::Subagent)
}

fn assemble(cfg: &EngineConfig, audience: Audience) -> String {
    let name = cfg.resolved_agent_name();
    let answer_language = if cfg.locale.starts_with("zh") {
        "Default to Chinese (简体中文). Follow an explicit language request if the user switches."
    } else {
        "Default to English. Follow an explicit language request if the user switches."
    };
    let current_date = chrono::Utc::now().format("%Y-%m-%d").to_string();

    let identity = match audience {
        Audience::Primary => format!(
            "You are {name}, a mobile assistant that completes tasks with the tools in this sandbox. Complete the user's request."
        ),
        Audience::Subagent => format!(
            "You are a {name} subagent — a focused worker delegated a specific task. Complete the assigned task directly. Do not broaden scope. Do not launch further subagents. Report results clearly."
        ),
    };

    let mut prompt = format!(
        r#"{identity}

{answer_language}

Today's date is {current_date}.

<environment>
You run inside a mobile app sandbox. File tools only see the app workspace.
There is no OS shell and no child processes. Do not attempt bash or other shell commands.
For computation, write JavaScript and call run_script.
For file work, prefer read_file, edit_file, grep, and list_dir; use busybox applets only when no dedicated tool fits.
Subagents run in-memory via the task tool.
</environment>

<action_safety>
Reversible local work (reading, computing, writing new files) is fine to do freely.
Before deleting or overwriting existing user files, sending content off-device, or posting a user-visible notification, say what you will do.
One approval is not a blank check.
</action_safety>

<working>
Keep going until the request is done. Do not end a turn on a promise.
If a tool fails, report the failure with its output; do not claim success.
Do exactly what was asked — do not quietly narrow or widen the scope.
When you call tools, include a brief one-sentence preamble in the same message.
Skip the plan tool for simple single-step requests.
Keep replies proportional to the task; do not default to a formal report.
</working>
"#
    );

    if matches!(audience, Audience::Primary) {
        prompt.push_str(
            "\nUse `web_search` for current events, documentation, or facts beyond training knowledge. Cite used URLs as markdown links.\n",
        );
    }

    if let Some(extra) = &cfg.system_prompt_extra {
        let extra = extra.trim();
        if !extra.is_empty() {
            prompt.push_str("\n<product_instructions>\n");
            prompt.push_str(extra);
            prompt.push_str("\n</product_instructions>\n");
        }
    }
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{resolve_agent_name, EngineConfig, DEFAULT_AGENT_NAME};

    fn cfg() -> EngineConfig {
        EngineConfig {
            api_key: "test".into(),
            model: "test-model".into(),
            locale: "en".into(),
            ..Default::default()
        }
    }

    #[test]
    fn default_identity_is_phonebuddy() {
        let prompt = build_system_prompt(&cfg());
        assert!(prompt.starts_with("You are PhoneBuddy, a mobile assistant"));
        assert!(!prompt.contains("CRITICAL"));
        assert!(!prompt.contains("final report"));
        assert!(!prompt.contains("MUST use"));
    }

    #[test]
    fn custom_agent_name_is_interpolated() {
        let mut c = cfg();
        c.agent_name = "小智".into();
        let prompt = build_system_prompt(&c);
        assert!(prompt.starts_with("You are 小智, a mobile assistant"));
        assert!(!prompt.contains("You are PhoneBuddy"));
    }

    #[test]
    fn empty_and_multiline_names_are_sanitized() {
        assert_eq!(resolve_agent_name(""), DEFAULT_AGENT_NAME);
        assert_eq!(resolve_agent_name("   "), DEFAULT_AGENT_NAME);
        assert_eq!(resolve_agent_name("Acme\ninject"), "Acme");
        assert_eq!(resolve_agent_name(" Pal ").as_str(), "Pal");
    }

    #[test]
    fn web_search_line_is_on_primary_prompt() {
        let mut c = cfg();
        c.enable_web_search = false;
        assert!(build_system_prompt(&c).contains("web_search"));
        c.enable_web_search = true;
        assert!(build_system_prompt(&c).contains("web_search"));
        assert!(!build_subagent_prompt(&c).contains("web_search"));
    }

    #[test]
    fn extra_instructions_are_appended() {
        let mut c = cfg();
        c.system_prompt_extra = Some("  Speak like a concierge.  ".into());
        let prompt = build_system_prompt(&c);
        assert!(prompt.contains("<product_instructions>"));
        assert!(prompt.contains("Speak like a concierge."));
    }

    #[test]
    fn zh_locale_defaults_to_chinese() {
        let mut c = cfg();
        c.locale = "zh-CN".into();
        let prompt = build_system_prompt(&c);
        assert!(prompt.contains("Default to Chinese"));
        assert!(!prompt.contains("Always answer"));
    }

    #[test]
    fn subagent_prompt_forbids_nesting_and_uses_name() {
        let mut c = cfg();
        c.agent_name = "Pal".into();
        let prompt = build_subagent_prompt(&c);
        assert!(prompt.contains("You are a Pal subagent"));
        assert!(prompt.contains("Do not launch further subagents"));
        assert!(!prompt.contains("run_in_background"));
        assert!(!prompt.contains("resume_from"));
    }

    #[test]
    fn runtime_overrides_apply() {
        let mut c = cfg();
        let rt = PromptRuntime {
            agent_name: "Acme".into(),
            extra: Some("Be brief.".into()),
        };
        rt.apply_to(&mut c);
        let prompt = build_system_prompt(&c);
        assert!(prompt.starts_with("You are Acme,"));
        assert!(prompt.contains("Be brief."));
    }
}
