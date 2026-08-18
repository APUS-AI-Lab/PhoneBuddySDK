//
//  DemoApp.swift
//  PhoneBuddy iOS Demo
//
//  Demonstrates how to integrate the PhoneBuddy engine library into an iOS app:
//  - Real-time detailed progress display (searching, fetching, tool execution & results)
//  - Interactive Settings sheet for API Key, Base URL, Model, Backend protocol, Max turns
//  - Session history selection, resume & replay
//  - Headless WKWebView for web_search & web_fetch tools
//  - Real-time streaming reasoning, text deltas, tool calls & plan updates
//  - Interactive clarification questions (ask_user_question)
//  - In-flight turn cancellation
//

import SwiftUI
import UniformTypeIdentifiers

@main
struct PhoneBuddyDemoApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

// MARK: - Message Data Models

enum MessageRole: String {
    case user
    case assistant
    case toolCall
    case toolResult
    case plan
    case system
}

struct PlanItem: Identifiable {
    let id: String
    let content: String
    let status: String
}

struct UiChatMessage: Identifiable {
    let id = UUID()
    let role: MessageRole
    var text: String = ""
    var reasoning: String? = nil
    var callId: String? = nil
    var toolName: String? = nil
    var toolArgs: String? = nil
    var toolResult: String? = nil
    var toolSuccess: Bool = true
    var isRunning: Bool = false
    var planItems: [PlanItem] = []
    var tokenUsage: String? = nil
    var isThinkingExpanded: Bool = true
    var isOutputExpanded: Bool = false
}

struct ClarificationQuestion: Identifiable {
    let id = UUID()
    let callId: String
    let question: String
    let options: [String]
}

// MARK: - Tool Information Helper

struct ToolSummaryInfo {
    let icon: String
    let title: String
    let primaryParam: String?
    let detail: String?
}

func parseToolSummary(name: String, argsJson: String) -> ToolSummaryInfo {
    guard let data = argsJson.data(using: .utf8),
          let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        return ToolSummaryInfo(
            icon: "wrench.and.screwdriver.fill",
            title: "Tool: \(name)",
            primaryParam: nil,
            detail: argsJson.isEmpty ? nil : argsJson
        )
    }

    switch name {
    case "web_search":
        let query = obj["query"] as? String ?? obj["search_query"] as? String ?? ""
        return ToolSummaryInfo(
            icon: "magnifyingglass",
            title: "Web Search",
            primaryParam: query.isEmpty ? nil : "\"\(query)\"",
            detail: nil
        )
    case "web_fetch":
        let url = obj["url"] as? String ?? ""
        return ToolSummaryInfo(
            icon: "globe",
            title: "Web Fetch",
            primaryParam: url.isEmpty ? nil : url,
            detail: nil
        )
    case "read_file":
        let path = obj["path"] as? String ?? obj["file_path"] as? String ?? ""
        return ToolSummaryInfo(
            icon: "doc.text.fill",
            title: "Read File",
            primaryParam: path.isEmpty ? nil : path,
            detail: nil
        )
    case "write_file":
        let path = obj["path"] as? String ?? obj["file_path"] as? String ?? ""
        return ToolSummaryInfo(
            icon: "square.and.pencil",
            title: "Write File",
            primaryParam: path.isEmpty ? nil : path,
            detail: nil
        )
    case "edit_file":
        let path = obj["path"] as? String ?? obj["file_path"] as? String ?? ""
        return ToolSummaryInfo(
            icon: "pencil.tip.crop.circle",
            title: "Edit File",
            primaryParam: path.isEmpty ? nil : path,
            detail: nil
        )
    case "grep_search":
        let query = obj["query"] as? String ?? ""
        let path = obj["path"] as? String ?? ""
        let summary = path.isEmpty ? "\"\(query)\"" : "\"\(query)\" in \(path)"
        return ToolSummaryInfo(
            icon: "text.magnifyingglass",
            title: "Grep Search",
            primaryParam: summary,
            detail: nil
        )
    case "list_dir":
        let path = obj["path"] as? String ?? obj["directory"] as? String ?? "."
        return ToolSummaryInfo(
            icon: "folder.fill",
            title: "List Directory",
            primaryParam: path,
            detail: nil
        )
    case "plan":
        return ToolSummaryInfo(
            icon: "list.bullet.clipboard.fill",
            title: "Execution Plan",
            primaryParam: "Updating task plan...",
            detail: nil
        )
    case "task":
        let prompt = obj["prompt"] as? String ?? obj["role"] as? String ?? ""
        return ToolSummaryInfo(
            icon: "bolt.fill",
            title: "Subagent Task",
            primaryParam: prompt.isEmpty ? nil : "\"\(prompt)\"",
            detail: nil
        )
    case "ask_user_question":
        let q = obj["question"] as? String ?? ""
        return ToolSummaryInfo(
            icon: "questionmark.circle.fill",
            title: "Clarification Question",
            primaryParam: q.isEmpty ? nil : q,
            detail: nil
        )
    default:
        return ToolSummaryInfo(
            icon: "gearshape.fill",
            title: "Tool: \(name)",
            primaryParam: nil,
            detail: argsJson
        )
    }
}

// MARK: - Main ContentView

struct ContentView: View {
    @StateObject private var viewModel = ChatViewModel()
    @State private var showingSessionDrawer = false
    @State private var showingSettings = false

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Top real-time progress banner
                if viewModel.isProcessing {
                    HStack(spacing: 8) {
                        ProgressView()
                            .scaleEffect(0.8)

                        Text(viewModel.currentProgressText.isEmpty ? "Agent is working..." : viewModel.currentProgressText)
                            .font(.caption.weight(.medium))
                            .foregroundColor(.primary)
                            .lineLimit(1)
                            .truncationMode(.tail)

                        Spacer()

                        Button(action: { viewModel.cancelCurrentTurn() }) {
                            HStack(spacing: 4) {
                                Image(systemName: "stop.circle.fill")
                                Text("Stop")
                            }
                            .font(.caption.bold())
                            .foregroundColor(.red)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.red.opacity(0.12))
                            .cornerRadius(8)
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color(.systemGray6))
                    Divider()
                }

                // Chat message stream
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 12) {
                            ForEach(viewModel.messages) { msg in
                                MessageBubbleView(
                                    msg: msg,
                                    onToggleThinking: { viewModel.toggleThinking(for: msg.id) },
                                    onToggleOutput: { viewModel.toggleOutput(for: msg.id) }
                                )
                                .id(msg.id)
                            }
                            Color.clear
                                .frame(height: 1)
                                .id("bottom_anchor")
                        }
                        .padding()
                    }
                    .onChange(of: viewModel.messages.count) { _ in
                        proxy.scrollTo("bottom_anchor", anchor: .bottom)
                    }
                    .onChange(of: viewModel.messages.last?.text) { _ in
                        proxy.scrollTo("bottom_anchor", anchor: .bottom)
                    }
                    .onChange(of: viewModel.messages.last?.reasoning) { _ in
                        proxy.scrollTo("bottom_anchor", anchor: .bottom)
                    }
                }

                // Interactive Clarification Prompt (ask_user_question)
                if let question = viewModel.pendingClarification {
                    ClarificationPromptView(
                        question: question,
                        onSubmit: { answer in
                            viewModel.answerClarification(callId: question.callId, answer: answer)
                        }
                    )
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                }

                // Quick Suggestion Chips (Search, Fetch, Plan, Clarification)
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        SuggestionChip(title: "🔍 Search Rust 2026") {
                            viewModel.inputText = "Search the web for the latest updates on Rust language in 2026"
                        }
                        SuggestionChip(title: "🌐 WebFetch HackerNews") {
                            viewModel.inputText = "Fetch https://news.ycombinator.com and summarize the top 3 stories"
                        }
                        SuggestionChip(title: "📋 Plan Research") {
                            viewModel.inputText = "Create a multi-step research plan to analyze quantum computing breakthroughs"
                        }
                        SuggestionChip(title: "❓ Test Clarification") {
                            viewModel.inputText = "Ask me clarifying questions about my favorite programming tech stack"
                        }
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 4)
                }

                Divider()

                // Input bar
                HStack(spacing: 10) {
                    TextField("Ask a question, search, or run tasks...", text: $viewModel.inputText)
                        .textFieldStyle(.roundedBorder)
                        .disabled(viewModel.isProcessing)

                    Button(action: { viewModel.sendMessage() }) {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.system(size: 32))
                            .foregroundColor(viewModel.canSend ? .blue : .gray)
                    }
                    .disabled(!viewModel.canSend)
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
                .background(Color(.systemBackground))
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button(action: {
                        viewModel.loadSessionsList()
                        showingSessionDrawer = true
                    }) {
                        HStack(spacing: 6) {
                            Image(systemName: "clock.arrow.circlepath")
                            Text("Sessions")
                                .font(.subheadline.bold())
                        }
                    }
                }

                ToolbarItem(placement: .principal) {
                    VStack(spacing: 2) {
                        Text("PhoneBuddy Agent")
                            .font(.headline)
                        Text("\(viewModel.currentConfig.model) • \(viewModel.currentConfig.apiBackend)")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }

                ToolbarItem(placement: .navigationBarTrailing) {
                    HStack(spacing: 12) {
                        Button(action: { showingSettings = true }) {
                            Image(systemName: "gearshape")
                                .font(.system(size: 16, weight: .semibold))
                        }

                        Button(action: { viewModel.startNewSession() }) {
                            Image(systemName: "square.and.pencil")
                                .font(.system(size: 16, weight: .bold))
                        }
                    }
                }
            }
            .sheet(isPresented: $showingSessionDrawer) {
                SessionHistorySheet(
                    viewModel: viewModel,
                    isPresented: $showingSessionDrawer
                )
            }
            .sheet(isPresented: $showingSettings) {
                SettingsSheetView(
                    config: $viewModel.currentConfig,
                    onSave: { updatedConfig in
                        viewModel.applyNewConfig(updatedConfig)
                        showingSettings = false
                    },
                    isPresented: $showingSettings
                )
            }
        }
        .onAppear {
            viewModel.initialize()
        }
    }
}

// MARK: - Suggestion Chip

struct SuggestionChip: View {
    let title: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.caption)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(Color(.systemGray6))
                .foregroundColor(.primary)
                .cornerRadius(14)
        }
    }
}

// MARK: - Message Bubble Views

struct MessageBubbleView: View {
    let msg: UiChatMessage
    let onToggleThinking: () -> Void
    let onToggleOutput: () -> Void

    var body: some View {
        switch msg.role {
        case .user:
            HStack {
                Spacer()
                Text(msg.text)
                    .padding(12)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(16, corners: [.topLeft, .topRight, .bottomLeft])
            }

        case .assistant:
            VStack(alignment: .leading, spacing: 6) {
                if let reasoning = msg.reasoning, !reasoning.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("💭 Thinking...")
                                .font(.caption.bold())
                                .foregroundColor(.secondary)
                            Spacer()
                            Image(systemName: msg.isThinkingExpanded ? "chevron.up" : "chevron.down")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                        .contentShape(Rectangle())
                        .onTapGesture { onToggleThinking() }

                        if msg.isThinkingExpanded {
                            Text(reasoning)
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundColor(Color(.secondaryLabel))
                                .padding(8)
                                .background(Color(.systemGray6))
                                .cornerRadius(8)
                        }
                    }
                    .padding(8)
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(10)
                }

                if !msg.text.isEmpty {
                    Text(msg.text)
                        .padding(12)
                        .background(Color(.systemGray5))
                        .foregroundColor(.primary)
                        .cornerRadius(16, corners: [.topLeft, .topRight, .bottomRight])
                }

                if let usage = msg.tokenUsage {
                    Text(usage)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .padding(.leading, 4)
                }
            }

        case .toolCall:
            let info = parseToolSummary(name: msg.toolName ?? "tool", argsJson: msg.toolArgs ?? "")
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Image(systemName: info.icon)
                        .font(.caption.bold())
                        .foregroundColor(.orange)

                    Text(info.title)
                        .font(.caption.bold())
                        .foregroundColor(.orange)

                    if msg.isRunning {
                        ProgressView()
                            .scaleEffect(0.6)
                        Text("running...")
                            .font(.caption2.italic())
                            .foregroundColor(.secondary)
                    }

                    Spacer()
                }

                if let param = info.primaryParam, !param.isEmpty {
                    Text(param)
                        .font(.system(size: 12, weight: .medium, design: .monospaced))
                        .foregroundColor(.primary)
                        .lineLimit(2)
                } else if let detail = info.detail, !detail.isEmpty {
                    Text(detail)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(.secondary)
                        .lineLimit(2)
                }
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.orange.opacity(0.1))
            .cornerRadius(10)
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .stroke(Color.orange.opacity(0.25), lineWidth: 1)
            )

        case .toolResult:
            let isLong = (msg.toolResult ?? "").count > 200 || (msg.toolResult ?? "").contains("\n")
            VStack(alignment: .leading, spacing: 4) {
                HStack(alignment: .center, spacing: 6) {
                    Text(msg.toolSuccess ? "✓" : "✗")
                        .font(.caption.bold())
                        .foregroundColor(msg.toolSuccess ? .green : .red)

                    Text(msg.toolSuccess ? "\(msg.toolName ?? "tool") result" : "\(msg.toolName ?? "tool") failed")
                        .font(.caption.bold())
                        .foregroundColor(msg.toolSuccess ? .green : .red)

                    Spacer()

                    if isLong {
                        Button(action: onToggleOutput) {
                            Text(msg.isOutputExpanded ? "Collapse" : "Show full")
                                .font(.caption2.bold())
                                .foregroundColor(msg.toolSuccess ? .green : .red)
                        }
                    }
                }

                if let res = msg.toolResult, !res.isEmpty {
                    Text(res)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(.secondary)
                        .lineLimit(msg.isOutputExpanded ? nil : 4)
                        .padding(6)
                        .background(Color(.systemBackground).opacity(0.6))
                        .cornerRadius(6)
                }
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background((msg.toolSuccess ? Color.green : Color.red).opacity(0.08))
            .cornerRadius(10)
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .stroke((msg.toolSuccess ? Color.green : Color.red).opacity(0.2), lineWidth: 1)
            )

        case .plan:
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Image(systemName: "list.bullet.clipboard.fill")
                        .foregroundColor(.cyan)
                    Text("📋 Execution Plan:")
                        .font(.caption.bold())
                        .foregroundColor(.cyan)
                }

                ForEach(msg.planItems) { item in
                    let icon = (item.status == "completed") ? "✓" : (item.status == "in_progress" ? "⏳" : (item.status == "cancelled" ? "✕" : "○"))
                    let col: Color = (item.status == "completed") ? .green : (item.status == "in_progress" ? .orange : (item.status == "cancelled" ? .red : .secondary))
                    HStack(alignment: .top, spacing: 6) {
                        Text(icon)
                            .font(.caption.bold())
                            .foregroundColor(col)
                        Text("[\(item.id)] \(item.content)")
                            .font(.caption)
                            .foregroundColor(col)
                    }
                }
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.cyan.opacity(0.08))
            .cornerRadius(10)
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .stroke(Color.cyan.opacity(0.2), lineWidth: 1)
            )

        case .system:
            HStack {
                Spacer()
                Text(msg.text)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
                Spacer()
            }
        }
    }
}

// MARK: - Clarification Prompt View

struct ClarificationPromptView: View {
    let question: ClarificationQuestion
    let onSubmit: (String) -> Void
    @State private var textInput = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("❓")
                Text("Agent Clarification Request")
                    .font(.subheadline.bold())
                    .foregroundColor(.purple)
            }

            Text(question.question)
                .font(.body)
                .foregroundColor(.primary)

            if !question.options.isEmpty {
                VStack(spacing: 6) {
                    ForEach(Array(question.options.enumerated()), id: \.offset) { index, opt in
                        Button(action: { onSubmit(opt) }) {
                            HStack {
                                Text("\(index + 1)) \(opt)")
                                    .font(.subheadline)
                                Spacer()
                            }
                            .padding(8)
                            .background(Color.purple.opacity(0.1))
                            .cornerRadius(8)
                        }
                    }
                }
            }

            HStack {
                TextField("Your response...", text: $textInput)
                    .textFieldStyle(.roundedBorder)

                Button(action: {
                    let ans = textInput.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !ans.isEmpty {
                        onSubmit(ans)
                    }
                }) {
                    Text("Reply")
                        .font(.subheadline.bold())
                        .foregroundColor(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color.purple)
                        .cornerRadius(8)
                }
                .disabled(textInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding()
        .background(Color.purple.opacity(0.05))
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.purple.opacity(0.3), lineWidth: 1)
        )
        .padding(.horizontal)
    }
}

// MARK: - Settings Sheet View

struct SettingsSheetView: View {
    @Binding var config: PhoneBuddyConfig
    let onSave: (PhoneBuddyConfig) -> Void
    @Binding var isPresented: Bool

    @State private var apiKey: String = ""
    @State private var baseUrl: String = ""
    @State private var model: String = ""
    @State private var apiBackend: String = "responses"
    @State private var maxTurns: Int = 24
    @State private var enableWebSearch: Bool = true
    @State private var extraHeaders: [String: String]? = nil
    @State private var extraBody: [String: String]? = nil
    @State private var sandboxRoot: String = PhoneBuddyConfig.sandboxRoot(workspaceName: PhoneBuddyConfig.defaultWorkspaceName)
    @State private var isApiKeyVisible: Bool = false
    @State private var showingFileImporter: Bool = false
    @State private var importStatusMessage: String? = nil

    var body: some View {
        NavigationView {
            Form {
                // Section: Import from External File
                Section(header: Text("Import config.json File"), footer: Text("Tip: Put config.json in Files App -> 'On My iPhone' -> 'PhoneBuddy', or pick from iCloud/Downloads.")) {
                    Button(action: loadFromAppDocuments) {
                        HStack {
                            Image(systemName: "folder.badge.gearshape")
                                .foregroundColor(.blue)
                            Text("Load from App Documents (config.json)")
                                .foregroundColor(.primary)
                        }
                    }

                    Button(action: { showingFileImporter = true }) {
                        HStack {
                            Image(systemName: "doc.badge.plus")
                                .foregroundColor(.purple)
                            Text("Browse & Import config.json...")
                                .foregroundColor(.purple)
                        }
                    }

                    if let status = importStatusMessage {
                        Text(status)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }

                // Section: Provider Presets
                Section(header: Text("Quick Provider Presets")) {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            Button("🚀 xAI Grok") {
                                baseUrl = "https://api.x.ai/v1"
                                model = "grok-4.6"
                                apiBackend = "responses"
                                enableWebSearch = true
                            }
                            .buttonStyle(.bordered)

                            Button("⚡ DeepSeek") {
                                baseUrl = "https://api.deepseek.com/v1"
                                model = "deepseek-v4-flash"
                                apiBackend = "chat_completions"
                                enableWebSearch = false
                            }
                            .buttonStyle(.bordered)

                            Button("🧠 OpenAI") {
                                baseUrl = "https://api.openai.com/v1"
                                model = "gpt-5.6-sol"
                                apiBackend = "responses"
                                enableWebSearch = false
                            }
                            .buttonStyle(.bordered)

                            Button("🎭 Anthropic") {
                                baseUrl = "https://api.anthropic.com/v1"
                                model = "claude-fable-5"
                                apiBackend = "messages"
                                enableWebSearch = false
                            }
                            .buttonStyle(.bordered)

                            Button("🦙 Ollama Local") {
                                baseUrl = "http://localhost:11434/v1"
                                model = "llama3.1"
                                apiBackend = "chat_completions"
                                enableWebSearch = false
                            }
                            .buttonStyle(.bordered)

                            Button("⚡ vLLM Local") {
                                baseUrl = "http://localhost:8000/v1"
                                model = "deepseek-v4-flash"
                                apiBackend = "chat_completions"
                                enableWebSearch = false
                            }
                            .buttonStyle(.bordered)
                        }
                        .padding(.vertical, 4)
                    }
                }

                // Section: API Credentials
                Section(header: Text("API Credentials"), footer: Text("Refer to config.json.example for reference.")) {
                    HStack {
                        if isApiKeyVisible {
                            TextField("API Key", text: $apiKey)
                                .textInputAutocapitalization(.never)
                                .disableAutocorrection(true)
                        } else {
                            SecureField("API Key", text: $apiKey)
                        }

                        Button(action: { isApiKeyVisible.toggle() }) {
                            Image(systemName: isApiKeyVisible ? "eye.slash" : "eye")
                                .foregroundColor(.secondary)
                        }
                    }
                }

                // Section: Endpoint & Protocol
                Section(header: Text("API Endpoint & Protocol")) {
                    TextField("Base URL (e.g. https://api.x.ai/v1)", text: $baseUrl)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .disableAutocorrection(true)

                    Picker("API Backend Protocol", selection: $apiBackend) {
                        Text("responses (OpenAI/xAI Responses API)").tag("responses")
                        Text("chat_completions (OpenAI/DeepSeek Standard)").tag("chat_completions")
                        Text("messages (Anthropic Claude)").tag("messages")
                    }

                    TextField("Model ID (e.g. grok-4.6, deepseek-v4-flash)", text: $model)
                        .textInputAutocapitalization(.never)
                        .disableAutocorrection(true)
                }

                // Section: Agent Options
                Section(header: Text("Agent Loop Settings")) {
                    Stepper("Max Turns: \(maxTurns)", value: $maxTurns, in: 1...50)

                    Toggle("Enable Web Search & Fetch", isOn: $enableWebSearch)
                    if enableWebSearch {
                        Text("Uses hidden WKWebView for dynamic web rendering, JS execution & search.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }

                // Section: Sandbox Root Info
                Section(header: Text("Sandbox Workspace"), footer: Text("`root_dir` from config.json is used as the folder name under Documents. Example: `./workspace` → Documents/workspace.")) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(URL(fileURLWithPath: sandboxRoot).lastPathComponent)
                            .font(.system(.body, design: .monospaced))
                        Text(sandboxRoot)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(.secondary)
                    }
                }

                // Section: Reset to Default
                Section {
                    Button(action: resetToDefaults) {
                        HStack {
                            Spacer()
                            Text("Reset to config.json Defaults")
                                .foregroundColor(.red)
                            Spacer()
                        }
                    }
                }
            }
            .navigationTitle("Agent Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        isPresented = false
                    }
                }

                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        var updated = config
                        updated.apiKey = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
                        updated.baseUrl = baseUrl.trimmingCharacters(in: .whitespacesAndNewlines)
                        updated.model = model.trimmingCharacters(in: .whitespacesAndNewlines)
                        updated.apiBackend = apiBackend
                        updated.maxTurns = maxTurns
                        updated.enableWebSearch = enableWebSearch
                        updated.extraHeaders = extraHeaders
                        updated.extraBody = extraBody
                        updated.rootDir = sandboxRoot
                        updated.save()
                        onSave(updated)
                    }
                    .font(.headline)
                }
            }
            .onAppear {
                apiKey = config.apiKey
                baseUrl = config.baseUrl
                model = config.model
                apiBackend = config.apiBackend
                maxTurns = config.maxTurns
                enableWebSearch = config.enableWebSearch
                extraHeaders = config.extraHeaders
                extraBody = config.extraBody
                sandboxRoot = config.rootDir.isEmpty
                    ? PhoneBuddyConfig.sandboxRoot(workspaceName: PhoneBuddyConfig.defaultWorkspaceName)
                    : config.rootDir
            }
            .fileImporter(
                isPresented: $showingFileImporter,
                allowedContentTypes: [.json, .plainText],
                allowsMultipleSelection: false
            ) { result in
                switch result {
                case .success(let urls):
                    guard let selectedUrl = urls.first else { return }
                    guard selectedUrl.startAccessingSecurityScopedResource() else {
                        importStatusMessage = "Failed to access chosen file."
                        return
                    }
                    defer { selectedUrl.stopAccessingSecurityScopedResource() }
                    do {
                        let text = try String(contentsOf: selectedUrl, encoding: .utf8)
                        let importedConfig = try PhoneBuddyConfig.fromJsonString(text)
                        applyImportedConfig(importedConfig)
                        importStatusMessage = "✓ Imported \(selectedUrl.lastPathComponent) (Model: \(importedConfig.model), Workspace: \(importedConfig.workspaceName))"
                    } catch {
                        importStatusMessage = "Failed parsing config: \(error.localizedDescription)"
                    }
                case .failure(let error):
                    importStatusMessage = "File selection error: \(error.localizedDescription)"
                }
            }
        }
    }

    private func applyImportedConfig(_ cfg: PhoneBuddyConfig) {
        apiKey = cfg.apiKey
        baseUrl = cfg.baseUrl
        model = cfg.model
        apiBackend = cfg.apiBackend
        maxTurns = cfg.maxTurns
        enableWebSearch = cfg.enableWebSearch
        extraHeaders = cfg.extraHeaders
        extraBody = cfg.extraBody
        sandboxRoot = cfg.rootDir
    }

    private func loadFromAppDocuments() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        let candidates = [
            docs.first?.appendingPathComponent("config.json"),
            docs.first?.appendingPathComponent("PhoneBuddy/config.json"),
            URL(fileURLWithPath: config.rootDir).appendingPathComponent("config.json"),
            Bundle.main.url(forResource: "config", withExtension: "json")
        ].compactMap { $0 }

        for fileUrl in candidates {
            if FileManager.default.fileExists(atPath: fileUrl.path) {
                do {
                    let text = try String(contentsOf: fileUrl, encoding: .utf8)
                    let cfg = try PhoneBuddyConfig.fromJsonString(text)
                    applyImportedConfig(cfg)
                    importStatusMessage = "✓ Loaded from \(fileUrl.lastPathComponent) (Model: \(cfg.model), Workspace: \(cfg.workspaceName))"
                    return
                } catch {
                    importStatusMessage = "Failed parsing \(fileUrl.lastPathComponent): \(error.localizedDescription)"
                    return
                }
            }
        }
        importStatusMessage = "config.json not found in App Documents. Place it in Files App -> 'On My iPhone' -> 'PhoneBuddy'."
    }

    private func resetToDefaults() {
        apiKey = ""
        baseUrl = "https://api.x.ai/v1"
        model = "grok-4.6"
        apiBackend = "responses"
        maxTurns = 24
        enableWebSearch = true
        extraHeaders = nil
        extraBody = nil
        sandboxRoot = PhoneBuddyConfig.sandboxRoot(workspaceName: PhoneBuddyConfig.defaultWorkspaceName)
        importStatusMessage = "Reset to default values."
    }
}

// MARK: - Session History Sheet

struct SessionHistorySheet: View {
    @ObservedObject var viewModel: ChatViewModel
    @Binding var isPresented: Bool

    var body: some View {
        NavigationView {
            List {
                Section(header: Text("Active Session")) {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Current Session")
                                .font(.subheadline.bold())
                            Text(viewModel.sessionId)
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                        Spacer()
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                    }
                }

                Section(header: Text("Saved Sessions (\(viewModel.savedSessions.count))")) {
                    if viewModel.savedSessions.isEmpty {
                        Text("No saved sessions.")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    } else {
                        ForEach(viewModel.savedSessions) { s in
                            Button(action: {
                                viewModel.resumeSession(sessionId: s.id)
                                isPresented = false
                            }) {
                                HStack {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(s.title.isEmpty ? "Session \(s.id.prefix(8))" : s.title)
                                            .font(.subheadline.bold())
                                            .foregroundColor(.primary)
                                        HStack(spacing: 8) {
                                            Text("\(s.messageCount) msgs")
                                                .font(.caption2)
                                                .foregroundColor(.secondary)
                                            Text("•")
                                                .font(.caption2)
                                                .foregroundColor(.secondary)
                                            Text(s.updatedAt.prefix(19))
                                                .font(.caption2)
                                                .foregroundColor(.secondary)
                                        }
                                    }
                                    Spacer()
                                    if s.id == viewModel.sessionId {
                                        Image(systemName: "checkmark")
                                            .font(.caption.bold())
                                            .foregroundColor(.blue)
                                    }
                                }
                            }
                        }
                        .onDelete { indexSet in
                            for index in indexSet {
                                let session = viewModel.savedSessions[index]
                                viewModel.deleteSession(sessionId: session.id)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Session History")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("New Chat") {
                        viewModel.startNewSession()
                        isPresented = false
                    }
                }

                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        isPresented = false
                    }
                }
            }
        }
    }
}

// MARK: - View Model

class ChatViewModel: ObservableObject {
    @Published var messages: [UiChatMessage] = []
    @Published var inputText: String = ""
    @Published var isProcessing: Bool = false
    @Published var currentProgressText: String = ""
    @Published var sessionId: String = UUID().uuidString
    @Published var savedSessions: [SessionMetadata] = []
    @Published var pendingClarification: ClarificationQuestion? = nil
    @Published var currentConfig: PhoneBuddyConfig

    private var engine: PhoneBuddyEngine?

    init() {
        self.currentConfig = PhoneBuddyConfig.loadOrDefault()
    }

    var canSend: Bool {
        !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !isProcessing
    }

    func initialize() {
        rebuildEngine()
    }

    func applyNewConfig(_ newConfig: PhoneBuddyConfig) {
        var pinned = newConfig
        pinned.pinSandboxRoot(newConfig.rootDir)
        self.currentConfig = pinned
        NSLog("[DemoApp] Applied new configuration: Model=%@, Backend=%@, BaseURL=%@, Workspace=%@", pinned.model, pinned.apiBackend, pinned.baseUrl, pinned.rootDir)
        rebuildEngine()
        messages.append(UiChatMessage(
            role: .system,
            text: "⚙️ Configuration applied:\n• Model: \(pinned.model)\n• Backend: \(pinned.apiBackend)\n• Base URL: \(pinned.baseUrl)\n• Workspace: \(pinned.workspaceName)\n• Max Turns: \(pinned.maxTurns)"
        ))
    }

    private func rebuildEngine() {
        currentConfig.pinSandboxRoot(currentConfig.rootDir)
        NSLog("[DemoApp] Rebuilding engine for model: %@, backend: %@, rootDir: %@", currentConfig.model, currentConfig.apiBackend, currentConfig.rootDir)
        do {
            let newEngine = try PhoneBuddyEngine(config: currentConfig)

            // Register host tool callback for clarification questions
            newEngine.setHostToolCallback { [weak self] callId, name, argumentsJson in
                NSLog("[DemoApp] Host tool called: %@ (callId: %@)", name, callId)
                DispatchQueue.main.async {
                    if name == "ask_user_question" {
                        guard let data = argumentsJson.data(using: .utf8),
                              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                            try? self?.engine?.completeHostTool(callId: callId, ok: false, output: "Invalid question arguments")
                            return
                        }
                        let question = obj["question"] as? String ?? "Clarification requested:"
                        let options = obj["options"] as? [String] ?? []
                        self?.pendingClarification = ClarificationQuestion(callId: callId, question: question, options: options)
                        self?.currentProgressText = "❓ Clarification requested"
                    } else {
                        try? self?.engine?.completeHostTool(callId: callId, ok: false, output: "Unsupported host tool \(name)")
                    }
                }
            }

            self.engine = newEngine
            NSLog("[DemoApp] ✓ Engine initialized successfully")

            if messages.isEmpty {
                messages.append(UiChatMessage(
                    role: .system,
                    text: "PhoneBuddy Agent ready.\nModel: \(currentConfig.model) (\(currentConfig.apiBackend))\nWorkspace: \(currentConfig.workspaceName)\nHeadless WKWebView enabled for `web_search` & `web_fetch`."
                ))
            }

            loadSessionsList()
        } catch {
            NSLog("[DemoApp] ❌ Engine initialization failed: %@", error.localizedDescription)
            let description = error.localizedDescription
            let hint: String
            if description.localizedCaseInsensitiveContains("read-only file system")
                || description.contains("os error 30") {
                hint = "The engine could not create its workspace (read-only path). `root_dir` from config.json is used only as the Documents folder name (e.g. Documents/workspace)."
            } else if description.localizedCaseInsensitiveContains("api_key")
                || description.localizedCaseInsensitiveContains("base_url") {
                hint = "Please check your API Key & Base URL in Settings (⚙️)."
            } else {
                hint = "See Settings (⚙️) if the model, backend, or credentials look wrong."
            }
            messages.append(UiChatMessage(
                role: .system,
                text: "Initialization failed: \(description)\n\(hint)"
            ))
        }
    }

    func loadSessionsList() {
        guard let engine = engine else { return }
        DispatchQueue.global(qos: .userInitiated).async {
            if let list = try? engine.listSessionItems() {
                DispatchQueue.main.async {
                    self.savedSessions = list
                }
            }
        }
    }

    func startNewSession() {
        sessionId = UUID().uuidString
        pendingClarification = nil
        currentProgressText = ""
        messages = [
            UiChatMessage(
                role: .system,
                text: "Started new session: \(sessionId)\nModel: \(currentConfig.model)"
            )
        ]
        loadSessionsList()
    }

    func resumeSession(sessionId targetId: String) {
        guard let engine = engine else { return }
        sessionId = targetId
        pendingClarification = nil
        currentProgressText = ""

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let session = try engine.getSessionData(sessionId: targetId)
                DispatchQueue.main.async {
                    var restored: [UiChatMessage] = []
                    if let session = session, !session.messages.isEmpty {
                        for m in session.messages {
                            switch m.role.lowercased() {
                            case "user":
                                restored.append(UiChatMessage(role: .user, text: m.content ?? ""))
                            case "assistant":
                                restored.append(UiChatMessage(
                                    role: .assistant,
                                    text: m.content ?? "",
                                    reasoning: m.reasoningContent
                                ))
                                if let toolCalls = m.toolCalls {
                                    for tc in toolCalls {
                                        restored.append(UiChatMessage(
                                            role: .toolCall,
                                            toolName: tc.function.name,
                                            toolArgs: tc.function.arguments,
                                            isRunning: false
                                        ))
                                    }
                                }
                            case "tool":
                                restored.append(UiChatMessage(
                                    role: .toolResult,
                                    toolResult: m.content ?? "",
                                    toolSuccess: true
                                ))
                            default:
                                break
                            }
                        }
                    } else {
                        restored.append(UiChatMessage(
                            role: .system,
                            text: "Switched to session \(targetId) (no messages)."
                        ))
                    }
                    self.messages = restored
                }
            } catch {
                DispatchQueue.main.async {
                    self.messages = [
                        UiChatMessage(role: .system, text: "Failed to resume session: \(error.localizedDescription)")
                    ]
                }
            }
        }
    }

    func deleteSession(sessionId targetId: String) {
        guard let engine = engine else { return }
        DispatchQueue.global(qos: .userInitiated).async {
            try? engine.deleteSession(sessionId: targetId)
            self.loadSessionsList()
            if targetId == self.sessionId {
                DispatchQueue.main.async {
                    self.startNewSession()
                }
            }
        }
    }

    func toggleThinking(for id: UUID) {
        if let idx = messages.firstIndex(where: { $0.id == id }) {
            messages[idx].isThinkingExpanded.toggle()
        }
    }

    func toggleOutput(for id: UUID) {
        if let idx = messages.firstIndex(where: { $0.id == id }) {
            messages[idx].isOutputExpanded.toggle()
        }
    }

    func cancelCurrentTurn() {
        engine?.cancel(sessionId: sessionId)
        currentProgressText = "Cancelled"
    }

    func answerClarification(callId: String, answer: String) {
        pendingClarification = nil
        currentProgressText = "Sending reply..."
        DispatchQueue.global(qos: .userInitiated).async {
            try? self.engine?.completeHostTool(callId: callId, ok: true, output: answer)
        }
    }

    func sendMessage() {
        guard canSend else { return }

        let userInput = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        inputText = ""
        isProcessing = true
        currentProgressText = "Thinking..."

        NSLog("[DemoApp] 🚀 User prompt: '%@' (Session: %@)", userInput, sessionId)
        messages.append(UiChatMessage(role: .user, text: userInput))

        Task {
            do {
                guard let engine = engine else {
                    NSLog("[DemoApp] ❌ Attempted to chat but engine is closed/nil")
                    throw PhoneBuddyError.engineClosed
                }

                var currentReasoning = ""
                var currentText = ""

                let outcome = try await engine.chat(
                    sessionId: sessionId,
                    userInput: userInput
                ) { [weak self] eventJson in
                    guard let self = self,
                          let data = eventJson.data(using: .utf8),
                          let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                          let (tag, payloadVal) = obj.first else {
                        return
                    }

                    let payload = payloadVal as? [String: Any] ?? [:]

                    DispatchQueue.main.async {
                        switch tag {
                        case "ReasoningDelta":
                            let text = payload["text"] as? String ?? ""
                            currentReasoning += text
                            self.currentProgressText = "💭 Thinking..."
                            self.updateStreamingAssistant(reasoning: currentReasoning, text: currentText)

                        case "TextDelta":
                            let text = payload["text"] as? String ?? ""
                            currentText += text
                            self.currentProgressText = "✍️ Generating response..."
                            self.updateStreamingAssistant(reasoning: currentReasoning, text: currentText)

                        case "ToolCallStart":
                            let name = payload["name"] as? String ?? "tool"
                            let args = payload["arguments_json"] as? String ?? ""
                            let callId = payload["call_id"] as? String ?? ""

                            NSLog("[DemoApp] ⚙️ ToolCallStart: %@ (callId: %@, args: %@)", name, callId, args)

                            let summary = parseToolSummary(name: name, argsJson: args)
                            if let param = summary.primaryParam {
                                self.currentProgressText = "\(summary.title): \(param)"
                            } else {
                                self.currentProgressText = "⚙️ Executing \(name)..."
                            }

                            self.messages.append(UiChatMessage(
                                role: .toolCall,
                                callId: callId,
                                toolName: name,
                                toolArgs: args,
                                isRunning: true
                            ))

                        case "ToolCallResult":
                            let name = payload["name"] as? String ?? "tool"
                            let ok = payload["ok"] as? Bool ?? true
                            let out = payload["output"] as? String ?? ""
                            let callId = payload["call_id"] as? String

                            NSLog("[DemoApp] 🏁 ToolCallResult: %@ (callId: %@, ok: %d, outputBytes: %ld)", name, callId ?? "", ok, out.count)

                            // Mark the corresponding tool call as no longer running
                            if let lastToolIdx = self.messages.lastIndex(where: { $0.role == .toolCall && ($0.callId == callId || $0.toolName == name) }) {
                                self.messages[lastToolIdx].isRunning = false
                            }

                            self.currentProgressText = ok ? "✓ Finished \(name)" : "✗ Failed \(name)"

                            self.messages.append(UiChatMessage(
                                role: .toolResult,
                                callId: callId,
                                toolName: name,
                                toolResult: out,
                                toolSuccess: ok
                            ))

                        case "PlanUpdated":
                            self.currentProgressText = "📋 Plan updated"
                            if let itemsJson = payload["items_json"] as? String,
                               let pdata = itemsJson.data(using: .utf8),
                               let pArray = try? JSONSerialization.jsonObject(with: pdata) as? [[String: Any]] {
                                let planItems = pArray.map { dict in
                                    PlanItem(
                                        id: dict["id"] as? String ?? "-",
                                        content: dict["content"] as? String ?? "",
                                        status: dict["status"] as? String ?? "pending"
                                    )
                                }
                                NSLog("[DemoApp] 📋 PlanUpdated: %ld items", planItems.count)
                                self.messages.append(UiChatMessage(role: .plan, planItems: planItems))
                            }

                        case "Completed":
                            self.currentProgressText = ""
                            if let usage = payload["usage"] as? [String: Any] {
                                let p = usage["prompt_tokens"] as? Int ?? 0
                                let c = usage["completion_tokens"] as? Int ?? 0
                                let t = usage["total_tokens"] as? Int ?? 0
                                NSLog("[DemoApp] ✓ Completed turn (tokens: prompt=%ld, completion=%ld, total=%ld)", p, c, t)
                                if let lastIdx = self.messages.lastIndex(where: { $0.role == .assistant }) {
                                    self.messages[lastIdx].tokenUsage = "Tokens: prompt=\(p), completion=\(c), total=\(t)"
                                }
                            } else {
                                NSLog("[DemoApp] ✓ Completed turn")
                            }

                        case "Failed":
                            let msg = payload["message"] as? String ?? "Turn failed"
                            self.currentProgressText = ""
                            NSLog("[DemoApp] ❌ Turn failed: %@", msg)
                            self.messages.append(UiChatMessage(role: .system, text: "Turn failed: \(msg)"))

                        default:
                            break
                        }
                    }
                }

                NSLog("[DemoApp] ✓ Chat execution finished successfully for session: %@", sessionId)
                await MainActor.run {
                    self.loadSessionsList()
                    self.isProcessing = false
                    self.currentProgressText = ""
                }
            } catch {
                NSLog("[DemoApp] ❌ Chat execution threw error: %@", error.localizedDescription)
                await MainActor.run {
                    self.messages.append(UiChatMessage(
                        role: .system,
                        text: "Error: \(error.localizedDescription)"
                    ))
                    self.isProcessing = false
                    self.currentProgressText = ""
                }
            }
        }
    }

    private func updateStreamingAssistant(reasoning: String, text: String) {
        if let lastIdx = messages.lastIndex(where: { $0.role == .assistant }), lastIdx == messages.count - 1 {
            messages[lastIdx].reasoning = reasoning.isEmpty ? nil : reasoning
            messages[lastIdx].text = text
        } else {
            messages.append(UiChatMessage(
                role: .assistant,
                text: text,
                reasoning: reasoning.isEmpty ? nil : reasoning
            ))
        }
    }
}

// MARK: - View Extension for Custom Rounded Corners

extension View {
    func cornerRadius(_ radius: CGFloat, corners: UIRectCorner) -> some View {
        clipShape(RoundedCorner(radius: radius, corners: corners))
    }
}

struct RoundedCorner: Shape {
    var radius: CGFloat = .infinity
    var corners: UIRectCorner = .allCorners

    func path(in rect: CGRect) -> Path {
        let path = UIBezierPath(
            roundedRect: rect,
            byRoundingCorners: corners,
            cornerRadii: CGSize(width: radius, height: radius)
        )
        return Path(path.cgPath)
    }
}
