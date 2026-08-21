//
//  PhoneBuddy.swift
//  PhoneBuddy iOS Swift Wrapper
//

import Foundation
import UIKit
import WebKit

/// Configuration options for raw HTTP traffic dumps.
public struct HttpDumpConfig: Codable {
    public var mode: String // "off", "on_error", "all"
    public var dumpDir: String?
    public var maskSensitive: Bool
    public var maxFiles: Int

    public init(
        mode: String = "off",
        dumpDir: String? = nil,
        maskSensitive: Bool = true,
        maxFiles: Int = 30
    ) {
        self.mode = mode
        self.dumpDir = dumpDir
        self.maskSensitive = maskSensitive
        self.maxFiles = maxFiles
    }

    enum CodingKeys: String, CodingKey {
        case mode
        case dumpDir = "dump_dir"
        case maskSensitive = "mask_sensitive"
        case maxFiles = "max_files"
    }
}

/// Configuration options for PhoneBuddyEngine.
public struct PhoneBuddyConfig: Codable {
    public var apiKey: String
    public var baseUrl: String
    public var model: String
    public var apiBackend: String
    public var rootDir: String
    public var maxTurns: Int
    public var enableWebSearch: Bool
    /// Identity used in the system prompt (`You are {agentName}…`). Default: PhoneBuddy.
    public var agentName: String
    public var extraHeaders: [String: String]?
    public var extraBody: [String: String]?
    public var httpDump: HttpDumpConfig?

    public init(
        apiKey: String = "",
        baseUrl: String = "https://api.x.ai/v1",
        model: String = "grok-4.6",
        apiBackend: String = "responses",
        rootDir: String,
        maxTurns: Int = 24,
        enableWebSearch: Bool = true,
        agentName: String = "PhoneBuddy",
        extraHeaders: [String: String]? = nil,
        extraBody: [String: String]? = nil,
        httpDump: HttpDumpConfig? = nil
    ) {
        self.apiKey = apiKey
        self.baseUrl = baseUrl
        self.model = model
        self.apiBackend = apiBackend
        self.rootDir = rootDir
        self.maxTurns = maxTurns
        self.enableWebSearch = enableWebSearch
        self.agentName = agentName
        self.extraHeaders = extraHeaders
        self.extraBody = extraBody
        self.httpDump = httpDump
    }

    enum CodingKeys: String, CodingKey {
        case apiKey = "api_key"
        case baseUrl = "base_url"
        case model
        case apiBackend = "api_backend"
        case rootDir = "root_dir"
        case maxTurns = "max_turns"
        case enableWebSearch = "enable_web_search"
        case agentName = "agent_name"
        case extraHeaders = "extra_headers"
        case extraBody = "extra_body"
        case httpDump = "http_dump"
    }

    private enum AltCodingKeys: String, CodingKey {
        case apiKey
        case baseUrl
        case model
        case modelId = "model_id"
        case apiBackend
        case rootDir
        case maxTurns
        case enableWebSearch
        case extraHeaders
        case extraBody
        case agentName
        case httpDump
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let altContainer = try? decoder.container(keyedBy: AltCodingKeys.self)

        self.apiKey = (try? container.decodeIfPresent(String.self, forKey: .apiKey))
            ?? (try? altContainer?.decodeIfPresent(String.self, forKey: .apiKey))
            ?? ""
        self.baseUrl = (try? container.decodeIfPresent(String.self, forKey: .baseUrl))
            ?? (try? altContainer?.decodeIfPresent(String.self, forKey: .baseUrl))
            ?? "https://api.x.ai/v1"
        self.model = (try? container.decodeIfPresent(String.self, forKey: .model))
            ?? (try? altContainer?.decodeIfPresent(String.self, forKey: .modelId))
            ?? (try? altContainer?.decodeIfPresent(String.self, forKey: .model))
            ?? "grok-4.6"
        self.apiBackend = (try? container.decodeIfPresent(String.self, forKey: .apiBackend))
            ?? (try? altContainer?.decodeIfPresent(String.self, forKey: .apiBackend))
            ?? "responses"
        self.rootDir = (try? container.decodeIfPresent(String.self, forKey: .rootDir))
            ?? (try? altContainer?.decodeIfPresent(String.self, forKey: .rootDir))
            ?? ""
        self.maxTurns = (try? container.decodeIfPresent(Int.self, forKey: .maxTurns))
            ?? (try? altContainer?.decodeIfPresent(Int.self, forKey: .maxTurns))
            ?? 24
        self.enableWebSearch = (try? container.decodeIfPresent(Bool.self, forKey: .enableWebSearch))
            ?? (try? altContainer?.decodeIfPresent(Bool.self, forKey: .enableWebSearch))
            ?? true
        let decodedName = (try? container.decodeIfPresent(String.self, forKey: .agentName))
            ?? (try? altContainer?.decodeIfPresent(String.self, forKey: .agentName))
            ?? "PhoneBuddy"
        self.agentName = decodedName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? "PhoneBuddy"
            : decodedName
        self.extraHeaders = (try? container.decodeIfPresent([String: String].self, forKey: .extraHeaders))
            ?? (try? altContainer?.decodeIfPresent([String: String].self, forKey: .extraHeaders))
        self.extraBody = (try? container.decodeIfPresent([String: String].self, forKey: .extraBody))
            ?? (try? altContainer?.decodeIfPresent([String: String].self, forKey: .extraBody))
        self.httpDump = (try? container.decodeIfPresent(HttpDumpConfig.self, forKey: .httpDump))
            ?? (try? altContainer?.decodeIfPresent(HttpDumpConfig.self, forKey: .httpDump))
    }

    public static let userDefaultsKey = "phone_buddy_config"

    public static func loadOrDefault(rootDir: String = "workspace") -> PhoneBuddyConfig {
        // 1. Check for config.json in Documents or App Bundle first
        let sandbox = sandboxRoot(workspaceName: rootDir)
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        let candidates = [
            docs.first?.appendingPathComponent("config.json"),
            docs.first?.appendingPathComponent("PhoneBuddy/config.json"),
            URL(fileURLWithPath: sandbox).appendingPathComponent("config.json"),
            Bundle.main.url(forResource: "config", withExtension: "json")
        ].compactMap { $0 }

        for fileUrl in candidates {
            if FileManager.default.fileExists(atPath: fileUrl.path),
               let text = try? String(contentsOf: fileUrl, encoding: .utf8),
               let cfg = try? fromJsonString(text),
               !cfg.apiKey.isEmpty || !cfg.model.isEmpty {
                return cfg
            }
        }

        // 2. Fall back to UserDefaults
        if let data = UserDefaults.standard.data(forKey: userDefaultsKey),
           var config = try? JSONDecoder().decode(PhoneBuddyConfig.self, from: data) {
            // Re-pin so a leftover desktop/tmp path becomes Documents/<name>.
            config.pinSandboxRoot(config.rootDir.isEmpty ? rootDir : config.rootDir)
            return config
        }

        // 3. Fall back to default settings (Documents/workspace)
        return PhoneBuddyConfig(
            apiKey: "",
            baseUrl: "https://api.x.ai/v1",
            model: "grok-4.6",
            apiBackend: "responses",
            rootDir: Self.sandboxRoot(workspaceName: rootDir),
            maxTurns: 24,
            enableWebSearch: true,
            extraBody: nil
        )
    }

    public static let defaultWorkspaceName = "workspace"

    /// Folder name imported from config.json `root_dir` (e.g. `./workspace` → `workspace`).
    public var workspaceName: String {
        Self.workspaceName(from: rootDir)
    }

    public static func documentsDirectory() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
    }

    /// Take only the last path component from a desktop `root_dir`.
    public static func workspaceName(from rawRootDir: String) -> String {
        let trimmed = rawRootDir.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return defaultWorkspaceName }

        let last = trimmed
            .split { $0 == "/" || $0 == "\\" }
            .map(String.init)
            .last ?? ""
        if last.isEmpty || last == "." || last == ".." || last == "tmp" || last == "phone-buddy-demo" {
            return defaultWorkspaceName
        }
        return last
    }

    /// Writable sandbox: `<Documents>/<workspaceName>`.
    public static func sandboxRoot(workspaceName rawName: String) -> String {
        let name = workspaceName(from: rawName)
        let url = documentsDirectory().appendingPathComponent(name, isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url.standardizedFileURL.path
    }

    /// Map a config `root_dir` onto `Documents/<name>` so the engine never writes
    /// into the read-only app bundle (os error 30).
    public mutating func pinSandboxRoot(_ rootDir: String) {
        let pinned = Self.resolvedWritableRoot(rootDir)
        if !pinned.isEmpty {
            self.rootDir = pinned
        }
    }

    /// Resolve `root_dir` to `Documents/<imported-name>`.
    public static func resolvedWritableRoot(_ requested: String) -> String {
        let fm = FileManager.default
        let trimmed = requested.trimmingCharacters(in: .whitespacesAndNewlines)
        let docs = documentsDirectory().standardizedFileURL

        if !trimmed.isEmpty && (trimmed as NSString).isAbsolutePath {
            let url = URL(fileURLWithPath: trimmed).standardizedFileURL
            let docsPath = docs.path
            let urlPath = url.path
            let insideDocs = urlPath == docsPath || urlPath.hasPrefix(docsPath + "/")
            if insideDocs {
                if !fm.fileExists(atPath: urlPath) {
                    try? fm.createDirectory(at: url, withIntermediateDirectories: true)
                }
                if fm.isWritableFile(atPath: urlPath) {
                    return urlPath
                }
            }
            // tmp / desktop / other absolute paths: keep only the folder name.
            return sandboxRoot(workspaceName: url.lastPathComponent)
        }

        return sandboxRoot(workspaceName: trimmed)
    }

    public func save() {
        if let data = try? JSONEncoder().encode(self) {
            UserDefaults.standard.set(data, forKey: Self.userDefaultsKey)
            if let docsUrl = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
                let configUrl = docsUrl.appendingPathComponent("config.json")
                try? data.write(to: configUrl)
            }
        }
    }

    /// Strips JavaScript/C-style single-line (//) and multi-line (/* ... */) comments from JSON strings
    public static func stripJsonComments(_ input: String) -> String {
        var result = ""
        var inString = false
        var isEscaped = false
        var i = input.startIndex

        while i < input.endIndex {
            let c = input[i]
            let nextIndex = input.index(after: i)
            let nextC = nextIndex < input.endIndex ? input[nextIndex] : "\0"

            if inString {
                result.append(c)
                if isEscaped {
                    isEscaped = false
                } else if c == "\\" {
                    isEscaped = true
                } else if c == "\"" {
                    inString = false
                }
                i = nextIndex
            } else {
                if c == "\"" {
                    inString = true
                    result.append(c)
                    i = nextIndex
                } else if c == "/" && nextC == "/" {
                    // Single-line comment: skip until newline
                    i = input.index(after: nextIndex)
                    while i < input.endIndex && input[i] != "\n" && input[i] != "\r" {
                        i = input.index(after: i)
                    }
                } else if c == "/" && nextC == "*" {
                    // Multi-line comment: skip until */
                    i = input.index(after: nextIndex)
                    while i < input.endIndex {
                        let cur = input[i]
                        let nxt = input.index(after: i)
                        if cur == "*" && nxt < input.endIndex && input[nxt] == "/" {
                            i = input.index(after: nxt)
                            break
                        }
                        i = input.index(after: i)
                    }
                } else {
                    result.append(c)
                    i = nextIndex
                }
            }
        }
        return result
    }

    /// Parse a PhoneBuddyConfig from a JSON string, automatically supporting comments like in config.json.example.
    public static func fromJsonString(_ jsonStr: String, rootDir: String? = nil) throws -> PhoneBuddyConfig {
        let cleanJson = stripJsonComments(jsonStr)
        guard let data = cleanJson.data(using: .utf8) else {
            throw PhoneBuddyError.invalidConfig
        }
        var cfg = try JSONDecoder().decode(PhoneBuddyConfig.self, from: data)
        // `root_dir` from config.json is only a folder name (e.g. "./workspace").
        // The real sandbox is always Documents/<name>.
        if !cfg.rootDir.isEmpty {
            cfg.pinSandboxRoot(cfg.rootDir)
        } else if let root = rootDir, !root.isEmpty {
            cfg.pinSandboxRoot(root)
        } else {
            cfg.pinSandboxRoot(defaultWorkspaceName)
        }
        return cfg
    }
}

/// Session metadata representation.
public struct SessionMetadata: Identifiable, Codable {
    public let id: String
    public let title: String
    public let createdAt: String
    public let updatedAt: String
    public let messageCount: Int

    enum CodingKeys: String, CodingKey {
        case id
        case title
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case messageCount = "message_count"
    }
}

public struct StoredToolFunction: Codable {
    public let name: String
    public let arguments: String
}

public struct StoredToolCall: Codable {
    public let id: String
    public let function: StoredToolFunction
}

public struct StoredChatMessage: Codable {
    public let role: String
    public let content: String?
    public let reasoningContent: String?
    public let toolCalls: [StoredToolCall]?
    public let toolCallId: String?

    enum CodingKeys: String, CodingKey {
        case role
        case content
        case reasoningContent = "reasoning_content"
        case toolCalls = "tool_calls"
        case toolCallId = "tool_call_id"
    }
}

public struct StoredSession: Codable {
    public let id: String
    public let title: String
    public let createdAt: String
    public let updatedAt: String
    public let messages: [StoredChatMessage]

    enum CodingKeys: String, CodingKey {
        case id
        case title
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case messages
    }
}

/// Chat turn execution result.
public struct ChatOutcome: Codable {
    public let finalText: String
    public let turnsUsed: Int

    enum CodingKeys: String, CodingKey {
        case finalText = "final_text"
        case turnsUsed = "turns_used"
    }
}

/// Swift wrapper around the native C PhoneBuddy engine.
public final class PhoneBuddyEngine {
    private var enginePtr: OpaquePointer?
    private let webViewHost = SystemWebViewHost()
    private var hostToolBox: HostToolBox?

    public init(config: PhoneBuddyConfig) throws {
        var config = config
        config.pinSandboxRoot(config.rootDir)

        let encoder = JSONEncoder()
        let data = try encoder.encode(config)
        guard let jsonString = String(data: data, encoding: .utf8) else {
            throw PhoneBuddyError.invalidConfig
        }

        NSLog("[PhoneBuddy] Initializing engine with model: %@, backend: %@, baseURL: %@, rootDir: %@", config.model, config.apiBackend, config.baseUrl, config.rootDir)
        var errOut: UnsafeMutablePointer<CChar>? = nil
        let handle = pb_engine_new(jsonString, &errOut)
        if let err = errOut {
            let msg = String(cString: err)
            pb_string_free(err)
            NSLog("[PhoneBuddy] ❌ Engine creation failed: %@", msg)
            throw PhoneBuddyError.engineCreationFailed(msg)
        }
        guard let handle = handle else {
            NSLog("[PhoneBuddy] ❌ Engine creation returned null handle")
            throw PhoneBuddyError.engineCreationFailed("Unknown null pointer returned")
        }
        self.enginePtr = handle
        self.webViewHost.attach(engine: handle)
        NSLog("[PhoneBuddy] ✓ Engine initialized successfully")
    }

    /// Set the system-prompt identity (`You are {name}…`). Pass `nil` or empty to reset to `PhoneBuddy`.
    public func setAgentName(_ name: String?) {
        guard let ptr = enginePtr else { return }
        if let name, !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            pb_engine_set_agent_name(ptr, name)
        } else {
            pb_engine_set_agent_name(ptr, nil)
        }
    }

    /// Set or clear extra product instructions appended to the system prompt.
    public func setSystemPromptExtra(_ extra: String?) {
        guard let ptr = enginePtr else { return }
        if let extra, !extra.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            pb_engine_set_system_prompt_extra(ptr, extra)
        } else {
            pb_engine_set_system_prompt_extra(ptr, nil)
        }
    }

    deinit {
        webViewHost.shutdown()
        if let ptr = enginePtr {
            pb_engine_set_webview_callback(ptr, nil, nil)
            pb_engine_set_host_callbacks(ptr, nil, nil, nil)
            pb_engine_free(ptr)
        }
    }

    /// Run a chat turn asynchronously on a background thread.
    public func chat(
        sessionId: String,
        userInput: String,
        onEvent: ((String) -> Void)? = nil
    ) async throws -> ChatOutcome {
        guard let ptr = enginePtr else {
            throw PhoneBuddyError.engineClosed
        }

        NSLog("[PhoneBuddy] 🚀 Starting chat (sessionId: %@, input: '%@')", sessionId, userInput)

        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                var errOut: UnsafeMutablePointer<CChar>? = nil

                // Create callback context if onEvent is provided
                let context = EventContext(onEvent: onEvent)
                let unmanaged = Unmanaged.passRetained(context)
                let rawContext = unmanaged.toOpaque()

                let cCallback: PbEventCallback = { (eventJson, userData) in
                    guard let eventJson = eventJson, let userData = userData else { return }
                    let jsonStr = String(cString: eventJson)
                    let ctx = Unmanaged<EventContext>.fromOpaque(userData).takeUnretainedValue()
                    ctx.onEvent?(jsonStr)
                }

                let resPtr = pb_engine_chat(
                    ptr,
                    sessionId,
                    userInput,
                    cCallback,
                    rawContext,
                    &errOut
                )

                unmanaged.release()

                if let err = errOut {
                    let msg = String(cString: err)
                    pb_string_free(err)
                    NSLog("[PhoneBuddy] ❌ Chat turn failed (sessionId: %@): %@", sessionId, msg)
                    continuation.resume(throwing: PhoneBuddyError.chatFailed(msg))
                    return
                }

                guard let resPtr = resPtr else {
                    NSLog("[PhoneBuddy] ❌ Chat turn returned null result pointer (sessionId: %@)", sessionId)
                    continuation.resume(throwing: PhoneBuddyError.chatFailed("Null result"))
                    return
                }

                let jsonStr = String(cString: resPtr)
                pb_string_free(resPtr)

                if let data = jsonStr.data(using: .utf8),
                   let outcome = try? JSONDecoder().decode(ChatOutcome.self, from: data) {
                    NSLog("[PhoneBuddy] ✓ Chat turn completed (sessionId: %@, turnsUsed: %d)", sessionId, outcome.turnsUsed)
                    continuation.resume(returning: outcome)
                } else {
                    NSLog("[PhoneBuddy] ❌ Failed to parse ChatOutcome JSON (sessionId: %@): %@", sessionId, jsonStr)
                    continuation.resume(throwing: PhoneBuddyError.chatFailed("Failed to parse JSON result"))
                }
            }
        }
    }

    public func cancel(sessionId: String) {
        if let ptr = enginePtr {
            pb_engine_cancel(ptr, sessionId)
        }
    }

    /// Retrieve full session details (including all messages and reasoning) as a JSON string.
    public func getSession(sessionId: String) throws -> String? {
        guard let ptr = enginePtr else { throw PhoneBuddyError.engineClosed }
        var errOut: UnsafeMutablePointer<CChar>? = nil
        let resPtr = pb_engine_get_session(ptr, sessionId, &errOut)
        if let err = errOut {
            let msg = String(cString: err)
            pb_string_free(err)
            throw PhoneBuddyError.chatFailed(msg)
        }
        guard let resPtr = resPtr else { return nil }
        let jsonStr = String(cString: resPtr)
        pb_string_free(resPtr)
        return jsonStr
    }

    /// Retrieve full session details parsed as a typed StoredSession model.
    public func getSessionData(sessionId: String) throws -> StoredSession? {
        guard let jsonStr = try getSession(sessionId: sessionId) else { return nil }
        guard let data = jsonStr.data(using: .utf8) else { return nil }
        return try JSONDecoder().decode(StoredSession.self, from: data)
    }

    /// List all persisted sessions as a JSON array string.
    public func listSessions() throws -> String {
        guard let ptr = enginePtr else { throw PhoneBuddyError.engineClosed }
        var errOut: UnsafeMutablePointer<CChar>? = nil
        let resPtr = pb_engine_list_sessions(ptr, &errOut)
        if let err = errOut {
            let msg = String(cString: err)
            pb_string_free(err)
            throw PhoneBuddyError.chatFailed(msg)
        }
        guard let resPtr = resPtr else { return "[]" }
        let jsonStr = String(cString: resPtr)
        pb_string_free(resPtr)
        return jsonStr
    }

    /// List all persisted sessions parsed as an array of SessionMetadata objects.
    public func listSessionItems() throws -> [SessionMetadata] {
        let jsonStr = try listSessions()
        guard let data = jsonStr.data(using: .utf8) else { return [] }
        return (try? JSONDecoder().decode([SessionMetadata].self, from: data)) ?? []
    }

    /// Delete a persisted session by ID.
    public func deleteSession(sessionId: String) throws {
        guard let ptr = enginePtr else { throw PhoneBuddyError.engineClosed }
        let res = pb_engine_delete_session(ptr, sessionId)
        if res != 0 {
            throw PhoneBuddyError.chatFailed("Failed to delete session \(sessionId)")
        }
    }

    /// Register callback for host tools and ask_user_question UI prompts.
    public func setHostToolCallback(_ callback: @escaping (_ callId: String, _ name: String, _ argumentsJson: String) -> Void) {
        guard let ptr = enginePtr else { return }
        let box = HostToolBox(callback: callback)
        let rawContext = Unmanaged.passUnretained(box).toOpaque()

        let cCallback: PbHostToolCallback = { (callId, name, argsJson, userData) in
            guard let callId = callId, let name = name, let argsJson = argsJson, let userData = userData else { return }
            let callIdStr = String(cString: callId)
            let nameStr = String(cString: name)
            let argsStr = String(cString: argsJson)
            let box = Unmanaged<HostToolBox>.fromOpaque(userData).takeUnretainedValue()
            box.callback(callIdStr, nameStr, argsStr)
        }

        pb_engine_set_host_callbacks(ptr, nil, cCallback, rawContext)
        hostToolBox = box
    }

    /// Respond to a host tool or ask_user_question call.
    public func completeHostTool(callId: String, ok: Bool, output: String) throws {
        guard let ptr = enginePtr else { throw PhoneBuddyError.engineClosed }
        var errOut: UnsafeMutablePointer<CChar>? = nil
        let res = pb_engine_host_tool_result(ptr, callId, ok ? 1 : 0, output, &errOut)
        if let err = errOut {
            let msg = String(cString: err)
            pb_string_free(err)
            throw PhoneBuddyError.chatFailed(msg)
        }
        if res != 0 {
            throw PhoneBuddyError.chatFailed("Failed to complete host tool call \(callId)")
        }
    }
}

private class HostToolBox {
    let callback: (_ callId: String, _ name: String, _ argumentsJson: String) -> Void
    init(callback: @escaping (_ callId: String, _ name: String, _ argumentsJson: String) -> Void) {
        self.callback = callback
    }
}

private class EventContext {
    let onEvent: ((String) -> Void)?
    init(onEvent: ((String) -> Void)?) {
        self.onEvent = onEvent
    }
}

public enum PhoneBuddyError: Error, LocalizedError, CustomStringConvertible {
    case invalidConfig
    case engineCreationFailed(String)
    case engineClosed
    case chatFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfig:
            return "Invalid configuration: Failed to serialize config JSON"
        case .engineCreationFailed(let msg):
            return "Failed to initialize engine: \(msg)"
        case .engineClosed:
            return "Engine has been closed or uninitialized"
        case .chatFailed(let msg):
            return msg
        }
    }

    public var description: String {
        return errorDescription ?? "Unknown PhoneBuddy error"
    }
}

/// Hidden WKWebView used by `web_search` so DuckDuckGo sees system WebKit TLS.
private final class SystemWebViewHost: NSObject, WKNavigationDelegate {
    private var engine: OpaquePointer?
    private var webView: WKWebView?
    private var pendingCallId: String?
    private var pendingTimeout: DispatchWorkItem?
    private var timeoutMs: UInt64 = 20_000

    func shutdown() {
        pendingTimeout?.cancel()
        pendingTimeout = nil
        pendingCallId = nil
        engine = nil
        if Thread.isMainThread {
            teardownWebView()
        } else {
            DispatchQueue.main.sync { teardownWebView() }
        }
    }

    private func teardownWebView() {
        webView?.stopLoading()
        webView?.navigationDelegate = nil
        webView?.removeFromSuperview()
        webView = nil
    }

    func attach(engine: OpaquePointer) {
        self.engine = engine
        let unmanaged = Unmanaged.passUnretained(self).toOpaque()
        let callback: PbWebViewFetchCallback = { callId, requestJson, userData in
            guard let callId = callId, let requestJson = requestJson, let userData = userData else {
                return
            }
            let host = Unmanaged<SystemWebViewHost>.fromOpaque(userData).takeUnretainedValue()
            host.handle(callId: String(cString: callId), requestJson: String(cString: requestJson))
        }
        pb_engine_set_webview_callback(engine, callback, unmanaged)
    }

    private func handle(callId: String, requestJson: String) {
        DispatchQueue.main.async { [weak self] in
            self?.start(callId: callId, requestJson: requestJson)
        }
    }

    private func start(callId: String, requestJson: String) {
        failPending("superseded by a newer WebView fetch")

        guard let data = requestJson.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let urlString = obj["url"] as? String,
              let url = URL(string: urlString)
        else {
            complete(callId: callId, ok: false, output: "invalid WebView request JSON")
            return
        }

        let method = (obj["method"] as? String)?.uppercased() ?? "GET"
        let body = obj["body"] as? String ?? ""
        if let ms = obj["timeout_ms"] as? NSNumber {
            timeoutMs = ms.uint64Value
        }

        pendingCallId = callId
        let webView = ensureWebView()

        var request = URLRequest(url: url)
        request.httpMethod = method
        if let headers = obj["headers"] as? [String: String] {
            for (key, value) in headers {
                request.setValue(value, forHTTPHeaderField: key)
            }
        }
        if method != "GET" && !body.isEmpty {
            request.httpBody = body.data(using: .utf8)
            if request.value(forHTTPHeaderField: "Content-Type") == nil {
                request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
            }
        }

        let timeout = DispatchWorkItem { [weak self] in
            self?.finish(ok: false, output: "WKWebView navigation timed out")
        }
        pendingTimeout = timeout
        DispatchQueue.main.asyncAfter(deadline: .now() + Double(timeoutMs) / 1000.0, execute: timeout)
        webView.load(request)
    }

    private func ensureWebView() -> WKWebView {
        if let webView {
            return webView
        }
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .default()
        config.defaultWebpagePreferences.allowsContentJavaScript = true
        let webView = WKWebView(frame: CGRect(x: 0, y: 0, width: 390, height: 844), configuration: config)
        webView.navigationDelegate = self
        webView.isHidden = true
        attachToKeyWindow(webView)
        self.webView = webView
        return webView
    }

    private func attachToKeyWindow(_ webView: WKWebView) {
        let window = UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }
            .flatMap { $0.windows }
            .first { $0.isKeyWindow }
        guard let window else { return }
        webView.frame = CGRect(x: -4, y: -4, width: 2, height: 2)
        webView.alpha = 0.01
        window.addSubview(webView)
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        guard webView === self.webView, let callId = pendingCallId else { return }
        webView.evaluateJavaScript("document.documentElement.outerHTML") { [weak self, weak webView] result, error in
            guard let self,
                  let webView,
                  webView === self.webView,
                  self.pendingCallId == callId
            else {
                return
            }
            if let error {
                self.finish(ok: false, output: error.localizedDescription)
                return
            }
            let html = result as? String ?? ""
            self.finish(ok: true, output: html)
        }
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        guard webView === self.webView, pendingCallId != nil else { return }
        finish(ok: false, output: error.localizedDescription)
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        guard webView === self.webView, pendingCallId != nil else { return }
        finish(ok: false, output: error.localizedDescription)
    }

    private func finish(ok: Bool, output: String) {
        guard let callId = pendingCallId else { return }
        pendingCallId = nil
        pendingTimeout?.cancel()
        pendingTimeout = nil
        teardownWebView()
        complete(callId: callId, ok: ok, output: output)
    }

    private func failPending(_ message: String) {
        guard let callId = pendingCallId else { return }
        pendingCallId = nil
        pendingTimeout?.cancel()
        pendingTimeout = nil
        teardownWebView()
        complete(callId: callId, ok: false, output: message)
    }

    private func complete(callId: String, ok: Bool, output: String) {
        guard let engine else { return }
        var errOut: UnsafeMutablePointer<CChar>? = nil
        _ = pb_engine_webview_result(engine, callId, ok ? 1 : 0, output, &errOut)
        if let err = errOut {
            pb_string_free(err)
        }
    }
}
