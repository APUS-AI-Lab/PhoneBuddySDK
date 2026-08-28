package org.phonebuddy

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.webkit.WebView
import android.webkit.WebViewClient
import org.json.JSONArray
import org.json.JSONObject
import org.json.JSONTokener
import java.nio.charset.StandardCharsets

interface EventListener {
    fun onEvent(eventJson: String)
}

interface GenerateTextListener {
    fun onComplete(envelopeJson: String)
}

interface HostToolListener {
    fun onHostToolRequest(callId: String, name: String, argumentsJson: String)
}

data class SessionMetadata(
    val id: String,
    val title: String,
    val createdAt: String = "",
    val updatedAt: String = "",
    val messageCount: Int = 0
)

data class StoredToolCall(
    val id: String,
    val name: String,
    val arguments: String
)

data class StoredChatMessage(
    val role: String,
    val content: String?,
    val reasoningContent: String?,
    val toolCalls: List<StoredToolCall> = emptyList(),
    val toolCallId: String? = null
)

data class StoredSessionData(
    val id: String,
    val title: String,
    val createdAt: String,
    val updatedAt: String,
    val messages: List<StoredChatMessage>
)

class NativeAgent : AutoCloseable {
    private var enginePtr: Long = 0
    private var hostToolListener: HostToolListener? = null

    constructor(configJson: String, context: Context? = null) {
        enginePtr = nativeNewEngine(configJson)
        context?.let { enableSystemWebView(it) }
    }

    internal constructor(existingPtr: Long, context: Context? = null) {
        enginePtr = existingPtr
        context?.let { enableSystemWebView(it) }
    }

    fun chat(sessionId: String, userInput: String, listener: EventListener? = null): String? {
        check(enginePtr != 0L) { "Engine closed" }
        return nativeChat(enginePtr, sessionId, userInput, listener)
    }

    fun cancel(sessionId: String) {
        if (enginePtr != 0L) {
            nativeCancel(enginePtr, sessionId)
        }
    }

    fun getSession(sessionId: String): String? {
        check(enginePtr != 0L) { "Engine closed" }
        return nativeGetSession(enginePtr, sessionId)
    }

    fun getSessionData(sessionId: String): StoredSessionData? {
        val jsonStr = getSession(sessionId) ?: return null
        return try {
            val root = JSONObject(jsonStr)
            val id = root.optString("id", sessionId)
            val title = root.optString("title", "")
            val createdAt = root.optString("created_at", "")
            val updatedAt = root.optString("updated_at", "")
            val msgsArray = root.optJSONArray("items") ?: root.optJSONArray("messages") ?: JSONArray()
            val messages = mutableListOf<StoredChatMessage>()

            for (i in 0 until msgsArray.length()) {
                val m = msgsArray.getJSONObject(i)
                val itemType = m.optString("type", "")
                val role = when (itemType) {
                    "user" -> "user"
                    "assistant" -> "assistant"
                    "tool_result" -> "tool"
                    "system" -> "system"
                    "reasoning", "backend_tool_call" -> continue
                    else -> m.optString("role", "")
                }
                val content = if (m.has("content") && !m.isNull("content")) m.getString("content") else null
                val reasoning = if (m.has("reasoning_content") && !m.isNull("reasoning_content")) m.getString("reasoning_content") else null
                val toolCallId = if (m.has("tool_call_id") && !m.isNull("tool_call_id")) m.getString("tool_call_id") else null

                val toolCalls = mutableListOf<StoredToolCall>()
                val tcArray = m.optJSONArray("tool_calls")
                if (tcArray != null) {
                    for (j in 0 until tcArray.length()) {
                        val tc = tcArray.getJSONObject(j)
                        val tcId = tc.optString("id", "")
                        val fn = tc.optJSONObject("function")
                        val fnName = fn?.optString("name", "") ?: ""
                        val fnArgs = fn?.optString("arguments", "") ?: ""
                        toolCalls.add(StoredToolCall(tcId, fnName, fnArgs))
                    }
                }

                messages.add(
                    StoredChatMessage(
                        role = role,
                        content = content,
                        reasoningContent = reasoning,
                        toolCalls = toolCalls,
                        toolCallId = toolCallId
                    )
                )
            }

            StoredSessionData(
                id = id,
                title = title,
                createdAt = createdAt,
                updatedAt = updatedAt,
                messages = messages
            )
        } catch (_: Exception) {
            null
        }
    }

    fun listSessions(): String? {
        check(enginePtr != 0L) { "Engine closed" }
        return nativeListSessions(enginePtr)
    }

    fun listSessionItems(): List<SessionMetadata> {
        val jsonStr = listSessions() ?: return emptyList()
        return try {
            val array = JSONArray(jsonStr)
            val list = mutableListOf<SessionMetadata>()
            for (i in 0 until array.length()) {
                val item = array.getJSONObject(i)
                val id = item.optString("id", "")
                val title = item.optString("title", "")
                val createdAt = item.optString("created_at", "")
                val updatedAt = item.optString("updated_at", "")
                val messageCount = item.optInt("message_count", 0)
                if (id.isNotEmpty()) {
                    list.add(
                        SessionMetadata(
                            id = id,
                            title = title,
                            createdAt = createdAt,
                            updatedAt = updatedAt,
                            messageCount = messageCount
                        )
                    )
                }
            }
            list
        } catch (_: Exception) {
            emptyList()
        }
    }

    fun deleteSession(sessionId: String): Boolean {
        check(enginePtr != 0L) { "Engine closed" }
        return nativeDeleteSession(enginePtr, sessionId) == 0
    }

    /**
     * Register a hidden system WebView so `web_search` and `web_fetch` use Android's
     * system WebKit engine (supporting cookies, TLS, and JavaScript rendering).
     */
    fun enableSystemWebView(context: Context) {
        check(enginePtr != 0L) { "Engine closed" }
        SystemWebViewHost.install(this, context.applicationContext)
        nativeSetWebViewCallback(enginePtr)
    }

    /**
     * Register listener for host tools and ask_user_question clarification requests.
     */
    fun setHostToolListener(listener: HostToolListener?) {
        check(enginePtr != 0L) { "Engine closed" }
        this.hostToolListener = listener
        activeAgents[enginePtr] = this
        nativeSetHostCallbacks(enginePtr)
    }

    fun completeHostTool(callId: String, ok: Boolean, output: String): Boolean {
        check(enginePtr != 0L) { "Engine closed" }
        return nativeHostToolResult(enginePtr, callId, if (ok) 1 else 0, output) == 0
    }

    /**
     * Set the system-prompt identity (`You are {name}…`).
     * Pass null or blank to reset to `PhoneBuddy`.
     */
    fun setAgentName(name: String?) {
        check(enginePtr != 0L) { "Engine closed" }
        nativeSetAgentName(enginePtr, name)
    }

    /** Set or clear extra product instructions appended to the system prompt. */
    fun setSystemPromptExtra(extra: String?) {
        check(enginePtr != 0L) { "Engine closed" }
        nativeSetSystemPromptExtra(enginePtr, extra)
    }

    @Synchronized
    internal fun completeWebView(callId: String, ok: Boolean, output: String) {
        if (enginePtr != 0L) {
            nativeWebViewResult(enginePtr, callId, if (ok) 1 else 0, output)
        }
    }

    @Synchronized
    override fun close() {
        if (enginePtr != 0L) {
            activeAgents.remove(enginePtr)
            SystemWebViewHost.detach(this)
            nativeClearWebViewCallback(enginePtr)
            nativeFreeEngine(enginePtr)
            enginePtr = 0L
        }
    }

    companion object {
        init {
            System.loadLibrary("phone_buddy_ffi")
        }

        private val activeAgents = java.util.concurrent.ConcurrentHashMap<Long, NativeAgent>()

        @JvmStatic
        private external fun nativeNewEngine(configJson: String): Long

        @JvmStatic
        private external fun nativeFreeEngine(enginePtr: Long)

        @JvmStatic
        private external fun nativeChat(
            enginePtr: Long,
            sessionId: String,
            userInput: String,
            listener: EventListener?
        ): String?

        @JvmStatic
        private external fun nativeCancel(enginePtr: Long, sessionId: String)

        @JvmStatic
        private external fun nativeGetSession(enginePtr: Long, sessionId: String): String?

        @JvmStatic
        private external fun nativeListSessions(enginePtr: Long): String?

        @JvmStatic
        private external fun nativeDeleteSession(enginePtr: Long, sessionId: String): Int

        @JvmStatic
        private external fun nativeSetHostCallbacks(enginePtr: Long)

        @JvmStatic
        private external fun nativeHostToolResult(
            enginePtr: Long,
            callId: String,
            ok: Int,
            output: String
        ): Int

        @JvmStatic
        fun onWebViewFetch(callId: String, requestJson: String) {
            SystemWebViewHost.onFetch(callId, requestJson)
        }

        @JvmStatic
        fun onHostToolRequest(callId: String, name: String, argumentsJson: String) {
            for (agent in activeAgents.values) {
                agent.hostToolListener?.onHostToolRequest(callId, name, argumentsJson)
            }
        }

        @JvmStatic
        private external fun nativeSetWebViewCallback(enginePtr: Long)

        @JvmStatic
        private external fun nativeClearWebViewCallback(enginePtr: Long)

        @JvmStatic
        private external fun nativeWebViewResult(
            enginePtr: Long,
            callId: String,
            ok: Int,
            output: String
        ): Int

        @JvmStatic
        private external fun nativeSetAgentName(enginePtr: Long, name: String?)

        @JvmStatic
        private external fun nativeSetSystemPromptExtra(enginePtr: Long, extra: String?)
    }
}

class NativeRuntime(routingJson: String, rootDir: String) : AutoCloseable {
    private var runtimePtr: Long = 0

    init {
        runtimePtr = nativeNew(routingJson, rootDir)
    }

    fun updateRouting(routingJson: String) {
        check(runtimePtr != 0L) { "Runtime closed" }
        nativeUpdateRouting(runtimePtr, routingJson)
    }

    fun createEngine(configJson: String, mainPoolId: String = "main", context: Context? = null): NativeAgent {
        check(runtimePtr != 0L) { "Runtime closed" }
        val ptr = nativeCreateEngine(runtimePtr, configJson, mainPoolId)
        check(ptr != 0L) { "Engine creation failed" }
        return NativeAgent(ptr, context)
    }

    fun generateText(requestJson: String, listener: GenerateTextListener? = null): String {
        check(runtimePtr != 0L) { "Runtime closed" }
        return nativeGenerateTextAsync(runtimePtr, requestJson, listener)
            ?: throw IllegalStateException("generateText returned no operation id")
    }

    fun cancel(operationId: String) {
        if (runtimePtr != 0L) {
            nativeCancelOperation(runtimePtr, operationId)
        }
    }

    @Synchronized
    override fun close() {
        if (runtimePtr != 0L) {
            nativeFree(runtimePtr)
            runtimePtr = 0L
        }
    }

    companion object {
        init {
            System.loadLibrary("phone_buddy_ffi")
        }

        @JvmStatic
        private external fun nativeNew(routingJson: String, rootDir: String): Long

        @JvmStatic
        private external fun nativeFree(runtimePtr: Long)

        @JvmStatic
        private external fun nativeUpdateRouting(runtimePtr: Long, routingJson: String)

        @JvmStatic
        private external fun nativeCreateEngine(
            runtimePtr: Long,
            configJson: String,
            mainPoolId: String?
        ): Long

        @JvmStatic
        private external fun nativeGenerateTextAsync(
            runtimePtr: Long,
            requestJson: String,
            listener: GenerateTextListener?
        ): String?

        @JvmStatic
        private external fun nativeCancelOperation(runtimePtr: Long, operationId: String)
    }
}

private object SystemWebViewHost {
    private const val TAG = "PhoneBuddy-WebView"
    private val main = Handler(Looper.getMainLooper())
    @Volatile
    private var agent: NativeAgent? = null
    @Volatile
    private var appContext: Context? = null
    private var webView: WebView? = null
    private var webViewOwner: NativeAgent? = null
    private var pendingCallId: String? = null
    private var pendingOwner: NativeAgent? = null
    private var timeoutRunnable: Runnable? = null

    @JvmStatic
    fun install(owner: NativeAgent, context: Context) {
        agent = owner
        appContext = context.applicationContext
        Log.i(TAG, "Installed SystemWebViewHost for agent")
    }

    @JvmStatic
    fun detach(owner: NativeAgent) {
        val wasInstalledOwner = agent === owner
        if (wasInstalledOwner) {
            agent = null
            appContext = null
        }
        main.post {
            if (pendingOwner === owner) {
                timeoutRunnable?.let { main.removeCallbacks(it) }
                timeoutRunnable = null
                pendingCallId = null
                pendingOwner = null
            }
            releaseWebView(owner)
        }
        if (wasInstalledOwner) {
            Log.i(TAG, "Detached SystemWebViewHost for agent")
        }
    }

    @JvmStatic
    fun onFetch(callId: String, requestJson: String) {
        Log.d(TAG, "onFetch callback received from native engine: callId=$callId")
        main.post { start(callId, requestJson) }
    }

    private fun start(callId: String, requestJson: String) {
        failPending("superseded by a newer WebView fetch")
        val owner = agent
        val context = appContext
        if (owner == null || context == null) {
            Log.w(TAG, "Cannot start WebView fetch (owner=$owner, context=$context)")
            return
        }

        val obj = try {
            JSONObject(requestJson)
        } catch (e: Exception) {
            Log.e(TAG, "Invalid WebView request JSON: ${e.message}")
            owner.completeWebView(callId, false, "invalid WebView request JSON: ${e.message}")
            return
        }

        val url = obj.optString("url")
        if (url.isNullOrBlank()) {
            Log.e(TAG, "WebView request missing URL for callId=$callId")
            owner.completeWebView(callId, false, "WebView request missing url")
            return
        }
        val method = obj.optString("method", "GET").uppercase()
        val body = obj.optString("body", "")
        val timeoutMs = obj.optLong("timeout_ms", 20_000L)
        val headersObj = obj.optJSONObject("headers")

        val view = ensureWebView(context, owner)
        pendingCallId = callId
        pendingOwner = owner

        // Custom headers
        val headerMap = HashMap<String, String>()
        if (headersObj != null) {
            val keys = headersObj.keys()
            while (keys.hasNext()) {
                val k = keys.next()
                val v = headersObj.optString(k, "")
                if (k.equals("User-Agent", ignoreCase = true) && v.isNotBlank()) {
                    view.settings.userAgentString = v
                } else {
                    headerMap[k] = v
                }
            }
        }

        Log.i(
            TAG,
            "▶ [HeadlessWebView] Starting fetch: callId=$callId, method=$method, url=$url, timeout=${timeoutMs}ms, hasCustomHeaders=${headerMap.isNotEmpty()}"
        )

        val timeout = Runnable {
            Log.w(TAG, "✖ [HeadlessWebView] Navigation timed out after ${timeoutMs}ms: url=$url, callId=$callId")
            finish(false, "Android WebView navigation timed out")
        }
        timeoutRunnable = timeout
        main.postDelayed(timeout, timeoutMs)

        if (method == "POST") {
            view.postUrl(url, body.toByteArray(StandardCharsets.UTF_8))
        } else {
            if (headerMap.isNotEmpty()) {
                view.loadUrl(url, headerMap)
            } else {
                view.loadUrl(url)
            }
        }
    }

    private fun ensureWebView(context: Context, owner: NativeAgent): WebView {
        webView?.let {
            if (webViewOwner === owner) {
                return it
            }
            releaseWebView()
        }
        val view = WebView(context)
        view.settings.javaScriptEnabled = true
        view.settings.domStorageEnabled = true
        view.webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView, url: String) {
                if (view !== webView) return
                val callId = pendingCallId ?: return
                view.evaluateJavascript(
                    "(function(){return document.documentElement.outerHTML;})()"
                ) { value ->
                    if (view === webView && pendingCallId == callId) {
                        val html = decodeJsString(value)
                        Log.i(
                            TAG,
                            "✔ [HeadlessWebView] Succeeded loading HTML: callId=$callId, url=$url, html_bytes=${html.length}"
                        )
                        finish(true, html)
                    }
                }
            }

            @Deprecated("Deprecated in Java")
            override fun onReceivedError(
                view: WebView,
                errorCode: Int,
                description: String?,
                failingUrl: String?
            ) {
                if (view !== webView || pendingCallId == null) return
                Log.e(
                    TAG,
                    "✖ [HeadlessWebView] Navigation Error: callId=$pendingCallId, url=$failingUrl, errorCode=$errorCode, desc=$description"
                )
                finish(false, description ?: "WebView error $errorCode")
            }
        }
        webView = view
        webViewOwner = owner
        return view
    }

    private fun finish(ok: Boolean, output: String) {
        val callId = pendingCallId ?: return
        val owner = pendingOwner
        pendingCallId = null
        pendingOwner = null
        timeoutRunnable?.let { main.removeCallbacks(it) }
        timeoutRunnable = null
        owner?.let { releaseWebView(it) }
        Log.d(TAG, "Completed WebView request: callId=$callId, ok=$ok, output_len=${output.length}")
        owner?.completeWebView(callId, ok, output)
    }

    private fun failPending(message: String) {
        val callId = pendingCallId ?: return
        val owner = pendingOwner
        pendingCallId = null
        pendingOwner = null
        timeoutRunnable?.let { main.removeCallbacks(it) }
        timeoutRunnable = null
        owner?.let { releaseWebView(it) }
        Log.w(TAG, "Fail pending WebView request: callId=$callId, reason=$message")
        owner?.completeWebView(callId, false, message)
    }

    /** Release the current page so its DOM, JavaScript, and renderer resources cannot
     * remain active between headless fetches. Must be called on the main thread. */
    private fun releaseWebView(expectedOwner: NativeAgent? = null) {
        if (expectedOwner != null && webViewOwner !== expectedOwner) return
        val view = webView
        webView = null
        webViewOwner = null
        if (view != null) {
            view.stopLoading()
            view.webViewClient = WebViewClient()
            view.removeAllViews()
            view.destroy()
        }
    }

    private fun decodeJsString(value: String?): String {
        if (value.isNullOrBlank() || value == "null") {
            return ""
        }
        return try {
            JSONTokener(value).nextValue() as? String ?: value
        } catch (_: Exception) {
            value
        }
    }
}
