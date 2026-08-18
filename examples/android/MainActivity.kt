package org.phonebuddy.demo

import android.content.Context
import android.content.SharedPreferences
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import org.phonebuddy.EventListener
import org.phonebuddy.HostToolListener
import org.phonebuddy.NativeAgent
import org.phonebuddy.SessionMetadata
import java.io.File
import java.util.*

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            PhoneBuddyDemoTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    ChatScreen()
                }
            }
        }
    }
}

@Composable
fun PhoneBuddyDemoTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = lightColorScheme(
            primary = Color(0xFF0F62FE),
            onPrimary = Color.White,
            primaryContainer = Color(0xFFEDF5FF),
            onPrimaryContainer = Color(0xFF001F5C),
            surface = Color(0xFFFFFFFF),
            surfaceVariant = Color(0xFFF2F4F8),
            onSurfaceVariant = Color(0xFF393939)
        ),
        content = content
    )
}

// MARK: - Data Models

enum class MessageRole {
    USER, ASSISTANT, TOOL_CALL, TOOL_RESULT, PLAN, SYSTEM
}

data class PlanItem(
    val id: String,
    val content: String,
    val status: String
)

data class UiMessage(
    val id: String = UUID.randomUUID().toString(),
    val role: MessageRole,
    val text: String = "",
    val reasoning: String? = null,
    val callId: String? = null,
    val toolName: String? = null,
    val toolArgs: String? = null,
    val toolResult: String? = null,
    val toolSuccess: Boolean = true,
    var isRunning: Boolean = false,
    val planItems: List<PlanItem> = emptyList(),
    val tokenUsage: String? = null,
    var isThinkingExpanded: Boolean = true,
    var isOutputExpanded: Boolean = false
)

data class ClarificationRequest(
    val callId: String,
    val question: String,
    val options: List<String> = emptyList()
)

data class AppConfig(
    val apiKey: String = "",
    val baseUrl: String = "https://api.x.ai/v1",
    val model: String = "grok-4.6",
    val apiBackend: String = "responses",
    val maxTurns: Int = 24,
    val enableWebSearch: Boolean = true,
    val workspaceName: String = "workspace",
    val agentName: String = "PhoneBuddy"
) {
    fun resolveRootDir(context: Context): String {
        val name = sanitizeWorkspaceName(workspaceName)
        val dir = File(context.filesDir, name)
        dir.mkdirs()
        return dir.absolutePath
    }

    fun toJson(rootDir: String, context: Context? = null): String {
        return JSONObject().apply {
            put("api_key", apiKey)
            put("base_url", baseUrl)
            put("model", model)
            put("api_backend", apiBackend)
            put("root_dir", rootDir)
            put("max_turns", maxTurns)
            put("enable_web_search", enableWebSearch)
            put("agent_name", agentName.ifBlank { "PhoneBuddy" })
        }.toString()
    }

    companion object {
        private const val PREFS_NAME = "phone_buddy_settings"
        private const val KEY_API_KEY = "api_key"
        private const val KEY_BASE_URL = "base_url"
        private const val KEY_MODEL = "model"
        private const val KEY_API_BACKEND = "api_backend"
        private const val KEY_MAX_TURNS = "max_turns"
        private const val KEY_ENABLE_WEB_SEARCH = "enable_web_search"
        private const val KEY_WORKSPACE_NAME = "workspace_name"
        const val DEFAULT_WORKSPACE_NAME = "workspace"

        // Last path component of desktop `root_dir` (`./workspace` → `workspace`).
        fun sanitizeWorkspaceName(raw: String?): String {
            val last = raw.orEmpty().trim()
                .trimEnd('/', '\\')
                .substringAfterLast('/')
                .substringAfterLast('\\')
                .trim()
            if (last.isEmpty() || last == "." || last == ".." || last == "tmp" || last == "phone-buddy-demo") {
                return DEFAULT_WORKSPACE_NAME
            }
            if (last.any { it == '/' || it == '\\' || it == '\u0000' }) {
                return DEFAULT_WORKSPACE_NAME
            }
            return last
        }

        fun load(context: Context): AppConfig {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            return AppConfig(
                apiKey = prefs.getString(KEY_API_KEY, "") ?: "",
                baseUrl = prefs.getString(KEY_BASE_URL, "https://api.x.ai/v1") ?: "https://api.x.ai/v1",
                model = prefs.getString(KEY_MODEL, "grok-4.6") ?: "grok-4.6",
                apiBackend = prefs.getString(KEY_API_BACKEND, "responses") ?: "responses",
                maxTurns = prefs.getInt(KEY_MAX_TURNS, 24),
                enableWebSearch = prefs.getBoolean(KEY_ENABLE_WEB_SEARCH, true),
                workspaceName = sanitizeWorkspaceName(
                    prefs.getString(KEY_WORKSPACE_NAME, DEFAULT_WORKSPACE_NAME)
                )
            )
        }

        fun save(context: Context, config: AppConfig) {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            prefs.edit().apply {
                putString(KEY_API_KEY, config.apiKey)
                putString(KEY_BASE_URL, config.baseUrl)
                putString(KEY_MODEL, config.model)
                putString(KEY_API_BACKEND, config.apiBackend)
                putInt(KEY_MAX_TURNS, config.maxTurns)
                putBoolean(KEY_ENABLE_WEB_SEARCH, config.enableWebSearch)
                putString(KEY_WORKSPACE_NAME, sanitizeWorkspaceName(config.workspaceName))
                apply()
            }
        }

        fun stripJsonComments(input: String): String {
            val sb = StringBuilder()
            var inString = false
            var isEscaped = false
            var i = 0
            while (i < input.length) {
                val c = input[i]
                val nextC = if (i + 1 < input.length) input[i + 1] else '\u0000'
                if (inString) {
                    sb.append(c)
                    if (isEscaped) {
                        isEscaped = false
                    } else if (c == '\\') {
                        isEscaped = true
                    } else if (c == '"') {
                        inString = false
                    }
                    i++
                } else {
                    if (c == '"') {
                        inString = true
                        sb.append(c)
                        i++
                    } else if (c == '/' && nextC == '/') {
                        i += 2
                        while (i < input.length && input[i] != '\n' && input[i] != '\r') {
                            i++
                        }
                    } else if (c == '/' && nextC == '*') {
                        i += 2
                        while (i < input.length) {
                            val cur = input[i]
                            val nxt = if (i + 1 < input.length) input[i + 1] else '\u0000'
                            if (cur == '*' && nxt == '/') {
                                i += 2
                                break
                            }
                            i++
                        }
                    } else {
                        sb.append(c)
                        i++
                    }
                }
            }
            return sb.toString()
        }

        fun fromJsonString(jsonStr: String): AppConfig? {
            return try {
                val clean = stripJsonComments(jsonStr)
                val obj = JSONObject(clean)
                AppConfig(
                    apiKey = obj.optString("api_key", ""),
                    baseUrl = obj.optString("base_url", "https://api.x.ai/v1"),
                    model = obj.optString("model", "grok-4.6"),
                    apiBackend = obj.optString("api_backend", "responses"),
                    maxTurns = obj.optInt("max_turns", 24),
                    enableWebSearch = obj.optBoolean("enable_web_search", true),
                    workspaceName = sanitizeWorkspaceName(obj.optString("root_dir", DEFAULT_WORKSPACE_NAME))
                )
            } catch (_: Exception) {
                null
            }
        }

        fun findDownloadConfigFiles(context: Context): List<File> {
            val candidates = mutableListOf<File>()
            // 1. App-specific external files dir (100% Zero-permission on all Android versions):
            // /sdcard/Android/data/org.phonebuddy.demo/files/config.json
            context.getExternalFilesDir(null)?.let { dir ->
                candidates.add(File(dir, "config.json"))
                candidates.add(File(dir, "phone_buddy_config.json"))
            }
            // 2. App internal files:
            candidates.add(File(context.filesDir, "config.json"))
            // 3. Public Download directory:
            candidates.add(File("/sdcard/Download/config.json"))
            candidates.add(File("/sdcard/Download/phone_buddy_config.json"))
            try {
                val pubDown = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                candidates.add(File(pubDown, "config.json"))
                candidates.add(File(pubDown, "phone_buddy_config.json"))
            } catch (_: Exception) {}
            return candidates.distinct().filter { it.exists() && it.isFile && it.canRead() }
        }

        fun loadFromFile(file: File): AppConfig? {
            return try {
                val text = file.readText(Charsets.UTF_8)
                fromJsonString(text)
            } catch (_: Exception) {
                null
            }
        }

        fun loadFromUri(context: Context, uri: Uri): AppConfig? {
            return try {
                context.contentResolver.openInputStream(uri)?.use { stream ->
                    val text = stream.bufferedReader(Charsets.UTF_8).use { it.readText() }
                    fromJsonString(text)
                }
            } catch (_: Exception) {
                null
            }
        }
    }
}

// MARK: - Tool Summary Formatter

data class ToolSummary(
    val iconEmoji: String,
    val title: String,
    val primaryParam: String?,
    val detail: String?
)

fun formatToolSummary(name: String, argsJson: String): ToolSummary {
    return try {
        val obj = JSONObject(argsJson)
        when (name) {
            "web_search" -> {
                val query = obj.optString("query", obj.optString("search_query", ""))
                ToolSummary("🔍", "Web Search", if (query.isNotBlank()) "\"$query\"" else null, null)
            }
            "web_fetch" -> {
                val url = obj.optString("url", "")
                ToolSummary("🌐", "Web Fetch", if (url.isNotBlank()) url else null, null)
            }
            "read_file" -> {
                val path = obj.optString("path", obj.optString("file_path", ""))
                ToolSummary("📁", "Read File", if (path.isNotBlank()) path else null, null)
            }
            "write_file" -> {
                val path = obj.optString("path", obj.optString("file_path", ""))
                ToolSummary("✏️", "Write File", if (path.isNotBlank()) path else null, null)
            }
            "edit_file" -> {
                val path = obj.optString("path", obj.optString("file_path", ""))
                ToolSummary("📝", "Edit File", if (path.isNotBlank()) path else null, null)
            }
            "grep_search" -> {
                val query = obj.optString("query", "")
                val path = obj.optString("path", "")
                val sum = if (path.isNotBlank()) "\"$query\" in $path" else "\"$query\""
                ToolSummary("🔎", "Grep Search", sum, null)
            }
            "list_dir" -> {
                val path = obj.optString("path", obj.optString("directory", "."))
                ToolSummary("📂", "List Directory", path, null)
            }
            "plan" -> {
                ToolSummary("📋", "Execution Plan", "Updating task plan...", null)
            }
            "task" -> {
                val prompt = obj.optString("prompt", obj.optString("role", ""))
                ToolSummary("🚀", "Subagent Task", if (prompt.isNotBlank()) "\"$prompt\"" else null, null)
            }
            "ask_user_question" -> {
                val q = obj.optString("question", "")
                ToolSummary("❓", "Clarification Question", if (q.isNotBlank()) q else null, null)
            }
            else -> {
                ToolSummary("⚙️", "Tool: $name", null, if (argsJson.isNotBlank()) argsJson else null)
            }
        }
    } catch (_: Exception) {
        ToolSummary("⚙️", "Tool: $name", null, if (argsJson.isNotBlank()) argsJson else null)
    }
}

// MARK: - Main Chat Screen

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen() {
    val scope = rememberCoroutineScope()
    val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)
    val context = androidx.compose.ui.platform.LocalContext.current

    var appConfig by remember { mutableStateOf(AppConfig.load(context)) }
    var agent: NativeAgent? by remember { mutableStateOf(null) }
    var sessionId by remember { mutableStateOf(UUID.randomUUID().toString()) }
    var sessionsList by remember { mutableStateOf(listOf<SessionMetadata>()) }
    var messages by remember { mutableStateOf(listOf<UiMessage>()) }
    var inputText by remember { mutableStateOf("") }
    var isProcessing by remember { mutableStateOf(false) }
    var currentProgressText by remember { mutableStateOf("") }
    var pendingClarification by remember { mutableStateOf<ClarificationRequest?>(null) }
    var clarificationInput by remember { mutableStateOf("") }
    var showSettingsDialog by remember { mutableStateOf(false) }

    val listState = rememberLazyListState()

    val workDir = remember(appConfig.workspaceName) {
        appConfig.resolveRootDir(context)
    }

    // Function to re-instantiate NativeAgent with new configuration
    val reconfigureAgent: (AppConfig) -> Unit = { newConfig ->
        appConfig = newConfig
        AppConfig.save(context, newConfig)
        scope.launch(Dispatchers.IO) {
            var candidateAgent: NativeAgent? = null
            try {
                agent?.close()
                val configJson = newConfig.toJson(newConfig.resolveRootDir(context), context)
                val newAgent = NativeAgent(configJson, context.applicationContext)
                candidateAgent = newAgent

                newAgent.setHostToolListener(object : HostToolListener {
                    override fun onHostToolRequest(callId: String, name: String, argumentsJson: String) {
                        if (name == "ask_user_question") {
                            try {
                                val args = JSONObject(argumentsJson)
                                val question = args.optString("question", "Clarification requested:")
                                val optionsArray = args.optJSONArray("options")
                                val optionsList = mutableListOf<String>()
                                if (optionsArray != null) {
                                    for (i in 0 until optionsArray.length()) {
                                        optionsList.add(optionsArray.getString(i))
                                    }
                                }
                                scope.launch(Dispatchers.Main) {
                                    pendingClarification = ClarificationRequest(callId, question, optionsList)
                                    currentProgressText = "❓ Clarification requested"
                                }
                            } catch (_: Exception) {
                                newAgent.completeHostTool(callId, false, "Failed to parse question arguments")
                            }
                        } else {
                            newAgent.completeHostTool(callId, false, "Unsupported host tool: $name")
                        }
                    }
                })

                val initialSessions = newAgent.listSessionItems()

                withContext(Dispatchers.Main) {
                    agent = newAgent
                    sessionsList = initialSessions
                    messages = messages + UiMessage(
                        role = MessageRole.SYSTEM,
                        text = "⚙️ Configuration applied:\n• Model: ${newConfig.model}\n• Backend: ${newConfig.apiBackend}\n• Base URL: ${newConfig.baseUrl}\n• Workspace: ${newConfig.workspaceName}\n• Max Turns: ${newConfig.maxTurns}\n• Web Search: ${newConfig.enableWebSearch}"
                    )
                }
                candidateAgent = null
            } catch (e: Exception) {
                runCatching { candidateAgent?.close() }
                withContext(Dispatchers.Main) {
                    messages = messages + UiMessage(
                        role = MessageRole.SYSTEM,
                        text = "Engine reinitialization failed: ${e.message}\nPlease check your settings (⚙️)."
                    )
                }
            }
        }
    }

    // Reload sessions list from disk
    val reloadSessions: () -> Unit = {
        agent?.let { ag ->
            scope.launch(Dispatchers.IO) {
                try {
                    val list = ag.listSessionItems()
                    withContext(Dispatchers.Main) {
                        sessionsList = list
                    }
                } catch (_: Exception) {}
            }
        }
    }

    // Switch or resume a session by loading past history
    val switchSession: (String) -> Unit = { targetSessionId ->
        sessionId = targetSessionId
        pendingClarification = null
        currentProgressText = ""
        scope.launch(Dispatchers.IO) {
            try {
                val sessionData = agent?.getSessionData(targetSessionId)
                withContext(Dispatchers.Main) {
                    val newMessages = mutableListOf<UiMessage>()
                if (sessionData != null && sessionData.messages.isNotEmpty()) {
                    for (m in sessionData.messages) {
                        when (m.role.lowercase()) {
                            "user" -> {
                                newMessages.add(
                                    UiMessage(
                                        role = MessageRole.USER,
                                        text = m.content ?: ""
                                    )
                                )
                            }
                            "assistant" -> {
                                newMessages.add(
                                    UiMessage(
                                        role = MessageRole.ASSISTANT,
                                        reasoning = if (!m.reasoningContent.isNullOrBlank()) m.reasoningContent else null,
                                        text = m.content ?: ""
                                    )
                                )
                                for (tc in m.toolCalls) {
                                    newMessages.add(
                                        UiMessage(
                                            role = MessageRole.TOOL_CALL,
                                            toolName = tc.name,
                                            toolArgs = tc.arguments,
                                            isRunning = false
                                        )
                                    )
                                }
                            }
                            "tool" -> {
                                newMessages.add(
                                    UiMessage(
                                        role = MessageRole.TOOL_RESULT,
                                        toolResult = m.content ?: "",
                                        toolSuccess = true
                                    )
                                )
                            }
                        }
                    }
                } else {
                    newMessages.add(
                        UiMessage(
                            role = MessageRole.SYSTEM,
                            text = "Switched to session $targetSessionId (empty history)."
                        )
                    )
                }
                messages = newMessages
                drawerState.close()
            }
        } catch (_: Exception) {}
    }
}

    // Start fresh session
    val startNewSession: () -> Unit = {
        val newId = UUID.randomUUID().toString()
        sessionId = newId
        pendingClarification = null
        currentProgressText = ""
        messages = listOf(
            UiMessage(
                role = MessageRole.SYSTEM,
                text = "Started fresh session: $newId\nModel: ${appConfig.model} (${appConfig.apiBackend})"
            )
        )
        reloadSessions()
        scope.launch { drawerState.close() }
    }

    // Initial Engine Startup
    LaunchedEffect(Unit) {
        withContext(Dispatchers.IO) {
            var candidateAgent: NativeAgent? = null
            try {
                val configJson = appConfig.toJson(appConfig.resolveRootDir(context), context)
                val newAgent = NativeAgent(configJson, context.applicationContext)
                candidateAgent = newAgent

                newAgent.setHostToolListener(object : HostToolListener {
                    override fun onHostToolRequest(callId: String, name: String, argumentsJson: String) {
                        if (name == "ask_user_question") {
                            try {
                                val args = JSONObject(argumentsJson)
                                val question = args.optString("question", "Clarification requested:")
                                val optionsArray = args.optJSONArray("options")
                                val optionsList = mutableListOf<String>()
                                if (optionsArray != null) {
                                    for (i in 0 until optionsArray.length()) {
                                        optionsList.add(optionsArray.getString(i))
                                    }
                                }
                                scope.launch(Dispatchers.Main) {
                                    pendingClarification = ClarificationRequest(callId, question, optionsList)
                                    currentProgressText = "❓ Clarification requested"
                                }
                            } catch (_: Exception) {
                                newAgent.completeHostTool(callId, false, "Failed to parse question arguments")
                            }
                        } else {
                            newAgent.completeHostTool(callId, false, "Unsupported host tool: $name")
                        }
                    }
                })

                val initialSessions = newAgent.listSessionItems()

                withContext(Dispatchers.Main) {
                    agent = newAgent
                    sessionsList = initialSessions
                    messages = listOf(
                        UiMessage(
                            role = MessageRole.SYSTEM,
                            text = "PhoneBuddy Agent engine ready (v1.0.0).\n• Model: ${appConfig.model} (${appConfig.apiBackend})\n• Workspace: ${appConfig.workspaceName}\n• Headless WebView configured for `web_search` & `web_fetch`.\n• Tap ⚙️ in top bar to configure API Key & Base URL."
                        )
                    )
                }
                candidateAgent = null
            } catch (e: Exception) {
                runCatching { candidateAgent?.close() }
                withContext(Dispatchers.Main) {
                    messages = listOf(
                        UiMessage(
                            role = MessageRole.SYSTEM,
                            text = "Engine initialization failed: ${e.message}\nPlease check settings (⚙️) to configure API Key."
                        )
                    )
                }
            }
        }
    }

    // Auto-scroll to bottom on message list change or streaming text update
    LaunchedEffect(messages.size, messages.lastOrNull()?.text, messages.lastOrNull()?.reasoning) {
        if (messages.isNotEmpty()) {
            listState.scrollToItem(messages.size - 1, scrollOffset = 100000)
        }
    }

    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            ModalDrawerSheet(modifier = Modifier.width(320.dp)) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(16.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "Chat History",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                        IconButton(onClick = startNewSession) {
                            Icon(Icons.Default.Add, contentDescription = "New Session")
                        }
                    }

                    HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

                    if (sessionsList.isEmpty()) {
                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .fillMaxWidth(),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = "No saved sessions found.",
                                color = Color.Gray,
                                fontSize = 14.sp
                            )
                        }
                    } else {
                        LazyColumn(
                            modifier = Modifier.weight(1f),
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            items(sessionsList, key = { it.id }) { s ->
                                val isSelected = s.id == sessionId
                                Card(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .clickable { switchSession(s.id) },
                                    colors = CardDefaults.cardColors(
                                        containerColor = if (isSelected) {
                                            MaterialTheme.colorScheme.primaryContainer
                                        } else {
                                            MaterialTheme.colorScheme.surfaceVariant
                                        }
                                    ),
                                    shape = RoundedCornerShape(12.dp)
                                ) {
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(12.dp),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Column(modifier = Modifier.weight(1f)) {
                                            Text(
                                                text = if (s.title.isNotBlank()) s.title else "Session ${s.id.take(8)}",
                                                fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Medium,
                                                fontSize = 14.sp,
                                                maxLines = 1,
                                                overflow = TextOverflow.Ellipsis
                                            )
                                            Spacer(modifier = Modifier.height(4.dp))
                                            Text(
                                                text = "${s.messageCount} msgs • ${s.updatedAt.take(16)}",
                                                fontSize = 11.sp,
                                                color = Color.Gray
                                            )
                                        }

                                        IconButton(
                                            onClick = {
                                                scope.launch(Dispatchers.IO) {
                                                    try {
                                                        agent?.deleteSession(s.id)
                                                        reloadSessions()
                                                        if (s.id == sessionId) {
                                                            withContext(Dispatchers.Main) {
                                                                startNewSession()
                                                            }
                                                        }
                                                    } catch (_: Exception) {}
                                                }
                                            }
                                        ) {
                                            Icon(
                                                Icons.Default.Delete,
                                                contentDescription = "Delete Session",
                                                tint = Color.Gray,
                                                modifier = Modifier.size(18.dp)
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }

                    HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

                    Button(
                        onClick = startNewSession,
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Icon(Icons.Default.Add, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("New Chat Session")
                    }
                }
            }
        }
    ) {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = {
                        Column {
                            Text(
                                text = "PhoneBuddy Agent",
                                fontWeight = FontWeight.Bold,
                                fontSize = 17.sp
                            )
                            Text(
                                text = "${appConfig.model} • ${appConfig.apiBackend}",
                                fontSize = 11.sp,
                                color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f)
                            )
                        }
                    },
                    navigationIcon = {
                        IconButton(onClick = { scope.launch { drawerState.open() } }) {
                            Icon(Icons.Default.Menu, contentDescription = "History")
                        }
                    },
                    actions = {
                        IconButton(onClick = { showSettingsDialog = true }) {
                            Icon(Icons.Default.Settings, contentDescription = "Settings")
                        }

                        if (isProcessing) {
                            IconButton(
                                onClick = {
                                    agent?.cancel(sessionId)
                                    currentProgressText = "Cancelled"
                                }
                            ) {
                                Icon(
                                    Icons.Default.Close,
                                    contentDescription = "Cancel Turn",
                                    tint = Color.Red
                                )
                            }
                        } else {
                            IconButton(onClick = startNewSession) {
                                Icon(Icons.Default.Add, contentDescription = "New Chat")
                            }
                        }
                    },
                    colors = TopAppBarDefaults.topAppBarColors(
                        containerColor = MaterialTheme.colorScheme.primaryContainer
                    )
                )
            }
        ) { padding ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
            ) {
                // Real-time progress banner (like c_demo)
                if (isProcessing) {
                    Surface(
                        color = MaterialTheme.colorScheme.surfaceVariant,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 14.dp, vertical = 8.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier.weight(1f)
                            ) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(16.dp),
                                    strokeWidth = 2.dp,
                                    color = MaterialTheme.colorScheme.primary
                                )
                                Spacer(modifier = Modifier.width(10.dp))
                                Text(
                                    text = if (currentProgressText.isNotBlank()) currentProgressText else "Agent is working...",
                                    fontSize = 12.sp,
                                    fontWeight = FontWeight.Medium,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis
                                )
                            }

                            Spacer(modifier = Modifier.width(8.dp))

                            TextButton(
                                onClick = {
                                    agent?.cancel(sessionId)
                                    currentProgressText = "Cancelled"
                                },
                                colors = ButtonDefaults.textButtonColors(contentColor = Color.Red),
                                contentPadding = PaddingValues(horizontal = 8.dp, vertical = 2.dp)
                            ) {
                                Text("Stop", fontSize = 12.sp, fontWeight = FontWeight.Bold)
                            }
                        }
                    }
                    HorizontalDivider()
                }

                // Messages list
                LazyColumn(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth(),
                    state = listState,
                    contentPadding = PaddingValues(12.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    items(messages, key = { it.id }) { msg ->
                        MessageItemView(
                            msg = msg,
                            onToggleThinking = {
                                val idx = messages.indexOfFirst { it.id == msg.id }
                                if (idx >= 0) {
                                    val updated = messages.toMutableList()
                                    updated[idx] = updated[idx].copy(isThinkingExpanded = !updated[idx].isThinkingExpanded)
                                    messages = updated
                                }
                            },
                            onToggleOutput = {
                                val idx = messages.indexOfFirst { it.id == msg.id }
                                if (idx >= 0) {
                                    val updated = messages.toMutableList()
                                    updated[idx] = updated[idx].copy(isOutputExpanded = !updated[idx].isOutputExpanded)
                                    messages = updated
                                }
                            }
                        )
                    }
                }

                // Interactive clarification question card (ask_user_question)
                pendingClarification?.let { req ->
                    ClarificationCard(
                        request = req,
                        inputText = clarificationInput,
                        onInputChange = { clarificationInput = it },
                        onSubmit = { responseText ->
                            val callId = req.callId
                            pendingClarification = null
                            clarificationInput = ""
                            currentProgressText = "Sending reply..."
                            scope.launch(Dispatchers.IO) {
                                try {
                                    agent?.completeHostTool(callId, true, responseText)
                                } catch (_: Exception) {}
                            }
                        }
                    )
                }

                // Quick Prompt Suggestion Chips (Search, Fetch, Plan, Clarification)
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .horizontalScroll(rememberScrollState())
                        .padding(horizontal = 12.dp, vertical = 4.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    SuggestionChip(
                        onClick = {
                            inputText = "Search the web for the latest updates on Rust language in 2026"
                        },
                        label = { Text("🔍 Search Rust 2026", fontSize = 12.sp) }
                    )
                    SuggestionChip(
                        onClick = {
                            inputText = "Fetch https://news.ycombinator.com and summarize the top 3 headlines"
                        },
                        label = { Text("🌐 WebFetch HackerNews", fontSize = 12.sp) }
                    )
                    SuggestionChip(
                        onClick = {
                            inputText = "Create a multi-step research plan to analyze quantum computing breakthroughs"
                        },
                        label = { Text("📋 Plan Research", fontSize = 12.sp) }
                    )
                    SuggestionChip(
                        onClick = {
                            inputText = "Ask me clarifying questions about my favorite programming tech stack"
                        },
                        label = { Text("❓ Test Clarification", fontSize = 12.sp) }
                    )
                }

                HorizontalDivider()

                // Input bar
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    OutlinedTextField(
                        value = inputText,
                        onValueChange = { inputText = it },
                        modifier = Modifier.weight(1f),
                        placeholder = { Text("Ask a question, search, or run tasks...") },
                        enabled = !isProcessing,
                        maxLines = 4,
                        shape = RoundedCornerShape(20.dp)
                    )

                    IconButton(
                        onClick = {
                            val userText = inputText.trim()
                            if (userText.isEmpty() || isProcessing || agent == null) return@IconButton

                            inputText = ""
                            isProcessing = true
                            currentProgressText = "Thinking..."

                            // Append user message
                            messages = messages + UiMessage(role = MessageRole.USER, text = userText)

                            scope.launch(Dispatchers.IO) {
                                var currentReasoning = ""
                                var currentText = ""

                                try {
                                    val resultJson = agent?.chat(
                                        sessionId,
                                        userText,
                                        object : EventListener {
                                            override fun onEvent(eventJson: String) {
                                                try {
                                                    val root = JSONObject(eventJson)
                                                    val keys = root.keys()
                                                    if (!keys.hasNext()) return
                                                    val tag = keys.next()
                                                    val payload = root.optJSONObject(tag)

                                                    when (tag) {
                                                        "ReasoningDelta" -> {
                                                            val delta = payload?.optString("text", "") ?: ""
                                                            currentReasoning += delta
                                                            scope.launch(Dispatchers.Main) {
                                                                currentProgressText = "💭 Thinking..."
                                                                updateStreamingMessage(currentReasoning, currentText)
                                                            }
                                                        }
                                                        "TextDelta" -> {
                                                            val delta = payload?.optString("text", "") ?: ""
                                                            currentText += delta
                                                            scope.launch(Dispatchers.Main) {
                                                                currentProgressText = "✍️ Generating response..."
                                                                updateStreamingMessage(currentReasoning, currentText)
                                                            }
                                                        }
                                                        "ToolCallStart" -> {
                                                            val name = payload?.optString("name", "tool") ?: "tool"
                                                            val args = payload?.optString("arguments_json", "") ?: ""
                                                            val callId = payload?.optString("call_id", "") ?: ""
                                                            val summary = formatToolSummary(name, args)

                                                            Log.i("PhoneBuddy-Tool", "▶ [ToolCallStart] tool=$name, callId=$callId, args=$args")

                                                            scope.launch(Dispatchers.Main) {
                                                                currentProgressText = if (summary.primaryParam != null) {
                                                                    "${summary.title}: ${summary.primaryParam}"
                                                                } else {
                                                                    "⚙️ Executing $name..."
                                                                }
                                                                messages = messages + UiMessage(
                                                                    role = MessageRole.TOOL_CALL,
                                                                    callId = callId,
                                                                    toolName = name,
                                                                    toolArgs = args,
                                                                    isRunning = true
                                                                )
                                                            }
                                                        }
                                                        "ToolCallResult" -> {
                                                            val name = payload?.optString("name", "tool") ?: "tool"
                                                            val ok = payload?.optBoolean("ok", true) ?: true
                                                            val out = payload?.optString("output", "") ?: ""
                                                            val callId = payload?.optString("call_id", "")

                                                            Log.i("PhoneBuddy-Tool", "✔ [ToolCallResult] tool=$name, callId=$callId, ok=$ok, output_len=${out.length}")

                                                            when (name) {
                                                                "web_search" -> {
                                                                    if (out.contains("(via DuckDuckGo WebView)")) {
                                                                        Log.i(
                                                                            "PhoneBuddy-Search",
                                                                            "🔍 [web_search] Execution Mode: [Headless WebView (DuckDuckGo Lite)] | Success: $ok | Output Length: ${out.length}\n$out"
                                                                        )
                                                                    } else if (out.contains("via OpenAI Responses API") ||
                                                                        out.contains("via Claude Messages API") ||
                                                                        out.contains("via LLM ChatCompletions")
                                                                    ) {
                                                                        Log.i(
                                                                            "PhoneBuddy-Search",
                                                                            "🔍 [web_search] Execution Mode: [LLM Search API Fallback] | Success: $ok | Output Length: ${out.length}\n$out"
                                                                        )
                                                                    } else {
                                                                        Log.w(
                                                                            "PhoneBuddy-Search",
                                                                            "🔍 [web_search] Execution Mode: [Unknown/Error] | Success: $ok | Output:\n$out"
                                                                        )
                                                                    }
                                                                }
                                                                "web_fetch" -> {
                                                                    if (out.contains("[WebFetch Anti-Bot Block]") || out.contains("[WebFetch Error]")) {
                                                                        Log.w(
                                                                            "PhoneBuddy-Fetch",
                                                                            "🌐 [web_fetch] Failed/Blocked: ok=$ok | Output:\n${out.take(500)}"
                                                                        )
                                                                    } else {
                                                                        Log.i(
                                                                            "PhoneBuddy-Fetch",
                                                                            "🌐 [web_fetch] Succeeded: ok=$ok | Output Length: ${out.length} | Preview:\n${out.take(500)}"
                                                                        )
                                                                    }
                                                                }
                                                            }

                                                            scope.launch(Dispatchers.Main) {
                                                                // Mark tool call not running
                                                                val lastToolIdx = messages.indexOfLast {
                                                                    it.role == MessageRole.TOOL_CALL && (it.callId == callId || it.toolName == name)
                                                                }
                                                                if (lastToolIdx >= 0) {
                                                                    val updated = messages.toMutableList()
                                                                    updated[lastToolIdx] = updated[lastToolIdx].copy(isRunning = false)
                                                                    messages = updated
                                                                }

                                                                currentProgressText = if (ok) "✓ Finished $name" else "✗ Failed $name"
                                                                messages = messages + UiMessage(
                                                                    role = MessageRole.TOOL_RESULT,
                                                                    callId = callId,
                                                                    toolName = name,
                                                                    toolResult = out,
                                                                    toolSuccess = ok
                                                                )
                                                            }
                                                        }
                                                        "PlanUpdated" -> {
                                                            val itemsJson = payload?.optString("items_json", "") ?: ""
                                                            val planList = parsePlanItems(itemsJson)
                                                            Log.i("PhoneBuddy-Agent", "📋 [PlanUpdated] items_count=${planList.size}")
                                                            scope.launch(Dispatchers.Main) {
                                                                currentProgressText = "📋 Plan updated"
                                                                messages = messages + UiMessage(
                                                                    role = MessageRole.PLAN,
                                                                    planItems = planList
                                                                )
                                                            }
                                                        }
                                                        "Completed" -> {
                                                            val usageObj = payload?.optJSONObject("usage")
                                                            val pTok = usageObj?.optInt("prompt_tokens", 0) ?: 0
                                                            val cTok = usageObj?.optInt("completion_tokens", 0) ?: 0
                                                            val tTok = usageObj?.optInt("total_tokens", 0) ?: 0
                                                            val usageStr = "Tokens: prompt=$pTok, completion=$cTok, total=$tTok"
                                                            Log.i("PhoneBuddy-Agent", "🏁 [Completed] $usageStr")
                                                            scope.launch(Dispatchers.Main) {
                                                                currentProgressText = ""
                                                                appendUsage(usageStr)
                                                            }
                                                        }
                                                        "Failed" -> {
                                                            val errMsg = payload?.optString("message", "Error") ?: "Error"
                                                            Log.e("PhoneBuddy-Agent", "✖ [Failed] $errMsg")
                                                            scope.launch(Dispatchers.Main) {
                                                                currentProgressText = ""
                                                                messages = messages + UiMessage(
                                                                    role = MessageRole.SYSTEM,
                                                                    text = "Turn stopped: $errMsg"
                                                                )
                                                            }
                                                        }
                                                    }
                                                } catch (_: Exception) {
                                                }
                                            }

                                            private fun updateStreamingMessage(reasoning: String, text: String) {
                                                val lastIdx = messages.indexOfLast { it.role == MessageRole.ASSISTANT }
                                                if (lastIdx >= 0 && lastIdx == messages.lastIndex) {
                                                    val updated = messages.toMutableList()
                                                    updated[lastIdx] = updated[lastIdx].copy(
                                                        reasoning = if (reasoning.isNotEmpty()) reasoning else null,
                                                        text = text
                                                    )
                                                    messages = updated
                                                } else {
                                                    messages = messages + UiMessage(
                                                        role = MessageRole.ASSISTANT,
                                                        reasoning = if (reasoning.isNotEmpty()) reasoning else null,
                                                        text = text
                                                    )
                                                }
                                            }

                                            private fun appendUsage(usage: String) {
                                                val lastIdx = messages.indexOfLast { it.role == MessageRole.ASSISTANT }
                                                if (lastIdx >= 0) {
                                                    val updated = messages.toMutableList()
                                                    updated[lastIdx] = updated[lastIdx].copy(tokenUsage = usage)
                                                    messages = updated
                                                }
                                            }
                                        }
                                    )
                                } catch (e: Exception) {
                                    val errMsg = e.message ?: "Request failed"
                                    scope.launch(Dispatchers.Main) {
                                        messages = messages + UiMessage(
                                            role = MessageRole.SYSTEM,
                                            text = "❌ Error: $errMsg"
                                        )
                                    }
                                } finally {
                                    reloadSessions()
                                    withContext(Dispatchers.Main) {
                                        isProcessing = false
                                        currentProgressText = ""
                                    }
                                }
                            }
                        },
                        enabled = inputText.trim().isNotEmpty() && !isProcessing && agent != null,
                        modifier = Modifier
                            .size(48.dp)
                            .background(
                                color = if (inputText.trim().isNotEmpty() && !isProcessing) {
                                    MaterialTheme.colorScheme.primary
                                } else {
                                    Color.LightGray
                                },
                                shape = CircleShape
                            )
                    ) {
                        Icon(
                            Icons.Default.Send,
                            contentDescription = "Send",
                            tint = Color.White,
                            modifier = Modifier.size(20.dp)
                        )
                    }
                }
            }
        }
    }

    // Settings Dialog
    if (showSettingsDialog) {
        SettingsDialog(
            config = appConfig,
            rootDir = workDir,
            onDismiss = { showSettingsDialog = false },
            onSave = { updated ->
                showSettingsDialog = false
                reconfigureAgent(updated)
            }
        )
    }

    DisposableEffect(Unit) {
        onDispose {
            agent?.close()
        }
    }
}

private fun parsePlanItems(jsonStr: String): List<PlanItem> {
    return try {
        val array = JSONArray(jsonStr)
        val list = mutableListOf<PlanItem>()
        for (i in 0 until array.length()) {
            val obj = array.getJSONObject(i)
            list.add(
                PlanItem(
                    id = obj.optString("id", "-"),
                    content = obj.optString("content", ""),
                    status = obj.optString("status", "pending")
                )
            )
        }
        list
    } catch (_: Exception) {
        emptyList()
    }
}

// MARK: - Message Item Views

@Composable
fun MessageItemView(
    msg: UiMessage,
    onToggleThinking: () -> Unit,
    onToggleOutput: () -> Unit
) {
    when (msg.role) {
        MessageRole.USER -> {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End
            ) {
                Surface(
                    shape = RoundedCornerShape(16.dp, 16.dp, 2.dp, 16.dp),
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.widthIn(max = 300.dp)
                ) {
                    Text(
                        text = msg.text,
                        modifier = Modifier.padding(12.dp),
                        color = Color.White,
                        fontSize = 15.sp
                    )
                }
            }
        }
        MessageRole.ASSISTANT -> {
            Column(
                modifier = Modifier.fillMaxWidth(),
                horizontalAlignment = Alignment.Start
            ) {
                // Thinking block
                if (!msg.reasoning.isNullOrBlank()) {
                    ElevatedCard(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(bottom = 6.dp),
                        shape = RoundedCornerShape(12.dp),
                        colors = CardDefaults.elevatedCardColors(containerColor = Color(0xFFF8F9FA))
                    ) {
                        Column(modifier = Modifier.padding(10.dp)) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clickable { onToggleThinking() },
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Text(
                                        text = "💭 Thinking...",
                                        fontWeight = FontWeight.SemiBold,
                                        fontSize = 12.sp,
                                        color = Color(0xFF555555)
                                    )
                                }
                                Icon(
                                    if (msg.isThinkingExpanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                                    contentDescription = null,
                                    tint = Color.Gray,
                                    modifier = Modifier.size(16.dp)
                                )
                            }
                            if (msg.isThinkingExpanded) {
                                Spacer(modifier = Modifier.height(6.dp))
                                Text(
                                    text = msg.reasoning,
                                    fontFamily = FontFamily.Monospace,
                                    fontSize = 12.sp,
                                    color = Color(0xFF444444),
                                    lineHeight = 16.sp
                                )
                            }
                        }
                    }
                }

                // Assistant text
                if (msg.text.isNotBlank()) {
                    Surface(
                        shape = RoundedCornerShape(16.dp, 16.dp, 16.dp, 2.dp),
                        color = MaterialTheme.colorScheme.surfaceVariant,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(modifier = Modifier.padding(14.dp)) {
                            Text(
                                text = msg.text,
                                color = Color(0xFF1E1E1E),
                                fontSize = 15.sp,
                                lineHeight = 21.sp
                            )

                            msg.tokenUsage?.let { usage ->
                                Spacer(modifier = Modifier.height(8.dp))
                                Text(
                                    text = usage,
                                    fontSize = 11.sp,
                                    color = Color.Gray
                                )
                            }
                        }
                    }
                }
            }
        }
        MessageRole.TOOL_CALL -> {
            val summary = formatToolSummary(msg.toolName ?: "tool", msg.toolArgs ?: "")
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(10.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFFFFF8E1))
            ) {
                Column(modifier = Modifier.padding(10.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(summary.iconEmoji, fontSize = 14.sp)
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = summary.title,
                                fontWeight = FontWeight.Bold,
                                fontSize = 13.sp,
                                color = Color(0xFFBF360C)
                            )
                        }

                        if (msg.isRunning) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(12.dp),
                                    strokeWidth = 1.5.dp,
                                    color = Color(0xFFBF360C)
                                )
                                Spacer(modifier = Modifier.width(4.dp))
                                Text("running...", fontSize = 10.sp, color = Color.Gray)
                            }
                        }
                    }

                    if (!summary.primaryParam.isNullOrBlank()) {
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = summary.primaryParam,
                            fontSize = 12.sp,
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Medium,
                            color = Color(0xFF3E2723),
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                    } else if (!summary.detail.isNullOrBlank()) {
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = summary.detail,
                            fontSize = 11.sp,
                            fontFamily = FontFamily.Monospace,
                            color = Color(0xFF5D4037),
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                }
            }
        }
        MessageRole.TOOL_RESULT -> {
            val isLong = (msg.toolResult ?: "").length > 180 || (msg.toolResult ?: "").contains("\n")
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(10.dp),
                colors = CardDefaults.cardColors(
                    containerColor = if (msg.toolSuccess) Color(0xFFE8F5E9) else Color(0xFFFFEBEE)
                )
            ) {
                Column(modifier = Modifier.padding(10.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(
                                text = if (msg.toolSuccess) "✓" else "✗",
                                color = if (msg.toolSuccess) Color(0xFF2E7D32) else Color(0xFFC62828),
                                fontWeight = FontWeight.Bold
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = if (msg.toolSuccess) "${msg.toolName ?: "tool"} result" else "${msg.toolName ?: "tool"} failed",
                                fontWeight = FontWeight.SemiBold,
                                fontSize = 12.sp,
                                color = if (msg.toolSuccess) Color(0xFF1B5E20) else Color(0xFFB71C1C)
                            )
                        }

                        if (isLong) {
                            Text(
                                text = if (msg.isOutputExpanded) "Collapse" else "Show full",
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Bold,
                                color = if (msg.toolSuccess) Color(0xFF2E7D32) else Color(0xFFC62828),
                                modifier = Modifier.clickable { onToggleOutput() }
                            )
                        }
                    }

                    if (!msg.toolResult.isNullOrBlank()) {
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = msg.toolResult,
                            fontSize = 11.sp,
                            fontFamily = FontFamily.Monospace,
                            color = Color(0xFF37474F),
                            maxLines = if (msg.isOutputExpanded) Int.MAX_VALUE else 4,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                }
            }
        }
        MessageRole.PLAN -> {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFFE0F7FA))
            ) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Text(
                        text = "📋 Execution Plan:",
                        fontWeight = FontWeight.Bold,
                        fontSize = 13.sp,
                        color = Color(0xFF006064)
                    )
                    Spacer(modifier = Modifier.height(6.dp))
                    for (item in msg.planItems) {
                        val icon = when (item.status.lowercase()) {
                            "completed" -> "✓"
                            "in_progress" -> "⏳"
                            "cancelled" -> "✕"
                            else -> "○"
                        }
                        val color = when (item.status.lowercase()) {
                            "completed" -> Color(0xFF2E7D32)
                            "in_progress" -> Color(0xFFE65100)
                            "cancelled" -> Color(0xFFC62828)
                            else -> Color(0xFF546E7A)
                        }
                        Row(
                            modifier = Modifier.padding(vertical = 2.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(icon, color = color, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = "[${item.id}] ${item.content}",
                                fontSize = 12.sp,
                                color = color
                            )
                        }
                    }
                }
            }
        }
        MessageRole.SYSTEM -> {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp),
                contentAlignment = Alignment.Center
            ) {
                Surface(
                    shape = RoundedCornerShape(8.dp),
                    color = Color(0xFFECEFF1)
                ) {
                    Text(
                        text = msg.text,
                        fontSize = 12.sp,
                        color = Color(0xFF455A64),
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
                    )
                }
            }
        }
    }
}

// MARK: - Clarification Card

@Composable
fun ClarificationCard(
    request: ClarificationRequest,
    inputText: String,
    onInputChange: (String) -> Unit,
    onSubmit: (String) -> Unit
) {
    ElevatedCard(
        modifier = Modifier
            .fillMaxWidth()
            .padding(12.dp),
        shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.elevatedCardColors(containerColor = Color(0xFFF3E5F5))
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("❓", fontSize = 16.sp)
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = "Agent Clarification Request",
                    fontWeight = FontWeight.Bold,
                    fontSize = 14.sp,
                    color = Color(0xFF6A1B9A)
                )
            }

            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = request.question,
                fontSize = 14.sp,
                color = Color(0xFF4A148C),
                fontWeight = FontWeight.Medium
            )

            if (request.options.isNotEmpty()) {
                Spacer(modifier = Modifier.height(8.dp))
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    request.options.forEachIndexed { index, opt ->
                        OutlinedButton(
                            onClick = { onSubmit(opt) },
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(8.dp)
                        ) {
                            Text("${index + 1}) $opt", fontSize = 13.sp)
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedTextField(
                    value = inputText,
                    onValueChange = onInputChange,
                    modifier = Modifier.weight(1f),
                    placeholder = { Text("Your reply...") },
                    singleLine = true,
                    shape = RoundedCornerShape(8.dp)
                )
                Button(
                    onClick = { onSubmit(inputText.trim()) },
                    enabled = inputText.trim().isNotEmpty(),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Text("Reply")
                }
            }
        }
    }
}

// MARK: - Settings Dialog

@Composable
fun SettingsDialog(
    config: AppConfig,
    rootDir: String,
    onDismiss: () -> Unit,
    onSave: (AppConfig) -> Unit
) {
    var apiKey by remember { mutableStateOf(config.apiKey) }
    var baseUrl by remember { mutableStateOf(config.baseUrl) }
    var model by remember { mutableStateOf(config.model) }
    var apiBackend by remember { mutableStateOf(config.apiBackend) }
    var maxTurns by remember { mutableStateOf(config.maxTurns) }
    var enableWebSearch by remember { mutableStateOf(config.enableWebSearch) }
    var workspaceName by remember { mutableStateOf(config.workspaceName) }
    var sandboxPath by remember { mutableStateOf(rootDir) }
    var passwordVisible by remember { mutableStateOf(false) }
    var importStatus by remember { mutableStateOf<String?>(null) }
    val context = androidx.compose.ui.platform.LocalContext.current

    fun applyLoadedConfig(loaded: AppConfig, sourceLabel: String) {
        apiKey = loaded.apiKey
        baseUrl = loaded.baseUrl
        model = loaded.model
        apiBackend = loaded.apiBackend
        maxTurns = loaded.maxTurns
        enableWebSearch = loaded.enableWebSearch
        workspaceName = loaded.workspaceName
        sandboxPath = loaded.resolveRootDir(context)
        importStatus = "✓ $sourceLabel (Workspace: ${loaded.workspaceName})"
    }

    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri != null) {
            val loaded = AppConfig.loadFromUri(context, uri)
            if (loaded != null) {
                applyLoadedConfig(loaded, "Loaded config from selected file")
            } else {
                importStatus = "Failed to parse config from selected file"
            }
        }
    }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false)
    ) {
        Surface(
            modifier = Modifier
                .fillMaxWidth(0.92f)
                .fillMaxHeight(0.88f),
            shape = RoundedCornerShape(16.dp),
            color = MaterialTheme.colorScheme.surface,
            tonalElevation = 6.dp
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(20.dp)
            ) {
                // Header
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Agent Settings",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold
                    )
                    IconButton(onClick = onDismiss) {
                        Icon(Icons.Default.Close, contentDescription = "Close")
                    }
                }

                HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

                // Scrollable Form
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(14.dp)
                ) {
                    // Import Section
                    Text(
                        text = "Import config.json File",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        OutlinedButton(
                            onClick = {
                                val found = AppConfig.findDownloadConfigFiles(context)
                                if (found.isNotEmpty()) {
                                    val target = found.first()
                                    val loaded = AppConfig.loadFromFile(target)
                                    if (loaded != null) {
                                        applyLoadedConfig(loaded, "Loaded from ${target.path}")
                                    } else {
                                        importStatus = "Failed parsing ${target.path}"
                                    }
                                } else {
                                    importStatus = "No config.json found in /sdcard/Download"
                                }
                            },
                            modifier = Modifier.weight(1f),
                            contentPadding = PaddingValues(horizontal = 8.dp, vertical = 6.dp)
                        ) {
                            Text("📥 /sdcard/Download", fontSize = 12.sp, maxLines = 1)
                        }

                        OutlinedButton(
                            onClick = {
                                filePickerLauncher.launch(arrayOf("application/json", "text/plain", "*/*"))
                            },
                            modifier = Modifier.weight(1f),
                            contentPadding = PaddingValues(horizontal = 8.dp, vertical = 6.dp)
                        ) {
                            Text("📁 Browse File...", fontSize = 12.sp, maxLines = 1)
                        }
                    }

                    if (importStatus != null) {
                        Text(
                            text = importStatus!!,
                            fontSize = 11.sp,
                            color = if (importStatus!!.startsWith("✓")) Color(0xFF2E7D32) else Color(0xFFC62828)
                        )
                    }

                    HorizontalDivider()

                    // Quick Presets Row
                    Text(
                        text = "Quick Provider Presets",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )

                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        OutlinedButton(
                            onClick = {
                                baseUrl = "https://api.x.ai/v1"
                                model = "grok-4.6"
                                apiBackend = "responses"
                                enableWebSearch = true
                            },
                            contentPadding = PaddingValues(horizontal = 10.dp, vertical = 6.dp)
                        ) {
                            Text("🚀 xAI Grok", fontSize = 12.sp)
                        }

                        OutlinedButton(
                            onClick = {
                                baseUrl = "https://api.deepseek.com/v1"
                                model = "deepseek-v4-flash"
                                apiBackend = "chat_completions"
                                enableWebSearch = false
                            },
                            contentPadding = PaddingValues(horizontal = 10.dp, vertical = 6.dp)
                        ) {
                            Text("⚡ DeepSeek", fontSize = 12.sp)
                        }

                        OutlinedButton(
                            onClick = {
                                baseUrl = "https://api.openai.com/v1"
                                model = "gpt-5.6-sol"
                                apiBackend = "responses"
                                enableWebSearch = false
                            },
                            contentPadding = PaddingValues(horizontal = 10.dp, vertical = 6.dp)
                        ) {
                            Text("🧠 OpenAI", fontSize = 12.sp)
                        }

                        OutlinedButton(
                            onClick = {
                                baseUrl = "https://api.anthropic.com/v1"
                                model = "claude-fable-5"
                                apiBackend = "messages"
                                enableWebSearch = false
                            },
                            contentPadding = PaddingValues(horizontal = 10.dp, vertical = 6.dp)
                        ) {
                            Text("🎭 Anthropic", fontSize = 12.sp)
                        }

                        OutlinedButton(
                            onClick = {
                                baseUrl = "http://10.0.2.2:11434/v1"
                                model = "llama3.1"
                                apiBackend = "chat_completions"
                                enableWebSearch = false
                            },
                            contentPadding = PaddingValues(horizontal = 10.dp, vertical = 6.dp)
                        ) {
                            Text("🦙 Ollama Local", fontSize = 12.sp)
                        }

                        OutlinedButton(
                            onClick = {
                                baseUrl = "http://10.0.2.2:8000/v1"
                                model = "deepseek-v4-flash"
                                apiBackend = "chat_completions"
                                enableWebSearch = false
                            },
                            contentPadding = PaddingValues(horizontal = 10.dp, vertical = 6.dp)
                        ) {
                            Text("⚡ vLLM Local", fontSize = 12.sp)
                        }
                    }

                    HorizontalDivider()

                    // API Key Field
                    Text(
                        text = "API Key",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold
                    )
                    OutlinedTextField(
                        value = apiKey,
                        onValueChange = { apiKey = it },
                        modifier = Modifier.fillMaxWidth(),
                        placeholder = { Text("API Key (e.g. sk-... or xai-...)") },
                        singleLine = true,
                        visualTransformation = if (passwordVisible) VisualTransformation.None else PasswordVisualTransformation(),
                        trailingIcon = {
                            IconButton(onClick = { passwordVisible = !passwordVisible }) {
                                Icon(
                                    if (passwordVisible) Icons.Default.VisibilityOff else Icons.Default.Visibility,
                                    contentDescription = "Toggle password"
                                )
                            }
                        }
                    )

                    // Base URL Field
                    Text(
                        text = "Base URL",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold
                    )
                    OutlinedTextField(
                        value = baseUrl,
                        onValueChange = { baseUrl = it },
                        modifier = Modifier.fillMaxWidth(),
                        placeholder = { Text("https://api.x.ai/v1") },
                        singleLine = true
                    )

                    // Backend Protocol Selection
                    Text(
                        text = "API Backend Protocol",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        listOf(
                            Triple("responses", "responses", "OpenAI / xAI Responses API (Recommended for agent turns)"),
                            Triple("chat_completions", "chat_completions", "Standard OpenAI / DeepSeek / Ollama Chat Completions"),
                            Triple("messages", "messages", "Anthropic Claude Messages API")
                        ).forEach { (id, label, desc) ->
                            val isSelected = apiBackend == id
                            Surface(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clickable { apiBackend = id },
                                shape = RoundedCornerShape(8.dp),
                                color = if (isSelected) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant
                            ) {
                                Row(
                                    modifier = Modifier.padding(10.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    RadioButton(
                                        selected = isSelected,
                                        onClick = { apiBackend = id }
                                    )
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Column {
                                        Text(label, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                                        Text(desc, fontSize = 11.sp, color = Color.Gray)
                                    }
                                }
                            }
                        }
                    }

                    // Model Name Field
                    Text(
                        text = "Model Identifier",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold
                    )
                    OutlinedTextField(
                        value = model,
                        onValueChange = { model = it },
                        modifier = Modifier.fillMaxWidth(),
                        placeholder = { Text("deepseek-v4-flash, grok-4.6, gpt-5.6-sol...") },
                        singleLine = true
                    )

                    // Max Turns Slider
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "Max Turns per Turn",
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "$maxTurns turns",
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                    Slider(
                        value = maxTurns.toFloat(),
                        onValueChange = { maxTurns = it.toInt() },
                        valueRange = 1f..50f,
                        steps = 48
                    )

                    // Web Search Switch
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = "Live Web Search & Fetch",
                                fontSize = 13.sp,
                                fontWeight = FontWeight.Bold
                            )
                            Text(
                                text = "Uses hidden Android WebView for JavaScript rendering & DuckDuckGo search.",
                                fontSize = 11.sp,
                                color = Color.Gray
                            )
                        }
                        Switch(
                            checked = enableWebSearch,
                            onCheckedChange = { enableWebSearch = it }
                        )
                    }

                    // Root Directory Display
                    Text(
                        text = "Sandbox Workspace",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "`root_dir` from config.json is used as the folder name under files/. Example: ./workspace → files/workspace.",
                        fontSize = 11.sp,
                        color = Color.Gray
                    )
                    Surface(
                        color = MaterialTheme.colorScheme.surfaceVariant,
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(modifier = Modifier.padding(10.dp)) {
                            Text(
                                text = workspaceName,
                                fontSize = 14.sp,
                                fontFamily = FontFamily.Monospace,
                                fontWeight = FontWeight.Bold
                            )
                            Text(
                                text = sandboxPath,
                                fontSize = 11.sp,
                                fontFamily = FontFamily.Monospace,
                                color = Color.DarkGray
                            )
                        }
                    }

                    // Reset Button
                    TextButton(
                        onClick = {
                            apiKey = ""
                            baseUrl = "https://api.x.ai/v1"
                            model = "grok-4.6"
                            apiBackend = "responses"
                            maxTurns = 24
                            enableWebSearch = true
                            workspaceName = AppConfig.DEFAULT_WORKSPACE_NAME
                            sandboxPath = AppConfig(workspaceName = AppConfig.DEFAULT_WORKSPACE_NAME).resolveRootDir(context)
                        },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Reset to config.json Defaults", color = Color.Red)
                    }
                }

                HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

                // Action Buttons
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.End,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    OutlinedButton(onClick = onDismiss) {
                        Text("Cancel")
                    }
                    Spacer(modifier = Modifier.width(10.dp))
                    Button(
                        onClick = {
                            onSave(
                                AppConfig(
                                    apiKey = apiKey.trim(),
                                    baseUrl = baseUrl.trim(),
                                    model = model.trim(),
                                    apiBackend = apiBackend,
                                    maxTurns = maxTurns,
                                    enableWebSearch = enableWebSearch,
                                    workspaceName = AppConfig.sanitizeWorkspaceName(workspaceName)
                                )
                            )
                        }
                    ) {
                        Text("Save & Apply")
                    }
                }
            }
        }
    }
}
