#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <signal.h>
#include <errno.h>
#include <ctype.h>
#include <unistd.h>
#include <time.h>
#include "phone_buddy.h"

/* ══════════════════════════════════════════════════════════════════════════ */
/*                           Lightweight JSON Parser                          */
/* ══════════════════════════════════════════════════════════════════════════ */

typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} JsonType;

typedef struct JsonValue JsonValue;

typedef struct JsonMember {
    char *key;
    JsonValue *value;
    struct JsonMember *next;
} JsonMember;

typedef struct JsonElement {
    JsonValue *value;
    struct JsonElement *next;
} JsonElement;

struct JsonValue {
    JsonType type;
    union {
        bool bool_val;
        double num_val;
        char *str_val;
        JsonMember *obj_head;
        JsonElement *arr_head;
    } u;
};

static void json_free(JsonValue *v) {
    if (!v) return;
    switch (v->type) {
        case JSON_STRING:
            free(v->u.str_val);
            break;
        case JSON_ARRAY: {
            JsonElement *curr = v->u.arr_head;
            while (curr) {
                JsonElement *next = curr->next;
                json_free(curr->value);
                free(curr);
                curr = next;
            }
            break;
        }
        case JSON_OBJECT: {
            JsonMember *curr = v->u.obj_head;
            while (curr) {
                JsonMember *next = curr->next;
                free(curr->key);
                json_free(curr->value);
                free(curr);
                curr = next;
            }
            break;
        }
        default:
            break;
    }
    free(v);
}

static void skip_ws(const char **p) {
    while (**p && (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r')) {
        (*p)++;
    }
}

static bool parse_hex4(const char **p, uint32_t *out) {
    uint32_t val = 0;
    for (int i = 0; i < 4; i++) {
        char c = **p;
        (*p)++;
        val <<= 4;
        if (c >= '0' && c <= '9') val |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') val |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') val |= (uint32_t)(c - 'A' + 10);
        else return false;
    }
    *out = val;
    return true;
}

static void utf8_encode(uint32_t cp, char *buf, size_t *len) {
    if (cp <= 0x7F) {
        buf[(*len)++] = (char)cp;
    } else if (cp <= 0x7FF) {
        buf[(*len)++] = (char)(0xC0 | ((cp >> 6) & 0x1F));
        buf[(*len)++] = (char)(0x80 | (cp & 0x3F));
    } else if (cp <= 0xFFFF) {
        buf[(*len)++] = (char)(0xE0 | ((cp >> 12) & 0x0F));
        buf[(*len)++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[(*len)++] = (char)(0x80 | (cp & 0x3F));
    } else if (cp <= 0x10FFFF) {
        buf[(*len)++] = (char)(0xF0 | ((cp >> 18) & 0x07));
        buf[(*len)++] = (char)(0x80 | ((cp >> 12) & 0x3F));
        buf[(*len)++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[(*len)++] = (char)(0x80 | (cp & 0x3F));
    }
}

static char *parse_json_string(const char **p) {
    if (**p != '"') return NULL;
    (*p)++; // Skip opening quote

    size_t cap = 64;
    size_t len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) return NULL;

    while (**p) {
        if (**p == '"') {
            (*p)++; // Skip closing quote
            buf[len] = '\0';
            return buf;
        }

        if (len + 8 >= cap) {
            cap *= 2;
            char *nb = (char *)realloc(buf, cap);
            if (!nb) { free(buf); return NULL; }
            buf = nb;
        }

        if (**p == '\\') {
            (*p)++;
            switch (**p) {
                case '"':  buf[len++] = '"';  (*p)++; break;
                case '\\': buf[len++] = '\\'; (*p)++; break;
                case '/':  buf[len++] = '/';  (*p)++; break;
                case 'b':  buf[len++] = '\b'; (*p)++; break;
                case 'f':  buf[len++] = '\f'; (*p)++; break;
                case 'n':  buf[len++] = '\n'; (*p)++; break;
                case 'r':  buf[len++] = '\r'; (*p)++; break;
                case 't':  buf[len++] = '\t'; (*p)++; break;
                case 'u': {
                    (*p)++;
                    uint32_t cp = 0;
                    if (parse_hex4(p, &cp)) {
                        if (cp >= 0xD800 && cp <= 0xDBFF && **p == '\\' && *(*p + 1) == 'u') {
                            *p += 2;
                            uint32_t low = 0;
                            if (parse_hex4(p, &low) && low >= 0xDC00 && low <= 0xDFFF) {
                                cp = 0x10000 + (((cp & 0x3FF) << 10) | (low & 0x3FF));
                            }
                        }
                        utf8_encode(cp, buf, &len);
                    }
                    break;
                }
                default:
                    if (**p) {
                        buf[len++] = **p;
                        (*p)++;
                    }
                    break;
            }
        } else {
            buf[len++] = **p;
            (*p)++;
        }
    }

    free(buf);
    return NULL;
}

static JsonValue *parse_json_value(const char **p);

static JsonValue *parse_json_object(const char **p) {
    if (**p != '{') return NULL;
    (*p)++; // skip '{'

    JsonValue *val = (JsonValue *)calloc(1, sizeof(JsonValue));
    if (!val) return NULL;
    val->type = JSON_OBJECT;

    JsonMember **tail = &val->u.obj_head;

    while (**p) {
        skip_ws(p);
        if (**p == '}') {
            (*p)++;
            return val;
        }

        if (**p != '"') break;
        char *key = parse_json_string(p);
        if (!key) break;

        skip_ws(p);
        if (**p != ':') { free(key); break; }
        (*p)++; // skip ':'

        skip_ws(p);
        JsonValue *child = parse_json_value(p);
        if (!child) { free(key); break; }

        JsonMember *mem = (JsonMember *)malloc(sizeof(JsonMember));
        if (!mem) { free(key); json_free(child); break; }
        mem->key = key;
        mem->value = child;
        mem->next = NULL;

        *tail = mem;
        tail = &mem->next;

        skip_ws(p);
        if (**p == ',') {
            (*p)++;
        } else if (**p == '}') {
            (*p)++;
            return val;
        } else {
            break;
        }
    }

    json_free(val);
    return NULL;
}

static JsonValue *parse_json_array(const char **p) {
    if (**p != '[') return NULL;
    (*p)++; // skip '['

    JsonValue *val = (JsonValue *)calloc(1, sizeof(JsonValue));
    if (!val) return NULL;
    val->type = JSON_ARRAY;

    JsonElement **tail = &val->u.arr_head;

    while (**p) {
        skip_ws(p);
        if (**p == ']') {
            (*p)++;
            return val;
        }

        JsonValue *elem_val = parse_json_value(p);
        if (!elem_val) break;

        JsonElement *elem = (JsonElement *)malloc(sizeof(JsonElement));
        if (!elem) { json_free(elem_val); break; }
        elem->value = elem_val;
        elem->next = NULL;

        *tail = elem;
        tail = &elem->next;

        skip_ws(p);
        if (**p == ',') {
            (*p)++;
        } else if (**p == ']') {
            (*p)++;
            return val;
        } else {
            break;
        }
    }

    json_free(val);
    return NULL;
}

static JsonValue *parse_json_value(const char **p) {
    skip_ws(p);
    if (!**p) return NULL;

    if (**p == '{') return parse_json_object(p);
    if (**p == '[') return parse_json_array(p);
    if (**p == '"') {
        char *s = parse_json_string(p);
        if (!s) return NULL;
        JsonValue *v = (JsonValue *)calloc(1, sizeof(JsonValue));
        v->type = JSON_STRING;
        v->u.str_val = s;
        return v;
    }
    if (strncmp(*p, "true", 4) == 0) {
        *p += 4;
        JsonValue *v = (JsonValue *)calloc(1, sizeof(JsonValue));
        v->type = JSON_BOOL;
        v->u.bool_val = true;
        return v;
    }
    if (strncmp(*p, "false", 5) == 0) {
        *p += 5;
        JsonValue *v = (JsonValue *)calloc(1, sizeof(JsonValue));
        v->type = JSON_BOOL;
        v->u.bool_val = false;
        return v;
    }
    if (strncmp(*p, "null", 4) == 0) {
        *p += 4;
        JsonValue *v = (JsonValue *)calloc(1, sizeof(JsonValue));
        v->type = JSON_NULL;
        return v;
    }

    if (**p == '-' || (**p >= '0' && **p <= '9')) {
        char *endptr = NULL;
        double num = strtod(*p, &endptr);
        if (endptr != *p) {
            *p = endptr;
            JsonValue *v = (JsonValue *)calloc(1, sizeof(JsonValue));
            v->type = JSON_NUMBER;
            v->u.num_val = num;
            return v;
        }
    }

    return NULL;
}

static JsonValue *json_parse(const char *json_str) {
    if (!json_str) return NULL;
    const char *p = json_str;
    JsonValue *root = parse_json_value(&p);
    return root;
}

static JsonValue *json_obj_get(const JsonValue *obj, const char *key) {
    if (!obj || obj->type != JSON_OBJECT) return NULL;
    for (JsonMember *m = obj->u.obj_head; m; m = m->next) {
        if (strcmp(m->key, key) == 0) return m->value;
    }
    return NULL;
}

static const char *json_obj_get_str(const JsonValue *obj, const char *key) {
    JsonValue *v = json_obj_get(obj, key);
    return (v && v->type == JSON_STRING) ? v->u.str_val : NULL;
}

static bool json_obj_get_bool(const JsonValue *obj, const char *key, bool def_val) {
    JsonValue *v = json_obj_get(obj, key);
    return (v && v->type == JSON_BOOL) ? v->u.bool_val : def_val;
}

static double json_obj_get_num(const JsonValue *obj, const char *key, double def_val) {
    JsonValue *v = json_obj_get(obj, key);
    return (v && v->type == JSON_NUMBER) ? v->u.num_val : def_val;
}

/* ══════════════════════════════════════════════════════════════════════════ */
/*                           UI State & TUI Helpers                           */
/* ══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    bool in_reasoning;
    bool in_text;
    int tool_call_count;
    PbEngine *engine;
    const char *session_id;
} AgentUiContext;

static volatile sig_atomic_t g_running = 1;
static volatile sig_atomic_t g_in_turn = 0;
static volatile sig_atomic_t g_cancel_requested = 0;
static PbEngine *g_engine = NULL;
static char g_session_id[128] = {0};
static bool g_is_resumed_session = false;

static void generate_uuid_v4(char *out, size_t out_len) {
    if (!out || out_len < 37) return;
    uint8_t bytes[16];
    bool got_random = false;
    FILE *f = fopen("/dev/urandom", "rb");
    if (f) {
        if (fread(bytes, 1, 16, f) == 16) {
            got_random = true;
        }
        fclose(f);
    }
    if (!got_random) {
        srand((unsigned int)(time(NULL) ^ getpid() ^ (uintptr_t)&bytes));
        for (int i = 0; i < 16; i++) {
            bytes[i] = (uint8_t)(rand() & 0xFF);
        }
    }
    bytes[6] = (bytes[6] & 0x0F) | 0x40; // UUID version 4
    bytes[8] = (bytes[8] & 0x3F) | 0x80; // RFC 4122 variant
    snprintf(out, out_len,
             "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             bytes[0], bytes[1], bytes[2], bytes[3],
             bytes[4], bytes[5],
             bytes[6], bytes[7],
             bytes[8], bytes[9],
             bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]);
}

static void handle_sigint(int signo) {
    (void)signo;
    if (g_in_turn && g_engine) {
        if (g_cancel_requested) {
            // Second Ctrl+C pressed while cancel is in progress: force exit immediately
            const char msg[] = "\n\x1b[1;31m[Force exiting PhoneBuddy Agent...]\x1b[0m\n";
            (void)write(STDOUT_FILENO, msg, sizeof(msg) - 1);
            _exit(130);
        }
        g_cancel_requested = 1;
        // Cancel the current in-flight chat turn
        pb_engine_cancel(g_engine, g_session_id);
        const char msg[] = "\n\x1b[1;33m[Cancelling current turn... (press Ctrl+C again to force exit)]\x1b[0m\n";
        (void)write(STDOUT_FILENO, msg, sizeof(msg) - 1);
    } else {
        // Signal to exit the REPL
        g_running = 0;
        const char msg[] = "\n\x1b[1;33m[Exiting PhoneBuddy Agent...]\x1b[0m\n";
        (void)write(STDOUT_FILENO, msg, sizeof(msg) - 1);
    }
}

static void print_compact_preview(const char *text, size_t max_len) {
    if (!text || !*text) return;
    char buf[512];
    size_t bi = 0;
    bool in_space = false;

    for (const char *p = text; *p && bi < sizeof(buf) - 5; p++) {
        if (*p == '\n' || *p == '\r' || *p == '\t' || *p == ' ') {
            if (!in_space && bi > 0) {
                buf[bi++] = ' ';
                in_space = true;
            }
        } else {
            buf[bi++] = *p;
            in_space = false;
        }
        if (bi >= max_len) break;
    }

    if (bi >= max_len && strlen(text) > max_len) {
        strcpy(&buf[bi], "...");
    } else {
        buf[bi] = '\0';
    }

    printf("%s", buf);
}

static void print_output_preview(const char *output, size_t max_lines, size_t max_line_chars) {
    if (!output || !*output) {
        printf("    \x1b[2m(empty output)\x1b[0m\n");
        return;
    }

    size_t total_lines = 1;
    for (const char *p = output; *p; p++) {
        if (*p == '\n' && *(p + 1)) total_lines++;
    }

    size_t lines_printed = 0;
    const char *curr = output;

    while (*curr && lines_printed < max_lines) {
        const char *next = strchr(curr, '\n');
        size_t len = next ? (size_t)(next - curr) : strlen(curr);

        if (len > max_line_chars) {
            printf("    \x1b[2m%.*s...\x1b[0m\n", (int)(max_line_chars - 3), curr);
        } else {
            printf("    \x1b[2m%.*s\x1b[0m\n", (int)len, curr);
        }

        lines_printed++;
        if (!next) break;
        curr = next + 1;
    }

    if (total_lines > lines_printed) {
        printf("    \x1b[2;3m... (+%zu more lines)\x1b[0m\n", total_lines - lines_printed);
    }
}

static void render_plan_items(const char *items_json) {
    if (!items_json || !*items_json) return;
    JsonValue *root = json_parse(items_json);
    if (!root) {
        printf("  \x1b[1;36m📋 Plan:\x1b[0m \x1b[36m%s\x1b[0m\n", items_json);
        return;
    }

    if (root->type == JSON_ARRAY) {
        printf("  \x1b[1;36m📋 Plan updated:\x1b[0m\n");
        for (JsonElement *elem = root->u.arr_head; elem; elem = elem->next) {
            if (elem->value->type == JSON_OBJECT) {
                const char *id = json_obj_get_str(elem->value, "id");
                const char *content = json_obj_get_str(elem->value, "content");
                const char *status = json_obj_get_str(elem->value, "status");
                if (!content) content = "";
                if (!id) id = "-";

                if (status && strcmp(status, "completed") == 0) {
                    printf("    \x1b[1;32m✓\x1b[0m \x1b[2;37m[%s] %s\x1b[0m\n", id, content);
                } else if (status && strcmp(status, "in_progress") == 0) {
                    printf("    \x1b[1;33m⏳\x1b[0m \x1b[1;37m[%s] %s\x1b[0m\n", id, content);
                } else if (status && strcmp(status, "cancelled") == 0) {
                    printf("    \x1b[1;31m✕\x1b[0m \x1b[2;37m[%s] %s\x1b[0m\n", id, content);
                } else {
                    printf("    \x1b[2m○ [%s] %s\x1b[0m\n", id, content);
                }
            }
        }
    } else {
        printf("  \x1b[1;36m📋 Plan:\x1b[0m \x1b[36m%s\x1b[0m\n", items_json);
    }
    json_free(root);
}

/* ══════════════════════════════════════════════════════════════════════════ */
/*                          Agent Event Callback                              */
/* ══════════════════════════════════════════════════════════════════════════ */

static void on_agent_event(const char *event_json, void *user_data) {
    if (!event_json) return;
    AgentUiContext *ctx = (AgentUiContext *)user_data;

    JsonValue *root = json_parse(event_json);
    if (!root) {
        printf("%s\n", event_json);
        fflush(stdout);
        return;
    }

    if (root->type != JSON_OBJECT) {
        json_free(root);
        return;
    }

    // Check externally tagged variant in root object:
    // e.g. {"TextDelta": {"text": "..."}}
    JsonMember *first = root->u.obj_head;
    if (!first) {
        json_free(root);
        return;
    }

    const char *tag = first->key;
    JsonValue *payload = first->value;

    if (strcmp(tag, "ReasoningDelta") == 0) {
        const char *text = json_obj_get_str(payload, "text");
        if (text && *text) {
            if (!ctx->in_reasoning) {
                if (ctx->in_text) {
                    printf("\n");
                    ctx->in_text = false;
                }
                printf("\x1b[2;3m💭 Thinking...\x1b[0m\n\x1b[2m");
                ctx->in_reasoning = true;
            }
            printf("%s", text);
            fflush(stdout);
        }
    } else if (strcmp(tag, "TextDelta") == 0) {
        const char *text = json_obj_get_str(payload, "text");
        if (text && *text) {
            if (ctx->in_reasoning) {
                printf("\x1b[0m\n\n");
                ctx->in_reasoning = false;
            }
            printf("%s", text);
            fflush(stdout);
            ctx->in_text = true;
        }
    } else if (strcmp(tag, "ToolCallStart") == 0) {
        if (ctx->in_reasoning) {
            printf("\x1b[0m\n");
            ctx->in_reasoning = false;
        }
        if (ctx->in_text) {
            printf("\n");
            ctx->in_text = false;
        }

        const char *name = json_obj_get_str(payload, "name");
        const char *args = json_obj_get_str(payload, "arguments_json");
        ctx->tool_call_count++;

        printf("\n\x1b[1;33m▶ tool call:\x1b[0m \x1b[1;37m%s\x1b[0m\x1b[2m(", name ? name : "tool");
        if (args) {
            print_compact_preview(args, 140);
        }
        printf(")\x1b[0m\n");
        fflush(stdout);
    } else if (strcmp(tag, "ToolCallResult") == 0) {
        if (ctx->in_reasoning) {
            printf("\x1b[0m\n");
            ctx->in_reasoning = false;
        }
        if (ctx->in_text) {
            printf("\n");
            ctx->in_text = false;
        }

        const char *name = json_obj_get_str(payload, "name");
        bool ok = json_obj_get_bool(payload, "ok", true);
        const char *output = json_obj_get_str(payload, "output");

        if (ok) {
            printf("  \x1b[1;32m✓\x1b[0m \x1b[1;37m%s\x1b[0m →\n", name ? name : "tool");
        } else {
            printf("  \x1b[1;31m✗\x1b[0m \x1b[1;31m%s (failed)\x1b[0m →\n", name ? name : "tool");
        }

        if (output) {
            print_output_preview(output, 6, 120);
        }
        fflush(stdout);
    } else if (strcmp(tag, "PlanUpdated") == 0) {
        if (ctx->in_reasoning) {
            printf("\x1b[0m\n");
            ctx->in_reasoning = false;
        }
        if (ctx->in_text) {
            printf("\n");
            ctx->in_text = false;
        }

        const char *items_json = json_obj_get_str(payload, "items_json");
        render_plan_items(items_json);
        fflush(stdout);
    } else if (strcmp(tag, "Completed") == 0) {
        if (ctx->in_reasoning) {
            printf("\x1b[0m\n");
            ctx->in_reasoning = false;
        }
        if (ctx->in_text) {
            printf("\n");
            ctx->in_text = false;
        }

        JsonValue *usage = json_obj_get(payload, "usage");
        if (usage && usage->type == JSON_OBJECT) {
            double p_tok = json_obj_get_num(usage, "prompt_tokens", 0);
            double c_tok = json_obj_get_num(usage, "completion_tokens", 0);
            double t_tok = json_obj_get_num(usage, "total_tokens", 0);
            printf("\x1b[2m[Done. Tokens: prompt=%.0f, completion=%.0f, total=%.0f]\x1b[0m\n",
                   p_tok, c_tok, t_tok);
        }
        fflush(stdout);
    } else if (strcmp(tag, "Failed") == 0) {
        if (ctx->in_reasoning) {
            printf("\x1b[0m\n");
            ctx->in_reasoning = false;
        }
        if (ctx->in_text) {
            printf("\n");
            ctx->in_text = false;
        }

        const char *msg = json_obj_get_str(payload, "message");
        if (msg && strcmp(msg, "cancelled") == 0) {
            printf("\x1b[1;33m[Turn cancelled]\x1b[0m\n");
        } else {
            printf("\x1b[1;31m✗ Turn failed: %s\x1b[0m\n", msg ? msg : "unknown error");
        }
        fflush(stdout);
    }

    json_free(root);
}

/* ══════════════════════════════════════════════════════════════════════════ */
/*                    Host Tool Callback (ask_user_question)                  */
/* ══════════════════════════════════════════════════════════════════════════ */

static void on_host_tool_request(const char *call_id,
                                 const char *name,
                                 const char *arguments_json,
                                 void *user_data) {
    PbEngine *engine = (PbEngine *)user_data;
    if (!call_id || !name || !engine) return;

    if (strcmp(name, "ask_user_question") == 0) {
        JsonValue *args = json_parse(arguments_json);
        const char *question = args ? json_obj_get_str(args, "question") : NULL;

        printf("\n\x1b[1;35m❓ [Agent Clarification Request]:\x1b[0m \x1b[1;37m%s\x1b[0m\n",
               question ? question : "The agent has a question for you:");

        if (args) {
            JsonValue *options = json_obj_get(args, "options");
            if (options && options->type == JSON_ARRAY) {
                int idx = 1;
                for (JsonElement *elem = options->u.arr_head; elem; elem = elem->next) {
                    if (elem->value->type == JSON_STRING) {
                        printf("   \x1b[35m%d)\x1b[0m %s\n", idx++, elem->value->u.str_val);
                    }
                }
            }
        }

        printf("\x1b[1;35mYour reply ❯\x1b[0m ");
        fflush(stdout);

        char reply_buf[1024];
        if (fgets(reply_buf, sizeof(reply_buf), stdin)) {
            // Trim trailing newline
            size_t len = strlen(reply_buf);
            while (len > 0 && (reply_buf[len - 1] == '\n' || reply_buf[len - 1] == '\r')) {
                reply_buf[--len] = '\0';
            }
            char *err = NULL;
            pb_engine_host_tool_result(engine, call_id, 1, reply_buf, &err);
            if (err) pb_string_free(err);
        } else {
            char *err = NULL;
            pb_engine_host_tool_result(engine, call_id, 0, "No response provided", &err);
            if (err) pb_string_free(err);
        }

        if (args) json_free(args);
    } else {
        // Unknown host tool
        char *err = NULL;
        pb_engine_host_tool_result(engine, call_id, 0, "Unsupported host tool", &err);
        if (err) pb_string_free(err);
    }
}

/* ══════════════════════════════════════════════════════════════════════════ */
/*                          Config File Reader                                */
/* ══════════════════════════════════════════════════════════════════════════ */

static char *read_config_file(const char *filepath) {
    if (!filepath || !*filepath) {
        fprintf(stderr, "\x1b[1;31m[Error] Configuration file path is empty.\x1b[0m\n");
        return NULL;
    }

    FILE *f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "\x1b[1;31m[Error] Cannot open configuration file '%s': %s\x1b[0m\n",
                filepath, strerror(errno));
        return NULL;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "\x1b[1;31m[Error] Cannot seek in configuration file '%s': %s\x1b[0m\n",
                filepath, strerror(errno));
        fclose(f);
        return NULL;
    }

    long file_size = ftell(f);
    if (file_size < 0) {
        fprintf(stderr, "\x1b[1;31m[Error] Invalid size for configuration file '%s'\x1b[0m\n", filepath);
        fclose(f);
        return NULL;
    }
    if (file_size == 0) {
        fprintf(stderr, "\x1b[1;31m[Error] Configuration file '%s' is empty.\x1b[0m\n", filepath);
        fclose(f);
        return NULL;
    }

    if (fseek(f, 0, SEEK_SET) != 0) {
        fprintf(stderr, "\x1b[1;31m[Error] Failed to rewind configuration file '%s'\x1b[0m\n", filepath);
        fclose(f);
        return NULL;
    }

    char *buffer = (char *)malloc(file_size + 1);
    if (!buffer) {
        fprintf(stderr, "\x1b[1;31m[Error] Out of memory allocating %ld bytes for '%s'\x1b[0m\n",
                file_size, filepath);
        fclose(f);
        return NULL;
    }

    size_t read_bytes = fread(buffer, 1, file_size, f);
    fclose(f);

    if (read_bytes != (size_t)file_size) {
        fprintf(stderr, "\x1b[1;31m[Error] Failed to read complete file '%s' (read %zu of %ld bytes)\x1b[0m\n",
                filepath, read_bytes, file_size);
        free(buffer);
        return NULL;
    }

    buffer[file_size] = '\0';

    // Strip // and /* */ comments so users can annotate their config.json
    char *clean = (char *)malloc(file_size + 1);
    if (!clean) {
        return buffer;
    }

    size_t ci = 0;
    bool in_str = false;
    bool esc = false;

    for (size_t i = 0; i < (size_t)file_size; i++) {
        char c = buffer[i];
        if (in_str) {
            clean[ci++] = c;
            if (esc) {
                esc = false;
            } else if (c == '\\') {
                esc = true;
            } else if (c == '"') {
                in_str = false;
            }
        } else {
            if (c == '"') {
                in_str = true;
                clean[ci++] = c;
            } else if (c == '/' && i + 1 < (size_t)file_size && buffer[i + 1] == '/') {
                i += 2;
                while (i < (size_t)file_size && buffer[i] != '\n') {
                    i++;
                }
                if (i < (size_t)file_size && buffer[i] == '\n') {
                    clean[ci++] = '\n';
                }
            } else if (c == '/' && i + 1 < (size_t)file_size && buffer[i + 1] == '*') {
                i += 2;
                while (i + 1 < (size_t)file_size && !(buffer[i] == '*' && buffer[i + 1] == '/')) {
                    if (buffer[i] == '\n') clean[ci++] = '\n';
                    i++;
                }
                if (i + 1 < (size_t)file_size) {
                    i++;
                }
            } else {
                clean[ci++] = c;
            }
        }
    }
    clean[ci] = '\0';
    free(buffer);
    return clean;
}

/* ══════════════════════════════════════════════════════════════════════════ */
/*                           Session Management                               */
/* ══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    char id[128];
    char title[256];
    char updated_at[64];
    int message_count;
    bool found;
} SessionInfo;

static SessionInfo get_latest_session(PbEngine *engine) {
    SessionInfo info;
    memset(&info, 0, sizeof(info));
    if (!engine) return info;

    char *err = NULL;
    char *json_str = pb_engine_list_sessions(engine, &err);
    if (err) {
        pb_string_free(err);
        return info;
    }
    if (!json_str) return info;

    JsonValue *root = json_parse(json_str);
    if (root && root->type == JSON_ARRAY) {
        for (JsonElement *elem = root->u.arr_head; elem; elem = elem->next) {
            if (elem->value->type == JSON_OBJECT) {
                const char *id = json_obj_get_str(elem->value, "id");
                const char *title = json_obj_get_str(elem->value, "title");
                const char *updated_at = json_obj_get_str(elem->value, "updated_at");
                double msg_count = json_obj_get_num(elem->value, "message_count", 0);

                if (id && *id) {
                    if (!info.found || (updated_at && strcmp(updated_at, info.updated_at) > 0)) {
                        info.found = true;
                        strncpy(info.id, id, sizeof(info.id) - 1);
                        info.id[sizeof(info.id) - 1] = '\0';
                        if (title) {
                            strncpy(info.title, title, sizeof(info.title) - 1);
                            info.title[sizeof(info.title) - 1] = '\0';
                        } else {
                            info.title[0] = '\0';
                        }
                        if (updated_at) {
                            strncpy(info.updated_at, updated_at, sizeof(info.updated_at) - 1);
                            info.updated_at[sizeof(info.updated_at) - 1] = '\0';
                        } else {
                            info.updated_at[0] = '\0';
                        }
                        info.message_count = (int)msg_count;
                    }
                }
            }
        }
    }
    if (root) json_free(root);
    pb_string_free(json_str);
    return info;
}

static void list_all_sessions(PbEngine *engine) {
    if (!engine) return;
    char *err = NULL;
    char *json_str = pb_engine_list_sessions(engine, &err);
    if (err) {
        printf("\x1b[1;31mError listing sessions: %s\x1b[0m\n", err);
        pb_string_free(err);
        return;
    }
    if (!json_str) {
        printf("\x1b[2mNo sessions found.\x1b[0m\n\n");
        return;
    }

    JsonValue *root = json_parse(json_str);
    if (!root || root->type != JSON_ARRAY || !root->u.arr_head) {
        printf("\x1b[2mNo existing sessions found.\x1b[0m\n\n");
    } else {
        printf("\n\x1b[1mSaved Sessions:\x1b[0m\n");
        int count = 0;
        for (JsonElement *elem = root->u.arr_head; elem; elem = elem->next) {
            if (elem->value->type == JSON_OBJECT) {
                const char *id = json_obj_get_str(elem->value, "id");
                const char *title = json_obj_get_str(elem->value, "title");
                const char *updated_at = json_obj_get_str(elem->value, "updated_at");
                double msg_count = json_obj_get_num(elem->value, "message_count", 0);
                if (id) {
                    count++;
                    bool is_current = (strcmp(id, g_session_id) == 0);
                    printf("  %s \x1b[1;36m%-36s\x1b[0m  \x1b[2m(%2.0f msgs)\x1b[0m  \x1b[33m%-20s\x1b[0m  \x1b[2m%s\x1b[0m\n",
                           is_current ? "\x1b[1;32m*\x1b[0m" : " ",
                           id,
                           msg_count,
                           (title && *title) ? title : "(no title)",
                           updated_at ? updated_at : "");
                }
            }
        }
        printf("  \x1b[2mTotal: %d session(s) | * indicates active session\x1b[0m\n\n", count);
    }
    if (root) json_free(root);
    pb_string_free(json_str);
}

/**
 * Load and display the complete message history of a session using pb_engine_get_session.
 *
 * Demonstrates the chat history retrieval API: fetches all past turns (user prompts,
 * assistant reasoning, tool calls, and final responses) and renders them in formatted TUI.
 *
 * @param engine Pointer to the active PbEngine
 * @param session_id Session UUID to retrieve and replay
 * @return true if the session was found and replayed successfully, false otherwise
 */
static bool replay_session_history(PbEngine *engine, const char *session_id) {
    if (!engine || !session_id || !*session_id) return false;

    char *err = NULL;
    char *session_json = pb_engine_get_session(engine, session_id, &err);
    if (err) {
        printf("\x1b[1;31mError retrieving session '%s': %s\x1b[0m\n", session_id, err);
        pb_string_free(err);
        return false;
    }
    if (!session_json) {
        printf("\x1b[1;33mSession '%s' not found.\x1b[0m\n", session_id);
        return false;
    }

    JsonValue *root = json_parse(session_json);
    if (!root || root->type != JSON_OBJECT) {
        printf("\x1b[1;31mFailed to parse session JSON for '%s'.\x1b[0m\n", session_id);
        if (root) json_free(root);
        pb_string_free(session_json);
        return false;
    }

    const char *title = json_obj_get_str(root, "title");
    const char *updated_at = json_obj_get_str(root, "updated_at");
    JsonValue *messages_val = json_obj_get(root, "messages");

    printf("\n\x1b[1;36m╭────────────────────────────────────────────────────────────────────────╮\x1b[0m\n");
    printf("\x1b[1;36m│\x1b[0m  \x1b[1;37m📜 Session History Replay\x1b[0m                                             \x1b[1;36m│\x1b[0m\n");
    printf("\x1b[1;36m│\x1b[0m  ID       : \x1b[36m%-57s\x1b[0m\x1b[1;36m│\x1b[0m\n", session_id);
    if (title && *title) {
        printf("\x1b[1;36m│\x1b[0m  Title    : \x1b[33m%-57s\x1b[0m\x1b[1;36m│\x1b[0m\n", title);
    }
    if (updated_at && *updated_at) {
        printf("\x1b[1;36m│\x1b[0m  Updated  : \x1b[2m%-57s\x1b[0m\x1b[1;36m│\x1b[0m\n", updated_at);
    }
    printf("\x1b[1;36m╰────────────────────────────────────────────────────────────────────────╯\x1b[0m\n\n");

    int msg_count = 0;
    if (messages_val && messages_val->type == JSON_ARRAY) {
        for (JsonElement *elem = messages_val->u.arr_head; elem; elem = elem->next) {
            if (!elem->value || elem->value->type != JSON_OBJECT) continue;
            JsonValue *msg = elem->value;
            const char *role = json_obj_get_str(msg, "role");
            const char *content = json_obj_get_str(msg, "content");
            const char *reasoning = json_obj_get_str(msg, "reasoning_content");
            JsonValue *tool_calls = json_obj_get(msg, "tool_calls");

            if (!role) continue;
            msg_count++;

            if (strcmp(role, "user") == 0) {
                printf("\x1b[1;32mUser\x1b[0m \x1b[1;34m❯\x1b[0m %s\n\n", content ? content : "");
            } else if (strcmp(role, "assistant") == 0) {
                if (reasoning && *reasoning) {
                    printf("\x1b[2;37m💭 %s\x1b[0m\n\n", reasoning);
                }
                if (tool_calls && tool_calls->type == JSON_ARRAY) {
                    for (JsonElement *tc = tool_calls->u.arr_head; tc; tc = tc->next) {
                        if (!tc->value || tc->value->type != JSON_OBJECT) continue;
                        JsonValue *fn = json_obj_get(tc->value, "function");
                        if (fn && fn->type == JSON_OBJECT) {
                            const char *fn_name = json_obj_get_str(fn, "name");
                            const char *fn_args = json_obj_get_str(fn, "arguments");
                            printf("\x1b[2m  ▶ tool call: %s(%s)\x1b[0m\n",
                                   fn_name ? fn_name : "unknown",
                                   fn_args ? fn_args : "");
                        }
                    }
                }
                if (content && *content) {
                    printf("\x1b[1;35mAgent\x1b[0m \x1b[1;34m❯\x1b[0m %s\n\n", content);
                }
            } else if (strcmp(role, "tool") == 0) {
                printf("\x1b[2m  ✓ tool result → ");
                print_compact_preview(content ? content : "", 120);
                printf("\x1b[0m\n\n");
            }
        }
    }

    printf("\x1b[1;32m✓ Replayed %d message(s) from session history.\x1b[0m\n\n", msg_count);

    json_free(root);
    pb_string_free(session_json);
    return true;
}

/**
 * Interactive session resume handler.
 *
 * Implements grok-build style `/resume` behavior:
 * - If called with an ID (`/resume <uuid>`), directly switches to and replays that session.
 * - If called without an ID (`/resume`), presents a numbered picker list of saved sessions,
 *   allows the user to enter a list index or UUID, and rehydrates the selected session.
 */
static void handle_resume_command(PbEngine *engine, AgentUiContext *ui_ctx, const char *arg) {
    if (!engine) return;

    // If an argument was provided (e.g. "/resume <uuid>"), resume directly
    if (arg && *arg) {
        char target_uuid[128];
        size_t len = 0;
        const char *p = arg;
        while (*p && *p != ' ' && *p != '\t' && len < sizeof(target_uuid) - 1) {
            target_uuid[len++] = *p++;
        }
        target_uuid[len] = '\0';

        if (replay_session_history(engine, target_uuid)) {
            strncpy(g_session_id, target_uuid, sizeof(g_session_id) - 1);
            g_session_id[sizeof(g_session_id) - 1] = '\0';
            if (ui_ctx) ui_ctx->session_id = g_session_id;
            g_is_resumed_session = true;
            printf("\x1b[1;32m✓ Switched active session to:\x1b[0m \x1b[1;36m%s\x1b[0m\n\n", g_session_id);
        }
        return;
    }

    // Interactive picker: list sessions and let user select
    char *err = NULL;
    char *json_str = pb_engine_list_sessions(engine, &err);
    if (err) {
        printf("\x1b[1;31mError listing sessions: %s\x1b[0m\n", err);
        pb_string_free(err);
        return;
    }
    if (!json_str) {
        printf("\x1b[2mNo saved sessions found.\x1b[0m\n\n");
        return;
    }

    JsonValue *root = json_parse(json_str);
    if (!root || root->type != JSON_ARRAY || !root->u.arr_head) {
        printf("\x1b[2mNo existing saved sessions found.\x1b[0m\n\n");
        if (root) json_free(root);
        pb_string_free(json_str);
        return;
    }

    #define MAX_PICKER_SESSIONS 64
    SessionInfo sessions[MAX_PICKER_SESSIONS];
    int count = 0;

    printf("\n\x1b[1mSelect a session to resume:\x1b[0m\n");
    for (JsonElement *elem = root->u.arr_head; elem && count < MAX_PICKER_SESSIONS; elem = elem->next) {
        if (elem->value && elem->value->type == JSON_OBJECT) {
            const char *id = json_obj_get_str(elem->value, "id");
            const char *title = json_obj_get_str(elem->value, "title");
            const char *updated_at = json_obj_get_str(elem->value, "updated_at");
            double msg_count = json_obj_get_num(elem->value, "message_count", 0);
            if (id && *id) {
                memset(&sessions[count], 0, sizeof(SessionInfo));
                sessions[count].found = true;
                strncpy(sessions[count].id, id, sizeof(sessions[count].id) - 1);
                if (title) strncpy(sessions[count].title, title, sizeof(sessions[count].title) - 1);
                if (updated_at) strncpy(sessions[count].updated_at, updated_at, sizeof(sessions[count].updated_at) - 1);
                sessions[count].message_count = (int)msg_count;

                bool is_current = (strcmp(id, g_session_id) == 0);
                printf("  \x1b[1;33m[%2d]\x1b[0m %s \x1b[1;36m%-36s\x1b[0m  \x1b[2m(%2d msgs)\x1b[0m  \x1b[33m%-20s\x1b[0m  \x1b[2m%s\x1b[0m\n",
                       count + 1,
                       is_current ? "\x1b[1;32m*\x1b[0m" : " ",
                       id,
                       (int)msg_count,
                       (title && *title) ? title : "(no title)",
                       updated_at ? updated_at : "");
                count++;
            }
        }
    }
    printf("  \x1b[2m* indicates active session\x1b[0m\n\n");

    if (root) json_free(root);
    pb_string_free(json_str);

    if (count == 0) {
        return;
    }

    printf("\x1b[1;32mSelect session number [1-%d], enter UUID, or press Enter to cancel:\x1b[0m ", count);
    fflush(stdout);

    char *choice_buf = NULL;
    size_t choice_cap = 0;
    ssize_t nread = getline(&choice_buf, &choice_cap, stdin);
    if (nread <= 0) {
        if (choice_buf) free(choice_buf);
        printf("\x1b[2m[Cancelled]\x1b[0m\n\n");
        return;
    }

    while (nread > 0 && (choice_buf[nread - 1] == '\n' || choice_buf[nread - 1] == '\r')) {
        choice_buf[--nread] = '\0';
    }

    char *trimmed = choice_buf;
    while (*trimmed == ' ' || *trimmed == '\t') trimmed++;

    if (!*trimmed) {
        free(choice_buf);
        printf("\x1b[2m[Cancelled]\x1b[0m\n\n");
        return;
    }

    char chosen_uuid[128] = {0};
    char *endptr = NULL;
    long num = strtol(trimmed, &endptr, 10);
    if (endptr != trimmed && *endptr == '\0') {
        // User entered a valid session number
        if (num >= 1 && num <= count) {
            strncpy(chosen_uuid, sessions[num - 1].id, sizeof(chosen_uuid) - 1);
        } else {
            printf("\x1b[1;31mInvalid selection number %ld.\x1b[0m\n\n", num);
            free(choice_buf);
            return;
        }
    } else {
        // User entered a UUID string
        strncpy(chosen_uuid, trimmed, sizeof(chosen_uuid) - 1);
    }
    free(choice_buf);

    if (chosen_uuid[0]) {
        if (replay_session_history(engine, chosen_uuid)) {
            strncpy(g_session_id, chosen_uuid, sizeof(g_session_id) - 1);
            g_session_id[sizeof(g_session_id) - 1] = '\0';
            if (ui_ctx) ui_ctx->session_id = g_session_id;
            g_is_resumed_session = true;
            printf("\x1b[1;32m✓ Switched active session to:\x1b[0m \x1b[1;36m%s\x1b[0m\n\n", g_session_id);
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════════ */
/*                               Main REPL                                    */
/* ══════════════════════════════════════════════════════════════════════════ */

static void print_banner(const char *config_path, const JsonValue *config_json, const SessionInfo *latest) {
    const char *model = config_json ? json_obj_get_str(config_json, "model") : NULL;
    const char *base_url = config_json ? json_obj_get_str(config_json, "base_url") : NULL;
    const char *root_dir = config_json ? json_obj_get_str(config_json, "root_dir") : NULL;

    printf("\n");
    printf("\x1b[1;36m╭────────────────────────────────────────────────────────────────────────╮\x1b[0m\n");
    printf("\x1b[1;36m│\x1b[0m  \x1b[1;37m🤖 PhoneBuddy C Agent\x1b[0m \x1b[2m(SDK v%s)\x1b[0m                                 \x1b[1;36m│\x1b[0m\n", pb_version());
    printf("\x1b[1;36m│\x1b[0m  Config   : \x1b[33m%-57s\x1b[0m\x1b[1;36m│\x1b[0m\n", config_path);
    printf("\x1b[1;36m│\x1b[0m  Model    : \x1b[32m%-57s\x1b[0m\x1b[1;36m│\x1b[0m\n", model ? model : "default");
    printf("\x1b[1;36m│\x1b[0m  Base URL : \x1b[34m%-57s\x1b[0m\x1b[1;36m│\x1b[0m\n", base_url ? base_url : "https://api.x.ai/v1");
    printf("\x1b[1;36m│\x1b[0m  Sandbox  : \x1b[35m%-57s\x1b[0m\x1b[1;36m│\x1b[0m\n", root_dir ? root_dir : "./workspace");

    char session_display[80];
    if (g_is_resumed_session) {
        snprintf(session_display, sizeof(session_display), "%s (resumed)", g_session_id);
    } else {
        snprintf(session_display, sizeof(session_display), "%s (new)", g_session_id);
    }
    printf("\x1b[1;36m│\x1b[0m  Session  : \x1b[36m%-57s\x1b[0m\x1b[1;36m│\x1b[0m\n", session_display);
    printf("\x1b[1;36m│\x1b[0m  \x1b[2mType 'exit' / 'quit' or press Ctrl+C to quit\x1b[0m                          \x1b[1;36m│\x1b[0m\n");
    printf("\x1b[1;36m╰────────────────────────────────────────────────────────────────────────╯\x1b[0m\n");

    if (latest && latest->found && strcmp(latest->id, g_session_id) != 0) {
        printf("\x1b[1;33m💡 Last session:\x1b[0m \x1b[1;36m%s\x1b[0m", latest->id);
        if (latest->title[0]) {
            printf(" \x1b[2m(\"%s\", %d msgs, updated %s)\x1b[0m\n",
                   latest->title, latest->message_count, latest->updated_at);
        } else {
            printf(" \x1b[2m(%d msgs, updated %s)\x1b[0m\n",
                   latest->message_count, latest->updated_at);
        }
        printf("   \x1b[2mTo resume: enter '\x1b[1;37m/resume %s\x1b[0;2m' or pick via '\x1b[1;37m/resume\x1b[0;2m' or run '\x1b[1;37m./demo -r %s\x1b[0;2m'\x1b[0m\n",
               latest->id, latest->id);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    const char *config_path = "config.json";
    const char *cli_resume_id = NULL;

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--resume") == 0) {
            if (i + 1 < argc) {
                cli_resume_id = argv[++i];
            } else {
                fprintf(stderr, "\x1b[1;31mError: --resume requires a session ID\x1b[0m\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--config") == 0) {
            if (i + 1 < argc) {
                config_path = argv[++i];
            } else {
                fprintf(stderr, "\x1b[1;31mError: --config requires a config file path\x1b[0m\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options] [config.json]\n", argv[0]);
            printf("Options:\n");
            printf("  -r, --resume <uuid>     Resume a previous session by UUID\n");
            printf("  -c, --config <path>     Path to configuration file (default: config.json)\n");
            printf("  -h, --help              Show this help message\n");
            return 0;
        } else if (argv[i][0] != '-') {
            // Positional argument: if it's an existing file or ends in .json, treat as config;
            // otherwise treat as a session uuid to resume.
            if (strstr(argv[i], ".json") != NULL || access(argv[i], F_OK) == 0) {
                config_path = argv[i];
            } else {
                cli_resume_id = argv[i];
            }
        }
    }

    // 1. Read configuration from config.json (no fallbacks)
    char *config_content = read_config_file(config_path);
    if (!config_content) {
        fprintf(stderr, "\x1b[1;31mFatal: Failed to load config from '%s'. Exiting.\x1b[0m\n", config_path);
        return 1;
    }

    JsonValue *parsed_cfg = json_parse(config_content);
    if (!parsed_cfg) {
        fprintf(stderr, "\x1b[1;31mFatal: File '%s' contains invalid JSON. Exiting.\x1b[0m\n", config_path);
        free(config_content);
        return 1;
    }

    // 2. Initialize the PhoneBuddy engine
    char *err = NULL;
    PbEngine *engine = pb_engine_new(config_content, &err);
    free(config_content);

    if (!engine) {
        fprintf(stderr, "\x1b[1;31mFatal: Engine initialization failed: %s\x1b[0m\n",
                err ? err : "unknown error");
        if (err) pb_string_free(err);
        json_free(parsed_cfg);
        return 1;
    }

    g_engine = engine;

    // 3. Register host callbacks for interactive questions.
    // Do not register pb_engine_set_webview_callback here: C / desktop hosts
    // have no system WebView, so web_search skips DuckDuckGo and uses the API.
    pb_engine_set_host_callbacks(engine, NULL, on_host_tool_request, (void *)engine);

    // 4. Setup signal handler for graceful Ctrl+C cancellation & exit
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_sigint;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);

    // 5. Determine active session ID & query latest session
    SessionInfo latest = get_latest_session(engine);

    if (cli_resume_id && *cli_resume_id) {
        strncpy(g_session_id, cli_resume_id, sizeof(g_session_id) - 1);
        g_session_id[sizeof(g_session_id) - 1] = '\0';
        g_is_resumed_session = true;
    } else {
        generate_uuid_v4(g_session_id, sizeof(g_session_id));
        g_is_resumed_session = false;
    }

    // 6. Display banner
    print_banner(config_path, parsed_cfg, &latest);
    json_free(parsed_cfg);

    // If started with --resume flag, replay previous session history immediately
    if (g_is_resumed_session) {
        replay_session_history(engine, g_session_id);
    }

    // 7. Interactive REPL loop
    char *line_buf = NULL;
    size_t line_cap = 0;

    AgentUiContext ui_ctx = {
        .in_reasoning = false,
        .in_text = false,
        .tool_call_count = 0,
        .engine = engine,
        .session_id = g_session_id
    };

    while (g_running) {
        printf("\x1b[1;32mUser\x1b[0m \x1b[1;34m❯\x1b[0m ");
        fflush(stdout);

        ssize_t nread = getline(&line_buf, &line_cap, stdin);
        if (nread == -1) {
            // EOF (Ctrl+D) or interrupted by signal
            if (!g_running) {
                break;
            }
            if (feof(stdin)) {
                printf("\n\x1b[1;33m[EOF received. Exiting...]\x1b[0m\n");
                break;
            }
            clearerr(stdin);
            continue;
        }

        // Strip trailing carriage returns and newlines
        while (nread > 0 && (line_buf[nread - 1] == '\n' || line_buf[nread - 1] == '\r')) {
            line_buf[--nread] = '\0';
        }

        // Trim leading whitespace for command checks
        char *input = line_buf;
        while (*input == ' ' || *input == '\t') input++;

        if (!*input) {
            continue;
        }

        if (strcmp(input, "exit") == 0 || strcmp(input, "quit") == 0 ||
            strcmp(input, "/exit") == 0 || strcmp(input, "/quit") == 0) {
            printf("\x1b[1;33m[Exiting PhoneBuddy Agent. Goodbye!]\x1b[0m\n");
            break;
        }

        if (strcmp(input, "clear") == 0 || strcmp(input, "/clear") == 0) {
            printf("\x1b[H\x1b[J");
            continue;
        }

        if (strcmp(input, "help") == 0 || strcmp(input, "/help") == 0) {
            printf("\x1b[1mCommands:\x1b[0m\n");
            printf("  /resume [uuid]  - Interactive picker or direct resume of session history\n");
            printf("  /new            - Start a fresh session with a new UUID\n");
            printf("  /sessions       - List all saved sessions\n");
            printf("  clear, /clear   - Clear console screen\n");
            printf("  help, /help     - Show this help message\n");
            printf("  exit, quit      - Exit the agent\n");
            printf("  Ctrl+C          - Cancel running turn or exit\n\n");
            continue;
        }

        if (strncmp(input, "/resume", 7) == 0 || strncmp(input, "resume", 6) == 0) {
            const char *target = input + (input[0] == '/' ? 7 : 6);
            while (*target == ' ' || *target == '\t') target++;
            handle_resume_command(engine, &ui_ctx, target);
            continue;
        }

        if (strcmp(input, "/new") == 0 || strcmp(input, "new") == 0) {
            generate_uuid_v4(g_session_id, sizeof(g_session_id));
            ui_ctx.session_id = g_session_id;
            g_is_resumed_session = false;
            printf("\x1b[1;32m✓ Started new session:\x1b[0m \x1b[1;36m%s\x1b[0m\n\n", g_session_id);
            continue;
        }

        if (strcmp(input, "/sessions") == 0 || strcmp(input, "sessions") == 0 ||
            strcmp(input, "/list") == 0 || strcmp(input, "list") == 0) {
            list_all_sessions(engine);
            continue;
        }

        // Execute chat turn
        g_in_turn = 1;
        g_cancel_requested = 0;
        ui_ctx.in_reasoning = false;
        ui_ctx.in_text = false;
        ui_ctx.tool_call_count = 0;

        char *chat_err = NULL;
        char *chat_outcome_json = pb_engine_chat(
            engine,
            g_session_id,
            input,
            on_agent_event,
            &ui_ctx,
            &chat_err
        );

        g_in_turn = 0;
        g_cancel_requested = 0;

        if (ui_ctx.in_reasoning) {
            printf("\x1b[0m\n");
            ui_ctx.in_reasoning = false;
        }
        if (ui_ctx.in_text) {
            printf("\n");
            ui_ctx.in_text = false;
        }

        if (chat_err) {
            if (strcmp(chat_err, "cancelled") != 0) {
                printf("\x1b[1;31m[Agent Error]: %s\x1b[0m\n", chat_err);
            }
            pb_string_free(chat_err);
        }

        if (chat_outcome_json) {
            pb_string_free(chat_outcome_json);
        }

        printf("\n");
    }

    if (line_buf) {
        free(line_buf);
    }

    // 8. Clean up engine and exit
    g_engine = NULL;
    pb_engine_free(engine);
    return 0;
}
