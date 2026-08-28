#include <jni.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "phone_buddy.h"

static JavaVM *g_vm = NULL;
static jclass g_agent_cls = NULL;
static jmethodID g_on_webview_fetch = NULL;

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    (void)reserved;
    g_vm = vm;
    return JNI_VERSION_1_6;
}

static void webview_fetch_cb(const char *call_id, const char *request_json, void *user_data) {
    (void)user_data;
    if (!g_vm || !g_agent_cls || !g_on_webview_fetch || !call_id || !request_json) {
        return;
    }

    JNIEnv *env = NULL;
    int did_attach = 0;
    jint status = (*g_vm)->GetEnv(g_vm, (void **)&env, JNI_VERSION_1_6);
    if (status == JNI_EDETACHED) {
        if ((*g_vm)->AttachCurrentThread(g_vm, &env, NULL) != 0) {
            return;
        }
        did_attach = 1;
    } else if (status != JNI_OK || env == NULL) {
        return;
    }

    jstring j_call = (*env)->NewStringUTF(env, call_id);
    jstring j_req = (*env)->NewStringUTF(env, request_json);
    if (j_call && j_req) {
        (*env)->CallStaticVoidMethod(env, g_agent_cls, g_on_webview_fetch, j_call, j_req);
        if ((*env)->ExceptionCheck(env)) {
            (*env)->ExceptionClear(env);
        }
    }
    if (j_call) {
        (*env)->DeleteLocalRef(env, j_call);
    }
    if (j_req) {
        (*env)->DeleteLocalRef(env, j_req);
    }
    if (did_attach) {
        (*g_vm)->DetachCurrentThread(g_vm);
    }
}

// Callback context for JNI event forwarding
typedef struct {
    JNIEnv *env;
    jobject listener;
    jmethodID on_event_mid;
} JniCallbackContext;

static void c_event_callback(const char *event_json, void *user_data) {
    if (!user_data || !event_json) return;
    JniCallbackContext *ctx = (JniCallbackContext *)user_data;
    jstring jstr = (*ctx->env)->NewStringUTF(ctx->env, event_json);
    (*ctx->env)->CallVoidMethod(ctx->env, ctx->listener, ctx->on_event_mid, jstr);
    (*ctx->env)->DeleteLocalRef(ctx->env, jstr);
}

JNIEXPORT jlong JNICALL
Java_org_phonebuddy_NativeAgent_nativeNewEngine(JNIEnv *env, jclass clazz, jstring config_json) {
    const char *c_config = (*env)->GetStringUTFChars(env, config_json, NULL);
    char *err_out = NULL;
    PbEngine *engine = pb_engine_new(c_config, &err_out);
    (*env)->ReleaseStringUTFChars(env, config_json, c_config);

    if (err_out != NULL) {
        jclass ex_cls = (*env)->FindClass(env, "java/lang/RuntimeException");
        (*env)->ThrowNew(env, ex_cls, err_out);
        pb_string_free(err_out);
        return 0;
    }
    return (jlong)engine;
}

JNIEXPORT void JNICALL
Java_org_phonebuddy_NativeAgent_nativeFreeEngine(JNIEnv *env, jclass clazz, jlong engine_ptr) {
    if (engine_ptr != 0) {
        pb_engine_free((PbEngine *)engine_ptr);
    }
}

JNIEXPORT jstring JNICALL
Java_org_phonebuddy_NativeAgent_nativeChat(
    JNIEnv *env, jclass clazz, jlong engine_ptr, jstring session_id, jstring user_input, jobject listener
) {
    if (engine_ptr == 0) return NULL;
    const char *c_session = (*env)->GetStringUTFChars(env, session_id, NULL);
    const char *c_input = (*env)->GetStringUTFChars(env, user_input, NULL);
    char *err_out = NULL;

    JniCallbackContext ctx;
    ctx.env = env;
    ctx.listener = listener;
    if (listener != NULL) {
        jclass l_cls = (*env)->GetObjectClass(env, listener);
        ctx.on_event_mid = (*env)->GetMethodID(env, l_cls, "onEvent", "(Ljava/lang/String;)V");
    } else {
        ctx.on_event_mid = NULL;
    }

    char *result_json = pb_engine_chat(
        (PbEngine *)engine_ptr,
        c_session,
        c_input,
        listener ? c_event_callback : NULL,
        listener ? &ctx : NULL,
        &err_out
    );

    (*env)->ReleaseStringUTFChars(env, session_id, c_session);
    (*env)->ReleaseStringUTFChars(env, user_input, c_input);

    if (err_out != NULL) {
        jclass ex_cls = (*env)->FindClass(env, "java/lang/RuntimeException");
        (*env)->ThrowNew(env, ex_cls, err_out);
        pb_string_free(err_out);
        return NULL;
    }

    if (result_json == NULL) return NULL;
    jstring res = (*env)->NewStringUTF(env, result_json);
    pb_string_free(result_json);
    return res;
}

JNIEXPORT jstring JNICALL
Java_org_phonebuddy_NativeAgent_nativeGetSession(
    JNIEnv *env, jclass clazz, jlong engine_ptr, jstring session_id
) {
    if (engine_ptr == 0) return NULL;
    const char *c_session = (*env)->GetStringUTFChars(env, session_id, NULL);
    char *err_out = NULL;
    char *session_json = pb_engine_get_session((PbEngine *)engine_ptr, c_session, &err_out);
    (*env)->ReleaseStringUTFChars(env, session_id, c_session);

    if (err_out != NULL) {
        jclass ex_cls = (*env)->FindClass(env, "java/lang/RuntimeException");
        (*env)->ThrowNew(env, ex_cls, err_out);
        pb_string_free(err_out);
        return NULL;
    }

    if (session_json == NULL) return NULL;
    jstring res = (*env)->NewStringUTF(env, session_json);
    pb_string_free(session_json);
    return res;
}

JNIEXPORT jstring JNICALL
Java_org_phonebuddy_NativeAgent_nativeListSessions(
    JNIEnv *env, jclass clazz, jlong engine_ptr
) {
    if (engine_ptr == 0) return NULL;
    char *err_out = NULL;
    char *sessions_json = pb_engine_list_sessions((PbEngine *)engine_ptr, &err_out);

    if (err_out != NULL) {
        jclass ex_cls = (*env)->FindClass(env, "java/lang/RuntimeException");
        (*env)->ThrowNew(env, ex_cls, err_out);
        pb_string_free(err_out);
        return NULL;
    }

    if (sessions_json == NULL) return NULL;
    jstring res = (*env)->NewStringUTF(env, sessions_json);
    pb_string_free(sessions_json);
    return res;
}

JNIEXPORT jint JNICALL
Java_org_phonebuddy_NativeAgent_nativeDeleteSession(
    JNIEnv *env, jclass clazz, jlong engine_ptr, jstring session_id
) {
    if (engine_ptr == 0) return -1;
    const char *c_session = (*env)->GetStringUTFChars(env, session_id, NULL);
    int res = pb_engine_delete_session((PbEngine *)engine_ptr, c_session);
    (*env)->ReleaseStringUTFChars(env, session_id, c_session);
    return (jint)res;
}

JNIEXPORT void JNICALL
Java_org_phonebuddy_NativeAgent_nativeSetWebViewCallback(
    JNIEnv *env, jclass clazz, jlong engine_ptr
) {
    (void)clazz;
    if (engine_ptr == 0) {
        return;
    }
    if (g_agent_cls == NULL) {
        jclass local = (*env)->FindClass(env, "org/phonebuddy/NativeAgent");
        if (local == NULL) {
            return;
        }
        g_agent_cls = (*env)->NewGlobalRef(env, local);
        (*env)->DeleteLocalRef(env, local);
        if (g_agent_cls == NULL) {
            return;
        }
        g_on_webview_fetch = (*env)->GetStaticMethodID(
            env, g_agent_cls, "onWebViewFetch", "(Ljava/lang/String;Ljava/lang/String;)V"
        );
        if (g_on_webview_fetch == NULL) {
            return;
        }
    }
    pb_engine_set_webview_callback((PbEngine *)engine_ptr, webview_fetch_cb, NULL);
}

JNIEXPORT void JNICALL
Java_org_phonebuddy_NativeAgent_nativeClearWebViewCallback(
    JNIEnv *env, jclass clazz, jlong engine_ptr
) {
    (void)env;
    (void)clazz;
    if (engine_ptr != 0) {
        pb_engine_set_webview_callback((PbEngine *)engine_ptr, NULL, NULL);
    }
}

static jmethodID g_on_host_tool_request = NULL;

static void host_tool_cb(const char *call_id, const char *name, const char *arguments_json, void *user_data) {
    (void)user_data;
    if (!g_vm || !g_agent_cls || !g_on_host_tool_request || !call_id || !name || !arguments_json) {
        return;
    }

    JNIEnv *env = NULL;
    int did_attach = 0;
    jint status = (*g_vm)->GetEnv(g_vm, (void **)&env, JNI_VERSION_1_6);
    if (status == JNI_EDETACHED) {
        if ((*g_vm)->AttachCurrentThread(g_vm, &env, NULL) != 0) {
            return;
        }
        did_attach = 1;
    } else if (status != JNI_OK || env == NULL) {
        return;
    }

    jstring j_call = (*env)->NewStringUTF(env, call_id);
    jstring j_name = (*env)->NewStringUTF(env, name);
    jstring j_args = (*env)->NewStringUTF(env, arguments_json);
    if (j_call && j_name && j_args) {
        (*env)->CallStaticVoidMethod(env, g_agent_cls, g_on_host_tool_request, j_call, j_name, j_args);
        if ((*env)->ExceptionCheck(env)) {
            (*env)->ExceptionClear(env);
        }
    }
    if (j_call) (*env)->DeleteLocalRef(env, j_call);
    if (j_name) (*env)->DeleteLocalRef(env, j_name);
    if (j_args) (*env)->DeleteLocalRef(env, j_args);
    if (did_attach) {
        (*g_vm)->DetachCurrentThread(g_vm);
    }
}

JNIEXPORT void JNICALL
Java_org_phonebuddy_NativeAgent_nativeCancel(
    JNIEnv *env, jclass clazz, jlong engine_ptr, jstring session_id
) {
    (void)clazz;
    if (engine_ptr == 0 || session_id == NULL) return;
    const char *c_session = (*env)->GetStringUTFChars(env, session_id, NULL);
    pb_engine_cancel((PbEngine *)engine_ptr, c_session);
    (*env)->ReleaseStringUTFChars(env, session_id, c_session);
}

JNIEXPORT void JNICALL
Java_org_phonebuddy_NativeAgent_nativeSetHostCallbacks(
    JNIEnv *env, jclass clazz, jlong engine_ptr
) {
    (void)clazz;
    if (engine_ptr == 0) return;
    if (g_agent_cls == NULL) {
        jclass local = (*env)->FindClass(env, "org/phonebuddy/NativeAgent");
        if (local == NULL) return;
        g_agent_cls = (*env)->NewGlobalRef(env, local);
        (*env)->DeleteLocalRef(env, local);
        if (g_agent_cls == NULL) return;
    }
    if (g_on_host_tool_request == NULL) {
        g_on_host_tool_request = (*env)->GetStaticMethodID(
            env, g_agent_cls, "onHostToolRequest", "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V"
        );
        if (g_on_host_tool_request == NULL) return;
    }
    pb_engine_set_host_callbacks((PbEngine *)engine_ptr, NULL, host_tool_cb, NULL);
}

JNIEXPORT jint JNICALL
Java_org_phonebuddy_NativeAgent_nativeHostToolResult(
    JNIEnv *env, jclass clazz, jlong engine_ptr, jstring call_id, jint ok, jstring output
) {
    (void)clazz;
    if (engine_ptr == 0) return -1;
    const char *c_call = (*env)->GetStringUTFChars(env, call_id, NULL);
    const char *c_out = output ? (*env)->GetStringUTFChars(env, output, NULL) : "";
    char *err_out = NULL;
    int res = pb_engine_host_tool_result(
        (PbEngine *)engine_ptr,
        c_call,
        (int32_t)ok,
        c_out,
        &err_out
    );
    (*env)->ReleaseStringUTFChars(env, call_id, c_call);
    if (output) {
        (*env)->ReleaseStringUTFChars(env, output, c_out);
    }
    if (err_out != NULL) {
        pb_string_free(err_out);
        return -2;
    }
    return (jint)res;
}

JNIEXPORT jint JNICALL
Java_org_phonebuddy_NativeAgent_nativeWebViewResult(
    JNIEnv *env, jclass clazz, jlong engine_ptr, jstring call_id, jint ok, jstring output
) {
    (void)clazz;
    if (engine_ptr == 0) {
        return -1;
    }
    const char *c_call = (*env)->GetStringUTFChars(env, call_id, NULL);
    const char *c_out = output ? (*env)->GetStringUTFChars(env, output, NULL) : "";
    char *err_out = NULL;
    int res = pb_engine_webview_result(
        (PbEngine *)engine_ptr,
        c_call,
        (int32_t)ok,
        c_out,
        &err_out
    );
    (*env)->ReleaseStringUTFChars(env, call_id, c_call);
    if (output) {
        (*env)->ReleaseStringUTFChars(env, output, c_out);
    }
    if (err_out != NULL) {
        pb_string_free(err_out);
        return -3;
    }
    return (jint)res;
}

typedef struct {
    JavaVM *vm;
    jobject listener;
    jmethodID on_complete_mid;
} GenerateTextJniContext;

static void generate_text_jni_cb(const char *envelope_json, void *user_data) {
    GenerateTextJniContext *ctx = (GenerateTextJniContext *)user_data;
    if (!ctx) {
        return;
    }
    JNIEnv *env = NULL;
    int did_attach = 0;
    if (ctx->vm) {
        jint status = (*ctx->vm)->GetEnv(ctx->vm, (void **)&env, JNI_VERSION_1_6);
        if (status == JNI_EDETACHED) {
            if ((*ctx->vm)->AttachCurrentThread(ctx->vm, &env, NULL) == 0) {
                did_attach = 1;
            } else {
                env = NULL;
            }
        } else if (status != JNI_OK) {
            env = NULL;
        }
    }
    if (env && ctx->listener && ctx->on_complete_mid && envelope_json) {
        jstring j_env = (*env)->NewStringUTF(env, envelope_json);
        if (j_env) {
            (*env)->CallVoidMethod(env, ctx->listener, ctx->on_complete_mid, j_env);
            if ((*env)->ExceptionCheck(env)) {
                (*env)->ExceptionClear(env);
            }
            (*env)->DeleteLocalRef(env, j_env);
        }
    }
    if (env && ctx->listener) {
        (*env)->DeleteGlobalRef(env, ctx->listener);
        ctx->listener = NULL;
    }
    if (did_attach && ctx->vm) {
        (*ctx->vm)->DetachCurrentThread(ctx->vm);
    }
    free(ctx);
}

JNIEXPORT jlong JNICALL
Java_org_phonebuddy_NativeRuntime_nativeNew(JNIEnv *env, jclass clazz, jstring routing_json, jstring root_dir) {
    (void)clazz;
    const char *c_routing = (*env)->GetStringUTFChars(env, routing_json, NULL);
    const char *c_root = (*env)->GetStringUTFChars(env, root_dir, NULL);
    char *err_out = NULL;
    PbRuntime *runtime = pb_runtime_new(c_routing, c_root, &err_out);
    (*env)->ReleaseStringUTFChars(env, routing_json, c_routing);
    (*env)->ReleaseStringUTFChars(env, root_dir, c_root);
    if (err_out != NULL) {
        jclass ex_cls = (*env)->FindClass(env, "java/lang/RuntimeException");
        (*env)->ThrowNew(env, ex_cls, err_out);
        pb_string_free(err_out);
        return 0;
    }
    return (jlong)runtime;
}

JNIEXPORT void JNICALL
Java_org_phonebuddy_NativeRuntime_nativeFree(JNIEnv *env, jclass clazz, jlong runtime_ptr) {
    (void)env;
    (void)clazz;
    if (runtime_ptr != 0) {
        pb_runtime_free((PbRuntime *)runtime_ptr);
    }
}

JNIEXPORT void JNICALL
Java_org_phonebuddy_NativeRuntime_nativeUpdateRouting(JNIEnv *env, jclass clazz, jlong runtime_ptr, jstring routing_json) {
    (void)clazz;
    if (runtime_ptr == 0) {
        return;
    }
    const char *c_routing = (*env)->GetStringUTFChars(env, routing_json, NULL);
    char *err_out = NULL;
    int rc = pb_runtime_update_routing((PbRuntime *)runtime_ptr, c_routing, &err_out);
    (*env)->ReleaseStringUTFChars(env, routing_json, c_routing);
    if (rc != 0 && err_out != NULL) {
        jclass ex_cls = (*env)->FindClass(env, "java/lang/RuntimeException");
        (*env)->ThrowNew(env, ex_cls, err_out);
        pb_string_free(err_out);
    } else if (err_out != NULL) {
        pb_string_free(err_out);
    }
}

JNIEXPORT jlong JNICALL
Java_org_phonebuddy_NativeRuntime_nativeCreateEngine(
    JNIEnv *env, jclass clazz, jlong runtime_ptr, jstring config_json, jstring main_pool_id
) {
    (void)clazz;
    if (runtime_ptr == 0) {
        return 0;
    }
    const char *c_config = (*env)->GetStringUTFChars(env, config_json, NULL);
    const char *c_pool = main_pool_id ? (*env)->GetStringUTFChars(env, main_pool_id, NULL) : NULL;
    char *err_out = NULL;
    PbEngine *engine = pb_engine_new_with_runtime(
        (PbRuntime *)runtime_ptr, c_config, c_pool, &err_out
    );
    (*env)->ReleaseStringUTFChars(env, config_json, c_config);
    if (main_pool_id) {
        (*env)->ReleaseStringUTFChars(env, main_pool_id, c_pool);
    }
    if (err_out != NULL) {
        jclass ex_cls = (*env)->FindClass(env, "java/lang/RuntimeException");
        (*env)->ThrowNew(env, ex_cls, err_out);
        pb_string_free(err_out);
        return 0;
    }
    return (jlong)engine;
}

JNIEXPORT jstring JNICALL
Java_org_phonebuddy_NativeRuntime_nativeGenerateTextAsync(
    JNIEnv *env, jclass clazz, jlong runtime_ptr, jstring request_json, jobject listener
) {
    (void)clazz;
    if (runtime_ptr == 0) {
        return NULL;
    }
    const char *c_req = (*env)->GetStringUTFChars(env, request_json, NULL);
    char *err_out = NULL;
    GenerateTextJniContext *ctx = NULL;
    PbOperationCallback cb = NULL;
    if (listener != NULL) {
        ctx = (GenerateTextJniContext *)calloc(1, sizeof(GenerateTextJniContext));
        if (ctx == NULL) {
            (*env)->ReleaseStringUTFChars(env, request_json, c_req);
            return NULL;
        }
        (*env)->GetJavaVM(env, &ctx->vm);
        ctx->listener = (*env)->NewGlobalRef(env, listener);
        jclass l_cls = (*env)->GetObjectClass(env, listener);
        ctx->on_complete_mid = (*env)->GetMethodID(env, l_cls, "onComplete", "(Ljava/lang/String;)V");
        cb = generate_text_jni_cb;
    }
    char *op = pb_runtime_generate_text_async(
        (PbRuntime *)runtime_ptr, c_req, cb, ctx, &err_out
    );
    (*env)->ReleaseStringUTFChars(env, request_json, c_req);
    if (err_out != NULL) {
        if (ctx) {
            if (ctx->listener) {
                (*env)->DeleteGlobalRef(env, ctx->listener);
            }
            free(ctx);
        }
        jclass ex_cls = (*env)->FindClass(env, "java/lang/RuntimeException");
        (*env)->ThrowNew(env, ex_cls, err_out);
        pb_string_free(err_out);
        return NULL;
    }
    if (op == NULL) {
        if (ctx) {
            if (ctx->listener) {
                (*env)->DeleteGlobalRef(env, ctx->listener);
            }
            free(ctx);
        }
        return NULL;
    }
    jstring res = (*env)->NewStringUTF(env, op);
    pb_string_free(op);
    return res;
}

JNIEXPORT void JNICALL
Java_org_phonebuddy_NativeRuntime_nativeCancelOperation(
    JNIEnv *env, jclass clazz, jlong runtime_ptr, jstring operation_id
) {
    (void)clazz;
    if (runtime_ptr == 0 || operation_id == NULL) {
        return;
    }
    const char *c_op = (*env)->GetStringUTFChars(env, operation_id, NULL);
    pb_runtime_cancel_operation((PbRuntime *)runtime_ptr, c_op);
    (*env)->ReleaseStringUTFChars(env, operation_id, c_op);
}

void phone_buddy_jni_link_anchor(void) {
}

