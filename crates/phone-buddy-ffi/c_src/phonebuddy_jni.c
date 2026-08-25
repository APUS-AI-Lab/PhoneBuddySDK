#include <jni.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "phone_buddy.h"

#ifdef __ANDROID__
#include <android/log.h>
#define PB_LOG_TAG "PhoneBuddyJNI"
#define PB_LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, PB_LOG_TAG, __VA_ARGS__)
#define PB_LOGI(...) __android_log_print(ANDROID_LOG_INFO, PB_LOG_TAG, __VA_ARGS__)
#define PB_LOGW(...) __android_log_print(ANDROID_LOG_WARN, PB_LOG_TAG, __VA_ARGS__)
#define PB_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, PB_LOG_TAG, __VA_ARGS__)
#else
#define PB_LOGD(...) ((void)0)
#define PB_LOGI(...) ((void)0)
#define PB_LOGW(...) ((void)0)
#define PB_LOGE(...) ((void)0)
#endif

static JavaVM *g_vm = NULL;
static jclass g_agent_cls = NULL;
static jmethodID g_on_webview_fetch = NULL;

jint pb_jni_on_load(JavaVM *vm, void *reserved) {
    (void)reserved;
    g_vm = vm;
    PB_LOGI("pb_jni_on_load initialized JavaVM");
    return JNI_VERSION_1_6;
}

static void webview_fetch_cb(const char *call_id, const char *request_json, void *user_data) {
    (void)user_data;
    PB_LOGD("[JNI] webview_fetch_cb invoked for call_id=%s", call_id ? call_id : "null");
    if (!g_vm || !g_agent_cls || !g_on_webview_fetch || !call_id || !request_json) {
        PB_LOGW("[JNI] webview_fetch_cb skipped (missing vm=%p, cls=%p, mid=%p)",
                g_vm, g_agent_cls, g_on_webview_fetch);
        return;
    }

    JNIEnv *env = NULL;
    int did_attach = 0;
    jint status = (*g_vm)->GetEnv(g_vm, (void **)&env, JNI_VERSION_1_6);
    if (status == JNI_EDETACHED) {
        if ((*g_vm)->AttachCurrentThread(g_vm, &env, NULL) != 0) {
            PB_LOGE("[JNI] webview_fetch_cb failed to attach current thread");
            return;
        }
        did_attach = 1;
    } else if (status != JNI_OK || env == NULL) {
        PB_LOGE("[JNI] webview_fetch_cb failed to get JNIEnv (status=%d)", status);
        return;
    }

    jstring j_call = (*env)->NewStringUTF(env, call_id);
    jstring j_req = (*env)->NewStringUTF(env, request_json);
    if (j_call && j_req) {
        (*env)->CallStaticVoidMethod(env, g_agent_cls, g_on_webview_fetch, j_call, j_req);
        if ((*env)->ExceptionCheck(env)) {
            (*env)->ExceptionDescribe(env);
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

jlong pb_jni_nativeNewEngine(JNIEnv *env, jclass clazz, jstring config_json) {
    (void)clazz;
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

void pb_jni_nativeFreeEngine(JNIEnv *env, jclass clazz, jlong engine_ptr) {
    (void)env;
    (void)clazz;
    if (engine_ptr != 0) {
        pb_engine_free((PbEngine *)engine_ptr);
    }
}

jstring pb_jni_nativeChat(
    JNIEnv *env, jclass clazz, jlong engine_ptr, jstring session_id, jstring user_input, jobject listener
) {
    (void)clazz;
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

jstring pb_jni_nativeChatV2(
    JNIEnv *env, jclass clazz, jlong engine_ptr, jstring session_id, jstring turn_json, jobject listener
) {
    (void)clazz;
    if (engine_ptr == 0) return NULL;
    const char *c_session = (*env)->GetStringUTFChars(env, session_id, NULL);
    const char *c_turn = (*env)->GetStringUTFChars(env, turn_json, NULL);
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

    char *result_json = pb_engine_chat_v2(
        (PbEngine *)engine_ptr,
        c_session,
        c_turn,
        listener ? c_event_callback : NULL,
        listener ? &ctx : NULL,
        &err_out
    );

    (*env)->ReleaseStringUTFChars(env, session_id, c_session);
    (*env)->ReleaseStringUTFChars(env, turn_json, c_turn);

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

jstring pb_jni_nativeGetSession(
    JNIEnv *env, jclass clazz, jlong engine_ptr, jstring session_id
) {
    (void)clazz;
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

jstring pb_jni_nativeListSessions(
    JNIEnv *env, jclass clazz, jlong engine_ptr
) {
    (void)clazz;
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

jint pb_jni_nativeDeleteSession(
    JNIEnv *env, jclass clazz, jlong engine_ptr, jstring session_id
) {
    (void)clazz;
    if (engine_ptr == 0) return -1;
    const char *c_session = (*env)->GetStringUTFChars(env, session_id, NULL);
    int res = pb_engine_delete_session((PbEngine *)engine_ptr, c_session);
    (*env)->ReleaseStringUTFChars(env, session_id, c_session);
    return (jint)res;
}

void pb_jni_nativeSetWebViewCallback(
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

void pb_jni_nativeClearWebViewCallback(
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

void pb_jni_nativeCancel(
    JNIEnv *env, jclass clazz, jlong engine_ptr, jstring session_id
) {
    (void)clazz;
    if (engine_ptr == 0 || session_id == NULL) return;
    const char *c_session = (*env)->GetStringUTFChars(env, session_id, NULL);
    pb_engine_cancel((PbEngine *)engine_ptr, c_session);
    (*env)->ReleaseStringUTFChars(env, session_id, c_session);
}

void pb_jni_nativeSetHostCallbacks(
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

jint pb_jni_nativeHostToolResult(
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

jint pb_jni_nativeWebViewResult(
    JNIEnv *env, jclass clazz, jlong engine_ptr, jstring call_id, jint ok, jstring output
) {
    (void)clazz;
    if (engine_ptr == 0) {
        return -1;
    }
    const char *c_call = (*env)->GetStringUTFChars(env, call_id, NULL);
    const char *c_out = output ? (*env)->GetStringUTFChars(env, output, NULL) : "";
    PB_LOGD("[JNI] nativeWebViewResult: call_id=%s, ok=%d, output_bytes=%zu",
            c_call ? c_call : "null", ok, strlen(c_out));
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

void pb_jni_nativeSetAgentName(
    JNIEnv *env, jclass clazz, jlong engine_ptr, jstring name
) {
    (void)clazz;
    if (engine_ptr == 0) {
        return;
    }
    if (name == NULL) {
        pb_engine_set_agent_name((PbEngine *)engine_ptr, NULL);
        return;
    }
    const char *c_name = (*env)->GetStringUTFChars(env, name, NULL);
    pb_engine_set_agent_name((PbEngine *)engine_ptr, c_name);
    (*env)->ReleaseStringUTFChars(env, name, c_name);
}

void pb_jni_nativeSetSystemPromptExtra(
    JNIEnv *env, jclass clazz, jlong engine_ptr, jstring extra
) {
    (void)clazz;
    if (engine_ptr == 0) {
        return;
    }
    if (extra == NULL) {
        pb_engine_set_system_prompt_extra((PbEngine *)engine_ptr, NULL);
        return;
    }
    const char *c_extra = (*env)->GetStringUTFChars(env, extra, NULL);
    pb_engine_set_system_prompt_extra((PbEngine *)engine_ptr, c_extra);
    (*env)->ReleaseStringUTFChars(env, extra, c_extra);
}
