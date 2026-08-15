// server.js — Robust Hybrid OpenAI ↔ NIM Proxy
// Express 5 Compatible
// Fixes: auth bypass, startup DDoS, silent stream failures, memory leaks, Express 5 deprecations
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { StringDecoder } = require('string_decoder');
const { timingSafeEqual } = require('crypto');
const app = express();
const PORT = process.env.PORT || 3000;

// ─── Configuration ───────────────────────────────────────────────────────────
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const CLIENT_AUTH_KEY = process.env.CLIENT_AUTH_KEY;
const SHOW_REASONING = process.env.SHOW_REASONING === 'true';
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';
const MAX_TOKENS_LIMIT = 8192;
const REQUEST_TIMEOUT_MS = 300000;
const MAX_BUFFER_SIZE = 6 * 1024 * 1024; // 6MB

// ─── Upstream concurrency control ────────────────────────────────────────────
// Default: only 1 active NIM inference at a time.
// Additional requests wait locally instead of hitting NIM simultaneously.
const MAX_CONCURRENT_NIM_REQUESTS = Math.max(1, Number(process.env.MAX_CONCURRENT_NIM_REQUESTS || 1));
let activeNimRequests = 0;
const nimRequestQueue = [];

function acquireNimSlot(req) {
    if (activeNimRequests < MAX_CONCURRENT_NIM_REQUESTS) {
        activeNimRequests++;
        return Promise.resolve(true);
    }
    return new Promise(resolve => {
        const entry = {
            resolve,
            req,
            cancel: null
        };
        // If the client disconnects while waiting in the queue,
        // cancel the queued request instead of sending it later.
        entry.cancel = () => {
            const index = nimRequestQueue.indexOf(entry);
            if (index !== -1) {
                nimRequestQueue.splice(index, 1);
                resolve(false);
            }
        };
        req.once('close', entry.cancel);
        nimRequestQueue.push(entry);
    });
}

function releaseNimSlot() {
    const next = nimRequestQueue.shift();
    if (next) {
        next.req.removeListener('close', next.cancel);
        // Transfer the existing slot directly to the next request.
        next.resolve(true);
    }
    else {
        activeNimRequests = Math.max(0, activeNimRequests - 1);
    }
}
if (SHOW_REASONING)
    console.log('[CONFIG] Reasoning display: ENABLED');
if (ENABLE_THINKING_MODE)
    console.log('[CONFIG] Thinking mode: ENABLED');

// ─── Config validation ──────────────────────────────────────────────────────

function validateConfig() {
    const fatal = (msg) => { console.error(`[FATAL] ${msg}`); process.exit(1); };
    if (!NIM_API_KEY)
        fatal('NIM_API_KEY is required. Get one at https://build.nvidia.com/');
    if (!CLIENT_AUTH_KEY) {
        console.warn('[WARN] CLIENT_AUTH_KEY not set. All requests will be rejected with 403.');
    }
}

validateConfig();

// ─── Model Mapping ─────────────────────────────────────────────────────────
const MODEL_MAPPING = {
    'gpt-3.5-turbo': 'nvidia/nemotron-3-super-120b-a12b',
    'gpt-4': 'nvidia/nemotron-3-ultra-550b-a55b',
    'gpt-3.5': 'qwen/qwen3.5-397b-a17b',
    'gpt-4-turbo': 'moonshotai/kimi-k2.6',
    'gpt-4o': 'deepseek-ai/deepseek-v4-pro',
    'claude-3-opus': 'openai/gpt-oss-120b',
    'claude-3-sonnet': 'openai/gpt-oss-20b',
    'gemini-pro': 'nvidia/llama-3.3-nemotron-super-49b-v1.5',
    'gemini-turbo': 'meta/llama-3.3-70b-instruct',
    'gemini-turbo?': 'abacusai/dracarys-llama-3.1-70b-instruct',
    'gpt-3.5o': 'nvidia/nemotron-mini-4b-instruct',
    'gpt-4-flash': 'deepseek-ai/deepseek-v4-flash',
    'glm-5.2': 'z-ai/glm-5.2',
    'mistral': 'mistralai/mistral-large-3-675b-instruct-2512',
    'mistral-turbo': 'mistralai/mistral-medium-3.5-128b',
    'mistral-pro': 'mistralai/mistral-small-4-119b-2603',
    'mistral-nemo': 'mistralai/mistral-nemotron',
    'mistral-fast': 'mistralai/ministral-14b-instruct-2512',
    'google-light': 'google/gemma-4-31b-it',
    'google-lightest': 'google/gemma-2-2b-it',
    'google-lighter': 'google/gemma-3n-e4b-it',
    'm2.7': 'minimaxai/minimax-m2.7',
    'm3': 'minimaxai/minimax-m3',
    'step-3.5-flash': 'stepfun-ai/step-3.5-flash',
    'step-3.7-flash': 'stepfun-ai/step-3.7-flash'
};

// ─── Middleware ─────────────────────────────────────────────────────────────

app.use(cors());

app.use(express.json({ limit: '50mb' }));
// FIX: Extract token AFTER "Bearer " prefix, compare only the token
// Prevents bypass when CLIENT_AUTH_KEY is empty (expected would be "Bearer " which is 7 chars)
function extractBearerToken(authHeader) {
    if (!authHeader || typeof authHeader !== 'string')
        return null;
    const parts = authHeader.trim().split(' ');
    if (parts.length !== 2 || parts[0] !== 'Bearer')
        return null;
    return parts[1];
}

function safeTimingEqual(a, b) {
    if (!a || !b || a.length !== b.length)
        return false;
    try {
        return timingSafeEqual(Buffer.from(a), Buffer.from(b));
    }
    catch {
        return false;
    }
}

app.use((req, res, next) => {
    if (req.path === '/health' || req.path === '/v1/models') {
        return next();
    }
    const token = extractBearerToken(req.headers.authorization);
    if (!token || !CLIENT_AUTH_KEY) {
        return res.status(403).json({
            error: {
                message: 'Forbidden: Invalid or missing authentication',
                type: 'authentication_error',
                code: 403
            }
        });
    }
    if (!safeTimingEqual(token, CLIENT_AUTH_KEY)) {
        return res.status(403).json({
            error: {
                message: 'Forbidden: Invalid authentication credentials',
                type: 'authentication_error',
                code: 403
            }
        });
    }
    next();
});

// ─── Helper: Safe Stream Writing ───────────────────────────────────────────
// FIX: Wrap res.write in try/catch to prevent crashes on closed sockets
function safeWrite(res, data) {
    try {
        if (!res.writableEnded && !res.destroyed && res.writable) {
            res.write(data);
            return true;
        }
    }
    catch (err) {
        console.warn('[STREAM] Write failed:', err.message);
    }
    return false;
}

// ─── Routes ────────────────────────────────────────────────────────────────

app.get('/health', (req, res) => {
    res.json({ status: 'ok', version: '2.1.0' });
});

app.get('/v1/models', (req, res) => {
    res.json({
        object: 'list',
        data: Object.keys(MODEL_MAPPING).map(id => ({
            id,
            object: 'model',
            created: Date.now(),
            owned_by: 'nim-proxy'
        }))
    });
});

app.post('/v1/chat/completions', async (req, res) => {
    let streamEndedCleanly = false;
    let upstreamStream = null;
    let nimSlotReleased = true;
    let releaseNimRequestSlot = () => { };
    // FIX: declared here (not with `const` inside the try block) so it's still
    // readable from the catch block below for error logging.
    let primaryModel;
    try {
        const { model, messages, top_p, temperature, max_tokens, stream } = req.body;
        primaryModel = MODEL_MAPPING[model];
        if (!primaryModel) {
            return res.status(400).json({
                error: {
                    message: `Unsupported model: ${model}`,
                    type: "invalid_request_error",
                    code: 400
                }
            });
        }
        if (!Array.isArray(messages) || messages.length === 0) {
            return res.status(400).json({
                error: {
                    message: '"messages" must be a non-empty array.',
                    type: 'invalid_request_error',
                    code: 400
                }
            });
        }
        const baseRequest = {
            messages,
            temperature: temperature ?? 1,
            top_p: top_p ?? 1,
            max_tokens: Math.min(max_tokens ?? 4096, MAX_TOKENS_LIMIT),
            stream: stream || false,
            extra_body: ENABLE_THINKING_MODE
                ? { chat_template_kwargs: { thinking: true } }
                : undefined
        };
        // ─── Limit simultaneous NIM requests ────────────────────────────────────────
        const acquiredNimSlot = await acquireNimSlot(req);
        if (!acquiredNimSlot) {
            // Client disconnected while waiting in the local queue.
            return;
        }
        nimSlotReleased = false;
        releaseNimRequestSlot = () => {
            if (nimSlotReleased)
                return;
            nimSlotReleased = true;
            releaseNimSlot();
        };
        let response;
        try {
            response = await axios.post(`${NIM_API_BASE}/chat/completions`, {
                ...baseRequest,
                model: primaryModel
            }, {
                headers: {
                    Authorization: `Bearer ${NIM_API_KEY}`,
                    'Content-Type': 'application/json'
                },
                responseType: stream ? 'stream' : 'json',
                timeout: REQUEST_TIMEOUT_MS
            });
        }
        catch (upstreamError) {
            releaseNimRequestSlot();
            throw upstreamError;
        }
        upstreamStream = response.data;
        console.log(`[PROXY] ${primaryModel} | stream=${Boolean(stream)} | ` +
            `active=${activeNimRequests}/${MAX_CONCURRENT_NIM_REQUESTS} | ` +
            `queued=${nimRequestQueue.length}`);
        if (stream) {
            res.setHeader('Content-Type', 'text/event-stream');
            res.setHeader('Cache-Control', 'no-cache');
            res.setHeader('Connection', 'keep-alive');
            const decoder = new StringDecoder('utf8');
            let buffer = '';
            let reasoningOpen = false;
            let doneSent = false;
            let cleanedUp = false;
            const cleanup = () => {
                if (cleanedUp)
                    return;
                cleanedUp = true;
                if (upstreamStream) {
                    upstreamStream.removeAllListeners();
                }
                req.removeAllListeners('close');
            };
            const processLine = (line) => {
                if (!line.startsWith('data: '))
                    return;
                if (line.includes('[DONE]')) {
                    if (!doneSent) {
                        safeWrite(res, 'data: [DONE]\n\n');
                        doneSent = true;
                    }
                    streamEndedCleanly = true;
                    return;
                }
                try {
                    const data = JSON.parse(line.slice(6));
                    const delta = data.choices?.[0]?.delta;
                    if (delta) {
                        let content = delta.content || '';
                        const reasoning = delta.reasoning_content;
                        if (SHOW_REASONING) {
                            if (reasoning) {
                                // Reasoning present this chunk: open the tag if needed, then
                                // use ONLY the reasoning text (never delta.content) so we
                                // don't duplicate anything.
                                content = reasoningOpen ? reasoning : `<think>\n${reasoning}`;
                                reasoningOpen = true;
                                // Boundary chunk: this same delta also carries real content,
                                // so close the tag and append it exactly once.
                                if (delta.content) {
                                    content += `\n</think>\n\n${delta.content}`;
                                    reasoningOpen = false;
                                }
                            }
                            else if (delta.content && reasoningOpen) {
                                // Reasoning finished in an earlier chunk; this chunk is pure
                                // content. Close the tag and use delta.content exactly once
                                // (it must NOT also be sitting in `content` from the top).
                                content = `\n</think>\n\n${delta.content}`;
                                reasoningOpen = false;
                            }
                            // else: ordinary content chunk, nothing to do — content is
                            // already delta.content from initialization above.
                        }
                        delta.content = content;
                        delete delta.reasoning_content;
                    }
                    safeWrite(res, `data: ${JSON.stringify(data)}\n\n`);
                }
                catch (parseErr) {
                    // FIX: Don't silently swallow—send error to client so they know data was lost
                    console.warn('[STREAM] Invalid upstream chunk');
                    safeWrite(res, `data: ${JSON.stringify({
                        error: {
                            message: 'Upstream sent malformed chunk',
                            type: 'stream_parse_error',
                            details: 'Malformed upstream chunk'
                        }
                    })}\n\n`);
                }
            };
            upstreamStream.on('data', chunk => {
                buffer += decoder.write(chunk);
                if (buffer.length > MAX_BUFFER_SIZE) {
                    console.error('[STREAM] Buffer overflow, destroying connection');
                    safeWrite(res, `data: ${JSON.stringify({
                        error: {
                            message: 'Stream buffer overflow',
                            type: 'stream_error'
                        }
                    })}\n\n`);
                    safeWrite(res, 'data: [DONE]\n\n');
                    res.end();
                    upstreamStream.destroy();
                    releaseNimRequestSlot()
                    cleanup();
                    return;
                }
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';
                for (const line of lines) {
                    processLine(line);
                }
            });
            upstreamStream.on('end', () => {
                buffer += decoder.end();
                if (buffer.trim()) {
                    for (const line of buffer.split('\n')) {
                        processLine(line);
                    }
                }
                if (!doneSent) {
                    safeWrite(res, 'data: [DONE]\n\n');
                }
                streamEndedCleanly = true;
                if (!res.writableEnded) {
                    res.end();
                }
                releaseNimRequestSlot();
                cleanup();
            });
            upstreamStream.on('error', err => {
                console.error('[STREAM] Upstream error:', err.message);
                if (!res.writableEnded) {
                    safeWrite(res, `data: ${JSON.stringify({
                        error: {
                            message: 'Stream interrupted by upstream error',
                            type: 'stream_error'
                        }
                    })}\n\n`);
                    safeWrite(res, 'data: [DONE]\n\n');
                    res.end();
                }
                releaseNimRequestSlot();
                cleanup();
            });
            // FIX: Check req.destroyed (Node/Express 5) 
            // Don't destroy already-finished streams
            req.on('close', () => {
                const clientGone = req.destroyed || !res.writable;
                if (!streamEndedCleanly && clientGone) {
                    console.warn('[STREAM] Client disconnected prematurely');
                }
                if (upstreamStream &&
                    !upstreamStream.destroyed &&
                    !streamEndedCleanly) {
                    upstreamStream.destroy();
                }
                releaseNimRequestSlot();
                cleanup();
            });
        }
        else {
            // Non-streaming response
            const openaiResponse = {
                id: `chatcmpl-${Date.now()}`,
                object: 'chat.completion',
                created: Math.floor(Date.now() / 1000),
                model: model,
                choices: (response.data.choices || []).map((choice, i) => {
                    let content = choice.message?.content || '';
                    if (SHOW_REASONING && choice.message?.reasoning_content) {
                        content = `<think>\n${choice.message.reasoning_content}\n</think>\n\n${content}`;
                    }
                    return {
                        index: i,
                        message: {
                            role: choice.message?.role || 'assistant',
                            content,
                            tool_calls: choice.message?.tool_calls
                        },
                        finish_reason: choice.finish_reason || 'stop'
                    };
                }),
                usage: response.data.usage || {
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    total_tokens: 0
                }
            };
            res.json(openaiResponse);
            releaseNimRequestSlot();
        }
    }
    catch (error) {
        releaseNimRequestSlot();
        // Keep error logs compact.
        // Never dump Axios response/request objects because they can contain
        // huge socket internals and sensitive Authorization headers.
        const status = error.response?.status;
        const retryAfter = error.response?.headers?.['retry-after'];
        if (status === 429) {
            console.warn(`[NIM 429] model=${primaryModel || 'unknown'} ` +
                `retry-after=${retryAfter || 'unknown'} ` +
                `message=${error.message}`);
        }
        else {
            console.error(`[PROXY ERROR] model=${primaryModel || 'unknown'} ` +
                `status=${status || 'N/A'} ` +
                `code=${error.code || 'N/A'} ` +
                `message=${error.message}`);
        }
        if (!res.headersSent) {
            res.status(error.response?.status || 500).json({
                error: {
                    message: error.message,
                    type: 'invalid_request_error',
                    code: error.response?.status || 500
                }
            });
        }
        else if (!res.writableEnded) {
            safeWrite(res, `data: ${JSON.stringify({
                error: {
                    message: error.message,
                    type: 'proxy_error'
                }
            })}\n\n`);
            safeWrite(res, 'data: [DONE]\n\n');
            res.end();
        }
        // Clean up upstream stream if we have it
        if (upstreamStream && !upstreamStream.destroyed) {
            upstreamStream.destroy();
        }
    }
});
// FIX: Express 5 named wildcard — but use proper 404 handler
app.use((req, res) => {
    res.status(404).json({
        error: {
            message: `Endpoint ${req.method} ${req.path} not found`,
            type: 'invalid_request_error',
            code: 404
        }
    });
});

// ─── Startup ───────────────────────────────────────────────────────────────

app.listen(PORT, () => {
    console.log(`[PROXY] Hybrid proxy running on port ${PORT}`);
    console.log(`[PROXY] Max tokens limit: ${MAX_TOKENS_LIMIT}`);
    console.log(`[PROXY] Max concurrent NIM requests: ${MAX_CONCURRENT_NIM_REQUESTS}`);
});
