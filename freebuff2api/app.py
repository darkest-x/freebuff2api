from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
from typing import Any, AsyncIterator
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .codebuff import (
    CodebuffClient,
    CodebuffError,
    FreebuffRun,
    SessionManager,
    utc_now_iso,
)
from .config import Settings, load_settings
from .logging_config import configure_logging, redact_headers, render_debug
from .openai_compat import (
    CompletionAccumulator,
    build_upstream_payload,
    sanitize_stream_chunk,
)
from .models import CONTEXT_PRUNER_AGENT_ID, models_response, resolve_model
from .sse import decode_sse_data, encode_sse


logger = logging.getLogger("freebuff2api.app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    configure_logging(settings)
    client = CodebuffClient(settings)
    app.state.settings = settings
    app.state.codebuff = client
    app.state.sessions = SessionManager(client, settings)
    try:
        yield
    finally:
        await client.aclose()


app = FastAPI(title="freebuff2api", version="0.1.0", lifespan=lifespan)


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _client(request: Request) -> CodebuffClient:
    return request.app.state.codebuff


def _sessions(request: Request) -> SessionManager:
    return request.app.state.sessions


def _check_local_auth(request: Request) -> None:
    api_key = _settings(request).local_api_key
    if not api_key:
        return
    expected = f"Bearer {api_key}"
    if request.headers.get("authorization") != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _error_response(error: Exception) -> JSONResponse:
    if isinstance(error, CodebuffError):
        return JSONResponse(
            status_code=error.status_code,
            content={
                "error": {
                    "message": str(error),
                    "type": "upstream_error",
                    "code": "codebuff_error",
                }
            },
        )
    raise error


@app.get("/healthz")
async def healthz(request: Request) -> dict[str, Any]:
    _check_local_auth(request)
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models(request: Request) -> dict[str, Any]:
    _check_local_auth(request)
    return models_response()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    _check_local_auth(request)
    body = await request.json()
    settings = _settings(request)
    try:
        model_config = resolve_model(body.get("model"))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    model = model_config.id
    logger.info(
        "chat completion request model=%s stream=%s messages=%s",
        model,
        body.get("stream") is True,
        len(body.get("messages") or []),
    )
    if settings.debug:
        logger.debug(
            "incoming request headers=%s",
            redact_headers(dict(request.headers)),
        )
        logger.debug(
            "chat completion request body=%s",
            render_debug(body, settings.log_body_chars),
        )

    try:
        session = await _sessions(request).ensure_session(
            model,
            messages=body.get("messages"),
        )
        await _client(request).validate_agents()
        await _client(request).request_ad_chain(messages=body.get("messages"))
        run = await _start_freebuff_run_chain(_client(request), model_config.agent_id)
        trace_session_id = str(uuid.uuid4())
        payload = build_upstream_payload(
            body,
            session=session,
            run_id=run.run_id,
            client_id=settings.client_id,
            trace_session_id=trace_session_id,
        )
        if settings.debug:
            logger.debug(
                "prepared upstream chat trace=%s run=%s payload=%s",
                trace_session_id,
                run,
                render_debug(payload, settings.log_body_chars),
            )
    except Exception as error:
        logger.exception("failed to prepare chat completion")
        return _error_response(error)

    if body.get("stream") is True:
        return StreamingResponse(
            _stream_openai_chunks(request, payload, run),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        response = await _collect_completion(request, payload, run, model)
        return JSONResponse(response)
    except Exception as error:
        return _error_response(error)


async def _stream_openai_chunks(
    request: Request,
    payload: dict[str, Any],
    run: FreebuffRun,
) -> AsyncIterator[bytes]:
    message_id: str | None = None
    client = _client(request)
    settings = _settings(request)
    try:
        async for line in client.chat_events(payload):
            data = decode_sse_data(line)
            if data is None:
                continue
            if data == "[DONE]":
                if settings.debug:
                    logger.debug(
                        "chat stream done run_id=%s message_id=%s",
                        run.run_id,
                        message_id,
                    )
                yield encode_sse("[DONE]")
                break

            message_id = data.get("id") or message_id
            chunk = sanitize_stream_chunk(data)
            if chunk is not None:
                if settings.debug:
                    logger.debug(
                        "chat stream chunk=%s",
                        render_debug(chunk, settings.log_body_chars),
                    )
                yield encode_sse(chunk)
            elif settings.debug:
                logger.debug(
                    "chat stream ignored data=%s",
                    render_debug(data, settings.log_body_chars),
                )
    finally:
        _schedule_finalize_run(client, run, message_id)


async def _collect_completion(
    request: Request,
    payload: dict[str, Any],
    run: FreebuffRun,
    model: str,
) -> dict[str, Any]:
    message_id: str | None = None
    accumulator = CompletionAccumulator(model)
    try:
        async for line in _client(request).chat_events(payload):
            data = decode_sse_data(line)
            if data is None:
                continue
            if data == "[DONE]":
                break
            message_id = data.get("id") or message_id
            accumulator.add(data)
        response = accumulator.final_response()
        logger.info(
            "chat completion response run_id=%s message_id=%s content_chars=%s finish_reason=%s",
            run.run_id,
            message_id,
            len(response["choices"][0]["message"].get("content") or ""),
            response["choices"][0].get("finish_reason"),
        )
        if _settings(request).debug:
            logger.debug(
                "chat completion response body=%s",
                render_debug(response, _settings(request).log_body_chars),
            )
        return response
    finally:
        await _finalize_run(request, run, message_id)


async def _start_freebuff_run_chain(
    client: CodebuffClient,
    agent_id: str,
) -> FreebuffRun:
    started_at = utc_now_iso()
    run_id = await client.start_run(agent_id)
    child_started_at = utc_now_iso()
    child_run_id = await client.start_run(
        CONTEXT_PRUNER_AGENT_ID,
        ancestor_run_ids=[run_id],
    )
    await client.record_run_step(
        child_run_id,
        step_number=1,
        child_run_ids=[],
        message_id=None,
        start_time=child_started_at,
    )
    await client.finish_run(child_run_id, total_steps=2)
    await client.record_run_step(
        run_id,
        step_number=1,
        child_run_ids=[child_run_id],
        message_id=None,
        start_time=started_at,
    )
    return FreebuffRun(
        run_id=run_id,
        agent_id=agent_id,
        started_at=started_at,
        child_run_id=child_run_id,
    )


async def _finalize_run(
    request: Request,
    run: FreebuffRun,
    message_id: str | None,
) -> None:
    await _finalize_run_with_client(_client(request), run, message_id)


def _schedule_finalize_run(
    client: CodebuffClient,
    run: FreebuffRun,
    message_id: str | None,
) -> None:
    task = asyncio.create_task(_finalize_run_with_client(client, run, message_id))

    def _log_background_error(done: asyncio.Task[None]) -> None:
        try:
            done.result()
        except asyncio.CancelledError:
            logger.debug("background finalize task cancelled run_id=%s", run.run_id)
        except Exception:
            logger.exception("background finalize task failed run_id=%s", run.run_id)

    task.add_done_callback(_log_background_error)


async def _finalize_run_with_client(
    client: CodebuffClient,
    run: FreebuffRun,
    message_id: str | None,
) -> None:
    try:
        logger.debug(
            "finalize run start run_id=%s message_id=%s started_at=%s",
            run.run_id,
            message_id,
            run.started_at,
        )
        await client.record_run_step(
            run.run_id,
            step_number=2,
            child_run_ids=[],
            message_id=message_id,
            start_time=run.started_at,
        )
        await client.finish_run(run.run_id, total_steps=3)
        logger.debug("finalize run done run_id=%s", run.run_id)
    except Exception:
        logger.exception("finalize run failed run_id=%s", run.run_id)
