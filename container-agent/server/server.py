import os
import uuid
import asyncio
import json
import pty
import fcntl
import termios
import struct
import signal
import shutil
from typing import AsyncIterator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, StreamingResponse
from langchain_openai import ChatOpenAI
from server.reasoning_graph import build_reasoning_graph

# Load .env copied into the image at build time
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set; ensure .env is provided at build time")


# ---------- LangGraph plumbing ----------------------------------------------
LLM = ChatOpenAI(model="gpt-4o", temperature=0.2, streaming=True, openai_api_key=api_key)
GRAPH = build_reasoning_graph(LLM)

# ---------- FastAPI -----------------------------------------------------------
app = FastAPI(title="Web Agent Demo", version="0.1.0")


@app.get("/", response_class=PlainTextResponse)
def index() -> str:
    return (
        "Web Agent Demo is running.\n"
        "POST /chat with {'message': '...'} to stream a response.\n"
        "GET /health for health check.\n"
    )


@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok"


async def _token_stream(user_message: str) -> AsyncIterator[str]:
    last = ""
    try:
        for event in GRAPH.stream(
            {"messages": [{"role": "user", "content": user_message}]},
            stream_mode="values",
            config={"configurable": {"thread_id": "default"}},
        ):
            if not isinstance(event, dict):
                continue
            messages = event.get("messages")
            if not messages:
                continue
            last_msg = messages[-1]
            role = (getattr(last_msg, "type", None) or getattr(last_msg, "role", ""))
            if str(role).lower() not in {"ai", "assistant"}:
                # Skip echoing user/tool/system messages
                continue
            content = getattr(last_msg, "content", None)
            if content:
                text = str(content)
                if len(text) >= len(last) and text.startswith(last):
                    delta = text[len(last):]
                else:
                    delta = text
                last = text
                if delta:
                    yield delta
    except Exception as e:
        yield f"\n[ERROR] {e}\n"


@app.post("/chat")
async def chat(payload: dict):
    msg = payload.get("message", "")
    thread_id = payload.get("thread_id") or "default"
    if not msg:
        raise HTTPException(422, "message field required")

    async def _stream() -> AsyncIterator[str]:
        last = ""
        try:
            for event in GRAPH.stream(
                {"messages": [{"role": "user", "content": msg}]},
                stream_mode="values",
                config={"configurable": {"thread_id": str(thread_id)}},
            ):
                if not isinstance(event, dict):
                    continue
                messages = event.get("messages")
                if not messages:
                    continue
                last_msg = messages[-1]
                role = (getattr(last_msg, "type", None) or getattr(last_msg, "role", ""))
                if str(role).lower() not in {"ai", "assistant"}:
                    # Skip echoing user/tool/system messages
                    continue
                content = getattr(last_msg, "content", None)
                if content:
                    text = str(content)
                    if len(text) >= len(last) and text.startswith(last):
                        delta = text[len(last):]
                    else:
                        delta = text
                    last = text
                    if delta:
                        yield delta
        except Exception as e:
            yield f"\n[ERROR] {e}\n"

    return StreamingResponse(_stream(), media_type="text/plain")


# -------------------- Interactive shell over WebSocket -----------------------

def _set_winsize(fd: int, rows: int, cols: int) -> None:
    # TIOCSWINSZ expects unsigned short (rows, cols, xpix, ypix)
    try:
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    except Exception:
        pass


@app.websocket("/shell")
async def shell_ws(ws: WebSocket) -> None:
    # Optional bearer/token check to avoid unauthenticated remote exec
    expected = os.getenv("SHELL_TOKEN")

    # Accept early to be able to send error messages; validate right after
    await ws.accept()

    if expected:
        supplied: Optional[str] = None
        # Prefer Authorization header; fall back to query param `token`
        auth = ws.headers.get("authorization")
        if auth and auth.lower().startswith("bearer "):
            supplied = auth.split(" ", 1)[1].strip()
        else:
            supplied = ws.query_params.get("token")
        if not supplied or supplied != expected:
            await ws.send_text(json.dumps({"type": "error", "message": "unauthorized"}))
            await ws.close(code=4401)
            return

    # Expect an optional init message with rows/cols/cmd
    rows, cols = 24, 80
    cmd: Optional[list[str]] = None
    try:
        init_msg = await asyncio.wait_for(ws.receive_text(), timeout=3.0)
        try:
            payload = json.loads(init_msg)
            if isinstance(payload, dict) and payload.get("type") == "init":
                rows = int(payload.get("rows", rows))
                cols = int(payload.get("cols", cols))
                _cmd = payload.get("cmd")
                if isinstance(_cmd, list) and _cmd:
                    cmd = [str(x) for x in _cmd]
                elif isinstance(_cmd, str) and _cmd:
                    # Execute string via shell -lc "..." to allow compound commands
                    sh = shutil.which("bash") or shutil.which("sh") or "/bin/sh"
                    cmd = [sh, "-lc", _cmd]
        except json.JSONDecodeError:
            # Not JSON; treat as regular input and proceed with defaults
            pass
    except asyncio.TimeoutError:
        # No init sent; proceed with defaults
        pass

    # Determine default command if not provided
    if cmd is None:
        shell_path = os.environ.get("SHELL") or shutil.which("bash") or shutil.which("sh") or "/bin/sh"
        # Prefer interactive login shell where possible
        # Not all shells support -l/-i, so fall back gracefully
        default_cmds = [[shell_path, "-l"], [shell_path, "-i"], [shell_path]]
        for candidate in default_cmds:
            cmd = candidate
            break

    # Spawn PTY-backed child
    pid, master_fd = pty.fork()
    if pid == 0:
        # Child: execute the shell/command; set TERM
        os.environ.setdefault("TERM", "xterm-256color")
        try:
            os.execvp(cmd[0], cmd)
        except Exception:
            # If exec fails, exit child with non-zero status
            os._exit(127)

    # Parent: bridge between PTY and websocket
    loop = asyncio.get_running_loop()

    # Apply initial window size
    _set_winsize(master_fd, rows, cols)

    # Ensure non-blocking reads
    try:
        os.set_blocking(master_fd, False)
    except Exception:
        pass

    # Reader: when PTY is readable, forward data to client
    stop = asyncio.Event()

    def _on_pty_readable() -> None:
        try:
            data = os.read(master_fd, 4096)
            if data:
                asyncio.create_task(ws.send_bytes(data))
            else:
                # EOF
                loop.remove_reader(master_fd)
                if not stop.is_set():
                    stop.set()
        except OSError:
            # Likely EIO when child exits
            try:
                loop.remove_reader(master_fd)
            except Exception:
                pass
            if not stop.is_set():
                stop.set()

    loop.add_reader(master_fd, _on_pty_readable)

    async def _recv_from_client() -> None:
        try:
            while True:
                msg = await ws.receive()
                mtype = msg.get("type")
                if mtype == "websocket.disconnect":
                    break
                if "bytes" in msg and msg["bytes"] is not None:
                    try:
                        os.write(master_fd, msg["bytes"])  # send raw input
                    except OSError:
                        break
                elif "text" in msg and msg["text"] is not None:
                    try:
                        payload = json.loads(msg["text"]) if msg["text"].startswith("{") else None
                    except Exception:
                        payload = None
                    if isinstance(payload, dict) and payload.get("type") == "resize":
                        r = int(payload.get("rows", rows))
                        c = int(payload.get("cols", cols))
                        _set_winsize(master_fd, r, c)
                        rows, cols = r, c
                    else:
                        # Treat as plain text input
                        try:
                            os.write(master_fd, msg["text"].encode())
                        except OSError:
                            break
        except WebSocketDisconnect:
            pass
        finally:
            if not stop.is_set():
                stop.set()

    # Run until PTY closes or client disconnects
    recv_task = asyncio.create_task(_recv_from_client())

    try:
        await stop.wait()
    finally:
        # Cleanup
        try:
            loop.remove_reader(master_fd)
        except Exception:
            pass
        try:
            os.close(master_fd)
        except Exception:
            pass
        try:
            # Try to send an exit notice if still connected
            await ws.send_text(json.dumps({"type": "exit"}))
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass
        try:
            recv_task.cancel()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000)
