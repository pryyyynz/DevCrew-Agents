"""
Web interface for DevCrew Agents system.
Provides a chat-like interface for interacting with the agent orchestrator.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import os
import threading
from pathlib import Path
import concurrent.futures

from orchestrator import AgentOrchestrator
from communication import SharedMemoryManager, MessageBus

# Initialize FastAPI app
app = FastAPI(title="DevCrew Agents Web Interface", version="1.0.0")

# Setup static files and templates
web_dir = Path(__file__).parent / "web"
web_dir.mkdir(exist_ok=True)
(web_dir / "static").mkdir(exist_ok=True)
(web_dir / "templates").mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(web_dir / "static")), name="static")
templates = Jinja2Templates(directory=str(web_dir / "templates"))

# Global orchestrator instance
orchestrator = None
connected_clients: List[WebSocket] = []

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread executor for background tasks
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")

    async def broadcast(self, message: str):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return

        # Create a copy of the connections list to avoid modification during iteration
        connections_to_remove = []

        for connection in self.active_connections.copy():
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                connections_to_remove.append(connection)

        # Remove failed connections
        for connection in connections_to_remove:
            self.disconnect(connection)


manager = ConnectionManager()


def initialize_orchestrator():
    """Initialize the orchestrator instance."""
    global orchestrator
    try:
        from dotenv import load_dotenv
        load_dotenv()
        orchestrator = AgentOrchestrator()
        logger.info("Orchestrator initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting DevCrew Agents Web Interface...")
    success = initialize_orchestrator()
    if not success:
        logger.error("Failed to initialize orchestrator on startup")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
async def get_status():
    """Get current system status."""
    if not orchestrator:
        return JSONResponse({"error": "Orchestrator not initialized"}, status_code=500)

    try:
        status = orchestrator.get_completion_status()
        project_status = orchestrator._get_project_status()

        return {
            "status": "ready",
            "current_phase": status.get("current_phase", "unknown"),
            "phase_completion": status.get("phase_completion", 0),
            "total_tasks": status.get("total_tasks_completed", 0),
            "completed_tasks": project_status.get("completed_tasks", 0),
            "failed_tasks": project_status.get("failed_tasks", 0),
            "active_queries": project_status.get("active_queries", 0),
            "last_agent": project_status.get("last_agent", None),
            "agents": list(orchestrator.agents.keys()) if orchestrator.agents else [],
            "models": orchestrator.model_mapping if orchestrator else {}
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/agent-performance")
async def get_agent_performance():
    """Get agent performance metrics."""
    if not orchestrator:
        return JSONResponse({"error": "Orchestrator not initialized"}, status_code=500)

    try:
        performance = orchestrator._get_agent_performance()
        return {"performance": performance}
    except Exception as e:
        logger.error(f"Error getting agent performance: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/conversation-history")
async def get_conversation_history():
    """Get conversation history."""
    if not orchestrator:
        return JSONResponse({"error": "Orchestrator not initialized"}, status_code=500)

    try:
        # Get recent user queries and responses
        conversations = []
        for query in orchestrator.user_queries[-20:]:  # Last 20 queries
            response = orchestrator.query_responses.get(query.id)
            conversations.append({
                "id": query.id,
                "timestamp": query.timestamp,
                "query": query.query,
                "response": response,
                "phase": orchestrator.current_phase.value
            })

        return {"conversations": conversations}
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/agent-history")
async def get_agent_history():
    """Get agent turn history."""
    if not orchestrator:
        return JSONResponse({"error": "Orchestrator not initialized"}, status_code=500)

    try:
        return {
            # Last 50 turns
            "agent_turns": orchestrator.agent_turn_history[-50:],
            "current_agent": orchestrator.current_agent
        }
    except Exception as e:
        logger.error(f"Error getting agent history: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/logs")
async def get_logs():
    """Get system logs."""
    try:
        logs = []

        # Read orchestrator log
        if os.path.exists("orchestrator.log"):
            with open("orchestrator.log", "r") as f:
                log_lines = f.readlines()[-100:]  # Last 100 lines
                for line in log_lines:
                    logs.append({
                        "source": "orchestrator",
                        "content": line.strip(),
                        "timestamp": datetime.now().isoformat()
                    })

        # Read agent logs
        for agent_type in ["project_manager", "designer", "coder", "tester"]:
            log_file = f"agent_{agent_type}.log"
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    log_lines = f.readlines()[-20:]  # Last 20 lines per agent
                    for line in log_lines:
                        logs.append({
                            "source": agent_type,
                            "content": line.strip(),
                            "timestamp": datetime.now().isoformat()
                        })

        # Sort by timestamp (approximate)
        logs = sorted(logs, key=lambda x: x["timestamp"], reverse=True)

        return {"logs": logs[:200]}  # Return latest 200 logs
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def safe_broadcast(message_data: dict):
    """Safely broadcast message to all connected clients."""
    try:
        message_json = json.dumps(message_data)
        await manager.broadcast(message_json)
    except Exception as e:
        logger.error(f"Error in safe_broadcast: {e}")


def process_message_sync(user_message: str, loop: asyncio.AbstractEventLoop):
    """Process message in synchronous context but schedule async broadcasts."""
    try:
        response = orchestrator.handle_user_query(user_message)

        # Schedule the broadcast in the event loop
        asyncio.run_coroutine_threadsafe(
            safe_broadcast({
                "type": "agent_response",
                "content": response,
                "timestamp": datetime.now().isoformat(),
                "phase": orchestrator.current_phase.value,
                "current_agent": orchestrator.current_agent
            }),
            loop
        )

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        # Schedule error broadcast in the event loop
        asyncio.run_coroutine_threadsafe(
            safe_broadcast({
                "type": "error",
                "content": f"Error processing message: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }),
            loop
        )


@app.post("/api/send-message")
async def send_message(request: Request):
    """Process user message and return agent response."""
    if not orchestrator:
        return JSONResponse({"error": "Orchestrator not initialized"}, status_code=500)

    try:
        data = await request.json()
        user_message = data.get("message", "").strip()

        if not user_message:
            return JSONResponse({"error": "Empty message"}, status_code=400)

        # Broadcast user message to connected clients
        await safe_broadcast({
            "type": "user_message",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })

        # Get the current event loop
        loop = asyncio.get_running_loop()

        # Process message with orchestrator in thread pool
        executor.submit(process_message_sync, user_message, loop)

        return {"status": "processing", "message": "Query sent to agents for processing"}

    except Exception as e:
        logger.error(f"Error in send_message: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/files")
async def get_project_files():
    """Get list of project files."""
    try:
        files = []

        # Common file types to look for
        file_patterns = ["*.py", "*.md", "*.txt", "*.json",
                         "*.yaml", "*.yml", "*.js", "*.html", "*.css"]

        for pattern in file_patterns:
            for file_path in Path(".").glob(f"**/{pattern}"):
                if any(skip in str(file_path) for skip in [".git", "__pycache__", "node_modules", ".env"]):
                    continue

                try:
                    stat = file_path.stat()
                    files.append({
                        "path": str(file_path),
                        "name": file_path.name,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "type": file_path.suffix
                    })
                except Exception:
                    continue

        # Sort by modification time (newest first)
        files.sort(key=lambda x: x["modified"], reverse=True)

        return {"files": files[:100]}  # Limit to 100 files

    except Exception as e:
        logger.error(f"Error getting files: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/file/{file_path:path}")
async def get_file_content(file_path: str):
    """Get content of a specific file."""
    try:
        file_path_obj = Path(file_path)

        # Security check - ensure file is within current directory
        if not file_path_obj.is_relative_to(Path.cwd()):
            return JSONResponse({"error": "Access denied"}, status_code=403)

        if not file_path_obj.exists():
            return JSONResponse({"error": "File not found"}, status_code=404)

        # Check file size (limit to 1MB)
        if file_path_obj.stat().st_size > 1024 * 1024:
            return JSONResponse({"error": "File too large"}, status_code=413)

        try:
            content = file_path_obj.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            content = "[Binary file - cannot display]"

        return {
            "path": str(file_path_obj),
            "content": content,
            "size": file_path_obj.stat().st_size,
            "type": file_path_obj.suffix
        }

    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/execute-code")
async def execute_code(request: Request):
    """Execute code in the sandbox using a simplified code execution function."""
    logger.info("Code execution endpoint called")  # Add debug logging
    try:
        data = await request.json()
        code = data.get("code", "").strip()
        language = data.get("language", "python")

        # Log the request
        logger.info(f"Executing {language} code: {code[:50]}...")

        if not code:
            return JSONResponse({"error": "No code provided"}, status_code=400)

        # Use a simplified code execution function instead of the tool
        result = execute_code_safely(code, language, timeout=30)

        # Log the result
        logger.info(f"Code execution result: {result[:100]}...")

        return {
            "status": "success",
            "output": result,
            "language": language,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error executing code: {e}")
        return JSONResponse({
            "error": f"Execution failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }, status_code=500)


def execute_code_safely(code: str, language: str = "python", timeout: int = 30) -> str:
    """
    Execute code safely in a sandboxed environment.
    This is a simplified version that doesn't depend on crewai.
    """
    import subprocess
    import tempfile
    import os
    import sys

    try:
        # Security check - basic code filtering
        dangerous_patterns = [
            'import os', 'import sys', 'import subprocess',
            'exec(', 'eval(', '__import__', 'open(',
            'file(', 'input(', 'raw_input('
        ]

        # Allow some safe imports for demo purposes
        safe_patterns = [
            'import datetime', 'import math', 'import random',
            'import json', 'import re', 'from datetime import',
            'from math import', 'from random import'
        ]

        # Check for dangerous patterns
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                # Check if it's in our safe list
                is_safe = any(
                    safe in code_lower for safe in safe_patterns if pattern.split()[1] in safe)
                if not is_safe:
                    return f"🚫 Security Error: '{pattern}' is not allowed in sandbox"

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{language}', delete=False) as f:
            f.write(code)
            temp_file = f.name

        # Execute based on language
        if language == "python":
            # Use python executable with restricted environment
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                # Add some security by limiting environment
                env={'PATH': os.environ.get('PATH', ''), 'PYTHONPATH': ''}
            )
        elif language == "javascript":
            # Execute with node if available
            result = subprocess.run(
                ['node', temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
        elif language == "html":
            # For HTML, just return the code as it would be rendered
            os.unlink(temp_file)  # Clean up first
            return f"HTML Code Preview:\n{code}\n\n(Note: HTML rendering would show in browser)"
        else:
            os.unlink(temp_file)
            return f"❌ Language '{language}' is not supported. Supported: python, javascript, html"

        # Clean up temporary file
        os.unlink(temp_file)

        # Return results
        if result.returncode == 0:
            output = result.stdout.strip()
            if not output:
                output = "(Code executed successfully with no output)"
            return output
        else:
            error_msg = result.stderr.strip()
            if not error_msg:
                error_msg = f"Process exited with code {result.returncode}"
            return f"❌ Execution Error:\n{error_msg}"

    except subprocess.TimeoutExpired:
        # Clean up temp file
        try:
            os.unlink(temp_file)
        except:
            pass
        return f"⏰ Execution timeout ({timeout}s limit exceeded)"
    except FileNotFoundError as e:
        if 'node' in str(e):
            return "❌ Node.js not found. Please install Node.js to run JavaScript code."
        return f"❌ Runtime not found: {str(e)}"
    except Exception as e:
        # Clean up temp file
        try:
            os.unlink(temp_file)
        except:
            pass
        return f"🚫 Execution Error: {str(e)}"


@app.get("/api/generated-code")
async def get_generated_code():
    """Get the latest generated code for the sandbox."""
    try:
        from communication.memory_manager import SharedMemoryManager

        memory_manager = SharedMemoryManager("sandbox_code.db")
        latest_code = memory_manager.get("latest_generated_code", "web_api")

        if latest_code:
            return {
                "status": "success",
                "code_data": latest_code,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "no_code",
                "message": "No generated code available",
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Error getting generated code: {e}")
        return JSONResponse({
            "error": f"Failed to retrieve generated code: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }, status_code=500)


@app.get("/api/generated-code/history")
async def get_generated_code_history():
    """Get history of all generated code."""
    try:
        from communication.memory_manager import SharedMemoryManager

        memory_manager = SharedMemoryManager("sandbox_code.db")

        # Get all generated code entries
        all_keys = memory_manager.list_keys("generated_code_")
        code_history = []

        for key in all_keys:
            code_data = memory_manager.get(key, "web_api")
            if code_data:
                code_history.append({
                    "id": key,
                    "code": code_data.get("code", ""),
                    "language": code_data.get("language", "python"),
                    "filename": code_data.get("filename"),
                    "description": code_data.get("description"),
                    "timestamp": code_data.get("timestamp"),
                    "agent": code_data.get("agent", "unknown")
                })

        # Sort by timestamp (newest first)
        code_history.sort(key=lambda x: x["timestamp"], reverse=True)

        return {
            "status": "success",
            "history": code_history[:20],  # Return latest 20
            "count": len(code_history)
        }

    except Exception as e:
        logger.error(f"Error getting code history: {e}")
        return JSONResponse({
            "error": f"Failed to retrieve code history: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }, status_code=500)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()

            # Handle different message types from client
            try:
                message = json.loads(data)
                msg_type = message.get("type", "unknown")

                if msg_type == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}))

                elif msg_type == "get_status":
                    if orchestrator:
                        status = orchestrator.get_completion_status()
                        await websocket.send_text(json.dumps({"type": "status", "data": status}))

            except json.JSONDecodeError:
                logger.warning("Received invalid JSON from WebSocket client")

    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
