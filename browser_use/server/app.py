"""FastAPI Web Server for Browser-Use.

Provides a REST API, health check endpoints, 9router / custom OpenAI API compatibility,
permanent task log disk persistence, date range log filtering, JSON export,
and an interactive web dashboard for executing browser automation tasks.
Designed for deployment on cloud platforms and Docker containers like Easypanel.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
	title="Browser-Use API Server",
	description="REST API and Web Interface for Browser-Use AI Web Automation Agent",
	version="0.13.6",
)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Setup Data Directory & Persistent Logs Directory
DATA_DIR = os.environ.get("DATA_DIR", "/data")
if not os.path.exists(DATA_DIR):
	try:
		os.makedirs(DATA_DIR, exist_ok=True)
	except Exception:
		DATA_DIR = os.path.join(os.getcwd(), "data")
		os.makedirs(DATA_DIR, exist_ok=True)

LOGS_DIR = os.path.join(DATA_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


def save_task_to_disk(task_data: dict) -> None:
	"""Persist task log entry permanently to disk as JSON."""
	try:
		task_id = task_data.get("task_id")
		if not task_id:
			return
		file_path = os.path.join(LOGS_DIR, f"{task_id}.json")
		with open(file_path, "w", encoding="utf-8") as f:
			json.dump(task_data, f, ensure_ascii=False, indent=2)
	except Exception as exc:
		logger.exception(f"Failed to save task log {task_data.get('task_id')} to disk")


def load_all_tasks_from_disk() -> Dict[str, Dict[str, Any]]:
	"""Load all task logs from disk on startup."""
	tasks: Dict[str, Dict[str, Any]] = {}
	if not os.path.exists(LOGS_DIR):
		return tasks
	for filename in os.listdir(LOGS_DIR):
		if filename.endswith(".json"):
			file_path = os.path.join(LOGS_DIR, filename)
			try:
				with open(file_path, "r", encoding="utf-8") as f:
					task_data = json.load(f)
					if isinstance(task_data, dict) and "task_id" in task_data:
						tasks[task_data["task_id"]] = task_data
			except Exception:
				pass
	return tasks


# Initialize tasks database permanently from disk
tasks_db: Dict[str, Dict[str, Any]] = load_all_tasks_from_disk()


class TaskRequest(BaseModel):
	"""Request schema for initiating a browser automation task."""

	task: str = Field(..., description="The task description for the AI agent to execute.")
	llm_provider: str = Field(
		default="browser_use",
		description="LLM provider: 'browser_use' (recommended), '9router', 'custom', 'openai', 'google', or 'anthropic'.",
	)
	model_name: Optional[str] = Field(
		default=None,
		description="Optional specific model name (e.g. 'gpt-4o', 'claude-3-5-sonnet', 'gemini-2.5-flash').",
	)
	api_key: Optional[str] = Field(
		default=None,
		description="API key for the chosen LLM provider if not configured via environment variables.",
	)
	api_base_url: Optional[str] = Field(
		default=None,
		description="Custom OpenAI-compatible API base URL (e.g. https://terbaik-9router.3obhmi.easypanel.host/v1).",
	)
	use_vision: bool = Field(default=True, description="Enable vision capabilities for screenshot processing.")
	headless: bool = Field(default=True, description="Run browser in headless mode.")
	max_steps: int = Field(default=100, ge=1, le=500, description="Maximum execution steps allowed.")
	use_cloud: bool = Field(
		default=False,
		description="Use Browser Use Cloud remote browser for stealth, high performance, and captcha bypass.",
	)


class TaskResponse(BaseModel):
	"""Response schema after initiating a task."""

	task_id: str
	status: str
	message: str
	created_at: str


class TaskStatusResponse(BaseModel):
	"""Status schema for querying task execution progress."""

	task_id: str
	status: str  # "pending", "running", "completed", "failed"
	task: str
	created_at: str
	completed_at: Optional[str] = None
	result: Optional[str] = None
	errors: Optional[List[str]] = None
	urls_visited: Optional[List[str]] = None
	steps_completed: Optional[int] = None


class FetchModelsRequest(BaseModel):
	"""Request schema for importing/fetching models from a custom OpenAI-compatible API gateway (e.g. 9router)."""

	api_base_url: str = Field(..., description="API base URL (e.g. https://terbaik-9router.3obhmi.easypanel.host/v1).")
	api_key: Optional[str] = Field(default=None, description="Optional API key for authentication.")


class FetchModelsResponse(BaseModel):
	"""Response schema containing list of imported models."""

	api_base_url: str
	models: List[str]


def filter_tasks_list(
	tasks: List[Dict[str, Any]],
	start_date: Optional[str] = None,
	end_date: Optional[str] = None,
	status_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
	"""Filter task logs by date range (start_date, end_date) and status."""
	filtered = []
	for t in tasks:
		# Status filter
		if status_filter and status_filter.strip() and status_filter.lower() != "all":
			if t.get("status", "").lower() != status_filter.strip().lower():
				continue

		# Date filter (compare YYYY-MM-DD strings)
		created_at = t.get("created_at", "")
		if created_at:
			task_date = created_at[:10]  # YYYY-MM-DD
			if start_date and start_date.strip():
				if task_date < start_date.strip()[:10]:
					continue
			if end_date and end_date.strip():
				if task_date > end_date.strip()[:10]:
					continue

		filtered.append(t)

	# Sort newest first
	filtered.sort(key=lambda x: x.get("created_at", ""), reverse=True)
	return filtered


async def execute_task_background(task_id: str, request: TaskRequest) -> None:
	"""Background worker function executing the Browser-Use Agent."""
	tasks_db[task_id]["status"] = "running"
	save_task_to_disk(tasks_db[task_id])

	# Set API key into environment if provided
	if request.api_key:
		if request.llm_provider == "browser_use":
			os.environ["BROWSER_USE_API_KEY"] = request.api_key
		elif request.llm_provider in ("openai", "9router", "custom"):
			os.environ["OPENAI_API_KEY"] = request.api_key
		elif request.llm_provider == "google":
			os.environ["GOOGLE_API_KEY"] = request.api_key
		elif request.llm_provider == "anthropic":
			os.environ["ANTHROPIC_API_KEY"] = request.api_key

	try:
		# Import browser-use modules inside task runner
		from browser_use import Agent, Browser
		from browser_use.llm import ChatAnthropic, ChatBrowserUse, ChatGoogle, ChatOpenAI

		# Select LLM
		llm = None
		provider = request.llm_provider.lower()

		if provider in ("9router", "custom", "openai_compatible") or request.api_base_url:
			model = request.model_name or "gpt-4o"
			llm = ChatOpenAI(
				model=model,
				base_url=request.api_base_url,
				api_key=request.api_key or os.environ.get("OPENAI_API_KEY"),
			)
		elif provider == "browser_use":
			llm = ChatBrowserUse()
		elif provider == "openai":
			model = request.model_name or "gpt-4.1-mini"
			llm = ChatOpenAI(model=model, api_key=request.api_key)
		elif provider == "google":
			model = request.model_name or "gemini-2.5-flash"
			llm = ChatGoogle(model=model, api_key=request.api_key)
		elif provider == "anthropic":
			model = request.model_name or "claude-sonnet-4-0"
			llm = ChatAnthropic(model=model, api_key=request.api_key)
		else:
			# Default fallback to ChatBrowserUse
			llm = ChatBrowserUse()

		# Setup browser options
		browser = None
		if request.use_cloud:
			browser = Browser(use_cloud=True, headless=request.headless)
		else:
			browser = Browser(headless=request.headless)

		agent = Agent(
			task=request.task,
			llm=llm,
			browser=browser,
			use_vision=request.use_vision,
		)

		history = await agent.run(max_steps=request.max_steps)

		tasks_db[task_id]["status"] = "completed"
		tasks_db[task_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
		tasks_db[task_id]["result"] = history.final_result() or "Task finished successfully."
		tasks_db[task_id]["urls_visited"] = history.urls()
		tasks_db[task_id]["steps_completed"] = history.number_of_steps()
		tasks_db[task_id]["errors"] = [e for e in history.errors() if e]
		save_task_to_disk(tasks_db[task_id])

	except Exception as exc:
		logger.exception(f"Error executing task {task_id}")
		tasks_db[task_id]["status"] = "failed"
		tasks_db[task_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
		tasks_db[task_id]["result"] = f"Error: {str(exc)}"
		tasks_db[task_id]["errors"] = [str(exc)]
		save_task_to_disk(tasks_db[task_id])


@app.get("/health", summary="Easypanel Healthcheck Endpoint")
async def health_check():
	"""Healthcheck endpoint used by Easypanel and load balancers."""
	return {
		"status": "ok",
		"service": "browser-use",
		"version": "0.13.6",
		"timestamp": datetime.now(timezone.utc).isoformat(),
	}


@app.post("/api/v1/models", response_model=FetchModelsResponse, summary="Import models from 9router / custom OpenAI API base URL")
async def fetch_models(request: FetchModelsRequest):
	"""Fetch available AI model list from a custom OpenAI-compatible API base URL (e.g. 9router)."""
	base_url = request.api_base_url.rstrip("/")
	if not base_url.endswith("/models"):
		url = f"{base_url}/models"
	else:
		url = base_url

	headers = {}
	if request.api_key:
		headers["Authorization"] = f"Bearer {request.api_key}"

	try:
		async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
			resp = await client.get(url, headers=headers)
			if resp.status_code != 200:
				raise HTTPException(
					status_code=resp.status_code,
					detail=f"Failed to fetch models from {url} (HTTP {resp.status_code}): {resp.text[:300]}",
				)

			data = resp.json()
			models_list: List[str] = []

			if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
				for item in data["data"]:
					if isinstance(item, dict) and "id" in item:
						models_list.append(str(item["id"]))
					elif isinstance(item, str):
						models_list.append(item)
			elif isinstance(data, list):
				for item in data:
					if isinstance(item, dict) and "id" in item:
						models_list.append(str(item["id"]))
					elif isinstance(item, str):
						models_list.append(item)

			if not models_list:
				models_list = ["default"]

			return FetchModelsResponse(api_base_url=request.api_base_url, models=models_list)
	except HTTPException:
		raise
	except Exception as exc:
		logger.exception(f"Error fetching models from {url}")
		raise HTTPException(status_code=500, detail=f"Could not connect to {url}: {str(exc)}")


@app.post("/api/v1/run", response_model=TaskResponse, summary="Execute a Browser-Use Automation Task")
async def run_task(request: TaskRequest, background_tasks: BackgroundTasks):
	"""Endpoint to submit and launch a browser automation task."""
	task_id = str(uuid.uuid4())
	created_at = datetime.now(timezone.utc).isoformat()

	tasks_db[task_id] = {
		"task_id": task_id,
		"task": request.task,
		"status": "pending",
		"created_at": created_at,
		"completed_at": None,
		"result": None,
		"errors": [],
		"urls_visited": [],
		"steps_completed": 0,
	}

	save_task_to_disk(tasks_db[task_id])
	background_tasks.add_task(execute_task_background, task_id, request)

	return TaskResponse(
		task_id=task_id,
		status="pending",
		message="Task queued and executing in background.",
		created_at=created_at,
	)


@app.get("/api/v1/tasks", summary="List All Tasks with Optional Date Range & Status Filtering")
async def list_tasks(
	start_date: Optional[str] = None,
	end_date: Optional[str] = None,
	status: Optional[str] = None,
):
	"""List task logs with support for date range filtering (start_date, end_date) and status filtering."""
	all_tasks = list(tasks_db.values())
	return filter_tasks_list(all_tasks, start_date=start_date, end_date=end_date, status_filter=status)


@app.get("/api/v1/tasks/export", summary="Export Filtered Task Logs as JSON File")
async def export_tasks_json(
	start_date: Optional[str] = None,
	end_date: Optional[str] = None,
	status: Optional[str] = None,
):
	"""Export task logs as a downloadable JSON file attachment."""
	all_tasks = list(tasks_db.values())
	filtered = filter_tasks_list(all_tasks, start_date=start_date, end_date=end_date, status_filter=status)

	content = json.dumps(filtered, ensure_ascii=False, indent=2)
	filename = f"browser_use_logs_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"

	return Response(
		content=content,
		media_type="application/json",
		headers={"Content-Disposition": f'attachment; filename="{filename}"'},
	)


@app.get("/api/v1/tasks/{task_id}", response_model=TaskStatusResponse, summary="Get Task Status")
async def get_task_status(task_id: str):
	"""Get status and output results of a specific task."""
	if task_id not in tasks_db:
		raise HTTPException(status_code=404, detail="Task ID not found")
	return TaskStatusResponse(**tasks_db[task_id])


@app.get("/", response_class=HTMLResponse, summary="Embedded Web Dashboard")
async def dashboard():
	"""Embedded interactive web dashboard for Browser-Use with permanent logs, date range search, and JSON export."""
	html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Browser-Use Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #0f172a; color: #f8fafc; padding-top: 2rem; }
        .card { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; color: #f8fafc; }
        .form-control, .form-select { background-color: #0f172a; border: 1px solid #475569; color: #f8fafc; }
        .form-control:focus, .form-select:focus { background-color: #0f172a; color: #f8fafc; border-color: #3b82f6; box-shadow: none; }
        .btn-primary { background-color: #3b82f6; border: none; font-weight: 600; }
        .btn-primary:hover { background-color: #2563eb; }
        .btn-success { font-weight: 600; }
        .badge-pending { background-color: #f59e0b; }
        .badge-running { background-color: #3b82f6; }
        .badge-completed { background-color: #10b981; }
        .badge-failed { background-color: #ef4444; }
        pre { background-color: #0f172a; border: 1px solid #334155; padding: 1rem; border-radius: 8px; color: #38bdf8; max-height: 400px; overflow-y: auto; }
        .table-dark { --bs-table-bg: #0f172a; --bs-table-border-color: #334155; }
        .result-truncate { max-width: 280px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: block; }
    </style>
</head>
<body>
    <div class="container mb-5">
        <div class="d-flex align-items-center justify-content-between mb-4">
            <div>
                <h1 class="fw-bold mb-0">🌐 Browser-Use Dashboard</h1>
                <p class="text-secondary mb-0">Autonomous AI Web Automation Agent with Permanent Log History & 9router Support</p>
            </div>
            <span class="badge bg-success px-3 py-2">Health: OK</span>
        </div>

        <!-- Task Creator & Live Execution View -->
        <div class="row g-4 mb-4">
            <div class="col-lg-5">
                <div class="card p-4">
                    <h5 class="fw-bold mb-3">Create Automation Task</h5>
                    <form id="taskForm">
                        <div class="mb-3">
                            <label class="form-label text-secondary">Task Prompt</label>
                            <textarea id="taskInput" class="form-control" rows="3" placeholder="e.g. Find the top post on Hacker News and extract the summary" required></textarea>
                        </div>
                        
                        <div class="mb-3">
                            <label class="form-label text-secondary">LLM Model Provider</label>
                            <select id="providerSelect" class="form-select" onchange="toggleProviderFields()">
                                <option value="browser_use">ChatBrowserUse (Recommended - Fast & Accurate)</option>
                                <option value="9router" selected>9router / Custom OpenAI Compatible Endpoint</option>
                                <option value="openai">OpenAI (Official)</option>
                                <option value="google">Google Gemini</option>
                                <option value="anthropic">Anthropic Claude</option>
                            </select>
                        </div>

                        <div id="apiBaseUrlContainer" class="mb-3">
                            <label class="form-label text-secondary">API Base URL (e.g. 9router)</label>
                            <div class="input-group">
                                <input type="text" id="apiBaseUrlInput" class="form-control" value="https://terbaik-9router.3obhmi.easypanel.host/v1" placeholder="https://terbaik-9router.3obhmi.easypanel.host/v1">
                                <button type="button" class="btn btn-outline-info" onclick="importModels()">📥 Import Models</button>
                            </div>
                            <small id="importStatus" class="form-text text-muted"></small>
                        </div>

                        <div class="mb-3">
                            <label class="form-label text-secondary">Model Name</label>
                            <select id="modelSelect" class="form-select">
                                <option value="gpt-4o">gpt-4o (Default)</option>
                            </select>
                        </div>

                        <div class="mb-3">
                            <label class="form-label text-secondary">API Key (Optional if set in ENV)</label>
                            <input type="password" id="apiKeyInput" class="form-control" placeholder="API Key...">
                        </div>

                        <div class="form-check form-switch mb-3">
                            <input class="form-check-input" type="checkbox" id="useCloudCheck">
                            <label class="form-check-label text-secondary" for="useCloudCheck">Use Browser Use Cloud (use_cloud=True)</label>
                        </div>

                        <button type="submit" class="btn btn-primary w-100 py-2">🚀 Launch Task</button>
                    </form>
                </div>
            </div>

            <div class="col-lg-7">
                <div class="card p-4">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <h5 class="fw-bold mb-0">Live Active Task Monitor</h5>
                        <button class="btn btn-sm btn-outline-secondary" onclick="checkStatus()">Refresh</button>
                    </div>
                    <div id="statusContainer" class="text-secondary">No active task running. Submit a prompt or view historical logs below.</div>
                    <div id="outputContainer" style="display: none;" class="mt-3">
                        <h6>Result Output:</h6>
                        <pre id="resultText"></pre>
                    </div>
                </div>
            </div>
        </div>

        <!-- Permanent Logs & Filter Section -->
        <div class="row">
            <div class="col-12">
                <div class="card p-4">
                    <div class="d-flex flex-wrap align-items-center justify-content-between mb-3 gap-2">
                        <h5 class="fw-bold mb-0">📜 Permanent Execution Logs & History</h5>
                        <button class="btn btn-success btn-sm" onclick="downloadJsonLogs()">📥 Download JSON</button>
                    </div>

                    <!-- Filter Controls -->
                    <div class="row g-2 align-items-end mb-4 bg-dark p-3 rounded border border-secondary">
                        <div class="col-md-3">
                            <label class="form-label text-secondary mb-1">Start Date</label>
                            <input type="date" id="filterStartDate" class="form-control form-control-sm">
                        </div>
                        <div class="col-md-3">
                            <label class="form-label text-secondary mb-1">End Date</label>
                            <input type="date" id="filterEndDate" class="form-control form-control-sm">
                        </div>
                        <div class="col-md-3">
                            <label class="form-label text-secondary mb-1">Status</label>
                            <select id="filterStatus" class="form-select form-select-sm">
                                <option value="all" selected>All Statuses</option>
                                <option value="completed">Completed</option>
                                <option value="failed">Failed</option>
                                <option value="running">Running</option>
                                <option value="pending">Pending</option>
                            </select>
                        </div>
                        <div class="col-md-3 d-flex gap-2">
                            <button onclick="loadLogsTable()" class="btn btn-primary btn-sm flex-grow-1">🔍 Filter Logs</button>
                            <button onclick="resetFilters()" class="btn btn-outline-secondary btn-sm">Reset</button>
                        </div>
                    </div>

                    <!-- Logs Table -->
                    <div class="table-responsive">
                        <table class="table table-dark table-hover align-middle mb-0">
                            <thead>
                                <tr>
                                    <th>Task ID</th>
                                    <th>Date & Time</th>
                                    <th>Status</th>
                                    <th>Steps</th>
                                    <th>Result / Details</th>
                                    <th>Action</th>
                                </tr>
                            </thead>
                            <tbody id="logsTableBody">
                                <tr>
                                    <td colspan="6" class="text-center text-secondary py-4">Loading persistent logs...</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Task Detail Modal -->
    <div class="modal fade" id="detailModal" tabindex="-1" aria-hidden="true">
        <div class="modal-dialog modal-lg">
            <div class="modal-content card">
                <div class="modal-header border-secondary">
                    <h5 class="modal-title fw-bold" id="detailModalTitle">Task Details</h5>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <pre id="detailModalBody"></pre>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let currentTaskId = null;
        let pollInterval = null;
        let currentLogsData = [];

        function toggleProviderFields() {
            const provider = document.getElementById('providerSelect').value;
            const baseUrlContainer = document.getElementById('apiBaseUrlContainer');
            const modelSelect = document.getElementById('modelSelect');

            if (provider === '9router' || provider === 'custom') {
                baseUrlContainer.style.display = 'block';
            } else {
                baseUrlContainer.style.display = 'none';
            }

            modelSelect.innerHTML = '';
            if (provider === 'browser_use') {
                modelSelect.innerHTML = '<option value="ChatBrowserUse">ChatBrowserUse</option>';
            } else if (provider === '9router') {
                modelSelect.innerHTML = '<option value="gpt-4o">gpt-4o</option><option value="gpt-4.1-mini">gpt-4.1-mini</option><option value="claude-3-5-sonnet">claude-3-5-sonnet</option>';
            } else if (provider === 'openai') {
                modelSelect.innerHTML = '<option value="gpt-4.1-mini">gpt-4.1-mini</option><option value="gpt-4o">gpt-4o</option><option value="o3-mini">o3-mini</option>';
            } else if (provider === 'google') {
                modelSelect.innerHTML = '<option value="gemini-2.5-flash">gemini-2.5-flash</option><option value="gemini-2.5-pro">gemini-2.5-pro</option>';
            } else if (provider === 'anthropic') {
                modelSelect.innerHTML = '<option value="claude-sonnet-4-0">claude-sonnet-4-0</option><option value="claude-3-5-haiku">claude-3-5-haiku</option>';
            }
        }

        async function importModels() {
            const apiBaseUrl = document.getElementById('apiBaseUrlInput').value.trim();
            const apiKey = document.getElementById('apiKeyInput').value.trim();
            const importStatus = document.getElementById('importStatus');
            const modelSelect = document.getElementById('modelSelect');

            if (!apiBaseUrl) {
                importStatus.innerHTML = '<span class="text-danger">Please enter an API Base URL.</span>';
                return;
            }

            importStatus.innerHTML = '<span class="text-info">⏳ Fetching models from ' + apiBaseUrl + '...</span>';

            try {
                const res = await fetch('/api/v1/models', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        api_base_url: apiBaseUrl,
                        api_key: apiKey || null
                    })
                });

                if (!res.ok) {
                    const errData = await res.json();
                    throw new Error(errData.detail || 'Failed to fetch models');
                }

                const data = await res.json();
                if (data.models && data.models.length > 0) {
                    modelSelect.innerHTML = '';
                    data.models.forEach(m => {
                        const opt = document.createElement('option');
                        opt.value = m;
                        opt.textContent = m;
                        modelSelect.appendChild(opt);
                    });
                    importStatus.innerHTML = `<span class="text-success">✅ Successfully imported ${data.models.length} model(s) from 9router!</span>`;
                } else {
                    importStatus.innerHTML = '<span class="text-warning">No models found in response.</span>';
                }
            } catch (err) {
                importStatus.innerHTML = '<span class="text-danger">❌ Error: ' + err.message + '</span>';
            }
        }

        document.getElementById('taskForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const task = document.getElementById('taskInput').value;
            const provider = document.getElementById('providerSelect').value;
            const apiBaseUrl = document.getElementById('apiBaseUrlInput').value.trim();
            const modelName = document.getElementById('modelSelect').value;
            const apiKey = document.getElementById('apiKeyInput').value;
            const useCloud = document.getElementById('useCloudCheck').checked;

            document.getElementById('statusContainer').innerHTML = '<div class="spinner-border spinner-border-sm text-primary"></div> Submitting task...';

            try {
                const res = await fetch('/api/v1/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        task: task,
                        llm_provider: provider,
                        model_name: modelName || null,
                        api_base_url: (provider === '9router' || provider === 'custom') ? apiBaseUrl : null,
                        api_key: apiKey || null,
                        use_cloud: useCloud,
                        headless: true
                    })
                });
                const data = await res.json();
                currentTaskId = data.task_id;
                startPolling();
                loadLogsTable();
            } catch (err) {
                document.getElementById('statusContainer').innerHTML = '<span class="text-danger">Failed to submit task: ' + err.message + '</span>';
            }
        });

        function startPolling() {
            if (pollInterval) clearInterval(pollInterval);
            pollInterval = setInterval(checkStatus, 2000);
            checkStatus();
        }

        async function checkStatus() {
            if (!currentTaskId) return;
            try {
                const res = await fetch('/api/v1/tasks/' + currentTaskId);
                const data = await res.json();
                
                let badgeClass = 'badge-pending';
                if (data.status === 'running') badgeClass = 'badge-running';
                if (data.status === 'completed') badgeClass = 'badge-completed';
                if (data.status === 'failed') badgeClass = 'badge-failed';

                document.getElementById('statusContainer').innerHTML = `
                    <div><strong>Task ID:</strong> <code>${data.task_id}</code></div>
                    <div><strong>Status:</strong> <span class="badge ${badgeClass}">${data.status.toUpperCase()}</span></div>
                    <div><strong>Steps Completed:</strong> ${data.steps_completed}</div>
                    ${data.urls_visited && data.urls_visited.length ? '<div><strong>Visited URLs:</strong> ' + data.urls_visited.join(', ') + '</div>' : ''}
                `;

                if (data.status === 'completed' || data.status === 'failed') {
                    clearInterval(pollInterval);
                    document.getElementById('outputContainer').style.display = 'block';
                    document.getElementById('resultText').textContent = data.result || 'No output.';
                    loadLogsTable();
                }
            } catch (err) {
                console.error(err);
            }
        }

        async function loadLogsTable() {
            const startDate = document.getElementById('filterStartDate').value;
            const endDate = document.getElementById('filterEndDate').value;
            const status = document.getElementById('filterStatus').value;

            let url = '/api/v1/tasks?';
            const params = new URLSearchParams();
            if (startDate) params.append('start_date', startDate);
            if (endDate) params.append('end_date', endDate);
            if (status && status !== 'all') params.append('status', status);

            url += params.toString();

            try {
                const res = await fetch(url);
                const data = await res.json();
                currentLogsData = data;
                renderLogsTable(data);
            } catch (err) {
                document.getElementById('logsTableBody').innerHTML = `<tr><td colspan="6" class="text-danger text-center">Failed to load logs: ${err.message}</td></tr>`;
            }
        }

        function renderLogsTable(logs) {
            const tbody = document.getElementById('logsTableBody');
            tbody.innerHTML = '';

            if (!logs || logs.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" class="text-center text-secondary py-4">No matching task logs found.</td></tr>';
                return;
            }

            logs.forEach(log => {
                let badgeClass = 'badge-pending';
                if (log.status === 'running') badgeClass = 'badge-running';
                if (log.status === 'completed') badgeClass = 'badge-completed';
                if (log.status === 'failed') badgeClass = 'badge-failed';

                const formattedDate = log.created_at ? log.created_at.replace('T', ' ').substring(0, 19) : '-';
                const resultSnippet = log.result ? log.result.substring(0, 80) + (log.result.length > 80 ? '...' : '') : (log.status === 'running' ? 'Executing...' : '-');

                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><code>${log.task_id.substring(0, 8)}...</code></td>
                    <td class="small text-secondary">${formattedDate}</td>
                    <td><span class="badge ${badgeClass}">${(log.status || 'unknown').toUpperCase()}</span></td>
                    <td>${log.steps_completed || 0}</td>
                    <td><span class="result-truncate text-secondary" title="${log.result || ''}">${resultSnippet}</span></td>
                    <td><button class="btn btn-sm btn-outline-info" onclick="viewLogDetail('${log.task_id}')">View</button></td>
                `;
                tbody.appendChild(tr);
            });
        }

        function viewLogDetail(taskId) {
            const item = currentLogsData.find(x => x.task_id === taskId);
            if (!item) return;
            document.getElementById('detailModalTitle').textContent = `Task Log: ${item.task_id}`;
            document.getElementById('detailModalBody').textContent = JSON.stringify(item, null, 2);
            const modal = new bootstrap.Modal(document.getElementById('detailModal'));
            modal.show();
        }

        function resetFilters() {
            document.getElementById('filterStartDate').value = '';
            document.getElementById('filterEndDate').value = '';
            document.getElementById('filterStatus').value = 'all';
            loadLogsTable();
        }

        function downloadJsonLogs() {
            const startDate = document.getElementById('filterStartDate').value;
            const endDate = document.getElementById('filterEndDate').value;
            const status = document.getElementById('filterStatus').value;

            let url = '/api/v1/tasks/export?';
            const params = new URLSearchParams();
            if (startDate) params.append('start_date', startDate);
            if (endDate) params.append('end_date', endDate);
            if (status && status !== 'all') params.append('status', status);

            url += params.toString();
            window.open(url, '_blank');
        }

        // Initialize on page load
        toggleProviderFields();
        loadLogsTable();
    </script>
</body>
</html>
"""
	return HTMLResponse(content=html_content)


def start_server(host: str = "0.0.0.0", port: int = 8000) -> None:
	"""Run the uvicorn web server."""
	import uvicorn

	uvicorn.run("browser_use.server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
	port = int(os.environ.get("PORT", 8000))
	host = os.environ.get("HOST", "0.0.0.0")
	start_server(host=host, port=port)
