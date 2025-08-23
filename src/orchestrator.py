"""Orchestrator for coordinating multiple agent loops."""

import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from agent_loop import AgentLoop, AgentOutput, AgentState
from communication import SharedMemoryManager, MessageBus, MessagePriority


class ProjectPhase(Enum):
    """Project phases for tracking completion."""
    INITIALIZATION = "initialization"
    PLANNING = "planning"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    REVIEW = "review"
    COMPLETE = "complete"


class AgentTurnReason(Enum):
    """Reasons for agent turn selection."""
    INITIAL_PM = "initial_pm"
    DEPENDENCY_READY = "dependency_ready"
    EVENT_TRIGGERED = "event_triggered"
    ROTATION_SCHEDULED = "rotation_scheduled"
    COMPLETION_CHECK = "completion_check"
    USER_REQUEST = "user_request"


@dataclass
class UserQuery:
    """User query/request structure."""
    id: str
    query: str
    timestamp: str
    priority: int = 5
    context: Dict[str, Any] = None
    requires_pm_first: bool = True


@dataclass
class OrchestratorTask:
    """Task for the orchestrator to assign to agents."""
    id: str
    description: str
    agent_type: str
    priority: int
    dependencies: List[str]
    inputs: Dict[str, Any]
    timeout: int = 300
    created_from_query: str = None  # Track which user query created this task


class AgentOrchestrator:
    """Orchestrates multiple agents with different models in a sandbox environment."""

    def __init__(self):
        self.agents = {}
        self.model_mapping = self._load_model_mapping()
        self.task_queue = []
        self.completed_tasks = {}
        self.failed_tasks = {}

        # User query management
        self.user_queries = []
        self.query_responses = {}

        # Project state tracking
        self.current_phase = ProjectPhase.INITIALIZATION
        self.project_completion_criteria = {}
        self.phase_outputs = {}

        # Agent turn management
        self.current_agent = None
        self.agent_turn_history = []
        self.turn_rotation_enabled = True
        self.max_consecutive_turns = 3

        # Sandbox code management
        self.last_published_code_id = None
        self.force_sandbox_clear = True

        # Global communication
        self.shared_memory = SharedMemoryManager("orchestrator_memory.db")
        self.message_bus = MessageBus("orchestrator_messages.db")

        # Task execution control
        self.continuous_execution = True
        self.max_execution_cycles = 50  # Prevent infinite loops

        # Setup logging
        self.logger = self._setup_logging()

        # Initialize agents
        self._initialize_agents()

        # Initialize project completion criteria
        self._initialize_completion_criteria()

    def _load_model_mapping(self) -> Dict[str, str]:
        """Load model assignments for each agent type."""
        return {
            # Project Manager: Needs strategic thinking and coordination
            'project_manager': os.getenv('PROJECT_MANAGER_MODEL', 'groq/llama-3.3-70b-versatile'),
            # Designer: Benefits from creative and analytical thinking
            'designer': os.getenv('DESIGNER_MODEL', 'groq/llama-3.1-70b-versatile'),
            # Coder: Needs deep technical reasoning and code generation
            'coder': os.getenv('CODER_MODEL', 'groq/deepseek-r1-distill-llama-70b'),
            # Tester: Requires systematic and thorough analysis
            'tester': os.getenv('TESTER_MODEL', 'groq/llama-3.3-70b-versatile')
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup orchestrator logging."""
        logger = logging.getLogger("orchestrator")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.FileHandler("orchestrator.log")
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _initialize_agents(self):
        """Initialize all agent loops with their specific models."""
        for agent_type, model in self.model_mapping.items():
            try:
                self.agents[agent_type] = AgentLoop(
                    agent_id=agent_type,
                    model_name=model,
                    tools=self._get_tools_for_agent(agent_type)
                )
                self.logger.info(
                    f"Initialized {agent_type} with model {model}")
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize {agent_type}: {str(e)}")

    def _get_tools_for_agent(self, agent_type: str) -> List:
        """Get tools for a specific agent type optimized for DevCrew coordination."""
        from tools.custom_tool import (
            RequirementsClarifierTool, ArchitectureReviewerTool, TaskManagerTool,
            TaskAssignmentTool, KnowledgeBaseSearchTool, SummarizerTool,
            ArchitectureDocGeneratorTool, DesignSystemGeneratorTool, CritiqueTool,
            WriteFileTool, ReadFileTool, CodeExecutionTool, CodeQualityAnalyzerTool,
            PackageInstallerTool, SearchDocsTool, UnitTestRunnerTool,
            TestCaseGeneratorTool, BugLoggerTool, SandboxCodeTool
        )
        from communication.communication_tools import (
            SharedMemoryTool, MessagePassingTool, KnowledgeStoreTool, TeamCommunicationTool
        )

        # Core communication tools for all agents
        communication_tools = [
            SharedMemoryTool(),
            MessagePassingTool(),
            KnowledgeStoreTool(),
            TeamCommunicationTool()
        ]

        # Common tools for basic functionality
        common_tools = [RequirementsClarifierTool()] + communication_tools

        if agent_type == "project_manager":
            return common_tools + [
                ArchitectureReviewerTool(),
                TaskManagerTool(),
                TaskAssignmentTool(),  # Critical for task delegation
                KnowledgeBaseSearchTool(),
                SummarizerTool(),
                ReadFileTool(),  # For reviewing project artifacts
            ]
        elif agent_type == "designer":
            return common_tools + [
                DesignSystemGeneratorTool(),
                ArchitectureReviewerTool(),
                ArchitectureDocGeneratorTool(),
                CritiqueTool(),
                ReadFileTool(),  # For reviewing design specs
                WriteFileTool()  # For creating design documentation
            ]
        elif agent_type == "coder":
            return common_tools + [
                WriteFileTool(),
                ReadFileTool(),
                CodeExecutionTool(),
                CodeQualityAnalyzerTool(),
                PackageInstallerTool(),
                SearchDocsTool(),
                SandboxCodeTool(),  # Critical for prototyping
                ArchitectureReviewerTool()
            ]
        elif agent_type == "tester":
            return common_tools + [
                TestCaseGeneratorTool(),
                UnitTestRunnerTool(),
                CodeQualityAnalyzerTool(),
                BugLoggerTool(),
                ReadFileTool(),  # For reviewing code and specs
                WriteFileTool()  # For test documentation
            ]

        return common_tools

    def _initialize_completion_criteria(self):
        """Initialize project completion criteria for each phase."""
        self.project_completion_criteria = {
            ProjectPhase.PLANNING: [
                "project_requirements_defined",
                "success_metrics_established",
                "feature_priorities_set",
                "risks_identified"
            ],
            ProjectPhase.DESIGN: [
                "user_journey_mapped",
                "component_inventory_created",
                "data_model_defined",
                "api_endpoints_specified"
            ],
            ProjectPhase.IMPLEMENTATION: [
                "architecture_finalized",
                "coding_standards_defined",
                "implementation_plan_created",
                "ci_cd_outlined"
            ],
            ProjectPhase.TESTING: [
                "test_strategy_defined",
                "test_cases_created",
                "automation_plan_ready",
                "exit_criteria_established"
            ],
            ProjectPhase.COMPLETE: [
                "all_phases_completed",
                "deliverables_validated",
                "stakeholder_approval_received"
            ]
        }

    def add_task(self, task: OrchestratorTask) -> str:
        """Add a task to the orchestrator queue."""
        self.task_queue.append(task)
        self.logger.info(f"Added task {task.id} for {task.agent_type}")

        # Store task in shared memory for transparency
        self.shared_memory.set(
            f"task_{task.id}",
            asdict(task),
            "orchestrator"
        )

        return task.id

    def handle_user_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """Main entry point for user queries - PM acts first."""
        query_id = f"query_{int(datetime.now().timestamp())}"

        user_query = UserQuery(
            id=query_id,
            query=query,
            timestamp=datetime.now().isoformat(),
            context=context or {},
            requires_pm_first=self._requires_pm_first(query)
        )

        self.user_queries.append(user_query)
        self.logger.info(f"Received user query {query_id}: {query[:100]}...")

        # Store in shared memory for agent access
        self.shared_memory.set(f"user_query_{query_id}", {
            "query": query,
            "context": context,
            "timestamp": user_query.timestamp,
            "status": "processing"
        }, "orchestrator")

        # Process the query
        return self._process_user_query(user_query)

    def _requires_pm_first(self, query: str) -> bool:
        """Determine if query requires PM to act first."""
        # ALL queries now require PM first for proper workflow
        return True

    def _process_user_query(self, user_query: UserQuery) -> str:
        """Process user query through agent orchestration - PM ALWAYS acts first."""
        try:
            # Step 1: PM ALWAYS processes the query first
            pm_task = self._create_pm_query_task(user_query)
            pm_result = self._execute_single_task(pm_task)

            if not pm_result:
                return f"❌ Failed to process query with Project Manager"

            # NEW: Create a PM-controlled workflow for all roles based on this query
            self._create_pm_controlled_workflow(user_query, pm_result)

            # Step 2: Execute all tasks until completion
            # This includes any tasks the PM might create
            self.execute_continuous_until_complete()

            # Step 3: Determine final status and concise response
            final_response = pm_result.content
            is_validated = False

            # Prefer PM validation output if available
            pm_validation_outputs = [output for output in self.completed_tasks.values()
                                     if getattr(output, 'agent_id', '') == 'project_manager' and
                                     'pm_validate' in getattr(output, 'task_id', '')]
            if pm_validation_outputs:
                latest_validation = max(
                    pm_validation_outputs, key=lambda x: getattr(x, 'timestamp', ''))
                final_response = latest_validation.content
                is_validated = True
            else:
                # Check if PM created a final compilation task
                final_tasks = [task for task in self.completed_tasks.values()
                               if task.agent_id == "project_manager" and "final_response" in task.task_id]
                if final_tasks:
                    latest_final = max(final_tasks, key=lambda x: x.timestamp)
                    final_response = latest_final.content

            # Store response
            self.query_responses[user_query.id] = final_response

            # Update project phase if needed
            self._check_and_update_project_phase()

            # Return concise confirmation to the user
            return self._format_pm_response(user_query, final_response, is_validated)

        except Exception as e:
            error_msg = f"❌ Error processing user query: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    def _create_pm_query_task(self, user_query: UserQuery) -> OrchestratorTask:
        """Create initial PM task for query processing."""
        return OrchestratorTask(
            id=f"pm_query_{user_query.id}",
            description=f"Analyze and respond to user query: {user_query.query}",
            agent_type="project_manager",
            priority=10,
            dependencies=[],
            inputs={
                "user_query": user_query.query,
                "context": user_query.context,
                "current_phase": self.current_phase.value,
                "project_status": self._get_project_status()
            },
            timeout=300,
            created_from_query=user_query.id
        )

    def _format_pm_response(self, user_query: UserQuery, pm_response: str, is_validated: bool = False) -> str:
        """Return a concise confirmation response for the user without internal details."""
        query_text = user_query.query.strip()
        if is_validated:
            return (
                f"Request completed successfully.\n"
                f"- Query: {query_text}\n"
                f"If you need changes, reply with the specifics."
            )
        else:
            return (
                f"Request received. The team is working on it.\n"
                f"- Query: {query_text}\n"
                f"I'll confirm once it's complete."
            )

    def _create_pm_controlled_workflow(self, user_query: UserQuery, pm_result):
        """Create and enqueue a PM-controlled end-to-end workflow for this query."""
        try:
            base_id = user_query.id
            pm_query_task_id = f"pm_query_{base_id}"

            tasks = [
                OrchestratorTask(
                    id=f"design_{base_id}",
                    description=f"Create architecture and UX design for query: {user_query.query}",
                    agent_type="designer",
                    priority=9,
                    dependencies=[pm_query_task_id],
                    inputs={
                        "user_query": user_query.query,
                        "pm_analysis": pm_result.content,
                        "context": user_query.context,
                        "current_phase": self.current_phase.value
                    },
                    timeout=400,
                    created_from_query=user_query.id
                ),
                OrchestratorTask(
                    id=f"pm_forward_design_{base_id}",
                    description="Review design and forward to implementation",
                    agent_type="project_manager",
                    priority=8,
                    dependencies=[f"design_{base_id}"],
                    inputs={
                        "action": "forward_design",
                        "current_phase": self.current_phase.value
                    },
                    timeout=300,
                    created_from_query=user_query.id
                ),
                OrchestratorTask(
                    id=f"implementation_{base_id}",
                    description="Implement based on approved design",
                    agent_type="coder",
                    priority=7,
                    dependencies=[f"pm_forward_design_{base_id}"],
                    inputs={
                        "user_query": user_query.query,
                        "context": user_query.context,
                        "current_phase": self.current_phase.value
                    },
                    timeout=500,
                    created_from_query=user_query.id
                ),
                OrchestratorTask(
                    id=f"pm_forward_implementation_{base_id}",
                    description="Review implementation and forward to testing",
                    agent_type="project_manager",
                    priority=6,
                    dependencies=[f"implementation_{base_id}"],
                    inputs={
                        "action": "forward_implementation",
                        "current_phase": self.current_phase.value
                    },
                    timeout=300,
                    created_from_query=user_query.id
                ),
                OrchestratorTask(
                    id=f"testing_{base_id}",
                    description="Test implementation and report findings",
                    agent_type="tester",
                    priority=5,
                    dependencies=[f"pm_forward_implementation_{base_id}"],
                    inputs={
                        "user_query": user_query.query,
                        "context": user_query.context,
                        "current_phase": self.current_phase.value
                    },
                    timeout=300,
                    created_from_query=user_query.id
                ),
                OrchestratorTask(
                    id=f"pm_validate_{base_id}",
                    description="Validate project results and compile final response",
                    agent_type="project_manager",
                    priority=10,
                    dependencies=[f"testing_{base_id}"],
                    inputs={
                        "action": "final_validation",
                        "current_phase": self.current_phase.value
                    },
                    timeout=300,
                    created_from_query=user_query.id
                ),
            ]

            for t in tasks:
                self.add_task(t)

        except Exception as e:
            self.logger.error(f"Failed to create PM-controlled workflow: {e}")

    def execute_continuous_until_complete(self, project_inputs: Dict[str, Any] = None) -> Dict[str, AgentOutput]:
        """Execute all tasks in the queue continuously until completion."""
        self.logger.info(
            "Starting continuous execution until all tasks complete")
        results: Dict[str, AgentOutput] = {}
        execution_cycles = 0
        project_inputs = project_inputs or {}

        while self.task_queue and execution_cycles < self.max_execution_cycles:
            execution_cycles += 1
            self.logger.info(
                f"Execution cycle {execution_cycles}: {len(self.task_queue)} tasks in queue")

            ready_tasks = []
            completed_task_ids = set(self.completed_tasks.keys())

            for task in self.task_queue:
                if task.id not in completed_task_ids and task.id not in self.failed_tasks:
                    if all(dep_id in completed_task_ids for dep_id in task.dependencies):
                        ready_tasks.append(task)

            if not ready_tasks:
                remaining_tasks = [
                    t for t in self.task_queue if t.id not in completed_task_ids and t.id not in self.failed_tasks]
                if remaining_tasks:
                    self.logger.warning(
                        "No ready tasks but some remain - possible circular dependency")
                    ready_tasks = remaining_tasks[:1]
                else:
                    break

            cycle_results = {}
            for task in ready_tasks:
                if task.agent_type not in self.agents:
                    self.logger.error(
                        f"No agent found for type: {task.agent_type}")
                    self.failed_tasks[task.id] = f"No agent found for type: {task.agent_type}"
                    continue

                try:
                    result = self._execute_single_task(
                        task, project_inputs, results)
                    if result:
                        cycle_results[task.id] = result
                        results[task.id] = result
                        self._check_for_triggered_events(result)
                        self._check_for_pm_task_assignments(result)
                        self.task_queue = [
                            t for t in self.task_queue if t.id != task.id]
                        self.logger.info(
                            f"Completed task {task.id} by {task.agent_type}")
                    else:
                        self.task_queue = [
                            t for t in self.task_queue if t.id != task.id]
                except Exception as e:
                    self.logger.error(
                        f"Failed to execute task {task.id}: {str(e)}")
                    self.failed_tasks[task.id] = str(e)
                    self.task_queue = [
                        t for t in self.task_queue if t.id != task.id]

            self._check_and_update_project_phase()

            if self.is_project_complete():
                self.logger.info("Project completed - stopping execution")
                break

            if not cycle_results and self.task_queue:
                self.logger.warning(
                    "No tasks executed this cycle but queue not empty")
                break

        if execution_cycles >= self.max_execution_cycles:
            self.logger.warning(
                f"Reached maximum execution cycles ({self.max_execution_cycles})")

        self.logger.info(
            f"Continuous execution completed after {execution_cycles} cycles")
        return results

    def _execute_single_task(self, task: OrchestratorTask,
                             project_inputs: Dict[str, Any] = None,
                             previous_results: Dict[str, AgentOutput] = None) -> Optional[AgentOutput]:
        """Execute a single task."""
        if self.is_project_complete():
            self.logger.info(
                f"Skipping task {task.id} - project already complete")
            return None

        if task.agent_type not in self.agents:
            return None

        try:
            enriched_task = self._enrich_task(
                task, project_inputs or {}, previous_results or {})
            agent = self.agents[task.agent_type]
            self.logger.info(
                f"Executing task {task.id} with {task.agent_type}")
            output = agent.execute(enriched_task)
            self.completed_tasks[task.id] = output
            self._share_output_globally(output)
            return output
        except Exception as e:
            self.logger.error(f"Failed to execute task {task.id}: {str(e)}")
            self.failed_tasks[task.id] = str(e)
            return None

    def _enrich_task(self, task: OrchestratorTask,
                     project_inputs: Dict[str, Any],
                     previous_results: Dict[str, AgentOutput]) -> Dict[str, Any]:
        """Enrich task with context and dependencies."""
        enriched = {
            'id': task.id,
            'description': task.description,
            'agent_type': task.agent_type,
            'inputs': task.inputs.copy(),
            'project_context': project_inputs,
            'dependencies_output': {}
        }
        for dep_id in task.dependencies:
            if dep_id in previous_results:
                enriched['dependencies_output'][dep_id] = previous_results[dep_id].content
        return enriched

    def _share_output_globally(self, output: AgentOutput):
        """Share agent output globally for other agents to access."""
        self.shared_memory.set(
            f"global_output_{output.task_id}",
            output.to_dict(),
            "orchestrator"
        )
        self.message_bus.broadcast_message(
            sender_id="orchestrator",
            channel="task_completions",
            content={
                "task_id": output.task_id,
                "agent_id": output.agent_id,
                "status": output.final_state.value,
                "execution_time": output.execution_time
            },
            subject=f"Task {output.task_id} completed",
            priority=MessagePriority.NORMAL
        )

        # New: Try to publish any code in this output to the web sandbox
        self._try_publish_output_to_sandbox(output)

    def _try_publish_output_to_sandbox(self, output: AgentOutput):
        """If the agent output contains a code block, publish it to the sandbox for the web UI."""
        try:
            # Prefer code from coder or designer
            if output.agent_id not in ("coder", "designer"):
                return

            code_info = self._extract_code_block(output.content)
            if not code_info:
                return

            language, code = code_info

            # Generate a unique ID for this code
            code_id = f"generated_code_{int(datetime.now().timestamp())}"

            # Build code metadata
            code_data = {
                "code": code,
                "language": language,
                "filename": None,
                "description": f"Generated by {output.agent_id} for task {output.task_id}",
                "timestamp": datetime.now().isoformat(),
                "agent": output.agent_id,
                "force_clear": self.force_sandbox_clear,
                "code_id": code_id
            }

            # Store into sandbox shared memory so frontend can load it
            from communication.memory_manager import SharedMemoryManager as _SMM
            smm = _SMM("sandbox_code.db")

            # Save with unique ID
            smm.set(code_id, code_data, "orchestrator")

            # Also update the latest code for easy access
            smm.set("latest_generated_code", code_data, "orchestrator")

            # Track that we've published this code to avoid duplicate notifications
            self.last_published_code_id = code_id

            self.logger.info(
                f"Published {language} code from {output.agent_id} to sandbox for task {output.task_id}")
        except Exception as e:
            self.logger.warning(f"Failed to publish code to sandbox: {e}")

    def _extract_code_block(self, text: str) -> Optional[Tuple[str, str]]:
        """Extract the largest fenced code block (```lang ... ```) and return (language, code)."""
        try:
            pattern = re.compile(r"```(\w+)?\n([\s\S]*?)```", re.MULTILINE)
            matches = list(pattern.finditer(text or ""))
            if not matches:
                return None
            # Pick the largest block by content length
            best = max(matches, key=lambda m: len(m.group(2) or ""))
            lang = (best.group(1) or "python").lower().strip()
            code = best.group(2) or ""
            # Normalize language
            if lang in ("py", "python3"):
                lang = "python"
            if lang in ("js", "node", "nodejs"):
                lang = "javascript"
            if lang not in ("python", "javascript", "html"):
                # Default to python for unsupported languages
                lang = "python"
            return (lang, code)
        except Exception:
            return None

    def _check_for_pm_task_assignments(self, result: AgentOutput):
        """Check if PM output contains task assignments for other agents."""
        if result.agent_id != "project_manager":
            return
        content = result.content.lower()
        if "task assigned successfully" in content and "task id:" in content:
            lines = result.content.split('\n')
            task_id = None
            agent_type = None
            task_title = None
            task_description = None
            priority = 5
            dependencies: List[str] = []
            for line in lines:
                line = line.strip()
                if line.startswith("**Task ID**:"):
                    task_id = line.split(":", 1)[1].strip()
                elif line.startswith("**Assigned to**:"):
                    agent_info = line.split(":", 1)[1].strip().lower()
                    if "designer" in agent_info:
                        agent_type = "designer"
                    elif "coder" in agent_info:
                        agent_type = "coder"
                    elif "tester" in agent_info:
                        agent_type = "tester"
                elif line.startswith("**Title**:"):
                    task_title = line.split(":", 1)[1].strip()
                elif line.startswith("**Priority**:"):
                    priority_str = line.split(":", 1)[1].strip().split("/")[0]
                    try:
                        priority = int(priority_str)
                    except:
                        priority = 5
                elif line.startswith("**Dependencies**:"):
                    deps_str = line.split(":", 1)[1].strip()
                    if deps_str and deps_str != "None":
                        dependencies = [dep.strip()
                                        for dep in deps_str.split(",")]
            desc_start = False
            description_lines = []
            for line in lines:
                if line.strip().startswith("**Description**:"):
                    desc_start = True
                    continue
                elif desc_start and line.strip().startswith("The task has been"):
                    break
                elif desc_start:
                    description_lines.append(line)
            if description_lines:
                task_description = "\n".join(description_lines).strip()
            if agent_type and task_id:
                new_task = OrchestratorTask(
                    id=task_id,
                    description=task_description or f"Execute {agent_type} work as assigned by Project Manager",
                    agent_type=agent_type,
                    priority=priority,
                    dependencies=dependencies + [result.task_id],
                    inputs={
                        "pm_assignment": result.content,
                        "assigned_by": "project_manager",
                        "current_phase": self.current_phase.value,
                        "task_title": task_title
                    },
                    timeout=400
                )
                self.add_task(new_task)
                self.logger.info(
                    f"PM created task via tool: {task_id} for {agent_type}")
                return
        # Fallback: natural language patterns
        import re as _re
        task_patterns = [
            ("designer", ["assign.*designer",
             "designer.*task", "design.*needed", "ui.*task"]),
            ("coder", ["assign.*coder", "coder.*task",
             "implement.*task", "coding.*needed"]),
            ("tester", ["assign.*tester", "tester.*task",
             "test.*needed", "testing.*task"])
        ]
        for agent_type, patterns in task_patterns:
            if any(_re.search(pattern, content) for pattern in patterns):
                task_id = f"pm_assigned_{agent_type}_{int(datetime.now().timestamp())}"
                lines = result.content.split('\n')
                task_description = f"Execute {agent_type} work as assigned by Project Manager"
                for line in lines:
                    if agent_type in line.lower() and any(word in line.lower() for word in ['task', 'work', 'need', 'should']):
                        task_description = line.strip()
                        break
                new_task = OrchestratorTask(
                    id=task_id,
                    description=task_description,
                    agent_type=agent_type,
                    priority=6,
                    dependencies=[result.task_id],
                    inputs={
                        "pm_assignment": result.content,
                        "assigned_by": "project_manager",
                        "current_phase": self.current_phase.value
                    },
                    timeout=400
                )
                self.add_task(new_task)
                self.logger.info(
                    f"PM assigned new task to {agent_type}: {task_id}")

    def _check_for_triggered_events(self, result: AgentOutput):
        """Create follow-up tasks based on certain outputs."""
        content = result.content.lower()
        if result.agent_id == "designer" and "design complete" in content:
            self._create_triggered_task(
                "coder", "Begin implementation based on completed design", result.task_id)
        if result.agent_id == "coder" and "implementation complete" in content:
            self._create_triggered_task(
                "tester", "Create test strategy for completed implementation", result.task_id)
        if result.agent_id == "tester" and "testing complete" in content:
            self._create_triggered_task(
                "project_manager", "Review completed testing and project status", result.task_id)

    def _create_triggered_task(self, agent_type: str, description: str, trigger_task_id: str):
        task = OrchestratorTask(
            id=f"triggered_{agent_type}_{int(datetime.now().timestamp())}",
            description=description,
            agent_type=agent_type,
            priority=7,
            dependencies=[trigger_task_id],
            inputs={"triggered_by": trigger_task_id,
                    "current_phase": self.current_phase.value},
            timeout=300
        )
        self.add_task(task)
        self.logger.info(f"Created triggered task for {agent_type}: {task.id}")

    def _check_and_update_project_phase(self):
        """Check if project phase should be updated based on completed tasks."""
        current_criteria = self.project_completion_criteria.get(
            self.current_phase, [])
        completed_criteria = []
        for criterion in current_criteria:
            if self._is_criterion_met(criterion):
                completed_criteria.append(criterion)
        completion_rate = len(completed_criteria) / \
            len(current_criteria) if current_criteria else 0
        if completion_rate >= 0.8:
            self._advance_project_phase()

    def _is_criterion_met(self, criterion: str) -> bool:
        """Check if a specific completion criterion is met."""
        criterion_keywords = criterion.replace("_", " ").split()
        for task_output in self.completed_tasks.values():
            content = task_output.content.lower()
            if any(keyword in content for keyword in criterion_keywords):
                return True
        return False

    def _advance_project_phase(self):
        """Advance to the next project phase and broadcast update."""
        phase_order = [
            ProjectPhase.INITIALIZATION,
            ProjectPhase.PLANNING,
            ProjectPhase.DESIGN,
            ProjectPhase.IMPLEMENTATION,
            ProjectPhase.TESTING,
            ProjectPhase.REVIEW,
            ProjectPhase.COMPLETE
        ]
        current_index = phase_order.index(self.current_phase)
        if current_index < len(phase_order) - 1:
            old_phase = self.current_phase
            self.current_phase = phase_order[current_index + 1]
            self.logger.info(
                f"Advanced project phase: {old_phase.value} → {self.current_phase.value}")
            self.message_bus.broadcast_message(
                sender_id="orchestrator",
                channel="project_updates",
                content={
                    "event": "phase_change",
                    "old_phase": old_phase.value,
                    "new_phase": self.current_phase.value,
                    "timestamp": datetime.now().isoformat()
                },
                subject="Project Phase Advanced",
                priority=MessagePriority.HIGH
            )

    def _get_project_status(self) -> Dict[str, Any]:
        """Get current project status for external APIs."""
        return {
            "current_phase": self.current_phase.value,
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "active_queries": len([q for q in self.user_queries if q.id not in self.query_responses]),
            "last_agent": self.current_agent,
            "turn_history_length": len(self.agent_turn_history)
        }

    def get_completion_status(self) -> Dict[str, Any]:
        """Get detailed completion status for the current phase."""
        current_criteria = self.project_completion_criteria.get(
            self.current_phase, [])
        met_criteria = [
            c for c in current_criteria if self._is_criterion_met(c)]
        return {
            "current_phase": self.current_phase.value,
            "is_complete": self.is_project_complete(),
            "phase_completion": len(met_criteria) / len(current_criteria) if current_criteria else 0,
            "met_criteria": met_criteria,
            "remaining_criteria": [c for c in current_criteria if c not in met_criteria],
            "total_tasks_completed": len(self.completed_tasks),
            "user_queries_processed": len(self.query_responses)
        }

    def is_project_complete(self) -> bool:
        """Check if the project is complete."""
        return self.current_phase == ProjectPhase.COMPLETE

    def _get_agent_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for each agent."""
        performance: Dict[str, Dict[str, Any]] = {}
        for agent_type in self.model_mapping.keys():
            agent_tasks = [
                task for task in self.completed_tasks.values() if task.agent_id == agent_type]
            if agent_tasks:
                avg_time = sum(
                    task.execution_time for task in agent_tasks) / len(agent_tasks)
                avg_reasoning = sum(len(task.reasoning_chain)
                                    for task in agent_tasks) / len(agent_tasks)
                avg_actions = sum(len(task.actions_taken)
                                  for task in agent_tasks) / len(agent_tasks)
                performance[agent_type] = {
                    'tasks_completed': len(agent_tasks),
                    'avg_execution_time': avg_time,
                    'avg_reasoning_steps': avg_reasoning,
                    'avg_actions_taken': avg_actions,
                    'model': self.model_mapping.get(agent_type, 'unknown')
                }
            else:
                performance[agent_type] = {
                    'tasks_completed': 0,
                    'avg_execution_time': 0,
                    'avg_reasoning_steps': 0,
                    'avg_actions_taken': 0,
                    'model': self.model_mapping.get(agent_type, 'unknown')
                }
        return performance
