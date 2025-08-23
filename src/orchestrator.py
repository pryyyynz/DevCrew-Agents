"""Orchestrator for coordinating multiple agent loops."""

import os
import asyncio
import threading
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import logging

from agent_loop import AgentLoop, AgentOutput, AgentState
from communication import SharedMemoryManager, MessageBus, MessagePriority
from tools import (
    WriteFileTool,
    ReadFileTool,
    CodeExecutionTool,
    UnitTestRunnerTool,
    ArchitectureDocGeneratorTool,
    ArchitectureReviewerTool,
    DesignSystemGeneratorTool,
    CodeQualityAnalyzerTool,
    TestCaseGeneratorTool,
    RequirementsClarifierTool,
    SearchDocsTool,
    TaskManagerTool,
    KnowledgeBaseSearchTool,
    SummarizerTool,
    CritiqueTool,
    PackageInstallerTool,
    BugLoggerTool
)


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

            # Step 2: Execute all tasks until completion
            # This includes any tasks the PM might create
            self.execute_continuous_until_complete()

            # Step 3: Get final PM response
            final_response = pm_result.content

            # Check if PM created a final compilation task
            final_tasks = [task for task in self.completed_tasks.values()
                           if task.agent_id == "project_manager" and "final_response" in task.task_id]

            if final_tasks:
                # Use the most recent final compilation
                latest_final = max(final_tasks, key=lambda x: x.timestamp)
                final_response = latest_final.content

            # Store response
            self.query_responses[user_query.id] = final_response

            # Update project phase if needed
            self._check_and_update_project_phase()

            return self._format_pm_response(user_query, final_response)

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

    def execute_continuous_until_complete(self, project_inputs: Dict[str, Any] = None) -> Dict[str, AgentOutput]:
        """Execute all tasks in the queue continuously until completion."""
        self.logger.info(
            "Starting continuous execution until all tasks complete")
        results = {}
        execution_cycles = 0
        project_inputs = project_inputs or {}

        while self.task_queue and execution_cycles < self.max_execution_cycles:
            execution_cycles += 1
            self.logger.info(
                f"Execution cycle {execution_cycles}: {len(self.task_queue)} tasks in queue")

            # Get tasks that are ready to execute (dependencies satisfied)
            ready_tasks = []
            completed_task_ids = set(self.completed_tasks.keys())

            for task in self.task_queue:
                if task.id not in completed_task_ids and task.id not in self.failed_tasks:
                    # Check if all dependencies are satisfied
                    if all(dep_id in completed_task_ids for dep_id in task.dependencies):
                        ready_tasks.append(task)

            if not ready_tasks:
                # No ready tasks - check if we have unsatisfied dependencies
                remaining_tasks = [t for t in self.task_queue
                                   if t.id not in completed_task_ids and t.id not in self.failed_tasks]

                if remaining_tasks:
                    self.logger.warning(
                        f"No ready tasks but {len(remaining_tasks)} remain - possible circular dependency")
                    # Execute one task with unsatisfied dependencies to break deadlock
                    ready_tasks = remaining_tasks[:1]
                else:
                    break

            # Execute ready tasks
            cycle_results = {}
            for task in ready_tasks:
                if task.agent_type not in self.agents:
                    self.logger.error(
                        f"No agent found for type: {task.agent_type}")
                    self.failed_tasks[task.id] = f"No agent found for type: {task.agent_type}"
                    continue

                try:
                    # Execute the task
                    result = self._execute_single_task(
                        task, project_inputs, results)
                    if result:
                        cycle_results[task.id] = result
                        results[task.id] = result

                        # Check if this task completion triggers new tasks
                        self._check_for_triggered_events(result)

                        # Check for PM-created tasks in the output
                        self._check_for_pm_task_assignments(result)

                        # Remove completed task from queue
                        self.task_queue = [
                            t for t in self.task_queue if t.id != task.id]

                        self.logger.info(
                            f"Completed task {task.id} by {task.agent_type}")
                    else:
                        # Task failed, remove from queue
                        self.task_queue = [
                            t for t in self.task_queue if t.id != task.id]

                except Exception as e:
                    self.logger.error(
                        f"Failed to execute task {task.id}: {str(e)}")
                    self.failed_tasks[task.id] = str(e)
                    # Remove failed task from queue
                    self.task_queue = [
                        t for t in self.task_queue if t.id != task.id]

            # Update project phase after each cycle
            self._check_and_update_project_phase()

            # Check if project is complete
            if self.is_project_complete():
                self.logger.info("Project completed - stopping execution")
                break

            # Prevent infinite loops
            if not cycle_results and self.task_queue:
                self.logger.warning(
                    "No tasks were executed this cycle but queue is not empty")
                break

        if execution_cycles >= self.max_execution_cycles:
            self.logger.warning(
                f"Reached maximum execution cycles ({self.max_execution_cycles})")

        self.logger.info(
            f"Continuous execution completed after {execution_cycles} cycles")
        return results

    def _check_for_pm_task_assignments(self, result: AgentOutput):
        """Check if PM output contains task assignments for other agents."""
        if result.agent_id != "project_manager":
            return

        content = result.content.lower()

        # Check for explicit tool usage pattern (TaskAssignmentTool output)
        if "task assigned successfully" in content and "task id:" in content:
            # Extract task details from tool output
            lines = result.content.split('\n')
            task_id = None
            agent_type = None
            task_title = None
            task_description = None
            priority = 5
            dependencies = []

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

            # Extract description from the tool output
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

            # Create the actual orchestrator task
            if agent_type and task_id:
                new_task = OrchestratorTask(
                    id=task_id,
                    description=task_description or f"Execute {agent_type} work as assigned by Project Manager",
                    agent_type=agent_type,
                    priority=priority,
                    # Always depend on PM task that created it
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

        # Fallback: Look for task assignment patterns in PM output (natural language)
        task_patterns = [
            ("designer", ["assign.*designer",
             "designer.*task", "design.*needed", "ui.*task"]),
            ("coder", ["assign.*coder", "coder.*task",
             "implement.*task", "coding.*needed"]),
            ("tester", ["assign.*tester", "tester.*task",
             "test.*needed", "testing.*task"])
        ]

        for agent_type, patterns in task_patterns:
            if any(re.search(pattern, content) for pattern in patterns):
                # Create a follow-up task for this agent
                task_id = f"pm_assigned_{agent_type}_{int(datetime.now().timestamp())}"

                # Extract task description from PM output if possible
                lines = result.content.split('\n')
                task_description = f"Execute {agent_type} work as assigned by Project Manager"

                # Look for specific task descriptions in PM output
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

    def _analyze_pm_output_for_next_agents(self, pm_result: AgentOutput) -> List[str]:
        """Analyze PM output to determine which agents should act next."""
        content = pm_result.content.lower()
        next_agents = []

        # Keyword-based analysis
        if any(keyword in content for keyword in ["design", "ui", "ux", "interface", "user experience"]):
            next_agents.append("designer")

        if any(keyword in content for keyword in ["code", "implement", "develop", "programming", "technical"]):
            next_agents.append("coder")

        if any(keyword in content for keyword in ["test", "quality", "qa", "validation", "verification"]):
            next_agents.append("tester")

        # Default to designer if no specific keywords found
        if not next_agents and self.current_phase in [ProjectPhase.PLANNING, ProjectPhase.DESIGN]:
            next_agents.append("designer")
        elif not next_agents and self.current_phase == ProjectPhase.IMPLEMENTATION:
            next_agents.append("coder")
        elif not next_agents and self.current_phase == ProjectPhase.TESTING:
            next_agents.append("tester")

        return next_agents

    def _create_follow_up_tasks(self, user_query: UserQuery, pm_result: AgentOutput,
                                next_agents: List[str]) -> List[OrchestratorTask]:
        """Create follow-up tasks based on PM analysis."""
        tasks = []

        for i, agent_type in enumerate(next_agents):
            task = OrchestratorTask(
                id=f"{agent_type}_followup_{user_query.id}_{i}",
                description=f"Handle {agent_type} aspects of query: {user_query.query}",
                agent_type=agent_type,
                priority=8 - i,  # Decreasing priority
                dependencies=[f"pm_query_{user_query.id}"],
                inputs={
                    "user_query": user_query.query,
                    "pm_analysis": pm_result.content,
                    "context": user_query.context,
                    "current_phase": self.current_phase.value
                },
                timeout=400,
                created_from_query=user_query.id
            )
            tasks.append(task)

            # Enqueue the task to the orchestrator queue
            self.add_task(task)

        return tasks

    def _execute_with_rotation(self, tasks: List[OrchestratorTask],
                               project_inputs: Dict[str, Any]) -> Dict[str, AgentOutput]:
        """Execute tasks with intelligent agent rotation."""
        results = {}

        # Filter to only the provided tasks and sort by priority and dependencies
        task_ids = {task.id for task in tasks}
        relevant_tasks = [
            task for task in self.task_queue if task.id in task_ids]
        ordered_tasks = self._sort_tasks_by_dependencies_subset(relevant_tasks)

        for task in ordered_tasks:
            if task.agent_type not in self.agents:
                self.logger.error(
                    f"No agent found for type: {task.agent_type}")
                continue

            # Check if agent turn is appropriate
            turn_reason = self._determine_turn_reason(task)

            if self._should_allow_agent_turn(task.agent_type, turn_reason):
                try:
                    result = self._execute_single_task(
                        task, project_inputs, results)
                    if result:
                        results[task.id] = result
                        self._record_agent_turn(
                            task.agent_type, turn_reason, task.id)

                        # Check for event-triggered next steps
                        self._check_for_triggered_events(result)

                except Exception as e:
                    self.logger.error(
                        f"Failed to execute task {task.id}: {str(e)}")
                    self.failed_tasks[task.id] = str(e)
            else:
                self.logger.info(
                    f"Skipping agent turn for {task.agent_type} - rotation limits")

        return results

    def _sort_tasks_by_dependencies_subset(self, tasks: List[OrchestratorTask]) -> List[OrchestratorTask]:
        """Sort a subset of tasks by dependencies and priority."""
        # Simple topological sort for the subset
        completed = set()
        ordered = []
        remaining = tasks.copy()

        while remaining:
            ready_tasks = [
                task for task in remaining
                if all(dep in completed for dep in task.dependencies)
            ]

            if not ready_tasks:
                # Circular dependency or missing dependency within subset
                self.logger.warning(
                    "Circular or missing dependencies detected in task subset")
                ready_tasks = remaining

            # Sort by priority
            ready_tasks.sort(key=lambda x: x.priority, reverse=True)

            for task in ready_tasks:
                ordered.append(task)
                completed.add(task.id)
                remaining.remove(task)

        return ordered

    def _check_and_update_project_phase(self):
        """Check if project phase should be updated based on completed tasks."""
        current_criteria = self.project_completion_criteria.get(
            self.current_phase, [])

        # Simple completion check based on task outputs
        completed_criteria = []
        for criterion in current_criteria:
            if self._is_criterion_met(criterion):
                completed_criteria.append(criterion)

        completion_rate = len(completed_criteria) / \
            len(current_criteria) if current_criteria else 0

        if completion_rate >= 0.8:  # 80% of criteria met
            self._advance_project_phase()

    def _is_criterion_met(self, criterion: str) -> bool:
        """Check if a specific completion criterion is met."""
        # Check shared memory and completed tasks for evidence
        criterion_keywords = criterion.replace("_", " ").split()

        for task_output in self.completed_tasks.values():
            content = task_output.content.lower()
            if any(keyword in content for keyword in criterion_keywords):
                return True

        return False

    def _advance_project_phase(self):
        """Advance to the next project phase."""
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

            # Broadcast phase change
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
        """Get current project status."""
        return {
            "current_phase": self.current_phase.value,
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "active_queries": len([q for q in self.user_queries if q.id not in self.query_responses]),
            "last_agent": self.current_agent,
            "turn_history_length": len(self.agent_turn_history)
        }

    def _execute_single_task(self, task: OrchestratorTask,
                             project_inputs: Dict[str, Any] = None,
                             previous_results: Dict[str, AgentOutput] = None) -> Optional[AgentOutput]:
        """Execute a single task."""
        # Skip execution if project is already complete
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

    def _compile_query_response(self, user_query: UserQuery, pm_result: AgentOutput,
                                follow_up_results: Dict[str, AgentOutput]) -> str:
        """Compile final response to user query."""
        response_parts = [
            f"# Response to Query: {user_query.query}",
            f"**Processed at:** {datetime.now().isoformat()}",
            f"**Current Phase:** {self.current_phase.value}",
            "",
            "## Project Manager Analysis",
            pm_result.content,
            ""
        ]

        if follow_up_results:
            response_parts.append("## Specialist Responses")

            for task_id, output in follow_up_results.items():
                agent_name = output.agent_id.replace("_", " ").title()
                response_parts.extend([
                    f"### {agent_name}",
                    output.content,
                    ""
                ])

        # Add project status
        status = self._get_project_status()
        response_parts.extend([
            "## Project Status",
            f"- **Phase:** {status['current_phase']}",
            f"- **Completed Tasks:** {status['completed_tasks']}",
            f"- **Failed Tasks:** {status['failed_tasks']}",
            f"- **Last Active Agent:** {status['last_agent']}",
            ""
        ])

        return "\n".join(response_parts)

    def is_project_complete(self) -> bool:
        """Check if the project is complete."""
        return self.current_phase == ProjectPhase.COMPLETE

    def get_completion_status(self) -> Dict[str, Any]:
        """Get detailed completion status."""
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

    def execute_sequential(self, project_inputs: Dict[str, Any]) -> Dict[str, AgentOutput]:
        """Execute tasks sequentially according to dependencies."""
        self.logger.info("Starting sequential execution")
        results = {}

        # Sort tasks by dependencies and priority
        ordered_tasks = self._sort_tasks_by_dependencies()

        for task in ordered_tasks:
            if task.agent_type not in self.agents:
                self.logger.error(
                    f"No agent found for type: {task.agent_type}")
                continue

            try:
                # Prepare task with project inputs and previous results
                enriched_task = self._enrich_task(
                    task, project_inputs, results)

                # Execute task
                agent = self.agents[task.agent_type]
                self.logger.info(
                    f"Executing task {task.id} with {task.agent_type}")

                output = agent.execute(enriched_task)
                results[task.id] = output
                self.completed_tasks[task.id] = output

                # Share output with other agents
                self._share_output_globally(output)

                self.logger.info(f"Completed task {task.id}")

            except Exception as e:
                self.logger.error(
                    f"Failed to execute task {task.id}: {str(e)}")
                self.failed_tasks[task.id] = str(e)

        return results

    def execute_parallel(self, project_inputs: Dict[str, Any]) -> Dict[str, AgentOutput]:
        """Execute independent tasks in parallel."""
        self.logger.info("Starting parallel execution")
        results = {}

        # Group tasks by dependency level
        task_levels = self._group_tasks_by_level()

        for level, tasks in task_levels.items():
            self.logger.info(
                f"Executing level {level} with {len(tasks)} tasks")

            # Execute tasks at this level in parallel
            level_results = self._execute_level_parallel(
                tasks, project_inputs, results)
            results.update(level_results)

        return results

    def _execute_level_parallel(self, tasks: List[OrchestratorTask],
                                project_inputs: Dict[str, Any],
                                previous_results: Dict[str, AgentOutput]) -> Dict[str, AgentOutput]:
        """Execute a level of tasks in parallel."""
        results = {}

        with ThreadPoolExecutor(max_workers=min(len(tasks), 4)) as executor:
            # Submit all tasks
            future_to_task = {}
            for task in tasks:
                if task.agent_type in self.agents:
                    enriched_task = self._enrich_task(
                        task, project_inputs, previous_results)
                    agent = self.agents[task.agent_type]
                    future = executor.submit(agent.execute, enriched_task)
                    future_to_task[future] = task

            # Collect results
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    output = future.result(timeout=task.timeout)
                    results[task.id] = output
                    self.completed_tasks[task.id] = output
                    self._share_output_globally(output)
                    self.logger.info(f"Completed parallel task {task.id}")
                except Exception as e:
                    self.logger.error(
                        f"Failed parallel task {task.id}: {str(e)}")
                    self.failed_tasks[task.id] = str(e)

        return results

    def _sort_tasks_by_dependencies(self) -> List[OrchestratorTask]:
        """Sort tasks by dependencies and priority."""
        # Simple topological sort
        completed = set()
        ordered = []
        remaining = self.task_queue.copy()

        while remaining:
            ready_tasks = [
                task for task in remaining
                if all(dep in completed for dep in task.dependencies)
            ]

            if not ready_tasks:
                # Circular dependency or missing dependency
                self.logger.warning(
                    "Circular or missing dependencies detected")
                ready_tasks = remaining

            # Sort by priority
            ready_tasks.sort(key=lambda x: x.priority, reverse=True)

            for task in ready_tasks:
                ordered.append(task)
                completed.add(task.id)
                remaining.remove(task)

        return ordered

    def _group_tasks_by_level(self) -> Dict[int, List[OrchestratorTask]]:
        """Group tasks by dependency level for parallel execution."""
        levels = {}
        task_levels = {}

        # Calculate level for each task
        for task in self.task_queue:
            level = self._calculate_task_level(task, task_levels)
            task_levels[task.id] = level

            if level not in levels:
                levels[level] = []
            levels[level].append(task)

        return levels

    def _calculate_task_level(self, task: OrchestratorTask, memo: Dict[str, int]) -> int:
        """Calculate dependency level for a task."""
        if task.id in memo:
            return memo[task.id]

        if not task.dependencies:
            memo[task.id] = 0
            return 0

        # Find maximum level of dependencies
        max_dep_level = -1
        for dep_id in task.dependencies:
            dep_task = next(
                (t for t in self.task_queue if t.id == dep_id), None)
            if dep_task:
                dep_level = self._calculate_task_level(dep_task, memo)
                max_dep_level = max(max_dep_level, dep_level)

        level = max_dep_level + 1
        memo[task.id] = level
        return level

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

        # Add outputs from dependency tasks
        for dep_id in task.dependencies:
            if dep_id in previous_results:
                enriched['dependencies_output'][dep_id] = previous_results[dep_id].content

        return enriched

    def _share_output_globally(self, output: AgentOutput):
        """Share agent output globally for other agents to access."""
        # Store in shared memory (with proper serialization)
        self.shared_memory.set(
            f"global_output_{output.task_id}",
            output.to_dict(),
            "orchestrator"
        )

        # Broadcast completion message
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

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of execution status."""
        total_tasks = len(self.task_queue)
        completed = len(self.completed_tasks)
        failed = len(self.failed_tasks)

        return {
            'total_tasks': total_tasks,
            'completed_tasks': completed,
            'failed_tasks': failed,
            'success_rate': completed / total_tasks if total_tasks > 0 else 0,
            'agent_performance': self._get_agent_performance(),
            'execution_summary': self._generate_execution_summary()
        }

    def _get_agent_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for each agent."""
        performance = {}

        for agent_type in self.model_mapping.keys():
            # Count tasks completed by each agent
            agent_tasks = [task for task in self.completed_tasks.values()
                           if task.agent_id == agent_type]

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

    def _generate_execution_summary(self) -> str:
        """Generate human-readable execution summary."""
        summary_parts = [
            f"## Orchestrator Execution Summary",
            f"**Timestamp:** {datetime.now().isoformat()}",
            f"**Total Tasks:** {len(self.task_queue)}",
            f"**Completed:** {len(self.completed_tasks)}",
            f"**Failed:** {len(self.failed_tasks)}",
            "",
            "### Agent Models:",
        ]

        for agent_type, model in self.model_mapping.items():
            summary_parts.append(f"- **{agent_type}**: {model}")

        if self.completed_tasks:
            summary_parts.extend([
                "",
                "### Completed Tasks:",
            ])
            for task_id, output in self.completed_tasks.items():
                summary_parts.append(
                    f"- **{task_id}** ({output.agent_id}): "
                    f"{output.execution_time:.2f}s, {len(output.reasoning_chain)} reasoning steps"
                )

        if self.failed_tasks:
            summary_parts.extend([
                "",
                "### Failed Tasks:",
            ])
            for task_id, error in self.failed_tasks.items():
                summary_parts.append(f"- **{task_id}**: {error}")

        return "\n".join(summary_parts)

    def create_development_workflow(self, project_inputs: Dict[str, Any]) -> List[OrchestratorTask]:
        """Create a typical development workflow with tasks."""
        tasks = [
            OrchestratorTask(
                id="planning",
                description="Create project initiation and planning brief",
                agent_type="project_manager",
                priority=10,
                dependencies=[],
                inputs=project_inputs,
                timeout=300
            ),
            OrchestratorTask(
                id="design",
                description="Produce detailed UX & system design artifacts",
                agent_type="designer",
                priority=8,
                dependencies=["planning"],
                inputs=project_inputs,
                timeout=400
            ),
            OrchestratorTask(
                id="implementation",
                description="Convert design into implementation plan",
                agent_type="coder",
                priority=6,
                dependencies=["planning", "design"],
                inputs=project_inputs,
                timeout=500
            ),
            OrchestratorTask(
                id="testing",
                description="Produce comprehensive test strategy",
                agent_type="tester",
                priority=4,
                dependencies=["implementation"],
                inputs=project_inputs,
                timeout=300
            )
        ]

        for task in tasks:
            self.add_task(task)

        return tasks

    def execute_development_cycle(self, project_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complete development cycle."""
        self.logger.info("Starting development cycle execution")

        # Create workflow
        tasks = self.create_development_workflow(project_inputs)

        # Execute sequentially (respecting dependencies)
        results = self.execute_sequential(project_inputs)

        # Generate final summary
        summary = self.get_execution_summary()

        # Store final results (with proper serialization)
        self.shared_memory.set("final_results", {
            'results': {k: v.to_dict() for k, v in results.items()},
            'summary': summary,
            'project_inputs': project_inputs
        }, "orchestrator")

        return {
            'task_outputs': results,
            'execution_summary': summary,
            'project_inputs': project_inputs
        }

    def _check_for_triggered_events(self, result: AgentOutput):
        """Check if agent output triggers events requiring other agents."""
        content = result.content.lower()

        # Design completion might trigger coder
        if result.agent_id == "designer" and "design complete" in content:
            self._create_triggered_task(
                "coder", "Begin implementation based on completed design", result.task_id)

        # Code completion might trigger tester
        if result.agent_id == "coder" and "implementation complete" in content:
            self._create_triggered_task(
                "tester", "Create test strategy for completed implementation", result.task_id)

        # Test completion might trigger PM for review
        if result.agent_id == "tester" and "testing complete" in content:
            self._create_triggered_task(
                "project_manager", "Review completed testing and project status", result.task_id)

    def _create_triggered_task(self, agent_type: str, description: str, trigger_task_id: str):
        """Create a new task triggered by another agent's completion."""
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

    def _determine_turn_reason(self, task: OrchestratorTask) -> AgentTurnReason:
        """Determine the reason for this agent turn."""
        if task.created_from_query and not self.agent_turn_history:
            return AgentTurnReason.INITIAL_PM

        if task.dependencies:
            return AgentTurnReason.DEPENDENCY_READY

        if task.created_from_query:
            return AgentTurnReason.USER_REQUEST

        return AgentTurnReason.ROTATION_SCHEDULED

    def _should_allow_agent_turn(self, agent_type: str, reason: AgentTurnReason) -> bool:
        """Determine if agent should be allowed to take a turn."""
        # Always allow initial PM or user requests
        if reason in [AgentTurnReason.INITIAL_PM, AgentTurnReason.USER_REQUEST]:
            return True

        # Check consecutive turn limits
        if self.turn_rotation_enabled:
            recent_turns = [
                turn for turn in self.agent_turn_history[-self.max_consecutive_turns:]]
            consecutive_same_agent = all(
                turn['agent_type'] == agent_type for turn in recent_turns)

            if consecutive_same_agent and len(recent_turns) >= self.max_consecutive_turns:
                return False

        # Dependency-based turns are usually allowed
        if reason == AgentTurnReason.DEPENDENCY_READY:
            return True

        return True

    def _record_agent_turn(self, agent_type: str, reason: AgentTurnReason, task_id: str):
        """Record an agent turn for rotation tracking."""
        turn_record = {
            'agent_type': agent_type,
            'reason': reason.value,
            'task_id': task_id,
            'timestamp': datetime.now().isoformat()
        }

        self.agent_turn_history.append(turn_record)
        self.current_agent = agent_type

        # Keep only recent history
        if len(self.agent_turn_history) > 20:
            self.agent_turn_history = self.agent_turn_history[-20:]

    def _format_pm_response(self, user_query: UserQuery, pm_response: str) -> str:
        """Format the final PM response for the user."""
        return f"""# DevCrew Project Manager Response

**Query:** {user_query.query}
**Timestamp:** {datetime.now().isoformat()}
**Project Phase:** {self.current_phase.value}

## Response

{pm_response}

---
*All team coordination and specialist work has been managed by the Project Manager*"""

    def query_interactive_mode(self) -> None:
        """Interactive mode for processing user queries."""
        print("🎭 DevCrew Agents Interactive Query Mode")
        print("=" * 50)
        print("✅ All queries are processed by the Project Manager first")
        print("📋 Workflow: PM → Designer → PM → Coder → PM → Tester → PM → User")
        print("Type 'quit' or 'exit' to stop")
        print("Type 'status' to see project status")
        print("Type 'help' for more commands")
        print()

        while True:
            try:
                query = input("🤔 Your query (to Project Manager): ").strip()

                if not query:
                    continue

                if query.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break

                if query.lower() == 'status':
                    status = self.get_completion_status()
                    print(f"""
📊 **Project Status** (Managed by Project Manager)
- Phase: {status['current_phase']} ({status['phase_completion']:.1%} complete)
- Total Tasks: {status['total_tasks_completed']}
- Queries Processed: {status['user_queries_processed']}
- Project Complete: {'✅ Yes' if status['is_complete'] else '❌ No'}
                    """)
                    continue

                if query.lower() == 'help':
                    print("""
🛠️  **Available Commands**
- status: Show project status
- agents: Show agent performance
- history: Show recent agent turns
- reset: Reset project to initialization
- quit/exit: Exit interactive mode

📝 **DevCrew Workflow** (PM-Controlled)
1. 📋 PM receives and analyzes your query
2. 🎨 PM assigns design work to Designer (if needed)
3. 📋 PM reviews and forwards design
4. 👨‍💻 PM assigns coding work to Coder (if needed)  
5. 📋 PM reviews and forwards implementation
6. 🧪 PM assigns testing work to Tester (if needed)
7. 📋 PM validates and responds to you

**Example Queries:**
- "Create a project plan for a task management app"
- "Design a user login interface"
- "Implement user authentication"
- "Create tests for the API"

*Note: Only the Project Manager communicates directly with you*
                    """)
                    continue

                if query.lower() == 'agents':
                    performance = self._get_agent_performance()
                    print("\n🤖 **Agent Performance** (Coordinated by PM)")
                    for agent, metrics in performance.items():
                        print(f"- {agent}: {metrics['tasks_completed']} tasks, "
                              f"{metrics['avg_execution_time']:.1f}s avg")
                    continue

                if query.lower() == 'history':
                    print("\n📜 **Recent Agent Turns** (PM-Orchestrated)")
                    for turn in self.agent_turn_history[-5:]:
                        print(
                            f"- {turn['agent_type']} ({turn['reason']}) at {turn['timestamp']}")
                    continue

                if query.lower() == 'reset':
                    self.current_phase = ProjectPhase.INITIALIZATION
                    self.completed_tasks.clear()
                    self.failed_tasks.clear()
                    self.agent_turn_history.clear()
                    print(
                        "🔄 Project reset to initialization phase (PM will coordinate restart)")
                    continue

                # Process the query
                print(
                    f"\n🔄 Project Manager is processing your query and coordinating the team...")
                response = self.handle_user_query(query)

                print("\n" + "="*60)
                print(response)
                print("="*60 + "\n")

                # Show completion status if project is complete
                if self.is_project_complete():
                    print(
                        "🎉 **Project Complete!** All phases finished under PM coordination.")
                    break

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
