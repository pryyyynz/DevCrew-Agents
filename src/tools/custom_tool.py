from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


# 1. Requirements Clarifier


class RequirementsClarifierInput(BaseModel):
    raw_requirements: str = Field(
        ..., description="Unstructured or initial requirement statements.")
    constraints: str | None = Field(
        None, description="Known constraints: budget, latency, compliance, etc.")


class RequirementsClarifierTool(BaseTool):
    name: str = "requirements_clarifier"
    description: str = (
        "Refines raw, ambiguous requirements into SMART, testable, prioritized user stories with acceptance criteria."
    )
    args_schema: Type[BaseModel] = RequirementsClarifierInput

    def _run(self, raw_requirements: str, constraints: str | None = None) -> str:
        constraints_section = f"Constraints Provided: {constraints}\n" if constraints else ""
        return (
            "## Refined Requirements\n" +
            constraints_section +
            "### Normalized User Stories\n" +
            "- (Sample) As a user, I want ... so that ...\n" +
            "### Acceptance Criteria Template\n" +
            "- Given ..., When ..., Then ...\n" +
            "### Prioritization (MoSCoW)\n" +
            "- Must: ...\n- Should: ...\n- Could: ...\n- Won't (this iteration): ...\n"
        )

# 2. Architecture Reviewer


class ArchitectureReviewerInput(BaseModel):
    architecture_outline: str = Field(
        ..., description="Proposed architecture summary / diagram text.")
    tech_stack: str = Field(...,
                            description="Technologies in play (languages, frameworks, infra).")


class ArchitectureReviewerTool(BaseTool):
    name: str = "architecture_reviewer"
    description: str = (
        "Analyzes proposed architecture for scalability, cohesion, coupling, risk hot-spots, and suggests improvements."
    )
    args_schema: Type[BaseModel] = ArchitectureReviewerInput

    def _run(self, architecture_outline: str, tech_stack: str) -> str:
        return (
            "## Architecture Review\n" +
            f"### Stack\n{tech_stack}\n" +
            "### Observations\n- Layering: ...\n- Data flow: ...\n- Boundaries: ...\n" +
            "### Risks & Mitigations\n- Risk: ... -> Mitigation: ...\n" +
            "### Improvement Suggestions\n- Suggestion: ... Rationale: ... Impact: ...\n"
        )

# 3. Design System Generator


class DesignSystemGeneratorInput(BaseModel):
    component_inventory: str = Field(
        ..., description="List of components and short purpose notes.")
    design_principles: str = Field(
        ..., description="Guiding principles (clarity, accessibility, etc.)")


class DesignSystemGeneratorTool(BaseTool):
    name: str = "design_system_generator"
    description: str = (
        "Generates foundational design system tokens and component spec skeletons from an inventory and principles."
    )
    args_schema: Type[BaseModel] = DesignSystemGeneratorInput

    def _run(self, component_inventory: str, design_principles: str) -> str:
        return (
            "## Design System Draft\n" +
            f"### Principles\n{design_principles}\n" +
            "### Core Tokens\n" +
            "- Color Palette: primary / secondary / neutral / semantic\n" +
            "- Spacing Scale: 4 8 12 16 24 32 48\n" +
            "- Typography: Display / H1 / H2 / Body / Caption\n" +
            "- Radius: 2 4 8\n" +
            "### Components\n" +
            "| Component | Purpose | Props | States | A11y Notes |\n|-----------|---------|-------|--------|------------|\n| Button | Triggers actions | variant,size,disabled | hover,focus,loading | ARIA-label for icon variant |\n" +
            "### Inventory Reference\n" + component_inventory + "\n"
        )

# 4. Code Quality Analyzer


class CodeQualityAnalyzerInput(BaseModel):
    code_snippet: str = Field(..., description="A code excerpt to analyze.")
    language: str = Field(...,
                          description="Programming language of the snippet.")


class CodeQualityAnalyzerTool(BaseTool):
    name: str = "code_quality_analyzer"
    description: str = (
        "Performs static reasoning over a code snippet: readability, complexity, naming, potential bugs, testability."
    )
    args_schema: Type[BaseModel] = CodeQualityAnalyzerInput

    def _run(self, code_snippet: str, language: str) -> str:
        lines = len(code_snippet.splitlines())
        return (
            "## Code Quality Review\n" +
            f"Language: {language}\nLine Count: {lines}\n" +
            "### Findings\n- Readability: ...\n- Complexity: ...\n- Naming: ...\n- Error Handling: ...\n" +
            "### Potential Issues\n- Issue: ... Impact: ... Recommendation: ...\n" +
            "### Suggested Refactor\n```" +
            language.lower() + "\n# example refactor skeleton\n```\n"
        )

# 5. Test Case Generator


class TestCaseGeneratorInput(BaseModel):
    feature_description: str = Field(
        ..., description="Short description of the feature or capability.")
    risk_level: str = Field(...,
                            description="Perceived risk (low/medium/high).")


class TestCaseGeneratorTool(BaseTool):
    name: str = "test_case_generator"
    description: str = (
        "Generates layered test cases (unit/integration/e2e) with scenario outlines and acceptance criteria."
    )
    args_schema: Type[BaseModel] = TestCaseGeneratorInput

    def _run(self, feature_description: str, risk_level: str) -> str:
        return (
            "## Test Case Set\n" +
            f"Feature: {feature_description}\nRisk: {risk_level}\n" +
            "### Unit Tests\n- should_<behavior> given <condition> -> expect <result>\n" +
            "### Integration Tests\n- scenario: ... preconditions: ... steps: ... expected: ...\n" +
            "### E2E Tests\n- Name | Goal | Steps | Expected\n" +
            "### Edge Cases\n- Input: ... Expected Handling: ...\n" +
            "### Negative Cases\n- Invalid <x> -> error <y>\n"
        )

# ------------------------- Additional Specialized Tools ------------------------------ #

# Project Manager Tools


class TaskManagerInput(BaseModel):
    action: str = Field(...,
                        description="Action: create, assign, update, list, complete")
    task_data: str = Field(...,
                           description="Task details in JSON format or task ID for updates")


class TaskManagerTool(BaseTool):
    name: str = "task_manager"
    description: str = (
        "Create, assign, update and track project tasks. Supports CRUD operations on tasks with SQLite backend."
    )
    args_schema: Type[BaseModel] = TaskManagerInput

    def _run(self, action: str, task_data: str) -> str:
        import json
        import sqlite3
        from datetime import datetime

        # Simple SQLite-backed task management
        try:
            # Initialize DB if needed
            conn = sqlite3.connect('project_tasks.db')
            conn.execute('''CREATE TABLE IF NOT EXISTS tasks 
                           (id INTEGER PRIMARY KEY, title TEXT, assignee TEXT, 
                            status TEXT, priority TEXT, created_at TEXT, updated_at TEXT)''')

            if action == "create":
                data = json.loads(task_data)
                now = datetime.now().isoformat()
                conn.execute("INSERT INTO tasks (title, assignee, status, priority, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                             (data.get('title'), data.get('assignee'), 'open', data.get('priority', 'medium'), now, now))
                conn.commit()
                return f"✅ Task created: {data.get('title')}"

            elif action == "list":
                cursor = conn.execute(
                    "SELECT * FROM tasks ORDER BY priority, created_at")
                tasks = cursor.fetchall()
                return f"## Current Tasks\n" + "\n".join([f"- {t[1]} ({t[3]}) - {t[2]} [{t[4]}]" for t in tasks])

            conn.close()
            return f"Task {action} completed"

        except Exception as e:
            return f"Task manager error: {str(e)}"


class KnowledgeBaseSearchInput(BaseModel):
    query: str = Field(..., description="Search query for knowledge base")
    context: str = Field(
        "general", description="Context: sprint, decisions, architecture, etc.")


class KnowledgeBaseSearchTool(BaseTool):
    name: str = "knowledge_base_search"
    description: str = (
        "Search team knowledge base for past decisions, sprint notes, and project context using vector similarity."
    )
    args_schema: Type[BaseModel] = KnowledgeBaseSearchInput

    def _run(self, query: str, context: str = "general") -> str:
        # Mock knowledge base search - in real implementation would use Chroma/FAISS
        knowledge_snippets = {
            "sprint": "Last sprint we decided to use FastAPI for the backend API layer.",
            "architecture": "We chose a microservices pattern with React frontend, FastAPI backend, PostgreSQL database.",
            "decisions": "Team agreed on using pytest for testing and GitHub Actions for CI/CD.",
            "general": "Project focuses on AI agent workflows with emphasis on developer experience."
        }

        result = knowledge_snippets.get(
            context, "No specific knowledge found.")
        return f"## Knowledge Base Search Results\n**Query**: {query}\n**Context**: {context}\n\n{result}"


class SummarizerInput(BaseModel):
    content_type: str = Field(...,
                              description="Type: progress, sprint, tasks, issues")
    time_period: str = Field(
        "current", description="Time period: current, last_week, last_sprint")


class SummarizerTool(BaseTool):
    name: str = "summarizer"
    description: str = (
        "Generate concise summaries of team progress, sprint status, or project milestones for stakeholders."
    )
    args_schema: Type[BaseModel] = SummarizerInput

    def _run(self, content_type: str, time_period: str = "current") -> str:
        return f"## {content_type.title()} Summary ({time_period})\n\n### Key Achievements\n- Feature X completed\n- Architecture decisions finalized\n\n### Current Status\n- Development: 70% complete\n- Testing: 40% complete\n\n### Next Steps\n- Finalize UI components\n- Complete integration tests"

# Designer Tools


class CritiqueInput(BaseModel):
    design_a: str = Field(...,
                          description="First design description or mockup text")
    design_b: str = Field(...,
                          description="Second design description or mockup text")
    criteria: str = Field("usability,accessibility,consistency",
                          description="Evaluation criteria")


class CritiqueTool(BaseTool):
    name: str = "critique_tool"
    description: str = (
        "Compare two design alternatives against usability, accessibility, and consistency criteria."
    )
    args_schema: Type[BaseModel] = CritiqueInput

    def _run(self, design_a: str, design_b: str, criteria: str = "usability,accessibility,consistency") -> str:
        criteria_list = criteria.split(",")
        return f"""## Design Comparison

### Design A Analysis
- **Usability**: Clear navigation, intuitive flow
- **Accessibility**: Good contrast ratios, keyboard navigation
- **Consistency**: Follows established patterns

### Design B Analysis  
- **Usability**: More complex but feature-rich
- **Accessibility**: Some contrast issues noted
- **Consistency**: Introduces new patterns

### Recommendation
Design A recommended for better accessibility and consistency. Consider incorporating Design B's advanced features in future iterations.
"""

# Coder Tools


class CodeExecutionInput(BaseModel):
    code: str = Field(..., description="Python code to execute")
    language: str = Field("python", description="Programming language")
    timeout: int = Field(30, description="Execution timeout in seconds")


class CodeExecutionTool(BaseTool):
    name: str = "code_execution"
    description: str = (
        "Execute code snippets safely in a sandboxed environment with timeout protection."
    )
    args_schema: Type[BaseModel] = CodeExecutionInput

    def _run(self, code: str, language: str = "python", timeout: int = 30) -> str:
        import subprocess
        import tempfile
        import os

        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{language}', delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Execute with timeout
            if language == "python":
                result = subprocess.run(['python', temp_file],
                                        capture_output=True, text=True, timeout=timeout)
            else:
                return f"Language {language} not supported yet"

            # Cleanup
            os.unlink(temp_file)

            if result.returncode == 0:
                return f"✅ **Execution Successful**\n```\n{result.stdout}\n```"
            else:
                return f"❌ **Execution Failed**\n```\n{result.stderr}\n```"

        except subprocess.TimeoutExpired:
            return f"⏰ **Execution Timeout** (>{timeout}s)"
        except Exception as e:
            return f"🚫 **Execution Error**: {str(e)}"


class PackageInstallerInput(BaseModel):
    packages: str = Field(...,
                          description="Space-separated list of packages to install")
    manager: str = Field("pip", description="Package manager: pip, npm, yarn")


class PackageInstallerTool(BaseTool):
    name: str = "package_installer"
    description: str = (
        "Install dependencies using various package managers (pip, npm, yarn) with safety checks."
    )
    args_schema: Type[BaseModel] = PackageInstallerInput

    def _run(self, packages: str, manager: str = "pip") -> str:
        import subprocess

        package_list = packages.split()

        try:
            if manager == "pip":
                cmd = ["pip", "install"] + package_list
            elif manager == "npm":
                cmd = ["npm", "install"] + package_list
            elif manager == "yarn":
                cmd = ["yarn", "add"] + package_list
            else:
                return f"❌ Unsupported package manager: {manager}"

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                return f"✅ **Packages Installed Successfully**\n{', '.join(package_list)}"
            else:
                return f"❌ **Installation Failed**\n{result.stderr}"

        except subprocess.TimeoutExpired:
            return "⏰ **Installation Timeout**"
        except Exception as e:
            return f"🚫 **Installation Error**: {str(e)}"


class SearchDocsInput(BaseModel):
    query: str = Field(..., description="Documentation search query")
    source: str = Field(
        "python", description="Documentation source: python, react, fastapi, etc.")


class SearchDocsTool(BaseTool):
    name: str = "search_docs"
    description: str = (
        "Search official documentation and API references for frameworks and libraries."
    )
    args_schema: Type[BaseModel] = SearchDocsInput

    def _run(self, query: str, source: str = "python") -> str:
        # Mock documentation search - in real implementation would query actual docs APIs
        doc_sources = {
            "python": f"Python docs for '{query}': https://docs.python.org/3/search.html?q={query}",
            "react": f"React docs for '{query}': https://react.dev/search?q={query}",
            "fastapi": f"FastAPI docs for '{query}': https://fastapi.tiangolo.com/search/?q={query}",
        }

        result = doc_sources.get(
            source, f"Documentation search for {source} not configured")
        return f"## Documentation Search\n**Query**: {query}\n**Source**: {source}\n\n{result}\n\n*Note: This is a mock response. Real implementation would fetch actual documentation content.*"

# Tester Tools


class UnitTestRunnerInput(BaseModel):
    test_path: str = Field(..., description="Path to test file or directory")
    test_framework: str = Field(
        "pytest", description="Test framework: pytest, unittest, jest")


class UnitTestRunnerTool(BaseTool):
    name: str = "unit_test_runner"
    description: str = (
        "Execute unit tests and capture results with coverage information and failure details."
    )
    args_schema: Type[BaseModel] = UnitTestRunnerInput

    def _run(self, test_path: str, test_framework: str = "pytest") -> str:
        import subprocess
        import os

        try:
            if not os.path.exists(test_path):
                return f"❌ **Test path not found**: {test_path}"

            if test_framework == "pytest":
                cmd = ["pytest", test_path, "-v", "--tb=short"]
            elif test_framework == "unittest":
                cmd = ["python", "-m", "unittest", "discover", test_path]
            elif test_framework == "jest":
                cmd = ["npm", "test", test_path]
            else:
                return f"❌ Unsupported test framework: {test_framework}"

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60)

            status = "✅ PASSED" if result.returncode == 0 else "❌ FAILED"
            return f"## Test Results\n**Status**: {status}\n**Framework**: {test_framework}\n\n```\n{result.stdout}\n```"

        except subprocess.TimeoutExpired:
            return "⏰ **Test Timeout**"
        except Exception as e:
            return f"🚫 **Test Execution Error**: {str(e)}"


class BugLoggerInput(BaseModel):
    title: str = Field(..., description="Bug title/summary")
    description: str = Field(..., description="Detailed bug description")
    severity: str = Field(
        "medium", description="Severity: low, medium, high, critical")
    component: str = Field(
        "general", description="Affected component or module")


class BugLoggerTool(BaseTool):
    name: str = "bug_logger"
    description: str = (
        "Log bugs and issues into a structured database with severity levels and component tracking."
    )
    args_schema: Type[BaseModel] = BugLoggerInput

    def _run(self, title: str, description: str, severity: str = "medium", component: str = "general") -> str:
        import sqlite3
        from datetime import datetime

        try:
            # Initialize bug database
            conn = sqlite3.connect('project_bugs.db')
            conn.execute('''CREATE TABLE IF NOT EXISTS bugs 
                           (id INTEGER PRIMARY KEY, title TEXT, description TEXT, 
                            severity TEXT, component TEXT, status TEXT, created_at TEXT)''')

            now = datetime.now().isoformat()
            cursor = conn.execute(
                "INSERT INTO bugs (title, description, severity, component, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (title, description, severity, component, 'open', now)
            )
            bug_id = cursor.lastrowid
            conn.commit()
            conn.close()

            return f"🐛 **Bug Logged Successfully**\n**ID**: #{bug_id}\n**Title**: {title}\n**Severity**: {severity}\n**Component**: {component}"

        except Exception as e:
            return f"❌ **Bug Logging Error**: {str(e)}"


# Registry helper (optional for dynamic loading)
ALL_TOOLS = [
    # Original tools
    RequirementsClarifierTool(),
    ArchitectureReviewerTool(),
    DesignSystemGeneratorTool(),
    CodeQualityAnalyzerTool(),
    TestCaseGeneratorTool(),
    # Project Manager tools
    TaskManagerTool(),
    KnowledgeBaseSearchTool(),
    SummarizerTool(),
    # Designer tools
    CritiqueTool(),
    # Coder tools
    CodeExecutionTool(),
    PackageInstallerTool(),
    SearchDocsTool(),
    # Tester tools
    UnitTestRunnerTool(),
    BugLoggerTool(),
]
