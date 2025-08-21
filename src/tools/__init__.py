"""Custom tools for DevCrew Agents."""

from .custom_tool import (
    # Core tools
    RequirementsClarifierTool,
    ArchitectureReviewerTool,
    DesignSystemGeneratorTool,
    CodeQualityAnalyzerTool,
    TestCaseGeneratorTool,
    # Project Manager tools
    TaskManagerTool,
    KnowledgeBaseSearchTool,
    SummarizerTool,
    # Designer tools
    CritiqueTool,
    # Coder tools
    CodeExecutionTool,
    PackageInstallerTool,
    SearchDocsTool,
    # Tester tools
    UnitTestRunnerTool,
    BugLoggerTool,
    # File I/O and Architecture tools
    WriteFileTool,
    ReadFileTool,
    ArchitectureDocGeneratorTool,
    # Tool registry
    ALL_TOOLS
)

# Communication tools - fix the import path
from communication.communication_tools import (
    SharedMemoryTool,
    MessagePassingTool,
    KnowledgeStoreTool,
    TeamCommunicationTool
)

__all__ = [
    'RequirementsClarifierTool',
    'ArchitectureReviewerTool',
    'DesignSystemGeneratorTool',
    'CodeQualityAnalyzerTool',
    'TestCaseGeneratorTool',
    'TaskManagerTool',
    'KnowledgeBaseSearchTool',
    'SummarizerTool',
    'CritiqueTool',
    'CodeExecutionTool',
    'PackageInstallerTool',
    'SearchDocsTool',
    'UnitTestRunnerTool',
    'BugLoggerTool',
    'WriteFileTool',
    'ReadFileTool',
    'ArchitectureDocGeneratorTool',
    'SharedMemoryTool',
    'MessagePassingTool',
    'KnowledgeStoreTool',
    'TeamCommunicationTool',
    'ALL_TOOLS'
]
