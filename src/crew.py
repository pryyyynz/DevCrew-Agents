from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from tools import (
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
    # Communication tools
    SharedMemoryTool,
    MessagePassingTool,
    KnowledgeStoreTool,
    TeamCommunicationTool,
)

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class DevcrewAgents():
    """DevcrewAgents crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def project_manager(self) -> Agent:
        project_manager_tools = [
            RequirementsClarifierTool(),
            ArchitectureReviewerTool(),
            TaskManagerTool(),
            KnowledgeBaseSearchTool(),
            SummarizerTool(),
            # Communication tools
            SharedMemoryTool(),
            MessagePassingTool(),
            KnowledgeStoreTool(),
            TeamCommunicationTool(),
        ]
        return Agent(
            # type: ignore[index]
            config=self.agents_config['project_manager'],
            tools=project_manager_tools,
            verbose=True
        )

    @agent
    def designer(self) -> Agent:
        designer_tools = [
            DesignSystemGeneratorTool(),
            ArchitectureReviewerTool(),
            CritiqueTool(),
            RequirementsClarifierTool(),  # To understand requirements for design
            # Communication tools
            SharedMemoryTool(),
            MessagePassingTool(),
            KnowledgeStoreTool(),
            TeamCommunicationTool(),
        ]
        return Agent(
            config=self.agents_config['designer'],  # type: ignore[index]
            tools=designer_tools,
            verbose=True
        )

    @agent
    def coder(self) -> Agent:
        coder_tools = [
            CodeQualityAnalyzerTool(),
            ArchitectureReviewerTool(),
            RequirementsClarifierTool(),
            CodeExecutionTool(),
            PackageInstallerTool(),
            SearchDocsTool(),
            # Communication tools
            SharedMemoryTool(),
            MessagePassingTool(),
            KnowledgeStoreTool(),
            TeamCommunicationTool(),
        ]
        return Agent(
            config=self.agents_config['coder'],  # type: ignore[index]
            tools=coder_tools,
            verbose=True
        )

    @agent
    def tester(self) -> Agent:
        tester_tools = [
            TestCaseGeneratorTool(),
            CodeQualityAnalyzerTool(),
            UnitTestRunnerTool(),
            BugLoggerTool(),
            # Communication tools
            SharedMemoryTool(),
            MessagePassingTool(),
            KnowledgeStoreTool(),
            TeamCommunicationTool(),
        ]
        return Agent(
            config=self.agents_config['tester'],  # type: ignore[index]
            tools=tester_tools,
            verbose=True
        )

    # Tasks reflecting a typical dev lifecycle
    @task
    def planning_task(self) -> Task:
        return Task(
            config=self.tasks_config['planning_task'],  # type: ignore[index]
        )

    @task
    def design_task(self) -> Task:
        return Task(
            config=self.tasks_config['design_task'],  # type: ignore[index]
            output_file='design.md',
            context=[self.planning_task]
        )

    @task
    def implementation_task(self) -> Task:
        return Task(
            # type: ignore[index]
            config=self.tasks_config['implementation_task'],
            output_file='implementation_plan.md',
            context=[self.planning_task, self.design_task]
        )

    @task
    def testing_task(self) -> Task:
        return Task(
            config=self.tasks_config['testing_task'],  # type: ignore[index]
            output_file='test_report.md',
            context=[self.implementation_task]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the DevcrewAgents crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
