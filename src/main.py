#!/usr/bin/env python
import sys
import warnings
import os
from datetime import datetime

# Import the new orchestrator system
from orchestrator import AgentOrchestrator
from crew import DevcrewAgents

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run_interactive_queries():
    """Run interactive query mode - main entry point for user queries."""
    print("🚀 Starting DevCrew Agents Interactive Query Mode")
    print("=" * 60)

    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()

        # Initialize orchestrator
        orchestrator = AgentOrchestrator()

        print(f"✅ Initialized orchestrator with models:")
        for agent_type, model in orchestrator.model_mapping.items():
            print(f"   • {agent_type}: {model}")

        print(f"📋 Current project phase: {orchestrator.current_phase.value}")
        print()

        # Start interactive mode
        orchestrator.query_interactive_mode()

    except Exception as e:
        print(f"❌ Error in interactive query mode: {e}")
        raise Exception(f"An error occurred: {e}")


def demo_single_query():
    """Demonstrate single query processing."""
    print("🎯 Single Query Demo")
    print("=" * 50)

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    orchestrator = AgentOrchestrator()

    # Example queries to demonstrate different flows
    sample_queries = [
        "Create a project plan for a task management app",
        "Design a clean user interface for the dashboard",
        "How should we implement user authentication?",
        "What testing strategy do we need?"
    ]

    for i, query in enumerate(sample_queries, 1):
        print(f"\n🔄 Query {i}: {query}")
        print("-" * 50)

        response = orchestrator.handle_user_query(query)

        # Show abbreviated response
        lines = response.split('\n')
        print('\n'.join(lines[:15]))  # First 15 lines
        if len(lines) > 15:
            print(f"\n... (truncated, full response has {len(lines)} lines)")

        print(f"\n📊 Project Phase: {orchestrator.current_phase.value}")
        print(f"📈 Tasks Completed: {len(orchestrator.completed_tasks)}")

        if orchestrator.is_project_complete():
            print("🎉 Project Complete!")
            break


def run_with_agent_loop():
    """Run the crew using the new agent loop system."""
    print("🚀 Starting DevCrew Agents with Agent Loop System")
    print("=" * 60)

    inputs = {
        'project_name': 'AI Dev Platform',
        'tech_stack': 'Python FastAPI PostgreSQL React',
        'target_user': 'AI Engineers building agent workflows',
        'design_principles': 'clarity, accessibility, performance, modularity',
        'current_year': str(datetime.now().year)
    }

    try:
        # Initialize orchestrator
        orchestrator = AgentOrchestrator()

        print(f"✅ Initialized orchestrator with models:")
        for agent_type, model in orchestrator.model_mapping.items():
            print(f"   • {agent_type}: {model}")

        print("\n🔄 Executing development cycle...")

        # Execute the development workflow
        results = orchestrator.execute_development_cycle(inputs)

        print("\n📊 Execution Summary:")
        print("=" * 40)
        print(results['execution_summary']['execution_summary'])

        print("\n✅ Agent loop execution completed successfully!")

        return results

    except Exception as e:
        print(f"❌ Error in agent loop execution: {e}")
        raise Exception(f"An error occurred while running the agent loop: {e}")


def run_traditional():
    """Run the traditional crew system for comparison."""
    print("🔄 Running traditional CrewAI system for comparison...")

    inputs = {
        'project_name': 'AI Dev Platform',
        'tech_stack': 'Python FastAPI PostgreSQL React',
        'target_user': 'AI Engineers building agent workflows',
        'design_principles': 'clarity, accessibility, performance, modularity',
        'current_year': str(datetime.now().year)
    }

    try:
        DevcrewAgents().crew().kickoff(inputs=inputs)
        print("✅ Traditional crew execution completed!")
    except Exception as e:
        print(f"❌ Error in traditional crew: {e}")
        raise Exception(f"An error occurred while running the crew: {e}")


def run():
    """Main run function - uses interactive query mode by default."""
    run_interactive_queries()


def demo_agent_loop():
    """Demonstrate the agent loop system with detailed output."""
    print("🎭 Agent Loop System Demonstration")
    print("=" * 50)

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    inputs = {
        'project_name': 'Task Management App',
        'tech_stack': 'React TypeScript Node.js MongoDB',
        'target_user': 'Small development teams',
        'design_principles': 'simplicity, efficiency, collaboration'
    }

    orchestrator = AgentOrchestrator()

    print("🏗️ Creating custom workflow...")

    # Create custom tasks for demonstration
    from orchestrator import OrchestratorTask

    custom_tasks = [
        OrchestratorTask(
            id="requirements_analysis",
            description="Analyze and clarify project requirements",
            agent_type="project_manager",
            priority=10,
            dependencies=[],
            inputs=inputs,
            timeout=180
        ),
        OrchestratorTask(
            id="ui_design",
            description="Design user interface and experience",
            agent_type="designer",
            priority=8,
            dependencies=["requirements_analysis"],
            inputs=inputs,
            timeout=200
        ),
        OrchestratorTask(
            id="backend_architecture",
            description="Design backend architecture and API",
            agent_type="coder",
            priority=7,
            dependencies=["requirements_analysis"],
            inputs=inputs,
            timeout=250
        ),
        OrchestratorTask(
            id="integration_testing",
            description="Plan integration and testing strategy",
            agent_type="tester",
            priority=5,
            dependencies=["ui_design", "backend_architecture"],
            inputs=inputs,
            timeout=180
        )
    ]

    # Add tasks to orchestrator
    for task in custom_tasks:
        orchestrator.add_task(task)

    print(f"📋 Added {len(custom_tasks)} tasks to orchestrator")

    # Execute with parallel processing where possible
    results = orchestrator.execute_parallel(inputs)

    print("\n📈 Detailed Results:")
    print("=" * 30)

    for task_id, output in results.items():
        print(f"\n🎯 Task: {task_id}")
        print(f"   Agent: {output.agent_id}")
        print(f"   Execution Time: {output.execution_time:.2f}s")
        print(f"   Reasoning Steps: {len(output.reasoning_chain)}")
        print(f"   Actions Taken: {len(output.actions_taken)}")
        print(f"   Final State: {output.final_state.value}")

        # Show sample reasoning
        if output.reasoning_chain:
            latest_reasoning = output.reasoning_chain[-1]
            print(f"   Latest Thought: {latest_reasoning.thought[:100]}...")

    # Show performance summary
    summary = orchestrator.get_execution_summary()
    print(f"\n📊 Performance Summary:")
    print(f"   Success Rate: {summary['success_rate']:.1%}")

    agent_perf = summary['agent_performance']
    for agent, metrics in agent_perf.items():
        if metrics['tasks_completed'] > 0:
            print(f"   {agent}: {metrics['avg_execution_time']:.1f}s avg, "
                  f"{metrics['avg_reasoning_steps']:.1f} reasoning steps avg")


def train():
    """Train the crew for a given number of iterations."""
    inputs = {
        'project_name': 'AI Dev Platform',
        'tech_stack': 'Python FastAPI PostgreSQL React',
        'target_user': 'AI Engineers building agent workflows',
        'design_principles': 'clarity, accessibility, performance, modularity',
        'current_year': str(datetime.now().year)
    }
    try:
        DevcrewAgents().crew().train(n_iterations=int(
            sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """Replay the crew execution from a specific task."""
    try:
        DevcrewAgents().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """Test the crew execution and returns the results."""
    inputs = {
        'project_name': 'AI Dev Platform',
        'tech_stack': 'Python FastAPI PostgreSQL React',
        'target_user': 'AI Engineers building agent workflows',
        'design_principles': 'clarity, accessibility, performance, modularity',
        'current_year': str(datetime.now().year)
    }

    try:
        DevcrewAgents().crew().test(n_iterations=int(
            sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "demo":
            demo_agent_loop()
        elif command == "traditional":
            run_traditional()
        elif command == "agent_loop":
            run_with_agent_loop()
        elif command == "interactive":
            run_interactive_queries()
        elif command == "single_query_demo":
            demo_single_query()
        else:
            print(
                "Available commands: demo, traditional, agent_loop, interactive, single_query_demo")
    else:
        run()
