#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from crew import DevcrewAgents

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """
    Run the crew.
    """
    inputs = {
        'project_name': 'AI Dev Platform',
        'tech_stack': 'Python FastAPI PostgreSQL React',
        'target_user': 'AI Engineers building agent workflows',
        'design_principles': 'clarity, accessibility, performance, modularity',
        'current_year': str(datetime.now().year)
    }

    try:
        DevcrewAgents().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
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
    """
    Replay the crew execution from a specific task.
    """
    try:
        DevcrewAgents().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and returns the results.
    """
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
