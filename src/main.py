#!/usr/bin/env python
import warnings

# Only import web dependencies
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run_web():
    """Run the web interface for DevCrew Agents."""
    print("🌐 Starting DevCrew Agents Web Interface")

    try:
        from web_app import app
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

    except ImportError as e:
        print(f"❌ Missing web dependencies: {e}")
        print("💡 Install web dependencies with: pip install fastapi uvicorn jinja2 python-multipart websockets")
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")
        raise


if __name__ == "__main__":
    run_web()
