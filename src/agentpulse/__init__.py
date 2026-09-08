"""AgentPulse — observability and drift detection for multi-agent systems.

Zero-code-change usage (recommended):

    agentpulse run python main.py

Or, if you can't change how your app is launched, add two lines instead:

    import agentpulse
    agentpulse.instrument()
"""

from agentpulse.sdk.instrument import instrument

__version__ = "0.3.3"
__all__ = ["instrument", "__version__"]
