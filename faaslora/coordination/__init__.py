"""
Coordination Module

Handles scaling orchestration, resource coordination, and event-driven
instance management for the FaaSLoRA system.
"""

from faaslora.coordination.coordinator import Coordinator
from faaslora.coordination.autoscaler import AutoScaler

__all__ = ["Coordinator", "AutoScaler"]