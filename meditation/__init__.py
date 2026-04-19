"""
Hermes Meditation — Cross-session reflection pipeline.

Inspired by Pokoclaw's meditation system, adapted for Hermes Agent:
- Harvest friction episodes from recent sessions
- Bucket them by recurring pattern
- Consolidate into candidate rules for skills/memory
"""

from meditation.runner import MeditationPipelineRunner
from meditation.episode_extractor import EpisodeExtractor, Episode

__all__ = ["MeditationPipelineRunner", "EpisodeExtractor", "Episode"]
