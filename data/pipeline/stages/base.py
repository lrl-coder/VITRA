"""
Base classes for pipeline stages.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path
import logging
import time


class PipelineStage(ABC):
    """Abstract base class for all pipeline stages."""
    
    def __init__(self, config: Any, logger: Optional[logging.Logger] = None):
        """
        Initialize the pipeline stage.
        
        Args:
            config: Stage-specific configuration object
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._is_initialized = False
    
    @property
    def name(self) -> str:
        """Return the name of this stage."""
        return self.__class__.__name__
    
    def initialize(self) -> None:
        """
        Initialize resources needed by this stage.
        Called once before processing begins.
        """
        if self._is_initialized:
            return
        self._do_initialize()
        self._is_initialized = True
        self.logger.info(f"Stage '{self.name}' initialized successfully")
    
    @abstractmethod
    def _do_initialize(self) -> None:
        """Implementation-specific initialization."""
        pass
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return results.
        
        Args:
            input_data: Dictionary containing input data for this stage
            
        Returns:
            Dictionary containing processed results
        """
        pass
    
    def cleanup(self) -> None:
        """
        Clean up resources used by this stage.
        Called after processing is complete.
        """
        self._do_cleanup()
        self._is_initialized = False
        self.logger.info(f"Stage '{self.name}' cleaned up")
    
    def _do_cleanup(self) -> None:
        """Implementation-specific cleanup. Override if needed."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False


class TimedStage(PipelineStage):
    """Pipeline stage with timing information."""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process with timing."""
        start_time = time.time()
        try:
            result = self._do_process(input_data)
            elapsed = time.time() - start_time
            self.logger.info(f"Stage '{self.name}' completed in {elapsed:.2f}s")
            result['_processing_time'] = elapsed
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"Stage '{self.name}' failed after {elapsed:.2f}s: {e}")
            raise
    
    @abstractmethod
    def _do_process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation-specific processing."""
        pass


class StageResult:
    """Container for stage processing results."""
    
    def __init__(
        self,
        success: bool,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.data = data or {}
        self.error = error
        self.metadata = metadata or {}
    
    def __bool__(self) -> bool:
        return self.success
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata
        }
