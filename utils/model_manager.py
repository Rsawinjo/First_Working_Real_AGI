"""
Model Manager - Handles downloading, loading, and management of AI models
Provides interface for working with different Hugging Face models
"""

import os
import json
import logging
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import threading
import hashlib
from pathlib import Path
import shutil

try:
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class ModelManager:
    def __init__(self, cache_dir: str = "./data/models", config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.cache_dir = Path(cache_dir)
        self.config = config or {}
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self.model_registry = {
            "microsoft/DialoGPT-small": {
                "name": "DialoGPT Small",
                "description": "Conversational AI model - Fast and efficient",
                "size": "117MB",
                "type": "conversational",
                "recommended": True,
                "requirements": ["transformers", "torch"]
            },
            "microsoft/DialoGPT-medium": {
                "name": "DialoGPT Medium",
                "description": "Conversational AI model - Balanced performance",
                "size": "345MB",
                "type": "conversational",
                "recommended": True,
                "requirements": ["transformers", "torch"]
            },
            "microsoft/DialoGPT-large": {
                "name": "DialoGPT Large",
                "description": "Conversational AI model - High quality responses",
                "size": "1.4GB",
                "type": "conversational",
                "recommended": False,
                "requirements": ["transformers", "torch", "8GB+ RAM"]
            },
            "facebook/blenderbot_small-90M": {
                "name": "BlenderBot Small",
                "description": "Open-domain chatbot - Lightweight",
                "size": "90MB",
                "type": "conversational",
                "recommended": True,
                "requirements": ["transformers", "torch"]
            },
            "facebook/blenderbot-400M-distill": {
                "name": "BlenderBot 400M",
                "description": "Open-domain chatbot - Good balance",
                "size": "400MB",
                "type": "conversational",
                "recommended": True,
                "requirements": ["transformers", "torch"]
            },
            "gpt2": {
                "name": "GPT-2",
                "description": "General purpose language model",
                "size": "500MB",
                "type": "generative",
                "recommended": True,
                "requirements": ["transformers", "torch"]
            }
        }
        
        # Download status tracking
        self.download_status = {}
        self.installation_status = {}
        
        # Load local model info
        self._load_local_models()
    
    def get_available_models(self) -> List[Dict]:
        """Get list of available models"""
        try:
            models = []
            for model_id, info in self.model_registry.items():
                model_info = info.copy()
                model_info['id'] = model_id
                model_info['is_downloaded'] = self._is_model_downloaded(model_id)
                model_info['local_path'] = self._get_model_path(model_id) if model_info['is_downloaded'] else None
                models.append(model_info)
            
            return models
            
        except Exception as e:
            self.logger.error(f"Error getting available models: {e}")
            return []
    
    def get_recommended_models(self) -> List[Dict]:
        """Get list of recommended models"""
        try:
            all_models = self.get_available_models()
            recommended = [model for model in all_models if model.get('recommended', False)]
            return recommended
            
        except Exception as e:
            self.logger.error(f"Error getting recommended models: {e}")
            return []
    
    def _is_model_downloaded(self, model_id: str) -> bool:
        """Check if model is downloaded locally"""
        try:
            model_path = self._get_model_path(model_id)
            return model_path.exists() and any(model_path.iterdir())
            
        except Exception as e:
            self.logger.error(f"Error checking if model is downloaded: {e}")
            return False
    
    def _get_model_path(self, model_id: str) -> Path:
        """Get local path for model"""
        # Convert model ID to filesystem-safe name
        safe_name = model_id.replace("/", "_").replace("\\", "_")
        return self.cache_dir / safe_name
    
    def download_model(self, model_id: str, callback=None) -> bool:
        """Download a model from Hugging Face Hub"""
        try:
            if not TRANSFORMERS_AVAILABLE:
                self.logger.error("Transformers library not available")
                return False
            
            if model_id not in self.model_registry:
                self.logger.error(f"Unknown model: {model_id}")
                return False
            
            # Check if already downloaded
            if self._is_model_downloaded(model_id):
                self.logger.info(f"Model {model_id} already downloaded")
                if callback:
                    callback("already_downloaded", model_id)
                return True
            
            # Set download status
            self.download_status[model_id] = {
                'status': 'downloading',
                'progress': 0,
                'start_time': datetime.now()
            }
            
            if callback:
                callback("download_started", model_id)
            
            # Create model directory
            model_path = self._get_model_path(model_id)
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Download model components
            success = self._download_model_components(model_id, model_path, callback)
            
            if success:
                self.download_status[model_id]['status'] = 'completed'
                self.download_status[model_id]['end_time'] = datetime.now()
                
                # Save model info
                self._save_model_info(model_id, model_path)
                
                if callback:
                    callback("download_completed", model_id)
                
                self.logger.info(f"Successfully downloaded model: {model_id}")
                return True
            else:
                self.download_status[model_id]['status'] = 'failed'
                if callback:
                    callback("download_failed", model_id)
                return False
                
        except Exception as e:
            self.logger.error(f"Error downloading model {model_id}: {e}")
            if model_id in self.download_status:
                self.download_status[model_id]['status'] = 'failed'
                self.download_status[model_id]['error'] = str(e)
            if callback:
                callback("download_error", f"{model_id}: {e}")
            return False
    
    def _download_model_components(self, model_id: str, model_path: Path, callback=None) -> bool:
        """Download individual model components"""
        try:
            # Download tokenizer
            if callback:
                callback("downloading_component", f"{model_id}: tokenizer")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=str(model_path),
                local_files_only=False
            )
            
            # Download model config
            if callback:
                callback("downloading_component", f"{model_id}: config")
            
            config = AutoConfig.from_pretrained(
                model_id,
                cache_dir=str(model_path),
                local_files_only=False
            )
            
            # Download model weights
            if callback:
                callback("downloading_component", f"{model_id}: model weights")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=str(model_path),
                local_files_only=False
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading components for {model_id}: {e}")
            return False
    
    def _save_model_info(self, model_id: str, model_path: Path):
        """Save model information to local file"""
        try:
            model_info = {
                'model_id': model_id,
                'download_date': datetime.now().isoformat(),
                'local_path': str(model_path),
                'registry_info': self.model_registry.get(model_id, {}),
                'status': 'ready'
            }
            
            info_file = model_path / 'model_info.json'
            with open(info_file, 'w') as f:
                json.dump(model_info, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving model info: {e}")
    
    def _load_local_models(self):
        """Load information about locally available models"""
        try:
            for model_path in self.cache_dir.iterdir():
                if model_path.is_dir():
                    info_file = model_path / 'model_info.json'
                    if info_file.exists():
                        try:
                            with open(info_file, 'r') as f:
                                model_info = json.load(f)
                            
                            model_id = model_info.get('model_id')
                            if model_id and model_id not in self.model_registry:
                                # Add to registry if not already present
                                self.model_registry[model_id] = {
                                    'name': model_id.split('/')[-1],
                                    'description': 'Locally installed model',
                                    'type': 'unknown',
                                    'is_local': True
                                }
                        except Exception as e:
                            self.logger.error(f"Error loading model info from {info_file}: {e}")
                            
        except Exception as e:
            self.logger.error(f"Error loading local models: {e}")
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a downloaded model"""
        try:
            model_path = self._get_model_path(model_id)
            
            if not model_path.exists():
                self.logger.warning(f"Model {model_id} not found locally")
                return False
            
            # Remove the model directory
            shutil.rmtree(model_path)
            
            # Clean up download status
            if model_id in self.download_status:
                del self.download_status[model_id]
            
            self.logger.info(f"Removed model: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing model {model_id}: {e}")
            return False
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get detailed information about a model"""
        try:
            if model_id not in self.model_registry:
                return None
            
            info = self.model_registry[model_id].copy()
            info['id'] = model_id
            info['is_downloaded'] = self._is_model_downloaded(model_id)
            
            if info['is_downloaded']:
                model_path = self._get_model_path(model_id)
                info_file = model_path / 'model_info.json'
                
                if info_file.exists():
                    try:
                        with open(info_file, 'r') as f:
                            local_info = json.load(f)
                        info.update(local_info)
                    except Exception as e:
                        self.logger.error(f"Error reading local model info: {e}")
                
                # Get size information
                try:
                    size = self._get_directory_size(model_path)
                    info['local_size'] = self._format_size(size)
                except Exception as e:
                    self.logger.error(f"Error getting model size: {e}")
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting model info for {model_id}: {e}")
            return None
    
    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        try:
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            self.logger.error(f"Error calculating directory size: {e}")
        return total_size
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def get_download_status(self, model_id: str) -> Optional[Dict]:
        """Get download status for a model"""
        return self.download_status.get(model_id)
    
    def get_model_requirements(self, model_id: str) -> List[str]:
        """Get requirements for a model"""
        try:
            model_info = self.model_registry.get(model_id, {})
            return model_info.get('requirements', [])
            
        except Exception as e:
            self.logger.error(f"Error getting requirements for {model_id}: {e}")
            return []
    
    def check_system_compatibility(self, model_id: str) -> Dict[str, bool]:
        """Check if system is compatible with model requirements"""
        compatibility = {
            'transformers': TRANSFORMERS_AVAILABLE,
            'sufficient_memory': True,  # Simplified check
            'disk_space': True  # Simplified check
        }
        
        try:
            requirements = self.get_model_requirements(model_id)
            
            # Check specific requirements
            for req in requirements:
                if 'GB' in req and 'RAM' in req:
                    # Parse memory requirement
                    try:
                        import psutil
                        total_memory = psutil.virtual_memory().total / (1024**3)  # GB
                        required_gb = float(req.split('GB')[0])
                        compatibility['sufficient_memory'] = total_memory >= required_gb
                    except:
                        compatibility['sufficient_memory'] = True  # Assume OK if can't check
            
            return compatibility
            
        except Exception as e:
            self.logger.error(f"Error checking compatibility: {e}")
            return compatibility
    
    def auto_select_model(self) -> Optional[str]:
        """Automatically select the best model for the system"""
        try:
            recommended_models = self.get_recommended_models()
            
            # Filter by compatibility
            compatible_models = []
            for model in recommended_models:
                compatibility = self.check_system_compatibility(model['id'])
                if all(compatibility.values()):
                    compatible_models.append(model)
            
            if not compatible_models:
                # Fall back to any recommended model
                compatible_models = recommended_models
            
            if compatible_models:
                # Prefer downloaded models
                downloaded_models = [m for m in compatible_models if m['is_downloaded']]
                if downloaded_models:
                    return downloaded_models[0]['id']
                
                # Otherwise, return smallest recommended model
                compatible_models.sort(key=lambda x: x.get('size', '0MB'))
                return compatible_models[0]['id']
            
            # Last resort - return any available model
            all_models = self.get_available_models()
            if all_models:
                return all_models[0]['id']
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in auto model selection: {e}")
            return None
    
    def download_recommended_model(self, callback=None) -> Optional[str]:
        """Download the best recommended model for the system"""
        try:
            model_id = self.auto_select_model()
            if not model_id:
                return None
            
            if self._is_model_downloaded(model_id):
                return model_id
            
            success = self.download_model(model_id, callback)
            return model_id if success else None
            
        except Exception as e:
            self.logger.error(f"Error downloading recommended model: {e}")
            return None
    
    def export_model_registry(self, filepath: str) -> bool:
        """Export model registry to file"""
        try:
            export_data = {
                'model_registry': self.model_registry,
                'download_status': self.download_status,
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Model registry exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting model registry: {e}")
            return False
    
    def cleanup_cache(self, keep_downloaded: bool = True) -> bool:
        """Clean up model cache"""
        try:
            if not keep_downloaded:
                # Remove all cached models
                if self.cache_dir.exists():
                    shutil.rmtree(self.cache_dir)
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                self.download_status.clear()
            else:
                # Remove only failed downloads
                failed_models = [
                    model_id for model_id, status in self.download_status.items()
                    if status.get('status') == 'failed'
                ]
                
                for model_id in failed_models:
                    model_path = self._get_model_path(model_id)
                    if model_path.exists():
                        shutil.rmtree(model_path)
                    del self.download_status[model_id]
            
            self.logger.info("Cache cleanup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning up cache: {e}")
            return False
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        try:
            stats = {
                'total_models': len(self.model_registry),
                'downloaded_models': 0,
                'cache_size': 0,
                'cache_path': str(self.cache_dir)
            }
            
            # Count downloaded models and calculate cache size
            for model_id in self.model_registry:
                if self._is_model_downloaded(model_id):
                    stats['downloaded_models'] += 1
                    model_path = self._get_model_path(model_id)
                    stats['cache_size'] += self._get_directory_size(model_path)
            
            stats['cache_size_formatted'] = self._format_size(stats['cache_size'])
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {}
    
    def validate_model(self, model_id: str) -> bool:
        """Validate that a downloaded model is working correctly"""
        try:
            if not self._is_model_downloaded(model_id):
                return False
            
            if not TRANSFORMERS_AVAILABLE:
                return False
            
            model_path = self._get_model_path(model_id)
            
            # Try to load the model components
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
                config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
                # Don't load the full model for validation as it's memory intensive
                
                return True
                
            except Exception as e:
                self.logger.error(f"Model validation failed for {model_id}: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error validating model {model_id}: {e}")
            return False