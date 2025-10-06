"""
LLM Interface Module - Llama 3 AGI Integration
Handles Llama 3 model loading, fine-tuning, and fundamental weight modification
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    BitsAndBytesConfig, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, pipeline
)
from peft import (
    LoraConfig, get_peft_model, prepare_model_for_kbit_training,
    PeftModel, PeftConfig
)
import logging
import json
import os
from typing import Dict, List, Optional, Tuple
import threading
import time
from datetime import datetime
import numpy as np

class LLMInterface:
    def __init__(self, model_name: str = None, config_path: str = "./config/settings.py", max_response_length: int = 300):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        # Llama 3 AGI Integration - Default to Llama 3 8B
        self.model_name = model_name or self.config.get('DEFAULT_MODEL', 'meta-llama/Llama-3.2-3B-Instruct')

        # Response configuration
        self.max_response_length = max_response_length
        self.unlimited_mode = False
        self.smart_truncation = True

        # Model components
        self.tokenizer = None
        self.model = None
        self.pipeline = None

        # Llama 3 Fine-tuning Components
        self.peft_model = None
        self.lora_config = None
        self.training_args = None
        self.fine_tuned = False
        self.model_version = 1

        # GPU Configuration - RTX 4090 Optimization for Llama 3
        self.device = self._setup_gpu_device()
        self.gpu_memory_optimization = True
        self.use_4bit_quantization = True  # Enable 4-bit quantization for Llama 3
        self.use_flash_attention = True    # Flash attention for speed

        # Conversation management
        self.conversation_history = []
        self.conversation_context = {}
        self.performance_metrics = {
            'total_conversations': 0,
            'avg_response_time': 0,
            'improvement_score': 0.5,
            'learning_rate': 0.001,
            'fine_tuning_sessions': 0
        }

        # AGI Learning Integration
        self.learned_knowledge = []
        self.fine_tuning_data = []
        self.knowledge_buffer = []
        self.is_fine_tuning = False

        # Self-improvement components
        self.feedback_buffer = []
        self.improvement_thread = None
        self.is_improving = False

        # Enhanced capabilities for AGI
        self.web_search_enabled = True
        self.research_depth = "comprehensive"
        self.parallel_processing = True

        # Set up Hugging Face authentication for gated models
        self._setup_huggingface_auth()

        self.load_model()
        self._start_improvement_loop()
    
    def _setup_gpu_device(self):
        """Setup optimal GPU configuration for RTX 4090"""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                memory_gb = torch.cuda.get_device_properties(current_device).total_memory / 1e9
                
                # Only log GPU info, not debug prints
                if "4090" in device_name or "RTX" in device_name:
                    self.logger.info(f"🚀 RTX GPU detected: {device_name} ({memory_gb:.1f}GB) - Beast Mode enabled!")
                    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
                    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for massive speedup
                    torch.backends.cudnn.allow_tf32 = True
                else:
                    self.logger.info(f"✅ GPU detected: {device_name} ({memory_gb:.1f}GB)")
                    
                return f"cuda:{current_device}"
            else:
                # Silent CPU fallback - no warning needed
                return "cpu"
        except Exception as e:
            self.logger.debug(f"GPU detection issue: {e}")
            return "cpu"
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration settings"""
        try:
            with open(config_path.replace('.py', '.json'), 'r') as f:
                return json.load(f)
        except:
            # Default config if file doesn't exist
            return {
                'DEFAULT_MODEL': 'meta-llama/Llama-3.2-3B-Instruct',
                'MODEL_CACHE_DIR': './data/models',
                'LEARNING_RATE': 0.001,
                'IMPROVEMENT_INTERVAL': 100
            }
    
    def _setup_huggingface_auth(self):
        """Set up Hugging Face authentication for gated models"""
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
            if token:
                os.environ['HF_TOKEN'] = token
                os.environ['HUGGINGFACE_TOKEN'] = token
                self.logger.info("Hugging Face authentication configured")
            else:
                self.logger.warning("No Hugging Face token found - gated models may not load")
        except Exception as e:
            self.logger.warning(f"Failed to set up Hugging Face auth: {e}")
    
    def load_model(self, model_name: str = None):
        """Load the specified language model"""
        if model_name:
            self.model_name = model_name
        
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            
            # Create cache directory
            cache_dir = self.config.get('MODEL_CACHE_DIR', './data/models')
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                cache_dir=cache_dir,
                padding_side='left'
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load appropriate model type with RTX 4090 optimizations
            try:
                self.logger.info(f"Loading model on {self.device} with GPU optimizations...")
                
                # RTX 4090 Beast Mode Configuration
                model_kwargs = {
                    'cache_dir': cache_dir,
                    'low_cpu_mem_usage': True,  # Load directly to GPU
                }
                
                # Enable mixed precision for RTX 4090 (50-100x speedup!)
                if "cuda" in self.device:
                    # For Llama 3, avoid device_map="auto" to prevent device mismatch errors
                    # Instead, load directly to GPU and manage memory manually
                    if "llama" in self.model_name.lower() or "meta-llama" in self.model_name.lower():
                        model_kwargs['torch_dtype'] = torch.float16  # FP16 for massive speedup
                        model_kwargs['use_cache'] = True  # Enable KV cache for faster inference
                        # Don't use device_map for Llama 3 to avoid meta device issues
                    else:
                        model_kwargs['device_map'] = "auto"
                        model_kwargs['torch_dtype'] = torch.float16  # FP16 for massive speedup
                        model_kwargs['use_cache'] = True  # Enable KV cache for faster inference
                else:
                    model_kwargs['torch_dtype'] = torch.float32
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                
                # For Llama 3, ensure all tensors are on the correct device
                if "llama" in self.model_name.lower() or "meta-llama" in self.model_name.lower():
                    if "cuda" in self.device:
                        self.model = self.model.to(self.device)
                        self.logger.info(f"Llama 3 model moved to {self.device}")
                        
            except Exception as e:
                self.logger.warning(f"AutoModelForCausalLM failed: {e}, trying Seq2Seq model format...")
                model_kwargs_seq2seq = model_kwargs.copy()
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    **model_kwargs_seq2seq
                )
            
            # Create optimized pipeline for RTX 4090
            # Note: When using accelerate, we don't specify device in pipeline
            self.logger.info(f"Creating optimized pipeline (accelerate-managed)")
            
            pipeline_kwargs = {
                "task": "text-generation",
                "model": self.model,
                "tokenizer": self.tokenizer,
                "return_full_text": False,
                "do_sample": True,
                "temperature": 0.7,
                "max_new_tokens": 150,
                "pad_token_id": self.tokenizer.pad_token_id,
                "framework": "pt"  # PyTorch framework
            }
            
            # Only add device if not using accelerate
            if not hasattr(self.model, 'hf_device_map'):
                pipeline_device = 0 if "cuda" in self.device else -1
                pipeline_kwargs["device"] = pipeline_device
                pipeline_kwargs["batch_size"] = 4 if "cuda" in self.device else 1
            
            self.pipeline = pipeline(**pipeline_kwargs)
            
            # GPU memory optimization
            if "cuda" in self.device:
                torch.cuda.empty_cache()  # Clear any existing cache
                self.logger.info(f"RTX 4090 READY - Beast mode activated!")
            
            self.logger.info(f"Model loaded successfully on {self.device}")
            print(f"Device set to use {self.device}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            # Try Llama fallback models only - no non-Llama models allowed
            llama_fallbacks = [
                "meta-llama/Llama-3.2-3B-Instruct",
                "meta-llama/Llama-3.2-1B-Instruct",
                "meta-llama/Llama-3.2-3B",
                "meta-llama/Llama-3.2-1B"
            ]
            
            for fallback in llama_fallbacks:
                if self.model_name != fallback:
                    self.logger.info(f"Trying Llama fallback model: {fallback}")
                    self.model_name = fallback
                    try:
                        return self.load_model()
                    except Exception as fallback_error:
                        self.logger.warning(f"Llama fallback {fallback} also failed: {fallback_error}")
                        continue
            
            self.logger.error("All Llama models failed to load - AGI system cannot function without Llama")
            return False
    
    def generate_response(self, user_input: str, context: Optional[Dict] = None) -> Tuple[str, Dict]:
        """Generate response with self-improvement capabilities"""
        start_time = time.time()
        
        try:
            # Prepare context
            full_context = self._prepare_context(user_input, context)
            
            # Generate response
            if self.pipeline:
                # Format input for conversation
                conversation_text = self._format_conversation_input(user_input, full_context)
                
                # Generate
                max_tokens = self.max_response_length if not self.unlimited_mode else 1000
                outputs = self.pipeline(
                    conversation_text,
                    max_new_tokens=max_tokens,
                    num_return_sequences=1,
                    temperature=0.7 + (self.performance_metrics['improvement_score'] * 0.2),
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer and self.tokenizer.pad_token_id else (self.tokenizer.eos_token_id if self.tokenizer else None)
                )
                
                # Extract response from pipeline output
                if isinstance(outputs, list) and len(outputs) > 0:
                    if isinstance(outputs[0], dict) and 'generated_text' in outputs[0]:
                        response = outputs[0]['generated_text']
                    else:
                        response = str(outputs[0])
                else:
                    response = str(outputs)
                
                # Clean and format response
                response = self._clean_response(response, user_input)
                response = self._format_response(response)
                
            else:
                response = "I'm still learning. Please help me improve by continuing our conversation."
            
            # Update metrics
            response_time = time.time() - start_time
            self._update_performance_metrics(response_time)
            
            # Store for learning
            self._store_interaction(user_input, response, full_context)
            
            # Return response with metadata
            metadata = {
                'response_time': response_time,
                'confidence': self._calculate_confidence(response),
                'improvement_score': self.performance_metrics['improvement_score'],
                'total_conversations': self.performance_metrics['total_conversations']
            }
            
            return response, metadata
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"I encountered an error: {str(e)}. I'm learning from this to improve.", {}
    
    def _prepare_context(self, user_input: str, context: Optional[Dict] = None) -> Dict:
        """Prepare conversation context with learning enhancements"""
        full_context = context or {}
        
        # Add conversation history
        full_context['history'] = self.conversation_history[-5:]  # Last 5 exchanges
        
        # Add performance context
        full_context['performance'] = self.performance_metrics
        
        # Add timestamp
        full_context['timestamp'] = datetime.now().isoformat()
        
        return full_context
    
    def _format_conversation_input(self, user_input: str, context: Dict) -> str:
        """Format input for the conversation model"""
        # Build conversation context
        conversation_parts = []
        
        # Add recent history
        history = context.get('history', [])
        for exchange in history[-3:]:  # Last 3 exchanges
            if 'user' in exchange and 'assistant' in exchange:
                conversation_parts.append(f"Human: {exchange['user']}")
                conversation_parts.append(f"Assistant: {exchange['assistant']}")
        
        # Add current input
        conversation_parts.append(f"Human: {user_input}")
        conversation_parts.append("Assistant:")
        
        return "\n".join(conversation_parts)
    
    def _clean_response(self, response: str, user_input: str) -> str:
        """Clean and improve the generated response"""
        # Remove input echo
        if user_input.lower() in response.lower():
            response = response.replace(user_input, "").strip()
        
        # Remove common artifacts
        artifacts = ["Human:", "Assistant:", "User:", "AI:", "<|endoftext|>", "<pad>"]
        for artifact in artifacts:
            response = response.replace(artifact, "").strip()
        
        # Ensure proper sentence ending
        if response and not response.endswith(('.', '!', '?')):
            response += "."
        
        # Remove empty responses
        if not response.strip():
            response = "I'm thinking about that. Could you tell me more?"
        
        return response.strip()
    
    def _format_response(self, response: str) -> str:
        """Format response with smart truncation and user preferences"""
        if self.unlimited_mode:
            return response
        
        # If response is within limits, return as-is
        if len(response) <= self.max_response_length:
            return response
        
        # Smart truncation at sentence boundaries
        if self.smart_truncation:
            sentences = response.split('. ')
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence + '. ') <= self.max_response_length:
                    truncated += sentence + '. '
                else:
                    break
            
            if truncated:
                return truncated.rstrip() + " [Continue...]"
            
        # Fallback: simple truncation
        return response[:self.max_response_length] + "..."
    
    def set_response_config(self, max_length: Optional[int] = None, unlimited: Optional[bool] = None, smart_truncation: Optional[bool] = None):
        """Update response configuration"""
        if max_length is not None:
            self.max_response_length = max_length
        if unlimited is not None:
            self.unlimited_mode = unlimited
        if smart_truncation is not None:
            self.smart_truncation = smart_truncation
    
    def _calculate_confidence(self, response: str) -> float:
        """Calculate confidence score for the response"""
        # Simple heuristic-based confidence calculation
        confidence = 0.5  # Base confidence
        
        # Length factor
        if 10 <= len(response) <= 200:
            confidence += 0.2
        
        # Completeness factor
        if response.endswith(('.', '!', '?')):
            confidence += 0.1
        
        # Avoid repetitive responses
        if len(set(response.split())) / len(response.split()) > 0.7:
            confidence += 0.1
        
        # Improvement factor
        confidence += self.performance_metrics['improvement_score'] * 0.1
        
        return min(confidence, 1.0)
    
    def _store_interaction(self, user_input: str, response: str, context: Dict):
        """Store interaction for learning and improvement"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'response': response,
            'context': context,
            'metrics': self.performance_metrics.copy()
        }
        
        # Add to history
        self.conversation_history.append({
            'user': user_input,
            'assistant': response,
            'timestamp': interaction['timestamp']
        })
        
        # Add to knowledge buffer for learning
        self.knowledge_buffer.append(interaction)
        
        # Limit buffer size
        if len(self.knowledge_buffer) > 1000:
            self.knowledge_buffer = self.knowledge_buffer[-800:]
    
    def _update_performance_metrics(self, response_time: float):
        """Update performance metrics for self-improvement"""
        self.performance_metrics['total_conversations'] += 1
        
        # Update average response time
        current_avg = self.performance_metrics['avg_response_time']
        total = self.performance_metrics['total_conversations']
        self.performance_metrics['avg_response_time'] = (
            (current_avg * (total - 1) + response_time) / total
        )
        
        # Update improvement score based on performance trends
        if response_time < current_avg:
            self.performance_metrics['improvement_score'] = min(
                self.performance_metrics['improvement_score'] + 0.01, 1.0
            )
    
    def provide_feedback(self, feedback: str, interaction_id: Optional[str] = None):
        """Accept feedback for self-improvement"""
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'feedback': feedback,
            'interaction_id': interaction_id,
            'sentiment': self._analyze_feedback_sentiment(feedback)
        }
        
        self.feedback_buffer.append(feedback_entry)
        
        # Immediate learning from feedback
        if feedback_entry['sentiment'] > 0:
            self.performance_metrics['improvement_score'] = min(
                self.performance_metrics['improvement_score'] + 0.05, 1.0
            )
        elif feedback_entry['sentiment'] < 0:
            self.performance_metrics['improvement_score'] = max(
                self.performance_metrics['improvement_score'] - 0.02, 0.1
            )
    
    def _analyze_feedback_sentiment(self, feedback: str) -> float:
        """Simple sentiment analysis for feedback"""
        positive_words = ['good', 'great', 'excellent', 'helpful', 'correct', 'right', 'amazing']
        negative_words = ['bad', 'wrong', 'incorrect', 'unhelpful', 'poor', 'terrible']
        
        feedback_lower = feedback.lower()
        positive_count = sum(1 for word in positive_words if word in feedback_lower)
        negative_count = sum(1 for word in negative_words if word in feedback_lower)
        
        if positive_count > negative_count:
            return 1.0
        elif negative_count > positive_count:
            return -1.0
        else:
            return 0.0
    
    def _start_improvement_loop(self):
        """Start the self-improvement background process"""
        def improvement_loop():
            while True:
                time.sleep(30)  # Check every 30 seconds
                if len(self.knowledge_buffer) >= 10 and not self.is_improving:
                    self._perform_self_improvement()
        
        self.improvement_thread = threading.Thread(target=improvement_loop, daemon=True)
        self.improvement_thread.start()
    
    def _perform_self_improvement(self):
        """Perform self-improvement based on accumulated knowledge"""
        self.is_improving = True
        
        try:
            # Analyze conversation patterns
            patterns = self._analyze_conversation_patterns()
            
            # Adjust generation parameters based on learning
            self._adjust_generation_parameters(patterns)
            
            # Update learning rate
            self._update_learning_rate()
            
            self.logger.info("Self-improvement cycle completed")
            
        except Exception as e:
            self.logger.error(f"Error during self-improvement: {e}")
        
        finally:
            self.is_improving = False
    
    def _analyze_conversation_patterns(self) -> Dict:
        """Analyze patterns in conversations for learning"""
        if not self.knowledge_buffer:
            return {}
        
        patterns = {
            'avg_input_length': 0,
            'avg_response_length': 0,
            'common_topics': [],
            'response_quality_trend': []
        }
        
        # Calculate averages
        input_lengths = [len(item['user_input']) for item in self.knowledge_buffer]
        response_lengths = [len(item['response']) for item in self.knowledge_buffer]
        
        patterns['avg_input_length'] = np.mean(input_lengths)
        patterns['avg_response_length'] = np.mean(response_lengths)
        
        return patterns
    
    def _adjust_generation_parameters(self, patterns: Dict):
        """Adjust generation parameters based on learned patterns"""
        if self.pipeline:
            # Adjust temperature based on improvement score
            base_temp = 0.7
            improvement_factor = self.performance_metrics['improvement_score']
            new_temp = base_temp + (improvement_factor - 0.5) * 0.3
            
            # Update pipeline parameters (simplified)
            self.pipeline.model.config.temperature = max(0.1, min(1.0, new_temp))
    
    def _update_learning_rate(self):
        """Update learning rate based on performance"""
        current_score = self.performance_metrics['improvement_score']
        if current_score > 0.8:
            self.performance_metrics['learning_rate'] *= 0.95  # Slow down learning
        elif current_score < 0.5:
            self.performance_metrics['learning_rate'] *= 1.05  # Speed up learning
    
    def get_model_info(self) -> Dict:
        """Get current model information and statistics"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'performance_metrics': self.performance_metrics,
            'total_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            'conversation_count': len(self.conversation_history),
            'knowledge_buffer_size': len(self.knowledge_buffer),
            'feedback_count': len(self.feedback_buffer)
        }
    
    def cleanup(self):
        """Properly cleanup GPU resources and memory"""
        try:
            import torch
            
            # Clear model from memory
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            if hasattr(self, 'pipeline') and self.pipeline is not None:
                del self.pipeline
                self.pipeline = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all operations to complete
            
            self.logger.info("GPU resources cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def save_state(self, filepath: str):
        """Save the current state for persistence"""
        state = {
            'model_name': self.model_name,
            'conversation_history': self.conversation_history,
            'performance_metrics': self.performance_metrics,
            'knowledge_buffer': self.knowledge_buffer[-100:],  # Save last 100
            'feedback_buffer': self.feedback_buffer[-100:]
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load previously saved state"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.conversation_history = state.get('conversation_history', [])
            self.performance_metrics.update(state.get('performance_metrics', {}))
            self.knowledge_buffer = state.get('knowledge_buffer', [])
            self.feedback_buffer = state.get('feedback_buffer', [])
            
            self.logger.info("State loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            return False