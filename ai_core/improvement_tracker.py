"""
Improvement Tracker - Monitors and analyzes self-improvement progress
Tracks learning metrics, performance trends, and optimization opportunities
"""

import json
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import logging
from collections import defaultdict, deque
import threading
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import sqlite3

@dataclass
class PerformanceMetric:
    """Represents a performance metric at a point in time"""
    timestamp: str
    metric_name: str
    value: float
    context: Dict[str, Any]
    improvement_score: float
    confidence: float

@dataclass
class LearningMilestone:
    """Represents a significant learning achievement"""
    timestamp: str
    milestone_type: str
    description: str
    metrics_snapshot: Dict[str, float]
    significance_score: float

class ImprovementTracker:
    def __init__(self, data_dir: str = "./data", config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        self.config = config or {}
        
        # Performance tracking
        self.performance_history = defaultdict(deque)
        self.learning_curves = defaultdict(list)
        self.improvement_trends = {}
        self.baseline_metrics = {}
        
        # Milestone tracking
        self.milestones = []
        self.skill_progression = defaultdict(list)
        self.achievement_thresholds = {
            'conversation_mastery': 0.85,
            'reasoning_excellence': 0.80,
            'creativity_breakthrough': 0.75,
            'knowledge_expert': 0.90,
            'learning_efficiency': 0.80
        }
        
        # Analytics components
        self.metrics_buffer = deque(maxlen=10000)
        self.trend_analyzer = TrendAnalyzer()
        self.optimization_engine = OptimizationEngine()
        
        # Database for persistent tracking
        self.tracking_db_path = os.path.join(data_dir, "improvement_tracking.db")
        
        # Reporting and visualization
        self.report_generator = ReportGenerator(data_dir)
        self.visualization_engine = VisualizationEngine(data_dir)
        
        # Initialize system
        self._initialize_tracking_database()
        self._load_baseline_metrics()
        self._start_analysis_loop()
    
    def _initialize_tracking_database(self):
        """Initialize database for tracking improvement metrics"""
        os.makedirs(self.data_dir, exist_ok=True)
        
        try:
            with sqlite3.connect(self.tracking_db_path) as conn:
                # Performance metrics table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        value REAL NOT NULL,
                        context TEXT,
                        improvement_score REAL,
                        confidence REAL,
                        session_id TEXT
                    )
                ''')
                
                # Learning milestones table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS learning_milestones (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        milestone_type TEXT NOT NULL,
                        description TEXT NOT NULL,
                        metrics_snapshot TEXT,
                        significance_score REAL
                    )
                ''')
                
                # Optimization suggestions table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS optimization_suggestions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        suggestion_type TEXT NOT NULL,
                        description TEXT NOT NULL,
                        expected_impact REAL,
                        implementation_status TEXT DEFAULT 'pending',
                        actual_impact REAL
                    )
                ''')
                
                # Create indexes
                conn.execute('CREATE INDEX IF NOT EXISTS idx_metric_name ON performance_metrics(metric_name)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON performance_metrics(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_milestone_type ON learning_milestones(milestone_type)')
                
                conn.commit()
            
            self.logger.info("Improvement tracking database initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing tracking database: {e}")
    
    def record_performance_metric(self, metric_name: str, value: float, 
                                 context: Dict = None, session_id: str = None) -> bool:
        """Record a performance metric"""
        try:
            timestamp = datetime.now().isoformat()
            context = context or {}
            
            # Calculate improvement score
            improvement_score = self._calculate_improvement_score(metric_name, value)
            
            # Calculate confidence based on consistency
            confidence = self._calculate_metric_confidence(metric_name, value)
            
            # Create metric object
            metric = PerformanceMetric(
                timestamp=timestamp,
                metric_name=metric_name,
                value=value,
                context=context,
                improvement_score=improvement_score,
                confidence=confidence
            )
            
            # Store in memory
            self.metrics_buffer.append(metric)
            self.performance_history[metric_name].append((timestamp, value))
            
            # Limit history size
            if len(self.performance_history[metric_name]) > 1000:
                self.performance_history[metric_name] = deque(
                    list(self.performance_history[metric_name])[-800:], maxlen=1000
                )
            
            # Store in database
            with sqlite3.connect(self.tracking_db_path) as conn:
                conn.execute('''
                    INSERT INTO performance_metrics 
                    (timestamp, metric_name, value, context, improvement_score, confidence, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (timestamp, metric_name, value, json.dumps(context), 
                      improvement_score, confidence, session_id))
                conn.commit()
            
            # Check for milestones
            self._check_for_milestones(metric_name, value, improvement_score)
            
            # Update learning curves
            self._update_learning_curves(metric_name, value, timestamp)
            
            self.logger.debug(f"Recorded metric: {metric_name} = {value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error recording performance metric: {e}")
            return False
    
    def _calculate_improvement_score(self, metric_name: str, current_value: float) -> float:
        """Calculate improvement score compared to baseline and recent performance"""
        try:
            # Get baseline
            baseline = self.baseline_metrics.get(metric_name, current_value)
            
            # Get recent values
            recent_values = []
            if metric_name in self.performance_history:
                recent_data = list(self.performance_history[metric_name])[-10:]  # Last 10 values
                recent_values = [value for _, value in recent_data]
            
            if not recent_values:
                return 0.5  # Neutral score for first measurement
            
            # Calculate improvement from baseline
            baseline_improvement = (current_value - baseline) / max(abs(baseline), 0.1)
            
            # Calculate improvement from recent average
            recent_avg = np.mean(recent_values)
            recent_improvement = (current_value - recent_avg) / max(abs(recent_avg), 0.1)
            
            # Combine scores
            improvement_score = (baseline_improvement * 0.3 + recent_improvement * 0.7)
            
            # Normalize to 0-1 range
            improvement_score = max(0.0, min(1.0, (improvement_score + 1.0) / 2.0))
            
            return improvement_score
            
        except Exception as e:
            self.logger.error(f"Error calculating improvement score: {e}")
            return 0.5
    
    def _calculate_metric_confidence(self, metric_name: str, current_value: float) -> float:
        """Calculate confidence based on measurement consistency"""
        try:
            if metric_name not in self.performance_history:
                return 0.5  # Neutral confidence for first measurement
            
            recent_data = list(self.performance_history[metric_name])[-20:]  # Last 20 values
            if len(recent_data) < 3:
                return 0.5
            
            recent_values = [value for _, value in recent_data]
            
            # Calculate coefficient of variation
            mean_value = np.mean(recent_values)
            std_value = np.std(recent_values)
            
            if mean_value == 0:
                return 0.5
            
            cv = std_value / abs(mean_value)
            
            # Convert to confidence (lower variation = higher confidence)
            confidence = max(0.1, min(1.0, 1.0 - cv))
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating metric confidence: {e}")
            return 0.5
    
    def _check_for_milestones(self, metric_name: str, value: float, improvement_score: float):
        """Check if current performance represents a milestone"""
        try:
            milestone_triggered = False
            milestone_type = None
            description = None
            significance_score = 0.0
            
            # Check achievement thresholds
            for achievement, threshold in self.achievement_thresholds.items():
                if metric_name.startswith(achievement.split('_')[0]) and value >= threshold:
                    if not self._milestone_already_achieved(achievement):
                        milestone_type = achievement
                        description = f"Achieved {achievement.replace('_', ' ')} with score {value:.3f}"
                        significance_score = value
                        milestone_triggered = True
                        break
            
            # Check for significant improvements
            if improvement_score > 0.8 and not milestone_triggered:
                milestone_type = "significant_improvement"
                description = f"Significant improvement in {metric_name}: {value:.3f}"
                significance_score = improvement_score
                milestone_triggered = True
            
            # Check for consistency milestones
            if self._check_consistency_milestone(metric_name, value):
                milestone_type = "consistency_achievement"
                description = f"Consistent high performance in {metric_name}"
                significance_score = 0.8
                milestone_triggered = True
            
            # Record milestone if triggered
            if milestone_triggered:
                self._record_milestone(milestone_type, description, significance_score)
            
        except Exception as e:
            self.logger.error(f"Error checking for milestones: {e}")
    
    def _milestone_already_achieved(self, achievement: str) -> bool:
        """Check if milestone has already been achieved"""
        for milestone in self.milestones:
            if milestone.milestone_type == achievement:
                return True
        return False
    
    def _check_consistency_milestone(self, metric_name: str, current_value: float) -> bool:
        """Check if recent performance shows consistent high quality"""
        try:
            if metric_name not in self.performance_history:
                return False
            
            recent_data = list(self.performance_history[metric_name])[-10:]  # Last 10 values
            if len(recent_data) < 10:
                return False
            
            recent_values = [value for _, value in recent_data]
            
            # Check if all recent values are above threshold
            threshold = 0.7
            all_above_threshold = all(value >= threshold for value in recent_values)
            
            # Check if improvement is stable
            improvement_trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
            stable_or_improving = improvement_trend >= -0.01  # Small negative slope allowed
            
            return all_above_threshold and stable_or_improving
            
        except Exception as e:
            self.logger.error(f"Error checking consistency milestone: {e}")
            return False
    
    def _record_milestone(self, milestone_type: str, description: str, significance_score: float):
        """Record a learning milestone"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Get current metrics snapshot
            metrics_snapshot = self._get_current_metrics_snapshot()
            
            # Create milestone object
            milestone = LearningMilestone(
                timestamp=timestamp,
                milestone_type=milestone_type,
                description=description,
                metrics_snapshot=metrics_snapshot,
                significance_score=significance_score
            )
            
            # Store in memory
            self.milestones.append(milestone)
            
            # Store in database
            with sqlite3.connect(self.tracking_db_path) as conn:
                conn.execute('''
                    INSERT INTO learning_milestones 
                    (timestamp, milestone_type, description, metrics_snapshot, significance_score)
                    VALUES (?, ?, ?, ?, ?)
                ''', (timestamp, milestone_type, description, 
                      json.dumps(metrics_snapshot), significance_score))
                conn.commit()
            
            self.logger.info(f"Milestone achieved: {description}")
            
        except Exception as e:
            self.logger.error(f"Error recording milestone: {e}")
    
    def _get_current_metrics_snapshot(self) -> Dict[str, float]:
        """Get snapshot of current performance metrics"""
        snapshot = {}
        
        for metric_name, history in self.performance_history.items():
            if history:
                # Get most recent value
                snapshot[metric_name] = history[-1][1]
        
        return snapshot
    
    def _update_learning_curves(self, metric_name: str, value: float, timestamp: str):
        """Update learning curves for visualization and analysis"""
        try:
            # Add to learning curve data
            if metric_name not in self.learning_curves:
                self.learning_curves[metric_name] = []
            
            self.learning_curves[metric_name].append({
                'timestamp': timestamp,
                'value': value,
                'moving_average': self._calculate_moving_average(metric_name, window=10)
            })
            
            # Limit curve data
            if len(self.learning_curves[metric_name]) > 1000:
                self.learning_curves[metric_name] = self.learning_curves[metric_name][-800:]
            
        except Exception as e:
            self.logger.error(f"Error updating learning curves: {e}")
    
    def _calculate_moving_average(self, metric_name: str, window: int = 10) -> float:
        """Calculate moving average for smoothed trend analysis"""
        try:
            if metric_name not in self.performance_history:
                return 0.0
            
            recent_values = [value for _, value in list(self.performance_history[metric_name])[-window:]]
            
            if not recent_values:
                return 0.0
            
            return np.mean(recent_values)
            
        except Exception as e:
            self.logger.error(f"Error calculating moving average: {e}")
            return 0.0
    
    def _load_baseline_metrics(self):
        """Load or establish baseline performance metrics"""
        try:
            baseline_file = os.path.join(self.data_dir, "baseline_metrics.json")
            
            if os.path.exists(baseline_file):
                with open(baseline_file, 'r') as f:
                    self.baseline_metrics = json.load(f)
                self.logger.info("Loaded baseline metrics")
            else:
                # Establish baselines from initial performance
                self.baseline_metrics = {
                    'conversation_quality': 0.5,
                    'reasoning_accuracy': 0.5,
                    'creativity_score': 0.5,
                    'knowledge_retention': 0.5,
                    'learning_speed': 0.5,
                    'response_coherence': 0.5,
                    'improvement_rate': 0.5
                }
                self._save_baseline_metrics()
                self.logger.info("Established initial baseline metrics")
            
        except Exception as e:
            self.logger.error(f"Error loading baseline metrics: {e}")
            self.baseline_metrics = {}
    
    def _save_baseline_metrics(self):
        """Save baseline metrics to file"""
        try:
            baseline_file = os.path.join(self.data_dir, "baseline_metrics.json")
            os.makedirs(os.path.dirname(baseline_file), exist_ok=True)
            
            with open(baseline_file, 'w') as f:
                json.dump(self.baseline_metrics, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error saving baseline metrics: {e}")
    
    def get_improvement_analysis(self, days_back: int = 30) -> Dict[str, Any]:
        """Get comprehensive improvement analysis"""
        try:
            analysis = {
                'summary': {},
                'trends': {},
                'milestones': [],
                'recommendations': [],
                'performance_overview': {}
            }
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Analyze each metric
            for metric_name, history in self.performance_history.items():
                if not history:
                    continue
                
                # Filter data by date range
                period_data = [
                    (timestamp, value) for timestamp, value in history
                    if start_date <= datetime.fromisoformat(timestamp) <= end_date
                ]
                
                if len(period_data) < 2:
                    continue
                
                values = [value for _, value in period_data]
                
                # Calculate trend
                trend = self.trend_analyzer.calculate_trend(values)
                improvement_rate = self.trend_analyzer.calculate_improvement_rate(values)
                
                analysis['trends'][metric_name] = {
                    'trend_direction': trend['direction'],
                    'trend_strength': trend['strength'],
                    'improvement_rate': improvement_rate,
                    'current_value': values[-1],
                    'period_average': np.mean(values),
                    'volatility': np.std(values)
                }
                
                # Performance overview
                analysis['performance_overview'][metric_name] = {
                    'current': values[-1],
                    'baseline': self.baseline_metrics.get(metric_name, values[0]),
                    'best': max(values),
                    'average': np.mean(values),
                    'improvement_from_baseline': values[-1] - self.baseline_metrics.get(metric_name, values[0])
                }
            
            # Recent milestones
            recent_milestones = [
                milestone for milestone in self.milestones
                if start_date <= datetime.fromisoformat(milestone.timestamp) <= end_date
            ]
            analysis['milestones'] = [asdict(milestone) for milestone in recent_milestones]
            
            # Generate recommendations
            analysis['recommendations'] = self.optimization_engine.generate_recommendations(
                analysis['trends'], analysis['performance_overview']
            )
            
            # Summary statistics
            analysis['summary'] = self._generate_improvement_summary(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error generating improvement analysis: {e}")
            return {}
    
    def _generate_improvement_summary(self, analysis: Dict) -> Dict:
        """Generate summary of improvement progress"""
        try:
            summary = {
                'overall_improvement_score': 0.0,
                'strongest_areas': [],
                'improvement_areas': [],
                'milestone_count': len(analysis['milestones']),
                'trend_summary': {}
            }
            
            # Calculate overall improvement score
            improvement_scores = []
            for metric_name, trend_data in analysis['trends'].items():
                improvement_scores.append(trend_data['improvement_rate'])
            
            if improvement_scores:
                summary['overall_improvement_score'] = np.mean(improvement_scores)
            
            # Identify strongest and weakest areas
            performance_data = analysis['performance_overview']
            sorted_by_current = sorted(
                performance_data.items(), 
                key=lambda x: x[1]['current'], 
                reverse=True
            )
            
            summary['strongest_areas'] = [item[0] for item in sorted_by_current[:3]]
            
            sorted_by_improvement = sorted(
                performance_data.items(),
                key=lambda x: x[1]['improvement_from_baseline']
            )
            
            summary['improvement_areas'] = [item[0] for item in sorted_by_improvement[:3]]
            
            # Trend summary
            positive_trends = sum(1 for trend in analysis['trends'].values() 
                                if trend['trend_direction'] == 'improving')
            negative_trends = sum(1 for trend in analysis['trends'].values() 
                                if trend['trend_direction'] == 'declining')
            stable_trends = sum(1 for trend in analysis['trends'].values() 
                              if trend['trend_direction'] == 'stable')
            
            summary['trend_summary'] = {
                'improving': positive_trends,
                'declining': negative_trends,
                'stable': stable_trends
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating improvement summary: {e}")
            return {}
    
    def generate_progress_report(self, format: str = 'json') -> str:
        """Generate comprehensive progress report"""
        try:
            report_data = {
                'report_generated': datetime.now().isoformat(),
                'improvement_analysis': self.get_improvement_analysis(),
                'learning_milestones': [asdict(m) for m in self.milestones[-10:]],  # Recent 10
                'performance_trends': self._get_performance_trends(),
                'optimization_suggestions': self._get_optimization_suggestions(),
                'future_projections': self._generate_future_projections()
            }
            
            if format.lower() == 'json':
                return json.dumps(report_data, indent=2, default=str)
            elif format.lower() == 'html':
                return self.report_generator.generate_html_report(report_data)
            else:
                return str(report_data)
                
        except Exception as e:
            self.logger.error(f"Error generating progress report: {e}")
            return "{}"
    
    def _get_performance_trends(self) -> Dict:
        """Get detailed performance trends for all metrics"""
        trends = {}
        
        for metric_name, history in self.performance_history.items():
            if len(history) < 5:
                continue
            
            values = [value for _, value in list(history)]
            timestamps = [timestamp for timestamp, _ in list(history)]
            
            trends[metric_name] = {
                'values': values[-50:],  # Last 50 values
                'timestamps': timestamps[-50:],
                'trend_analysis': self.trend_analyzer.detailed_trend_analysis(values),
                'moving_averages': {
                    'short_term': self._calculate_moving_average(metric_name, 5),
                    'medium_term': self._calculate_moving_average(metric_name, 20),
                    'long_term': self._calculate_moving_average(metric_name, 50)
                }
            }
        
        return trends
    
    def _get_optimization_suggestions(self) -> List[Dict]:
        """Get optimization suggestions from database"""
        try:
            suggestions = []
            
            with sqlite3.connect(self.tracking_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM optimization_suggestions 
                    WHERE implementation_status = 'pending'
                    ORDER BY expected_impact DESC, timestamp DESC
                    LIMIT 10
                ''')
                
                suggestions = [dict(row) for row in cursor.fetchall()]
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error getting optimization suggestions: {e}")
            return []
    
    def _generate_future_projections(self) -> Dict:
        """Generate projections for future performance"""
        try:
            projections = {}
            
            for metric_name, history in self.performance_history.items():
                if len(history) < 10:
                    continue
                
                values = [value for _, value in list(history)[-30:]]  # Last 30 values
                
                # Simple linear projection
                if len(values) >= 5:
                    x = np.arange(len(values))
                    coeffs = np.polyfit(x, values, 1)
                    
                    # Project next 10 time periods
                    future_x = np.arange(len(values), len(values) + 10)
                    future_values = np.polyval(coeffs, future_x)
                    
                    projections[metric_name] = {
                        'projected_values': future_values.tolist(),
                        'trend_slope': coeffs[0],
                        'confidence': min(0.9, max(0.1, 1.0 - np.std(values) / max(np.mean(values), 0.1)))
                    }
            
            return projections
            
        except Exception as e:
            self.logger.error(f"Error generating future projections: {e}")
            return {}
    
    def _start_analysis_loop(self):
        """Start background analysis and optimization loop"""
        def analysis_loop():
            while True:
                try:
                    time.sleep(1800)  # Run every 30 minutes
                    
                    # Update trend analysis
                    self._update_trend_analysis()
                    
                    # Generate optimization suggestions
                    self._generate_optimization_suggestions()
                    
                    # Clean up old data
                    self._cleanup_old_data()
                    
                except Exception as e:
                    self.logger.error(f"Error in analysis loop: {e}")
        
        analysis_thread = threading.Thread(target=analysis_loop, daemon=True)
        analysis_thread.start()
    
    def _update_trend_analysis(self):
        """Update trend analysis for all metrics"""
        for metric_name in self.performance_history:
            self.improvement_trends[metric_name] = self.trend_analyzer.analyze_metric_trend(
                metric_name, list(self.performance_history[metric_name])
            )
    
    def _generate_optimization_suggestions(self):
        """Generate and store optimization suggestions"""
        try:
            suggestions = self.optimization_engine.analyze_and_suggest(
                self.performance_history, self.improvement_trends
            )
            
            with sqlite3.connect(self.tracking_db_path) as conn:
                for suggestion in suggestions:
                    conn.execute('''
                        INSERT INTO optimization_suggestions 
                        (timestamp, suggestion_type, description, expected_impact)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        datetime.now().isoformat(),
                        suggestion['type'],
                        suggestion['description'],
                        suggestion['expected_impact']
                    ))
                conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error generating optimization suggestions: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old tracking data to maintain performance"""
        try:
            # Keep only last 90 days of data in database
            cutoff_date = (datetime.now() - timedelta(days=90)).isoformat()
            
            with sqlite3.connect(self.tracking_db_path) as conn:
                conn.execute('''
                    DELETE FROM performance_metrics 
                    WHERE timestamp < ?
                ''', (cutoff_date,))
                
                conn.execute('''
                    DELETE FROM optimization_suggestions 
                    WHERE timestamp < ? AND implementation_status = 'completed'
                ''', (cutoff_date,))
                
                conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")

class TrendAnalyzer:
    """Analyzes performance trends and patterns"""
    
    def calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend direction and strength"""
        if len(values) < 2:
            return {'direction': 'unknown', 'strength': 0.0}
        
        # Linear regression
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        slope = coeffs[0]
        
        # Determine direction
        if slope > 0.01:
            direction = 'improving'
        elif slope < -0.01:
            direction = 'declining'
        else:
            direction = 'stable'
        
        # Calculate strength (correlation coefficient)
        correlation = abs(np.corrcoef(x, values)[0, 1])
        strength = correlation if not np.isnan(correlation) else 0.0
        
        return {
            'direction': direction,
            'strength': strength,
            'slope': slope,
            'correlation': correlation
        }
    
    def calculate_improvement_rate(self, values: List[float]) -> float:
        """Calculate rate of improvement"""
        if len(values) < 2:
            return 0.0
        
        # Compare recent period to earlier period
        if len(values) >= 10:
            recent = np.mean(values[-5:])
            earlier = np.mean(values[:5])
        else:
            recent = values[-1]
            earlier = values[0]
        
        if earlier == 0:
            return 0.0
        
        return (recent - earlier) / abs(earlier)
    
    def detailed_trend_analysis(self, values: List[float]) -> Dict[str, Any]:
        """Perform detailed trend analysis"""
        analysis = {
            'basic_trend': self.calculate_trend(values),
            'volatility': np.std(values) if values else 0.0,
            'momentum': self._calculate_momentum(values),
            'seasonality': self._detect_seasonality(values),
            'anomalies': self._detect_anomalies(values)
        }
        
        return analysis
    
    def _calculate_momentum(self, values: List[float]) -> float:
        """Calculate momentum of the trend"""
        if len(values) < 5:
            return 0.0
        
        # Compare rate of change in recent vs earlier periods
        recent_changes = np.diff(values[-5:])
        earlier_changes = np.diff(values[:5]) if len(values) >= 10 else np.diff(values[:-5])
        
        recent_momentum = np.mean(recent_changes) if len(recent_changes) > 0 else 0.0
        earlier_momentum = np.mean(earlier_changes) if len(earlier_changes) > 0 else 0.0
        
        return recent_momentum - earlier_momentum
    
    def _detect_seasonality(self, values: List[float]) -> Dict[str, Any]:
        """Detect seasonal patterns in the data"""
        # Simplified seasonality detection
        if len(values) < 20:
            return {'has_seasonality': False, 'period': None}
        
        # Check for periodic patterns (simplified)
        autocorrelations = []
        for lag in range(1, min(10, len(values) // 2)):
            corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
            if not np.isnan(corr):
                autocorrelations.append((lag, abs(corr)))
        
        if autocorrelations:
            best_lag, best_corr = max(autocorrelations, key=lambda x: x[1])
            if best_corr > 0.5:
                return {'has_seasonality': True, 'period': best_lag, 'strength': best_corr}
        
        return {'has_seasonality': False, 'period': None}
    
    def _detect_anomalies(self, values: List[float]) -> List[int]:
        """Detect anomalous values in the series"""
        if len(values) < 5:
            return []
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        anomalies = []
        for i, value in enumerate(values):
            z_score = abs(value - mean_val) / max(std_val, 0.01)
            if z_score > 2.5:  # 2.5 standard deviations
                anomalies.append(i)
        
        return anomalies
    
    def analyze_metric_trend(self, metric_name: str, history_data: List[Tuple[str, float]]) -> Dict:
        """Analyze trend for a specific metric"""
        if not history_data:
            return {}
        
        values = [value for _, value in history_data]
        timestamps = [timestamp for timestamp, _ in history_data]
        
        return {
            'metric_name': metric_name,
            'trend_analysis': self.detailed_trend_analysis(values),
            'recent_performance': values[-10:] if len(values) >= 10 else values,
            'data_quality': {
                'completeness': len(values),
                'consistency': 1.0 - (np.std(values) / max(np.mean(values), 0.01))
            }
        }

class OptimizationEngine:
    """Generates optimization suggestions based on performance analysis"""
    
    def generate_recommendations(self, trends: Dict, performance_overview: Dict) -> List[Dict]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Analyze declining trends
        for metric_name, trend_data in trends.items():
            if trend_data['trend_direction'] == 'declining':
                recommendations.append({
                    'type': 'improvement_needed',
                    'metric': metric_name,
                    'description': f"Focus on improving {metric_name} - showing declining trend",
                    'priority': 'high' if trend_data['trend_strength'] > 0.5 else 'medium',
                    'suggested_actions': self._get_metric_specific_suggestions(metric_name)
                })
        
        # Analyze volatile metrics
        for metric_name, trend_data in trends.items():
            if trend_data['volatility'] > 0.3:  # High volatility threshold
                recommendations.append({
                    'type': 'stability_improvement',
                    'metric': metric_name,
                    'description': f"Work on stabilizing {metric_name} performance",
                    'priority': 'medium',
                    'suggested_actions': ['Identify sources of variability', 'Implement consistency measures']
                })
        
        # Identify optimization opportunities
        best_performing = max(performance_overview.items(), key=lambda x: x[1]['current'])
        recommendations.append({
            'type': 'leverage_strength',
            'metric': best_performing[0],
            'description': f"Leverage strength in {best_performing[0]} to improve other areas",
            'priority': 'low',
            'suggested_actions': ['Apply successful patterns to other metrics']
        })
        
        return recommendations
    
    def _get_metric_specific_suggestions(self, metric_name: str) -> List[str]:
        """Get specific suggestions for different metrics"""
        suggestions_map = {
            'conversation': [
                'Practice active listening patterns',
                'Improve response coherence',
                'Enhance contextual understanding'
            ],
            'reasoning': [
                'Break down complex problems systematically',
                'Improve logical flow in responses',
                'Practice analytical thinking patterns'
            ],
            'creativity': [
                'Explore diverse response approaches',
                'Practice novel concept combinations',
                'Encourage innovative thinking patterns'
            ],
            'learning': [
                'Increase knowledge retention strategies',
                'Improve pattern recognition',
                'Enhance memory consolidation'
            ]
        }
        
        # Find matching suggestions
        for key, suggestions in suggestions_map.items():
            if key in metric_name.lower():
                return suggestions
        
        return ['Analyze performance patterns', 'Focus on consistent improvement']
    
    def analyze_and_suggest(self, performance_history: Dict, improvement_trends: Dict) -> List[Dict]:
        """Analyze performance and generate optimization suggestions"""
        suggestions = []
        
        # Analyze correlation between metrics
        correlations = self._analyze_metric_correlations(performance_history)
        
        for (metric1, metric2), correlation in correlations.items():
            if correlation > 0.7:  # Strong positive correlation
                suggestions.append({
                    'type': 'leverage_correlation',
                    'description': f"Improving {metric1} will likely improve {metric2}",
                    'expected_impact': correlation
                })
        
        # Analyze learning efficiency
        efficiency_suggestions = self._analyze_learning_efficiency(performance_history)
        suggestions.extend(efficiency_suggestions)
        
        return suggestions
    
    def _analyze_metric_correlations(self, performance_history: Dict) -> Dict[Tuple[str, str], float]:
        """Analyze correlations between different metrics"""
        correlations = {}
        metrics = list(performance_history.keys())
        
        for i, metric1 in enumerate(metrics):
            for metric2 in metrics[i+1:]:
                if len(performance_history[metric1]) > 5 and len(performance_history[metric2]) > 5:
                    # Get overlapping time periods
                    values1 = [value for _, value in list(performance_history[metric1])[-20:]]
                    values2 = [value for _, value in list(performance_history[metric2])[-20:]]
                    
                    min_len = min(len(values1), len(values2))
                    if min_len > 3:
                        corr = np.corrcoef(values1[:min_len], values2[:min_len])[0, 1]
                        if not np.isnan(corr):
                            correlations[(metric1, metric2)] = abs(corr)
        
        return correlations
    
    def _analyze_learning_efficiency(self, performance_history: Dict) -> List[Dict]:
        """Analyze learning efficiency and suggest improvements"""
        suggestions = []
        
        # Calculate learning rates for each metric
        learning_rates = {}
        for metric_name, history in performance_history.items():
            if len(history) > 10:
                values = [value for _, value in history]
                # Simple learning rate calculation
                early_avg = np.mean(values[:5])
                recent_avg = np.mean(values[-5:])
                learning_rate = (recent_avg - early_avg) / len(values)
                learning_rates[metric_name] = learning_rate
        
        # Identify slow learning metrics
        if learning_rates:
            avg_learning_rate = np.mean(list(learning_rates.values()))
            for metric_name, rate in learning_rates.items():
                if rate < avg_learning_rate * 0.5:  # Significantly slower
                    suggestions.append({
                        'type': 'accelerate_learning',
                        'description': f"Focus on accelerating learning in {metric_name}",
                        'expected_impact': 0.6
                    })
        
        return suggestions

class ReportGenerator:
    """Generates formatted reports"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    def generate_html_report(self, report_data: Dict) -> str:
        """Generate HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Improvement Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ margin: 10px 0; }}
                .milestone {{ background: #f0f8ff; padding: 10px; margin: 5px 0; }}
                .recommendation {{ background: #fff8dc; padding: 10px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <h1>AI Self-Improvement Report</h1>
            <p>Generated: {report_data['report_generated']}</p>
            
            <h2>Improvement Summary</h2>
            <div class="summary">
                <!-- Summary content -->
            </div>
            
            <h2>Recent Milestones</h2>
            <div class="milestones">
                <!-- Milestones content -->
            </div>
            
            <h2>Optimization Recommendations</h2>
            <div class="recommendations">
                <!-- Recommendations content -->
            </div>
        </body>
        </html>
        """
        return html

class VisualizationEngine:
    """Creates visualizations for performance tracking"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.plots_dir = os.path.join(data_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def create_performance_charts(self, performance_data: Dict) -> List[str]:
        """Create performance visualization charts"""
        chart_files = []
        
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            
            for metric_name, history in performance_data.items():
                if len(history) < 3:
                    continue
                
                # Extract data
                timestamps = [datetime.fromisoformat(ts) for ts, _ in history]
                values = [value for _, value in history]
                
                # Create plot
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(timestamps, values, marker='o', linewidth=2, markersize=4)
                ax.set_title(f'{metric_name.replace("_", " ").title()} Performance Over Time')
                ax.set_xlabel('Time')
                ax.set_ylabel('Performance Score')
                ax.grid(True, alpha=0.3)
                
                # Add trend line
                x_numeric = np.arange(len(values))
                z = np.polyfit(x_numeric, values, 1)
                p = np.poly1d(z)
                ax.plot(timestamps, p(x_numeric), "--", alpha=0.8, linewidth=2)
                
                # Save plot
                filename = f"{metric_name}_performance.png"
                filepath = os.path.join(self.plots_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                
                chart_files.append(filepath)
            
            return chart_files
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error creating performance charts: {e}")
            return []