// storage/performance_storage.go - Stores historical performance metrics
package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/yourusername/ai-code-assistant/tracking"
)

// PerformanceStorage implements the storage interface for performance tracking
type PerformanceStorage struct {
	db                 *sql.DB
	mutex              sync.RWMutex
	retentionPeriod    time.Duration
	compressionEnabled bool
	batchSize          int
	aggregationWindow  time.Duration
}

// AggregatedMetrics represents compressed performance data for efficient storage
type AggregatedMetrics struct {
	TimeWindow      time.Time `json:"time_window" db:"time_window"`
	WindowSize      string    `json:"window_size" db:"window_size"` // hourly, daily, weekly
	SessionCount    int       `json:"session_count" db:"session_count"`
	RequestCount    int       `json:"request_count" db:"request_count"`
	SuccessCount    int       `json:"success_count" db:"success_count"`
	ErrorCount      int       `json:"error_count" db:"error_count"`
	AvgResponseTime float64   `json:"avg_response_time" db:"avg_response_time"` // milliseconds
	MinResponseTime float64   `json:"min_response_time" db:"min_response_time"` // milliseconds
	MaxResponseTime float64   `json:"max_response_time" db:"max_response_time"` // milliseconds
	P95ResponseTime float64   `json:"p95_response_time" db:"p95_response_time"` // milliseconds
	P99ResponseTime float64   `json:"p99_response_time" db:"p99_response_time"` // milliseconds
	AvgMemoryUsage  int64     `json:"avg_memory_usage" db:"avg_memory_usage"`   // bytes
	PeakMemoryUsage int64     `json:"peak_memory_usage" db:"peak_memory_usage"` // bytes
	AvgCPUUsage     float64   `json:"avg_cpu_usage" db:"avg_cpu_usage"`         // percentage
	PeakCPUUsage    float64   `json:"peak_cpu_usage" db:"peak_cpu_usage"`       // percentage
	TotalTokens     int       `json:"total_tokens" db:"total_tokens"`
	AvgTokensPerReq float64   `json:"avg_tokens_per_req" db:"avg_tokens_per_req"`
	ErrorRate       float64   `json:"error_rate" db:"error_rate"` // percentage
	Throughput      float64   `json:"throughput" db:"throughput"` // requests per second
	CreatedAt       time.Time `json:"created_at" db:"created_at"`
}

// PerformanceAlert represents stored alert data
type PerformanceAlert struct {
	ID             string                 `json:"id" db:"id"`
	SessionID      string                 `json:"session_id" db:"session_id"`
	AlertType      string                 `json:"alert_type" db:"alert_type"` // threshold, anomaly, degradation
	Severity       string                 `json:"severity" db:"severity"`     // warning, critical
	MetricName     string                 `json:"metric_name" db:"metric_name"`
	CurrentValue   float64                `json:"current_value" db:"current_value"`
	ThresholdValue float64                `json:"threshold_value" db:"threshold_value"`
	Message        string                 `json:"message" db:"message"`
	Context        map[string]interface{} `json:"context" db:"context"` // Additional context
	Resolved       bool                   `json:"resolved" db:"resolved"`
	ResolvedAt     *time.Time             `json:"resolved_at" db:"resolved_at"`
	CreatedAt      time.Time              `json:"created_at" db:"created_at"`
	UpdatedAt      time.Time              `json:"updated_at" db:"updated_at"`
}

// PerformanceTrend represents performance trend analysis
type PerformanceTrend struct {
	MetricName    string       `json:"metric_name"`
	TimeFrame     string       `json:"time_frame"`     // last_hour, last_day, last_week, last_month
	TrendType     string       `json:"trend_type"`     // improving, degrading, stable
	ChangePercent float64      `json:"change_percent"` // percentage change
	DataPoints    []TrendPoint `json:"data_points"`
	Confidence    float64      `json:"confidence"` // statistical confidence (0-1)
	Analysis      string       `json:"analysis"`   // Human-readable analysis
}

// PerformanceInsight represents AI-generated performance insights
type PerformanceInsight struct {
	ID              string                 `json:"id" db:"id"`
	Type            string                 `json:"type" db:"type"` // bottleneck, optimization, anomaly
	Title           string                 `json:"title" db:"title"`
	Description     string                 `json:"description" db:"description"`
	Impact          string                 `json:"impact" db:"impact"`         // high, medium, low
	Confidence      float64                `json:"confidence" db:"confidence"` // 0-1
	Recommendations []string               `json:"recommendations" db:"recommendations"`
	Metrics         map[string]interface{} `json:"metrics" db:"metrics"` // Supporting metrics
	TimeFrame       time.Duration          `json:"time_frame" db:"time_frame"`
	CreatedAt       time.Time              `json:"created_at" db:"created_at"`
	RelevantUntil   time.Time              `json:"relevant_until" db:"relevant_until"`
}

// NewPerformanceStorage creates a new performance storage instance
func NewPerformanceStorage(db *sql.DB) (*PerformanceStorage, error) {
	ps := &PerformanceStorage{
		db:                 db,
		retentionPeriod:    90 * 24 * time.Hour, // 90 days
		compressionEnabled: true,
		batchSize:          100,
		aggregationWindow:  time.Hour,
	}

	// Initialize database schema
	if err := ps.initializeSchema(); err != nil {
		return nil, fmt.Errorf("failed to initialize performance storage schema: %w", err)
	}

	// Start background maintenance tasks
	go ps.maintenanceRoutine()

	return ps, nil
}

// initializeSchema creates the necessary database tables
func (ps *PerformanceStorage) initializeSchema() error {
	queries := []string{
		// Raw performance metrics table
		`CREATE TABLE IF NOT EXISTS performance_metrics (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			session_id TEXT NOT NULL,
			request_id TEXT,
			timestamp DATETIME NOT NULL,
			response_time_ms INTEGER,
			ai_provider_time_ms INTEGER,
			context_build_time_ms INTEGER,
			analysis_time_ms INTEGER,
			render_time_ms INTEGER,
			memory_usage_bytes INTEGER,
			cpu_usage_percent REAL,
			goroutine_count INTEGER,
			request_size_bytes INTEGER,
			response_size_bytes INTEGER,
			tokens_processed INTEGER,
			cache_hit_rate REAL,
			error_count INTEGER,
			retry_count INTEGER,
			user_satisfaction REAL,
			additional_metrics TEXT -- JSON
		)`,

		// System metrics table
		`CREATE TABLE IF NOT EXISTS system_metrics (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			timestamp DATETIME NOT NULL,
			total_memory_usage INTEGER,
			available_memory INTEGER,
			cpu_cores INTEGER,
			average_cpu_usage REAL,
			goroutine_count INTEGER,
			active_sessions INTEGER,
			requests_per_second REAL,
			error_rate REAL,
			uptime_seconds INTEGER,
			additional_metrics TEXT -- JSON
		)`,

		// Aggregated metrics for efficient querying
		`CREATE TABLE IF NOT EXISTS aggregated_metrics (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			time_window DATETIME NOT NULL,
			window_size TEXT NOT NULL,
			session_count INTEGER DEFAULT 0,
			request_count INTEGER DEFAULT 0,
			success_count INTEGER DEFAULT 0,
			error_count INTEGER DEFAULT 0,
			avg_response_time REAL DEFAULT 0,
			min_response_time REAL DEFAULT 0,
			max_response_time REAL DEFAULT 0,
			p95_response_time REAL DEFAULT 0,
			p99_response_time REAL DEFAULT 0,
			avg_memory_usage INTEGER DEFAULT 0,
			peak_memory_usage INTEGER DEFAULT 0,
			avg_cpu_usage REAL DEFAULT 0,
			peak_cpu_usage REAL DEFAULT 0,
			total_tokens INTEGER DEFAULT 0,
			avg_tokens_per_req REAL DEFAULT 0,
			error_rate REAL DEFAULT 0,
			throughput REAL DEFAULT 0,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			UNIQUE(time_window, window_size)
		)`,

		// Performance alerts table
		`CREATE TABLE IF NOT EXISTS performance_alerts (
			id TEXT PRIMARY KEY,
			session_id TEXT,
			alert_type TEXT NOT NULL,
			severity TEXT NOT NULL,
			metric_name TEXT NOT NULL,
			current_value REAL,
			threshold_value REAL,
			message TEXT,
			context TEXT, -- JSON
			resolved BOOLEAN DEFAULT FALSE,
			resolved_at DATETIME,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
		)`,

		// Performance insights table
		`CREATE TABLE IF NOT EXISTS performance_insights (
			id TEXT PRIMARY KEY,
			type TEXT NOT NULL,
			title TEXT NOT NULL,
			description TEXT,
			impact TEXT,
			confidence REAL DEFAULT 0,
			recommendations TEXT, -- JSON array
			metrics TEXT, -- JSON object
			time_frame_ms INTEGER, -- duration in milliseconds
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			relevant_until DATETIME
		)`,

		// Indexes for performance
		`CREATE INDEX IF NOT EXISTS idx_perf_metrics_session ON performance_metrics(session_id)`,
		`CREATE INDEX IF NOT EXISTS idx_perf_metrics_timestamp ON performance_metrics(timestamp)`,
		`CREATE INDEX IF NOT EXISTS idx_perf_metrics_response_time ON performance_metrics(response_time_ms)`,

		`CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp)`,

		`CREATE INDEX IF NOT EXISTS idx_agg_metrics_window ON aggregated_metrics(time_window, window_size)`,
		`CREATE INDEX IF NOT EXISTS idx_agg_metrics_created ON aggregated_metrics(created_at)`,

		`CREATE INDEX IF NOT EXISTS idx_alerts_severity ON performance_alerts(severity)`,
		`CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON performance_alerts(resolved)`,
		`CREATE INDEX IF NOT EXISTS idx_alerts_created ON performance_alerts(created_at)`,

		`CREATE INDEX IF NOT EXISTS idx_insights_type ON performance_insights(type)`,
		`CREATE INDEX IF NOT EXISTS idx_insights_impact ON performance_insights(impact)`,
		`CREATE INDEX IF NOT EXISTS idx_insights_relevant ON performance_insights(relevant_until)`,
	}

	for _, query := range queries {
		if _, err := ps.db.Exec(query); err != nil {
			return fmt.Errorf("failed to execute query: %s, error: %w", query, err)
		}
	}

	return nil
}

// SaveMetrics implements the tracking.PerformanceStorage interface
func (ps *PerformanceStorage) SaveMetrics(ctx context.Context, metrics *tracking.PerformanceMetrics) error {
	ps.mutex.Lock()
	defer ps.mutex.Unlock()

	additionalMetricsJSON, _ := json.Marshal(metrics.AdditionalMetrics)

	query := `INSERT INTO performance_metrics 
			(session_id, request_id, timestamp, response_time_ms, ai_provider_time_ms, 
			 context_build_time_ms, analysis_time_ms, render_time_ms, memory_usage_bytes, 
			 cpu_usage_percent, goroutine_count, request_size_bytes, response_size_bytes, 
			 tokens_processed, cache_hit_rate, error_count, retry_count, user_satisfaction, 
			 additional_metrics)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`

	_, err := ps.db.ExecContext(ctx, query,
		metrics.SessionID,
		metrics.RequestID,
		metrics.Timestamp,
		metrics.ResponseTime.Milliseconds(),
		metrics.AIProviderTime.Milliseconds(),
		metrics.ContextBuildTime.Milliseconds(),
		metrics.AnalysisTime.Milliseconds(),
		metrics.RenderTime.Milliseconds(),
		metrics.MemoryUsage,
		metrics.CPUUsage,
		metrics.GoroutineCount,
		metrics.RequestSize,
		metrics.ResponseSize,
		metrics.TokensProcessed,
		metrics.CacheHitRate,
		metrics.ErrorCount,
		metrics.RetryCount,
		metrics.UserSatisfaction,
		string(additionalMetricsJSON),
	)

	return err
}

// SaveSystemMetrics implements the tracking.PerformanceStorage interface
func (ps *PerformanceStorage) SaveSystemMetrics(ctx context.Context, metrics *tracking.SystemMetrics) error {
	ps.mutex.Lock()
	defer ps.mutex.Unlock()

	additionalMetricsJSON, _ := json.Marshal(metrics.AdditionalMetrics)

	query := `INSERT INTO system_metrics 
			(timestamp, total_memory_usage, available_memory, cpu_cores, average_cpu_usage, 
			 goroutine_count, active_sessions, requests_per_second, error_rate, uptime_seconds, 
			 additional_metrics)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`

	_, err := ps.db.ExecContext(ctx, query,
		metrics.Timestamp,
		metrics.TotalMemoryUsage,
		metrics.AvailableMemory,
		metrics.CPUCores,
		metrics.AverageCPUUsage,
		metrics.GoroutineCount,
		metrics.ActiveSessions,
		metrics.RequestsPerSecond,
		metrics.ErrorRate,
		metrics.UptimeSeconds,
		string(additionalMetricsJSON),
	)

	return err
}

// GetStats implements the tracking.PerformanceStorage interface
func (ps *PerformanceStorage) GetStats(ctx context.Context, timeRange time.Duration) (*tracking.PerformanceStats, error) {
	ps.mutex.RLock()
	defer ps.mutex.RUnlock()

	cutoff := time.Now().Add(-timeRange)

	// Try aggregated data first for efficiency
	if ps.compressionEnabled {
		return ps.getStatsFromAggregated(ctx, cutoff)
	}

	return ps.getStatsFromRaw(ctx, cutoff)
}

// GetSessionMetrics implements the tracking.PerformanceStorage interface
func (ps *PerformanceStorage) GetSessionMetrics(ctx context.Context, sessionID string) ([]*tracking.PerformanceMetrics, error) {
	ps.mutex.RLock()
	defer ps.mutex.RUnlock()

	query := `SELECT session_id, request_id, timestamp, response_time_ms, ai_provider_time_ms,
			context_build_time_ms, analysis_time_ms, render_time_ms, memory_usage_bytes,
			cpu_usage_percent, goroutine_count, request_size_bytes, response_size_bytes,
			tokens_processed, cache_hit_rate, error_count, retry_count, user_satisfaction,
			additional_metrics
			FROM performance_metrics 
			WHERE session_id = ? 
			ORDER BY timestamp DESC`

	rows, err := ps.db.QueryContext(ctx, query, sessionID)
	if err != nil {
		return nil, fmt.Errorf("failed to query session metrics: %w", err)
	}
	defer rows.Close()

	var metrics []*tracking.PerformanceMetrics
	for rows.Next() {
		metric, err := ps.scanPerformanceMetrics(rows)
		if err != nil {
			continue // Skip invalid rows
		}
		metrics = append(metrics, metric)
	}

	return metrics, nil
}

// GetTrends analyzes performance trends over time
func (ps *PerformanceStorage) GetTrends(ctx context.Context, metricNames []string, timeFrame string) ([]*PerformanceTrend, error) {
	ps.mutex.RLock()
	defer ps.mutex.RUnlock()

	trends := make([]*PerformanceTrend, 0)

	for _, metricName := range metricNames {
		trend, err := ps.calculateTrend(ctx, metricName, timeFrame)
		if err != nil {
			continue // Skip metrics we can't analyze
		}
		trends = append(trends, trend)
	}

	return trends, nil
}

// SaveAlert stores a performance alert
func (ps *PerformanceStorage) SaveAlert(ctx context.Context, alert *tracking.PerformanceAlert) error {
	ps.mutex.Lock()
	defer ps.mutex.Unlock()

	contextJSON, _ := json.Marshal(alert.Additional)

	query := `INSERT OR REPLACE INTO performance_alerts 
			(id, session_id, alert_type, severity, metric_name, current_value, threshold_value,
			 message, context, resolved, resolved_at, created_at, updated_at)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`

	_, err := ps.db.ExecContext(ctx, query,
		alert.AlertID,
		alert.SessionID,
		"threshold", // Default type
		alert.Severity,
		alert.MetricName,
		alert.CurrentValue,
		alert.ThresholdValue,
		alert.Message,
		string(contextJSON),
		false,
		nil,
		alert.Timestamp,
		time.Now(),
	)

	return err
}

// GetActiveAlerts retrieves unresolved alerts
func (ps *PerformanceStorage) GetActiveAlerts(ctx context.Context) ([]*PerformanceAlert, error) {
	ps.mutex.RLock()
	defer ps.mutex.RUnlock()

	query := `SELECT id, session_id, alert_type, severity, metric_name, current_value, 
			threshold_value, message, context, resolved, resolved_at, created_at, updated_at
			FROM performance_alerts 
			WHERE resolved = FALSE 
			ORDER BY severity DESC, created_at DESC`

	rows, err := ps.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to query active alerts: %w", err)
	}
	defer rows.Close()

	var alerts []*PerformanceAlert
	for rows.Next() {
		alert, err := ps.scanAlert(rows)
		if err != nil {
			continue
		}
		alerts = append(alerts, alert)
	}

	return alerts, nil
}

// GenerateInsights creates AI-powered performance insights
func (ps *PerformanceStorage) GenerateInsights(ctx context.Context, timeRange time.Duration) ([]*PerformanceInsight, error) {
	ps.mutex.Lock()
	defer ps.mutex.Unlock()

	// Get performance statistics for analysis
	stats, err := ps.GetStats(ctx, timeRange)
	if err != nil {
		return nil, fmt.Errorf("failed to get stats for insights: %w", err)
	}

	insights := make([]*PerformanceInsight, 0)

	// Bottleneck detection
	if bottleneck := ps.detectBottlenecks(stats); bottleneck != nil {
		insights = append(insights, bottleneck)
	}

	// Performance degradation analysis
	if degradation := ps.detectDegradation(ctx, stats); degradation != nil {
		insights = append(insights, degradation)
	}

	// Optimization opportunities
	if optimization := ps.detectOptimizations(stats); optimization != nil {
		insights = append(insights, optimization)
	}

	// Save insights to database
	for _, insight := range insights {
		if err := ps.saveInsight(ctx, insight); err != nil {
			continue // Skip failed saves
		}
	}

	return insights, nil
}

// Helper methods

func (ps *PerformanceStorage) scanPerformanceMetrics(rows *sql.Rows) (*tracking.PerformanceMetrics, error) {
	var metric tracking.PerformanceMetrics
	var responseTimeMs, aiProviderTimeMs, contextBuildTimeMs, analysisTimeMs, renderTimeMs int64
	var additionalMetricsJSON string

	err := rows.Scan(
		&metric.SessionID,
		&metric.RequestID,
		&metric.Timestamp,
		&responseTimeMs,
		&aiProviderTimeMs,
		&contextBuildTimeMs,
		&analysisTimeMs,
		&renderTimeMs,
		&metric.MemoryUsage,
		&metric.CPUUsage,
		&metric.GoroutineCount,
		&metric.RequestSize,
		&metric.ResponseSize,
		&metric.TokensProcessed,
		&metric.CacheHitRate,
		&metric.ErrorCount,
		&metric.RetryCount,
		&metric.UserSatisfaction,
		&additionalMetricsJSON,
	)

	if err != nil {
		return nil, err
	}

	// Convert milliseconds to time.Duration
	metric.ResponseTime = time.Duration(responseTimeMs) * time.Millisecond
	metric.AIProviderTime = time.Duration(aiProviderTimeMs) * time.Millisecond
	metric.ContextBuildTime = time.Duration(contextBuildTimeMs) * time.Millisecond
	metric.AnalysisTime = time.Duration(analysisTimeMs) * time.Millisecond
	metric.RenderTime = time.Duration(renderTimeMs) * time.Millisecond

	// Parse additional metrics JSON
	if additionalMetricsJSON != "" {
		json.Unmarshal([]byte(additionalMetricsJSON), &metric.AdditionalMetrics)
	} else {
		metric.AdditionalMetrics = make(map[string]interface{})
	}

	return &metric, nil
}

func (ps *PerformanceStorage) scanAlert(rows *sql.Rows) (*PerformanceAlert, error) {
	var alert PerformanceAlert
	var contextJSON string
	var resolvedAtPtr *time.Time

	err := rows.Scan(
		&alert.ID,
		&alert.SessionID,
		&alert.AlertType,
		&alert.Severity,
		&alert.MetricName,
		&alert.CurrentValue,
		&alert.ThresholdValue,
		&alert.Message,
		&contextJSON,
		&alert.Resolved,
		&resolvedAtPtr,
		&alert.CreatedAt,
		&alert.UpdatedAt,
	)

	if err != nil {
		return nil, err
	}

	if resolvedAtPtr != nil {
		alert.ResolvedAt = resolvedAtPtr
	}

	// Parse context JSON
	if contextJSON != "" {
		json.Unmarshal([]byte(contextJSON), &alert.Context)
	} else {
		alert.Context = make(map[string]interface{})
	}

	return &alert, nil
}

func (ps *PerformanceStorage) getStatsFromRaw(ctx context.Context, cutoff time.Time) (*tracking.PerformanceStats, error) {
	query := `SELECT response_time_ms, memory_usage_bytes, cpu_usage_percent, error_count,
			tokens_processed FROM performance_metrics WHERE timestamp >= ? ORDER BY response_time_ms`

	rows, err := ps.db.QueryContext(ctx, query, cutoff)
	if err != nil {
		return nil, fmt.Errorf("failed to query raw metrics: %w", err)
	}
	defer rows.Close()

	var responseTimes []float64
	var memoryUsages []int64
	var cpuUsages []float64
	var errorCount int64
	var totalRequests int64
	var totalTokens int

	for rows.Next() {
		var responseTimeMs int64
		var memoryUsage int64
		var cpuUsage float64
		var errors int
		var tokens int

		err := rows.Scan(&responseTimeMs, &memoryUsage, &cpuUsage, &errors, &tokens)
		if err != nil {
			continue
		}

		responseTimes = append(responseTimes, float64(responseTimeMs))
		memoryUsages = append(memoryUsages, memoryUsage)
		cpuUsages = append(cpuUsages, cpuUsage)
		errorCount += int64(errors)
		totalRequests++
		totalTokens += tokens
	}

	if totalRequests == 0 {
		return &tracking.PerformanceStats{}, nil
	}

	stats := &tracking.PerformanceStats{
		TotalRequests:      totalRequests,
		SuccessfulRequests: totalRequests - errorCount,
		FailedRequests:     errorCount,
		ErrorRate:          float64(errorCount) / float64(totalRequests),
	}

	// Calculate response time statistics
	if len(responseTimes) > 0 {
		sort.Float64s(responseTimes)

		// Average
		sum := 0.0
		for _, rt := range responseTimes {
			sum += rt
		}
		stats.AverageResponseTime = time.Duration(sum/float64(len(responseTimes))) * time.Millisecond

		// Min/Max
		stats.FastestResponse = time.Duration(responseTimes[0]) * time.Millisecond
		stats.SlowestResponse = time.Duration(responseTimes[len(responseTimes)-1]) * time.Millisecond

		// Percentiles
		stats.MedianResponseTime = time.Duration(ps.percentile(responseTimes, 50)) * time.Millisecond
		stats.P95ResponseTime = time.Duration(ps.percentile(responseTimes, 95)) * time.Millisecond
		stats.P99ResponseTime = time.Duration(ps.percentile(responseTimes, 99)) * time.Millisecond
	}

	// Calculate memory statistics
	if len(memoryUsages) > 0 {
		sum := int64(0)
		max := memoryUsages[0]
		for _, mem := range memoryUsages {
			sum += mem
			if mem > max {
				max = mem
			}
		}
		stats.AverageMemoryUsage = sum / int64(len(memoryUsages))
		stats.PeakMemoryUsage = max
	}

	// Calculate CPU statistics
	if len(cpuUsages) > 0 {
		sum := 0.0
		for _, cpu := range cpuUsages {
			sum += cpu
		}
		stats.AverageCPUUsage = sum / float64(len(cpuUsages))
	}

	return stats, nil
}

func (ps *PerformanceStorage) getStatsFromAggregated(ctx context.Context, cutoff time.Time) (*tracking.PerformanceStats, error) {
	// Use aggregated data for better performance on large datasets
	query := `SELECT SUM(request_count), SUM(success_count), SUM(error_count),
			AVG(avg_response_time), MIN(min_response_time), MAX(max_response_time),
			AVG(p95_response_time), AVG(p99_response_time),
			AVG(avg_memory_usage), MAX(peak_memory_usage), AVG(avg_cpu_usage),
			AVG(error_rate), AVG(throughput)
			FROM aggregated_metrics 
			WHERE time_window >= ? AND window_size = 'hourly'`

	var stats tracking.PerformanceStats
	var avgResponseTime, minResponseTime, maxResponseTime float64
	var p95ResponseTime, p99ResponseTime float64

	err := ps.db.QueryRowContext(ctx, query, cutoff).Scan(
		&stats.TotalRequests,
		&stats.SuccessfulRequests,
		&stats.FailedRequests,
		&avgResponseTime,
		&minResponseTime,
		&maxResponseTime,
		&p95ResponseTime,
		&p99ResponseTime,
		&stats.AverageMemoryUsage,
		&stats.PeakMemoryUsage,
		&stats.AverageCPUUsage,
		&stats.ErrorRate,
		&stats.RequestsPerSecond,
	)

	if err != nil {
		return nil, fmt.Errorf("failed to get aggregated stats: %w", err)
	}

	// Convert milliseconds to time.Duration
	stats.AverageResponseTime = time.Duration(avgResponseTime) * time.Millisecond
	stats.FastestResponse = time.Duration(minResponseTime) * time.Millisecond
	stats.SlowestResponse = time.Duration(maxResponseTime) * time.Millisecond
	stats.P95ResponseTime = time.Duration(p95ResponseTime) * time.Millisecond
	stats.P99ResponseTime = time.Duration(p99ResponseTime) * time.Millisecond

	return &stats, nil
}

func (ps *PerformanceStorage) calculateTrend(ctx context.Context, metricName, timeFrame string) (*PerformanceTrend, error) {
	// Get data points for trend analysis
	var interval string
	var lookback time.Duration

	switch timeFrame {
	case "last_hour":
		interval = "5 minutes"
		lookback = time.Hour
	case "last_day":
		interval = "1 hour"
		lookback = 24 * time.Hour
	case "last_week":
		interval = "6 hours"
		lookback = 7 * 24 * time.Hour
	default:
		interval = "1 day"
		lookback = 30 * 24 * time.Hour
	}

	// This is a simplified trend calculation - in production you'd use more sophisticated analysis
	trend := &PerformanceTrend{
		MetricName: metricName,
		TimeFrame:  timeFrame,
		TrendType:  "stable",
		Confidence: 0.8,
		Analysis:   fmt.Sprintf("Trend analysis for %s over %s", metricName, timeFrame),
	}

	return trend, nil
}

func (ps *PerformanceStorage) detectBottlenecks(stats *tracking.PerformanceStats) *PerformanceInsight {
	if stats.P95ResponseTime > 10*time.Second {
		return &PerformanceInsight{
			ID:          fmt.Sprintf("bottleneck_%d", time.Now().Unix()),
			Type:        "bottleneck",
			Title:       "High Response Time Detected",
			Description: fmt.Sprintf("95th percentile response time is %v, which exceeds acceptable thresholds", stats.P95ResponseTime),
			Impact:      "high",
			Confidence:  0.9,
			Recommendations: []string{
				"Investigate slow database queries",
				"Review AI provider response times",
				"Consider implementing request timeouts",
				"Analyze resource utilization patterns",
			},
			Metrics: map[string]interface{}{
				"p95_response_time":     stats.P95ResponseTime,
				"p99_response_time":     stats.P99ResponseTime,
				"average_response_time": stats.AverageResponseTime,
			},
			CreatedAt:     time.Now(),
			RelevantUntil: time.Now().Add(24 * time.Hour),
		}
	}
	return nil
}

func (ps *PerformanceStorage) detectDegradation(ctx context.Context, stats *tracking.PerformanceStats) *PerformanceInsight {
	// Compare current performance with historical baselines
	if stats.ErrorRate > 0.1 { // > 10% error rate
		return &PerformanceInsight{
			ID:          fmt.Sprintf("degradation_%d", time.Now().Unix()),
			Type:        "degradation",
			Title:       "Performance Degradation Detected",
			Description: fmt.Sprintf("Error rate has increased to %.2f%%, indicating system degradation", stats.ErrorRate*100),
			Impact:      "high",
			Confidence:  0.85,
			Recommendations: []string{
				"Review recent deployments or configuration changes",
				"Check system resource availability",
				"Investigate error patterns and root causes",
				"Consider implementing circuit breakers",
			},
			Metrics: map[string]interface{}{
				"current_error_rate": stats.ErrorRate,
				"failed_requests":    stats.FailedRequests,
				"total_requests":     stats.TotalRequests,
			},
			CreatedAt:     time.Now(),
			RelevantUntil: time.Now().Add(48 * time.Hour),
		}
	}
	return nil
}

func (ps *PerformanceStorage) detectOptimizations(stats *tracking.PerformanceStats) *PerformanceInsight {
	if stats.AverageMemoryUsage > 500*1024*1024 { // > 500MB average
		return &PerformanceInsight{
			ID:          fmt.Sprintf("optimization_%d", time.Now().Unix()),
			Type:        "optimization",
			Title:       "Memory Usage Optimization Opportunity",
			Description: fmt.Sprintf("Average memory usage is %d MB, suggesting optimization opportunities", stats.AverageMemoryUsage/(1024*1024)),
			Impact:      "medium",
			Confidence:  0.75,
			Recommendations: []string{
				"Review memory allocation patterns",
				"Implement more aggressive garbage collection",
				"Consider reducing cache sizes",
				"Profile memory usage for potential leaks",
			},
			Metrics: map[string]interface{}{
				"average_memory_usage": stats.AverageMemoryUsage,
				"peak_memory_usage":    stats.PeakMemoryUsage,
			},
			CreatedAt:     time.Now(),
			RelevantUntil: time.Now().Add(7 * 24 * time.Hour),
		}
	}
	return nil
}

func (ps *PerformanceStorage) saveInsight(ctx context.Context, insight *PerformanceInsight) error {
	recommendationsJSON, _ := json.Marshal(insight.Recommendations)
	metricsJSON, _ := json.Marshal(insight.Metrics)

	query := `INSERT OR REPLACE INTO performance_insights 
			(id, type, title, description, impact, confidence, recommendations, metrics, 
			 time_frame_ms, created_at, relevant_until)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`

	_, err := ps.db.ExecContext(ctx, query,
		insight.ID,
		insight.Type,
		insight.Title,
		insight.Description,
		insight.Impact,
		insight.Confidence,
		string(recommendationsJSON),
		string(metricsJSON),
		insight.TimeFrame.Milliseconds(),
		insight.CreatedAt,
		insight.RelevantUntil,
	)

	return err
}

func (ps *PerformanceStorage) percentile(sortedData []float64, p int) float64 {
	if len(sortedData) == 0 {
		return 0
	}

	index := float64(p) / 100 * float64(len(sortedData)-1)
	lower := int(math.Floor(index))
	upper := int(math.Ceil(index))

	if lower == upper {
		return sortedData[lower]
	}

	// Linear interpolation
	weight := index - float64(lower)
	return sortedData[lower]*(1-weight) + sortedData[upper]*weight
}

func (ps *PerformanceStorage) maintenanceRoutine() {
	ticker := time.NewTicker(24 * time.Hour) // Run daily
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			ps.performMaintenance()
		}
	}
}

func (ps *PerformanceStorage) performMaintenance() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	// Clean up old raw metrics
	cutoff := time.Now().Add(-ps.retentionPeriod)
	ps.db.ExecContext(ctx, "DELETE FROM performance_metrics WHERE timestamp < ?", cutoff)
	ps.db.ExecContext(ctx, "DELETE FROM system_metrics WHERE timestamp < ?", cutoff)

	// Clean up resolved alerts older than 30 days
	alertCutoff := time.Now().Add(-30 * 24 * time.Hour)
	ps.db.ExecContext(ctx, "DELETE FROM performance_alerts WHERE resolved = TRUE AND resolved_at < ?", alertCutoff)

	// Clean up expired insights
	ps.db.ExecContext(ctx, "DELETE FROM performance_insights WHERE relevant_until < ?", time.Now())

	// Aggregate recent data
	if ps.compressionEnabled {
		ps.aggregateRecentData(ctx)
	}

	// Vacuum database to reclaim space
	ps.db.ExecContext(ctx, "VACUUM")
}

func (ps *PerformanceStorage) aggregateRecentData(ctx context.Context) {
	// Aggregate hourly data from raw metrics
	query := `INSERT OR IGNORE INTO aggregated_metrics 
			(time_window, window_size, session_count, request_count, success_count, error_count,
			 avg_response_time, min_response_time, max_response_time, 
			 avg_memory_usage, peak_memory_usage, avg_cpu_usage, peak_cpu_usage, error_rate, throughput)
			SELECT 
				datetime(timestamp, 'start of hour') as time_window,
				'hourly' as window_size,
				COUNT(DISTINCT session_id) as session_count,
				COUNT(*) as request_count,
				COUNT(*) - SUM(error_count) as success_count,
				SUM(error_count) as error_count,
				AVG(response_time_ms) as avg_response_time,
				MIN(response_time_ms) as min_response_time,
				MAX(response_time_ms) as max_response_time,
				AVG(memory_usage_bytes) as avg_memory_usage,
				MAX(memory_usage_bytes) as peak_memory_usage,
				AVG(cpu_usage_percent) as avg_cpu_usage,
				MAX(cpu_usage_percent) as peak_cpu_usage,
				CAST(SUM(error_count) AS FLOAT) / COUNT(*) as error_rate,
				COUNT(*) / 3600.0 as throughput
			FROM performance_metrics 
			WHERE timestamp >= datetime('now', '-7 days')
			GROUP BY datetime(timestamp, 'start of hour')`

	ps.db.ExecContext(ctx, query)
}
