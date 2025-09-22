// tracking/performance_tracker.go - Tracks various system performance metrics
package tracking

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// PerformanceMetrics represents a single performance measurement
type PerformanceMetrics struct {
	SessionID         string                 `json:"session_id"`
	RequestID         string                 `json:"request_id"`
	Timestamp         time.Time              `json:"timestamp"`
	ResponseTime      time.Duration          `json:"response_time"`
	AIProviderTime    time.Duration          `json:"ai_provider_time"`
	ContextBuildTime  time.Duration          `json:"context_build_time"`
	AnalysisTime      time.Duration          `json:"analysis_time"`
	RenderTime        time.Duration          `json:"render_time"`
	MemoryUsage       int64                  `json:"memory_usage"` // bytes
	CPUUsage          float64                `json:"cpu_usage"`    // percentage
	GoroutineCount    int                    `json:"goroutine_count"`
	RequestSize       int64                  `json:"request_size"`  // bytes
	ResponseSize      int64                  `json:"response_size"` // bytes
	TokensProcessed   int                    `json:"tokens_processed"`
	CacheHitRate      float64                `json:"cache_hit_rate"`
	ErrorCount        int                    `json:"error_count"`
	RetryCount        int                    `json:"retry_count"`
	UserSatisfaction  float64                `json:"user_satisfaction"` // 0-1 scale
	AdditionalMetrics map[string]interface{} `json:"additional_metrics"`
}

// SystemMetrics represents overall system health metrics
type SystemMetrics struct {
	Timestamp         time.Time              `json:"timestamp"`
	TotalMemoryUsage  int64                  `json:"total_memory_usage"`
	AvailableMemory   int64                  `json:"available_memory"`
	CPUCores          int                    `json:"cpu_cores"`
	AverageCPUUsage   float64                `json:"average_cpu_usage"`
	GoroutineCount    int                    `json:"goroutine_count"`
	ActiveSessions    int                    `json:"active_sessions"`
	RequestsPerSecond float64                `json:"requests_per_second"`
	ErrorRate         float64                `json:"error_rate"`
	UptimeSeconds     int64                  `json:"uptime_seconds"`
	AdditionalMetrics map[string]interface{} `json:"additional_metrics"`
}

// PerformanceStats represents aggregated performance statistics
type PerformanceStats struct {
	TimeRange           time.Duration            `json:"time_range"`
	TotalRequests       int64                    `json:"total_requests"`
	SuccessfulRequests  int64                    `json:"successful_requests"`
	FailedRequests      int64                    `json:"failed_requests"`
	AverageResponseTime time.Duration            `json:"average_response_time"`
	MedianResponseTime  time.Duration            `json:"median_response_time"`
	P95ResponseTime     time.Duration            `json:"p95_response_time"`
	P99ResponseTime     time.Duration            `json:"p99_response_time"`
	FastestResponse     time.Duration            `json:"fastest_response"`
	SlowestResponse     time.Duration            `json:"slowest_response"`
	RequestsPerSecond   float64                  `json:"requests_per_second"`
	ErrorRate           float64                  `json:"error_rate"`
	AverageMemoryUsage  int64                    `json:"average_memory_usage"`
	PeakMemoryUsage     int64                    `json:"peak_memory_usage"`
	AverageCPUUsage     float64                  `json:"average_cpu_usage"`
	ThroughputTrend     []TrendPoint             `json:"throughput_trend"`
	ResponseTimeTrend   []TrendPoint             `json:"response_time_trend"`
	ErrorTrend          []TrendPoint             `json:"error_trend"`
	ComponentBreakdown  map[string]time.Duration `json:"component_breakdown"`
}

// AlertThreshold defines performance alert thresholds
type AlertThreshold struct {
	MaxResponseTime      time.Duration `json:"max_response_time"`
	MaxMemoryUsage       int64         `json:"max_memory_usage"`
	MaxCPUUsage          float64       `json:"max_cpu_usage"`
	MaxErrorRate         float64       `json:"max_error_rate"`
	MinRequestsPerSecond float64       `json:"min_requests_per_second"`
}

// PerformanceAlert represents a performance alert
type PerformanceAlert struct {
	AlertID        string                 `json:"alert_id"`
	Timestamp      time.Time              `json:"timestamp"`
	Severity       string                 `json:"severity"` // "warning", "critical"
	MetricName     string                 `json:"metric_name"`
	CurrentValue   interface{}            `json:"current_value"`
	ThresholdValue interface{}            `json:"threshold_value"`
	Message        string                 `json:"message"`
	SessionID      string                 `json:"session_id,omitempty"`
	Additional     map[string]interface{} `json:"additional,omitempty"`
}

// PerformanceStorage interface for persisting performance data
type PerformanceStorage interface {
	SaveMetrics(ctx context.Context, metrics *PerformanceMetrics) error
	SaveSystemMetrics(ctx context.Context, metrics *SystemMetrics) error
	GetStats(ctx context.Context, timeRange time.Duration) (*PerformanceStats, error)
	GetSessionMetrics(ctx context.Context, sessionID string) ([]*PerformanceMetrics, error)
}

// AlertHandler interface for handling performance alerts
type AlertHandler interface {
	HandleAlert(alert *PerformanceAlert) error
}

// PerformanceTracker tracks system and request performance
type PerformanceTracker struct {
	storage      PerformanceStorage
	alertHandler AlertHandler
	thresholds   AlertThreshold
	startTime    time.Time

	// Atomic counters for thread-safe updates
	requestCount  int64
	errorCount    int64
	totalRespTime int64 // nanoseconds

	// In-memory metrics storage
	recentMetrics []*PerformanceMetrics
	systemMetrics *SystemMetrics

	// Synchronization
	mutex         sync.RWMutex
	metricsBuffer chan *PerformanceMetrics
	stopChan      chan struct{}

	// Configuration
	bufferSize      int
	flushInterval   time.Duration
	retentionPeriod time.Duration
}

// NewPerformanceTracker creates a new performance tracker
func NewPerformanceTracker(storage PerformanceStorage, alertHandler AlertHandler) *PerformanceTracker {
	pt := &PerformanceTracker{
		storage:         storage,
		alertHandler:    alertHandler,
		startTime:       time.Now(),
		recentMetrics:   make([]*PerformanceMetrics, 0, 1000),
		metricsBuffer:   make(chan *PerformanceMetrics, 100),
		stopChan:        make(chan struct{}),
		bufferSize:      100,
		flushInterval:   time.Minute,
		retentionPeriod: 24 * time.Hour,
		thresholds: AlertThreshold{
			MaxResponseTime:      30 * time.Second,
			MaxMemoryUsage:       1024 * 1024 * 1024, // 1GB
			MaxCPUUsage:          80.0,               // 80%
			MaxErrorRate:         0.05,               // 5%
			MinRequestsPerSecond: 0.1,                // 0.1 RPS
		},
	}

	// Start background processing
	go pt.processMetrics()
	go pt.collectSystemMetrics()

	return pt
}

// StartRequestTracking starts tracking a new request
func (pt *PerformanceTracker) StartRequestTracking(sessionID, requestID string) *RequestTracker {
	return &RequestTracker{
		pt:          pt,
		sessionID:   sessionID,
		requestID:   requestID,
		startTime:   time.Now(),
		stages:      make(map[string]time.Duration),
		stageStarts: make(map[string]time.Time),
	}
}

// RequestTracker tracks individual request performance
type RequestTracker struct {
	pt              *PerformanceTracker
	sessionID       string
	requestID       string
	startTime       time.Time
	stages          map[string]time.Duration
	stageStarts     map[string]time.Time
	requestSize     int64
	responseSize    int64
	tokensProcessed int
	errorCount      int
	retryCount      int
	mutex           sync.Mutex
}

// StartStage starts tracking a specific stage of request processing
func (rt *RequestTracker) StartStage(stageName string) {
	rt.mutex.Lock()
	defer rt.mutex.Unlock()
	rt.stageStarts[stageName] = time.Now()
}

// EndStage ends tracking of a specific stage
func (rt *RequestTracker) EndStage(stageName string) {
	rt.mutex.Lock()
	defer rt.mutex.Unlock()

	if startTime, exists := rt.stageStarts[stageName]; exists {
		rt.stages[stageName] = time.Since(startTime)
		delete(rt.stageStarts, stageName)
	}
}

// RecordMetric records additional metrics for the request
func (rt *RequestTracker) RecordMetric(key string, value interface{}) {
	rt.mutex.Lock()
	defer rt.mutex.Unlock()
	// Store in a metrics map if needed
}

// SetRequestSize sets the size of the incoming request
func (rt *RequestTracker) SetRequestSize(size int64) {
	rt.requestSize = size
}

// SetResponseSize sets the size of the outgoing response
func (rt *RequestTracker) SetResponseSize(size int64) {
	rt.responseSize = size
}

// SetTokensProcessed sets the number of tokens processed
func (rt *RequestTracker) SetTokensProcessed(tokens int) {
	rt.tokensProcessed = tokens
}

// RecordError increments the error count
func (rt *RequestTracker) RecordError() {
	atomic.AddInt64(&rt.pt.errorCount, 1)
	rt.errorCount++
}

// RecordRetry increments the retry count
func (rt *RequestTracker) RecordRetry() {
	rt.retryCount++
}

// Complete completes the request tracking and records metrics
func (rt *RequestTracker) Complete() *PerformanceMetrics {
	rt.mutex.Lock()
	defer rt.mutex.Unlock()

	totalTime := time.Since(rt.startTime)
	atomic.AddInt64(&rt.pt.requestCount, 1)
	atomic.AddInt64(&rt.pt.totalRespTime, totalTime.Nanoseconds())

	// Get current system metrics
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	metrics := &PerformanceMetrics{
		SessionID:         rt.sessionID,
		RequestID:         rt.requestID,
		Timestamp:         rt.startTime,
		ResponseTime:      totalTime,
		AIProviderTime:    rt.stages["ai_provider"],
		ContextBuildTime:  rt.stages["context_build"],
		AnalysisTime:      rt.stages["analysis"],
		RenderTime:        rt.stages["render"],
		MemoryUsage:       int64(memStats.Alloc),
		GoroutineCount:    runtime.NumGoroutine(),
		RequestSize:       rt.requestSize,
		ResponseSize:      rt.responseSize,
		TokensProcessed:   rt.tokensProcessed,
		ErrorCount:        rt.errorCount,
		RetryCount:        rt.retryCount,
		AdditionalMetrics: make(map[string]interface{}),
	}

	// Calculate CPU usage (simplified)
	metrics.CPUUsage = rt.pt.getCurrentCPUUsage()

	// Send to processing channel
	select {
	case rt.pt.metricsBuffer <- metrics:
	default:
		// Buffer full, skip this metric or handle overflow
	}

	// Check for alerts
	rt.pt.checkAlerts(metrics)

	return metrics
}

// processMetrics processes metrics in background
func (pt *PerformanceTracker) processMetrics() {
	ticker := time.NewTicker(pt.flushInterval)
	defer ticker.Stop()

	batch := make([]*PerformanceMetrics, 0, pt.bufferSize)

	for {
		select {
		case metric := <-pt.metricsBuffer:
			batch = append(batch, metric)

			// Store in recent metrics for quick access
			pt.mutex.Lock()
			pt.recentMetrics = append(pt.recentMetrics, metric)
			// Keep only recent metrics
			if len(pt.recentMetrics) > 1000 {
				pt.recentMetrics = pt.recentMetrics[len(pt.recentMetrics)-1000:]
			}
			pt.mutex.Unlock()

			// Flush if batch is full
			if len(batch) >= pt.bufferSize {
				pt.flushBatch(batch)
				batch = make([]*PerformanceMetrics, 0, pt.bufferSize)
			}

		case <-ticker.C:
			// Flush on timer
			if len(batch) > 0 {
				pt.flushBatch(batch)
				batch = make([]*PerformanceMetrics, 0, pt.bufferSize)
			}

		case <-pt.stopChan:
			// Final flush on shutdown
			if len(batch) > 0 {
				pt.flushBatch(batch)
			}
			return
		}
	}
}

// flushBatch flushes a batch of metrics to storage
func (pt *PerformanceTracker) flushBatch(batch []*PerformanceMetrics) {
	if pt.storage == nil {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	for _, metric := range batch {
		if err := pt.storage.SaveMetrics(ctx, metric); err != nil {
			// Log error but continue processing
			fmt.Printf("Failed to save performance metrics: %v\n", err)
		}
	}
}

// collectSystemMetrics periodically collects system-wide metrics
func (pt *PerformanceTracker) collectSystemMetrics() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			pt.collectAndStoreSystemMetrics()
		case <-pt.stopChan:
			return
		}
	}
}

// collectAndStoreSystemMetrics collects current system metrics
func (pt *PerformanceTracker) collectAndStoreSystemMetrics() {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	requestCount := atomic.LoadInt64(&pt.requestCount)
	errorCount := atomic.LoadInt64(&pt.errorCount)
	totalRespTime := atomic.LoadInt64(&pt.totalRespTime)

	var avgRespTime time.Duration
	if requestCount > 0 {
		avgRespTime = time.Duration(totalRespTime / requestCount)
	}

	var errorRate float64
	if requestCount > 0 {
		errorRate = float64(errorCount) / float64(requestCount)
	}

	uptime := time.Since(pt.startTime)
	rps := float64(requestCount) / uptime.Seconds()

	systemMetrics := &SystemMetrics{
		Timestamp:         time.Now(),
		TotalMemoryUsage:  int64(memStats.Alloc),
		AvailableMemory:   int64(memStats.Sys - memStats.Alloc),
		CPUCores:          runtime.NumCPU(),
		AverageCPUUsage:   pt.getCurrentCPUUsage(),
		GoroutineCount:    runtime.NumGoroutine(),
		ActiveSessions:    pt.getActiveSessions(),
		RequestsPerSecond: rps,
		ErrorRate:         errorRate,
		UptimeSeconds:     int64(uptime.Seconds()),
		AdditionalMetrics: make(map[string]interface{}),
	}

	pt.mutex.Lock()
	pt.systemMetrics = systemMetrics
	pt.mutex.Unlock()

	// Save to storage
	if pt.storage != nil {
		go func() {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			pt.storage.SaveSystemMetrics(ctx, systemMetrics)
		}()
	}

	// Check system-level alerts
	pt.checkSystemAlerts(systemMetrics)
}

// getCurrentCPUUsage gets current CPU usage (simplified implementation)
func (pt *PerformanceTracker) getCurrentCPUUsage() float64 {
	// This is a simplified CPU usage calculation
	// In production, you might want to use a more sophisticated method
	return float64(runtime.NumGoroutine()) / float64(runtime.NumCPU()) * 10
}

// getActiveSessions returns the number of active sessions
func (pt *PerformanceTracker) getActiveSessions() int {
	pt.mutex.RLock()
	defer pt.mutex.RUnlock()

	// Count unique sessions in recent metrics
	sessions := make(map[string]bool)
	cutoff := time.Now().Add(-5 * time.Minute)

	for _, metric := range pt.recentMetrics {
		if metric.Timestamp.After(cutoff) {
			sessions[metric.SessionID] = true
		}
	}

	return len(sessions)
}

// checkAlerts checks if metrics exceed alert thresholds
func (pt *PerformanceTracker) checkAlerts(metrics *PerformanceMetrics) {
	if pt.alertHandler == nil {
		return
	}

	// Response time alert
	if metrics.ResponseTime > pt.thresholds.MaxResponseTime {
		alert := &PerformanceAlert{
			AlertID:        fmt.Sprintf("resp_time_%s_%d", metrics.RequestID, time.Now().Unix()),
			Timestamp:      time.Now(),
			Severity:       "warning",
			MetricName:     "response_time",
			CurrentValue:   metrics.ResponseTime,
			ThresholdValue: pt.thresholds.MaxResponseTime,
			Message:        fmt.Sprintf("Response time %v exceeded threshold %v", metrics.ResponseTime, pt.thresholds.MaxResponseTime),
			SessionID:      metrics.SessionID,
		}
		go pt.alertHandler.HandleAlert(alert)
	}

	// Memory usage alert
	if metrics.MemoryUsage > pt.thresholds.MaxMemoryUsage {
		alert := &PerformanceAlert{
			AlertID:        fmt.Sprintf("memory_%s_%d", metrics.RequestID, time.Now().Unix()),
			Timestamp:      time.Now(),
			Severity:       "critical",
			MetricName:     "memory_usage",
			CurrentValue:   metrics.MemoryUsage,
			ThresholdValue: pt.thresholds.MaxMemoryUsage,
			Message:        fmt.Sprintf("Memory usage %d bytes exceeded threshold %d bytes", metrics.MemoryUsage, pt.thresholds.MaxMemoryUsage),
			SessionID:      metrics.SessionID,
		}
		go pt.alertHandler.HandleAlert(alert)
	}
}

// checkSystemAlerts checks system-level alerts
func (pt *PerformanceTracker) checkSystemAlerts(metrics *SystemMetrics) {
	if pt.alertHandler == nil {
		return
	}

	// Error rate alert
	if metrics.ErrorRate > pt.thresholds.MaxErrorRate {
		alert := &PerformanceAlert{
			AlertID:        fmt.Sprintf("error_rate_%d", time.Now().Unix()),
			Timestamp:      time.Now(),
			Severity:       "critical",
			MetricName:     "error_rate",
			CurrentValue:   metrics.ErrorRate,
			ThresholdValue: pt.thresholds.MaxErrorRate,
			Message:        fmt.Sprintf("Error rate %.2f%% exceeded threshold %.2f%%", metrics.ErrorRate*100, pt.thresholds.MaxErrorRate*100),
		}
		go pt.alertHandler.HandleAlert(alert)
	}

	// Low throughput alert
	if metrics.RequestsPerSecond < pt.thresholds.MinRequestsPerSecond {
		alert := &PerformanceAlert{
			AlertID:        fmt.Sprintf("low_throughput_%d", time.Now().Unix()),
			Timestamp:      time.Now(),
			Severity:       "warning",
			MetricName:     "requests_per_second",
			CurrentValue:   metrics.RequestsPerSecond,
			ThresholdValue: pt.thresholds.MinRequestsPerSecond,
			Message:        fmt.Sprintf("Throughput %.2f RPS below threshold %.2f RPS", metrics.RequestsPerSecond, pt.thresholds.MinRequestsPerSecond),
		}
		go pt.alertHandler.HandleAlert(alert)
	}
}

// GetStats returns aggregated performance statistics
func (pt *PerformanceTracker) GetStats(timeRange time.Duration) (*PerformanceStats, error) {
	if pt.storage != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		return pt.storage.GetStats(ctx, timeRange)
	}

	// Fallback to in-memory calculation
	return pt.calculateInMemoryStats(timeRange), nil
}

// calculateInMemoryStats calculates statistics from in-memory metrics
func (pt *PerformanceTracker) calculateInMemoryStats(timeRange time.Duration) *PerformanceStats {
	pt.mutex.RLock()
	defer pt.mutex.RUnlock()

	cutoff := time.Now().Add(-timeRange)
	var relevantMetrics []*PerformanceMetrics

	for _, metric := range pt.recentMetrics {
		if metric.Timestamp.After(cutoff) {
			relevantMetrics = append(relevantMetrics, metric)
		}
	}

	if len(relevantMetrics) == 0 {
		return &PerformanceStats{TimeRange: timeRange}
	}

	// Calculate basic statistics
	stats := &PerformanceStats{
		TimeRange:          timeRange,
		TotalRequests:      int64(len(relevantMetrics)),
		ComponentBreakdown: make(map[string]time.Duration),
	}

	var responseTimes []time.Duration
	var memoryUsages []int64
	var cpuUsages []float64
	totalErrors := int64(0)

	for _, metric := range relevantMetrics {
		responseTimes = append(responseTimes, metric.ResponseTime)
		memoryUsages = append(memoryUsages, metric.MemoryUsage)
		cpuUsages = append(cpuUsages, metric.CPUUsage)
		totalErrors += int64(metric.ErrorCount)

		// Component breakdown
		stats.ComponentBreakdown["ai_provider"] += metric.AIProviderTime
		stats.ComponentBreakdown["context_build"] += metric.ContextBuildTime
		stats.ComponentBreakdown["analysis"] += metric.AnalysisTime
		stats.ComponentBreakdown["render"] += metric.RenderTime
	}

	// Calculate averages and percentiles
	if len(responseTimes) > 0 {
		stats.AverageResponseTime = pt.calculateAverage(responseTimes)
		stats.MedianResponseTime = pt.calculatePercentile(responseTimes, 50)
		stats.P95ResponseTime = pt.calculatePercentile(responseTimes, 95)
		stats.P99ResponseTime = pt.calculatePercentile(responseTimes, 99)
		stats.FastestResponse = pt.calculateMin(responseTimes)
		stats.SlowestResponse = pt.calculateMax(responseTimes)
	}

	stats.SuccessfulRequests = stats.TotalRequests - totalErrors
	stats.FailedRequests = totalErrors
	stats.ErrorRate = float64(totalErrors) / float64(stats.TotalRequests)
	stats.RequestsPerSecond = float64(stats.TotalRequests) / timeRange.Seconds()

	if len(memoryUsages) > 0 {
		stats.AverageMemoryUsage = pt.calculateAverageInt64(memoryUsages)
		stats.PeakMemoryUsage = pt.calculateMaxInt64(memoryUsages)
	}

	if len(cpuUsages) > 0 {
		stats.AverageCPUUsage = pt.calculateAverageFloat64(cpuUsages)
	}

	return stats
}

// Helper methods for statistics calculation
func (pt *PerformanceTracker) calculateAverage(durations []time.Duration) time.Duration {
	if len(durations) == 0 {
		return 0
	}

	total := time.Duration(0)
	for _, d := range durations {
		total += d
	}
	return total / time.Duration(len(durations))
}

func (pt *PerformanceTracker) calculatePercentile(durations []time.Duration, percentile int) time.Duration {
	if len(durations) == 0 {
		return 0
	}

	// Simple percentile calculation (would need proper sorting in production)
	index := (len(durations) * percentile) / 100
	if index >= len(durations) {
		index = len(durations) - 1
	}
	return durations[index]
}

func (pt *PerformanceTracker) calculateMin(durations []time.Duration) time.Duration {
	if len(durations) == 0 {
		return 0
	}

	min := durations[0]
	for _, d := range durations[1:] {
		if d < min {
			min = d
		}
	}
	return min
}

func (pt *PerformanceTracker) calculateMax(durations []time.Duration) time.Duration {
	if len(durations) == 0 {
		return 0
	}

	max := durations[0]
	for _, d := range durations[1:] {
		if d > max {
			max = d
		}
	}
	return max
}

func (pt *PerformanceTracker) calculateAverageInt64(values []int64) int64 {
	if len(values) == 0 {
		return 0
	}

	total := int64(0)
	for _, v := range values {
		total += v
	}
	return total / int64(len(values))
}

func (pt *PerformanceTracker) calculateMaxInt64(values []int64) int64 {
	if len(values) == 0 {
		return 0
	}

	max := values[0]
	for _, v := range values[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

func (pt *PerformanceTracker) calculateAverageFloat64(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	total := 0.0
	for _, v := range values {
		total += v
	}
	return total / float64(len(values))
}

// GetCurrentSystemMetrics returns the latest system metrics
func (pt *PerformanceTracker) GetCurrentSystemMetrics() *SystemMetrics {
	pt.mutex.RLock()
	defer pt.mutex.RUnlock()
	return pt.systemMetrics
}

// SetAlertThresholds updates the alert thresholds
func (pt *PerformanceTracker) SetAlertThresholds(thresholds AlertThreshold) {
	pt.mutex.Lock()
	defer pt.mutex.Unlock()
	pt.thresholds = thresholds
}

// Stop stops the performance tracker
func (pt *PerformanceTracker) Stop() {
	close(pt.stopChan)
}
