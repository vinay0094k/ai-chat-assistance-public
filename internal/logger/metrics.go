package logger

import (
	"fmt"
	"sync"
	"time"
)

// MetricsCollector collects and analyzes logging metrics
type MetricsCollector struct {
	mutex       sync.RWMutex
	metrics     map[string]*ComponentMetrics
	globalStats *GlobalStats
	startTime   time.Time
}

// ComponentMetrics represents metrics for a specific component
type ComponentMetrics struct {
	Component       string              `json:"component"`
	TotalLogs       int64               `json:"total_logs"`
	LogsByLevel     map[LogLevel]int64  `json:"logs_by_level"`
	ErrorCount      int64               `json:"error_count"`
	WarningCount    int64               `json:"warning_count"`
	LastLogTime     time.Time           `json:"last_log_time"`
	AvgLogsPerMin   float64             `json:"avg_logs_per_minute"`
	TopErrors       map[string]int64    `json:"top_errors"`
	PerformanceData *PerformanceMetrics `json:"performance_data"`
}

// GlobalStats represents system-wide logging statistics
type GlobalStats struct {
	TotalLogs         int64              `json:"total_logs"`
	LogsByLevel       map[LogLevel]int64 `json:"logs_by_level"`
	LogsByComponent   map[string]int64   `json:"logs_by_component"`
	ErrorRate         float64            `json:"error_rate"`
	WarningRate       float64            `json:"warning_rate"`
	TopErrorMessages  []ErrorFrequency   `json:"top_error_messages"`
	PeakLoggingPeriod time.Time          `json:"peak_logging_period"`
	SystemUptime      time.Duration      `json:"system_uptime"`
}

// PerformanceMetrics represents performance-related metrics
type PerformanceMetrics struct {
	AvgResponseTime  time.Duration    `json:"avg_response_time"`
	MaxResponseTime  time.Duration    `json:"max_response_time"`
	MinResponseTime  time.Duration    `json:"min_response_time"`
	TotalOperations  int64            `json:"total_operations"`
	SlowOperations   int64            `json:"slow_operations"`
	MemoryUsage      MemoryStats      `json:"memory_usage"`
	ResponseTimeHist map[string]int64 `json:"response_time_histogram"`
}

// MemoryStats represents memory usage statistics
type MemoryStats struct {
	CurrentUsage int64 `json:"current_usage"`
	PeakUsage    int64 `json:"peak_usage"`
	AvgUsage     int64 `json:"avg_usage"`
	GCCount      int64 `json:"gc_count"`
}

// ErrorFrequency represents error message frequency
type ErrorFrequency struct {
	Message   string    `json:"message"`
	Count     int64     `json:"count"`
	LastSeen  time.Time `json:"last_seen"`
	Component string    `json:"component"`
}

// AlertRule represents a rule for generating alerts
type AlertRule struct {
	ID            string                      `json:"id"`
	Name          string                      `json:"name"`
	Description   string                      `json:"description"`
	Condition     func(*LogEntry) bool        `json:"-"`
	Action        func(*LogEntry, *AlertRule) `json:"-"`
	Enabled       bool                        `json:"enabled"`
	Cooldown      time.Duration               `json:"cooldown"`
	LastTriggered time.Time                   `json:"last_triggered"`
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{
		metrics: make(map[string]*ComponentMetrics),
		globalStats: &GlobalStats{
			LogsByLevel:      make(map[LogLevel]int64),
			LogsByComponent:  make(map[string]int64),
			TopErrorMessages: make([]ErrorFrequency, 0),
		},
		startTime: time.Now(),
	}
}

// RecordLog records a log entry for metrics collection
func (mc *MetricsCollector) RecordLog(entry *LogEntry) {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()

	// Update global stats
	mc.globalStats.TotalLogs++
	mc.globalStats.LogsByLevel[entry.Level]++
	mc.globalStats.LogsByComponent[entry.Component]++
	mc.globalStats.SystemUptime = time.Since(mc.startTime)

	// Calculate error and warning rates
	if entry.Level >= ERROR {
		mc.globalStats.ErrorRate = float64(mc.globalStats.LogsByLevel[ERROR]+mc.globalStats.LogsByLevel[FATAL]) / float64(mc.globalStats.TotalLogs)
	}
	if entry.Level == WARN {
		mc.globalStats.WarningRate = float64(mc.globalStats.LogsByLevel[WARN]) / float64(mc.globalStats.TotalLogs)
	}

	// Update component metrics
	if _, exists := mc.metrics[entry.Component]; !exists {
		mc.metrics[entry.Component] = &ComponentMetrics{
			Component:   entry.Component,
			LogsByLevel: make(map[LogLevel]int64),
			TopErrors:   make(map[string]int64),
			PerformanceData: &PerformanceMetrics{
				ResponseTimeHist: make(map[string]int64),
				MemoryUsage:      MemoryStats{},
			},
		}
	}

	component := mc.metrics[entry.Component]
	component.TotalLogs++
	component.LogsByLevel[entry.Level]++
	component.LastLogTime = entry.Timestamp

	// Count errors and warnings
	if entry.Level >= ERROR {
		component.ErrorCount++
		if entry.Error != "" {
			component.TopErrors[entry.Error]++
		}
	}
	if entry.Level == WARN {
		component.WarningCount++
	}

	// Calculate average logs per minute
	elapsed := time.Since(mc.startTime).Minutes()
	if elapsed > 0 {
		component.AvgLogsPerMin = float64(component.TotalLogs) / elapsed
	}

	// Record performance metrics if duration is present
	if entry.Duration > 0 {
		mc.recordPerformanceMetrics(component.PerformanceData, entry)
	}

	// Update top error messages globally
	if entry.Level >= ERROR && entry.Error != "" {
		mc.updateTopErrors(entry)
	}
}

// recordPerformanceMetrics records performance-related metrics
func (mc *MetricsCollector) recordPerformanceMetrics(perf *PerformanceMetrics, entry *LogEntry) {
	perf.TotalOperations++

	// Update response time statistics
	if perf.AvgResponseTime == 0 {
		perf.AvgResponseTime = entry.Duration
		perf.MinResponseTime = entry.Duration
		perf.MaxResponseTime = entry.Duration
	} else {
		// Calculate running average
		perf.AvgResponseTime = time.Duration(
			(int64(perf.AvgResponseTime)*perf.TotalOperations + int64(entry.Duration)) /
				(perf.TotalOperations + 1),
		)

		if entry.Duration < perf.MinResponseTime {
			perf.MinResponseTime = entry.Duration
		}
		if entry.Duration > perf.MaxResponseTime {
			perf.MaxResponseTime = entry.Duration
		}
	}

	// Count slow operations (>1 second)
	if entry.Duration > time.Second {
		perf.SlowOperations++
	}

	// Update response time histogram
	bucket := mc.getResponseTimeBucket(entry.Duration)
	perf.ResponseTimeHist[bucket]++

	// Update memory stats if present
	if memUsage, ok := entry.Data["memory_used"].(int64); ok {
		if perf.MemoryUsage.CurrentUsage == 0 {
			perf.MemoryUsage.CurrentUsage = memUsage
			perf.MemoryUsage.PeakUsage = memUsage
			perf.MemoryUsage.AvgUsage = memUsage
		} else {
			perf.MemoryUsage.CurrentUsage = memUsage
			if memUsage > perf.MemoryUsage.PeakUsage {
				perf.MemoryUsage.PeakUsage = memUsage
			}

			// Update running average
			perf.MemoryUsage.AvgUsage = (perf.MemoryUsage.AvgUsage + memUsage) / 2
		}
	}
}

// getResponseTimeBucket categorizes response time into buckets
func (mc *MetricsCollector) getResponseTimeBucket(duration time.Duration) string {
	ms := duration.Milliseconds()
	switch {
	case ms < 10:
		return "<10ms"
	case ms < 50:
		return "10-50ms"
	case ms < 100:
		return "50-100ms"
	case ms < 500:
		return "100-500ms"
	case ms < 1000:
		return "500ms-1s"
	case ms < 5000:
		return "1-5s"
	case ms < 10000:
		return "5-10s"
	default:
		return ">10s"
	}
}

// updateTopErrors updates the global top errors list
func (mc *MetricsCollector) updateTopErrors(entry *LogEntry) {
	found := false
	for i := range mc.globalStats.TopErrorMessages {
		if mc.globalStats.TopErrorMessages[i].Message == entry.Error {
			mc.globalStats.TopErrorMessages[i].Count++
			mc.globalStats.TopErrorMessages[i].LastSeen = entry.Timestamp
			found = true
			break
		}
	}

	if !found {
		mc.globalStats.TopErrorMessages = append(mc.globalStats.TopErrorMessages, ErrorFrequency{
			Message:   entry.Error,
			Count:     1,
			LastSeen:  entry.Timestamp,
			Component: entry.Component,
		})
	}

	// Keep only top 10 errors by count
	if len(mc.globalStats.TopErrorMessages) > 10 {
		// Sort by count (descending)
		for i := 0; i < len(mc.globalStats.TopErrorMessages)-1; i++ {
			for j := i + 1; j < len(mc.globalStats.TopErrorMessages); j++ {
				if mc.globalStats.TopErrorMessages[j].Count > mc.globalStats.TopErrorMessages[i].Count {
					mc.globalStats.TopErrorMessages[i], mc.globalStats.TopErrorMessages[j] =
						mc.globalStats.TopErrorMessages[j], mc.globalStats.TopErrorMessages[i]
				}
			}
		}
		mc.globalStats.TopErrorMessages = mc.globalStats.TopErrorMessages[:10]
	}
}

// GetMetrics returns current metrics for all components
func (mc *MetricsCollector) GetMetrics() map[string]*ComponentMetrics {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()

	// Create a copy to avoid race conditions
	result := make(map[string]*ComponentMetrics)
	for k, v := range mc.metrics {
		// Deep copy the metrics
		copied := *v
		copiedLogs := make(map[LogLevel]int64)
		for level, count := range v.LogsByLevel {
			copiedLogs[level] = count
		}
		copied.LogsByLevel = copiedLogs

		copiedErrors := make(map[string]int64)
		for msg, count := range v.TopErrors {
			copiedErrors[msg] = count
		}
		copied.TopErrors = copiedErrors

		result[k] = &copied
	}

	return result
}

// GetGlobalStats returns global logging statistics
func (mc *MetricsCollector) GetGlobalStats() *GlobalStats {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()

	// Create a copy
	stats := *mc.globalStats

	// Copy maps
	stats.LogsByLevel = make(map[LogLevel]int64)
	for level, count := range mc.globalStats.LogsByLevel {
		stats.LogsByLevel[level] = count
	}

	stats.LogsByComponent = make(map[string]int64)
	for component, count := range mc.globalStats.LogsByComponent {
		stats.LogsByComponent[component] = count
	}

	// Copy top errors
	stats.TopErrorMessages = make([]ErrorFrequency, len(mc.globalStats.TopErrorMessages))
	copy(stats.TopErrorMessages, mc.globalStats.TopErrorMessages)

	return &stats
}

// GetHealthReport generates a health report based on metrics
func (mc *MetricsCollector) GetHealthReport() *HealthReport {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()

	report := &HealthReport{
		OverallHealth:   "healthy",
		Timestamp:       time.Now(),
		Issues:          make([]HealthIssue, 0),
		Recommendations: make([]string, 0),
	}

	// Check error rates
	if mc.globalStats.ErrorRate > 0.1 { // More than 10% errors
		report.OverallHealth = "warning"
		report.Issues = append(report.Issues, HealthIssue{
			Severity:    "high",
			Component:   "global",
			Description: fmt.Sprintf("High error rate: %.2f%%", mc.globalStats.ErrorRate*100),
			Count:       int64(mc.globalStats.ErrorRate * float64(mc.globalStats.TotalLogs)),
		})
		report.Recommendations = append(report.Recommendations, "Investigate high error rate causes")
	}

	// Check for components with excessive errors
	for component, metrics := range mc.metrics {
		if metrics.TotalLogs > 0 {
			errorRate := float64(metrics.ErrorCount) / float64(metrics.TotalLogs)
			if errorRate > 0.15 { // More than 15% errors for this component
				if report.OverallHealth == "healthy" {
					report.OverallHealth = "warning"
				}
				report.Issues = append(report.Issues, HealthIssue{
					Severity:    "medium",
					Component:   component,
					Description: fmt.Sprintf("Component error rate: %.2f%%", errorRate*100),
					Count:       metrics.ErrorCount,
				})
			}
		}

		// Check for performance issues
		if metrics.PerformanceData.SlowOperations > 0 {
			slowRate := float64(metrics.PerformanceData.SlowOperations) / float64(metrics.PerformanceData.TotalOperations)
			if slowRate > 0.05 { // More than 5% slow operations
				report.Issues = append(report.Issues, HealthIssue{
					Severity:    "medium",
					Component:   component,
					Description: fmt.Sprintf("Slow operations: %.2f%%", slowRate*100),
					Count:       metrics.PerformanceData.SlowOperations,
				})
				report.Recommendations = append(report.Recommendations,
					fmt.Sprintf("Optimize performance for %s component", component))
			}
		}
	}

	// Check for memory issues
	for component, metrics := range mc.metrics {
		if metrics.PerformanceData.MemoryUsage.PeakUsage > 1024*1024*1024 { // > 1GB
			report.Issues = append(report.Issues, HealthIssue{
				Severity:  "low",
				Component: component,
				Description: fmt.Sprintf("High memory usage: %d MB",
					metrics.PerformanceData.MemoryUsage.PeakUsage/(1024*1024)),
				Count: metrics.PerformanceData.MemoryUsage.PeakUsage,
			})
		}
	}

	// Set overall health based on issues
	if len(report.Issues) == 0 {
		report.OverallHealth = "healthy"
	} else {
		hasHighSeverity := false
		for _, issue := range report.Issues {
			if issue.Severity == "high" {
				hasHighSeverity = true
				break
			}
		}
		if hasHighSeverity {
			report.OverallHealth = "critical"
		}
	}

	return report
}

// HealthReport represents a system health report
type HealthReport struct {
	OverallHealth   string        `json:"overall_health"`
	Timestamp       time.Time     `json:"timestamp"`
	Issues          []HealthIssue `json:"issues"`
	Recommendations []string      `json:"recommendations"`
}

// HealthIssue represents a specific health issue
type HealthIssue struct {
	Severity    string `json:"severity"` // low, medium, high, critical
	Component   string `json:"component"`
	Description string `json:"description"`
	Count       int64  `json:"count"`
}

// Reset clears all collected metrics
func (mc *MetricsCollector) Reset() {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()

	mc.metrics = make(map[string]*ComponentMetrics)
	mc.globalStats = &GlobalStats{
		LogsByLevel:      make(map[LogLevel]int64),
		LogsByComponent:  make(map[string]int64),
		TopErrorMessages: make([]ErrorFrequency, 0),
	}
	mc.startTime = time.Now()
}

// AlertManager manages logging alerts
type AlertManager struct {
	rules         []AlertRule
	mutex         sync.RWMutex
	notifications chan AlertNotification
}

// AlertNotification represents an alert notification
type AlertNotification struct {
	Rule      AlertRule `json:"rule"`
	Entry     LogEntry  `json:"entry"`
	Timestamp time.Time `json:"timestamp"`
	Message   string    `json:"message"`
}

// NewAlertManager creates a new alert manager
func NewAlertManager() *AlertManager {
	return &AlertManager{
		rules:         make([]AlertRule, 0),
		notifications: make(chan AlertNotification, 100),
	}
}

// AddRule adds an alert rule
func (am *AlertManager) AddRule(rule AlertRule) {
	am.mutex.Lock()
	defer am.mutex.Unlock()
	am.rules = append(am.rules, rule)
}

// ProcessLog processes a log entry against all alert rules
func (am *AlertManager) ProcessLog(entry *LogEntry) {
	am.mutex.RLock()
	defer am.mutex.RUnlock()

	for _, rule := range am.rules {
		if !rule.Enabled {
			continue
		}

		// Check cooldown
		if time.Since(rule.LastTriggered) < rule.Cooldown {
			continue
		}

		// Check condition
		if rule.Condition(entry) {
			// Update last triggered time
			rule.LastTriggered = time.Now()

			// Execute action
			if rule.Action != nil {
				rule.Action(entry, &rule)
			}

			// Send notification
			notification := AlertNotification{
				Rule:      rule,
				Entry:     *entry,
				Timestamp: time.Now(),
				Message:   fmt.Sprintf("Alert: %s triggered for %s", rule.Name, entry.Component),
			}

			select {
			case am.notifications <- notification:
			default:
				// Channel full, drop notification
			}
		}
	}
}

// GetNotifications returns the notifications channel
func (am *AlertManager) GetNotifications() <-chan AlertNotification {
	return am.notifications
}

// Common alert rule builders

// ErrorRateRule creates a rule that triggers on high error rates
func ErrorRateRule(threshold float64, windowMinutes int) AlertRule {
	errorCount := int64(0)
	totalCount := int64(0)
	windowStart := time.Now()

	return AlertRule{
		ID:          "error_rate",
		Name:        "High Error Rate",
		Description: fmt.Sprintf("Error rate exceeds %.2f%% in %d minutes", threshold*100, windowMinutes),
		Condition: func(entry *LogEntry) bool {
			now := time.Now()

			// Reset window if needed
			if now.Sub(windowStart) > time.Duration(windowMinutes)*time.Minute {
				errorCount = 0
				totalCount = 0
				windowStart = now
			}

			totalCount++
			if entry.Level >= ERROR {
				errorCount++
			}

			if totalCount < 10 { // Need minimum samples
				return false
			}

			rate := float64(errorCount) / float64(totalCount)
			return rate > threshold
		},
		Enabled:  true,
		Cooldown: time.Duration(windowMinutes) * time.Minute,
	}
}

// SlowResponseRule creates a rule that triggers on slow responses
func SlowResponseRule(threshold time.Duration, component string) AlertRule {
	return AlertRule{
		ID:          fmt.Sprintf("slow_response_%s", component),
		Name:        "Slow Response Time",
		Description: fmt.Sprintf("Response time exceeds %s for %s", threshold, component),
		Condition: func(entry *LogEntry) bool {
			return entry.Component == component && entry.Duration > threshold
		},
		Enabled:  true,
		Cooldown: 5 * time.Minute,
	}
}

// HighMemoryRule creates a rule that triggers on high memory usage
func HighMemoryRule(thresholdMB int64) AlertRule {
	return AlertRule{
		ID:          "high_memory",
		Name:        "High Memory Usage",
		Description: fmt.Sprintf("Memory usage exceeds %d MB", thresholdMB),
		Condition: func(entry *LogEntry) bool {
			if memUsage, ok := entry.Data["memory_used"].(int64); ok {
				return memUsage > thresholdMB*1024*1024
			}
			return false
		},
		Enabled:  true,
		Cooldown: 10 * time.Minute,
	}
}

// RepeatedErrorRule creates a rule that triggers on repeated errors
func RepeatedErrorRule(errorMessage string, count int, windowMinutes int) AlertRule {
	occurrences := make([]time.Time, 0)

	return AlertRule{
		ID:          fmt.Sprintf("repeated_error_%s", errorMessage),
		Name:        "Repeated Error",
		Description: fmt.Sprintf("Error '%s' occurred %d times in %d minutes", errorMessage, count, windowMinutes),
		Condition: func(entry *LogEntry) bool {
			if entry.Level < ERROR || entry.Error != errorMessage {
				return false
			}

			now := time.Now()
			window := time.Duration(windowMinutes) * time.Minute

			// Clean old occurrences
			validOccurrences := make([]time.Time, 0)
			for _, occurrence := range occurrences {
				if now.Sub(occurrence) <= window {
					validOccurrences = append(validOccurrences, occurrence)
				}
			}
			occurrences = validOccurrences

			// Add current occurrence
			occurrences = append(occurrences, now)

			return len(occurrences) >= count
		},
		Enabled:  true,
		Cooldown: time.Duration(windowMinutes) * time.Minute,
	}
}
