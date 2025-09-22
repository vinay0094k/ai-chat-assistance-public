package tests

import (
	"fmt"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/logger"
)

func LoggingTests() {
	// Create logger with different outputs
	log := logger.NewLogger("test", "session-123")

	// Test basic logging
	fmt.Println("✅ Testing basic logging levels:")
	log.Trace("This is a trace message")
	log.Debug("This is a debug message")
	log.Info("This is an info message", map[string]interface{}{
		"user_id": "user-123",
		"action":  "test",
	})
	log.Warn("This is a warning message")
	log.Error("This is an error message", fmt.Errorf("test error"))

	// Test step logging
	fmt.Println("\n✅ Testing step logging:")
	stepLogger := log.StepStart("database_connection", map[string]interface{}{
		"host":     "localhost",
		"database": "test",
	})

	time.Sleep(100 * time.Millisecond)
	stepLogger.Progress("Establishing connection", 0.5, map[string]interface{}{
		"retry_count": 1,
	})

	time.Sleep(100 * time.Millisecond)
	stepLogger.Success("Connected successfully", map[string]interface{}{
		"connection_id": "conn-456",
	})

	// Test sub-steps
	subStep := stepLogger.SubStep("schema_validation", nil)
	time.Sleep(50 * time.Millisecond)
	subStep.Success("Schema validated", nil)

	// Test formatters
	fmt.Println("\n✅ Testing formatters:")

	// JSON formatter
	jsonFormatter := logger.NewJSONFormatter()
	jsonFormatter.PrettyPrint = true

	entry := &logger.LogEntry{
		Timestamp: time.Now(),
		Level:     logger.INFO,
		Component: "formatter_test",
		SessionID: "test-session",
		Message:   "Test JSON formatting",
		Data: map[string]interface{}{
			"test":  true,
			"count": 42,
		},
		Duration: 123 * time.Millisecond,
	}

	jsonOutput, _ := jsonFormatter.Format(entry)
	fmt.Printf("JSON Format:\n%s\n", string(jsonOutput))

	// Structured formatter
	structFormatter := logger.NewStructuredFormatter()
	structOutput, _ := structFormatter.Format(entry)
	fmt.Printf("Structured Format:\n%s", string(structOutput))

	// Compact formatter
	compactFormatter := logger.NewCompactFormatter()
	compactOutput, _ := compactFormatter.Format(entry)
	fmt.Printf("Compact Format:\n%s", string(compactOutput))

	// Test metrics collection
	fmt.Println("\n✅ Testing metrics collection:")
	metrics := logger.NewMetricsCollector()

	// Simulate some log entries
	for i := 0; i < 100; i++ {
		testEntry := &logger.LogEntry{
			Timestamp: time.Now(),
			Level:     logger.INFO,
			Component: "test_component",
			SessionID: "test-session",
			Message:   fmt.Sprintf("Test message %d", i),
			Duration:  time.Duration(i*10) * time.Millisecond,
		}

		if i%10 == 0 {
			testEntry.Level = logger.ERROR
			testEntry.Error = "Test error"
		}

		metrics.RecordLog(testEntry)
	}

	// Get metrics
	componentMetrics := metrics.GetMetrics()
	globalStats := metrics.GetGlobalStats()

	fmt.Printf("Total logs recorded: %d\n", globalStats.TotalLogs)
	fmt.Printf("Error rate: %.2f%%\n", globalStats.ErrorRate*100)

	if testMetrics, exists := componentMetrics["test_component"]; exists {
		fmt.Printf("Test component logs: %d\n", testMetrics.TotalLogs)
		fmt.Printf("Test component errors: %d\n", testMetrics.ErrorCount)
	}

	// Test health report
	healthReport := metrics.GetHealthReport()
	fmt.Printf("System health: %s\n", healthReport.OverallHealth)
	fmt.Printf("Issues found: %d\n", len(healthReport.Issues))

	// Test alerts
	fmt.Println("\n✅ Testing alert system:")
	alertManager := logger.NewAlertManager()

	// Add error rate rule
	errorRule := logger.ErrorRateRule(0.05, 1) // 5% error rate in 1 minute
	alertManager.AddRule(errorRule)

	// Add slow response rule
	slowRule := logger.SlowResponseRule(500*time.Millisecond, "test_component")
	alertManager.AddRule(slowRule)

	// Simulate triggering alerts
	errorEntry := &logger.LogEntry{
		Level:     logger.ERROR,
		Component: "test_component",
		Error:     "Simulated error",
		Timestamp: time.Now(),
	}

	slowEntry := &logger.LogEntry{
		Level:     logger.INFO,
		Component: "test_component",
		Duration:  1 * time.Second, // Slow response
		Timestamp: time.Now(),
	}

	alertManager.ProcessLog(errorEntry)
	alertManager.ProcessLog(slowEntry)

	// Check for notifications
	select {
	case notification := <-alertManager.GetNotifications():
		fmt.Printf("Alert triggered: %s\n", notification.Message)
	case <-time.After(100 * time.Millisecond):
		fmt.Println("No alerts triggered")
	}

	fmt.Println("✅ All logging tests completed!")
}
