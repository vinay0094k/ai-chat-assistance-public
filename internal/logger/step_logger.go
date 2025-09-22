package logger

import (
	"fmt"
	"time"
)

// StepLogger provides step-by-step logging functionality
type StepLogger struct {
	parent    *Logger
	step      string
	startTime time.Time
	data      map[string]interface{}
	requestID string
	subSteps  []*StepLogger
	depth     int
}

// StepResult represents the result of a step
type StepResult struct {
	Step     string                 `json:"step"`
	Success  bool                   `json:"success"`
	Duration time.Duration          `json:"duration"`
	Message  string                 `json:"message,omitempty"`
	Data     map[string]interface{} `json:"data,omitempty"`
	Error    string                 `json:"error,omitempty"`
	SubSteps []*StepResult          `json:"sub_steps,omitempty"`
}

// StepStart creates a new step logger
func (l *Logger) StepStart(step string, data map[string]interface{}) *StepLogger {
	requestID := generateRequestID()

	stepLogger := &StepLogger{
		parent:    l,
		step:      step,
		startTime: time.Now(),
		data:      data,
		requestID: requestID,
		subSteps:  make([]*StepLogger, 0),
		depth:     0,
	}

	// Log step start
	logData := map[string]interface{}{
		"request_id": requestID,
		"step":       step,
		"action":     "start",
	}

	if data != nil {
		for k, v := range data {
			logData[k] = v
		}
	}

	l.Info("Step started", logData)
	return stepLogger
}

// SubStep creates a sub-step logger
func (sl *StepLogger) SubStep(name string, data map[string]interface{}) *StepLogger {
	subStep := &StepLogger{
		parent:    sl.parent,
		step:      fmt.Sprintf("%s.%s", sl.step, name),
		startTime: time.Now(),
		data:      data,
		requestID: sl.requestID,
		subSteps:  make([]*StepLogger, 0),
		depth:     sl.depth + 1,
	}

	sl.subSteps = append(sl.subSteps, subStep)

	// Log sub-step start
	logData := map[string]interface{}{
		"request_id": sl.requestID,
		"step":       subStep.step,
		"parent":     sl.step,
		"depth":      subStep.depth,
		"action":     "start",
	}

	if data != nil {
		for k, v := range data {
			logData[k] = v
		}
	}

	sl.parent.Info("Sub-step started", logData)
	return subStep
}

// Progress logs progress for a step
func (sl *StepLogger) Progress(message string, progress float64, data map[string]interface{}) {
	logData := map[string]interface{}{
		"request_id": sl.requestID,
		"step":       sl.step,
		"progress":   progress,
		"elapsed":    time.Since(sl.startTime).String(),
		"action":     "progress",
	}

	// Add original step data
	if sl.data != nil {
		for k, v := range sl.data {
			if _, exists := logData[k]; !exists {
				logData[k] = v
			}
		}
	}

	// Add progress data
	if data != nil {
		for k, v := range data {
			logData[k] = v
		}
	}

	sl.parent.Info(message, logData)
}

// Info logs an info message for the step
func (sl *StepLogger) Info(message string, data map[string]interface{}) {
	logData := map[string]interface{}{
		"request_id": sl.requestID,
		"step":       sl.step,
		"elapsed":    time.Since(sl.startTime).String(),
		"action":     "info",
	}

	if data != nil {
		for k, v := range data {
			logData[k] = v
		}
	}

	sl.parent.Info(message, logData)
}

// Warn logs a warning message for the step
func (sl *StepLogger) Warn(message string, data map[string]interface{}) {
	logData := map[string]interface{}{
		"request_id": sl.requestID,
		"step":       sl.step,
		"elapsed":    time.Since(sl.startTime).String(),
		"action":     "warning",
	}

	if data != nil {
		for k, v := range data {
			logData[k] = v
		}
	}

	sl.parent.Warn(message, logData)
}

// Success completes a step successfully
func (sl *StepLogger) Success(message string, data map[string]interface{}) {
	duration := time.Since(sl.startTime)

	logData := map[string]interface{}{
		"request_id": sl.requestID,
		"step":       sl.step,
		"duration":   duration.String(),
		"status":     "success",
		"action":     "complete",
	}

	// Add original step data
	if sl.data != nil {
		for k, v := range sl.data {
			if _, exists := logData[k]; !exists {
				logData[k] = v
			}
		}
	}

	// Add completion data
	if data != nil {
		for k, v := range data {
			logData[k] = v
		}
	}

	sl.parent.Info(message, logData)
}

// Error completes a step with an error
func (sl *StepLogger) Error(message string, err error, data map[string]interface{}) {
	duration := time.Since(sl.startTime)

	logData := map[string]interface{}{
		"request_id": sl.requestID,
		"step":       sl.step,
		"duration":   duration.String(),
		"status":     "error",
		"action":     "complete",
	}

	// Add original step data
	if sl.data != nil {
		for k, v := range sl.data {
			if _, exists := logData[k]; !exists {
				logData[k] = v
			}
		}
	}

	// Add error data
	if data != nil {
		for k, v := range data {
			logData[k] = v
		}
	}

	sl.parent.Error(message, err, logData)
}

// GetResult returns the step result
func (sl *StepLogger) GetResult(success bool, message string, err error) *StepResult {
	duration := time.Since(sl.startTime)

	result := &StepResult{
		Step:     sl.step,
		Success:  success,
		Duration: duration,
		Message:  message,
		Data:     sl.data,
	}

	if err != nil {
		result.Error = err.Error()
	}

	// Add sub-step results
	if len(sl.subSteps) > 0 {
		result.SubSteps = make([]*StepResult, len(sl.subSteps))
		for i, subStep := range sl.subSteps {
			result.SubSteps[i] = subStep.GetResult(true, "", nil) // Sub-steps need individual completion
		}
	}

	return result
}

// GetRequestID returns the request ID for this step
func (sl *StepLogger) GetRequestID() string {
	return sl.requestID
}

// GetStep returns the step name
func (sl *StepLogger) GetStep() string {
	return sl.step
}

// GetDuration returns the elapsed time for this step
func (sl *StepLogger) GetDuration() time.Duration {
	return time.Since(sl.startTime)
}

// GetData returns the step data
func (sl *StepLogger) GetData() map[string]interface{} {
	return sl.data
}

// SetData sets additional data for the step
func (sl *StepLogger) SetData(key string, value interface{}) {
	if sl.data == nil {
		sl.data = make(map[string]interface{})
	}
	sl.data[key] = value
}

// AddSubStepResult manually adds a sub-step result (for complex workflows)
func (sl *StepLogger) AddSubStepResult(result *StepResult) {
	// This would be used when sub-steps are managed externally
	// Implementation depends on specific use case
}

// Utility function to generate request ID
func generateRequestID() string {
	return fmt.Sprintf("req_%d", time.Now().UnixNano())
}

// CreateStepLogger creates a standalone step logger
func CreateStepLogger(logger *Logger, step string, data map[string]interface{}) *StepLogger {
	return logger.StepStart(step, data)
}

// LogStepCompletion logs the completion of multiple steps
func LogStepCompletion(logger *Logger, results []*StepResult) {
	totalDuration := time.Duration(0)
	successCount := 0
	errorCount := 0

	for _, result := range results {
		totalDuration += result.Duration
		if result.Success {
			successCount++
		} else {
			errorCount++
		}
	}

	summaryData := map[string]interface{}{
		"total_steps":    len(results),
		"successful":     successCount,
		"failed":         errorCount,
		"total_duration": totalDuration.String(),
		"avg_duration":   (totalDuration / time.Duration(len(results))).String(),
	}

	if errorCount > 0 {
		logger.Warn("Step execution completed with errors", summaryData)
	} else {
		logger.Info("All steps completed successfully", summaryData)
	}
}
