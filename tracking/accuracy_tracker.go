// tracking/accuracy_tracker.go - Measures accuracy of AI generated responses
package tracking

import (
	"context"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/learning"
)

// AccuracyMetrics represents accuracy measurement data
type AccuracyMetrics struct {
	ResponseID          string                 `json:"response_id"`
	SessionID           string                 `json:"session_id"`
	Query               string                 `json:"query"`
	AIResponse          string                 `json:"ai_response"`
	UserFeedbackScore   float64                `json:"user_feedback_score"` // 0-1 scale
	SemanticAccuracy    float64                `json:"semantic_accuracy"`   // 0-1 scale
	ContextRelevance    float64                `json:"context_relevance"`   // 0-1 scale
	CodeCompleteness    float64                `json:"code_completeness"`   // 0-1 scale
	OverallAccuracy     float64                `json:"overall_accuracy"`    // Weighted average
	Timestamp           time.Time              `json:"timestamp"`
	Tags                []string               `json:"tags"`
	AdditionalMetrics   map[string]interface{} `json:"additional_metrics"`
	ValidationAttempts  int                    `json:"validation_attempts"`
	SuccessfulExecution bool                   `json:"successful_execution"`
}

// AccuracyTracker tracks and measures AI response accuracy
type AccuracyTracker struct {
	metrics     map[string]*AccuracyMetrics
	storage     AccuracyStorage
	weights     AccuracyWeights
	mutex       sync.RWMutex
	feedbackSys *learning.FeedbackCollector
}

// AccuracyWeights defines how different accuracy factors are weighted
type AccuracyWeights struct {
	UserFeedback     float64 `json:"user_feedback"`
	SemanticMatch    float64 `json:"semantic_match"`
	ContextRelevance float64 `json:"context_relevance"`
	CodeQuality      float64 `json:"code_quality"`
}

// AccuracyStorage interface for persisting accuracy data
type AccuracyStorage interface {
	SaveMetrics(ctx context.Context, metrics *AccuracyMetrics) error
	LoadMetrics(ctx context.Context, responseID string) (*AccuracyMetrics, error)
	GetSessionMetrics(ctx context.Context, sessionID string) ([]*AccuracyMetrics, error)
	GetAggregatedAccuracy(ctx context.Context, timeRange time.Duration) (*AccuracyStats, error)
}

// AccuracyStats represents aggregated accuracy statistics
type AccuracyStats struct {
	AverageAccuracy     float64            `json:"average_accuracy"`
	MedianAccuracy      float64            `json:"median_accuracy"`
	AccuracyTrend       []TrendPoint       `json:"accuracy_trend"`
	CategoryBreakdown   map[string]float64 `json:"category_breakdown"`
	ImprovementRate     float64            `json:"improvement_rate"`
	TotalResponses      int                `json:"total_responses"`
	HighAccuracyCount   int                `json:"high_accuracy_count"`   // >0.8
	MediumAccuracyCount int                `json:"medium_accuracy_count"` // 0.5-0.8
	LowAccuracyCount    int                `json:"low_accuracy_count"`    // <0.5
}

// TrendPoint represents a point in the accuracy trend
type TrendPoint struct {
	Timestamp time.Time `json:"timestamp"`
	Accuracy  float64   `json:"accuracy"`
	Volume    int       `json:"volume"`
}

// NewAccuracyTracker creates a new accuracy tracker
func NewAccuracyTracker(storage AccuracyStorage, feedbackSys *learning.FeedbackCollector) *AccuracyTracker {
	return &AccuracyTracker{
		metrics:     make(map[string]*AccuracyMetrics),
		storage:     storage,
		feedbackSys: feedbackSys,
		weights: AccuracyWeights{
			UserFeedback:     0.4, // Highest weight - direct user input
			SemanticMatch:    0.25,
			ContextRelevance: 0.2,
			CodeQuality:      0.15,
		},
	}
}

// TrackResponse starts tracking accuracy for a new AI response
func (at *AccuracyTracker) TrackResponse(responseID, sessionID, query, aiResponse string, tags []string) *AccuracyMetrics {
	at.mutex.Lock()
	defer at.mutex.Unlock()

	metrics := &AccuracyMetrics{
		ResponseID:         responseID,
		SessionID:          sessionID,
		Query:              query,
		AIResponse:         aiResponse,
		Timestamp:          time.Now(),
		Tags:               tags,
		AdditionalMetrics:  make(map[string]interface{}),
		ValidationAttempts: 0,
	}

	// Initial automatic analysis
	go at.performAutomaticAnalysis(metrics)

	at.metrics[responseID] = metrics
	return metrics
}

// RecordUserFeedback records direct user feedback on response accuracy
func (at *AccuracyTracker) RecordUserFeedback(responseID string, feedbackScore float64, comments string) error {
	at.mutex.Lock()
	defer at.mutex.Unlock()

	metrics, exists := at.metrics[responseID]
	if !exists {
		return fmt.Errorf("response ID %s not found", responseID)
	}

	metrics.UserFeedbackScore = feedbackScore
	metrics.AdditionalMetrics["user_comments"] = comments

	// Update overall accuracy
	at.calculateOverallAccuracy(metrics)

	// Save to storage
	if at.storage != nil {
		go func() {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			at.storage.SaveMetrics(ctx, metrics)
		}()
	}

	// Send to feedback system for learning
	if at.feedbackSys != nil {
		feedback := &learning.FeedbackEntry{
			SessionID:    metrics.SessionID,
			ResponseID:   responseID,
			UserRating:   int(feedbackScore * 5), // Convert to 1-5 scale
			Comments:     comments,
			Timestamp:    time.Now(),
			FeedbackType: learning.FeedbackTypeRating,
		}
		at.feedbackSys.CollectFeedback(feedback)
	}

	return nil
}

// RecordCodeExecution records whether generated code executed successfully
func (at *AccuracyTracker) RecordCodeExecution(responseID string, successful bool, errorMsg string) error {
	at.mutex.Lock()
	defer at.mutex.Unlock()

	metrics, exists := at.metrics[responseID]
	if !exists {
		return fmt.Errorf("response ID %s not found", responseID)
	}

	metrics.SuccessfulExecution = successful
	metrics.ValidationAttempts++

	if !successful {
		metrics.AdditionalMetrics["execution_error"] = errorMsg
		// Penalize code completeness for failed execution
		metrics.CodeCompleteness *= 0.5
	} else {
		// Reward successful execution
		if metrics.CodeCompleteness == 0 {
			metrics.CodeCompleteness = 0.8
		} else {
			metrics.CodeCompleteness = math.Min(1.0, metrics.CodeCompleteness*1.2)
		}
	}

	at.calculateOverallAccuracy(metrics)
	return nil
}

// performAutomaticAnalysis performs automatic accuracy analysis
func (at *AccuracyTracker) performAutomaticAnalysis(metrics *AccuracyMetrics) {
	// Semantic accuracy analysis
	metrics.SemanticAccuracy = at.calculateSemanticAccuracy(metrics.Query, metrics.AIResponse)

	// Context relevance analysis
	metrics.ContextRelevance = at.calculateContextRelevance(metrics.Query, metrics.AIResponse, metrics.Tags)

	// Code completeness analysis (if response contains code)
	if at.containsCode(metrics.AIResponse) {
		metrics.CodeCompleteness = at.calculateCodeCompleteness(metrics.AIResponse)
	} else {
		metrics.CodeCompleteness = 1.0 // Not applicable
	}

	// Calculate initial overall accuracy (will be updated when user feedback arrives)
	at.calculateOverallAccuracy(metrics)
}

// calculateSemanticAccuracy analyzes semantic match between query and response
func (at *AccuracyTracker) calculateSemanticAccuracy(query, response string) float64 {
	// Extract key terms from query
	queryTerms := at.extractKeyTerms(query)
	responseTerms := at.extractKeyTerms(response)

	if len(queryTerms) == 0 {
		return 0.5 // Default for unclear queries
	}

	// Calculate term overlap
	matches := 0
	for _, term := range queryTerms {
		for _, respTerm := range responseTerms {
			if strings.Contains(strings.ToLower(respTerm), strings.ToLower(term)) ||
				strings.Contains(strings.ToLower(term), strings.ToLower(respTerm)) {
				matches++
				break
			}
		}
	}

	// Base semantic accuracy on term coverage
	termCoverage := float64(matches) / float64(len(queryTerms))

	// Bonus for comprehensive responses
	lengthBonus := math.Min(0.2, float64(len(response))/5000.0)

	return math.Min(1.0, termCoverage*0.8+lengthBonus)
}

// calculateContextRelevance analyzes how well response fits the context
func (at *AccuracyTracker) calculateContextRelevance(query, response string, tags []string) float64 {
	score := 0.5 // Base score

	// Check if response addresses the specific context indicated by tags
	for _, tag := range tags {
		if strings.Contains(strings.ToLower(response), strings.ToLower(tag)) {
			score += 0.1
		}
	}

	// Check for context-appropriate language and concepts
	if strings.Contains(query, "explain") && len(response) > 100 {
		score += 0.1 // Good explanations are typically longer
	}

	if strings.Contains(query, "example") && (strings.Contains(response, "```") || strings.Contains(response, "for example")) {
		score += 0.15 // Contains examples as requested
	}

	if strings.Contains(query, "how to") && (strings.Contains(response, "step") || strings.Contains(response, "first") || strings.Contains(response, "1.")) {
		score += 0.15 // Provides step-by-step guidance
	}

	return math.Min(1.0, score)
}

// calculateCodeCompleteness analyzes code quality and completeness
func (at *AccuracyTracker) calculateCodeCompleteness(response string) float64 {
	score := 0.0

	// Check for code blocks
	codeBlocks := strings.Count(response, "```")
	if codeBlocks >= 2 {
		score += 0.3 // Has properly formatted code
	}

	// Check for imports/includes
	if strings.Contains(response, "import ") || strings.Contains(response, "#include") || strings.Contains(response, "require(") {
		score += 0.1
	}

	// Check for function definitions
	if strings.Contains(response, "func ") || strings.Contains(response, "function ") || strings.Contains(response, "def ") {
		score += 0.2
	}

	// Check for error handling
	if strings.Contains(response, "error") || strings.Contains(response, "try") || strings.Contains(response, "catch") {
		score += 0.1
	}

	// Check for comments/documentation
	if strings.Contains(response, "//") || strings.Contains(response, "#") || strings.Contains(response, "/*") {
		score += 0.1
	}

	// Check for complete examples (basic heuristic)
	lines := strings.Split(response, "\n")
	codeLines := 0
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if len(line) > 0 && !strings.HasPrefix(line, "//") && !strings.HasPrefix(line, "#") {
			codeLines++
		}
	}

	if codeLines > 5 {
		score += 0.2 // Substantial code
	}

	return math.Min(1.0, score)
}

// calculateOverallAccuracy computes weighted overall accuracy
func (at *AccuracyTracker) calculateOverallAccuracy(metrics *AccuracyMetrics) {
	totalWeight := 0.0
	weightedSum := 0.0

	// Only include metrics that have been calculated
	if metrics.UserFeedbackScore > 0 {
		weightedSum += metrics.UserFeedbackScore * at.weights.UserFeedback
		totalWeight += at.weights.UserFeedback
	}

	if metrics.SemanticAccuracy > 0 {
		weightedSum += metrics.SemanticAccuracy * at.weights.SemanticMatch
		totalWeight += at.weights.SemanticMatch
	}

	if metrics.ContextRelevance > 0 {
		weightedSum += metrics.ContextRelevance * at.weights.ContextRelevance
		totalWeight += at.weights.ContextRelevance
	}

	if metrics.CodeCompleteness > 0 {
		weightedSum += metrics.CodeCompleteness * at.weights.CodeQuality
		totalWeight += at.weights.CodeQuality
	}

	if totalWeight > 0 {
		metrics.OverallAccuracy = weightedSum / totalWeight
	}
}

// GetAccuracyStats returns aggregated accuracy statistics
func (at *AccuracyTracker) GetAccuracyStats(timeRange time.Duration) (*AccuracyStats, error) {
	if at.storage != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		return at.storage.GetAggregatedAccuracy(ctx, timeRange)
	}

	// Fallback to in-memory calculation
	return at.calculateInMemoryStats(timeRange), nil
}

// Helper methods
func (at *AccuracyTracker) extractKeyTerms(text string) []string {
	words := strings.Fields(strings.ToLower(text))
	var terms []string

	stopWords := map[string]bool{
		"the": true, "a": true, "an": true, "and": true, "or": true, "but": true,
		"in": true, "on": true, "at": true, "to": true, "for": true, "of": true,
		"with": true, "by": true, "how": true, "what": true, "why": true, "when": true,
		"where": true, "is": true, "are": true, "was": true, "were": true, "be": true,
	}

	for _, word := range words {
		if len(word) > 2 && !stopWords[word] {
			terms = append(terms, word)
		}
	}

	return terms
}

func (at *AccuracyTracker) containsCode(text string) bool {
	return strings.Contains(text, "```") ||
		strings.Contains(text, "func ") ||
		strings.Contains(text, "function ") ||
		strings.Contains(text, "def ") ||
		strings.Contains(text, "class ") ||
		strings.Contains(text, "import ")
}

func (at *AccuracyTracker) calculateInMemoryStats(timeRange time.Duration) *AccuracyStats {
	at.mutex.RLock()
	defer at.mutex.RUnlock()

	cutoff := time.Now().Add(-timeRange)
	var accuracies []float64
	stats := &AccuracyStats{
		CategoryBreakdown: make(map[string]float64),
	}

	for _, metrics := range at.metrics {
		if metrics.Timestamp.After(cutoff) && metrics.OverallAccuracy > 0 {
			accuracies = append(accuracies, metrics.OverallAccuracy)
			stats.TotalResponses++

			if metrics.OverallAccuracy >= 0.8 {
				stats.HighAccuracyCount++
			} else if metrics.OverallAccuracy >= 0.5 {
				stats.MediumAccuracyCount++
			} else {
				stats.LowAccuracyCount++
			}
		}
	}

	if len(accuracies) > 0 {
		sum := 0.0
		for _, acc := range accuracies {
			sum += acc
		}
		stats.AverageAccuracy = sum / float64(len(accuracies))
	}

	return stats
}

// GetResponseMetrics returns metrics for a specific response
func (at *AccuracyTracker) GetResponseMetrics(responseID string) (*AccuracyMetrics, error) {
	at.mutex.RLock()
	defer at.mutex.RUnlock()

	metrics, exists := at.metrics[responseID]
	if !exists {
		return nil, fmt.Errorf("response ID %s not found", responseID)
	}

	return metrics, nil
}
