package models

import (
	"time"
)

// TokenUsageInfo represents token usage for a single request
type TokenUsageInfo struct {
	Provider      string         `json:"provider"` // openai, gemini, cohere, claude
	Model         string         `json:"model"`    // specific model used
	InputTokens   int            `json:"input_tokens"`
	OutputTokens  int            `json:"output_tokens"`
	TotalTokens   int            `json:"total_tokens"`
	Cost          float64        `json:"cost"` // USD cost
	CostBreakdown *CostBreakdown `json:"cost_breakdown,omitempty"`
	Timestamp     time.Time      `json:"timestamp"`
	RequestID     string         `json:"request_id,omitempty"`
}

// CostBreakdown represents detailed cost breakdown
type CostBreakdown struct {
	InputCost  float64 `json:"input_cost"`
	OutputCost float64 `json:"output_cost"`
	InputRate  float64 `json:"input_rate"`  // Cost per 1K tokens
	OutputRate float64 `json:"output_rate"` // Cost per 1K tokens
	Currency   string  `json:"currency"`    // USD, EUR, etc.
}

// SessionTokenUsage represents token usage for an entire session
type SessionTokenUsage struct {
	SessionID       string                    `json:"session_id" db:"session_id"`
	StartTime       time.Time                 `json:"start_time" db:"start_time"`
	EndTime         *time.Time                `json:"end_time,omitempty" db:"end_time"`
	TotalQueries    int64                     `json:"total_queries" db:"total_queries"`
	TotalTokens     int64                     `json:"total_tokens" db:"total_tokens"`
	TotalCost       float64                   `json:"total_cost" db:"total_cost"`
	ProviderUsage   map[string]*ProviderUsage `json:"provider_usage" db:"-"`
	AvgResponseTime time.Duration             `json:"avg_response_time" db:"avg_response_time"`
	CreatedAt       time.Time                 `json:"created_at" db:"created_at"`
	UpdatedAt       time.Time                 `json:"updated_at" db:"updated_at"`
}

// ProviderUsage represents usage statistics for a specific provider
type ProviderUsage struct {
	Provider        string        `json:"provider" db:"provider"`
	SessionID       string        `json:"session_id" db:"session_id"`
	RequestCount    int64         `json:"request_count" db:"request_count"`
	InputTokens     int64         `json:"input_tokens" db:"input_tokens"`
	OutputTokens    int64         `json:"output_tokens" db:"output_tokens"`
	TotalTokens     int64         `json:"total_tokens" db:"total_tokens"`
	TotalCost       float64       `json:"total_cost" db:"total_cost"`
	AvgTokensPerReq float64       `json:"avg_tokens_per_request" db:"avg_tokens_per_request"`
	AvgCostPerReq   float64       `json:"avg_cost_per_request" db:"avg_cost_per_request"`
	AvgResponseTime time.Duration `json:"avg_response_time" db:"avg_response_time"`
	SuccessRate     float64       `json:"success_rate" db:"success_rate"`
	LastUsed        time.Time     `json:"last_used" db:"last_used"`
	CreatedAt       time.Time     `json:"created_at" db:"created_at"`
	UpdatedAt       time.Time     `json:"updated_at" db:"updated_at"`
}

// TokenBudget represents token budget and limits
type TokenBudget struct {
	ID               string    `json:"id" db:"id"`
	UserID           string    `json:"user_id,omitempty" db:"user_id"`
	SessionID        string    `json:"session_id,omitempty" db:"session_id"`
	DailyLimit       int64     `json:"daily_limit" db:"daily_limit"`
	MonthlyLimit     int64     `json:"monthly_limit" db:"monthly_limit"`
	DailyUsed        int64     `json:"daily_used" db:"daily_used"`
	MonthlyUsed      int64     `json:"monthly_used" db:"monthly_used"`
	CostLimit        float64   `json:"cost_limit" db:"cost_limit"`
	CostUsed         float64   `json:"cost_used" db:"cost_used"`
	WarningThreshold float64   `json:"warning_threshold" db:"warning_threshold"` // 0.0 to 1.0
	AlertThreshold   float64   `json:"alert_threshold" db:"alert_threshold"`     // 0.0 to 1.0
	ResetDate        time.Time `json:"reset_date" db:"reset_date"`
	IsActive         bool      `json:"is_active" db:"is_active"`
	CreatedAt        time.Time `json:"created_at" db:"created_at"`
	UpdatedAt        time.Time `json:"updated_at" db:"updated_at"`
}

// TokenEstimate represents estimated token usage for a request
type TokenEstimate struct {
	Provider         string  `json:"provider"`
	Model            string  `json:"model"`
	EstimatedInput   int     `json:"estimated_input"`
	EstimatedOutput  int     `json:"estimated_output"`
	EstimatedTotal   int     `json:"estimated_total"`
	EstimatedCost    float64 `json:"estimated_cost"`
	Confidence       float64 `json:"confidence"`        // 0.0 to 1.0
	Method           string  `json:"method"`            // character_count, word_count, ml_model
	ContextSize      int     `json:"context_size"`      // Size of context provided
	ComplexityFactor float64 `json:"complexity_factor"` // Complexity multiplier
}

// TokenMetrics represents aggregated token metrics
type TokenMetrics struct {
	Period             string                       `json:"period"` // hour, day, week, month
	StartTime          time.Time                    `json:"start_time"`
	EndTime            time.Time                    `json:"end_time"`
	TotalRequests      int64                        `json:"total_requests"`
	TotalTokens        int64                        `json:"total_tokens"`
	TotalCost          float64                      `json:"total_cost"`
	AvgTokensPerReq    float64                      `json:"avg_tokens_per_request"`
	AvgCostPerReq      float64                      `json:"avg_cost_per_request"`
	ProviderBreakdown  map[string]*ProviderMetrics  `json:"provider_breakdown"`
	QueryTypeBreakdown map[string]*QueryTypeMetrics `json:"query_type_breakdown"`
	PeakUsageHour      int                          `json:"peak_usage_hour"` // Hour of day (0-23)
	TrendDirection     string                       `json:"trend_direction"` // up, down, stable
}

// ProviderMetrics represents metrics for a specific provider
type ProviderMetrics struct {
	Provider        string        `json:"provider"`
	RequestCount    int64         `json:"request_count"`
	TokenCount      int64         `json:"token_count"`
	Cost            float64       `json:"cost"`
	SuccessRate     float64       `json:"success_rate"`
	AvgResponseTime time.Duration `json:"avg_response_time"`
	ShareOfTotal    float64       `json:"share_of_total"` // Percentage of total usage
}

// QueryTypeMetrics represents metrics for different query types
type QueryTypeMetrics struct {
	QueryType       string  `json:"query_type"`
	RequestCount    int64   `json:"request_count"`
	TokenCount      int64   `json:"token_count"`
	Cost            float64 `json:"cost"`
	AvgTokensPerReq float64 `json:"avg_tokens_per_request"`
	SuccessRate     float64 `json:"success_rate"`
}

// TokenOptimization represents token optimization suggestions
type TokenOptimization struct {
	CurrentUsage         *TokenUsageInfo       `json:"current_usage"`
	OptimizedUsage       *TokenUsageInfo       `json:"optimized_usage"`
	Savings              *TokenSavings         `json:"savings"`
	Recommendations      []Recommendation      `json:"recommendations"`
	AlternativeProviders []ProviderAlternative `json:"alternative_providers"`
}

// TokenSavings represents potential savings from optimization
type TokenSavings struct {
	TokensSaved   int     `json:"tokens_saved"`
	CostSaved     float64 `json:"cost_saved"`
	PercentSaved  float64 `json:"percent_saved"`
	AnnualSavings float64 `json:"annual_savings"` // Projected annual savings
}

// Recommendation represents an optimization recommendation
type Recommendation struct {
	Type        string  `json:"type"` // provider_switch, context_reduction, etc.
	Title       string  `json:"title"`
	Description string  `json:"description"`
	Impact      string  `json:"impact"`  // low, medium, high
	Effort      string  `json:"effort"`  // low, medium, high
	Savings     float64 `json:"savings"` // Estimated savings
}

// ProviderAlternative represents an alternative provider option
type ProviderAlternative struct {
	Provider       string  `json:"provider"`
	Model          string  `json:"model"`
	EstimatedCost  float64 `json:"estimated_cost"`
	QualityScore   float64 `json:"quality_score"`  // 0.0 to 1.0
	SpeedScore     float64 `json:"speed_score"`    // 0.0 to 1.0
	Compatibility  float64 `json:"compatibility"`  // 0.0 to 1.0
	Recommendation string  `json:"recommendation"` // recommended, alternative, fallback
}

// TokenAlert represents a token usage alert
type TokenAlert struct {
	ID             string                 `json:"id" db:"id"`
	Type           string                 `json:"type" db:"type"`   // warning, limit_reached, budget_exceeded
	Level          string                 `json:"level" db:"level"` // info, warning, critical
	Title          string                 `json:"title" db:"title"`
	Message        string                 `json:"message" db:"message"`
	Threshold      float64                `json:"threshold" db:"threshold"`
	CurrentUsage   float64                `json:"current_usage" db:"current_usage"`
	Period         string                 `json:"period" db:"period"` // daily, monthly, session
	SessionID      string                 `json:"session_id,omitempty" db:"session_id"`
	Acknowledged   bool                   `json:"acknowledged" db:"acknowledged"`
	AcknowledgedAt *time.Time             `json:"acknowledged_at,omitempty" db:"acknowledged_at"`
	Metadata       map[string]interface{} `json:"metadata" db:"metadata"`
	CreatedAt      time.Time              `json:"created_at" db:"created_at"`
}

// TokenTransaction represents a single token transaction
type TokenTransaction struct {
	ID              string                 `json:"id" db:"id"`
	SessionID       string                 `json:"session_id" db:"session_id"`
	QueryID         string                 `json:"query_id,omitempty" db:"query_id"`
	Provider        string                 `json:"provider" db:"provider"`
	Model           string                 `json:"model" db:"model"`
	TransactionType string                 `json:"transaction_type" db:"transaction_type"` // request, refund, adjustment
	InputTokens     int                    `json:"input_tokens" db:"input_tokens"`
	OutputTokens    int                    `json:"output_tokens" db:"output_tokens"`
	TotalTokens     int                    `json:"total_tokens" db:"total_tokens"`
	Cost            float64                `json:"cost" db:"cost"`
	Status          string                 `json:"status" db:"status"` // pending, completed, failed
	ResponseTime    time.Duration          `json:"response_time" db:"response_time"`
	Quality         float64                `json:"quality,omitempty" db:"quality"` // User-rated quality 0.0-1.0
	Metadata        map[string]interface{} `json:"metadata" db:"metadata"`
	CreatedAt       time.Time              `json:"created_at" db:"created_at"`
}
