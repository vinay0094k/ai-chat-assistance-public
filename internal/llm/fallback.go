package llm

import (
	"context"
	"fmt"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/llm/providers"
)

// FallbackManager handles provider fallback logic
type FallbackManager struct {
	maxRetries    int
	retryDelay    time.Duration
	backoffFactor float64
}

// NewFallbackManager creates a new fallback manager
func NewFallbackManager(maxRetries int) *FallbackManager {
	return &FallbackManager{
		maxRetries:    maxRetries,
		retryDelay:    time.Second * 2,
		backoffFactor: 1.5,
	}
}

// HandleFailure handles provider failure and attempts fallback
func (f *FallbackManager) HandleFailure(
	ctx context.Context,
	req *providers.TextGenerationRequest,
	failedProvider string,
	err error,
	availableProviders map[string]providers.Provider,
) (*providers.TextGenerationResponse, error) {

	// Try other providers as fallback
	for name, provider := range availableProviders {
		if name == failedProvider || !provider.IsEnabled() {
			continue
		}

		// Add fallback metadata
		if req.Metadata == nil {
			req.Metadata = make(map[string]interface{})
		}
		req.Metadata["fallback_from"] = failedProvider
		req.Metadata["fallback_reason"] = err.Error()

		resp, retryErr := provider.GenerateText(ctx, req)
		if retryErr == nil {
			return resp, nil
		}
	}

	return nil, fmt.Errorf("all providers failed, last error: %v", err)
}

// ShouldRetry determines if an error should trigger a retry
func (f *FallbackManager) ShouldRetry(err error) bool {
	if err == nil {
		return false
	}

	// Check for retryable errors
	provErr, ok := err.(*providers.ProviderError)
	if !ok {
		return true // Unknown error, might be retryable
	}

	switch provErr.Code {
	case "RATE_LIMIT", "TIMEOUT", "HTTP_ERROR", "NETWORK_ERROR":
		return true
	case "INVALID_API_KEY", "UNAUTHORIZED", "INSUFFICIENT_QUOTA":
		return false
	default:
		return true
	}
}

// CalculateRetryDelay calculates delay for retry attempt
func (f *FallbackManager) CalculateRetryDelay(attempt int) time.Duration {
	if attempt <= 0 {
		return f.retryDelay
	}

	delay := f.retryDelay
	for i := 0; i < attempt; i++ {
		delay = time.Duration(float64(delay) * f.backoffFactor)
	}

	// Cap at 30 seconds
	if delay > time.Second*30 {
		delay = time.Second * 30
	}

	return delay
}
