package llm

import (
	"sort"
	"sync"
	"time"
)

// TokenTracker tracks token usage across providers
type TokenTracker struct {
	usage     map[string]*TokenUsageStats
	mu        sync.RWMutex
	startTime time.Time
}

// TokenUsageStats represents token usage statistics for a provider
type TokenUsageStats struct {
	InputTokens  int       `json:"input_tokens"`
	OutputTokens int       `json:"output_tokens"`
	TotalTokens  int       `json:"total_tokens"`
	RequestCount int       `json:"request_count"`
	LastUsed     time.Time `json:"last_used"`
}

// NewTokenTracker creates a new token tracker
func NewTokenTracker() *TokenTracker {
	return &TokenTracker{
		usage:     make(map[string]*TokenUsageStats),
		startTime: time.Now(),
	}
}

// Track adds token usage for a provider
func (t *TokenTracker) Track(provider string, inputTokens, outputTokens int) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.usage[provider] == nil {
		t.usage[provider] = &TokenUsageStats{}
	}

	stats := t.usage[provider]
	stats.InputTokens += inputTokens
	stats.OutputTokens += outputTokens
	stats.TotalTokens += inputTokens + outputTokens
	stats.RequestCount++
	stats.LastUsed = time.Now()
}

// GetUsage returns usage stats for a provider
func (t *TokenTracker) GetUsage(provider string) TokenUsageStats {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if stats, exists := t.usage[provider]; exists {
		return *stats
	}

	return TokenUsageStats{}
}

// GetAllUsage returns usage stats for all providers
func (t *TokenTracker) GetAllUsage() map[string]TokenUsageStats {
	t.mu.RLock()
	defer t.mu.RUnlock()

	usage := make(map[string]TokenUsageStats)
	for provider, stats := range t.usage {
		usage[provider] = *stats
	}

	return usage
}

// GetTotalTokens returns total tokens across all providers
func (t *TokenTracker) GetTotalTokens() int {
	t.mu.RLock()
	defer t.mu.RUnlock()

	total := 0
	for _, stats := range t.usage {
		total += stats.TotalTokens
	}

	return total
}

// Reset resets all usage tracking
func (t *TokenTracker) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.usage = make(map[string]*TokenUsageStats)
	t.startTime = time.Now()
}

// GetTopProviders returns providers sorted by token usage
func (t *TokenTracker) GetTopProviders(limit int) []ProviderUsage {
	t.mu.RLock()
	defer t.mu.RUnlock()

	var providers []ProviderUsage
	for name, stats := range t.usage {
		providers = append(providers, ProviderUsage{
			Provider: name,
			Stats:    *stats,
		})
	}

	// Sort by total tokens descending
	sort.Slice(providers, func(i, j int) bool {
		return providers[i].Stats.TotalTokens > providers[j].Stats.TotalTokens
	})

	if limit > 0 && len(providers) > limit {
		providers = providers[:limit]
	}

	return providers
}

// ProviderUsage represents usage for a specific provider
type ProviderUsage struct {
	Provider string          `json:"provider"`
	Stats    TokenUsageStats `json:"stats"`
}

// GetAverageTokensPerRequest returns average tokens per request for a provider
func (t *TokenTracker) GetAverageTokensPerRequest(provider string) float64 {
	stats := t.GetUsage(provider)
	if stats.RequestCount == 0 {
		return 0
	}

	return float64(stats.TotalTokens) / float64(stats.RequestCount)
}

// GetUsageRate returns tokens per time unit
func (t *TokenTracker) GetUsageRate(provider string, duration time.Duration) float64 {
	stats := t.GetUsage(provider)
	if stats.TotalTokens == 0 {
		return 0
	}

	elapsed := time.Since(t.startTime)
	if elapsed == 0 {
		return 0
	}

	tokensPerSecond := float64(stats.TotalTokens) / elapsed.Seconds()
	return tokensPerSecond * duration.Seconds()
}
