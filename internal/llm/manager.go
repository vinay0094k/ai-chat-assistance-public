package llm

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/llm/providers"
)

type Provider = providers.Provider

// Manager manages multiple AI providers and handles routing, fallback, and load balancing
type Manager struct {
	providers    map[string]providers.Provider
	fallback     *FallbackManager
	costTracker  *CostCalculator
	tokenTracker *TokenTracker
	config       *ManagerConfig
	mu           sync.RWMutex
}

// ManagerConfig contains configuration for the AI manager
type ManagerConfig struct {
	DefaultProvider   string                               `json:"default_provider"`
	EnableFallback    bool                                 `json:"enable_fallback"`
	EnableLoadBalance bool                                 `json:"enable_load_balance"`
	MaxRetries        int                                  `json:"max_retries"`
	Timeout           time.Duration                        `json:"timeout"`
	CostLimits        map[string]float64                   `json:"cost_limits"`
	ProviderConfigs   map[string]*providers.ProviderConfig `json:"provider_configs"`
}

// NewManager creates a new AI manager
func NewManager(config *ManagerConfig) *Manager {
	manager := &Manager{
		providers:    make(map[string]providers.Provider),
		config:       config,
		costTracker:  NewCostCalculator(),
		tokenTracker: NewTokenTracker(),
	}

	if config.EnableFallback {
		manager.fallback = NewFallbackManager(config.MaxRetries)
	}

	return manager
}

// RegisterProvider registers a new AI provider
func (m *Manager) RegisterProvider(name string, provider providers.Provider) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !provider.IsEnabled() {
		return fmt.Errorf("provider %s is not enabled", name)
	}

	// Health check
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := provider.HealthCheck(ctx); err != nil {
		return fmt.Errorf("provider %s failed health check: %v", name, err)
	}

	m.providers[name] = provider
	return nil
}

// GetProvider returns a specific provider
func (m *Manager) GetProvider(name string) (providers.Provider, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	provider, exists := m.providers[name]
	if !exists {
		return nil, fmt.Errorf("provider %s not found", name)
	}

	return provider, nil
}

// SelectProvider selects the best provider for a request
func (m *Manager) SelectProvider(req *providers.TextGenerationRequest) (providers.Provider, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if len(m.providers) == 0 {
		return nil, fmt.Errorf("no providers available")
	}

	// If specific provider requested in metadata
	if providerName, ok := req.Metadata["provider"].(string); ok {
		if provider, exists := m.providers[providerName]; exists && provider.IsEnabled() {
			return provider, nil
		}
	}

	if m.config.EnableLoadBalance {
		return m.selectByLoadBalance(req)
	}

	// Use default provider
	if provider, exists := m.providers[m.config.DefaultProvider]; exists && provider.IsEnabled() {
		return provider, nil
	}

	// Fallback to first available provider
	for _, provider := range m.providers {
		if provider.IsEnabled() {
			return provider, nil
		}
	}

	return nil, fmt.Errorf("no enabled providers available")
}

// GenerateText generates text using the best available provider
func (m *Manager) GenerateText(ctx context.Context, req *providers.TextGenerationRequest) (*providers.TextGenerationResponse, error) {
	provider, err := m.SelectProvider(req)
	if err != nil {
		return nil, err
	}

	// Add timeout to context
	if m.config.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, m.config.Timeout)
		defer cancel()
	}

	start := time.Now()
	resp, err := provider.GenerateText(ctx, req)

	if err != nil && m.config.EnableFallback {
		return m.fallback.HandleFailure(ctx, req, provider.GetName(), err, m.providers)
	}

	if resp != nil {
		// Track usage
		m.tokenTracker.Track(provider.GetName(), resp.TokensUsed.InputTokens, resp.TokensUsed.OutputTokens)
		m.costTracker.AddUsage(provider.GetName(), resp.TokensUsed.Cost)

		// Check cost limits
		if err := m.checkCostLimits(provider.GetName()); err != nil {
			return resp, fmt.Errorf("cost limit exceeded: %v", err)
		}

		resp.Duration = time.Since(start)
	}

	return resp, err
}

// GenerateCode generates code using the best available provider
func (m *Manager) GenerateCode(ctx context.Context, req *providers.CodeGenerationRequest) (*providers.CodeGenerationResponse, error) {
	// Convert to text request for provider selection
	textReq := &providers.TextGenerationRequest{
		Prompt:      req.Description,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		Metadata:    req.Metadata,
	}

	provider, err := m.SelectProvider(textReq)
	if err != nil {
		return nil, err
	}

	if m.config.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, m.config.Timeout)
		defer cancel()
	}

	resp, err := provider.GenerateCode(ctx, req)

	if resp != nil {
		m.tokenTracker.Track(provider.GetName(), resp.TokensUsed.InputTokens, resp.TokensUsed.OutputTokens)
		m.costTracker.AddUsage(provider.GetName(), resp.TokensUsed.Cost)
	}

	return resp, err
}

// ExplainCode explains code using the best available provider
func (m *Manager) ExplainCode(ctx context.Context, req *providers.CodeExplanationRequest) (*providers.CodeExplanationResponse, error) {
	textReq := &providers.TextGenerationRequest{
		Prompt:    fmt.Sprintf("Explain this %s code: %s", req.Language, req.Code),
		MaxTokens: req.MaxTokens,
		Metadata:  req.Metadata,
	}

	provider, err := m.SelectProvider(textReq)
	if err != nil {
		return nil, err
	}

	if m.config.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, m.config.Timeout)
		defer cancel()
	}

	resp, err := provider.ExplainCode(ctx, req)

	if resp != nil {
		m.tokenTracker.Track(provider.GetName(), resp.TokensUsed.InputTokens, resp.TokensUsed.OutputTokens)
		m.costTracker.AddUsage(provider.GetName(), resp.TokensUsed.Cost)
	}

	return resp, err
}

// StreamText streams text generation from the best available provider
func (m *Manager) StreamText(ctx context.Context, req *providers.TextGenerationRequest) (<-chan providers.StreamChunk, error) {
	provider, err := m.SelectProvider(req)
	if err != nil {
		return nil, err
	}

	if m.config.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, m.config.Timeout)
		defer cancel()
	}

	return provider.StreamText(ctx, req)
}

// GetProviderStatus returns status of all providers
func (m *Manager) GetProviderStatus() map[string]ProviderStatus {
	m.mu.RLock()
	defer m.mu.RUnlock()

	status := make(map[string]ProviderStatus)

	for name, provider := range m.providers {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)

		healthErr := provider.HealthCheck(ctx)
		cancel()

		status[name] = ProviderStatus{
			Name:         name,
			Model:        provider.GetModel(),
			Enabled:      provider.IsEnabled(),
			Healthy:      healthErr == nil,
			Capabilities: provider.GetCapabilities(),
			Usage:        m.tokenTracker.GetUsage(name),
			Cost:         m.costTracker.GetProviderCost(name),
		}
	}

	return status
}

// GetTotalCost returns total cost across all providers
func (m *Manager) GetTotalCost() float64 {
	return m.costTracker.GetTotalCost()
}

// GetUsageStats returns usage statistics
func (m *Manager) GetUsageStats() UsageStats {
	return UsageStats{
		TotalTokens: m.tokenTracker.GetTotalTokens(),
		TotalCost:   m.costTracker.GetTotalCost(),
		ByProvider:  m.tokenTracker.GetAllUsage(),
	}
}

// selectByLoadBalance selects provider based on load balancing
func (m *Manager) selectByLoadBalance(req *providers.TextGenerationRequest) (providers.Provider, error) {
	type providerScore struct {
		name     string
		provider providers.Provider
		score    float64
	}

	var candidates []providerScore

	for name, provider := range m.providers {
		if !provider.IsEnabled() {
			continue
		}

		// Calculate score based on multiple factors
		score := m.calculateProviderScore(name, provider, req)
		candidates = append(candidates, providerScore{
			name:     name,
			provider: provider,
			score:    score,
		})
	}

	if len(candidates) == 0 {
		return nil, fmt.Errorf("no available providers")
	}

	// Sort by score (higher is better)
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	return candidates[0].provider, nil
}

// calculateProviderScore calculates a score for provider selection
func (m *Manager) calculateProviderScore(name string, provider providers.Provider, req *providers.TextGenerationRequest) float64 {
	score := 1.0

	// Factor in cost efficiency
	config := m.config.ProviderConfigs[name]
	if config != nil {
		costFactor := 1.0 / (config.CostPer1KInput + config.CostPer1KOutput + 0.001)
		score *= costFactor

		// Factor in provider weight
		score *= config.Weight
	}

	// Factor in current load (lower usage = higher score)
	usage := m.tokenTracker.GetUsage(name)
	if usage.TotalTokens > 0 {
		score *= 1.0 / (float64(usage.TotalTokens)/10000.0 + 1.0)
	}

	// Factor in provider capabilities
	capabilities := provider.GetCapabilities()
	if req.MaxTokens > 0 && capabilities.MaxTokens > 0 {
		if req.MaxTokens <= capabilities.MaxTokens {
			score *= 1.2 // Bonus for meeting requirements
		} else {
			score *= 0.5 // Penalty for not meeting requirements
		}
	}

	return score
}

// checkCostLimits checks if cost limits are exceeded
func (m *Manager) checkCostLimits(providerName string) error {
	if limit, exists := m.config.CostLimits[providerName]; exists {
		if m.costTracker.GetProviderCost(providerName) > limit {
			return fmt.Errorf("cost limit exceeded for provider %s", providerName)
		}
	}

	if globalLimit, exists := m.config.CostLimits["global"]; exists {
		if m.costTracker.GetTotalCost() > globalLimit {
			return fmt.Errorf("global cost limit exceeded")
		}
	}

	return nil
}

// ProviderStatus represents the status of a provider
type ProviderStatus struct {
	Name         string                         `json:"name"`
	Model        string                         `json:"model"`
	Enabled      bool                           `json:"enabled"`
	Healthy      bool                           `json:"healthy"`
	Capabilities providers.ProviderCapabilities `json:"capabilities"`
	Usage        TokenUsageStats                `json:"usage"`
	Cost         float64                        `json:"cost"`
}

// UsageStats represents overall usage statistics
type UsageStats struct {
	TotalTokens int                        `json:"total_tokens"`
	TotalCost   float64                    `json:"total_cost"`
	ByProvider  map[string]TokenUsageStats `json:"by_provider"`
}
