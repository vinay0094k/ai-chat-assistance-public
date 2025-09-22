package llm

import (
	"sync"
	"time"
)

// CostCalculator tracks and calculates AI usage costs
type CostCalculator struct {
	providerCosts map[string]float64
	totalCost     float64
	mu            sync.RWMutex
	startTime     time.Time
}

// NewCostCalculator creates a new cost calculator
func NewCostCalculator() *CostCalculator {
	return &CostCalculator{
		providerCosts: make(map[string]float64),
		startTime:     time.Now(),
	}
}

// AddUsage adds usage cost for a provider
func (c *CostCalculator) AddUsage(provider string, cost float64) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.providerCosts[provider] += cost
	c.totalCost += cost
}

// GetProviderCost returns cost for a specific provider
func (c *CostCalculator) GetProviderCost(provider string) float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.providerCosts[provider]
}

// GetTotalCost returns total cost across all providers
func (c *CostCalculator) GetTotalCost() float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.totalCost
}

// GetAllProviderCosts returns costs for all providers
func (c *CostCalculator) GetAllProviderCosts() map[string]float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()

	costs := make(map[string]float64)
	for provider, cost := range c.providerCosts {
		costs[provider] = cost
	}

	return costs
}

// Reset resets all cost tracking
func (c *CostCalculator) Reset() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.providerCosts = make(map[string]float64)
	c.totalCost = 0
	c.startTime = time.Now()
}

// GetStats returns cost statistics
func (c *CostCalculator) GetStats() CostStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return CostStats{
		TotalCost:     c.totalCost,
		ProviderCosts: c.getAllProviderCosts(),
		Duration:      time.Since(c.startTime),
		StartTime:     c.startTime,
	}
}

func (c *CostCalculator) getAllProviderCosts() map[string]float64 {
	costs := make(map[string]float64)
	for provider, cost := range c.providerCosts {
		costs[provider] = cost
	}
	return costs
}

// CostStats represents cost statistics
type CostStats struct {
	TotalCost     float64            `json:"total_cost"`
	ProviderCosts map[string]float64 `json:"provider_costs"`
	Duration      time.Duration      `json:"duration"`
	StartTime     time.Time          `json:"start_time"`
}
