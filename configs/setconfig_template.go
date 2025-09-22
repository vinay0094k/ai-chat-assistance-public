package configs

import (
	"fmt"

	"github.com/yourusername/ai-code-assistant/internal/app"
)

// Template for implementing SetConfig in remaining agents
// Apply this pattern to each agent:

// For DebuggingAgent:
func (da *DebuggingAgent) SetConfig(config interface{}) error {
	appConfig, ok := config.(*app.DebuggingConfig)
	if !ok {
		return fmt.Errorf("invalid config type for DebuggingAgent, expected *app.DebuggingConfig")
	}
	da.mu.Lock()
	defer da.mu.Unlock()

	da.config = &DebuggingAgentConfig{
		EnableBugDetection:        appConfig.EnableBugDetection,
		EnablePerformanceAnalysis: appConfig.EnablePerformanceAnalysis,
		EnableSecurityAnalysis:    appConfig.EnableSecurityAnalysis,
		EnableFixSuggestions:      appConfig.EnableFixSuggestions,
		LLMModel:                  appConfig.LLMModel,
		MaxTokens:                 appConfig.MaxTokens,
		Temperature:               appConfig.Temperature,
	}
	da.initializeComponents()
	return nil
}

// For SearchAgent:
func (sa *SearchAgent) SetConfig(config interface{}) error {
	appConfig, ok := config.(*app.SearchAgentConfig)
	if !ok {
		return fmt.Errorf("invalid config type for SearchAgent, expected *app.SearchAgentConfig")
	}
	sa.mu.Lock()
	defer sa.mu.Unlock()

	sa.config = &SearchAgentConfig{
		MaxResults:          appConfig.MaxResults,
		EnableFuzzySearch:   appConfig.EnableFuzzySearch,
		EnableRegexSearch:   appConfig.EnableRegexSearch,
		SimilarityThreshold: appConfig.SimilarityThreshold,
	}
	sa.initializeComponents()
	return nil
}

// For ContextAwareSearchAgent:
func (casa *ContextAwareSearchAgent) SetConfig(config interface{}) error {
	appConfig, ok := config.(*app.ContextAwareSearchConfig)
	if !ok {
		return fmt.Errorf("invalid config type for ContextAwareSearchAgent, expected *app.ContextAwareSearchConfig")
	}
	casa.mu.Lock()
	defer casa.mu.Unlock()

	casa.config = &ContextAwareSearchAgentConfig{
		EnableSemanticSearch: appConfig.EnableSemanticSearch,
		EnableContextRanking: appConfig.EnableContextRanking,
		MaxResults:           appConfig.MaxResults,
		SimilarityThreshold:  appConfig.SimilarityThreshold,
		EnableCaching:        appConfig.EnableCaching,
	}
	casa.initializeComponents()
	return nil
}
