package configs

import (
	"fmt"
	"strconv"

	"github.com/yourusername/ai-code-assistant/internal/agents"
	"github.com/yourusername/ai-code-assistant/internal/app"
)

// ConfigUpdater provides methods for dynamic configuration updates
type ConfigUpdater struct {
	router *agents.AgentRouter
}

// NewConfigUpdater creates a new configuration updater
func NewConfigUpdater(router *agents.AgentRouter) *ConfigUpdater {
	return &ConfigUpdater{router: router}
}

// UpdateAgentConfig updates a specific agent's configuration
func (cu *ConfigUpdater) UpdateAgentConfig(agentName, configKey, configValue string, agentsConfig *app.AllAgentsConfig) error {
	switch agentName {
	case "coding":
		return cu.updateCodingConfig(configKey, configValue, agentsConfig.Coding)
	case "documentation":
		return cu.updateDocumentationConfig(configKey, configValue, agentsConfig.Documentation)
	case "testing":
		return cu.updateTestingConfig(configKey, configValue, agentsConfig.Testing)
	case "debugging":
		return cu.updateDebuggingConfig(configKey, configValue, agentsConfig.Debugging)
	case "search":
		return cu.updateSearchConfig(configKey, configValue, agentsConfig.Search)
	default:
		return fmt.Errorf("unknown agent: %s", agentName)
	}
}

func (cu *ConfigUpdater) updateCodingConfig(key, value string, currentConfig *app.CodingConfig) error {
	newConfig := *currentConfig

	switch key {
	case "llm_model":
		newConfig.LLMModel = value
	case "max_tokens":
		tokens, err := strconv.Atoi(value)
		if err != nil {
			return fmt.Errorf("invalid max_tokens: %v", err)
		}
		newConfig.MaxTokens = tokens
	case "temperature":
		temp, err := strconv.ParseFloat(value, 32)
		if err != nil {
			return fmt.Errorf("invalid temperature: %v", err)
		}
		newConfig.Temperature = float32(temp)
	case "enable_refactoring":
		newConfig.EnableRefactoring = value == "true"
	case "enable_quality_check":
		newConfig.EnableQualityCheck = value == "true"
	default:
		return fmt.Errorf("unknown config key: %s", key)
	}

	return cu.router.UpdateAgentConfig(agents.AgentTypeCoding, &newConfig)
}

func (cu *ConfigUpdater) updateDocumentationConfig(key, value string, currentConfig *app.DocumentationConfig) error {
	newConfig := *currentConfig

	switch key {
	case "default_format":
		newConfig.DefaultFormat = value
	case "llm_model":
		newConfig.LLMModel = value
	case "enable_examples":
		newConfig.EnableExampleGeneration = value == "true"
	case "enable_api_docs":
		newConfig.EnableAPIDocGeneration = value == "true"
	default:
		return fmt.Errorf("unknown config key: %s", key)
	}

	return cu.router.UpdateAgentConfig(agents.AgentTypeDocumentation, &newConfig)
}

func (cu *ConfigUpdater) updateTestingConfig(key, value string, currentConfig *app.TestingConfig) error {
	newConfig := *currentConfig

	switch key {
	case "coverage_target":
		target, err := strconv.ParseFloat(value, 32)
		if err != nil {
			return fmt.Errorf("invalid coverage_target: %v", err)
		}
		newConfig.CoverageTarget = float32(target)
	case "enable_mocks":
		newConfig.EnableMockGeneration = value == "true"
	case "enable_integration_tests":
		newConfig.EnableIntegrationTestGeneration = value == "true"
	default:
		return fmt.Errorf("unknown config key: %s", key)
	}

	return cu.router.UpdateAgentConfig(agents.AgentTypeTesting, &newConfig)
}

func (cu *ConfigUpdater) updateDebuggingConfig(key, value string, currentConfig *app.DebuggingConfig) error {
	newConfig := *currentConfig

	switch key {
	case "llm_model":
		newConfig.LLMModel = value
	case "enable_fix_suggestions":
		newConfig.EnableFixSuggestions = value == "true"
	case "enable_security_analysis":
		newConfig.EnableSecurityAnalysis = value == "true"
	default:
		return fmt.Errorf("unknown config key: %s", key)
	}

	return cu.router.UpdateAgentConfig(agents.AgentTypeDebugging, &newConfig)
}

func (cu *ConfigUpdater) updateSearchConfig(key, value string, currentConfig *app.SearchAgentConfig) error {
	newConfig := *currentConfig

	switch key {
	case "max_results":
		results, err := strconv.Atoi(value)
		if err != nil {
			return fmt.Errorf("invalid max_results: %v", err)
		}
		newConfig.MaxResults = results
	case "enable_fuzzy_search":
		newConfig.EnableFuzzySearch = value == "true"
	case "similarity_threshold":
		threshold, err := strconv.ParseFloat(value, 32)
		if err != nil {
			return fmt.Errorf("invalid similarity_threshold: %v", err)
		}
		newConfig.SimilarityThreshold = float32(threshold)
	default:
		return fmt.Errorf("unknown config key: %s", key)
	}

	return cu.router.UpdateAgentConfig(agents.AgentTypeSearch, &newConfig)
}
