package main

import (
	"fmt"
	"log"

	"github.com/yourusername/ai-code-assistant/internal/app"
)

func main() {
	// Test the new configuration loading system
	configLoader := app.NewConfigLoader()

	systemConfig, cliConfig, agentsConfig, err := configLoader.LoadAllConfigurations()
	if err != nil {
		log.Fatalf("Failed to load configurations: %v", err)
	}

	fmt.Println("=== Configuration Loading Test ===")

	// Test system config
	if systemConfig != nil {
		fmt.Printf("System Config Loaded: %s v%s\n",
			systemConfig.System.Name,
			systemConfig.System.Version)
	}

	// Test CLI config
	if cliConfig != nil {
		fmt.Printf("CLI Config Loaded: %d themes available\n",
			len(cliConfig.Themes))
	}

	// Test agents config
	if agentsConfig != nil {
		fmt.Println("Agent Configurations Loaded:")

		if agentsConfig.Coding != nil {
			fmt.Printf("  - Coding Agent: Model=%s, Tokens=%d\n",
				agentsConfig.Coding.LLMModel,
				agentsConfig.Coding.MaxTokens)
		}

		if agentsConfig.Documentation != nil {
			fmt.Printf("  - Documentation Agent: Format=%s, Examples=%t\n",
				agentsConfig.Documentation.DefaultFormat,
				agentsConfig.Documentation.EnableExampleGeneration)
		}

		if agentsConfig.Testing != nil {
			fmt.Printf("  - Testing Agent: Coverage=%.1f%%, Model=%s\n",
				agentsConfig.Testing.CoverageTarget,
				agentsConfig.Testing.LLMModel)
		}
	}

	fmt.Println("✅ Configuration system working correctly!")
}
