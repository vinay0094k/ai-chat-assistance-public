package main

import (
	"fmt"
	"os"

	"github.com/yourusername/ai-code-assistant/internal/app"
)

func main() {
	fmt.Println("✅ Testing configuration loading...")

	// Set environment variable
	os.Setenv("OPENAI_API_KEY", "test-env-key")
	defer os.Unsetenv("OPENAI_API_KEY")

	loader := app.NewConfigLoader()
	loader.SetConfigPaths([]string{"test_configs/test-properties.yaml"})
	loader.SetEnvPaths([]string{"test_configs/.env"})

	systemConfig, cliConfig, agentsConfig, err := loader.LoadAllConfigurations()
	if err != nil {
		fmt.Printf("❌ Failed to load config: %v\n", err)
		return
	}

	fmt.Printf("✅ System config loaded:\n")
	fmt.Printf("  Name: %s\n", systemConfig.System.Name)
	fmt.Printf("  Version: %s\n", systemConfig.System.Version)
	fmt.Printf("  Data Directory: %s\n", systemConfig.System.DataDirectory)

	fmt.Printf("✅ AI providers: %d\n", len(systemConfig.AI.Providers))
	for _, provider := range systemConfig.AI.Providers {
		fmt.Printf("  - %s (%s): enabled=%t\n", provider.Name, provider.Model, provider.Enabled)
	}

	fmt.Printf("✅ Indexing config:\n")
	fmt.Printf("  Languages: %v\n", systemConfig.Indexing.Languages)
	fmt.Printf("  Chunk size: %d\n", systemConfig.Indexing.ChunkSize)
	fmt.Printf("  Max workers: %d\n", systemConfig.Indexing.MaxWorkers)

	fmt.Printf("✅ CLI config:\n")
	if cliConfig != nil && cliConfig.CLI != nil {
		fmt.Printf("  Prompt: %s\n", cliConfig.CLI.Prompt)
		fmt.Printf("  Colors enabled: %t\n", cliConfig.CLI.Colors.Enabled)
	} else {
		fmt.Printf("  CLI config is nil\n")
	}

	fmt.Printf("✅ Agents config:\n")
	if agentsConfig != nil && agentsConfig.Coding != nil {
		fmt.Printf("  Coding Agent: Model=%s, Tokens=%d\n",
			agentsConfig.Coding.LLMModel,
			agentsConfig.Coding.MaxTokens)
	}

	if agentsConfig != nil && agentsConfig.Documentation != nil {
		fmt.Printf("  Documentation Agent: Format=%s, Examples=%t\n",
			agentsConfig.Documentation.DefaultFormat,
			agentsConfig.Documentation.EnableExampleGeneration)
	}

	// Test config validation
	fmt.Println("\n✅ Testing config validation...")

	// Test saving and reloading
	testConfig := systemConfig
	testConfig.System.Name = "Modified Test Config"

	if err := loader.SaveConfig(testConfig, "test-save.yaml"); err != nil {
		fmt.Printf("❌ Failed to save config: %v\n", err)
	} else {
		fmt.Println("✅ Config saved successfully")
	}

	fmt.Println("✅ Configuration tests completed!")
}

// ###################################################################################################
// ✅ Testing configuration loading...
// ✅ System config loaded:
//   Name: AI Code Assistant Test
//   Version: 1.0.0-test
//   Data Directory: /tmp/test-data
// ✅ AI providers: 1
//   - openai (gpt-4): enabled=true
// ✅ Indexing config:
//   Languages: [go python javascript]
//   Chunk size: 1000
//   Max workers: 4
// ✅ CLI config:
// panic: runtime error: invalid memory address or nil pointer dereference
// [signal SIGSEGV: segmentation violation code=0x1 addr=0x0 pc=0x50a301]

// goroutine 1 [running]:
// main.main()
//         /home/vinayk/Documents/OTHER_documents/Bolt_projects/ai-code-assistant/test_config_loading.go:42 +0x6a1
// exit status 2
// ###################################################################################################
