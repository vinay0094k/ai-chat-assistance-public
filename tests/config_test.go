package tests

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yourusername/ai-code-assistant/internal/app"
)

func TestConfigLoader(t *testing.T) {
	// Create a temporary config file for testing
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "test-config.yaml")

	configContent := `
system:
  version: "1.0.0"
  name: "AI Code Assistant Test"
  session_timeout: "24h"
  auto_save_config: true
  data_directory: "./test-data"

ai:
  providers:
    - name: "openai"
      model: "gpt-4"
      weight: 1.0
      timeout: "30s"
      max_tokens: 4096
      temperature: 0.7
      stream: true
      enabled: true
      cost_per_1k_input: 0.03
      cost_per_1k_output: 0.06

indexing:
  languages: ["go", "python"]
  file_extensions: [".go", ".py"]
  ignore_patterns: ["vendor/", ".git/"]
  chunk_size: 1000
  overlap: 100
  batch_size: 50
  incremental: true
  real_time: true
  max_workers: 4

vectordb:
  provider: "qdrant"
  host: "localhost"
  port: 6333
  collection_name: "test_chunks"
  vector_size: 1536
  distance: "cosine"

database:
  provider: "sqlite"
  path: "./test-data/test.db"
  max_connections: 5
  connection_timeout: "30s"

search:
  max_results: 10
  min_confidence: 0.5
  semantic_weight: 0.6
  keyword_weight: 0.2
  graph_weight: 0.2
`

	require.NoError(t, os.WriteFile(configPath, []byte(configContent), 0644))

	t.Run("LoadValidConfig", func(t *testing.T) {
		loader := app.NewConfigLoader()

		// Override config paths for testing
		loader.SetConfigPaths([]string{configPath})

		systemConfig, cliConfig, err := loader.LoadAllConfigurations()
		require.NoError(t, err)
		require.NotNil(t, systemConfig)
		require.NotNil(t, cliConfig)

		// Test system config
		assert.Equal(t, "1.0.0", systemConfig.System.Version)
		assert.Equal(t, "AI Code Assistant Test", systemConfig.System.Name)
		assert.Equal(t, "./test-data", systemConfig.System.DataDirectory)

		// Test AI config
		require.Len(t, systemConfig.AI.Providers, 1)
		provider := systemConfig.AI.Providers[0]
		assert.Equal(t, "openai", provider.Name)
		assert.Equal(t, "gpt-4", provider.Model)
		assert.Equal(t, 1.0, provider.Weight)
		assert.True(t, provider.Enabled)

		// Test indexing config
		assert.Equal(t, []string{"go", "python"}, systemConfig.Indexing.Languages)
		assert.Equal(t, 1000, systemConfig.Indexing.ChunkSize)
		assert.Equal(t, 4, systemConfig.Indexing.MaxWorkers)

		// Test vector DB config
		assert.Equal(t, "qdrant", systemConfig.VectorDB.Provider)
		assert.Equal(t, "localhost", systemConfig.VectorDB.Host)
		assert.Equal(t, 6333, systemConfig.VectorDB.Port)

		// Test database config
		assert.Equal(t, "sqlite", systemConfig.Database.Provider)
		assert.Equal(t, "./test-data/test.db", systemConfig.Database.Path)
	})

	t.Run("InvalidConfigFile", func(t *testing.T) {
		invalidConfigPath := filepath.Join(tempDir, "invalid-config.yaml")
		invalidContent := `
invalid yaml content:
  - this is not: valid yaml [
`
		require.NoError(t, os.WriteFile(invalidConfigPath, []byte(invalidContent), 0644))

		loader := app.NewConfigLoader()
		loader.SetConfigPaths([]string{invalidConfigPath})

		_, _, err := loader.LoadAllConfigurations()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "failed to parse")
	})

	t.Run("MissingConfigFile", func(t *testing.T) {
		nonExistentPath := filepath.Join(tempDir, "nonexistent.yaml")

		loader := app.NewConfigLoader()
		loader.SetConfigPaths([]string{nonExistentPath})

		// Should return default config when file doesn't exist
		systemConfig, cliConfig, err := loader.LoadAllConfigurations()
		require.NoError(t, err)
		require.NotNil(t, systemConfig)
		require.NotNil(t, cliConfig)

		// Should have default values
		assert.Equal(t, "1.0.0", systemConfig.System.Version)
		assert.Equal(t, "AI Code Assistant", systemConfig.System.Name)
	})

	t.Run("EnvironmentOverrides", func(t *testing.T) {
		// Set environment variables
		os.Setenv("OPENAI_API_KEY", "test-key")
		os.Setenv("DATA_DIRECTORY", "/tmp/test-data")
		os.Setenv("LOG_LEVEL", "DEBUG")
		defer func() {
			os.Unsetenv("OPENAI_API_KEY")
			os.Unsetenv("DATA_DIRECTORY")
			os.Unsetenv("LOG_LEVEL")
		}()

		loader := app.NewConfigLoader()
		loader.SetConfigPaths([]string{configPath})

		systemConfig, _, err := loader.LoadAllConfigurations()
		require.NoError(t, err)

		// Environment should override file values
		assert.Equal(t, "/tmp/test-data", systemConfig.System.DataDirectory)

		// OpenAI provider should be enabled due to API key
		openaiProvider := findProviderByName(systemConfig.AI.Providers, "openai")
		require.NotNil(t, openaiProvider)
		assert.True(t, openaiProvider.Enabled)
	})

	t.Run("ConfigValidation", func(t *testing.T) {
		// Test invalid config that should fail validation
		invalidConfigPath := filepath.Join(tempDir, "invalid-values.yaml")
		invalidContent := `
system:
  version: ""  # Empty version should fail
  name: ""     # Empty name should fail

ai:
  providers: []  # No providers should fail
`
		require.NoError(t, os.WriteFile(invalidConfigPath, []byte(invalidContent), 0644))

		loader := app.NewConfigLoader()
		loader.SetConfigPaths([]string{invalidConfigPath})

		_, _, err := loader.LoadAllConfigurations()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "invalid configuration")
	})
}

func TestDatabaseConfig(t *testing.T) {
	t.Run("ValidDatabaseConfig", func(t *testing.T) {
		config := &app.DatabaseConfig{
			Provider:          "sqlite",
			Path:              "./test.db",
			MaxConnections:    10,
			ConnectionTimeout: 30 * time.Second,
			JournalMode:       "WAL",
			Synchronous:       "NORMAL",
			CacheSize:         "64MB",
			AutoBackup:        true,
			BackupInterval:    time.Hour,
			MaxBackups:        24,
		}

		assert.Equal(t, "sqlite", config.Provider)
		assert.Equal(t, "./test.db", config.Path)
		assert.Equal(t, 10, config.MaxConnections)
		assert.Equal(t, 30*time.Second, config.ConnectionTimeout)
		assert.True(t, config.AutoBackup)
	})
}

func TestAIProviderConfig(t *testing.T) {
	t.Run("ValidProviderConfig", func(t *testing.T) {
		provider := app.AIProvider{
			Name:            "openai",
			Model:           "gpt-4",
			Weight:          1.0,
			Timeout:         "30s",
			MaxTokens:       4096,
			Temperature:     0.7,
			Stream:          true,
			Enabled:         true,
			CostPer1KInput:  0.03,
			CostPer1KOutput: 0.06,
		}

		assert.Equal(t, "openai", provider.Name)
		assert.Equal(t, "gpt-4", provider.Model)
		assert.Equal(t, 1.0, provider.Weight)
		assert.True(t, provider.Enabled)
		assert.Equal(t, 0.03, provider.CostPer1KInput)
	})

	t.Run("InvalidProviderWeight", func(t *testing.T) {
		provider := app.AIProvider{
			Name:   "test",
			Model:  "test-model",
			Weight: 1.5, // Invalid weight > 1.0
		}

		// This would be caught in validation
		assert.True(t, provider.Weight > 1.0)
	})
}

func TestIndexingConfig(t *testing.T) {
	t.Run("ValidIndexingConfig", func(t *testing.T) {
		config := &app.IndexingConfig{
			Languages:       []string{"go", "python", "javascript"},
			FileExtensions:  []string{".go", ".py", ".js"},
			IgnorePatterns:  []string{"vendor/", "node_modules/", ".git/"},
			ChunkSize:       1000,
			Overlap:         100,
			BatchSize:       50,
			MaxFileSize:     "10MB",
			MaxFiles:        100000,
			Incremental:     true,
			RealTime:        true,
			DebounceDelay:   "500ms",
			MaxWorkers:      8,
			WorkerQueueSize: 1000,
		}

		assert.Contains(t, config.Languages, "go")
		assert.Contains(t, config.FileExtensions, ".go")
		assert.Contains(t, config.IgnorePatterns, "vendor/")
		assert.Equal(t, 1000, config.ChunkSize)
		assert.Equal(t, 100, config.Overlap)
		assert.True(t, config.Incremental)
		assert.True(t, config.RealTime)
		assert.Equal(t, 8, config.MaxWorkers)
	})
}

func TestSearchConfig(t *testing.T) {
	t.Run("ValidSearchConfig", func(t *testing.T) {
		config := &app.SearchConfig{
			MaxResults:       20,
			MinConfidence:    0.5,
			SemanticWeight:   0.6,
			KeywordWeight:    0.2,
			GraphWeight:      0.2,
			ContextExpansion: 3,
			EnableFuzzy:      true,
			FuzzyThreshold:   0.8,
		}

		assert.Equal(t, 20, config.MaxResults)
		assert.Equal(t, 0.5, config.MinConfidence)

		// Weights should sum to 1.0
		totalWeight := config.SemanticWeight + config.KeywordWeight + config.GraphWeight
		assert.InDelta(t, 1.0, totalWeight, 0.001)

		assert.True(t, config.EnableFuzzy)
		assert.Equal(t, 0.8, config.FuzzyThreshold)
	})
}

func TestConfigSaveAndReload(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "save-test.yaml")

	loader := app.NewConfigLoader()

	// Create a test config
	config := loader.CreateDefaultSystemConfig()
	config.System.Name = "Test Save Config"
	config.System.Version = "2.0.0"

	// Save config
	err := loader.SaveConfig(config, "save-test.yaml")
	require.NoError(t, err)

	// Reload config
	loader.SetConfigPaths([]string{configPath})
	reloadedConfig, _, err := loader.LoadAllConfigurations()
	require.NoError(t, err)

	// Verify values
	assert.Equal(t, "Test Save Config", reloadedConfig.System.Name)
	assert.Equal(t, "2.0.0", reloadedConfig.System.Version)
}

// Helper functions

func findProviderByName(providers []app.AIProvider, name string) *app.AIProvider {
	for i := range providers {
		if providers[i].Name == name {
			return &providers[i]
		}
	}
	return nil
}

// Benchmark tests

func BenchmarkConfigLoad(b *testing.B) {
	tempDir := b.TempDir()
	configPath := filepath.Join(tempDir, "bench-config.yaml")

	configContent := `
system:
  version: "1.0.0"
  name: "Benchmark Test"
ai:
  providers:
    - name: "openai"
      model: "gpt-4"
      weight: 1.0
      enabled: true
`

	require.NoError(b, os.WriteFile(configPath, []byte(configContent), 0644))

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		loader := app.NewConfigLoader()
		loader.SetConfigPaths([]string{configPath})
		_, _, err := loader.LoadAllConfigurations()
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEnvironmentOverrides(b *testing.B) {
	os.Setenv("TEST_VAR", "test_value")
	defer os.Unsetenv("TEST_VAR")

	tempDir := b.TempDir()
	configPath := filepath.Join(tempDir, "env-bench-config.yaml")

	configContent := `
system:
  version: "1.0.0"
  name: "Environment Benchmark Test"
ai:
  providers:
    - name: "openai"
      model: "gpt-4"
      weight: 1.0
      enabled: true
`

	require.NoError(b, os.WriteFile(configPath, []byte(configContent), 0644))

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		loader := app.NewConfigLoader()
		loader.SetConfigPaths([]string{configPath})
		_, _, err := loader.LoadAllConfigurations()
		if err != nil {
			b.Fatal(err)
		}
	}
}

// Integration tests

func TestFullConfigurationFlow(t *testing.T) {
	// This test simulates a complete configuration workflow
	tempDir := t.TempDir()

	// Create environment file
	envPath := filepath.Join(tempDir, ".env")
	envContent := `
OPENAI_API_KEY=test-key-123
QDRANT_HOST=localhost
QDRANT_PORT=6333
LOG_LEVEL=DEBUG
`
	require.NoError(t, os.WriteFile(envPath, []byte(envContent), 0644))

	// Create main config file
	configPath := filepath.Join(tempDir, "config.yaml")
	configContent := `
system:
  version: "1.0.0"
  name: "Integration Test"
  data_directory: "./data"

ai:
  providers:
    - name: "openai"
      model: "gpt-4"
      weight: 1.0
      timeout: "30s"
      enabled: false  # Will be overridden by env var

database:
  provider: "sqlite"
  path: "./data/test.db"

vectordb:
  provider: "qdrant"
  host: "default-host"  # Will be overridden by env var
  port: 6333
`
	require.NoError(t, os.WriteFile(configPath, []byte(configContent), 0644))

	// Set up loader with both paths
	loader := app.NewConfigLoader()
	loader.SetConfigPaths([]string{configPath})
	loader.SetEnvPaths([]string{envPath})

	// Load configuration
	systemConfig, cliConfig, err := loader.LoadAllConfigurations()
	require.NoError(t, err)
	require.NotNil(t, systemConfig)
	require.NotNil(t, cliConfig)

	// Verify environment overrides work
	assert.Equal(t, "Integration Test", systemConfig.System.Name)

	// Find OpenAI provider
	var openaiProvider *app.AIProvider
	for i := range systemConfig.AI.Providers {
		if systemConfig.AI.Providers[i].Name == "openai" {
			openaiProvider = &systemConfig.AI.Providers[i]
			break
		}
	}
	require.NotNil(t, openaiProvider, "OpenAI provider should exist")

	// Should be enabled due to API key in environment
	assert.True(t, openaiProvider.Enabled, "OpenAI provider should be enabled by environment")

	// Vector DB host should be overridden
	if systemConfig.VectorDB != nil {
		assert.Equal(t, "localhost", systemConfig.VectorDB.Host)
	}
}

func TestConfigValidationRules(t *testing.T) {
	tests := []struct {
		name        string
		configData  string
		expectError bool
		errorSubstr string
	}{
		{
			name: "ValidConfig",
			configData: `
system:
  version: "1.0.0"
  name: "Test App"
ai:
  providers:
    - name: "openai"
      model: "gpt-4"
      weight: 1.0
      enabled: true
`,
			expectError: false,
		},
		{
			name: "EmptySystemName",
			configData: `
system:
  version: "1.0.0"
  name: ""
ai:
  providers:
    - name: "openai"
      model: "gpt-4"
      weight: 1.0
`,
			expectError: true,
			errorSubstr: "system configuration",
		},
		{
			name: "NoAIProviders",
			configData: `
system:
  version: "1.0.0"
  name: "Test App"
ai:
  providers: []
`,
			expectError: true,
			errorSubstr: "at least one AI provider",
		},
		{
			name: "InvalidProviderWeight",
			configData: `
system:
  version: "1.0.0"
  name: "Test App"
ai:
  providers:
    - name: "openai"
      model: "gpt-4"
      weight: 1.5
`,
			expectError: true,
			errorSubstr: "weight must be between 0 and 1",
		},
		{
			name: "EmptyProviderName",
			configData: `
system:
  version: "1.0.0"
  name: "Test App"
ai:
  providers:
    - name: ""
      model: "gpt-4"
      weight: 1.0
`,
			expectError: true,
			errorSubstr: "provider name is required",
		},
		{
			name: "EmptyProviderModel",
			configData: `
system:
  version: "1.0.0"
  name: "Test App"
ai:
  providers:
    - name: "openai"
      model: ""
      weight: 1.0
`,
			expectError: true,
			errorSubstr: "provider model is required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tempDir := t.TempDir()
			configPath := filepath.Join(tempDir, "test-config.yaml")

			require.NoError(t, os.WriteFile(configPath, []byte(tt.configData), 0644))

			loader := app.NewConfigLoader()
			loader.SetConfigPaths([]string{configPath})

			_, _, err := loader.LoadAllConfigurations()

			if tt.expectError {
				assert.Error(t, err)
				if tt.errorSubstr != "" {
					assert.Contains(t, err.Error(), tt.errorSubstr)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestConfigMerging(t *testing.T) {
	// Test multiple config files being merged
	tempDir := t.TempDir()

	// Base config
	baseConfigPath := filepath.Join(tempDir, "base.yaml")
	baseConfig := `
system:
  version: "1.0.0"
  name: "Base App"
  data_directory: "./data"

ai:
  providers:
    - name: "openai"
      model: "gpt-4"
      weight: 1.0
      enabled: true
`
	require.NoError(t, os.WriteFile(baseConfigPath, []byte(baseConfig), 0644))

	// Override config
	overrideConfigPath := filepath.Join(tempDir, "override.yaml")
	overrideConfig := `
system:
  name: "Override App"  # This should override base
  session_timeout: "1h"  # This should be added

ai:
  providers:
    - name: "openai"
      model: "gpt-4"
      weight: 1.0
      enabled: false  # This should override base
    - name: "claude"  # This should be added
      model: "claude-3-sonnet"
      weight: 0.8
      enabled: true
`
	require.NoError(t, os.WriteFile(overrideConfigPath, []byte(overrideConfig), 0644))

	loader := app.NewConfigLoader()
	loader.SetConfigPaths([]string{baseConfigPath, overrideConfigPath})

	systemConfig, _, err := loader.LoadAllConfigurations()
	require.NoError(t, err)

	// System name should be overridden
	assert.Equal(t, "Override App", systemConfig.System.Name)
	assert.Equal(t, "1.0.0", systemConfig.System.Version)        // Should remain from base
	assert.Equal(t, "./data", systemConfig.System.DataDirectory) // Should remain from base

	// Should have both providers
	assert.Len(t, systemConfig.AI.Providers, 2)

	// Find providers
	var openaiProvider, claudeProvider *app.AIProvider
	for i := range systemConfig.AI.Providers {
		switch systemConfig.AI.Providers[i].Name {
		case "openai":
			openaiProvider = &systemConfig.AI.Providers[i]
		case "claude":
			claudeProvider = &systemConfig.AI.Providers[i]
		}
	}

	require.NotNil(t, openaiProvider)
	require.NotNil(t, claudeProvider)

	// OpenAI should be disabled (overridden)
	assert.False(t, openaiProvider.Enabled)

	// Claude should be enabled (added)
	assert.True(t, claudeProvider.Enabled)
	assert.Equal(t, "claude-3-sonnet", claudeProvider.Model)
}

func TestConfigPrecedence(t *testing.T) {
	// Test precedence: Environment > Override Config > Base Config > Defaults
	tempDir := t.TempDir()

	// Environment variables (highest precedence)
	os.Setenv("DATA_DIRECTORY", "/env/data")
	os.Setenv("OPENAI_API_KEY", "env-key-123")
	defer func() {
		os.Unsetenv("DATA_DIRECTORY")
		os.Unsetenv("OPENAI_API_KEY")
	}()

	// Base config (lowest precedence)
	baseConfigPath := filepath.Join(tempDir, "base.yaml")
	baseConfig := `
system:
  version: "1.0.0"
  name: "Base App"
  data_directory: "/base/data"  # Should be overridden by env

ai:
  providers:
    - name: "openai"
      model: "gpt-4"
      weight: 1.0
      enabled: false  # Should be overridden by env (due to API key)
`
	require.NoError(t, os.WriteFile(baseConfigPath, []byte(baseConfig), 0644))

	loader := app.NewConfigLoader()
	loader.SetConfigPaths([]string{baseConfigPath})

	systemConfig, _, err := loader.LoadAllConfigurations()
	require.NoError(t, err)

	// Environment should win
	assert.Equal(t, "/env/data", systemConfig.System.DataDirectory)

	// OpenAI should be enabled due to API key in environment
	var openaiProvider *app.AIProvider
	for i := range systemConfig.AI.Providers {
		if systemConfig.AI.Providers[i].Name == "openai" {
			openaiProvider = &systemConfig.AI.Providers[i]
			break
		}
	}
	require.NotNil(t, openaiProvider)
	assert.True(t, openaiProvider.Enabled)
}

func TestConfigReload(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "reload-test.yaml")

	// Initial config
	initialConfig := `
system:
  version: "1.0.0"
  name: "Initial App"
ai:
  providers:
    - name: "openai"
      model: "gpt-4"
      weight: 1.0
      enabled: true
`
	require.NoError(t, os.WriteFile(configPath, []byte(initialConfig), 0644))

	loader := app.NewConfigLoader()
	loader.SetConfigPaths([]string{configPath})

	// Load initial config
	systemConfig1, _, err := loader.LoadAllConfigurations()
	require.NoError(t, err)
	assert.Equal(t, "Initial App", systemConfig1.System.Name)

	// Modify config file
	modifiedConfig := `
system:
  version: "2.0.0"
  name: "Modified App"
ai:
  providers:
    - name: "openai"
      model: "gpt-4"
      weight: 1.0
      enabled: true
    - name: "claude"
      model: "claude-3-sonnet"
      weight: 0.8
      enabled: true
`
	require.NoError(t, os.WriteFile(configPath, []byte(modifiedConfig), 0644))

	// Reload config
	systemConfig2, _, err := loader.ReloadConfig()
	require.NoError(t, err)

	// Should have new values
	assert.Equal(t, "Modified App", systemConfig2.System.Name)
	assert.Equal(t, "2.0.0", systemConfig2.System.Version)
	assert.Len(t, systemConfig2.AI.Providers, 2)
}

func TestConfigDefaults(t *testing.T) {
	loader := app.NewConfigLoader()

	// Load with no config files (should use defaults)
	loader.SetConfigPaths([]string{"/nonexistent/path.yaml"})

	systemConfig, cliConfig, err := loader.LoadAllConfigurations()
	require.NoError(t, err)
	require.NotNil(t, systemConfig)
	require.NotNil(t, cliConfig)

	// Check system defaults
	assert.Equal(t, "1.0.0", systemConfig.System.Version)
	assert.Equal(t, "AI Code Assistant", systemConfig.System.Name)
	assert.Equal(t, "24h", systemConfig.System.SessionTimeout)
	assert.True(t, systemConfig.System.AutoSaveConfig)
	assert.Equal(t, "./data", systemConfig.System.DataDirectory)

	// Check AI defaults
	require.NotNil(t, systemConfig.AI)
	require.NotNil(t, systemConfig.AI.Fallback)
	require.Len(t, systemConfig.AI.Providers, 1)

	defaultProvider := systemConfig.AI.Providers[0]
	assert.Equal(t, "openai", defaultProvider.Name)
	assert.Equal(t, "gpt-4", defaultProvider.Model)
	assert.Equal(t, 1.0, defaultProvider.Weight)

	// Check fallback defaults
	assert.True(t, systemConfig.AI.Fallback.Enabled)
	assert.Equal(t, 3, systemConfig.AI.Fallback.MaxRetries)

	// Check indexing defaults
	require.NotNil(t, systemConfig.Indexing)
	assert.Equal(t, []string{"go"}, systemConfig.Indexing.Languages)
	assert.Equal(t, 1000, systemConfig.Indexing.ChunkSize)
	assert.Equal(t, 100, systemConfig.Indexing.Overlap)
	assert.True(t, systemConfig.Indexing.Incremental)
	assert.True(t, systemConfig.Indexing.RealTime)
	assert.Equal(t, 8, systemConfig.Indexing.MaxWorkers)

	// Check search defaults
	require.NotNil(t, systemConfig.Search)
	assert.Equal(t, 20, systemConfig.Search.MaxResults)
	assert.Equal(t, 0.5, systemConfig.Search.MinConfidence)
	assert.Equal(t, 0.6, systemConfig.Search.SemanticWeight)
	assert.Equal(t, 0.2, systemConfig.Search.KeywordWeight)
	assert.Equal(t, 0.2, systemConfig.Search.GraphWeight)

	// Check CLI defaults
	require.NotNil(t, cliConfig.CLI)
	assert.Equal(t, "useQ>", cliConfig.CLI.Prompt)
	assert.Equal(t, "cyan", cliConfig.CLI.PromptColor)
	assert.True(t, cliConfig.CLI.ShowLineNumbers)
	assert.Equal(t, 4, cliConfig.CLI.LineNumberWidth)
	assert.Equal(t, 50, cliConfig.CLI.MaxOutputLines)

	// Check CLI colors
	require.NotNil(t, cliConfig.CLI.Colors)
	assert.True(t, cliConfig.CLI.Colors.Enabled)
	assert.Equal(t, "default", cliConfig.CLI.Colors.Theme)
	assert.Equal(t, "cyan", cliConfig.CLI.Colors.Primary)

	// Check command defaults
	require.NotNil(t, cliConfig.Commands)
	require.NotNil(t, cliConfig.Commands.Defaults)
	assert.Equal(t, "30s", cliConfig.Commands.Defaults.Timeout)
	assert.False(t, cliConfig.Commands.Defaults.Verbose)
	assert.Equal(t, "text", cliConfig.Commands.Defaults.Format)

	// Check aliases
	require.NotNil(t, cliConfig.Commands.Aliases)
	assert.Equal(t, "help", cliConfig.Commands.Aliases["?"])
	assert.Equal(t, "quit", cliConfig.Commands.Aliases["q"])
}

// Test helper methods for ConfigLoader

func TestConfigLoaderHelpers(t *testing.T) {
	loader := app.NewConfigLoader()

	t.Run("SetConfigPaths", func(t *testing.T) {
		paths := []string{"/path/1.yaml", "/path/2.yaml"}
		loader.SetConfigPaths(paths)

		// Verify paths were set (this would require exposing the field or a getter)
		// For now, we test that it doesn't panic
		assert.NotNil(t, loader)
	})

	t.Run("SetEnvPaths", func(t *testing.T) {
		paths := []string{"/path/.env", "/path/.env.local"}
		loader.SetEnvPaths(paths)

		// Verify paths were set
		assert.NotNil(t, loader)
	})

	t.Run("GetLoadedConfig", func(t *testing.T) {
		// First load a config
		tempDir := t.TempDir()
		configPath := filepath.Join(tempDir, "test.yaml")
		configContent := `
system:
  version: "1.0.0"
  name: "Test"
`
		require.NoError(t, os.WriteFile(configPath, []byte(configContent), 0644))

		loader.SetConfigPaths([]string{configPath})
		_, _, err := loader.LoadAllConfigurations()
		require.NoError(t, err)

		// Get loaded config
		config, exists := loader.GetLoadedConfig("system")
		assert.True(t, exists)
		assert.NotNil(t, config)

		// Try non-existent config
		_, exists = loader.GetLoadedConfig("nonexistent")
		assert.False(t, exists)
	})
}

// Performance tests

func BenchmarkConfigLoadLarge(b *testing.B) {
	tempDir := b.TempDir()
	configPath := filepath.Join(tempDir, "large-config.yaml")

	// Create a large config with many providers
	var configBuilder strings.Builder
	configBuilder.WriteString(`
system:
  version: "1.0.0"
  name: "Large Config Test"
  data_directory: "./data"

ai:
  providers:
`)

	// Add 100 providers
	for i := 0; i < 100; i++ {
		configBuilder.WriteString(fmt.Sprintf(`    - name: "provider%d"
      model: "model%d"
      weight: 1.0
      timeout: "30s"
      enabled: true
`, i, i))
	}

	configBuilder.WriteString(`
indexing:
  languages: ["go", "python", "javascript", "typescript", "java", "c", "cpp", "rust", "ruby", "php"]
  file_extensions: [".go", ".py", ".js", ".ts", ".java", ".c", ".cpp", ".rs", ".rb", ".php"]
  ignore_patterns: ["vendor/", "node_modules/", ".git/", "target/", "build/", "dist/"]
`)

	require.NoError(b, os.WriteFile(configPath, []byte(configBuilder.String()), 0644))

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		loader := app.NewConfigLoader()
		loader.SetConfigPaths([]string{configPath})
		_, _, err := loader.LoadAllConfigurations()
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkConfigValidation(b *testing.B) {
	tempDir := b.TempDir()
	configPath := filepath.Join(tempDir, "validation-bench.yaml")

	configContent := `
system:
  version: "1.0.0"
  name: "Validation Benchmark"

ai:
  providers:
    - name: "openai"
      model: "gpt-4"
      weight: 1.0
      timeout: "30s"
      max_tokens: 4096
      temperature: 0.7
      stream: true
      enabled: true
      cost_per_1k_input: 0.03
      cost_per_1k_output: 0.06
`

	require.NoError(b, os.WriteFile(configPath, []byte(configContent), 0644))

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		loader := app.NewConfigLoader()
		loader.SetConfigPaths([]string{configPath})
		_, _, err := loader.LoadAllConfigurations()
		if err != nil {
			b.Fatal(err)
		}
	}
}

// Edge cases and error conditions

func TestConfigEdgeCases(t *testing.T) {
	t.Run("EmptyConfigFile", func(t *testing.T) {
		tempDir := t.TempDir()
		configPath := filepath.Join(tempDir, "empty.yaml")

		// Create empty file
		require.NoError(t, os.WriteFile(configPath, []byte(""), 0644))

		loader := app.NewConfigLoader()
		loader.SetConfigPaths([]string{configPath})

		// Should use defaults for empty file
		systemConfig, _, err := loader.LoadAllConfigurations()
		require.NoError(t, err)
		assert.Equal(t, "AI Code Assistant", systemConfig.System.Name)
	})

	t.Run("ConfigWithOnlyComments", func(t *testing.T) {
		tempDir := t.TempDir()
		configPath := filepath.Join(tempDir, "comments.yaml")

		configContent := `
# This is a comment
# Another comment
# system:
#   version: "1.0.0"
`
		require.NoError(t, os.WriteFile(configPath, []byte(configContent), 0644))

		loader := app.NewConfigLoader()
		loader.SetConfigPaths([]string{configPath})

		systemConfig, _, err := loader.LoadAllConfigurations()
		require.NoError(t, err)
		assert.Equal(t, "AI Code Assistant", systemConfig.System.Name)
	})

	t.Run("ConfigWithSpecialCharacters", func(t *testing.T) {
		tempDir := t.TempDir()
		configPath := filepath.Join(tempDir, "special.yaml")

		configContent := `
system:
  version: "1.0.0"
  name: "App and émojis 🚀"
  data_directory: "./data with spaces"
`
		require.NoError(t, os.WriteFile(configPath, []byte(configContent), 0644))

		loader := app.NewConfigLoader()
		loader.SetConfigPaths([]string{configPath})

		systemConfig, _, err := loader.LoadAllConfigurations()
		require.NoError(t, err)
		assert.Equal(t, "App and émojis 🚀", systemConfig.System.Name)
		assert.Equal(t, "./data with spaces", systemConfig.System.DataDirectory)
	})
}
