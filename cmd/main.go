package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/joho/godotenv"
	"github.com/spf13/cobra"
	"github.com/yourusername/ai-code-assistant/internal/app"
)

var (
	version = "1.0.0"
	commit  = "dev"
	date    = "unknown"
)

func main() {
	// Load environment variables first
	if err := loadEnvironmentVariables(); err != nil {
		fmt.Printf("Warning: %v\n", err)
	}

	// Create root command
	rootCmd := createRootCommand()

	// Execute command
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func createRootCommand() *cobra.Command {
	rootCmd := &cobra.Command{
		Use:     "ai-assistant",
		Short:   "AI-powered code assistant CLI",
		Long:    `A CLI tool that helps with code generation, search, explanation, and more using AI with MCP integration.`,
		Version: fmt.Sprintf("%s (commit: %s, built: %s)", version, commit, date),
		Run:     runInteractiveCLI,
	}

	// Add flags
	rootCmd.Flags().StringP("query", "q", "", "Execute a single query and exit")
	rootCmd.Flags().StringP("project", "p", ".", "Project path to index")
	rootCmd.Flags().BoolP("verbose", "v", false, "Enable verbose logging")
	rootCmd.Flags().StringP("format", "f", "text", "Output format (text, json, table)")
	rootCmd.Flags().BoolP("no-color", "n", false, "Disable colored output")
	rootCmd.Flags().BoolP("debug", "d", false, "Enable debug mode")
	rootCmd.Flags().StringP("env-file", "e", "", "Path to .env file (overrides default locations)")

	// Add subcommands
	rootCmd.AddCommand(createVersionCmd())
	rootCmd.AddCommand(createConfigCmd())
	rootCmd.AddCommand(createInitCmd())
	rootCmd.AddCommand(createEnvCmd())

	return rootCmd
}

func runInteractiveCLI(cmd *cobra.Command, args []string) {
	// Parse flags
	query, _ := cmd.Flags().GetString("query")
	projectPath, _ := cmd.Flags().GetString("project")
	verbose, _ := cmd.Flags().GetBool("verbose")
	format, _ := cmd.Flags().GetString("format")
	noColor, _ := cmd.Flags().GetBool("no-color")
	debug, _ := cmd.Flags().GetBool("debug")
	envFile, _ := cmd.Flags().GetString("env-file")

	// Load specific env file if provided
	if envFile != "" {
		if err := godotenv.Load(envFile); err != nil {
			fmt.Printf("Warning: Could not load %s: %v\n", envFile, err)
		}
	}

	// Load all configurations
	configLoader := app.NewConfigLoader()
	systemConfig, cliConfig, agentsConfig, err := configLoader.LoadAllConfigurations()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load configurations: %v\n", err)
		os.Exit(1)
	}

	// Create CLI application with loaded configs
	cliApp, err := app.NewCLIApp(&app.CLIConfig{
		ProjectPath: projectPath,
		Verbose:     verbose,
		Format:      format,
		NoColor:     noColor,
		Debug:       debug,
	}, systemConfig, agentsConfig, cliConfig)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize CLI: %v\n", err)
		os.Exit(1)
	}

	// Run single query or interactive mode
	if query != "" {
		if err := cliApp.RunSingleQuery(query); err != nil {
			fmt.Fprintf(os.Stderr, "Query failed: %v\n", err)
			os.Exit(1)
		}
		return
	}

	// Run interactive mode
	if err := cliApp.RunInteractive(); err != nil {
		fmt.Fprintf(os.Stderr, "Interactive mode failed: %v\n", err)
		os.Exit(1)
	}
}

func loadEnvironmentVariables() error {
	// Priority order for loading .env files:
	// 1. .env.local (highest priority - for local overrides)
	// 2. .env
	// 3. ~/.ai-assistant/.env (user global config)
	// 4. /etc/ai-assistant/.env (system-wide config)

	homeDir, _ := os.UserHomeDir()

	envFiles := []string{
		".env.local", // Local overrides
		".env",       // Project specific
		filepath.Join(homeDir, ".ai-assistant", ".env"), // User global
		"/etc/ai-assistant/.env",                        // System-wide
	}

	var loadedFiles []string
	var lastError error

	for _, envFile := range envFiles {
		if _, err := os.Stat(envFile); err == nil {
			if err := godotenv.Load(envFile); err != nil {
				lastError = fmt.Errorf("failed to load %s: %v", envFile, err)
			} else {
				loadedFiles = append(loadedFiles, envFile)
			}
		}
	}

	// Print loaded files in verbose mode
	if len(loadedFiles) > 0 && os.Getenv("VERBOSE") == "true" {
		fmt.Printf("Loaded environment files: %v\n", loadedFiles)
	}

	return lastError
}

func createVersionCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "version",
		Short: "Show version information",
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Printf("AI Code Assistant %s\n", version)
			fmt.Printf("Commit: %s\n", commit)
			fmt.Printf("Built: %s\n", date)
			fmt.Printf("Go version: %s\n", runtime.Version())
		},
	}
}

func createConfigCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "config",
		Short: "Show current configuration",
		Run: func(cmd *cobra.Command, args []string) {
			// Load all configurations
			configLoader := app.NewConfigLoader()
			systemConfig, cliConfig, agentsConfig, err := configLoader.LoadAllConfigurations()
			if err != nil {
				fmt.Fprintf(os.Stderr, "Failed to load configurations: %v\n", err)
				os.Exit(1)
			}

			cliApp, err := app.NewCLIApp(&app.CLIConfig{}, systemConfig, agentsConfig, cliConfig)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Failed to initialize: %v\n", err)
				os.Exit(1)
			}
			cliApp.ShowConfig()
		},
	}
}

func createInitCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "init",
		Short: "Initialize configuration files",
		Long:  "Create initial configuration files including .env template",
		Run: func(cmd *cobra.Command, args []string) {
			if err := initializeConfiguration(); err != nil {
				fmt.Fprintf(os.Stderr, "Initialization failed: %v\n", err)
				os.Exit(1)
			}
			fmt.Println("✅ Configuration initialized successfully!")
		},
	}
}

func createEnvCmd() *cobra.Command {
	envCmd := &cobra.Command{
		Use:   "env",
		Short: "Environment variable management",
	}

	envCmd.AddCommand(&cobra.Command{
		Use:   "show",
		Short: "Show current environment variables",
		Run: func(cmd *cobra.Command, args []string) {
			showEnvironmentVariables()
		},
	})

	envCmd.AddCommand(&cobra.Command{
		Use:   "validate",
		Short: "Validate environment configuration",
		Run: func(cmd *cobra.Command, args []string) {
			validateEnvironmentConfiguration()
		},
	})

	envCmd.AddCommand(&cobra.Command{
		Use:   "template",
		Short: "Generate .env template",
		Run: func(cmd *cobra.Command, args []string) {
			generateEnvTemplate()
		},
	})

	return envCmd
}

func initializeConfiguration() error {
	// Create user config directory
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("could not get home directory: %v", err)
	}

	configDir := filepath.Join(homeDir, ".ai-assistant")
	if err := os.MkdirAll(configDir, 0755); err != nil {
		return fmt.Errorf("could not create config directory: %v", err)
	}

	// Create .env file if it doesn't exist
	envPath := filepath.Join(configDir, ".env")
	if _, err := os.Stat(envPath); os.IsNotExist(err) {
		if err := createDefaultEnvFile(envPath); err != nil {
			return err
		}
	}

	// Create local .env.example if it doesn't exist
	localEnvExample := ".env.example"
	if _, err := os.Stat(localEnvExample); os.IsNotExist(err) {
		if err := createEnvExample(localEnvExample); err != nil {
			return err
		}
	}

	fmt.Printf("Configuration directory: %s\n", configDir)
	fmt.Printf("User environment file: %s\n", envPath)
	fmt.Printf("Local environment example: %s\n", localEnvExample)
	fmt.Println("\nNext steps:")
	fmt.Println("1. Copy .env.example to .env")
	fmt.Println("2. Edit .env with your API keys")
	fmt.Println("3. Run: ai-assistant env validate")

	return nil
}

func createDefaultEnvFile(path string) error {
	content := `# AI Code Assistant Configuration
# Generated by 'ai-assistant init'
#
# Copy this file to your project root as '.env' and fill in your API keys
# You can also copy .env.example to .env and customize it

# AI Provider Configuration (at least one required)
OPENAI_API_KEY=
GEMINI_API_KEY=
COHERE_API_KEY=
CLAUDE_API_KEY=

# AI Model Selection (optional - uses defaults if not specified)
OPENAI_MODEL=gpt-4
GEMINI_MODEL=gemini-pro
COHERE_MODEL=command
CLAUDE_MODEL=claude-3-sonnet-20240229

# Database Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
SQLITE_PATH=./data/assistant.db

# MCP Server Configuration
MCP_FILESYSTEM_ENABLED=true
MCP_GIT_ENABLED=true
MCP_GITHUB_ENABLED=false
MCP_GITHUB_TOKEN=
MCP_SQLITE_ENABLED=true
MCP_DOCKER_ENABLED=false

# Display Configuration
ENABLE_COLORS=true
ENABLE_SYNTAX_HIGHLIGHTING=true
ENABLE_PROGRESS_BARS=true
STREAMING_DELAY_MS=50
MAX_LINE_LENGTH=120

# System Configuration
LOG_LEVEL=INFO
MAX_PARALLEL_INDEXING=4
TOKEN_WARNING_THRESHOLD=1000
COST_WARNING_THRESHOLD=1.0
SESSION_TIMEOUT=24h

# Development & Debug
DEBUG_MODE=false
VERBOSE=false
PROFILE_ENABLED=false
`

	return os.WriteFile(path, []byte(content), 0644)
}

func createEnvExample(path string) error {
	// Read from configs/.env.example if it exists, otherwise create default
	examplePath := "configs/.env.example"
	if content, err := os.ReadFile(examplePath); err == nil {
		return os.WriteFile(path, content, 0644)
	}

	// Create default .env.example
	return createDefaultEnvFile(path)
}

func showEnvironmentVariables() {
	fmt.Println("Current Environment Configuration:")
	fmt.Println(strings.Repeat("=", 50))

	envVars := []struct {
		Key         string
		Description string
		Required    bool
	}{
		{"OPENAI_API_KEY", "OpenAI API Key", false},
		{"GEMINI_API_KEY", "Google Gemini API Key", false},
		{"COHERE_API_KEY", "Cohere API Key", false},
		{"CLAUDE_API_KEY", "Anthropic Claude API Key", false},
		{"LOG_LEVEL", "Logging Level", false},
		{"ENABLE_COLORS", "Enable Color Output", false},
		{"DEBUG_MODE", "Debug Mode", false},
		{"QDRANT_HOST", "Qdrant Database Host", false},
		{"QDRANT_PORT", "Qdrant Database Port", false},
	}

	for _, env := range envVars {
		value := os.Getenv(env.Key)
		status := "✅"
		displayValue := value

		if value == "" {
			status = "❌"
			displayValue = "(not set)"
		} else if strings.Contains(strings.ToLower(env.Key), "key") || strings.Contains(strings.ToLower(env.Key), "token") {
			// Hide sensitive values
			displayValue = "***hidden***"
		}

		required := ""
		if env.Required {
			required = " (required)"
		}

		fmt.Printf("%s %-25s: %s%s\n", status, env.Key, displayValue, required)
	}
}

func validateEnvironmentConfiguration() {
	fmt.Println("Validating Environment Configuration...")
	fmt.Println(strings.Repeat("=", 50))

	hasError := false

	// Check for at least one AI provider
	providers := []string{"OPENAI_API_KEY", "GEMINI_API_KEY", "COHERE_API_KEY", "CLAUDE_API_KEY"}
	hasProvider := false
	for _, provider := range providers {
		if os.Getenv(provider) != "" {
			hasProvider = true
			break
		}
	}

	if !hasProvider {
		fmt.Println("❌ No AI provider API keys configured")
		fmt.Println("   At least one AI provider key is required:")
		for _, provider := range providers {
			fmt.Printf("   - %s\n", provider)
		}
		hasError = true
	} else {
		fmt.Println("✅ AI provider configuration found")
	}

	// Validate LOG_LEVEL
	logLevel := os.Getenv("LOG_LEVEL")
	validLogLevels := []string{"DEBUG", "INFO", "WARN", "ERROR"}
	if logLevel != "" {
		isValidLogLevel := false
		for _, level := range validLogLevels {
			if logLevel == level {
				isValidLogLevel = true
				break
			}
		}
		if !isValidLogLevel {
			fmt.Printf("❌ Invalid LOG_LEVEL: %s (valid: %v)\n", logLevel, validLogLevels)
			hasError = true
		} else {
			fmt.Printf("✅ LOG_LEVEL: %s\n", logLevel)
		}
	}

	// Validate boolean values
	boolEnvs := map[string]string{
		"ENABLE_COLORS":              os.Getenv("ENABLE_COLORS"),
		"ENABLE_SYNTAX_HIGHLIGHTING": os.Getenv("ENABLE_SYNTAX_HIGHLIGHTING"),
		"DEBUG_MODE":                 os.Getenv("DEBUG_MODE"),
		"VERBOSE":                    os.Getenv("VERBOSE"),
	}

	for key, value := range boolEnvs {
		if value != "" && value != "true" && value != "false" {
			fmt.Printf("❌ Invalid boolean value for %s: %s (must be 'true' or 'false')\n", key, value)
			hasError = true
		}
	}

	if !hasError {
		fmt.Println("✅ Environment configuration is valid!")
	} else {
		fmt.Println("\n❌ Environment configuration has errors. Please fix them and run 'ai-assistant env validate' again.")
		os.Exit(1)
	}
}

func generateEnvTemplate() {
	fmt.Println("Generating .env template...")

	if err := createDefaultEnvFile(".env.template"); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create template: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("✅ Created .env.template")
	fmt.Println("Copy this file to .env and fill in your configuration values")
}
