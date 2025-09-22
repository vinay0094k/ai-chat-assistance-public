package app

import (
	"bufio"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/joho/godotenv"

	"github.com/fatih/color"
)

type CLIApp struct {
	config       *CLIConfig
	sessionID    string
	startTime    time.Time
	queryCount   int64
	isRunning    bool
	envConfig    *EnvironmentConfig
	systemConfig *SystemConfig
	cliConfig    *CLISpecificConfig
	agentsConfig *AllAgentsConfig
	// logger       *logger.Logger
}

type CLIConfig struct {
	ProjectPath string
	Verbose     bool
	Format      string
	NoColor     bool
	Debug       bool
}

type EnvironmentConfig struct {
	// AI Providers
	OpenAIKey string
	GeminiKey string
	CohereKey string
	ClaudeKey string

	// AI Models
	OpenAIModel string
	GeminiModel string
	CohereModel string
	ClaudeModel string

	// Database
	QdrantHost string
	QdrantPort int
	SQLitePath string

	// MCP Servers
	MCPFilesystemEnabled bool
	MCPGitEnabled        bool
	MCPGithubEnabled     bool
	MCPGithubToken       string
	MCPSQLiteEnabled     bool
	MCPDockerEnabled     bool

	// Display
	EnableColors             bool
	EnableSyntaxHighlighting bool
	EnableProgressBars       bool
	StreamingDelayMS         int
	MaxLineLength            int

	// System
	LogLevel              string
	MaxParallelIndexing   int
	TokenWarningThreshold int
	CostWarningThreshold  float64
	SessionTimeout        string

	// Development
	DebugMode      bool
	Verbose        bool
	ProfileEnabled bool
}

func NewCLIApp(config *CLIConfig, systemConfig *SystemConfig, agentsConfig *AllAgentsConfig, cliConfig *CLISpecificConfig) (*CLIApp, error) {
	if config == nil {
		config = &CLIConfig{
			ProjectPath: ".",
			Format:      "text",
		}
	}

	// Load environment configuration
	envConfig, err := loadEnvironmentConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load environment config: %v", err)
	}

	// Override CLI config with environment settings
	if envConfig.Verbose {
		config.Verbose = true
	}
	if envConfig.DebugMode {
		config.Debug = true
	}
	if !envConfig.EnableColors {
		config.NoColor = true
	}

	// Disable colors if requested
	if config.NoColor {
		color.NoColor = true
	}

	sessionID := generateSessionID()

	app := &CLIApp{
		config:       config,
		envConfig:    envConfig,
		systemConfig: systemConfig,
		cliConfig:    cliConfig,
		agentsConfig: agentsConfig,
		sessionID:    sessionID,
		startTime:    time.Now(),
		isRunning:    false,
	}

	return app, nil
}

func loadEnvironmentConfig() (*EnvironmentConfig, error) {
	config := &EnvironmentConfig{
		// Set defaults
		OpenAIModel:              getEnvWithDefault("OPENAI_MODEL", "gpt-4"),
		GeminiModel:              getEnvWithDefault("GEMINI_MODEL", "gemini-pro"),
		CohereModel:              getEnvWithDefault("COHERE_MODEL", "command"),
		ClaudeModel:              getEnvWithDefault("CLAUDE_MODEL", "claude-3-sonnet-20240229"),
		QdrantHost:               getEnvWithDefault("QDRANT_HOST", "localhost"),
		QdrantPort:               getEnvIntWithDefault("QDRANT_PORT", 6333),
		SQLitePath:               getEnvWithDefault("SQLITE_PATH", "./data/assistant.db"),
		EnableColors:             getEnvBoolWithDefault("ENABLE_COLORS", true),
		EnableSyntaxHighlighting: getEnvBoolWithDefault("ENABLE_SYNTAX_HIGHLIGHTING", true),
		EnableProgressBars:       getEnvBoolWithDefault("ENABLE_PROGRESS_BARS", true),
		StreamingDelayMS:         getEnvIntWithDefault("STREAMING_DELAY_MS", 50),
		MaxLineLength:            getEnvIntWithDefault("MAX_LINE_LENGTH", 120),
		LogLevel:                 getEnvWithDefault("LOG_LEVEL", "INFO"),
		MaxParallelIndexing:      getEnvIntWithDefault("MAX_PARALLEL_INDEXING", 4),
		TokenWarningThreshold:    getEnvIntWithDefault("TOKEN_WARNING_THRESHOLD", 1000),
		CostWarningThreshold:     getEnvFloatWithDefault("COST_WARNING_THRESHOLD", 1.0),
		SessionTimeout:           getEnvWithDefault("SESSION_TIMEOUT", "24h"),
		DebugMode:                getEnvBoolWithDefault("DEBUG_MODE", false),
		Verbose:                  getEnvBoolWithDefault("VERBOSE", false),
		ProfileEnabled:           getEnvBoolWithDefault("PROFILE_ENABLED", false),
	}

	// Load API keys
	config.OpenAIKey = os.Getenv("OPENAI_API_KEY")
	config.GeminiKey = os.Getenv("GEMINI_API_KEY")
	config.CohereKey = os.Getenv("COHERE_API_KEY")
	config.ClaudeKey = os.Getenv("CLAUDE_API_KEY")

	// Load MCP configuration
	config.MCPFilesystemEnabled = getEnvBoolWithDefault("MCP_FILESYSTEM_ENABLED", true)
	config.MCPGitEnabled = getEnvBoolWithDefault("MCP_GIT_ENABLED", true)
	config.MCPGithubEnabled = getEnvBoolWithDefault("MCP_GITHUB_ENABLED", false)
	config.MCPGithubToken = os.Getenv("MCP_GITHUB_TOKEN")
	config.MCPSQLiteEnabled = getEnvBoolWithDefault("MCP_SQLITE_ENABLED", true)
	config.MCPDockerEnabled = getEnvBoolWithDefault("MCP_DOCKER_ENABLED", false)

	return config, nil
}

func (cli *CLIApp) RunInteractive() error {
	cli.isRunning = true

	// Setup signal handling for graceful shutdown
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		cli.shutdown()
		os.Exit(0)
	}()

	// Print welcome banner
	cli.printBanner()

	// Initialize system
	if err := cli.initialize(); err != nil {
		return fmt.Errorf("initialization failed: %v", err)
	}

	// Start interactive loop
	return cli.interactiveLoop()
}

func (cli *CLIApp) RunSingleQuery(query string) error {
	cli.printBanner()

	if err := cli.initialize(); err != nil {
		return fmt.Errorf("initialization failed: %v", err)
	}

	return cli.processQuery(query)
}

func (cli *CLIApp) initialize() error {
	if cli.config.Verbose || cli.envConfig.Verbose {
		fmt.Println("🔄 Initializing system...")
		fmt.Printf("├─ Session ID: %s\n", cli.sessionID)
		fmt.Printf("├─ Project Path: %s\n", cli.config.ProjectPath)
		fmt.Printf("├─ Output Format: %s\n", cli.config.Format)
		fmt.Printf("├─ Debug Mode: %t\n", cli.config.Debug)
		fmt.Printf("├─ Log Level: %s\n", cli.envConfig.LogLevel)

		// Show available AI providers
		providers := cli.getAvailableProviders()
		if len(providers) > 0 {
			fmt.Printf("├─ AI Providers: %s\n", strings.Join(providers, ", "))
		} else {
			fmt.Printf("├─ ⚠️  No AI providers configured\n")
		}

		// Show MCP servers status
		mcpServers := cli.getEnabledMCPServers()
		if len(mcpServers) > 0 {
			fmt.Printf("├─ MCP Servers: %s\n", strings.Join(mcpServers, ", "))
		}

		fmt.Printf("└─ Memory: %s\n", formatMemoryUsage())
	}

	// Validate configuration
	if err := cli.validateConfiguration(); err != nil {
		return fmt.Errorf("configuration validation failed: %v", err)
	}

	// TODO: Initialize components
	// - Database connections
	// - MCP servers
	// - AI providers

	if cli.config.Verbose || cli.envConfig.Verbose {
		fmt.Println("✅ System initialized")
	}

	return nil
}

func (cli *CLIApp) validateConfiguration() error {
	// Check for at least one AI provider
	providers := cli.getAvailableProviders()
	if len(providers) == 0 {
		return fmt.Errorf("no AI provider API keys configured. Please set at least one: OPENAI_API_KEY, GEMINI_API_KEY, COHERE_API_KEY, or CLAUDE_API_KEY")
	}

	// Validate log level
	validLogLevels := []string{"DEBUG", "INFO", "WARN", "ERROR"}
	isValidLogLevel := false
	for _, level := range validLogLevels {
		if cli.envConfig.LogLevel == level {
			isValidLogLevel = true
			break
		}
	}
	if !isValidLogLevel {
		return fmt.Errorf("invalid LOG_LEVEL: %s (valid: %v)", cli.envConfig.LogLevel, validLogLevels)
	}

	return nil
}

func (cli *CLIApp) getAvailableProviders() []string {
	var providers []string

	if cli.envConfig.OpenAIKey != "" {
		providers = append(providers, "openai")
	}
	if cli.envConfig.GeminiKey != "" {
		providers = append(providers, "gemini")
	}
	if cli.envConfig.CohereKey != "" {
		providers = append(providers, "cohere")
	}
	if cli.envConfig.ClaudeKey != "" {
		providers = append(providers, "claude")
	}

	return providers
}

func (cli *CLIApp) getEnabledMCPServers() []string {
	var servers []string

	if cli.envConfig.MCPFilesystemEnabled {
		servers = append(servers, "filesystem")
	}
	if cli.envConfig.MCPGitEnabled {
		servers = append(servers, "git")
	}
	if cli.envConfig.MCPGithubEnabled {
		servers = append(servers, "github")
	}
	if cli.envConfig.MCPSQLiteEnabled {
		servers = append(servers, "sqlite")
	}
	if cli.envConfig.MCPDockerEnabled {
		servers = append(servers, "docker")
	}

	return servers
}

func (cli *CLIApp) ShowConfig() {
	fmt.Println("\n" + strings.Repeat("═", 70))
	fmt.Println("                        CONFIGURATION")
	fmt.Println(strings.Repeat("═", 70))

	// CLI Configuration
	fmt.Println("CLI Settings:")
	fmt.Printf("  Project Path: %s\n", cli.config.ProjectPath)
	fmt.Printf("  Output Format: %s\n", cli.config.Format)
	fmt.Printf("  Verbose Mode: %t\n", cli.config.Verbose)
	fmt.Printf("  Debug Mode: %t\n", cli.config.Debug)
	fmt.Printf("  Colors Enabled: %t\n", !cli.config.NoColor)

	// Environment Configuration
	fmt.Println("\nEnvironment Settings:")
	fmt.Printf("  Log Level: %s\n", cli.envConfig.LogLevel)
	fmt.Printf("  Max Parallel Indexing: %d\n", cli.envConfig.MaxParallelIndexing)
	fmt.Printf("  Token Warning Threshold: %d\n", cli.envConfig.TokenWarningThreshold)
	fmt.Printf("  Cost Warning Threshold: $%.2f\n", cli.envConfig.CostWarningThreshold)

	// AI Providers
	fmt.Println("\nAI Providers:")
	providers := cli.getAvailableProviders()
	if len(providers) > 0 {
		for _, provider := range providers {
			var model string
			switch provider {
			case "openai":
				model = cli.envConfig.OpenAIModel
			case "gemini":
				model = cli.envConfig.GeminiModel
			case "cohere":
				model = cli.envConfig.CohereModel
			case "claude":
				model = cli.envConfig.ClaudeModel
			}
			fmt.Printf("  ✅ %s (%s)\n", provider, model)
		}
	} else {
		fmt.Printf("  ❌ No providers configured\n")
	}

	// MCP Servers
	fmt.Println("\nMCP Servers:")
	mcpServers := cli.getEnabledMCPServers()
	if len(mcpServers) > 0 {
		for _, server := range mcpServers {
			fmt.Printf("  ✅ %s\n", server)
		}
	} else {
		fmt.Printf("  ❌ No MCP servers enabled\n")
	}

	// Database Configuration
	fmt.Println("\nDatabase Configuration:")
	fmt.Printf("  Qdrant: %s:%d\n", cli.envConfig.QdrantHost, cli.envConfig.QdrantPort)
	fmt.Printf("  SQLite: %s\n", cli.envConfig.SQLitePath)

	// Display Configuration
	fmt.Println("\nDisplay Configuration:")
	fmt.Printf("  Syntax Highlighting: %t\n", cli.envConfig.EnableSyntaxHighlighting)
	fmt.Printf("  Progress Bars: %t\n", cli.envConfig.EnableProgressBars)
	fmt.Printf("  Streaming Delay: %dms\n", cli.envConfig.StreamingDelayMS)
	fmt.Printf("  Max Line Length: %d\n", cli.envConfig.MaxLineLength)

	fmt.Println(strings.Repeat("═", 70))
}

// Continue from internal/app/cli.go

func (cli *CLIApp) printBanner() {
	if cli.config.NoColor || !cli.envConfig.EnableColors {
		fmt.Println("AI Code Assistant CLI v1.0.0")
		fmt.Printf("Session: %s | Project: %s\n", cli.sessionID, cli.config.ProjectPath)
		fmt.Println("Type 'help' for commands or 'quit' to exit")
		fmt.Println(strings.Repeat("-", 60))
		return
	}

	// Colored banner
	cyan := color.New(color.FgCyan, color.Bold)
	green := color.New(color.FgGreen)
	yellow := color.New(color.FgYellow)

	fmt.Println()
	cyan.Println("╔══════════════════════════════════════════════════════════════╗")
	cyan.Println("║               AI Code Assistant CLI v1.0.0                   ║")
	cyan.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	green.Printf("Session: %s\n", cli.sessionID)
	yellow.Printf("Project: %s\n", cli.config.ProjectPath)

	// Show configuration status
	providers := cli.getAvailableProviders()
	if len(providers) > 0 {
		green.Printf("AI Providers: %s\n", strings.Join(providers, ", "))
	} else {
		color.New(color.FgRed).Printf("⚠️  No AI providers configured - run 'ai-assistant init' to setup\n")
	}

	fmt.Println()
	fmt.Println("Type 'help' for commands or 'quit' to exit")
	fmt.Println()
}

func (cli *CLIApp) interactiveLoop() error {
	scanner := bufio.NewScanner(os.Stdin)

	for cli.isRunning {
		// Print prompt
		cli.printPrompt()

		// Read input
		if !scanner.Scan() {
			break
		}

		query := strings.TrimSpace(scanner.Text())
		if query == "" {
			continue
		}

		// Handle system commands
		if cli.handleSystemCommand(query) {
			continue
		}

		// Handle exit commands
		if query == "quit" || query == "exit" {
			cli.printSessionSummary()
			break
		}

		// Process query
		if err := cli.processQuery(query); err != nil {
			cli.printError(fmt.Sprintf("Query failed: %v", err))
		}

		cli.queryCount++
	}

	return scanner.Err()
}

func (cli *CLIApp) printPrompt() {
	if cli.config.NoColor || !cli.envConfig.EnableColors {
		fmt.Print("useQ> ")
		return
	}

	cyan := color.New(color.FgCyan, color.Bold)
	cyan.Print("useQ> ")
}

func (cli *CLIApp) handleSystemCommand(query string) bool {
	switch query {
	case "help":
		cli.printHelp()
		return true
	case "status":
		cli.printStatus()
		return true
	case "config":
		cli.ShowConfig()
		return true
	case "env":
		cli.showEnvironmentStatus()
		return true
	case "providers":
		cli.showProviderStatus()
		return true
	case "mcp":
		cli.showMCPStatus()
		return true
	case "clear":
		cli.clearScreen()
		return true
	case "version":
		cli.printVersion()
		return true
	case "reload":
		cli.reloadConfiguration()
		return true
	default:
		if strings.HasPrefix(query, "set ") {
			cli.handleSetCommand(query[4:])
			return true
		}
		return false
	}
}

func (cli *CLIApp) processQuery(query string) error {
	startTime := time.Now()
	queryID := generateQueryID()

	if cli.config.Verbose || cli.envConfig.Verbose {
		fmt.Printf("\n┌─ Processing Query: %s\n", queryID)
		fmt.Printf("│  Query: %s\n", query)
		fmt.Printf("│  Time: %s\n", time.Now().Format("15:04:05"))
		fmt.Printf("│  Debug Mode: %t\n", cli.config.Debug)
		fmt.Printf("└─ \n")
	}

	// Parse the query using your existing prompt parser
	parser := NewPromptParser()
	parsedPrompt := parser.Parse(query)

	if cli.config.Debug || cli.envConfig.DebugMode {
		cli.printDebugInfo(parsedPrompt)
	}

	// Route to appropriate handler based on intent
	var err error
	switch parsedPrompt.Intent {
	case "search_code":
		err = cli.executeSearchCode(parsedPrompt)
	case "generate_code":
		err = cli.executeGenerateCode(parsedPrompt)
	case "explain_code":
		err = cli.executeExplainCode(parsedPrompt)
	case "generate_tests":
		err = cli.executeGenerateTests(parsedPrompt)
	case "refactor_code":
		err = cli.executeRefactorCode(parsedPrompt)
	case "generate_documentation":
		err = cli.executeGenerateDocumentation(parsedPrompt)
	case "fix_bug":
		err = cli.executeFixBug(parsedPrompt)
	case "review_code":
		err = cli.executeReviewCode(parsedPrompt)
	case "analyze_dependencies":
		err = cli.executeAnalyzeDependencies(parsedPrompt)
	default:
		err = cli.executeGenericQuery(parsedPrompt)
	}

	if err != nil {
		return err
	}

	// Show processing summary
	fmt.Printf("⏱️  Processing time: %v\n", time.Since(startTime))
	fmt.Printf("🆔 Query ID: %s\n", queryID)

	return nil
}

func (cli *CLIApp) printDebugInfo(parsedPrompt *ParsedPrompt) {
	if !cli.config.NoColor && cli.envConfig.EnableColors {
		yellow := color.New(color.FgYellow)
		yellow.Println("🐛 Debug Information:")
	} else {
		fmt.Println("🐛 Debug Information:")
	}

	fmt.Printf("   Raw Input: %s\n", parsedPrompt.RawInput)
	fmt.Printf("   Intent: %s\n", parsedPrompt.Intent)
	fmt.Printf("   Confidence: %.2f\n", parsedPrompt.Confidence)
	fmt.Printf("   Is System Command: %t\n", parsedPrompt.IsSystemCmd)
	fmt.Printf("   Context: %s\n", parsedPrompt.Context)
	fmt.Printf("   Entities: %v\n", parsedPrompt.Entities)
	fmt.Printf("   Parameters: %v\n", parsedPrompt.Parameters)
	fmt.Println()
}

func (cli *CLIApp) executeSearchCode(parsed *ParsedPrompt) error {
	fmt.Printf("🔍 Searching for code...\n")
	fmt.Printf("🎯 Intent: %s (confidence: %.2f)\n", parsed.Intent, parsed.Confidence)

	if functionName, exists := parsed.Entities["function_name"]; exists {
		fmt.Printf("🔎 Looking for function: %s\n", functionName)
	}
	if fileName, exists := parsed.Entities["file_name"]; exists {
		fmt.Printf("📁 In file: %s\n", fileName)
	}

	// Mock search results
	results := `📋 Found 3 results:

1. Score: 0.95
   File: auth/login.go:42
   Function: authenticateUser
   Context: Handles user authentication and login validation

2. Score: 0.87  
   File: middleware/auth.go:15
   Function: validateToken
   Context: JWT token validation middleware

3. Score: 0.78
   File: handlers/user.go:89
   Function: loginHandler
   Context: HTTP handler for user login endpoint`

	fmt.Println(results)
	fmt.Println("✅ Search completed successfully")
	return nil
}

func (cli *CLIApp) executeGenerateCode(parsed *ParsedPrompt) error {
	fmt.Printf("🤖 Generating code...\n")
	fmt.Printf("🎯 Intent: %s (confidence: %.2f)\n", parsed.Intent, parsed.Confidence)

	if genType, exists := parsed.Entities["generation_type"]; exists {
		fmt.Printf("📝 Generation type: %s\n", genType)
	}
	if codePattern, exists := parsed.Entities["code_pattern"]; exists {
		fmt.Printf("🏗️  Code pattern: %s\n", codePattern)
	}

	// Mock code generation with streaming effect
	code := `func authenticateUser(username, password string) (*User, error) {
    // Hash the provided password
    hashedPassword := hashPassword(password)
    
    // Query the database for the user
    user, err := db.GetUserByUsername(username)
    if err != nil {
        return nil, fmt.Errorf("user not found: %w", err)
    }
    
    // Verify password
    if user.PasswordHash != hashedPassword {
        return nil, errors.New("invalid credentials")
    }
    
    return user, nil
}`

	fmt.Println("\n📝 Generated code:")
	fmt.Println("```go")
	lines := strings.Split(code, "\n")
	for _, line := range lines {
		fmt.Println(line)
		time.Sleep(50 * time.Millisecond)
	}
	fmt.Println("```")

	fmt.Println("✅ Code generation completed successfully")
	return nil
}

func (cli *CLIApp) executeExplainCode(parsed *ParsedPrompt) error {
	fmt.Printf("📖 Explaining code...\n")
	fmt.Printf("🎯 Intent: %s (confidence: %.2f)\n", parsed.Intent, parsed.Confidence)

	if target, exists := parsed.Entities["explain_target"]; exists {
		fmt.Printf("🔍 Explaining: %s\n", target)
	}

	explanation := `📚 Code Explanation:

OVERVIEW:
This code implements user authentication functionality that validates credentials
and manages user sessions securely.

KEY CONCEPTS:
  🔐 Security: Uses password hashing for secure credential validation
  🗄️  Database: Queries user database for authentication data
  ⚡ Performance: Efficient lookup and validation process
  🛡️  Error Handling: Comprehensive error management for edge cases

IMPLEMENTATION DETAILS:
  • Password hashing prevents storing plaintext passwords
  • Database queries use prepared statements for security
  • Error messages are generic to prevent user enumeration attacks
  • Session management tracks user login state`

	fmt.Println(explanation)
	fmt.Println("✅ Code explanation completed")
	return nil
}

func (cli *CLIApp) executeGenerateTests(parsed *ParsedPrompt) error {
	fmt.Printf("🧪 Generating tests...\n")
	fmt.Printf("🎯 Intent: %s (confidence: %.2f)\n", parsed.Intent, parsed.Confidence)

	if target, exists := parsed.Entities["test_target"]; exists {
		fmt.Printf("🎯 Test target: %s\n", target)
	}
	if testType, exists := parsed.Entities["test_type"]; exists {
		fmt.Printf("📋 Test type: %s\n", testType)
	}

	testCode := `func TestAuthenticateUser(t *testing.T) {
    tests := []struct {
        name     string
        username string
        password string
        want     *User
        wantErr  bool
    }{
        {
            name:     "valid credentials",
            username: "testuser",
            password: "validpassword",
            want:     &User{Username: "testuser"},
            wantErr:  false,
        },
        {
            name:     "invalid password",
            username: "testuser", 
            password: "wrongpassword",
            want:     nil,
            wantErr:  true,
        },
        {
            name:     "user not found",
            username: "nonexistent",
            password: "anypassword",
            want:     nil,
            wantErr:  true,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got, err := authenticateUser(tt.username, tt.password)
            if (err != nil) != tt.wantErr {
                t.Errorf("authenticateUser() error = %v, wantErr %v", err, tt.wantErr)
                return
            }
            if !reflect.DeepEqual(got, tt.want) {
                t.Errorf("authenticateUser() = %v, want %v", got, tt.want)
            }
        })
    }
}`

	fmt.Println("\n🧪 Generated test code:")
	fmt.Println("```go")
	fmt.Println(testCode)
	fmt.Println("```")
	fmt.Println("✅ Test generation completed successfully")
	return nil
}

func (cli *CLIApp) executeRefactorCode(parsed *ParsedPrompt) error {
	fmt.Printf("🔧 Refactoring code...\n")
	fmt.Printf("🎯 Intent: %s (confidence: %.2f)\n", parsed.Intent, parsed.Confidence)

	if target, exists := parsed.Entities["refactor_target"]; exists {
		fmt.Printf("🎯 Refactor target: %s\n", target)
	}

	suggestions := `🔧 Refactoring Suggestions:

CURRENT ISSUES:
  ⚠️  Function complexity too high (cyclomatic complexity: 8)
  ⚠️  Multiple responsibilities in single function
  ⚠️  Hard-coded dependencies make testing difficult
  ⚠️  Error handling could be more specific

RECOMMENDED IMPROVEMENTS:
  1. Extract password validation to separate function
  2. Implement dependency injection for database
  3. Add structured error types
  4. Reduce cyclomatic complexity through early returns

REFACTORED STRUCTURE:
  • AuthService struct with injected dependencies
  • Separate validation, lookup, and session methods
  • Custom error types for better error handling
  • Interface-based design for testability`

	fmt.Println(suggestions)
	fmt.Println("✅ Refactoring analysis completed")
	return nil
}

func (cli *CLIApp) executeGenerateDocumentation(parsed *ParsedPrompt) error {
	fmt.Printf("📝 Generating documentation...\n")
	fmt.Printf("🎯 Intent: %s (confidence: %.2f)\n", parsed.Intent, parsed.Confidence)

	if target, exists := parsed.Entities["doc_target"]; exists {
		fmt.Printf("📋 Documentation target: %s\n", target)
	}

	documentation := `📖 Generated Documentation:

// authenticateUser validates user credentials and returns user data
//
// This function performs secure authentication by:
// 1. Hashing the provided password using a secure algorithm
// 2. Querying the database for user records
// 3. Comparing hashed passwords to prevent timing attacks
// 4. Returning user data on successful authentication
//
// Parameters:
//   username: The user's login identifier (must not be empty)
//   password: The user's plaintext password (will be hashed)
//
// Returns:
//   *User: User data structure on successful authentication
//   error: Authentication error or system error
//
// Example:
//   user, err := authenticateUser("john@example.com", "mypassword")
//   if err != nil {
//       log.Printf("Authentication failed: %v", err)
//       return
//   }
//   fmt.Printf("Welcome %s", user.Name)`

	fmt.Println(documentation)
	fmt.Println("✅ Documentation generation completed")
	return nil
}

func (cli *CLIApp) executeFixBug(parsed *ParsedPrompt) error {
	fmt.Printf("🐛 Analyzing bug...\n")
	fmt.Printf("🎯 Intent: %s (confidence: %.2f)\n", parsed.Intent, parsed.Confidence)

	if location, exists := parsed.Entities["bug_location"]; exists {
		fmt.Printf("📍 Bug location: %s\n", location)
	}

	bugAnalysis := `🔍 Bug Analysis:

POTENTIAL ISSUES DETECTED:
  🚨 Possible null pointer dereference
  🚨 Race condition in concurrent access
  🚨 Memory leak in resource cleanup
  🚨 Incorrect error handling pattern

RECOMMENDED FIXES:
  1. Add null checks before dereferencing pointers
  2. Implement proper mutex locking for shared resources
  3. Ensure all resources are properly closed with defer statements
  4. Use wrapped errors for better error context

PREVENTION STRATEGIES:
  • Add unit tests covering edge cases
  • Implement static analysis in CI pipeline
  • Use linting tools to catch common patterns
  • Add logging for better debugging`

	fmt.Println(bugAnalysis)
	fmt.Println("✅ Bug analysis completed")
	return nil
}

func (cli *CLIApp) executeReviewCode(parsed *ParsedPrompt) error {
	fmt.Printf("📋 Reviewing code...\n")
	fmt.Printf("🎯 Intent: %s (confidence: %.2f)\n", parsed.Intent, parsed.Confidence)

	if target, exists := parsed.Entities["review_target"]; exists {
		fmt.Printf("🎯 Review target: %s\n", target)
	}
	if reviewType, exists := parsed.Entities["review_type"]; exists {
		fmt.Printf("📋 Review type: %s\n", reviewType)
	}

	review := `📊 Code Review Report:

QUALITY SCORE: B+ (83/100)

STRENGTHS:
  ✅ Clear function naming and structure
  ✅ Proper error handling patterns
  ✅ Good separation of concerns
  ✅ Consistent code formatting

AREAS FOR IMPROVEMENT:
  ⚠️  Function complexity could be reduced
  ⚠️  Missing input validation
  ⚠️  Limited test coverage (current: 65%)
  ⚠️  Some magic numbers should be constants

SECURITY CONSIDERATIONS:
  🔐 Password handling looks secure
  🔐 SQL injection prevention in place
  ⚠️  Consider rate limiting for authentication attempts
  ⚠️  Add audit logging for security events`

	fmt.Println(review)
	fmt.Println("✅ Code review completed")
	return nil
}

func (cli *CLIApp) executeAnalyzeDependencies(parsed *ParsedPrompt) error {
	fmt.Printf("🔗 Analyzing dependencies...\n")
	fmt.Printf("🎯 Intent: %s (confidence: %.2f)\n", parsed.Intent, parsed.Confidence)

	if module, exists := parsed.Entities["module_name"]; exists {
		fmt.Printf("📦 Module: %s\n", module)
	}

	dependencies := `🔗 Dependency Analysis:

DIRECT DEPENDENCIES:
  📦 github.com/golang-jwt/jwt/v4 (authentication)
  📦 golang.org/x/crypto/bcrypt (password hashing)
  📦 github.com/lib/pq (PostgreSQL driver)
  📦 github.com/gorilla/mux (HTTP routing)

DEPENDENCY GRAPH:
  auth/login.go
  ├── database/user.go
  ├── crypto/hash.go
  └── middleware/jwt.go
      └── config/keys.go

POTENTIAL ISSUES:
  ⚠️  Circular dependency between auth and middleware packages
  ⚠️  Heavy dependency on external JWT library
  ⚠️  Database package tightly coupled to auth logic

RECOMMENDATIONS:
  1. Break circular dependencies with interfaces
  2. Consider dependency injection pattern
  3. Abstract database operations behind repository interface`

	fmt.Println(dependencies)
	fmt.Println("✅ Dependency analysis completed")
	return nil
}

func (cli *CLIApp) executeGenericQuery(parsed *ParsedPrompt) error {
	fmt.Printf("❓ Processing generic query...\n")
	fmt.Printf("🎯 Intent: %s (confidence: %.2f)\n", parsed.Intent, parsed.Confidence)

	if parsed.Context != "" {
		fmt.Printf("📝 Context: %s\n", parsed.Context)
	}

	genericResponse := `💡 I understand you're asking about: %s

Based on your query, here are some suggestions:

🔍 If you want to search for code, try: "find function functionName"
🤖 If you want to generate code, try: "create a REST API handler" 
📖 If you want explanations, try: "explain how authentication works"
🧪 If you want tests, try: "add tests for this function"

For more specific help, you can:
- Use more specific keywords in your query
- Type 'help' to see all available commands
- Check 'status' to see system information`

	fmt.Printf(genericResponse, parsed.RawInput)
	fmt.Println("\n💡 Try being more specific for better results")
	return nil
}

func (cli *CLIApp) showNextSteps(intent string) {
	nextSteps := map[string][]string{
		"search_code": {
			"Indexing project files for semantic search",
			"Connecting to vector database",
			"Analyzing code patterns and relationships",
		},
		"generate_code": {
			"Loading relevant code examples from your project",
			"Selecting optimal AI provider",
			"Generating code with project-specific context",
		},
		"explain_code": {
			"Analyzing code structure and dependencies",
			"Building explanation context",
			"Generating detailed explanation",
		},
		"generate_tests": {
			"Analyzing target code for test scenarios",
			"Identifying edge cases and test patterns",
			"Generating comprehensive test suite",
		},
	}

	if steps, exists := nextSteps[intent]; exists {
		fmt.Println("\n📋 Next steps:")
		for i, step := range steps {
			fmt.Printf("  %d. %s\n", i+1, step)
		}
	}
}

func (cli *CLIApp) showEnvironmentStatus() {
	fmt.Println("\n" + strings.Repeat("═", 60))
	fmt.Println("                ENVIRONMENT STATUS")
	fmt.Println(strings.Repeat("═", 60))

	// Environment file status
	fmt.Println("Environment Files:")
	envFiles := []string{".env.local", ".env", "~/.ai-assistant/.env"}
	for _, file := range envFiles {
		if file == "~/.ai-assistant/.env" {
			homeDir, _ := os.UserHomeDir()
			file = filepath.Join(homeDir, ".ai-assistant", ".env")
		}

		if _, err := os.Stat(file); err == nil {
			fmt.Printf("  ✅ %s\n", file)
		} else {
			fmt.Printf("  ❌ %s (not found)\n", file)
		}
	}

	// Environment variables status
	fmt.Println("\nEnvironment Variables:")
	envVars := []struct {
		key         string
		description string
		sensitive   bool
	}{
		{"LOG_LEVEL", "Logging Level", false},
		{"ENABLE_COLORS", "Color Output", false},
		{"DEBUG_MODE", "Debug Mode", false},
		{"VERBOSE", "Verbose Output", false},
		{"MAX_PARALLEL_INDEXING", "Parallel Processing", false},
		{"TOKEN_WARNING_THRESHOLD", "Token Warning", false},
		{"COST_WARNING_THRESHOLD", "Cost Warning", false},
	}

	for _, env := range envVars {
		value := os.Getenv(env.key)
		if value == "" {
			fmt.Printf("  ❌ %-25s: (not set)\n", env.key)
		} else {
			fmt.Printf("  ✅ %-25s: %s\n", env.key, value)
		}
	}

	fmt.Println(strings.Repeat("═", 60))
}

func (cli *CLIApp) showProviderStatus() {
	fmt.Println("\n" + strings.Repeat("═", 60))
	fmt.Println("                AI PROVIDER STATUS")
	fmt.Println(strings.Repeat("═", 60))

	providers := []struct {
		name     string
		keyEnv   string
		modelEnv string
		model    string
	}{
		{"OpenAI", "OPENAI_API_KEY", "OPENAI_MODEL", cli.envConfig.OpenAIModel},
		{"Google Gemini", "GEMINI_API_KEY", "GEMINI_MODEL", cli.envConfig.GeminiModel},
		{"Cohere", "COHERE_API_KEY", "COHERE_MODEL", cli.envConfig.CohereModel},
		{"Anthropic Claude", "CLAUDE_API_KEY", "CLAUDE_MODEL", cli.envConfig.ClaudeModel},
	}

	for _, provider := range providers {
		key := os.Getenv(provider.keyEnv)
		status := "❌"
		keyStatus := "(not configured)"

		if key != "" {
			status = "✅"
			keyStatus = "***configured***"
		}

		fmt.Printf("%s %-15s: %s | Model: %s\n", status, provider.name, keyStatus, provider.model)
	}

	// Show recommendations
	availableProviders := cli.getAvailableProviders()
	if len(availableProviders) == 0 {
		fmt.Println("\n⚠️  No AI providers configured!")
		fmt.Println("   Run 'ai-assistant init' to setup configuration")
		fmt.Println("   Then edit your .env file with API keys")
	} else {
		fmt.Printf("\n✅ %d provider(s) available: %s\n", len(availableProviders), strings.Join(availableProviders, ", "))
	}

	fmt.Println(strings.Repeat("═", 60))
}

func (cli *CLIApp) showMCPStatus() {
	fmt.Println("\n" + strings.Repeat("═", 60))
	fmt.Println("                  MCP SERVER STATUS")
	fmt.Println(strings.Repeat("═", 60))

	mcpServers := []struct {
		name        string
		envVar      string
		enabled     bool
		description string
	}{
		{"Filesystem", "MCP_FILESYSTEM_ENABLED", cli.envConfig.MCPFilesystemEnabled, "File operations and search"},
		{"Git", "MCP_GIT_ENABLED", cli.envConfig.MCPGitEnabled, "Git repository operations"},
		{"GitHub", "MCP_GITHUB_ENABLED", cli.envConfig.MCPGithubEnabled, "GitHub API integration"},
		{"SQLite", "MCP_SQLITE_ENABLED", cli.envConfig.MCPSQLiteEnabled, "Database operations"},
		{"Docker", "MCP_DOCKER_ENABLED", cli.envConfig.MCPDockerEnabled, "Container management"},
	}

	for _, server := range mcpServers {
		status := "❌"
		statusText := "disabled"

		if server.enabled {
			status = "✅"
			statusText = "enabled"
		}

		fmt.Printf("%s %-10s: %-8s | %s\n", status, server.name, statusText, server.description)
	}

	// Special handling for GitHub token
	if cli.envConfig.MCPGithubEnabled {
		githubToken := os.Getenv("MCP_GITHUB_TOKEN")
		if githubToken == "" {
			fmt.Println("\n⚠️  GitHub MCP server enabled but no token configured")
			fmt.Println("   Set MCP_GITHUB_TOKEN environment variable")
		}
	}

	enabledCount := len(cli.getEnabledMCPServers())
	fmt.Printf("\n📊 %d MCP server(s) enabled\n", enabledCount)

	fmt.Println(strings.Repeat("═", 60))
}

func (cli *CLIApp) reloadConfiguration() {
	fmt.Println("🔄 Reloading configuration...")

	// Reload environment variables
	if err := godotenv.Load(); err != nil {
		cli.printError(fmt.Sprintf("Failed to reload .env: %v", err))
		return
	}

	// Reload environment config
	newEnvConfig, err := loadEnvironmentConfig()
	if err != nil {
		cli.printError(fmt.Sprintf("Failed to reload environment config: %v", err))
		return
	}

	cli.envConfig = newEnvConfig

	// Update color settings
	if cli.config.NoColor || !cli.envConfig.EnableColors {
		color.NoColor = true
	} else {
		color.NoColor = false
	}

	fmt.Println("✅ Configuration reloaded successfully")

	// Show updated status
	providers := cli.getAvailableProviders()
	mcpServers := cli.getEnabledMCPServers()

	fmt.Printf("   AI Providers: %d configured\n", len(providers))
	fmt.Printf("   MCP Servers: %d enabled\n", len(mcpServers))
}

func (cli *CLIApp) printHelp() {
	fmt.Println(`
╔══════════════════════════════════════════════════════════════╗
║                         HELP MENU                            ║
╚══════════════════════════════════════════════════════════════╝

NATURAL LANGUAGE QUERIES:
  • "find function handleUser"          - Search for specific functions
  • "explain how authentication works"  - Get code explanations  
  • "create a REST API handler"         - Generate new code
  • "add tests for this function"       - Generate unit tests
  • "refactor this code"                - Improve existing code
  • "show dependencies of user.go"      - Analyze relationships

SYSTEM COMMANDS:
  help        - Show this help menu
  status      - Show system status
  config      - Show current configuration
  env         - Show environment status
  providers   - Show AI provider status
  mcp         - Show MCP server status
  reload      - Reload configuration
  clear       - Clear screen
  version     - Show version information
  quit/exit   - Exit the application

SETTINGS:
  set format <text|json|table>    - Change output format
  set verbose <true|false>        - Toggle verbose logging
  set debug <true|false>          - Toggle debug mode

CONFIGURATION:
  Run 'ai-assistant init' to setup initial configuration
  Edit .env file to configure API keys and settings
  Run 'ai-assistant env validate' to check configuration

EXAMPLES:
  useQ> find function handleRequest
  useQ> create a user authentication system
  useQ> explain the middleware pattern
  useQ> status
  useQ> providers
`)
}

func (cli *CLIApp) printStatus() {
	fmt.Println("\n" + strings.Repeat("═", 60))
	fmt.Println("                    SYSTEM STATUS")
	fmt.Println(strings.Repeat("═", 60))

	fmt.Printf("🟢 Status: Running\n")
	fmt.Printf("📁 Project: %s\n", cli.config.ProjectPath)
	fmt.Printf("⏱️  Uptime: %v\n", time.Since(cli.startTime))
	fmt.Printf("📊 Queries processed: %d\n", cli.queryCount)
	fmt.Printf("🆔 Session ID: %s\n", cli.sessionID)
	fmt.Printf("🧠 Memory usage: %s\n", formatMemoryUsage())

	// AI Providers status
	providers := cli.getAvailableProviders()
	if len(providers) > 0 {
		fmt.Printf("🤖 AI Providers: %s\n", strings.Join(providers, ", "))
	} else {
		fmt.Printf("⚠️  AI Providers: None configured\n")
	}

	// MCP Servers status
	mcpServers := cli.getEnabledMCPServers()
	if len(mcpServers) > 0 {
		fmt.Printf("🔌 MCP Servers: %s\n", strings.Join(mcpServers, ", "))
	} else {
		fmt.Printf("🔌 MCP Servers: None enabled\n")
	}

	// Configuration status
	fmt.Printf("⚙️  Log Level: %s\n", cli.envConfig.LogLevel)
	fmt.Printf("🎨 Colors: %t\n", cli.envConfig.EnableColors)
	fmt.Printf("🔍 Debug Mode: %t\n", cli.config.Debug || cli.envConfig.DebugMode)

	fmt.Println(strings.Repeat("═", 60))
}

func (cli *CLIApp) handleSetCommand(command string) {
	parts := strings.SplitN(command, " ", 2)
	if len(parts) != 2 {
		fmt.Println("❌ Usage: set <property> <value>")
		fmt.Println("Available properties: format, verbose, debug")
		return
	}

	property, value := parts[0], parts[1]

	switch property {
	case "format":
		if value == "text" || value == "json" || value == "table" {
			cli.config.Format = value
			fmt.Printf("✅ Output format set to: %s\n", value)
		} else {
			fmt.Println("❌ Invalid format. Use: text, json, or table")
		}
	case "verbose":
		if boolVal, err := strconv.ParseBool(value); err == nil {
			cli.config.Verbose = boolVal
			fmt.Printf("✅ Verbose mode: %t\n", boolVal)
		} else {
			fmt.Println("❌ Invalid value. Use: true or false")
		}
	case "debug":
		if boolVal, err := strconv.ParseBool(value); err == nil {
			cli.config.Debug = boolVal
			fmt.Printf("✅ Debug mode: %t\n", boolVal)
		} else {
			fmt.Println("❌ Invalid value. Use: true or false")
		}
	default:
		fmt.Printf("❌ Unknown property: %s\n", property)
		fmt.Println("Available properties: format, verbose, debug")
	}
}

func (cli *CLIApp) clearScreen() {
	fmt.Print("\033[H\033[2J")
}

func (cli *CLIApp) printVersion() {
	fmt.Printf("AI Code Assistant CLI v1.0.0\n")
	fmt.Printf("Go version: %s\n", runtime.Version())
	fmt.Printf("Platform: %s/%s\n", runtime.GOOS, runtime.GOARCH)
}

func (cli *CLIApp) printError(message string) {
	if cli.config.NoColor || !cli.envConfig.EnableColors {
		fmt.Printf("❌ %s\n", message)
		return
	}

	red := color.New(color.FgRed)
	red.Printf("❌ %s\n", message)
}

func (cli *CLIApp) printSessionSummary() {
	fmt.Println("\n" + strings.Repeat("═", 60))
	fmt.Println("                  SESSION SUMMARY")
	fmt.Println(strings.Repeat("═", 60))

	fmt.Printf("Session Duration: %v\n", time.Since(cli.startTime))
	fmt.Printf("Queries Processed: %d\n", cli.queryCount)
	fmt.Printf("Session ID: %s\n", cli.sessionID)
	fmt.Printf("Peak Memory Usage: %s\n", formatMemoryUsage())

	// TODO: Add more session statistics
	// - Token usage
	// - Cost information
	// - Performance metrics

	fmt.Println("\nThank you for using AI Code Assistant! 👋")
	fmt.Println(strings.Repeat("═", 60))
}

func (cli *CLIApp) shutdown() {
	cli.isRunning = false
	fmt.Println("\n\n🔄 Shutting down gracefully...")

	// TODO: Cleanup resources
	// - Close database connections
	// - Stop MCP servers
	// - Save session data

	fmt.Println("✅ Shutdown complete")
}

// Utility functions
func generateSessionID() string {
	return fmt.Sprintf("sess_%s", time.Now().Format("20060102_150405"))
}

func generateQueryID() string {
	return fmt.Sprintf("query_%d", time.Now().UnixNano())
}

func formatMemoryUsage() string {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return fmt.Sprintf("%.1f MB", float64(m.Alloc)/1024/1024)
}

// Environment helper functions
func getEnvWithDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvIntWithDefault(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getEnvFloatWithDefault(key string, defaultValue float64) float64 {
	if value := os.Getenv(key); value != "" {
		if floatValue, err := strconv.ParseFloat(value, 64); err == nil {
			return floatValue
		}
	}
	return defaultValue
}

func getEnvBoolWithDefault(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if boolValue, err := strconv.ParseBool(value); err == nil {
			return boolValue
		}
	}
	return defaultValue
}
