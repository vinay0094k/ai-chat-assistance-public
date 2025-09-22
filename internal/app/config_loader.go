package app

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/joho/godotenv"
	"gopkg.in/yaml.v3"
)

// ConfigLoader handles loading configuration from multiple sources
type ConfigLoader struct {
	configPaths []string
	envPaths    []string
	loaded      map[string]interface{}
}

// SystemConfig represents the complete system configuration
type SystemConfig struct {
	System      *SystemSettings    `yaml:"system"`
	AI          *AIConfig          `yaml:"ai"`
	Indexing    *IndexingConfig    `yaml:"indexing"`
	VectorDB    *VectorDBConfig    `yaml:"vectordb"`
	Database    *DatabaseConfig    `yaml:"database"`
	MCP         *MCPConfig         `yaml:"mcp"`
	Search      *SearchConfig      `yaml:"search"`
	Learning    *LearningConfig    `yaml:"learning"`
	Performance *PerformanceConfig `yaml:"performance"`
	Security    *SecurityConfig    `yaml:"security"`
}

// AllAgentsConfig represents configurations for all agents
type AllAgentsConfig struct {
	ArchitectureAware       *ArchitectureAwareConfig       `yaml:"architecture_aware"`
	CodeIntelligence        *CodeIntelligenceConfig        `yaml:"code_intelligence"`
	Coding                  *CodingConfig                  `yaml:"coding"`
	CodeReview              *CodeReviewConfig              `yaml:"code_review"`
	ContextAwareSearch      *ContextAwareSearchConfig      `yaml:"context_aware_search"`
	Debugging               *DebuggingConfig               `yaml:"debugging"`
	DependencyIntelligence  *DependencyIntelligenceConfig  `yaml:"dependency_intelligence"`
	Documentation           *DocumentationConfig           `yaml:"documentation"`
	PerformanceOptimization *PerformanceOptimizationConfig `yaml:"performance_optimization"`
	ProjectAnalysis         *ProjectAnalysisConfig         `yaml:"project_analysis"`
	Search                  *SearchAgentConfig             `yaml:"search"`
	Testing                 *TestingConfig                 `yaml:"testing"`
	TestIntelligence        *TestIntelligenceConfig        `yaml:"test_intelligence"`
}

// HealthCheckConfig holds configurable thresholds for health checks
type HealthCheckConfig struct {
	MaxLatencyMs            int64         `yaml:"max_latency_ms" json:"max_latency_ms"`
	MaxErrorRate            float64       `yaml:"max_error_rate" json:"max_error_rate"`
	DependencyCheckInterval time.Duration `yaml:"dependency_check_interval" json:"dependency_check_interval"`
	EnableDependencyCheck   bool          `yaml:"enable_dependency_check" json:"enable_dependency_check"`
	HealthCheckTimeout      time.Duration `yaml:"health_check_timeout" json:"health_check_timeout"`
}

// Agent-specific configuration structs
type ArchitectureAwareConfig struct {
	EnablePatternDetection       bool               `yaml:"enable_pattern_detection"`
	EnableDependencyAnalysis     bool               `yaml:"enable_dependency_analysis"`
	EnableArchitectureValidation bool               `yaml:"enable_architecture_validation"`
	MaxAnalysisDepth             int                `yaml:"max_analysis_depth"`
	LLMModel                     string             `yaml:"llm_model"`
	MaxTokens                    int                `yaml:"max_tokens"`
	Temperature                  float32            `yaml:"temperature"`
	HealthCheck                  *HealthCheckConfig `yaml:"health_check"`
}

type CodeIntelligenceConfig struct {
	EnableSemanticAnalysis  bool               `yaml:"enable_semantic_analysis"`
	EnableCodeUnderstanding bool               `yaml:"enable_code_understanding"`
	EnableContextAnalysis   bool               `yaml:"enable_context_analysis"`
	MaxContextSize          int                `yaml:"max_context_size"`
	LLMModel                string             `yaml:"llm_model"`
	MaxTokens               int                `yaml:"max_tokens"`
	Temperature             float32            `yaml:"temperature"`
	HealthCheck             *HealthCheckConfig `yaml:"health_check"`
}

type CodingConfig struct {
	EnableCodeGeneration   bool               `yaml:"enable_code_generation"`
	EnableCodeModification bool               `yaml:"enable_code_modification"`
	EnableRefactoring      bool               `yaml:"enable_refactoring"`
	EnableQualityCheck     bool               `yaml:"enable_quality_check"`
	LLMModel               string             `yaml:"llm_model"`
	MaxTokens              int                `yaml:"max_tokens"`
	Temperature            float32            `yaml:"temperature"`
	HealthCheck            *HealthCheckConfig `yaml:"health_check"`
}

type CodeReviewConfig struct {
	EnableSecurityAnalysis    bool    `yaml:"enable_security_analysis"`
	EnablePerformanceAnalysis bool    `yaml:"enable_performance_analysis"`
	EnableBestPracticesCheck  bool    `yaml:"enable_best_practices_check"`
	EnableStyleCheck          bool    `yaml:"enable_style_check"`
	MinQualityScore           float32 `yaml:"min_quality_score"`
	LLMModel                  string  `yaml:"llm_model"`
	MaxTokens                 int     `yaml:"max_tokens"`
	Temperature               float32 `yaml:"temperature"`
}

type ContextAwareSearchConfig struct {
	EnableSemanticSearch bool    `yaml:"enable_semantic_search"`
	EnableContextRanking bool    `yaml:"enable_context_ranking"`
	MaxResults           int     `yaml:"max_results"`
	SimilarityThreshold  float32 `yaml:"similarity_threshold"`
	EnableCaching        bool    `yaml:"enable_caching"`
}

type DebuggingConfig struct {
	EnableBugDetection        bool    `yaml:"enable_bug_detection"`
	EnablePerformanceAnalysis bool    `yaml:"enable_performance_analysis"`
	EnableSecurityAnalysis    bool    `yaml:"enable_security_analysis"`
	EnableFixSuggestions      bool    `yaml:"enable_fix_suggestions"`
	LLMModel                  string  `yaml:"llm_model"`
	MaxTokens                 int     `yaml:"max_tokens"`
	Temperature               float32 `yaml:"temperature"`
}

type DependencyIntelligenceConfig struct {
	EnableVulnerabilityScanning bool `yaml:"enable_vulnerability_scanning"`
	EnableUpdateSuggestions     bool `yaml:"enable_update_suggestions"`
	EnableCompatibilityCheck    bool `yaml:"enable_compatibility_check"`
	MaxDependencyDepth          int  `yaml:"max_dependency_depth"`
}

type DocumentationConfig struct {
	EnableDocGeneration     bool    `yaml:"enable_doc_generation"`
	EnableDocUpdate         bool    `yaml:"enable_doc_update"`
	EnableAPIDocGeneration  bool    `yaml:"enable_api_doc_generation"`
	EnableReadmeGeneration  bool    `yaml:"enable_readme_generation"`
	EnableExampleGeneration bool    `yaml:"enable_example_generation"`
	DefaultFormat           string  `yaml:"default_format"`
	LLMModel                string  `yaml:"llm_model"`
	MaxTokens               int     `yaml:"max_tokens"`
	Temperature             float32 `yaml:"temperature"`
}

type PerformanceOptimizationConfig struct {
	EnableBottleneckDetection     bool    `yaml:"enable_bottleneck_detection"`
	EnableOptimizationSuggestions bool    `yaml:"enable_optimization_suggestions"`
	EnableMemoryAnalysis          bool    `yaml:"enable_memory_analysis"`
	EnableCPUAnalysis             bool    `yaml:"enable_cpu_analysis"`
	LLMModel                      string  `yaml:"llm_model"`
	MaxTokens                     int     `yaml:"max_tokens"`
	Temperature                   float32 `yaml:"temperature"`
}

type ProjectAnalysisConfig struct {
	EnableStructureAnalysis  bool `yaml:"enable_structure_analysis"`
	EnableMetricsCalculation bool `yaml:"enable_metrics_calculation"`
	EnableQualityAssessment  bool `yaml:"enable_quality_assessment"`
	MaxAnalysisDepth         int  `yaml:"max_analysis_depth"`
}

type SearchAgentConfig struct {
	MaxResults          int     `yaml:"max_results"`
	EnableFuzzySearch   bool    `yaml:"enable_fuzzy_search"`
	EnableRegexSearch   bool    `yaml:"enable_regex_search"`
	SimilarityThreshold float32 `yaml:"similarity_threshold"`
}

type TestingConfig struct {
	EnableUnitTestGeneration        bool    `yaml:"enable_unit_test_generation"`
	EnableIntegrationTestGeneration bool    `yaml:"enable_integration_test_generation"`
	EnableMockGeneration            bool    `yaml:"enable_mock_generation"`
	CoverageTarget                  float32 `yaml:"coverage_target"`
	LLMModel                        string  `yaml:"llm_model"`
	MaxTokens                       int     `yaml:"max_tokens"`
	Temperature                     float32 `yaml:"temperature"`
}

type TestIntelligenceConfig struct {
	EnableTestAnalysis       bool `yaml:"enable_test_analysis"`
	EnableCoverageAnalysis   bool `yaml:"enable_coverage_analysis"`
	EnableTestOptimization   bool `yaml:"enable_test_optimization"`
	EnableFlakyTestDetection bool `yaml:"enable_flaky_test_detection"`
}

// SystemSettings represents basic system settings
type SystemSettings struct {
	Version        string `yaml:"version"`
	Name           string `yaml:"name"`
	SessionTimeout string `yaml:"session_timeout"`
	AutoSaveConfig bool   `yaml:"auto_save_config"`
	DataDirectory  string `yaml:"data_directory"`
}

// AIConfig represents AI provider configuration
type AIConfig struct {
	Providers []AIProvider    `yaml:"providers"`
	Fallback  *FallbackConfig `yaml:"fallback"`
}

// AIProvider represents a single AI provider configuration
type AIProvider struct {
	Name            string  `yaml:"name"`
	Model           string  `yaml:"model"`
	Weight          float64 `yaml:"weight"`
	Timeout         string  `yaml:"timeout"`
	MaxTokens       int     `yaml:"max_tokens"`
	Temperature     float64 `yaml:"temperature"`
	Stream          bool    `yaml:"stream"`
	Enabled         bool    `yaml:"enabled"`
	CostPer1KInput  float64 `yaml:"cost_per_1k_input"`
	CostPer1KOutput float64 `yaml:"cost_per_1k_output"`
}

// FallbackConfig represents AI provider fallback configuration
type FallbackConfig struct {
	Enabled             bool                  `yaml:"enabled"`
	MaxRetries          int                   `yaml:"max_retries"`
	RetryDelay          string                `yaml:"retry_delay"`
	HealthCheckInterval string                `yaml:"health_check_interval"`
	CircuitBreaker      *CircuitBreakerConfig `yaml:"circuit_breaker"`
}

// CircuitBreakerConfig represents circuit breaker configuration
type CircuitBreakerConfig struct {
	FailureThreshold int    `yaml:"failure_threshold"`
	RecoveryTimeout  string `yaml:"recovery_timeout"`
}

// IndexingConfig represents code indexing configuration
type IndexingConfig struct {
	Languages       []string `yaml:"languages"`
	FileExtensions  []string `yaml:"file_extensions"`
	IgnorePatterns  []string `yaml:"ignore_patterns"`
	ChunkSize       int      `yaml:"chunk_size"`
	Overlap         int      `yaml:"overlap"`
	BatchSize       int      `yaml:"batch_size"`
	MaxFileSize     string   `yaml:"max_file_size"`
	MaxFiles        int      `yaml:"max_files"`
	Incremental     bool     `yaml:"incremental"`
	RealTime        bool     `yaml:"real_time"`
	DebounceDelay   string   `yaml:"debounce_delay"`
	MaxWorkers      int      `yaml:"max_workers"`
	WorkerQueueSize int      `yaml:"worker_queue_size"`
}

// VectorDBConfig represents vector database configuration
type VectorDBConfig struct {
	Provider       string `yaml:"provider"`
	Host           string `yaml:"host"`
	Port           int    `yaml:"port"`
	CollectionName string `yaml:"collection_name"`
	VectorSize     int    `yaml:"vector_size"`
	Distance       string `yaml:"distance"`
	BatchSize      int    `yaml:"batch_size"`
	MaxConnections int    `yaml:"max_connections"`
	Timeout        string `yaml:"timeout"`
	M              int    `yaml:"m"`
	EfConstruct    int    `yaml:"ef_construct"`
	Ef             int    `yaml:"ef"`
}

// DatabaseConfig represents local database configuration
type DatabaseConfig struct {
	Provider       string `yaml:"provider"`
	Path           string `yaml:"path"`
	MaxConnections int    `yaml:"max_connections"`
	// ConnectionTimeout string `yaml:"connection_timeout"`
	ConnectionTimeout time.Duration `yaml:"connection_timeout"` // changed to time.Duration

	JournalMode string `yaml:"journal_mode"`
	Synchronous string `yaml:"synchronous"`
	CacheSize   string `yaml:"cache_size"`
	AutoBackup  bool   `yaml:"auto_backup"`
	// BackupInterval string `yaml:"backup_interval"`
	BackupInterval time.Duration `yaml:"backup_interval,omitempty"` // changed to time.Duration

	MaxBackups int `yaml:"max_backups"`
}

// MCPConfig represents MCP configuration
type MCPConfig struct {
	Enabled    bool              `yaml:"enabled"`
	Timeout    string            `yaml:"timeout"`
	MaxRetries int               `yaml:"max_retries"`
	Servers    []MCPServerConfig `yaml:"servers"`
}

// MCPServerConfig represents MCP server configuration
type MCPServerConfig struct {
	Name        string            `yaml:"name"`
	Description string            `yaml:"description"`
	Command     []string          `yaml:"command"`
	Args        []string          `yaml:"args"`
	Transport   string            `yaml:"transport"`
	AutoInstall bool              `yaml:"auto_install"`
	Enabled     bool              `yaml:"enabled"`
	Env         map[string]string `yaml:"env"`
}

// SearchConfig represents search configuration
type SearchConfig struct {
	MaxResults       int     `yaml:"max_results"`
	MinConfidence    float64 `yaml:"min_confidence"`
	SemanticWeight   float64 `yaml:"semantic_weight"`
	KeywordWeight    float64 `yaml:"keyword_weight"`
	GraphWeight      float64 `yaml:"graph_weight"`
	ContextExpansion int     `yaml:"context_expansion"`
	EnableFuzzy      bool    `yaml:"enable_fuzzy"`
	FuzzyThreshold   float64 `yaml:"fuzzy_threshold"`
}

// LearningConfig represents learning system configuration
type LearningConfig struct {
	Enabled                   bool    `yaml:"enabled"`
	MinPatternFrequency       int     `yaml:"min_pattern_frequency"`
	FeedbackWeight            float64 `yaml:"feedback_weight"`
	AccuracyTrackingWindow    string  `yaml:"accuracy_tracking_window"`
	AutoApplyPatterns         bool    `yaml:"auto_apply_patterns"`
	ConfidenceThreshold       float64 `yaml:"confidence_threshold"`
	MaxPatterns               int     `yaml:"max_patterns"`
	ProviderSelectionLearning bool    `yaml:"provider_selection_learning"`
	ContextOptimization       bool    `yaml:"context_optimization"`
}

// PerformanceConfig represents performance configuration
type PerformanceConfig struct {
	EnableMetrics   bool   `yaml:"enable_metrics"`
	MetricsInterval string `yaml:"metrics_interval"`
	MaxMemory       string `yaml:"max_memory"`
	GCThreshold     string `yaml:"gc_threshold"`
	EnableCache     bool   `yaml:"enable_cache"`
	CacheSize       string `yaml:"cache_size"`
	CacheTTL        string `yaml:"cache_ttl"`
}

// SecurityConfig represents security configuration
type SecurityConfig struct {
	EnableAPIKeyValidation bool `yaml:"enable_api_key_validation"`
	LogSensitiveData       bool `yaml:"log_sensitive_data"`
	EncryptStoredData      bool `yaml:"encrypt_stored_data"`
	EnableRateLimiting     bool `yaml:"enable_rate_limiting"`
	RequestsPerMinute      int  `yaml:"requests_per_minute"`
	BurstSize              int  `yaml:"burst_size"`
}

// CLISpecificConfig represents CLI-specific configuration
type CLISpecificConfig struct {
	CLI         *CLIDisplayConfig `yaml:"cli"`
	Commands    *CommandConfig    `yaml:"commands"`
	Shortcuts   *ShortcutConfig   `yaml:"shortcuts"`
	Output      *OutputConfig     `yaml:"output"`
	Development *DevConfig        `yaml:"development"`
}

// CLIDisplayConfig represents CLI display configuration
type CLIDisplayConfig struct {
	Prompt             string                    `yaml:"prompt"`
	PromptColor        string                    `yaml:"prompt_color"`
	ShowLineNumbers    bool                      `yaml:"show_line_numbers"`
	LineNumberWidth    int                       `yaml:"line_number_width"`
	MaxOutputLines     int                       `yaml:"max_output_lines"`
	WrapLines          bool                      `yaml:"wrap_lines"`
	Colors             *CLIColorConfig           `yaml:"colors"`
	SyntaxHighlighting *SyntaxHighlightingConfig `yaml:"syntax_highlighting"`
	Progress           *CLIProgressConfig        `yaml:"progress"`
	Completion         *CompletionConfig         `yaml:"completion"`
	History            *HistoryConfig            `yaml:"history"`
	Session            *SessionConfig            `yaml:"session"`
}

// CLIColorConfig represents CLI color configuration
type CLIColorConfig struct {
	Enabled   bool   `yaml:"enabled"`
	Theme     string `yaml:"theme"`
	Primary   string `yaml:"primary"`
	Secondary string `yaml:"secondary"`
	Warning   string `yaml:"warning"`
	Error     string `yaml:"error"`
	Success   string `yaml:"success"`
	Info      string `yaml:"info"`
	Debug     string `yaml:"debug"`
}

// SyntaxHighlightingConfig represents syntax highlighting configuration
type SyntaxHighlightingConfig struct {
	Enabled   bool                          `yaml:"enabled"`
	Theme     string                        `yaml:"theme"`
	Languages map[string]*LanguageHighlight `yaml:"languages"`
}

// LanguageHighlight represents language-specific highlighting
type LanguageHighlight struct {
	Keywords []string          `yaml:"keywords"`
	Types    []string          `yaml:"types"`
	Colors   map[string]string `yaml:"colors"`
}

// CLIProgressConfig represents CLI progress configuration
type CLIProgressConfig struct {
	ShowProgress     bool   `yaml:"show_progress"`
	ProgressBarWidth int    `yaml:"progress_bar_width"`
	ShowPercentage   bool   `yaml:"show_percentage"`
	ShowETA          bool   `yaml:"show_eta"`
	AnimationStyle   string `yaml:"animation_style"`
}

// CompletionConfig represents auto-completion configuration
type CompletionConfig struct {
	Enabled        bool `yaml:"enabled"`
	MaxSuggestions int  `yaml:"max_suggestions"`
	FuzzyMatching  bool `yaml:"fuzzy_matching"`
}

// HistoryConfig represents command history configuration
type HistoryConfig struct {
	Enabled    bool   `yaml:"enabled"`
	MaxEntries int    `yaml:"max_entries"`
	Persist    bool   `yaml:"persist"`
	File       string `yaml:"file"`
}

// SessionConfig represents session configuration
type SessionConfig struct {
	AutoSave     bool   `yaml:"auto_save"`
	SaveInterval string `yaml:"save_interval"`
	MaxSessions  int    `yaml:"max_sessions"`
}

// CommandConfig represents command configuration
type CommandConfig struct {
	Defaults      *CommandDefaults  `yaml:"defaults"`
	Aliases       map[string]string `yaml:"aliases"`
	QuickCommands []QuickCommand    `yaml:"quick_commands"`
}

// CommandDefaults represents default command settings
type CommandDefaults struct {
	Timeout string `yaml:"timeout"`
	Verbose bool   `yaml:"verbose"`
	Format  string `yaml:"format"`
}

// QuickCommand represents a quick command configuration
type QuickCommand struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description"`
	Pattern     string `yaml:"pattern"`
	Action      string `yaml:"action"`
}

// ShortcutConfig represents keyboard shortcut configuration
type ShortcutConfig struct {
	Enabled  bool              `yaml:"enabled"`
	Bindings map[string]string `yaml:"bindings"`
}

// OutputConfig represents output configuration
type OutputConfig struct {
	Formats    map[string]*FormatConfig `yaml:"formats"`
	Pagination *PaginationConfig        `yaml:"pagination"`
}

// FormatConfig represents output format configuration
type FormatConfig struct {
	Enabled    bool `yaml:"enabled"`
	Default    bool `yaml:"default"`
	Pretty     bool `yaml:"pretty,omitempty"`
	Indent     int  `yaml:"indent,omitempty"`
	Borders    bool `yaml:"borders,omitempty"`
	Headers    bool `yaml:"headers,omitempty"`
	CodeBlocks bool `yaml:"code_blocks,omitempty"`
}

// PaginationConfig represents pagination configuration
type PaginationConfig struct {
	Enabled         bool `yaml:"enabled"`
	PageSize        int  `yaml:"page_size"`
	ShowPageNumbers bool `yaml:"show_page_numbers"`
}

// DevConfig represents development configuration
type DevConfig struct {
	DebugMode          bool `yaml:"debug_mode"`
	VerboseLogging     bool `yaml:"verbose_logging"`
	ProfilePerformance bool `yaml:"profile_performance"`
	EnableExperimental bool `yaml:"enable_experimental"`
}

// NewConfigLoader creates a new configuration loader
func NewConfigLoader() *ConfigLoader {
	homeDir, _ := os.UserHomeDir()

	return &ConfigLoader{
		configPaths: []string{
			"./configs/properties.yaml",
			"./configs/cli-config.yaml",
			"./configs/display-config.yaml",
			"./configs/mcp-servers.yaml",
			filepath.Join(homeDir, ".ai-assistant", "config.yaml"),
			"/etc/ai-assistant/config.yaml",
		},
		envPaths: []string{
			".env.local",
			".env",
			filepath.Join(homeDir, ".ai-assistant", ".env"),
			"/etc/ai-assistant/.env",
		},
		loaded: make(map[string]interface{}),
	}
}

// LoadAllConfigurations loads all configuration files
func (cl *ConfigLoader) LoadAllConfigurations() (*SystemConfig, *CLISpecificConfig, *AllAgentsConfig, error) {
	// Load environment variables first
	if err := cl.loadEnvironmentVariables(); err != nil {
		return nil, nil, nil, fmt.Errorf("failed to load environment variables: %v", err)
	}

	// Load main system configuration
	systemConfig, err := cl.loadSystemConfig()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to load system config: %v", err)
	}

	// Load CLI-specific configuration
	cliConfig, err := cl.loadCLIConfig()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to load CLI config: %v", err)
	}

	// Load agent configurations
	agentsConfig, err := cl.loadAgentsConfig()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to load agents config: %v", err)
	}

	// Apply environment variable overrides
	cl.applyEnvironmentOverrides(systemConfig)
	cl.applyAgentEnvironmentOverrides(agentsConfig)

	return systemConfig, cliConfig, agentsConfig, nil
}

// loadEnvironmentVariables loads environment variables from .env files
func (cl *ConfigLoader) loadEnvironmentVariables() error {
	var lastError error
	loadedFiles := []string{}

	for _, envPath := range cl.envPaths {
		if _, err := os.Stat(envPath); err == nil {
			if err := godotenv.Load(envPath); err != nil {
				lastError = fmt.Errorf("failed to load %s: %v", envPath, err)
			} else {
				loadedFiles = append(loadedFiles, envPath)
			}
		}
	}

	if len(loadedFiles) > 0 && os.Getenv("VERBOSE") == "true" {
		fmt.Printf("Loaded environment files: %v\n", loadedFiles)
	}

	return lastError
}

// loadSystemConfig loads the main system configuration
func (cl *ConfigLoader) loadSystemConfig() (*SystemConfig, error) {
	configPath := cl.findConfigFile("properties.yaml")
	if configPath == "" {
		return cl.createDefaultSystemConfig(), nil
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file %s: %v", configPath, err)
	}

	var config SystemConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config file %s: %v", configPath, err)
	}

	// Validate configuration
	if err := cl.validateSystemConfig(&config); err != nil {
		return nil, fmt.Errorf("invalid configuration: %v", err)
	}

	cl.loaded["system"] = &config
	return &config, nil
}

// loadAgentsConfig loads agent configurations
func (cl *ConfigLoader) loadAgentsConfig() (*AllAgentsConfig, error) {
	configPath := cl.findConfigFile("agents-config.yaml")
	if configPath == "" {
		return cl.createDefaultAgentsConfig(), nil
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read agents config file %s: %v", configPath, err)
	}

	var config AllAgentsConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse agents config file %s: %v", configPath, err)
	}

	cl.loaded["agents"] = &config
	return &config, nil
}

// loadCLIConfig loads CLI-specific configuration
func (cl *ConfigLoader) loadCLIConfig() (*CLISpecificConfig, error) {
	configPath := cl.findConfigFile("cli-config.yaml")
	if configPath == "" {
		return cl.createDefaultCLIConfig(), nil
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read CLI config file %s: %v", configPath, err)
	}

	var config CLISpecificConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse CLI config file %s: %v", configPath, err)
	}

	cl.loaded["cli"] = &config
	return &config, nil
}

// findConfigFile finds the first existing config file
// func (cl *ConfigLoader) findConfigFile(filename string) string {
// 	// Try specific filename first
// 	if _, err := os.Stat(filepath.Join("configs", filename)); err == nil {
// 		return filepath.Join("configs", filename)
// 	}

// 	// Try general config paths
// 	for _, path := range cl.configPaths {
// 		if strings.Contains(path, filename) {
// 			if _, err := os.Stat(path); err == nil {
// 				return path
// 			}
// 		}
// 	}

// 	return ""
// }

func (cl *ConfigLoader) findConfigFile(filename string) string {
	// If caller or tests set explicit configPaths (full paths), prefer them first.
	for _, p := range cl.configPaths {
		// If the path is exactly the filename the caller passed (test passes full path),
		// or it's a path that exists on disk, return it.
		if filepath.Base(p) == filename {
			if _, err := os.Stat(p); err == nil {
				return p
			}
		}
		// If path exists and is a file, return it
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}

	// Try configs/<filename> first (local packaged configs)
	localPath := filepath.Join("configs", filename)
	if _, err := os.Stat(localPath); err == nil {
		return localPath
	}

	// Fallback: iterate configured default places and return first matching path
	for _, path := range cl.configPaths {
		// If the path contains the filename (old behavior), use it
		if strings.Contains(path, filename) {
			if _, err := os.Stat(path); err == nil {
				return path
			}
		}
	}

	// Nothing found
	return ""
}

// createDefaultSystemConfig creates default system configuration
func (cl *ConfigLoader) createDefaultSystemConfig() *SystemConfig {
	return &SystemConfig{
		System: &SystemSettings{
			Version:        "1.0.0",
			Name:           "AI Code Assistant",
			SessionTimeout: "24h",
			AutoSaveConfig: true,
			DataDirectory:  "./data",
		},
		AI: &AIConfig{
			Providers: []AIProvider{
				{
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
				},
			},
			Fallback: &FallbackConfig{
				Enabled:    true,
				MaxRetries: 3,
				RetryDelay: "2s",
			},
		},
		Indexing: &IndexingConfig{
			Languages:      []string{"go"},
			FileExtensions: []string{".go", ".mod", ".sum"},
			IgnorePatterns: []string{"vendor/", ".git/", "*.test"},
			ChunkSize:      1000,
			Overlap:        100,
			BatchSize:      50,
			Incremental:    true,
			RealTime:       true,
			MaxWorkers:     8,
		},
		Search: &SearchConfig{
			MaxResults:     20,
			MinConfidence:  0.5,
			SemanticWeight: 0.6,
			KeywordWeight:  0.2,
			GraphWeight:    0.2,
		},
	}
}

// createDefaultCLIConfig creates default CLI configuration
func (cl *ConfigLoader) createDefaultCLIConfig() *CLISpecificConfig {
	return &CLISpecificConfig{
		CLI: &CLIDisplayConfig{
			Prompt:          "useQ>",
			PromptColor:     "cyan",
			ShowLineNumbers: true,
			LineNumberWidth: 4,
			MaxOutputLines:  50,
			WrapLines:       true,
			Colors: &CLIColorConfig{
				Enabled:   true,
				Theme:     "default",
				Primary:   "cyan",
				Secondary: "green",
				Warning:   "yellow",
				Error:     "red",
				Success:   "green",
				Info:      "blue",
				Debug:     "magenta",
			},
		},
		Commands: &CommandConfig{
			Defaults: &CommandDefaults{
				Timeout: "30s",
				Verbose: false,
				Format:  "text",
			},
			Aliases: map[string]string{
				"?": "help",
				"q": "quit",
				"s": "status",
				"c": "config",
			},
		},
	}
}

// validateSystemConfig validates the system configuration
func (cl *ConfigLoader) validateSystemConfig(config *SystemConfig) error {
	if config.System == nil {
		return fmt.Errorf("system configuration is required")
	}

	if config.AI == nil || len(config.AI.Providers) == 0 {
		return fmt.Errorf("at least one AI provider must be configured")
	}

	// Validate AI providers
	for _, provider := range config.AI.Providers {
		if provider.Name == "" {
			return fmt.Errorf("AI provider name is required")
		}
		if provider.Model == "" {
			return fmt.Errorf("AI provider model is required")
		}
		if provider.Weight < 0 || provider.Weight > 1 {
			return fmt.Errorf("AI provider weight must be between 0 and 1")
		}
	}

	return nil
}

// applyEnvironmentOverrides applies environment variable overrides
func (cl *ConfigLoader) applyEnvironmentOverrides(config *SystemConfig) {
	// Override AI provider settings from environment
	if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey != "" {
		for i := range config.AI.Providers {
			if config.AI.Providers[i].Name == "openai" {
				config.AI.Providers[i].Enabled = true
				break
			}
		}
	}

	// Override system settings
	if dataDir := os.Getenv("DATA_DIRECTORY"); dataDir != "" {
		config.System.DataDirectory = dataDir
	}

	if logLevel := os.Getenv("LOG_LEVEL"); logLevel != "" {
		// Apply log level override
	}

	// Override database settings
	if dbPath := os.Getenv("SQLITE_PATH"); dbPath != "" && config.Database != nil {
		config.Database.Path = dbPath
	}

	// Override vector database settings
	if qdrantHost := os.Getenv("QDRANT_HOST"); qdrantHost != "" && config.VectorDB != nil {
		config.VectorDB.Host = qdrantHost
	}
}

// GetLoadedConfig returns a loaded configuration by name
func (cl *ConfigLoader) GetLoadedConfig(name string) (interface{}, bool) {
	config, exists := cl.loaded[name]
	return config, exists
}

// SaveConfig saves configuration to file
func (cl *ConfigLoader) SaveConfig(config interface{}, filename string) error {
	data, err := yaml.Marshal(config)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %v", err)
	}

	configPath := filepath.Join("configs", filename)
	if err := os.WriteFile(configPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %v", err)
	}

	return nil
}

// ReloadConfig reloads configuration from files
func (cl *ConfigLoader) ReloadConfig() (*SystemConfig, *CLISpecificConfig, *AllAgentsConfig, error) {
	// Clear loaded configs
	cl.loaded = make(map[string]interface{})

	// Reload all configurations
	return cl.LoadAllConfigurations()
}

func (cl *ConfigLoader) SetConfigPaths(paths []string) {
	if paths == nil {
		cl.configPaths = nil
		return
	}
	// Make a copy to avoid caller mutating our slice
	cp := make([]string, len(paths))
	copy(cp, paths)
	cl.configPaths = cp
}

func (cl *ConfigLoader) SetEnvPaths(paths []string) {
	if paths == nil {
		cl.envPaths = nil
		return
	}
	cp := make([]string, len(paths))
	copy(cp, paths)
	cl.envPaths = cp
}

func (cl *ConfigLoader) CreateDefaultSystemConfig() *SystemConfig {
	// Reuse your existing createDefaultSystemConfig helper
	return cl.createDefaultSystemConfig()
}

func (cl *ConfigLoader) createDefaultAgentsConfig() *AllAgentsConfig {
	return &AllAgentsConfig{
		ArchitectureAware: &ArchitectureAwareConfig{
			EnablePatternDetection:       true,
			EnableDependencyAnalysis:     true,
			EnableArchitectureValidation: true,
			MaxAnalysisDepth:             5,
			LLMModel:                     "gpt-4",
			MaxTokens:                    2048,
			Temperature:                  0.3,
		},
		CodeIntelligence: &CodeIntelligenceConfig{
			EnableSemanticAnalysis:  true,
			EnableCodeUnderstanding: true,
			EnableContextAnalysis:   true,
			MaxContextSize:          4096,
			LLMModel:                "gpt-4",
			MaxTokens:               2048,
			Temperature:             0.2,
		},
		Coding: &CodingConfig{
			EnableCodeGeneration:   true,
			EnableCodeModification: true,
			EnableRefactoring:      true,
			EnableQualityCheck:     true,
			LLMModel:               "gpt-4",
			MaxTokens:              4096,
			Temperature:            0.3,
		},
		CodeReview: &CodeReviewConfig{
			EnableSecurityAnalysis:    true,
			EnablePerformanceAnalysis: true,
			EnableBestPracticesCheck:  true,
			EnableStyleCheck:          true,
			MinQualityScore:           0.7,
			LLMModel:                  "gpt-4",
			MaxTokens:                 2048,
			Temperature:               0.2,
		},
		ContextAwareSearch: &ContextAwareSearchConfig{
			EnableSemanticSearch: true,
			EnableContextRanking: true,
			MaxResults:           50,
			SimilarityThreshold:  0.7,
			EnableCaching:        true,
		},
		Debugging: &DebuggingConfig{
			EnableBugDetection:        true,
			EnablePerformanceAnalysis: true,
			EnableSecurityAnalysis:    true,
			EnableFixSuggestions:      true,
			LLMModel:                  "gpt-4",
			MaxTokens:                 2048,
			Temperature:               0.2,
		},
		DependencyIntelligence: &DependencyIntelligenceConfig{
			EnableVulnerabilityScanning: true,
			EnableUpdateSuggestions:     true,
			EnableCompatibilityCheck:    true,
			MaxDependencyDepth:          10,
		},
		Documentation: &DocumentationConfig{
			EnableDocGeneration:     true,
			EnableDocUpdate:         true,
			EnableAPIDocGeneration:  true,
			EnableReadmeGeneration:  true,
			EnableExampleGeneration: true,
			DefaultFormat:           "markdown",
			LLMModel:                "gpt-4",
			MaxTokens:               2048,
			Temperature:             0.3,
		},
		PerformanceOptimization: &PerformanceOptimizationConfig{
			EnableBottleneckDetection:     true,
			EnableOptimizationSuggestions: true,
			EnableMemoryAnalysis:          true,
			EnableCPUAnalysis:             true,
			LLMModel:                      "gpt-4",
			MaxTokens:                     2048,
			Temperature:                   0.2,
		},
		ProjectAnalysis: &ProjectAnalysisConfig{
			EnableStructureAnalysis:  true,
			EnableMetricsCalculation: true,
			EnableQualityAssessment:  true,
			MaxAnalysisDepth:         10,
		},
		Search: &SearchAgentConfig{
			MaxResults:          20,
			EnableFuzzySearch:   true,
			EnableRegexSearch:   true,
			SimilarityThreshold: 0.6,
		},
		Testing: &TestingConfig{
			EnableUnitTestGeneration:        true,
			EnableIntegrationTestGeneration: true,
			EnableMockGeneration:            true,
			CoverageTarget:                  80.0,
			LLMModel:                        "gpt-4",
			MaxTokens:                       2048,
			Temperature:                     0.3,
		},
		TestIntelligence: &TestIntelligenceConfig{
			EnableTestAnalysis:       true,
			EnableCoverageAnalysis:   true,
			EnableTestOptimization:   true,
			EnableFlakyTestDetection: true,
		},
	}
}

// applyAgentEnvironmentOverrides applies environment variable overrides to agent configurations
func (cl *ConfigLoader) applyAgentEnvironmentOverrides(config *AllAgentsConfig) {
	// Example overrides for common agent settings
	if model := os.Getenv("AI_AGENT_LLM_MODEL"); model != "" {
		if config.Coding != nil {
			config.Coding.LLMModel = model
		}
		if config.Documentation != nil {
			config.Documentation.LLMModel = model
		}
		// Apply to other agents that use LLM
	}

	if maxTokens := os.Getenv("AI_AGENT_MAX_TOKENS"); maxTokens != "" {
		if tokens, err := strconv.Atoi(maxTokens); err == nil {
			if config.Coding != nil {
				config.Coding.MaxTokens = tokens
			}
			if config.Documentation != nil {
				config.Documentation.MaxTokens = tokens
			}
			// Apply to other agents that use tokens
		}
	}

	if temp := os.Getenv("AI_AGENT_TEMPERATURE"); temp != "" {
		if temperature, err := strconv.ParseFloat(temp, 32); err == nil {
			if config.Coding != nil {
				config.Coding.Temperature = float32(temperature)
			}
			if config.Documentation != nil {
				config.Documentation.Temperature = float32(temperature)
			}
			// Apply to other agents that use temperature
		}
	}
}
