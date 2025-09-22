package models

import (
	"time"
)

// Query represents a user query
type Query struct {
	ID          string                 `json:"id" db:"id"`
	SessionID   string                 `json:"session_id" db:"session_id"`
	RawInput    string                 `json:"raw_input" db:"raw_input"`
	ProcessedAt time.Time              `json:"processed_at" db:"processed_at"`
	Duration    time.Duration          `json:"duration" db:"duration"`
	Status      string                 `json:"status" db:"status"` // pending, processing, completed, failed
	Metadata    map[string]interface{} `json:"metadata" db:"metadata"`
	CreatedAt   time.Time              `json:"created_at" db:"created_at"`
}

// QueryIntent represents the parsed intent of a query
type QueryIntent struct {
	ID         string                 `json:"id" db:"id"`
	QueryID    string                 `json:"query_id" db:"query_id"`
	Type       string                 `json:"type" db:"type"`             // search_code, generate_code, explain_code, etc.
	Confidence float64                `json:"confidence" db:"confidence"` // 0.0 to 1.0
	Entities   map[string]string      `json:"entities" db:"entities"`     // Extracted entities
	Parameters map[string]interface{} `json:"parameters" db:"parameters"` // Intent parameters
	Context    string                 `json:"context" db:"context"`       // Additional context
	AgentType  string                 `json:"agent_type" db:"agent_type"` // Which agent should handle this
	Priority   int                    `json:"priority" db:"priority"`     // Processing priority
	Metadata   map[string]interface{} `json:"metadata" db:"metadata"`
	CreatedAt  time.Time              `json:"created_at" db:"created_at"`
}

// QueryContext represents the context in which a query is made
type QueryContext struct {
	SessionID        string                 `json:"session_id"`
	UserID           string                 `json:"user_id,omitempty"`
	ProjectPath      string                 `json:"project_path"`
	CurrentFile      string                 `json:"current_file,omitempty"`
	SelectedCode     string                 `json:"selected_code,omitempty"`
	CursorPosition   *CursorPosition        `json:"cursor_position,omitempty"`
	RelevantFiles    []string               `json:"relevant_files,omitempty"`
	RecentHistory    []string               `json:"recent_history,omitempty"`
	WorkingDirectory string                 `json:"working_directory"`
	GitBranch        string                 `json:"git_branch,omitempty"`
	GitCommit        string                 `json:"git_commit,omitempty"`
	Environment      string                 `json:"environment"`           // development, testing, production
	MCPResults       map[string]interface{} `json:"mcp_results,omitempty"` // Results from MCP tools
	Preferences      *UserPreferences       `json:"preferences,omitempty"`
	Timestamp        time.Time              `json:"timestamp"`
}

// CursorPosition represents cursor position in a file
type CursorPosition struct {
	Line   int `json:"line"`
	Column int `json:"column"`
}

// UserPreferences represents user-specific preferences
type UserPreferences struct {
	PreferredProvider string                 `json:"preferred_provider,omitempty"`
	PreferredLanguage string                 `json:"preferred_language,omitempty"`
	OutputFormat      string                 `json:"output_format"` // text, json, markdown
	VerboseMode       bool                   `json:"verbose_mode"`
	ShowTokens        bool                   `json:"show_tokens"`
	ShowCosts         bool                   `json:"show_costs"`
	CodeStyle         string                 `json:"code_style,omitempty"` // convention preference
	TestingFramework  string                 `json:"testing_framework,omitempty"`
	CustomSettings    map[string]interface{} `json:"custom_settings,omitempty"`
}

// SearchQuery represents a search request
type SearchQuery struct {
	ID            string         `json:"id"`
	Query         string         `json:"query"`
	Type          string         `json:"type"` // semantic, keyword, hybrid
	Filters       *SearchFilters `json:"filters,omitempty"`
	MaxResults    int            `json:"max_results"`
	MinConfidence float64        `json:"min_confidence"`
	Context       *QueryContext  `json:"context,omitempty"`
	SearchWeights *SearchWeights `json:"search_weights,omitempty"`
	ExpandContext bool           `json:"expand_context"`
	IncludeUsage  bool           `json:"include_usage"`
}

// SearchFilters represents search filtering options
type SearchFilters struct {
	Languages    []string   `json:"languages,omitempty"`     // Filter by programming language
	FileTypes    []string   `json:"file_types,omitempty"`    // Filter by file extension
	ChunkTypes   []string   `json:"chunk_types,omitempty"`   // function, struct, interface, etc.
	DateRange    *DateRange `json:"date_range,omitempty"`    // Filter by modification date
	SizeRange    *SizeRange `json:"size_range,omitempty"`    // Filter by code size
	Directories  []string   `json:"directories,omitempty"`   // Filter by directory
	ExcludeFiles []string   `json:"exclude_files,omitempty"` // Files to exclude
	OnlyTests    bool       `json:"only_tests"`              // Only test files
	ExcludeTests bool       `json:"exclude_tests"`           // Exclude test files
}

// DateRange represents a date range filter
type DateRange struct {
	From *time.Time `json:"from,omitempty"`
	To   *time.Time `json:"to,omitempty"`
}

// SizeRange represents a size range filter
type SizeRange struct {
	MinLines *int `json:"min_lines,omitempty"`
	MaxLines *int `json:"max_lines,omitempty"`
	MinBytes *int `json:"min_bytes,omitempty"`
	MaxBytes *int `json:"max_bytes,omitempty"`
}

// SearchWeights represents weights for different search types
type SearchWeights struct {
	Semantic float64 `json:"semantic"` // Weight for semantic similarity
	Keyword  float64 `json:"keyword"`  // Weight for keyword matching
	Graph    float64 `json:"graph"`    // Weight for graph relationships
	Usage    float64 `json:"usage"`    // Weight for usage frequency
	Recency  float64 `json:"recency"`  // Weight for recent modifications
}

// GenerationRequest represents a code generation request
type GenerationRequest struct {
	ID           string                 `json:"id"`
	Type         string                 `json:"type"`                // function, class, test, documentation
	Description  string                 `json:"description"`         // What to generate
	Language     string                 `json:"language"`            // Target language
	Framework    string                 `json:"framework,omitempty"` // Target framework
	Style        string                 `json:"style,omitempty"`     // Coding style
	Context      *QueryContext          `json:"context,omitempty"`
	Constraints  *GenerationConstraints `json:"constraints,omitempty"`
	Examples     []string               `json:"examples,omitempty"` // Example code to follow
	Temperature  float64                `json:"temperature"`        // AI creativity level
	MaxTokens    int                    `json:"max_tokens"`
	IncludeTests bool                   `json:"include_tests"`
	IncludeDocs  bool                   `json:"include_docs"`
}

// GenerationConstraints represents constraints for code generation
type GenerationConstraints struct {
	MaxLines         int      `json:"max_lines,omitempty"`
	RequiredImports  []string `json:"required_imports,omitempty"`
	ForbiddenWords   []string `json:"forbidden_words,omitempty"`
	MustInclude      []string `json:"must_include,omitempty"`
	Architecture     string   `json:"architecture,omitempty"` // Must follow specific architecture
	NamingConvention string   `json:"naming_convention,omitempty"`
	ErrorHandling    string   `json:"error_handling,omitempty"` // Required error handling pattern
}

// ExplanationRequest represents a code explanation request
type ExplanationRequest struct {
	ID           string        `json:"id"`
	Code         string        `json:"code"` // Code to explain
	Language     string        `json:"language"`
	Type         string        `json:"type"`            // overview, detailed, step-by-step
	Focus        []string      `json:"focus,omitempty"` // What to focus on
	Level        string        `json:"level"`           // beginner, intermediate, advanced
	Context      *QueryContext `json:"context,omitempty"`
	IncludeFlow  bool          `json:"include_flow"`  // Include execution flow
	IncludeUsage bool          `json:"include_usage"` // Include usage examples
}

// RefactoringRequest represents a code refactoring request
type RefactoringRequest struct {
	ID               string                  `json:"id"`
	Code             string                  `json:"code"` // Code to refactor
	Language         string                  `json:"language"`
	Type             string                  `json:"type"`  // extract, rename, optimize, simplify
	Goals            []string                `json:"goals"` // What to achieve
	Constraints      *RefactoringConstraints `json:"constraints,omitempty"`
	Context          *QueryContext           `json:"context,omitempty"`
	PreserveBehavior bool                    `json:"preserve_behavior"` // Must preserve existing behavior
}

// RefactoringConstraints represents constraints for refactoring
type RefactoringConstraints struct {
	PreserveInterface bool     `json:"preserve_interface"`    // Keep same public interface
	MaxChanges        int      `json:"max_changes,omitempty"` // Limit number of changes
	ForbiddenChanges  []string `json:"forbidden_changes,omitempty"`
	RequiredPatterns  []string `json:"required_patterns,omitempty"`
}

// AnalysisRequest represents a code analysis request
type AnalysisRequest struct {
	ID      string        `json:"id"`
	Target  string        `json:"target"` // file, function, project
	Type    string        `json:"type"`   // quality, performance, security, architecture
	Depth   string        `json:"depth"`  // surface, detailed, comprehensive
	Context *QueryContext `json:"context,omitempty"`
	Metrics []string      `json:"metrics,omitempty"` // Specific metrics to analyze
	Compare bool          `json:"compare"`           // Compare with project patterns
}
