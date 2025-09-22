package models

import (
	"time"
)

// Response represents a generic response from the system
type Response struct {
	ID          string                 `json:"id"`
	QueryID     string                 `json:"query_id"`
	Type        string                 `json:"type"`   // search, generation, explanation, etc.
	Status      string                 `json:"status"` // success, error, partial
	Message     string                 `json:"message,omitempty"`
	Data        interface{}            `json:"data,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	TokenUsage  *TokenUsageInfo        `json:"token_usage,omitempty"`
	Performance *PerformanceInfo       `json:"performance,omitempty"`
	Provider    string                 `json:"provider,omitempty"` // Which AI provider was used
	Model       string                 `json:"model,omitempty"`    // Which model was used
	Duration    time.Duration          `json:"duration"`
	Timestamp   time.Time              `json:"timestamp"`
	Error       *ErrorInfo             `json:"error,omitempty"`
}

// AgentResponse represents a response from a specific agent
type AgentResponse struct {
	AgentName    string                 `json:"agent_name"`
	AgentType    string                 `json:"agent_type"`
	Success      bool                   `json:"success"`
	Result       interface{}            `json:"result"`
	Message      string                 `json:"message"`
	Confidence   float64                `json:"confidence"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
	TokenUsage   *TokenUsageInfo        `json:"token_usage,omitempty"`
	Duration     time.Duration          `json:"duration"`
	NextSteps    []string               `json:"next_steps,omitempty"`
	Suggestions  []string               `json:"suggestions,omitempty"`
	SuggestionID string                 `json:"suggestion_id,omitempty"` // For feedback tracking
}

// SearchResponse represents search results
type SearchResponse struct {
	Query       string         `json:"query"`
	Results     []SearchResult `json:"results"`
	TotalFound  int            `json:"total_found"`
	SearchTime  time.Duration  `json:"search_time"`
	SearchType  string         `json:"search_type"`           // semantic, keyword, hybrid
	Confidence  float64        `json:"confidence"`            // Overall confidence
	SearchPath  []string       `json:"search_path"`           // How the search was executed
	Related     []SearchResult `json:"related,omitempty"`     // Related results
	Suggestions []string       `json:"suggestions,omitempty"` // Query suggestions
	Filters     *SearchFilters `json:"filters,omitempty"`     // Applied filters
}

// CodeGenerationResponse represents code generation results
type CodeGenerationResponse struct {
	Code             string                  `json:"code"`
	Language         string                  `json:"language"`
	Type             string                  `json:"type"` // function, class, test, etc.
	FileName         string                  `json:"file_name,omitempty"`
	Explanation      string                  `json:"explanation,omitempty"`
	Dependencies     []string                `json:"dependencies,omitempty"`  // Required imports/dependencies
	Tests            string                  `json:"tests,omitempty"`         // Generated tests
	Documentation    string                  `json:"documentation,omitempty"` // Generated documentation
	NextSteps        []string                `json:"next_steps,omitempty"`
	SimilarExamples  []ProjectExample        `json:"similar_examples,omitempty"` // Similar code in project
	ArchitecturalFit *ArchitecturalAlignment `json:"architectural_fit,omitempty"`
	QualityMetrics   *CodeQualityMetrics     `json:"quality_metrics,omitempty"`
	IntegrationGuide *IntegrationGuide       `json:"integration_guide,omitempty"`
}

// ProjectExample represents an example from the user's project
type ProjectExample struct {
	FilePath     string  `json:"file_path"`
	FunctionName string  `json:"function_name,omitempty"`
	Code         string  `json:"code"`
	Usage        string  `json:"usage,omitempty"`       // How it's used in the project
	Explanation  string  `json:"explanation,omitempty"` // Why it's relevant
	Similarity   float64 `json:"similarity"`            // Similarity score
}

// ArchitecturalAlignment represents how well generated code fits the project architecture
type ArchitecturalAlignment struct {
	Score           float64  `json:"score"`                // 0.0 to 1.0
	Architecture    string   `json:"architecture"`         // Detected architecture type
	Patterns        []string `json:"patterns"`             // Patterns followed
	Violations      []string `json:"violations,omitempty"` // Architecture violations
	Recommendations []string `json:"recommendations,omitempty"`
}

// CodeQualityMetrics represents quality metrics for generated code
type CodeQualityMetrics struct {
	Complexity      int            `json:"complexity"`              // Cyclomatic complexity
	Maintainability float64        `json:"maintainability"`         // 0.0 to 1.0
	Readability     float64        `json:"readability"`             // 0.0 to 1.0
	TestCoverage    float64        `json:"test_coverage,omitempty"` // If tests included
	Issues          []QualityIssue `json:"issues,omitempty"`
}

// QualityIssue represents a code quality issue
type QualityIssue struct {
	Type     string `json:"type"` // warning, error, suggestion
	Message  string `json:"message"`
	Line     int    `json:"line,omitempty"`
	Severity string `json:"severity"` // low, medium, high
	Rule     string `json:"rule,omitempty"`
}

// IntegrationGuide represents guidance for integrating generated code
type IntegrationGuide struct {
	Steps            []IntegrationStep `json:"steps"`
	Dependencies     []string          `json:"dependencies,omitempty"`
	Configuration    map[string]string `json:"configuration,omitempty"`
	TestInstructions string            `json:"test_instructions,omitempty"`
	Notes            []string          `json:"notes,omitempty"`
}

// IntegrationStep represents a step in the integration process
type IntegrationStep struct {
	Order       int    `json:"order"`
	Description string `json:"description"`
	Command     string `json:"command,omitempty"`
	File        string `json:"file,omitempty"`
	Content     string `json:"content,omitempty"`
}

// ExplanationResponse represents code explanation results
type ExplanationResponse struct {
	Code           string                `json:"code"`
	Language       string                `json:"language"`
	Overview       string                `json:"overview"`
	DetailedSteps  []ExplanationStep     `json:"detailed_steps,omitempty"`
	KeyConcepts    []string              `json:"key_concepts,omitempty"`
	Dependencies   []string              `json:"dependencies,omitempty"`
	UsageExamples  []ProjectExample      `json:"usage_examples,omitempty"`
	RelatedCode    []SearchResult        `json:"related_code,omitempty"`
	Architecture   *ArchitecturalContext `json:"architecture,omitempty"`
	ComplexityInfo *ComplexityInfo       `json:"complexity_info,omitempty"`
}

// ExplanationStep represents a step in code explanation
type ExplanationStep struct {
	Order       int        `json:"order"`
	Title       string     `json:"title"`
	Description string     `json:"description"`
	CodeSnippet string     `json:"code_snippet,omitempty"`
	LineRange   *LineRange `json:"line_range,omitempty"`
}

// LineRange represents a range of lines in code
type LineRange struct {
	Start int `json:"start"`
	End   int `json:"end"`
}

// ArchitecturalContext represents architectural context for explanations
type ArchitecturalContext struct {
	Layer        string   `json:"layer,omitempty"`        // Which architectural layer
	Pattern      string   `json:"pattern,omitempty"`      // Design pattern used
	Role         string   `json:"role,omitempty"`         // Role in architecture
	Interactions []string `json:"interactions,omitempty"` // What it interacts with
}

// ComplexityInfo represents complexity information
type ComplexityInfo struct {
	Cyclomatic  int      `json:"cyclomatic"`            // Cyclomatic complexity
	Cognitive   int      `json:"cognitive,omitempty"`   // Cognitive complexity
	Lines       int      `json:"lines"`                 // Lines of code
	Difficulty  string   `json:"difficulty"`            // easy, medium, hard
	Suggestions []string `json:"suggestions,omitempty"` // Simplification suggestions
}

// AnalysisResponse represents code analysis results
type AnalysisResponse struct {
	Target          string             `json:"target"`
	AnalysisType    string             `json:"analysis_type"`
	Summary         string             `json:"summary"`
	Score           float64            `json:"score"`   // Overall score 0.0 to 1.0
	Metrics         map[string]float64 `json:"metrics"` // Specific metrics
	Issues          []AnalysisIssue    `json:"issues,omitempty"`
	Recommendations []string           `json:"recommendations,omitempty"`
	Comparisons     *ProjectComparison `json:"comparisons,omitempty"` // Compare with project
	Timeline        []AnalysisPoint    `json:"timeline,omitempty"`    // Historical analysis
}

// AnalysisIssue represents an issue found during analysis
type AnalysisIssue struct {
	Type        string    `json:"type"`     // security, performance, quality, style
	Severity    string    `json:"severity"` // low, medium, high, critical
	Title       string    `json:"title"`
	Description string    `json:"description"`
	Location    *Location `json:"location,omitempty"`
	Suggestion  string    `json:"suggestion,omitempty"`
	Impact      string    `json:"impact,omitempty"`
	Effort      string    `json:"effort,omitempty"` // low, medium, high
}

// Location represents a location in code
type Location struct {
	File   string     `json:"file"`
	Line   int        `json:"line,omitempty"`
	Column int        `json:"column,omitempty"`
	Range  *LineRange `json:"range,omitempty"`
}

// ProjectComparison represents comparison with project patterns
type ProjectComparison struct {
	SimilarPatterns []string `json:"similar_patterns,omitempty"`
	Deviations      []string `json:"deviations,omitempty"`
	BestPractices   []string `json:"best_practices,omitempty"`
	Improvements    []string `json:"improvements,omitempty"`
}

// AnalysisPoint represents a point in time for analysis
type AnalysisPoint struct {
	Date    time.Time `json:"date"`
	Score   float64   `json:"score"`
	Changes []string  `json:"changes,omitempty"`
}

// StreamingResponse represents a streaming response
type StreamingResponse struct {
	ID         string                 `json:"id"`
	Type       string                 `json:"type"`              // start, chunk, end, error
	Content    string                 `json:"content,omitempty"` // Streamed content
	Chunk      *StreamChunk           `json:"chunk,omitempty"`   // Structured chunk
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
	Complete   bool                   `json:"complete"` // Is this the final chunk
	TokenCount int                    `json:"token_count,omitempty"`
	Timestamp  time.Time              `json:"timestamp"`
}

// StreamChunk represents a chunk of streamed content
type StreamChunk struct {
	LineNumber int    `json:"line_number,omitempty"`
	Content    string `json:"content"`
	Type       string `json:"type,omitempty"` // code, comment, explanation
	Language   string `json:"language,omitempty"`
	Important  bool   `json:"important"` // Should be emphasized
	TokenCount int    `json:"token_count,omitempty"`
}

// ErrorInfo represents error information
type ErrorInfo struct {
	Code       string                 `json:"code"`
	Message    string                 `json:"message"`
	Details    string                 `json:"details,omitempty"`
	Suggestion string                 `json:"suggestion,omitempty"`
	Retryable  bool                   `json:"retryable"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

// PerformanceInfo represents performance information
type PerformanceInfo struct {
	ResponseTime    time.Duration `json:"response_time"`
	ProcessingTime  time.Duration `json:"processing_time"`
	QueueTime       time.Duration `json:"queue_time,omitempty"`
	TokensPerSecond float64       `json:"tokens_per_second,omitempty"`
	MemoryUsed      int64         `json:"memory_used,omitempty"` // bytes
	CacheHit        bool          `json:"cache_hit,omitempty"`
}
