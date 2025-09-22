package agents

import (
	"context"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	// "github.com/yourusername/ai-code-assistant/internal/agents"
	"github.com/yourusername/ai-code-assistant/internal/app"
	"github.com/yourusername/ai-code-assistant/internal/indexer"
	"github.com/yourusername/ai-code-assistant/internal/llm"
	"github.com/yourusername/ai-code-assistant/internal/logger"
)

// CodingAgent handles code generation, modification, and refactoring requests
type CodingAgent struct {
	// Core components
	llmProvider    llm.Provider
	indexer        *indexer.UltraFastIndexer
	contextManager *app.ContextManager

	// Agent configuration
	config *CodingAgentConfig
	logger logger.Logger

	// Capabilities
	capabilities *AgentCapabilities

	// Code analysis
	codeAnalyzer    *CodeAnalyzer
	syntaxValidator *SyntaxValidator
	styleChecker    *CodeStyleChecker

	// Code generation
	templateEngine    *CodeTemplateEngine
	snippetManager    *CodeSnippetManager
	refactoringEngine *RefactoringEngine

	// Quality assurance
	qualityChecker      *CodeQualityChecker
	securityAnalyzer    *SecurityAnalyzer
	performanceAnalyzer *PerformanceAnalyzer

	// Statistics and monitoring
	metrics *CodingAgentMetrics

	// State management
	mu     sync.RWMutex
	status AgentStatus
}

// CodingAgentConfig contains coding agent configuration
type CodingAgentConfig struct {
	// Code generation settings
	EnableCodeGeneration   bool   `json:"enable_code_generation"`
	EnableCodeModification bool   `json:"enable_code_modification"`
	EnableRefactoring      bool   `json:"enable_refactoring"`
	MaxGeneratedLines      int    `json:"max_generated_lines"`
	DefaultLanguage        string `json:"default_language"`

	// Quality settings
	EnableQualityCheck     bool `json:"enable_quality_check"`
	EnableSecurityCheck    bool `json:"enable_security_check"`
	EnablePerformanceCheck bool `json:"enable_performance_check"`
	EnableStyleCheck       bool `json:"enable_style_check"`

	// Code analysis
	EnableSyntaxValidation   bool `json:"enable_syntax_validation"`
	EnableBestPracticeCheck  bool `json:"enable_best_practice_check"`
	EnableComplexityAnalysis bool `json:"enable_complexity_analysis"`

	// Template and snippet management
	EnableTemplates   bool   `json:"enable_templates"`
	EnableSnippets    bool   `json:"enable_snippets"`
	TemplateDirectory string `json:"template_directory"`

	// Language-specific settings
	LanguageConfigs map[string]*LanguageConfig `json:"language_configs"`

	// LLM settings
	LLMModel        string  `json:"llm_model"`
	MaxTokens       int     `json:"max_tokens"`
	Temperature     float32 `json:"temperature"`
	EnableStreaming bool    `json:"enable_streaming"`
}

// LanguageConfig contains language-specific configuration
type LanguageConfig struct {
	Language       string            `json:"language"`
	FileExtensions []string          `json:"file_extensions"`
	StyleGuide     string            `json:"style_guide"`
	Formatter      string            `json:"formatter"`
	Linter         string            `json:"linter"`
	TestFramework  string            `json:"test_framework"`
	BuildTool      string            `json:"build_tool"`
	PackageManager string            `json:"package_manager"`
	Conventions    map[string]string `json:"conventions"`
}

// CodingRequest represents a coding-related request
type CodingRequest struct {
	Type         CodingRequestType      `json:"type"`
	Language     string                 `json:"language"`
	Description  string                 `json:"description"`
	ExistingCode string                 `json:"existing_code,omitempty"`
	Requirements []string               `json:"requirements,omitempty"`
	Constraints  []string               `json:"constraints,omitempty"`
	Style        *CodeStyle             `json:"style,omitempty"`
	Context      *CodingContext         `json:"context,omitempty"`
	Options      map[string]interface{} `json:"options,omitempty"`
}

type CodingRequestType string

const (
	RequestTypeGenerate CodingRequestType = "generate"
	RequestTypeModify   CodingRequestType = "modify"
	RequestTypeRefactor CodingRequestType = "refactor"
	RequestTypeOptimize CodingRequestType = "optimize"
	RequestTypeExplain  CodingRequestType = "explain"
	RequestTypeReview   CodingRequestType = "review"
	RequestTypeComplete CodingRequestType = "complete"
	RequestTypeConvert  CodingRequestType = "convert"
)

// CodingResponse represents the agent's response
type CodingResponse struct {
	GeneratedCode   string               `json:"generated_code,omitempty"`
	ModifiedCode    string               `json:"modified_code,omitempty"`
	Explanation     string               `json:"explanation"`
	Changes         []*CodeChange        `json:"changes,omitempty"`
	Suggestions     []*CodeSuggestion    `json:"suggestions,omitempty"`
	QualityReport   *CodeQualityReport   `json:"quality_report,omitempty"`
	SecurityIssues  []*SecurityIssue     `json:"security_issues,omitempty"`
	Performance     *PerformanceAnalysis `json:"performance,omitempty"`
	Dependencies    []string             `json:"dependencies,omitempty"`
	TestSuggestions []*TestSuggestion    `json:"test_suggestions,omitempty"`
}

// Supporting types

type CodingContext struct {
	ProjectStructure   *ProjectStructure `json:"project_structure,omitempty"`
	RelatedFiles       []string          `json:"related_files,omitempty"`
	ImportStatements   []string          `json:"import_statements,omitempty"`
	AvailableFunctions []string          `json:"available_functions,omitempty"`
	AvailableClasses   []string          `json:"available_classes,omitempty"`
	Architecture       *ArchitectureInfo `json:"architecture,omitempty"`
	Standards          *CodingStandards  `json:"standards,omitempty"`
}

type ProjectStructure struct {
	RootPath     string            `json:"root_path"`
	SourceDirs   []string          `json:"source_dirs"`
	TestDirs     []string          `json:"test_dirs"`
	ConfigFiles  []string          `json:"config_files"`
	Dependencies map[string]string `json:"dependencies"`
	BuildSystem  string            `json:"build_system"`
}

type CodeChange struct {
	Type       ChangeType   `json:"type"`
	LineNumber int          `json:"line_number,omitempty"`
	OldCode    string       `json:"old_code,omitempty"`
	NewCode    string       `json:"new_code"`
	Reason     string       `json:"reason"`
	Impact     ChangeImpact `json:"impact"`
}

type ChangeType string

const (
	ChangeTypeAdd    ChangeType = "add"
	ChangeTypeModify ChangeType = "modify"
	ChangeTypeDelete ChangeType = "delete"
	ChangeTypeMove   ChangeType = "move"
)

type ChangeImpact string

const (
	ImpactLow      ChangeImpact = "low"
	ImpactMedium   ChangeImpact = "medium"
	ImpactHigh     ChangeImpact = "high"
	ImpactCritical ChangeImpact = "critical"
)

type CodeSuggestion struct {
	Type          SuggestionType `json:"type"`
	Title         string         `json:"title"`
	Description   string         `json:"description"`
	Code          string         `json:"code,omitempty"`
	Priority      Priority       `json:"priority"`
	Category      string         `json:"category"`
	EstimatedTime time.Duration  `json:"estimated_time,omitempty"`
}

type SuggestionType string

const (
	SuggestionTypeBestPractice SuggestionType = "best_practice"
	SuggestionTypeOptimization SuggestionType = "optimization"
	SuggestionTypeRefactoring  SuggestionType = "refactoring"
	SuggestionTypeSecurity     SuggestionType = "security"
	SuggestionTypeStyle        SuggestionType = "style"
	SuggestionTypePerformance  SuggestionType = "performance"
)

type Priority string

const (
	PriorityLow      Priority = "low"
	PriorityMedium   Priority = "medium"
	PriorityHigh     Priority = "high"
	PriorityCritical Priority = "critical"
)

// Quality and analysis types

type CodeQualityReport struct {
	OverallScore    float64            `json:"overall_score"`
	Maintainability float64            `json:"maintainability"`
	Readability     float64            `json:"readability"`
	Testability     float64            `json:"testability"`
	Complexity      *ComplexityMetrics `json:"complexity"`
	Coverage        *CoverageInfo      `json:"coverage,omitempty"`
	Issues          []*QualityIssue    `json:"issues"`
	Recommendations []string           `json:"recommendations"`
}

type ComplexityMetrics struct {
	CyclomaticComplexity int `json:"cyclomatic_complexity"`
	CognitiveComplexity  int `json:"cognitive_complexity"`
	NestingDepth         int `json:"nesting_depth"`
	LinesOfCode          int `json:"lines_of_code"`
	FunctionCount        int `json:"function_count"`
	ClassCount           int `json:"class_count"`
}

type QualityIssue struct {
	Type        string        `json:"type"`
	Severity    IssueSeverity `json:"severity"`
	Description string        `json:"description"`
	LineNumber  int           `json:"line_number,omitempty"`
	Column      int           `json:"column,omitempty"`
	Suggestion  string        `json:"suggestion,omitempty"`
	Rule        string        `json:"rule,omitempty"`
}

type IssueSeverity string

const (
	SeverityInfo     IssueSeverity = "info"
	SeverityWarning  IssueSeverity = "warning"
	SeverityError    IssueSeverity = "error"
	SeverityCritical IssueSeverity = "critical"
)

type SecurityIssue struct {
	Type        SecurityIssueType `json:"type"`
	Severity    IssueSeverity     `json:"severity"`
	Description string            `json:"description"`
	LineNumber  int               `json:"line_number,omitempty"`
	Mitigation  string            `json:"mitigation"`
	References  []string          `json:"references,omitempty"`
}

type SecurityIssueType string

const (
	SecurityTypeInjection SecurityIssueType = "injection"
	SecurityTypeXSS       SecurityIssueType = "xss"
	SecurityTypeAuth      SecurityIssueType = "authentication"
	SecurityTypeCrypto    SecurityIssueType = "cryptography"
	SecurityTypePrivacy   SecurityIssueType = "privacy"
	SecurityTypeHardcoded SecurityIssueType = "hardcoded_secrets"
)

type PerformanceAnalysis struct {
	Score                   float64                   `json:"score"`
	TimeComplexity          string                    `json:"time_complexity,omitempty"`
	SpaceComplexity         string                    `json:"space_complexity,omitempty"`
	BottleneckSummary       string                    `json:"bottleneck_summary,omitempty"`
	OptimizationSuggestions []*OptimizationSuggestion `json:"optimization_suggestions"`
	ProfileData             map[string]interface{}    `json:"profile_data,omitempty"`
}

type OptimizationSuggestion struct {
	Type           string `json:"type"`
	Description    string `json:"description"`
	ExpectedImpact string `json:"expected_impact"`
	Difficulty     string `json:"difficulty"`
	Example        string `json:"example,omitempty"`
}

type TestSuggestion struct {
	Type        string   `json:"type"`
	Description string   `json:"description"`
	TestCode    string   `json:"test_code,omitempty"`
	Framework   string   `json:"framework"`
	Priority    Priority `json:"priority"`
}

// CodingAgentMetrics tracks agent performance
type CodingAgentMetrics struct {
	RequestsProcessed       int64                       `json:"requests_processed"`
	RequestsByType          map[CodingRequestType]int64 `json:"requests_by_type"`
	RequestsByLanguage      map[string]int64            `json:"requests_by_language"`
	AverageResponseTime     time.Duration               `json:"average_response_time"`
	AverageCodeLength       int                         `json:"average_code_length"`
	SuccessRate             float64                     `json:"success_rate"`
	QualityScoreAverage     float64                     `json:"quality_score_average"`
	SecurityIssuesFound     int64                       `json:"security_issues_found"`
	PerformanceImprovements int64                       `json:"performance_improvements"`
	RefactoringsPerformed   int64                       `json:"refactorings_performed"`
	LastRequest             time.Time                   `json:"last_request"`
	mu                      sync.RWMutex
}

// NewCodingAgent creates a new coding agent
// func NewCodingAgent(llmProvider llm.Provider, indexer *indexer.UltraFastIndexer, config *CodingAgentConfig, logger logger.Logger) *CodingAgent {
func NewCodingAgent(config *app.CodingConfig, logger logger.Logger) *CodingAgent {
	if config == nil {
		return nil
	}

	// Convert app.CodingConfig to internal CodingAgentConfig
    agentConfig := &CodingAgentConfig{
		EnableCodeGeneration:     config.EnableCodeGeneration,
		EnableCodeModification:   config.EnableCodeModification,
		EnableRefactoring:        config.EnableRefactoring,
		EnableQualityCheck:       config.EnableQualityCheck,
		LLMModel:                 config.LLMModel,
		MaxTokens:                config.MaxTokens,
		Temperature:              config.Temperature,
		// Set other defaults for internal fields
		MaxGeneratedLines:        500,
		DefaultLanguage:          "go",
		EnableSecurityCheck:      true,
		EnablePerformanceCheck:   true,
		EnableStyleCheck:         true,
		EnableSyntaxValidation:   true,
		EnableBestPracticeCheck:  true,
		EnableComplexityAnalysis: true,
		EnableTemplates:          true,
		EnableSnippets:           true,
		EnableStreaming:          false,
		LanguageConfigs:          make(map[string]*LanguageConfig),
    }

	agent := &CodingAgent{
		config:    agentConfig,
		logger:      logger,
		status:      StatusIdle,
		metrics: &CodingAgentMetrics{
			RequestsByType:     make(map[CodingRequestType]int64),
			RequestsByLanguage: make(map[string]int64),
		},
	}

	// Initialize capabilities
	agent.initializeCapabilities()

	// Initialize components
	agent.initializeComponents()

	return agent
}

// ProcessRequest processes a coding request
func (ca *CodingAgent) ProcessRequest(ctx context.Context, request *AgentRequest) (*AgentResponse, error) {
	start := time.Now()
	ca.status = StatusBusy
	defer func() { ca.status = StatusIdle }()

	// Parse coding request
	codingRequest, err := ca.parseCodingRequest(request)
	if err != nil {
		return nil, fmt.Errorf("failed to parse coding request: %v", err)
	}

	// Validate request
	if err := ca.validateCodingRequest(codingRequest); err != nil {
		return nil, fmt.Errorf("invalid coding request: %v", err)
	}

	// Process based on request type
	var codingResponse *CodingResponse
	switch codingRequest.Type {
	case RequestTypeGenerate:
		codingResponse, err = ca.generateCode(ctx, codingRequest)
	case RequestTypeModify:
		codingResponse, err = ca.modifyCode(ctx, codingRequest)
	case RequestTypeRefactor:
		codingResponse, err = ca.refactorCode(ctx, codingRequest)
	case RequestTypeOptimize:
		codingResponse, err = ca.optimizeCode(ctx, codingRequest)
	case RequestTypeExplain:
		codingResponse, err = ca.explainCode(ctx, codingRequest)
	case RequestTypeReview:
		codingResponse, err = ca.reviewCode(ctx, codingRequest)
	case RequestTypeComplete:
		codingResponse, err = ca.completeCode(ctx, codingRequest)
	case RequestTypeConvert:
		codingResponse, err = ca.convertCode(ctx, codingRequest)
	default:
		return nil, fmt.Errorf("unsupported request type: %s", codingRequest.Type)
	}

	if err != nil {
		ca.updateMetrics(codingRequest.Type, codingRequest.Language, false, time.Since(start))
		return nil, fmt.Errorf("coding request processing failed: %v", err)
	}

	// Perform quality checks if enabled
	if ca.config.EnableQualityCheck && (codingRequest.Type == RequestTypeGenerate || codingRequest.Type == RequestTypeModify) {
		qualityReport, err := ca.performQualityCheck(codingResponse.GeneratedCode, codingRequest.Language)
		if err != nil {
			ca.logger.Warn("Quality check failed", "error", err)
		} else {
			codingResponse.QualityReport = qualityReport
		}
	}

	// Perform security checks if enabled
	if ca.config.EnableSecurityCheck && codingResponse.GeneratedCode != "" {
		securityIssues, err := ca.performSecurityCheck(codingResponse.GeneratedCode, codingRequest.Language)
		if err != nil {
			ca.logger.Warn("Security check failed", "error", err)
		} else {
			codingResponse.SecurityIssues = securityIssues
		}
	}

	// Create agent response
	response := &AgentResponse{
		RequestID:      request.ID,
		AgentType:      ca.GetType(),
		AgentVersion:   ca.GetVersion(),
		Result:         codingResponse,
		Confidence:     ca.calculateConfidence(codingRequest, codingResponse),
		ProcessingTime: time.Since(start),
		CreatedAt:      time.Now(),
	}

	// Update metrics
	ca.updateMetrics(codingRequest.Type, codingRequest.Language, true, time.Since(start))

	return response, nil
}

// generateCode generates new code based on description
func (ca *CodingAgent) generateCode(ctx context.Context, request *CodingRequest) (*CodingResponse, error) {
	if !ca.config.EnableCodeGeneration {
		return nil, fmt.Errorf("code generation is disabled")
	}

	// Prepare context for LLM
	prompt := ca.buildGenerationPrompt(request)

	// Get language-specific configuration
	langConfig := ca.getLanguageConfig(request.Language)

	// Call LLM
	llmRequest := &llm.CompletionRequest{
		Prompt:      prompt,
		Model:       ca.config.LLMModel,
		MaxTokens:   ca.config.MaxTokens,
		Temperature: ca.config.Temperature,
		Stop:        ca.getStopSequences(request.Language),
	}

	response, err := ca.llmProvider.Complete(ctx, llmRequest)
	if err != nil {
		return nil, fmt.Errorf("LLM completion failed: %v", err)
	}

	// Extract and clean generated code
	generatedCode := ca.extractCode(response.Text, request.Language)

	// Validate syntax if enabled
	if ca.config.EnableSyntaxValidation {
		if err := ca.syntaxValidator.Validate(generatedCode, request.Language); err != nil {
			ca.logger.Warn("Generated code has syntax errors", "error", err)
			// Try to fix common syntax issues
			generatedCode = ca.fixCommonSyntaxIssues(generatedCode, request.Language)
		}
	}

	// Generate explanation
	explanation := ca.generateExplanation(request, generatedCode)

	// Generate suggestions
	suggestions := ca.generateSuggestions(request, generatedCode)

	// Determine dependencies
	dependencies := ca.analyzeDependencies(generatedCode, request.Language)

	return &CodingResponse{
		GeneratedCode: generatedCode,
		Explanation:   explanation,
		Suggestions:   suggestions,
		Dependencies:  dependencies,
	}, nil
}

// modifyCode modifies existing code based on requirements
func (ca *CodingAgent) modifyCode(ctx context.Context, request *CodingRequest) (*CodingResponse, error) {
	if !ca.config.EnableCodeModification {
		return nil, fmt.Errorf("code modification is disabled")
	}

	if request.ExistingCode == "" {
		return nil, fmt.Errorf("existing code is required for modification")
	}

	// Analyze existing code
	analysis := ca.codeAnalyzer.Analyze(request.ExistingCode, request.Language)

	// Build modification prompt
	prompt := ca.buildModificationPrompt(request, analysis)

	// Call LLM
	llmRequest := &llm.CompletionRequest{
		Prompt:      prompt,
		Model:       ca.config.LLMModel,
		MaxTokens:   ca.config.MaxTokens,
		Temperature: ca.config.Temperature,
	}

	response, err := ca.llmProvider.Complete(ctx, llmRequest)
	if err != nil {
		return nil, fmt.Errorf("LLM completion failed: %v", err)
	}

	// Extract modified code
	modifiedCode := ca.extractCode(response.Text, request.Language)

	// Calculate changes
	changes := ca.calculateChanges(request.ExistingCode, modifiedCode)

	// Generate explanation
	explanation := ca.generateModificationExplanation(request, changes)

	return &CodingResponse{
		ModifiedCode: modifiedCode,
		Changes:      changes,
		Explanation:  explanation,
	}, nil
}

// refactorCode refactors code to improve quality while maintaining functionality
func (ca *CodingAgent) refactorCode(ctx context.Context, request *CodingRequest) (*CodingResponse, error) {
	if !ca.config.EnableRefactoring {
		return nil, fmt.Errorf("refactoring is disabled")
	}

	if request.ExistingCode == "" {
		return nil, fmt.Errorf("existing code is required for refactoring")
	}

	// Analyze code for refactoring opportunities
	opportunities := ca.identifyRefactoringOpportunities(request.ExistingCode, request.Language)

	// Apply refactoring using the refactoring engine
	refactoredCode, refactorings, err := ca.refactoringEngine.Refactor(request.ExistingCode, opportunities, request.Language)
	if err != nil {
		return nil, fmt.Errorf("refactoring failed: %v", err)
	}

	// Convert refactorings to changes
	changes := ca.convertRefactoringsToChanges(refactorings)

	// Generate explanation
	explanation := ca.generateRefactoringExplanation(refactorings)

	// Generate suggestions for additional improvements
	suggestions := ca.generateRefactoringSuggestions(refactoredCode, request.Language)

	return &CodingResponse{
		ModifiedCode: refactoredCode,
		Changes:      changes,
		Explanation:  explanation,
		Suggestions:  suggestions,
	}, nil
}

// optimizeCode optimizes code for performance
func (ca *CodingAgent) optimizeCode(ctx context.Context, request *CodingRequest) (*CodingResponse, error) {
	if request.ExistingCode == "" {
		return nil, fmt.Errorf("existing code is required for optimization")
	}

	// Analyze performance characteristics
	perfAnalysis := ca.performanceAnalyzer.Analyze(request.ExistingCode, request.Language)

	// Apply optimizations
	optimizedCode, optimizations, err := ca.performanceAnalyzer.Optimize(request.ExistingCode, request.Language)
	if err != nil {
		return nil, fmt.Errorf("optimization failed: %v", err)
	}

	// Calculate changes
	changes := ca.calculateChanges(request.ExistingCode, optimizedCode)

	// Generate explanation
	explanation := ca.generateOptimizationExplanation(optimizations, perfAnalysis)

	return &CodingResponse{
		ModifiedCode: optimizedCode,
		Changes:      changes,
		Explanation:  explanation,
		Performance:  perfAnalysis,
	}, nil
}

// explainCode provides detailed explanation of code functionality
func (ca *CodingAgent) explainCode(ctx context.Context, request *CodingRequest) (*CodingResponse, error) {
	code := request.ExistingCode
	if code == "" {
		return nil, fmt.Errorf("code is required for explanation")
	}

	// Analyze code structure
	analysis := ca.codeAnalyzer.Analyze(code, request.Language)

	// Build explanation prompt
	prompt := ca.buildExplanationPrompt(request, analysis)

	// Call LLM
	llmRequest := &llm.CompletionRequest{
		Prompt:      prompt,
		Model:       ca.config.LLMModel,
		MaxTokens:   ca.config.MaxTokens,
		Temperature: 0.2, // Lower temperature for explanations
	}

	response, err := ca.llmProvider.Complete(ctx, llmRequest)
	if err != nil {
		return nil, fmt.Errorf("LLM completion failed: %v", err)
	}

	// Generate additional insights
	suggestions := ca.generateCodeInsights(code, request.Language, analysis)

	return &CodingResponse{
		Explanation: response.Text,
		Suggestions: suggestions,
	}, nil
}

// reviewCode performs comprehensive code review
func (ca *CodingAgent) reviewCode(ctx context.Context, request *CodingRequest) (*CodingResponse, error) {
	code := request.ExistingCode
	if code == "" {
		return nil, fmt.Errorf("code is required for review")
	}

	response := &CodingResponse{}

	// Quality analysis
	if ca.config.EnableQualityCheck {
		qualityReport, err := ca.performQualityCheck(code, request.Language)
		if err != nil {
			ca.logger.Warn("Quality check failed during review", "error", err)
		} else {
			response.QualityReport = qualityReport
		}
	}

	// Security analysis
	if ca.config.EnableSecurityCheck {
		securityIssues, err := ca.performSecurityCheck(code, request.Language)
		if err != nil {
			ca.logger.Warn("Security check failed during review", "error", err)
		} else {
			response.SecurityIssues = securityIssues
		}
	}

	// Performance analysis
	if ca.config.EnablePerformanceCheck {
		perfAnalysis := ca.performanceAnalyzer.Analyze(code, request.Language)
		response.Performance = perfAnalysis
	}

	// Generate comprehensive review
	reviewPrompt := ca.buildReviewPrompt(request, response.QualityReport, response.SecurityIssues, response.Performance)

	llmRequest := &llm.CompletionRequest{
		Prompt:      reviewPrompt,
		Model:       ca.config.LLMModel,
		MaxTokens:   ca.config.MaxTokens,
		Temperature: 0.3,
	}

	llmResponse, err := ca.llmProvider.Complete(ctx, llmRequest)
	if err != nil {
		return nil, fmt.Errorf("LLM completion failed: %v", err)
	}

	response.Explanation = llmResponse.Text

	// Generate improvement suggestions
	response.Suggestions = ca.generateReviewSuggestions(code, request.Language, response.QualityReport)

	return response, nil
}

// completeCode completes partial code
func (ca *CodingAgent) completeCode(ctx context.Context, request *CodingRequest) (*CodingResponse, error) {
	if request.ExistingCode == "" {
		return nil, fmt.Errorf("existing code is required for completion")
	}

	// Build completion prompt
	prompt := ca.buildCompletionPrompt(request)

	// Call LLM
	llmRequest := &llm.CompletionRequest{
		Prompt:      prompt,
		Model:       ca.config.LLMModel,
		MaxTokens:   ca.config.MaxTokens,
		Temperature: ca.config.Temperature,
	}

	response, err := ca.llmProvider.Complete(ctx, llmRequest)
	if err != nil {
		return nil, fmt.Errorf("LLM completion failed: %v", err)
	}

	// Extract completion
	completion := ca.extractCompletion(response.Text, request.ExistingCode)
	completedCode := request.ExistingCode + completion

	// Generate explanation
	explanation := ca.generateCompletionExplanation(request, completion)

	return &CodingResponse{
		GeneratedCode: completedCode,
		Explanation:   explanation,
	}, nil
}

// convertCode converts code from one language to another
func (ca *CodingAgent) convertCode(ctx context.Context, request *CodingRequest) (*CodingResponse, error) {
	if request.ExistingCode == "" {
		return nil, fmt.Errorf("existing code is required for conversion")
	}

	targetLang, ok := request.Options["target_language"].(string)
	if !ok || targetLang == "" {
		return nil, fmt.Errorf("target_language option is required for conversion")
	}

	// Build conversion prompt
	prompt := ca.buildConversionPrompt(request, targetLang)

	// Call LLM
	llmRequest := &llm.CompletionRequest{
		Prompt:      prompt,
		Model:       ca.config.LLMModel,
		MaxTokens:   ca.config.MaxTokens,
		Temperature: 0.1, // Low temperature for accuracy
	}

	response, err := ca.llmProvider.Complete(ctx, llmRequest)
	if err != nil {
		return nil, fmt.Errorf("LLM completion failed: %v", err)
	}

	// Extract converted code
	convertedCode := ca.extractCode(response.Text, targetLang)

	// Generate explanation
	explanation := ca.generateConversionExplanation(request.Language, targetLang, request.ExistingCode, convertedCode)

	return &CodingResponse{
		GeneratedCode: convertedCode,
		Explanation:   explanation,
	}, nil
}

// Helper methods for prompt building

func (ca *CodingAgent) buildGenerationPrompt(request *CodingRequest) string {
	var prompt strings.Builder

	prompt.WriteString(fmt.Sprintf("Generate %s code based on the following description:\n\n", request.Language))
	prompt.WriteString(fmt.Sprintf("Description: %s\n\n", request.Description))

	if len(request.Requirements) > 0 {
		prompt.WriteString("Requirements:\n")
		for _, req := range request.Requirements {
			prompt.WriteString(fmt.Sprintf("- %s\n", req))
		}
		prompt.WriteString("\n")
	}

	if len(request.Constraints) > 0 {
		prompt.WriteString("Constraints:\n")
		for _, constraint := range request.Constraints {
			prompt.WriteString(fmt.Sprintf("- %s\n", constraint))
		}
		prompt.WriteString("\n")
	}

	// Add context if available
	if request.Context != nil {
		ca.addContextToPrompt(&prompt, request.Context)
	}

	// Add language-specific guidelines
	langConfig := ca.getLanguageConfig(request.Language)
	if langConfig != nil {
		prompt.WriteString("Follow these language conventions:\n")
		for convention, value := range langConfig.Conventions {
			prompt.WriteString(fmt.Sprintf("- %s: %s\n", convention, value))
		}
		prompt.WriteString("\n")
	}

	prompt.WriteString("Generate clean, well-commented, and maintainable code. Include error handling where appropriate.")

	return prompt.String()
}

func (ca *CodingAgent) buildModificationPrompt(request *CodingRequest, analysis *CodeAnalysis) string {
	var prompt strings.Builder

	prompt.WriteString(fmt.Sprintf("Modify the following %s code according to the requirements:\n\n", request.Language))
	prompt.WriteString("Current code:\n```\n")
	prompt.WriteString(request.ExistingCode)
	prompt.WriteString("\n```\n\n")

	prompt.WriteString(fmt.Sprintf("Modification requirements: %s\n\n", request.Description))

	if len(request.Requirements) > 0 {
		prompt.WriteString("Specific requirements:\n")
		for _, req := range request.Requirements {
			prompt.WriteString(fmt.Sprintf("- %s\n", req))
		}
		prompt.WriteString("\n")
	}

	// Add analysis insights
	if analysis != nil {
		prompt.WriteString("Code analysis insights:\n")
		prompt.WriteString(fmt.Sprintf("- Functions: %d\n", analysis.FunctionCount))
		prompt.WriteString(fmt.Sprintf("- Classes: %d\n", analysis.ClassCount))
		prompt.WriteString(fmt.Sprintf("- Complexity: %d\n", analysis.CyclomaticComplexity))
		prompt.WriteString("\n")
	}

	prompt.WriteString("Provide the modified code while maintaining existing functionality and improving code quality.")

	return prompt.String()
}

func (ca *CodingAgent) addContextToPrompt(prompt *strings.Builder, context *CodingContext) {
	if len(context.ImportStatements) > 0 {
		prompt.WriteString("Available imports:\n")
		for _, imp := range context.ImportStatements {
			prompt.WriteString(fmt.Sprintf("- %s\n", imp))
		}
		prompt.WriteString("\n")
	}

	if len(context.AvailableFunctions) > 0 {
		prompt.WriteString("Available functions:\n")
		for _, fn := range context.AvailableFunctions {
			prompt.WriteString(fmt.Sprintf("- %s\n", fn))
		}
		prompt.WriteString("\n")
	}

	if len(context.AvailableClasses) > 0 {
		prompt.WriteString("Available classes:\n")
		for _, cls := range context.AvailableClasses {
			prompt.WriteString(fmt.Sprintf("- %s\n", cls))
		}
		prompt.WriteString("\n")
	}
}

// Utility methods

func (ca *CodingAgent) parseCodingRequest(request *AgentRequest) (*CodingRequest, error) {
	// Try to parse from parameters first
	if params, ok := request.Parameters["coding_request"].(*CodingRequest); ok {
		return params, nil
	}

	// Parse from query using intent and context
	codingRequest := &CodingRequest{
		Type:        ca.inferRequestType(request.Intent.Type),
		Language:    ca.inferLanguage(request.Context),
		Description: request.Query,
	}

	// Extract existing code from context
	if request.Context != nil && request.Context.SelectedText != "" {
		codingRequest.ExistingCode = request.Context.SelectedText
	}

	// Set default language if not inferred
	if codingRequest.Language == "" {
		codingRequest.Language = ca.config.DefaultLanguage
	}

	return codingRequest, nil
}

func (ca *CodingAgent) inferRequestType(intentType IntentType) CodingRequestType {
	switch intentType {
	case IntentCodeGeneration:
		return RequestTypeGenerate
	case IntentCodeModification:
		return RequestTypeModify
	case IntentCodeRefactoring:
		return RequestTypeRefactor
	case IntentCodeExplanation:
		return RequestTypeExplain
	case IntentCodeReview:
		return RequestTypeReview
	default:
		return RequestTypeGenerate
	}
}

func (ca *CodingAgent) inferLanguage(context *RequestContext) string {
	if context == nil {
		return ""
	}

	// Try project language first
	if context.ProjectLanguage != "" {
		return context.ProjectLanguage
	}

	// Try to infer from current file
	if context.CurrentFile != "" {
		return ca.inferLanguageFromFile(context.CurrentFile)
	}

	return ""
}

func (ca *CodingAgent) inferLanguageFromFile(filename string) string {
	ext := strings.ToLower(strings.TrimPrefix(filepath.Ext(filename), "."))

	// Language mapping
	langMap := map[string]string{
		"go":   "go",
		"py":   "python",
		"js":   "javascript",
		"ts":   "typescript",
		"java": "java",
		"cpp":  "cpp",
		"c":    "c",
		"cs":   "csharp",
		"rb":   "ruby",
		"php":  "php",
		"rs":   "rust",
	}

	if lang, exists := langMap[ext]; exists {
		return lang
	}

	return ""
}

// Component initialization and management

func (ca *CodingAgent) initializeCapabilities() {
	ca.capabilities = &AgentCapabilities{
		AgentType: AgentTypeCoding,
		SupportedIntents: []IntentType{
			IntentCodeGeneration,
			IntentCodeModification,
			IntentCodeRefactoring,
			IntentCodeExplanation,
			IntentCodeReview,
		},
		SupportedLanguages: []string{
			"go", "python", "javascript", "typescript", "java",
			"cpp", "c", "csharp", "ruby", "php", "rust",
		},
		SupportedFileTypes: []string{
			"go", "py", "js", "ts", "java", "cpp", "c", "cs", "rb", "php", "rs",
		},
		MaxContextSize:    8192,
		SupportsStreaming: ca.config.EnableStreaming,
		SupportsAsync:     true,
		RequiresContext:   false,
		Capabilities: map[string]interface{}{
			"code_generation":   ca.config.EnableCodeGeneration,
			"code_modification": ca.config.EnableCodeModification,
			"refactoring":       ca.config.EnableRefactoring,
			"quality_check":     ca.config.EnableQualityCheck,
			"security_check":    ca.config.EnableSecurityCheck,
			"performance_check": ca.config.EnablePerformanceCheck,
			"syntax_validation": ca.config.EnableSyntaxValidation,
		},
	}
}

func (ca *CodingAgent) initializeComponents() {
	// Initialize code analyzer
	ca.codeAnalyzer = NewCodeAnalyzer()

	// Initialize syntax validator
	ca.syntaxValidator = NewSyntaxValidator()

	// Initialize style checker
	if ca.config.EnableStyleCheck {
		ca.styleChecker = NewCodeStyleChecker()
	}

	// Initialize quality checker
	if ca.config.EnableQualityCheck {
		ca.qualityChecker = NewCodeQualityChecker()
	}

	// Initialize security analyzer
	if ca.config.EnableSecurityCheck {
		ca.securityAnalyzer = NewSecurityAnalyzer()
	}

	// Initialize performance analyzer
	if ca.config.EnablePerformanceCheck {
		ca.performanceAnalyzer = NewPerformanceAnalyzer()
	}

	// Initialize template engine
	if ca.config.EnableTemplates {
		ca.templateEngine = NewCodeTemplateEngine(ca.config.TemplateDirectory)
	}

	// Initialize refactoring engine
	if ca.config.EnableRefactoring {
		ca.refactoringEngine = NewRefactoringEngine()
	}
}

// Required Agent interface methods

func (ca *CodingAgent) GetCapabilities() *AgentCapabilities {
	return ca.capabilities
}

func (ca *CodingAgent) GetType() AgentType {
	return AgentTypeCoding
}

func (ca *CodingAgent) GetVersion() string {
	return "1.0.0"
}

func (ca *CodingAgent) GetStatus() AgentStatus {
	ca.mu.RLock()
	defer ca.mu.RUnlock()
	return ca.status
}

func (ca *CodingAgent) Initialize(config interface{}) error {
	if cfg, ok := config.(*CodingAgentConfig); ok {
		ca.config = cfg
		ca.initializeComponents()
		return nil
	}
	return fmt.Errorf("invalid config type")
}


// SetConfig updates the agent's configuration dynamically
func (ca *CodingAgent) SetConfig(config interface{}) error {
  appConfig, ok := config.(*app.CodingConfig)
  if !ok {
          return fmt.Errorf("invalid config type for CodingAgent, expected *app.CodingConfig")
  }

  ca.mu.Lock()
  defer ca.mu.Unlock()

  // Convert app.CodingConfig to internal CodingAgentConfig
  newConfig := &CodingAgentConfig{
          EnableCodeGeneration:   appConfig.EnableCodeGeneration,
          EnableCodeModification: appConfig.EnableCodeModification,
          EnableRefactoring:      appConfig.EnableRefactoring,
          EnableQualityCheck:     appConfig.EnableQualityCheck,
          LLMModel:              appConfig.LLMModel,
          MaxTokens:             appConfig.MaxTokens,
          Temperature:           appConfig.Temperature,
          // Preserve existing internal settings
          MaxGeneratedLines:        ca.config.MaxGeneratedLines,
          DefaultLanguage:          ca.config.DefaultLanguage,
          EnableSecurityCheck:      ca.config.EnableSecurityCheck,
          EnablePerformanceCheck:   ca.config.EnablePerformanceCheck,
          EnableStyleCheck:         ca.config.EnableStyleCheck,
          EnableSyntaxValidation:   ca.config.EnableSyntaxValidation,
          EnableBestPracticeCheck:  ca.config.EnableBestPracticeCheck,
          EnableComplexityAnalysis: ca.config.EnableComplexityAnalysis,
          EnableTemplates:          ca.config.EnableTemplates,
          EnableSnippets:           ca.config.EnableSnippets,
          EnableStreaming:          ca.config.EnableStreaming,
          LanguageConfigs:          ca.config.LanguageConfigs,
  }

  // Update configuration
  ca.config = newConfig

  // Re-initialize components with new config
  ca.initializeComponents()

  ca.logger.Info("CodingAgent configuration updated",
          "llm_model", newConfig.LLMModel,
          "max_tokens", newConfig.MaxTokens,
          "temperature", newConfig.Temperature)

  return nil
}



func (ca *CodingAgent) Start() error {
	ca.mu.Lock()
	defer ca.mu.Unlock()

	ca.status = StatusIdle
	ca.logger.Info("Coding agent started")
	return nil
}

func (ca *CodingAgent) Stop() error {
	ca.mu.Lock()
	defer ca.mu.Unlock()

	ca.status = StatusStopped
	ca.logger.Info("Coding agent stopped")
	return nil
}

func (ca *CodingAgent) HealthCheck() *agents.HealthStatus {
	startTime := time.Now()
	status := &agents.HealthStatus{
		LastCheckTime:      startTime,
		DependenciesStatus: make(map[string]*agents.HealthStatus),
		Details:            make(map[string]interface{}),	
	}

	// Check LLM provider
	if ca.llmProvider == nil {
		status.Status = agents.HealthStatusUnhealthy
		status.Message = "LLM provider is not configured"
		status.Latency = time.Since(startTime)
		return status
	}
	
	// Test LLM connectivity
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
	defer cancel()

	testRequest := &llm.CompletionRequest{
		Prompt:    "test",
		Model:     ca.config.LLMModel,
		MaxTokens: 1,
	}

	_, err := ca.llmProvider.Complete(ctx, testRequest)
	status.Latency = time.Since(startTime)

	// Get metrics for health evaluation
	metrics := ca.GetMetrics()
	status.ErrorCount = metrics.ErrorCount

	// Evaluate health based on thresholds
	healthConfig := ca.getHealthCheckConfig()
	if err != nil {
			status.Status = agents.HealthStatusUnhealthy
			status.Message = fmt.Sprintf("LLM provider error: %v", err)
	} else if status.Latency > time.Duration(healthConfig.MaxLatencyMs)*time.Millisecond {
			status.Status = agents.HealthStatusDegraded
			status.Message = fmt.Sprintf("High latency: %v", status.Latency)
	} else if metrics.ErrorCount > 0 && float64(metrics.ErrorCount)/float64(metrics.RequestCount) > healthConfig.MaxErrorRate {
			status.Status = agents.HealthStatusDegraded
			status.Message = fmt.Sprintf("High error rate: %.2f%%", float64(metrics.ErrorCount)/float64(metrics.RequestCount)*100)
	} else {
			status.Status = agents.HealthStatusHealthy
			status.Message = "All systems operational"
	}

	// Add details
	status.Details["llm_model"] = ca.config.LLMModel
	status.Details["request_count"] = metrics.RequestCount
	status.Details["avg_response_time"] = metrics.AverageResponseTime

	return status
}

func (ca *CodingAgent) getHealthCheckConfig() *agents.HealthCheckConfig {
	// Default health check configuration
	defaultConfig := &agents.HealthCheckConfig{
			MaxLatencyMs:            2000,
			MaxErrorRate:            0.1,
			DependencyCheckInterval: time.Minute * 5,
			EnableDependencyCheck:   true,
			HealthCheckTimeout:      time.Second * 10,
	}

	// Use configured values if available
	if ca.config != nil && ca.config.HealthCheck != nil {
			return ca.config.HealthCheck
	}

	return defaultConfig
}

func (ca *CodingAgent) GetMetrics() *AgentMetrics {
	ca.metrics.mu.RLock()
	defer ca.metrics.mu.RUnlock()

	return &AgentMetrics{
		RequestsProcessed:   ca.metrics.RequestsProcessed,
		AverageResponseTime: ca.metrics.AverageResponseTime,
		SuccessRate:         ca.metrics.SuccessRate,
		AverageConfidence:   ca.metrics.QualityScoreAverage,
		LastRequestAt:       ca.metrics.LastRequest,
	}
}

func (ca *CodingAgent) ResetMetrics() {
	ca.metrics.mu.Lock()
	defer ca.metrics.mu.Unlock()

	ca.metrics = &CodingAgentMetrics{
		RequestsByType:     make(map[CodingRequestType]int64),
		RequestsByLanguage: make(map[string]int64),
	}
}

// Helper methods (simplified implementations)

func (ca *CodingAgent) validateCodingRequest(request *CodingRequest) error {
	if request.Description == "" {
		return fmt.Errorf("description is required")
	}

	if request.Language == "" {
		request.Language = ca.config.DefaultLanguage
	}

	return nil
}

func (ca *CodingAgent) getLanguageConfig(language string) *LanguageConfig {
	if config, exists := ca.config.LanguageConfigs[language]; exists {
		return config
	}
	return nil
}

func (ca *CodingAgent) getStopSequences(language string) []string {
	// Language-specific stop sequences
	switch language {
	case "python":
		return []string{"\n\n", "```"}
	case "go":
		return []string{"\n\n", "```", "package "}
	default:
		return []string{"\n\n", "```"}
	}
}

func (ca *CodingAgent) extractCode(text, language string) string {
	// Extract code from LLM response
	// Look for code blocks first
	codeBlockRegex := regexp.MustCompile("```(?:" + language + ")?\n?(.*?)\n?```")
	matches := codeBlockRegex.FindStringSubmatch(text)
	if len(matches) > 1 {
		return strings.TrimSpace(matches[1])
	}

	// If no code blocks, return the text as is (cleaned)
	return strings.TrimSpace(text)
}

func (ca *CodingAgent) calculateConfidence(request *CodingRequest, response *CodingResponse) float64 {
	confidence := 0.8 // Base confidence

	// Adjust based on request type
	switch request.Type {
	case RequestTypeGenerate:
		confidence = 0.7 // Lower for generation
	case RequestTypeExplain:
		confidence = 0.9 // Higher for explanations
	case RequestTypeReview:
		confidence = 0.85 // High for reviews
	}

	// Adjust based on quality report
	if response.QualityReport != nil {
		qualityBonus := response.QualityReport.OverallScore * 0.2
		confidence += qualityBonus
	}

	// Cap at 1.0
	if confidence > 1.0 {
		confidence = 1.0
	}

	return confidence
}

func (ca *CodingAgent) updateMetrics(requestType CodingRequestType, language string, success bool, duration time.Duration) {
	ca.metrics.mu.Lock()
	defer ca.metrics.mu.Unlock()

	ca.metrics.RequestsProcessed++
	ca.metrics.RequestsByType[requestType]++
	ca.metrics.RequestsByLanguage[language]++
	// Update success rate
	if ca.metrics.RequestsProcessed == 1 {
		if success {
			ca.metrics.SuccessRate = 1.0
		} else {
			ca.metrics.SuccessRate = 0.0
		}
	} else {
		oldSuccessCount := int64(ca.metrics.SuccessRate * float64(ca.metrics.RequestsProcessed-1))
		if success {
			oldSuccessCount++
		}
		ca.metrics.SuccessRate = float64(oldSuccessCount) / float64(ca.metrics.RequestsProcessed)
	}

	// Update average response time
	if ca.metrics.AverageResponseTime == 0 {
		ca.metrics.AverageResponseTime = duration
	} else {
		ca.metrics.AverageResponseTime = (ca.metrics.AverageResponseTime + duration) / 2
	}

	ca.metrics.LastRequest = time.Now()
}

// Additional helper methods
func (ca *CodingAgent) fixCommonSyntaxIssues(code, language string) string {
	// Simple syntax fixes - in practice this would be more sophisticated
	switch language {
	case "python":
		// Fix common Python indentation issues
		lines := strings.Split(code, "\n")
		for i, line := range lines {
			if strings.TrimSpace(line) != "" && !strings.HasPrefix(line, " ") && !strings.HasPrefix(line, "\t") {
				// Add basic indentation if missing
				if i > 0 && strings.HasSuffix(lines[i-1], ":") {
					lines[i] = "    " + line
				}
			}
		}
		return strings.Join(lines, "\n")
	case "go":
		// Add missing package declaration if not present
		if !strings.HasPrefix(strings.TrimSpace(code), "package ") {
			return "package main\n\n" + code
		}
		return code
	default:
		return code
	}
}
func (ca *CodingAgent) generateExplanation(request *CodingRequest, generatedCode string) string {
	explanation := fmt.Sprintf("Generated %s code based on your requirements:\n\n", request.Language)
	explanation += fmt.Sprintf("The code implements: %s\n\n", request.Description)

	if len(request.Requirements) > 0 {
		explanation += "Key features implemented:\n"
		for _, req := range request.Requirements {
			explanation += fmt.Sprintf("- %s\n", req)
		}
		explanation += "\n"
	}

	explanation += "The generated code follows best practices and includes appropriate error handling."

	return explanation
}

func (ca *CodingAgent) generateSuggestions(request *CodingRequest, generatedCode string) []*CodeSuggestion {
	var suggestions []*CodeSuggestion
	// Add unit testing suggestion
	suggestions = append(suggestions, &CodeSuggestion{
		Type:        SuggestionTypeBestPractice,
		Title:       "Add Unit Tests",
		Description: "Consider adding unit tests to verify the functionality of the generated code",
		Priority:    PriorityMedium,
		Category:    "testing",
	})

	// Add documentation suggestion
	if !strings.Contains(generatedCode, "/**") && !strings.Contains(generatedCode, "'''") {
		suggestions = append(suggestions, &CodeSuggestion{
			Type:        SuggestionTypeBestPractice,
			Title:       "Add Documentation",
			Description: "Add comprehensive documentation comments to explain the code functionality",
			Priority:    PriorityMedium,
			Category:    "documentation",
		})
	}

	// Add error handling suggestion if not present
	if !strings.Contains(strings.ToLower(generatedCode), "error") &&
		!strings.Contains(strings.ToLower(generatedCode), "exception") {
		suggestions = append(suggestions, &CodeSuggestion{
			Type:        SuggestionTypeBestPractice,
			Title:       "Enhance Error Handling",
			Description: "Consider adding more robust error handling and validation",
			Priority:    PriorityHigh,
			Category:    "reliability",
		})
	}

	return suggestions
}

func (ca *CodingAgent) analyzeDependencies(code, language string) []string {
	var dependencies []string
	switch language {
	case "go":
		// Extract Go imports
		importRegex := regexp.MustCompile(`import\s+"([^"]+)"`)
		matches := importRegex.FindAllStringSubmatch(code, -1)
		for _, match := range matches {
			if len(match) > 1 {
				dependencies = append(dependencies, match[1])
			}
		}
	case "python":
		// Extract Python imports
		importRegex := regexp.MustCompile(`(?:from\s+(\S+)\s+import|import\s+(\S+))`)
		matches := importRegex.FindAllStringSubmatch(code, -1)
		for _, match := range matches {
			if len(match) > 1 && match[1] != "" {
				dependencies = append(dependencies, match[1])
			} else if len(match) > 2 && match[2] != "" {
				dependencies = append(dependencies, match[2])
			}
		}
	case "javascript", "typescript":
		// Extract JS/TS imports
		importRegex := regexp.MustCompile(`(?:import.*from\s+['"]([^'"]+)['"]|require\(['"]([^'"]+)['"]\))`)
		matches := importRegex.FindAllStringSubmatch(code, -1)
		for _, match := range matches {
			if len(match) > 1 && match[1] != "" {
				dependencies = append(dependencies, match[1])
			} else if len(match) > 2 && match[2] != "" {
				dependencies = append(dependencies, match[2])
			}
		}
	}

	return dependencies
}

func (ca *CodingAgent) calculateChanges(oldCode, newCode string) []*CodeChange {
	// Simple diff algorithm - in practice would use more sophisticated diff
	oldLines := strings.Split(oldCode, "\n")
	newLines := strings.Split(newCode, "\n")
	var changes []*CodeChange

	// Find added/modified lines
	for i, newLine := range newLines {
		if i < len(oldLines) {
			if oldLines[i] != newLine {
				changes = append(changes, &CodeChange{
					Type:       ChangeTypeModify,
					LineNumber: i + 1,
					OldCode:    oldLines[i],
					NewCode:    newLine,
					Reason:     "Line modified",
					Impact:     ImpactLow,
				})
			}
		} else {
			changes = append(changes, &CodeChange{
				Type:       ChangeTypeAdd,
				LineNumber: i + 1,
				NewCode:    newLine,
				Reason:     "Line added",
				Impact:     ImpactLow,
			})
		}
	}

	// Find deleted lines
	if len(oldLines) > len(newLines) {
		for i := len(newLines); i < len(oldLines); i++ {
			changes = append(changes, &CodeChange{
				Type:       ChangeTypeDelete,
				LineNumber: i + 1,
				OldCode:    oldLines[i],
				Reason:     "Line removed",
				Impact:     ImpactLow,
			})
		}
	}

	return changes
}

func (ca *CodingAgent) performQualityCheck(code, language string) (*CodeQualityReport, error) {
	if ca.qualityChecker == nil {
		return nil, fmt.Errorf("quality checker not initialized")
	}
	return ca.qualityChecker.Check(code, language)
}
func (ca *CodingAgent) performSecurityCheck(code, language string) ([]*SecurityIssue, error) {
	if ca.securityAnalyzer == nil {
		return nil, fmt.Errorf("security analyzer not initialized")
	}
	return ca.securityAnalyzer.Analyze(code, language)
}

// Placeholder implementations for referenced components
type CodeAnalyzer struct{}
type CodeAnalysis struct {
FunctionCount         int                    json:"function_count"
ClassCount           int                    json:"class_count"
CyclomaticComplexity int                    json:"cyclomatic_complexity"
LinesOfCode          int                    json:"lines_of_code"
Imports              []string               json:"imports"
Functions            []string               json:"functions"
Classes              []string               json:"classes"
}
func NewCodeAnalyzer() *CodeAnalyzer {
return &CodeAnalyzer{}
}
func (ca *CodeAnalyzer) Analyze(code, language string) *CodeAnalysis {
analysis := &CodeAnalysis{
LinesOfCode: len(strings.Split(code, "\n")),
}
// Simple analysis - count functions and classes
switch language {
case "go":
	analysis.FunctionCount = strings.Count(code, "func ")
	analysis.ClassCount = strings.Count(code, "type ") // Simplified
case "python":
	analysis.FunctionCount = strings.Count(code, "def ")
	analysis.ClassCount = strings.Count(code, "class ")
case "javascript", "typescript":
	analysis.FunctionCount = strings.Count(code, "function ") + strings.Count(code, "=>")
	analysis.ClassCount = strings.Count(code, "class ")
}

// Simple complexity calculation
analysis.CyclomaticComplexity = 1 + 
	strings.Count(code, "if ") +
	strings.Count(code, "for ") +
	strings.Count(code, "while ") +
	strings.Count(code, "case ") +
	strings.Count(code, "catch ")

return analysis
}
type SyntaxValidator struct{}
func NewSyntaxValidator() *SyntaxValidator {
return &SyntaxValidator{}
}
func (sv *SyntaxValidator) Validate(code, language string) error {
// Simplified syntax validation
// In practice, would use language-specific parsers
// Check for basic syntax errors
openBraces := strings.Count(code, "{")
closeBraces := strings.Count(code, "}")
if openBraces != closeBraces {
	return fmt.Errorf("mismatched braces: %d open, %d close", openBraces, closeBraces)
}

openParens := strings.Count(code, "(")
closeParens := strings.Count(code, ")")
if openParens != closeParens {
	return fmt.Errorf("mismatched parentheses: %d open, %d close", openParens, closeParens)
}

return nil
}
type CodeStyleChecker struct{}
func NewCodeStyleChecker() *CodeStyleChecker {
return &CodeStyleChecker{}
}
type CodeQualityChecker struct{}
func NewCodeQualityChecker() *CodeQualityChecker {
return &CodeQualityChecker{}
}
func (cqc *CodeQualityChecker) Check(code, language string) (*CodeQualityReport, error) {
report := &CodeQualityReport{
OverallScore:    0.8, // Default score
Maintainability: 0.8,
Readability:     0.8,
Testability:     0.7,
Issues:          []*QualityIssue{},
Recommendations: []string{},
}
// Simple quality checks
lines := strings.Split(code, "\n")

// Check for long lines
for i, line := range lines {
	if len(line) > 120 {
		report.Issues = append(report.Issues, &QualityIssue{
			Type:        "long_line",
			Severity:    SeverityWarning,
			Description: "Line exceeds recommended length",
			LineNumber:  i + 1,
			Suggestion:  "Consider breaking long lines for better readability",
			Rule:        "line_length",
		})
	}
}

// Check for complex functions (simplified)
if strings.Count(code, "if ") > 10 {
	report.Issues = append(report.Issues, &QualityIssue{
		Type:        "high_complexity",
		Severity:    SeverityWarning,
		Description: "Function appears to be complex",
		Suggestion:  "Consider breaking down into smaller functions",
		Rule:        "complexity",
	})
}

// Add recommendations
if len(report.Issues) > 0 {
	report.Recommendations = append(report.Recommendations, 
		"Review and address the identified quality issues")
}

report.Recommendations = append(report.Recommendations, 
	"Add comprehensive unit tests",
	"Consider adding more documentation")

return report, nil
}
type SecurityAnalyzer struct{}
func NewSecurityAnalyzer() *SecurityAnalyzer {
return &SecurityAnalyzer{}
}
func (sa *SecurityAnalyzer) Analyze(code, language string) ([]*SecurityIssue, error) {
var issues []*SecurityIssue
codeLower := strings.ToLower(code)

// Check for hardcoded secrets
secretPatterns := []string{"password", "secret", "key", "token", "api_key"}
for _, pattern := range secretPatterns {
	if strings.Contains(codeLower, pattern+"=") || strings.Contains(codeLower, pattern+":") {
		issues = append(issues, &SecurityIssue{
			Type:        SecurityTypeHardcoded,
			Severity:    SeverityCritical,
			Description: "Potential hardcoded secret detected",
			Mitigation:  "Use environment variables or secure configuration management",
			References:  []string{"OWASP Top 10"},
		})
	}
}

// Check for SQL injection patterns
if strings.Contains(codeLower, "select ") && strings.Contains(code, "+") {
	issues = append(issues, &SecurityIssue{
		Type:        SecurityTypeInjection,
		Severity:    SeverityHigh,
		Description: "Potential SQL injection vulnerability",
		Mitigation:  "Use parameterized queries or prepared statements",
		References:  []string{"OWASP SQL Injection Prevention"},
	})
}

// Check for XSS patterns in web contexts
if strings.Contains(codeLower, "innerhtml") || strings.Contains(codeLower, "document.write") {
	issues = append(issues, &SecurityIssue{
		Type:        SecurityTypeXSS,
		Severity:    SeverityHigh,
		Description: "Potential XSS vulnerability",
		Mitigation:  "Sanitize user input and use safe DOM manipulation methods",
		References:  []string{"OWASP XSS Prevention"},
	})
}

return issues, nil
}
type PerformanceAnalyzer struct{}
func NewPerformanceAnalyzer() *PerformanceAnalyzer {
return &PerformanceAnalyzer{}
}
func (pa *PerformanceAnalyzer) Analyze(code, language string) *PerformanceAnalysis {
analysis := &PerformanceAnalysis{
Score:                   0.8, // Default score
OptimizationSuggestions: []*OptimizationSuggestion{},
}
// Simple performance analysis
codeLower := strings.ToLower(code)

// Check for nested loops
if strings.Count(code, "for ") > 1 && strings.Count(code, "{") > 2 {
	analysis.TimeComplexity = "O(n²) or higher"
	analysis.OptimizationSuggestions = append(analysis.OptimizationSuggestions, &OptimizationSuggestion{
		Type:           "algorithm",
		Description:    "Nested loops detected - consider algorithmic optimization",
		ExpectedImpact: "Significant performance improvement for large datasets",
		Difficulty:     "Medium",
	})
}

// Check for string concatenation in loops
if strings.Contains(codeLower, "for ") && strings.Contains(code, "+") {
	analysis.OptimizationSuggestions = append(analysis.OptimizationSuggestions, &OptimizationSuggestion{
		Type:           "memory",
		Description:    "String concatenation in loop - consider using string builder",
		ExpectedImpact: "Reduced memory allocation and improved performance",
		Difficulty:     "Easy",
	})
}

return analysis
}
func (pa *PerformanceAnalyzer) Optimize(code, language string) (string, []*OptimizationSuggestion, error) {
// Simple optimization - in practice would be much more sophisticated
optimizedCode := code
var optimizations []*OptimizationSuggestion
// Example: Replace string concatenation in Go
if language == "go" && strings.Contains(code, `result += `) {
	optimizedCode = strings.ReplaceAll(code, 
		`result += `, 
		`result = append(result, `)
	optimizations = append(optimizations, &OptimizationSuggestion{
		Type:           "performance",
		Description:    "Replaced string concatenation with more efficient approach",
		ExpectedImpact: "Improved performance for string operations",
		Difficulty:     "Easy",
	})
}

return optimizedCode, optimizations, nil
}

// Additional helper method implementations
func (ca *CodingAgent) identifyRefactoringOpportunities(code, language string) []RefactoringOpportunity {
var opportunities []RefactoringOpportunity
// Simple refactoring opportunity detection
lines := strings.Split(code, "\n")

// Check for long functions
if len(lines) > 50 {
	opportunities = append(opportunities, RefactoringOpportunity{
		Type:        "extract_method",
		Description: "Function is too long, consider extracting methods",
		Priority:    PriorityMedium,
	})
}

// Check for code duplication (simplified)
lineMap := make(map[string]int)
for _, line := range lines {
	if trimmed := strings.TrimSpace(line); trimmed != "" {
		lineMap[trimmed]++
	}
}

for line, count := range lineMap {
	if count > 3 && len(line) > 20 {
		opportunities = append(opportunities, RefactoringOpportunity{
			Type:        "extract_constant",
			Description: fmt.Sprintf("Duplicate code detected: %s", line[:50]),
			Priority:    PriorityHigh,
		})
		break
	}
}

return opportunities
}
type RefactoringOpportunity struct {
Type        string   json:"type"
Description string   json:"description"
Priority    Priority json:"priority"
}
type CodeTemplateEngine struct {
templateDir string
templates   map[string]string
}
func NewCodeTemplateEngine(templateDir string) *CodeTemplateEngine {
return &CodeTemplateEngine{
templateDir: templateDir,
templates:   make(map[string]string),
}
}
type CodeSnippetManager struct{}
type RefactoringEngine struct{}
func NewRefactoringEngine() *RefactoringEngine {
return &RefactoringEngine{}
}
func (re *RefactoringEngine) Refactor(code string, opportunities []RefactoringOpportunity, language string) (string, []string, error) {
// Simplified refactoring - in practice would be much more sophisticated
refactoredCode := code
var refactorings []string
for _, opp := range opportunities {
	switch opp.Type {
	case "extract_method":
		refactorings = append(refactorings, "Extracted long method into smaller functions")
		// Would implement actual method extraction
	case "extract_constant":
		refactorings = append(refactorings, "Extracted magic numbers into constants")
		// Would implement constant extraction
	}
}

return refactoredCode, refactorings, nil
}
// Additional helper methods for prompt building
func (ca *CodingAgent) buildExplanationPrompt(request *CodingRequest, analysis *CodeAnalysis) string {
var prompt strings.Builder
prompt.WriteString(fmt.Sprintf("Explain the following %s code in detail:\n\n", request.Language))
prompt.WriteString("```\n")
prompt.WriteString(request.ExistingCode)
prompt.WriteString("\n```\n\n")

prompt.WriteString("Please provide a comprehensive explanation that covers:\n")
prompt.WriteString("1. Overall purpose and functionality\n")
prompt.WriteString("2. Key components and their roles\n")
prompt.WriteString("3. Algorithm or logic flow\n")
prompt.WriteString("4. Input/output behavior\n")
prompt.WriteString("5. Any notable patterns or techniques used\n\n")

if analysis != nil {
	prompt.WriteString(fmt.Sprintf("Code statistics:\n"))
	prompt.WriteString(fmt.Sprintf("- Functions: %d\n", analysis.FunctionCount))
	prompt.WriteString(fmt.Sprintf("- Classes: %d\n", analysis.ClassCount))
	prompt.WriteString(fmt.Sprintf("- Lines of code: %d\n", analysis.LinesOfCode))
	prompt.WriteString(fmt.Sprintf("- Complexity: %d\n", analysis.CyclomaticComplexity))
}

return prompt.String()
}
func (ca *CodingAgent) buildReviewPrompt(request *CodingRequest, qualityReport *CodeQualityReport, securityIssues []*SecurityIssue, perfAnalysis *PerformanceAnalysis) string {
var prompt strings.Builder
prompt.WriteString(fmt.Sprintf("Provide a comprehensive code review for the following %s code:\n\n", request.Language))
prompt.WriteString("```\n")
prompt.WriteString(request.ExistingCode)
prompt.WriteString("\n```\n\n")

prompt.WriteString("Review aspects to cover:\n")
prompt.WriteString("1. Code quality and maintainability\n")
prompt.WriteString("2. Best practices adherence\n")
prompt.WriteString("3. Security considerations\n")
prompt.WriteString("4. Performance implications\n")
prompt.WriteString("5. Readability and documentation\n")
prompt.WriteString("6. Testing considerations\n")
prompt.WriteString("7. Improvement suggestions\n\n")

if qualityReport != nil {
	prompt.WriteString(fmt.Sprintf("Quality metrics:\n"))
	prompt.WriteString(fmt.Sprintf("- Overall score: %.2f\n", qualityReport.OverallScore))
	prompt.WriteString(fmt.Sprintf("- Issues found: %d\n", len(qualityReport.Issues)))
}

if len(securityIssues) > 0 {
	prompt.WriteString(fmt.Sprintf("Security issues found: %d\n", len(securityIssues)))
}

if perfAnalysis != nil {
	prompt.WriteString(fmt.Sprintf("Performance score: %.2f\n", perfAnalysis.Score))
}

return prompt.String()
}
func (ca *CodingAgent) buildCompletionPrompt(request *CodingRequest) string {
var prompt strings.Builder
prompt.WriteString(fmt.Sprintf("Complete the following %s code:\n\n", request.Language))
prompt.WriteString("```\n")
prompt.WriteString(request.ExistingCode)
prompt.WriteString("\n```\n\n")

if request.Description != "" {
	prompt.WriteString(fmt.Sprintf("Completion requirements: %s\n\n", request.Description))
}

prompt.WriteString("Complete the code following best practices and maintaining consistency with the existing style.")

return prompt.String()
}
func (ca *CodingAgent) buildConversionPrompt(request *CodingRequest, targetLanguage string) string {
var prompt strings.Builder
prompt.WriteString(fmt.Sprintf("Convert the following %s code to %s:\n\n", request.Language, targetLanguage))
prompt.WriteString("```\n")
prompt.WriteString(request.ExistingCode)
prompt.WriteString("\n```\n\n")

prompt.WriteString("Conversion requirements:\n")
prompt.WriteString("1. Maintain the same functionality\n")
prompt.WriteString("2. Follow target language best practices\n")
prompt.WriteString("3. Use idiomatic constructs for the target language\n")
prompt.WriteString("4. Preserve comments and documentation\n")
prompt.WriteString("5. Handle language-specific differences appropriately\n")

return prompt.String()
}
// Additional utility methods
func (ca *CodingAgent) extractCompletion(response, existingCode string) string {
// Extract only the new completion part
if strings.HasPrefix(response, existingCode) {
return strings.TrimPrefix(response, existingCode)
}
// If the response doesn't start with existing code, return as is
return response
}
func (ca *CodingAgent) generateModificationExplanation(request *CodingRequest, changes []*CodeChange) string {
explanation := fmt.Sprintf("Modified the %s code according to your requirements:\n\n", request.Language)
explanation += fmt.Sprintf("Changes made: %d\n\n", len(changes))
for _, change := range changes {
	explanation += fmt.Sprintf("- %s (Line %d): %s\n", change.Type, change.LineNumber, change.Reason)
}

return explanation
}
func (ca *CodingAgent) generateRefactoringExplanation(refactorings []string) string {
explanation := "Refactored code to improve quality and maintainability:\n\n"
for _, refactoring := range refactorings {
	explanation += fmt.Sprintf("- %s\n", refactoring)
}

explanation += "\nThe refactored code maintains the same functionality while improving code structure and readability."

return explanation
}
func (ca *CodingAgent) generateOptimizationExplanation(optimizations []*OptimizationSuggestion, analysis *PerformanceAnalysis) string {
explanation := "Optimized code for better performance:\n\n"
if analysis.TimeComplexity != "" {
	explanation += fmt.Sprintf("Time complexity: %s\n", analysis.TimeComplexity)
}

if analysis.SpaceComplexity != "" {
	explanation += fmt.Sprintf("Space complexity: %s\n", analysis.SpaceComplexity)
}

explanation += "\nOptimizations applied:\n"
for _, opt := range optimizations {
	explanation += fmt.Sprintf("- %s: %s\n", opt.Type, opt.Description)
}

return explanation
}
func (ca *CodingAgent) generateConversionExplanation(fromLang, toLang, originalCode, convertedCode string) string {
explanation := fmt.Sprintf("Converted code from %s to %s:\n\n", fromLang, toLang)
explanation += "Key conversion notes:\n"
switch toLang {
case "go":
	explanation += "- Added explicit error handling\n"
	explanation += "- Used Go-specific naming conventions\n"
	explanation += "- Implemented Go interfaces where appropriate\n"
case "python":
	explanation += "- Used Pythonic constructs and idioms\n"
	explanation += "- Applied PEP 8 style guidelines\n"
	explanation += "- Leveraged Python's built-in data structures\n"
case "javascript":
	explanation += "- Used modern JavaScript features\n"
	explanation += "- Applied proper async/await patterns where needed\n"
	explanation += "- Followed JavaScript naming conventions\n"
}

explanation += "\nThe converted code maintains the same functionality while following the target language's best practices."

return explanation
}
func (ca *CodingAgent) generateCodeInsights(code, language string, analysis *CodeAnalysis) []*CodeSuggestion {
var suggestions []*CodeSuggestion
// Complexity insight
if analysis.CyclomaticComplexity > 10 {
	suggestions = append(suggestions, &CodeSuggestion{
		Type:        SuggestionTypeRefactoring,
		Title:       "High Complexity Detected",
		Description: fmt.Sprintf("Cyclomatic complexity is %d. Consider breaking down complex functions.", analysis.CyclomaticComplexity),
		Priority:    PriorityHigh,
		Category:    "complexity",
	})
}

// Function count insight
if analysis.FunctionCount == 0 {
	suggestions = append(suggestions, &CodeSuggestion{
		Type:        SuggestionTypeBestPractice,
		Title:       "Consider Function Extraction",
		Description: "Code appears to be in a single block. Consider extracting reusable functions.",
		Priority:    PriorityMedium,
		Category:    "structure",
	})
}

// Size insight
if analysis.LinesOfCode > 100 {
	suggestions = append(suggestions, &CodeSuggestion{
		Type:        SuggestionTypeRefactoring,
		Title:       "Large Code Block",
		Description: "Code is quite large. Consider splitting into multiple files or modules.",
		Priority:    PriorityMedium,
		Category:    "organization",
	})
}

return suggestions
}
func (ca *CodingAgent) generateReviewSuggestions(code, language string, qualityReport *CodeQualityReport) []*CodeSuggestion {
var suggestions []*CodeSuggestion
if qualityReport != nil {
	// Convert quality issues to suggestions
	for _, issue := range qualityReport.Issues {
		suggestions = append(suggestions, &CodeSuggestion{
			Type:        SuggestionTypeRefactoring,
			Title:       issue.Description,
			Description: issue.Suggestion,
			Priority:    ca.convertSeverityToPriority(issue.Severity),
			Category:    issue.Type,
		})
	}
	
	// Add recommendations as suggestions
	for _, rec := range qualityReport.Recommendations {
		suggestions = append(suggestions, &CodeSuggestion{
			Type:        SuggestionTypeBestPractice,
			Title:       "Code Quality Improvement",
			Description: rec,
			Priority:    PriorityMedium,
			Category:    "quality",
		})
	}
}

return suggestions
}
func (ca *CodingAgent) convertRefactoringsToChanges(refactorings []string) []*CodeChange {
var changes []*CodeChange
for i, refactoring := range refactorings {
	changes = append(changes, &CodeChange{
		Type:    ChangeTypeModify,
		NewCode: refactoring,
		Reason:  "Refactoring improvement",
		Impact:  ImpactMedium,
	})
}

return changes
}
func (ca *CodingAgent) generateRefactoringSuggestions(code, language string) []*CodeSuggestion {
var suggestions []*CodeSuggestion
// Add general refactoring suggestions
suggestions = append(suggestions, &CodeSuggestion{
	Type:        SuggestionTypeRefactoring,
	Title:       "Add Unit Tests",
	Description: "Consider adding comprehensive unit tests for the refactored code",
	Priority:    PriorityHigh,
	Category:    "testing",
})

suggestions = append(suggestions, &CodeSuggestion{
	Type:        SuggestionTypeBestPractice,
	Title:       "Code Review",
	Description: "Have the refactored code reviewed by team members",
	Priority:    PriorityMedium,
	Category:    "process",
})

return suggestions
}

func (ca *CodingAgent) convertSeverityToPriority(severity IssueSeverity) Priority {
	switch severity {
	case SeverityCritical:
	return PriorityCritical
	case SeverityError:
	return PriorityHigh
	case SeverityWarning:
	return PriorityMedium
	case SeverityInfo:
	return PriorityLow
	default:
	return PriorityMedium
	}
}