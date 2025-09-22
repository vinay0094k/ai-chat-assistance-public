package agents

import (
	"context"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/app"
	"github.com/yourusername/ai-code-assistant/internal/indexer"
	"github.com/yourusername/ai-code-assistant/internal/llm"
	"github.com/yourusername/ai-code-assistant/internal/logger"
)

// DebuggingAgent assists with identifying and suggesting fixes for code bugs
type DebuggingAgent struct {
	// Core components
	llmProvider    llm.Provider
	indexer        *indexer.UltraFastIndexer
	contextManager *app.ContextManager

	// Agent configuration
	config *DebuggingAgentConfig
	logger logger.Logger

	// Capabilities
	capabilities *AgentCapabilities

	// Bug detection and analysis
	bugDetector        *BugDetector
	errorAnalyzer      *ErrorAnalyzer
	stackTraceAnalyzer *StackTraceAnalyzer

	// Code analysis
	staticAnalyzer  *StaticAnalyzer
	dynamicAnalyzer *DynamicAnalyzer
	patternMatcher  *AntiPatternMatcher

	// Fix generation
	fixGenerator   *FixGenerator
	solutionRanker *SolutionRanker
	testGenerator  *TestGenerator

	// Knowledge base
	knownIssuesDB    *KnownIssuesDatabase
	solutionPatterns *SolutionPatternLibrary

	// Statistics and monitoring
	metrics *DebuggingAgentMetrics

	// State management
	mu     sync.RWMutex
	status AgentStatus
}

// DebuggingAgentConfig contains debugging agent configuration
type DebuggingAgentConfig struct {
	// Detection settings
	EnableBugDetection    bool `json:"enable_bug_detection"`
	EnableStaticAnalysis  bool `json:"enable_static_analysis"`
	EnableDynamicAnalysis bool `json:"enable_dynamic_analysis"`
	EnablePatternMatching bool `json:"enable_pattern_matching"`

	// Error analysis
	EnableErrorAnalysis      bool `json:"enable_error_analysis"`
	EnableStackTraceAnalysis bool `json:"enable_stack_trace_analysis"`
	MaxStackTraceDepth       int  `json:"max_stack_trace_depth"`

	// Fix generation
	EnableFixGeneration  bool `json:"enable_fix_generation"`
	EnableMultipleFixes  bool `json:"enable_multiple_fixes"`
	MaxFixSuggestions    int  `json:"max_fix_suggestions"`
	EnableTestGeneration bool `json:"enable_test_generation"`

	// Severity assessment
	EnableSeverityAssessment bool     `json:"enable_severity_assessment"`
	CriticalPatterns         []string `json:"critical_patterns"`
	WarningPatterns          []string `json:"warning_patterns"`

	// Knowledge base
	EnableKnownIssuesCheck bool   `json:"enable_known_issues_check"`
	KnownIssuesDatabase    string `json:"known_issues_database"`
	EnablePatternLearning  bool   `json:"enable_pattern_learning"`

	// Performance optimization
	MaxAnalysisTime        time.Duration `json:"max_analysis_time"`
	EnableParallelAnalysis bool          `json:"enable_parallel_analysis"`

	// LLM settings
	LLMModel    string  `json:"llm_model"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float32 `json:"temperature"`

	// Language-specific settings
	LanguageAnalyzers map[string]*LanguageAnalyzerConfig `json:"language_analyzers"`
}

type LanguageAnalyzerConfig struct {
	StaticAnalyzer string   `json:"static_analyzer"`
	LintTool       string   `json:"lint_tool"`
	TestFramework  string   `json:"test_framework"`
	CommonPatterns []string `json:"common_patterns"`
	KnownIssues    []string `json:"known_issues"`
}

// Request and response structures

type DebuggingRequest struct {
	Type             DebuggingRequestType `json:"type"`
	Code             string               `json:"code"`
	ErrorMessage     string               `json:"error_message,omitempty"`
	StackTrace       string               `json:"stack_trace,omitempty"`
	Language         string               `json:"language"`
	Context          *DebuggingContext    `json:"context,omitempty"`
	Symptoms         []string             `json:"symptoms,omitempty"`
	ExpectedBehavior string               `json:"expected_behavior,omitempty"`
	ActualBehavior   string               `json:"actual_behavior,omitempty"`
	Options          *DebuggingOptions    `json:"options,omitempty"`
}

type DebuggingRequestType string

const (
	DebugTypeAnalyze       DebuggingRequestType = "analyze"
	DebugTypeFixSuggestion DebuggingRequestType = "fix_suggestion"
	DebugTypeExplain       DebuggingRequestType = "explain_error"
	DebugTypePrevent       DebuggingRequestType = "prevention"
	DebugTypeReview        DebuggingRequestType = "security_review"
	DebugTypePerformance   DebuggingRequestType = "performance"
)

type DebuggingContext struct {
	FilePath      string           `json:"file_path,omitempty"`
	LineNumber    int              `json:"line_number,omitempty"`
	FunctionName  string           `json:"function_name,omitempty"`
	ClassName     string           `json:"class_name,omitempty"`
	RelatedFiles  []string         `json:"related_files,omitempty"`
	Dependencies  []string         `json:"dependencies,omitempty"`
	Environment   *EnvironmentInfo `json:"environment,omitempty"`
	TestCases     []*TestCase      `json:"test_cases,omitempty"`
	RecentChanges []*CodeChange    `json:"recent_changes,omitempty"`
}

type EnvironmentInfo struct {
	OS            string                 `json:"os,omitempty"`
	Runtime       string                 `json:"runtime,omitempty"`
	Version       string                 `json:"version,omitempty"`
	Dependencies  map[string]string      `json:"dependencies,omitempty"`
	Configuration map[string]interface{} `json:"configuration,omitempty"`
}

type TestCase struct {
	Name           string     `json:"name"`
	Input          string     `json:"input"`
	ExpectedOutput string     `json:"expected_output"`
	ActualOutput   string     `json:"actual_output,omitempty"`
	Status         TestStatus `json:"status"`
}

type TestStatus string

const (
	TestStatusPassing TestStatus = "passing"
	TestStatusFailing TestStatus = "failing"
	TestStatusError   TestStatus = "error"
	TestStatusSkipped TestStatus = "skipped"
)

type DebuggingOptions struct {
	IncludeExplanation    bool        `json:"include_explanation"`
	IncludePreventionTips bool        `json:"include_prevention_tips"`
	IncludeTestCases      bool        `json:"include_test_cases"`
	DetailLevel           DetailLevel `json:"detail_level"`
	FocusAreas            []FocusArea `json:"focus_areas,omitempty"`
}

type FocusArea string

const (
	FocusSecurity        FocusArea = "security"
	FocusPerformance     FocusArea = "performance"
	FocusReliability     FocusArea = "reliability"
	FocusMaintainability FocusArea = "maintainability"
)

// Response structures

type DebuggingResponse struct {
	Analysis       *BugAnalysis         `json:"analysis,omitempty"`
	DetectedIssues []*DetectedIssue     `json:"detected_issues,omitempty"`
	Fixes          []*Fix               `json:"fixes,omitempty"`
	Explanation    string               `json:"explanation,omitempty"`
	PreventionTips []*PreventionTip     `json:"prevention_tips,omitempty"`
	TestCases      []*GeneratedTestCase `json:"test_cases,omitempty"`
	RelatedIssues  []*RelatedIssue      `json:"related_issues,omitempty"`
	Confidence     float32              `json:"confidence"`
}

type BugAnalysis struct {
	Summary          string            `json:"summary"`
	RootCause        string            `json:"root_cause"`
	ImpactAssessment *ImpactAssessment `json:"impact_assessment"`
	ComplexityScore  float32           `json:"complexity_score"`
	AnalysisMethod   []string          `json:"analysis_method"`
	Evidence         []*Evidence       `json:"evidence"`
}

type DetectedIssue struct {
	ID            string         `json:"id"`
	Type          IssueType      `json:"type"`
	Severity      IssueSeverity  `json:"severity"`
	Title         string         `json:"title"`
	Description   string         `json:"description"`
	Location      *IssueLocation `json:"location,omitempty"`
	Pattern       string         `json:"pattern,omitempty"`
	Confidence    float32        `json:"confidence"`
	Evidence      []*Evidence    `json:"evidence,omitempty"`
	RelatedIssues []string       `json:"related_issues,omitempty"`
}

type IssueType string

const (
	IssueTypeLogic       IssueType = "logic_error"
	IssueTypeSyntax      IssueType = "syntax_error"
	IssueTypeRuntime     IssueType = "runtime_error"
	IssueTypeMemory      IssueType = "memory_error"
	IssueTypeConcurrency IssueType = "concurrency_error"
	IssueTypeSecurity    IssueType = "security_vulnerability"
	IssueTypePerformance IssueType = "performance_issue"
	IssueTypeReliability IssueType = "reliability_issue"
)

type IssueLocation struct {
	FilePath    string `json:"file_path"`
	LineStart   int    `json:"line_start"`
	LineEnd     int    `json:"line_end"`
	ColumnStart int    `json:"column_start,omitempty"`
	ColumnEnd   int    `json:"column_end,omitempty"`
	Function    string `json:"function,omitempty"`
	Class       string `json:"class,omitempty"`
}

type Evidence struct {
	Type        EvidenceType   `json:"type"`
	Description string         `json:"description"`
	CodeSnippet string         `json:"code_snippet,omitempty"`
	Location    *IssueLocation `json:"location,omitempty"`
	Relevance   float32        `json:"relevance"`
}

type EvidenceType string

const (
	EvidenceCode           EvidenceType = "code_pattern"
	EvidenceError          EvidenceType = "error_message"
	EvidenceStackTrace     EvidenceType = "stack_trace"
	EvidenceTestFailure    EvidenceType = "test_failure"
	EvidenceStaticAnalysis EvidenceType = "static_analysis"
)

type Fix struct {
	ID              string              `json:"id"`
	Title           string              `json:"title"`
	Description     string              `json:"description"`
	Type            FixType             `json:"type"`
	Approach        string              `json:"approach"`
	CodeChanges     []*CodeChange       `json:"code_changes"`
	Explanation     string              `json:"explanation"`
	Confidence      float32             `json:"confidence"`
	EstimatedEffort string              `json:"estimated_effort"`
	RiskAssessment  *RiskAssessment     `json:"risk_assessment"`
	Validation      *ValidationStrategy `json:"validation,omitempty"`
	Prerequisites   []string            `json:"prerequisites,omitempty"`
}

type FixType string

const (
	FixTypeQuick         FixType = "quick_fix"
	FixTypeRefactor      FixType = "refactoring"
	FixTypeArchitecture  FixType = "architecture_change"
	FixTypeConfiguration FixType = "configuration"
	FixTypeWorkaround    FixType = "workaround"
)

type ImpactAssessment struct {
	Severity           IssueSeverity `json:"severity"`
	Scope              string        `json:"scope"`
	AffectedComponents []string      `json:"affected_components"`
	UserImpact         string        `json:"user_impact"`
	BusinessImpact     string        `json:"business_impact"`
	TechnicalDebt      float32       `json:"technical_debt"`
}

type RiskAssessment struct {
	RiskLevel       RiskLevel `json:"risk_level"`
	PotentialIssues []string  `json:"potential_issues"`
	Mitigations     []string  `json:"mitigations"`
	TestingNeeded   bool      `json:"testing_needed"`
}

type RiskLevel string

const (
	RiskLow      RiskLevel = "low"
	RiskMedium   RiskLevel = "medium"
	RiskHigh     RiskLevel = "high"
	RiskCritical RiskLevel = "critical"
)

type ValidationStrategy struct {
	TestCases       []*GeneratedTestCase `json:"test_cases"`
	ManualSteps     []string             `json:"manual_steps"`
	AutomatedChecks []string             `json:"automated_checks"`
}

type GeneratedTestCase struct {
	Name           string   `json:"name"`
	Description    string   `json:"description"`
	TestCode       string   `json:"test_code"`
	Framework      string   `json:"framework"`
	Type           TestType `json:"type"`
	ExpectedResult string   `json:"expected_result"`
}

type TestType string

const (
	TestTypeUnit        TestType = "unit"
	TestTypeIntegration TestType = "integration"
	TestTypeRegression  TestType = "regression"
	TestTypeSecurity    TestType = "security"
	TestTypePerformance TestType = "performance"
)

type PreventionTip struct {
	Category       string   `json:"category"`
	Title          string   `json:"title"`
	Description    string   `json:"description"`
	Implementation string   `json:"implementation"`
	Tools          []string `json:"tools,omitempty"`
	Examples       []string `json:"examples,omitempty"`
}

type RelatedIssue struct {
	ID         string  `json:"id"`
	Title      string  `json:"title"`
	Similarity float32 `json:"similarity"`
	URL        string  `json:"url,omitempty"`
	Resolution string  `json:"resolution,omitempty"`
}

// DebuggingAgentMetrics tracks debugging performance
type DebuggingAgentMetrics struct {
	TotalRequests       int64                          `json:"total_requests"`
	RequestsByType      map[DebuggingRequestType]int64 `json:"requests_by_type"`
	IssuesDetected      int64                          `json:"issues_detected"`
	IssuesByType        map[IssueType]int64            `json:"issues_by_type"`
	FixesGenerated      int64                          `json:"fixes_generated"`
	AverageConfidence   float32                        `json:"average_confidence"`
	AverageResponseTime time.Duration                  `json:"average_response_time"`
	SuccessRate         float64                        `json:"success_rate"`
	TestCasesGenerated  int64                          `json:"test_cases_generated"`
	KnownIssueMatches   int64                          `json:"known_issue_matches"`
	LastRequest         time.Time                      `json:"last_request"`
	mu                  sync.RWMutex
}

// NewDebuggingAgent creates a new debugging agent
func NewDebuggingAgent(llmProvider llm.Provider, indexer *indexer.UltraFastIndexer, config *DebuggingAgentConfig, logger logger.Logger) *DebuggingAgent {
	if config == nil {
		config = &DebuggingAgentConfig{
			EnableBugDetection:       true,
			EnableStaticAnalysis:     true,
			EnableDynamicAnalysis:    false, // Requires runtime environment
			EnablePatternMatching:    true,
			EnableErrorAnalysis:      true,
			EnableStackTraceAnalysis: true,
			MaxStackTraceDepth:       20,
			EnableFixGeneration:      true,
			EnableMultipleFixes:      true,
			MaxFixSuggestions:        5,
			EnableTestGeneration:     true,
			EnableSeverityAssessment: true,
			EnableKnownIssuesCheck:   true,
			EnablePatternLearning:    true,
			MaxAnalysisTime:          time.Minute * 2,
			EnableParallelAnalysis:   true,
			LLMModel:                 "gpt-4",
			MaxTokens:                2048,
			Temperature:              0.1, // Low temperature for consistency
			CriticalPatterns: []string{
				"null pointer", "buffer overflow", "sql injection",
				"xss", "infinite loop", "deadlock",
			},
			WarningPatterns: []string{
				"deprecated", "unused variable", "magic number",
				"code smell", "complexity",
			},
			LanguageAnalyzers: make(map[string]*LanguageAnalyzerConfig),
		}
	}

	agent := &DebuggingAgent{
		llmProvider: llmProvider,
		indexer:     indexer,
		config:      config,
		logger:      logger,
		status:      StatusIdle,
		metrics: &DebuggingAgentMetrics{
			RequestsByType: make(map[DebuggingRequestType]int64),
			IssuesByType:   make(map[IssueType]int64),
		},
	}

	// Initialize capabilities
	agent.initializeCapabilities()

	// Initialize components
	agent.initializeComponents()

	return agent
}

// ProcessRequest processes a debugging request
func (da *DebuggingAgent) ProcessRequest(ctx context.Context, request *AgentRequest) (*AgentResponse, error) {
	start := time.Now()
	da.status = StatusBusy
	defer func() { da.status = StatusIdle }()

	// Parse debugging request
	debugRequest, err := da.parseDebuggingRequest(request)
	if err != nil {
		return nil, fmt.Errorf("failed to parse debugging request: %v", err)
	}

	// Apply timeout
	debugCtx := ctx
	if da.config.MaxAnalysisTime > 0 {
		var cancel context.CancelFunc
		debugCtx, cancel = context.WithTimeout(ctx, da.config.MaxAnalysisTime)
		defer cancel()
	}

	// Process based on request type
	var debugResponse *DebuggingResponse
	switch debugRequest.Type {
	case DebugTypeAnalyze:
		debugResponse, err = da.analyzeBug(debugCtx, debugRequest)
	case DebugTypeFixSuggestion:
		debugResponse, err = da.generateFixSuggestions(debugCtx, debugRequest)
	case DebugTypeExplain:
		debugResponse, err = da.explainError(debugCtx, debugRequest)
	case DebugTypePrevent:
		debugResponse, err = da.generatePreventionTips(debugCtx, debugRequest)
	case DebugTypeReview:
		debugResponse, err = da.performSecurityReview(debugCtx, debugRequest)
	case DebugTypePerformance:
		debugResponse, err = da.analyzePerformance(debugCtx, debugRequest)
	default:
		return nil, fmt.Errorf("unsupported debugging type: %s", debugRequest.Type)
	}

	if err != nil {
		da.updateMetrics(debugRequest.Type, false, time.Since(start))
		return nil, fmt.Errorf("debugging analysis failed: %v", err)
	}

	// Create agent response
	response := &AgentResponse{
		RequestID:      request.ID,
		AgentType:      da.GetType(),
		AgentVersion:   da.GetVersion(),
		Result:         debugResponse,
		Confidence:     da.calculateConfidence(debugRequest, debugResponse),
		ProcessingTime: time.Since(start),
		CreatedAt:      time.Now(),
	}

	// Update metrics
	da.updateMetrics(debugRequest.Type, true, time.Since(start))

	return response, nil
}

// analyzeBug performs comprehensive bug analysis
func (da *DebuggingAgent) analyzeBug(ctx context.Context, request *DebuggingRequest) (*DebuggingResponse, error) {
	var detectedIssues []*DetectedIssue
	var analysis *BugAnalysis
	var err error

	// Static analysis
	if da.config.EnableStaticAnalysis {
		staticIssues, err := da.staticAnalyzer.AnalyzeCode(request.Code, request.Language)
		if err != nil {
			da.logger.Warn("Static analysis failed", "error", err)
		} else {
			detectedIssues = append(detectedIssues, staticIssues...)
		}
	}

	// Pattern matching
	if da.config.EnablePatternMatching {
		patternIssues := da.patternMatcher.FindAntiPatterns(request.Code, request.Language)
		detectedIssues = append(detectedIssues, patternIssues...)
	}

	// Error analysis
	if da.config.EnableErrorAnalysis && request.ErrorMessage != "" {
		errorIssues, err := da.errorAnalyzer.AnalyzeError(request.ErrorMessage, request.Code, request.Language)
		if err != nil {
			da.logger.Warn("Error analysis failed", "error", err)
		} else {
			detectedIssues = append(detectedIssues, errorIssues...)
		}
	}

	// Stack trace analysis
	if da.config.EnableStackTraceAnalysis && request.StackTrace != "" {
		stackTraceAnalysis, err := da.stackTraceAnalyzer.AnalyzeStackTrace(request.StackTrace, request.Code)
		if err != nil {
			da.logger.Warn("Stack trace analysis failed", "error", err)
		} else {
			if stackTraceAnalysis != nil {
				detectedIssues = append(detectedIssues, stackTraceAnalysis.Issues...)
				analysis = &BugAnalysis{
					Summary:   stackTraceAnalysis.Summary,
					RootCause: stackTraceAnalysis.RootCause,
					Evidence:  stackTraceAnalysis.Evidence,
				}
			}
		}
	}

	// Check known issues
	if da.config.EnableKnownIssuesCheck {
		knownIssues := da.knownIssuesDB.CheckKnownIssues(request.Code, request.ErrorMessage, request.Language)
		for _, issue := range knownIssues {
			detectedIssue := &DetectedIssue{
				ID:          issue.ID,
				Type:        IssueType(issue.Type),
				Severity:    IssueSeverity(issue.Severity),
				Title:       issue.Title,
				Description: issue.Description,
				Confidence:  issue.Confidence,
			}
			detectedIssues = append(detectedIssues, detectedIssue)
		}
	}

	// Generate comprehensive analysis if not from stack trace
	if analysis == nil && len(detectedIssues) > 0 {
		analysis = da.generateBugAnalysis(request, detectedIssues)
	}

	// Generate fixes
	var fixes []*Fix
	if da.config.EnableFixGeneration {
		fixes = da.generateFixes(ctx, request, detectedIssues)
	}

	// Generate explanation
	explanation := da.generateExplanation(request, detectedIssues, analysis)

	response := &DebuggingResponse{
		Analysis:       analysis,
		DetectedIssues: detectedIssues,
		Fixes:          fixes,
		Explanation:    explanation,
		Confidence:     da.calculateIssuesConfidence(detectedIssues),
	}

	return response, nil
}

// generateFixSuggestions generates fix suggestions for detected issues
func (da *DebuggingAgent) generateFixSuggestions(ctx context.Context, request *DebuggingRequest) (*DebuggingResponse, error) {
	// First analyze to detect issues
	analysisResponse, err := da.analyzeBug(ctx, request)
	if err != nil {
		return nil, err
	}

	// Generate additional fixes using LLM
	llmFixes, err := da.generateLLMFixes(ctx, request, analysisResponse.DetectedIssues)
	if err != nil {
		da.logger.Warn("LLM fix generation failed", "error", err)
	} else {
		analysisResponse.Fixes = append(analysisResponse.Fixes, llmFixes...)
	}

	// Rank fixes by effectiveness and confidence
	if da.solutionRanker != nil {
		analysisResponse.Fixes = da.solutionRanker.RankFixes(analysisResponse.Fixes, request)
	}

	// Limit to max suggestions
	if len(analysisResponse.Fixes) > da.config.MaxFixSuggestions {
		analysisResponse.Fixes = analysisResponse.Fixes[:da.config.MaxFixSuggestions]
	}

	// Generate test cases for fixes
	if da.config.EnableTestGeneration {
		for _, fix := range analysisResponse.Fixes {
			testCases := da.testGenerator.GenerateTestsForFix(fix, request)
			if fix.Validation == nil {
				fix.Validation = &ValidationStrategy{}
			}
			fix.Validation.TestCases = append(fix.Validation.TestCases, testCases...)
		}
	}

	return analysisResponse, nil
}

// explainError provides detailed explanation of an error
func (da *DebuggingAgent) explainError(ctx context.Context, request *DebuggingRequest) (*DebuggingResponse, error) {
	if request.ErrorMessage == "" {
		return nil, fmt.Errorf("error message is required for explanation")
	}

	// Build explanation prompt
	prompt := da.buildErrorExplanationPrompt(request)

	// Call LLM for explanation
	llmRequest := &llm.CompletionRequest{
		Prompt:      prompt,
		Model:       da.config.LLMModel,
		MaxTokens:   da.config.MaxTokens,
		Temperature: da.config.Temperature,
	}

	llmResponse, err := da.llmProvider.Complete(ctx, llmRequest)
	if err != nil {
		return nil, fmt.Errorf("LLM completion failed: %v", err)
	}

	// Analyze the error for context
	var detectedIssues []*DetectedIssue
	if da.config.EnableErrorAnalysis {
		issues, err := da.errorAnalyzer.AnalyzeError(request.ErrorMessage, request.Code, request.Language)
		if err == nil {
			detectedIssues = issues
		}
	}

	return &DebuggingResponse{
		DetectedIssues: detectedIssues,
		Explanation:    llmResponse.Text,
		Confidence:     0.8,
	}, nil
}

// generatePreventionTips generates tips to prevent similar bugs
func (da *DebuggingAgent) generatePreventionTips(ctx context.Context, request *DebuggingRequest) (*DebuggingResponse, error) {
	// First analyze to understand the issues
	analysisResponse, err := da.analyzeBug(ctx, request)
	if err != nil {
		return nil, err
	}

	// Generate prevention tips based on detected issues
	var preventionTips []*PreventionTip

	for _, issue := range analysisResponse.DetectedIssues {
		tips := da.generatePreventionTipsForIssue(issue, request.Language)
		preventionTips = append(preventionTips, tips...)
	}

	// Add general prevention tips
	generalTips := da.generateGeneralPreventionTips(request.Language)
	preventionTips = append(preventionTips, generalTips...)

	analysisResponse.PreventionTips = preventionTips

	return analysisResponse, nil
}

// performSecurityReview performs security-focused code review
func (da *DebuggingAgent) performSecurityReview(ctx context.Context, request *DebuggingRequest) (*DebuggingResponse, error) {
	// Focus on security-related patterns and vulnerabilities
	securityIssues := da.patternMatcher.FindSecurityVulnerabilities(request.Code, request.Language)

	// Generate security-specific fixes
	var securityFixes []*Fix
	for _, issue := range securityIssues {
		fixes := da.generateSecurityFixes(issue, request)
		securityFixes = append(securityFixes, fixes...)
	}

	// Generate security prevention tips
	securityTips := da.generateSecurityPreventionTips(request.Language)

	return &DebuggingResponse{
		DetectedIssues: securityIssues,
		Fixes:          securityFixes,
		PreventionTips: securityTips,
		Confidence:     da.calculateIssuesConfidence(securityIssues),
	}, nil
}

// analyzePerformance analyzes performance issues
func (da *DebuggingAgent) analyzePerformance(ctx context.Context, request *DebuggingRequest) (*DebuggingResponse, error) {
	// Focus on performance-related patterns
	performanceIssues := da.patternMatcher.FindPerformanceIssues(request.Code, request.Language)

	// Generate performance fixes
	var performanceFixes []*Fix
	for _, issue := range performanceIssues {
		fixes := da.generatePerformanceFixes(issue, request)
		performanceFixes = append(performanceFixes, fixes...)
	}

	// Generate performance tips
	performanceTips := da.generatePerformancePreventionTips(request.Language)

	return &DebuggingResponse{
		DetectedIssues: performanceIssues,
		Fixes:          performanceFixes,
		PreventionTips: performanceTips,
		Confidence:     da.calculateIssuesConfidence(performanceIssues),
	}, nil
}

// Helper methods

func (da *DebuggingAgent) parseDebuggingRequest(request *AgentRequest) (*DebuggingRequest, error) {
	// Try to parse from parameters first
	if params, ok := request.Parameters["debugging_request"].(*DebuggingRequest); ok {
		return params, nil
	}

	// Parse from query and context
	debugRequest := &DebuggingRequest{
		Type:     da.inferDebuggingType(request.Intent.Type),
		Language: da.inferLanguage(request.Context),
	}

	// Extract code from context
	if request.Context != nil && request.Context.SelectedText != "" {
		debugRequest.Code = request.Context.SelectedText
	}

	// Try to extract error message from query
	if strings.Contains(strings.ToLower(request.Query), "error") {
		debugRequest.ErrorMessage = request.Query
	}
	// Set default language if not inferred
	if debugRequest.Language == "" {
		debugRequest.Language = "unknown"
	}
	return debugRequest, nil
}

func (da *DebuggingAgent) inferDebuggingType(intentType IntentType) DebuggingRequestType {
	switch intentType {
	case IntentBugIdentification:
		return DebugTypeAnalyze
	case IntentBugFix:
		return DebugTypeFixSuggestion
	case IntentCodeReview:
		return DebugTypeReview
	case IntentPerformanceAnalysis:
		return DebugTypePerformance
	default:
		return DebugTypeAnalyze
	}
}

func (da *DebuggingAgent) inferLanguage(context *RequestContext) string {
	if context == nil {
		return ""
	}
	if context.ProjectLanguage != "" {
		return context.ProjectLanguage
	}

	if context.CurrentFile != "" {
		return da.inferLanguageFromFile(context.CurrentFile)
	}

	return ""
}

func (da *DebuggingAgent) inferLanguageFromFile(filename string) string {
	ext := strings.ToLower(strings.TrimPrefix(filepath.Ext(filename), "."))
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

	return "unknown"
}

func (da *DebuggingAgent) generateBugAnalysis(request *DebuggingRequest, issues []*DetectedIssue) *BugAnalysis {
	if len(issues) == 0 {
		return nil
	}
	// Find the most severe issue as the primary focus
	var primaryIssue *DetectedIssue
	for _, issue := range issues {
		if primaryIssue == nil || issue.Severity > primaryIssue.Severity {
			primaryIssue = issue
		}
	}

	analysis := &BugAnalysis{
		Summary:         da.generateAnalysisSummary(issues),
		RootCause:       da.inferRootCause(primaryIssue, request),
		ComplexityScore: da.calculateComplexityScore(issues),
		AnalysisMethod:  []string{"static_analysis", "pattern_matching"},
		Evidence:        da.collectEvidence(issues),
	}

	// Add impact assessment
	analysis.ImpactAssessment = &ImpactAssessment{
		Severity:           primaryIssue.Severity,
		Scope:              da.assessScope(issues, request),
		AffectedComponents: da.identifyAffectedComponents(issues),
		UserImpact:         da.assessUserImpact(primaryIssue),
		BusinessImpact:     da.assessBusinessImpact(primaryIssue),
		TechnicalDebt:      da.calculateTechnicalDebt(issues),
	}

	return analysis
}

func (da *DebuggingAgent) generateFixes(ctx context.Context, request *DebuggingRequest, issues []*DetectedIssue) []*Fix {
	var fixes []*Fix
	for _, issue := range issues {
		issueFixes := da.generateFixesForIssue(issue, request)
		fixes = append(fixes, issueFixes...)
	}

	return fixes
}

func (da *DebuggingAgent) generateFixesForIssue(issue *DetectedIssue, request *DebuggingRequest) []*Fix {
	var fixes []*Fix
	switch issue.Type {
	case IssueTypeLogic:
		fixes = append(fixes, da.generateLogicErrorFixes(issue, request)...)
	case IssueTypeSyntax:
		fixes = append(fixes, da.generateSyntaxErrorFixes(issue, request)...)
	case IssueTypeRuntime:
		fixes = append(fixes, da.generateRuntimeErrorFixes(issue, request)...)
	case IssueTypeMemory:
		fixes = append(fixes, da.generateMemoryErrorFixes(issue, request)...)
	case IssueTypeSecurity:
		fixes = append(fixes, da.generateSecurityFixes(issue, request)...)
	case IssueTypePerformance:
		fixes = append(fixes, da.generatePerformanceFixes(issue, request)...)
	}

	return fixes
}

func (da *DebuggingAgent) generateLogicErrorFixes(issue *DetectedIssue, request *DebuggingRequest) []*Fix {
	var fixes []*Fix
	// Quick fix: Add null checks
	if strings.Contains(strings.ToLower(issue.Description), "null") {
		fix := &Fix{
			ID:              da.generateFixID(),
			Title:           "Add Null Check",
			Description:     "Add null/undefined check to prevent null pointer errors",
			Type:            FixTypeQuick,
			Approach:        "defensive_programming",
			Confidence:      0.8,
			EstimatedEffort: "2-5 minutes",
			RiskAssessment: &RiskAssessment{
				RiskLevel:       RiskLow,
				TestingNeeded:   true,
				PotentialIssues: []string{"May change program flow"},
				Mitigations:     []string{"Add comprehensive tests"},
			},
		}

		// Generate code change based on language
		codeChange := da.generateNullCheckCodeChange(request.Language, issue.Location)
		if codeChange != nil {
			fix.CodeChanges = []*CodeChange{codeChange}
		}

		fixes = append(fixes, fix)
	}

	// Refactoring fix: Extract method
	if strings.Contains(strings.ToLower(issue.Description), "complex") {
		fix := &Fix{
			ID:              da.generateFixID(),
			Title:           "Extract Method",
			Description:     "Break down complex logic into smaller, manageable methods",
			Type:            FixTypeRefactor,
			Approach:        "method_extraction",
			Confidence:      0.7,
			EstimatedEffort: "15-30 minutes",
			RiskAssessment: &RiskAssessment{
				RiskLevel:       RiskMedium,
				TestingNeeded:   true,
				PotentialIssues: []string{"May introduce new bugs", "Requires careful refactoring"},
				Mitigations:     []string{"Comprehensive testing", "Incremental refactoring"},
			},
		}
		fixes = append(fixes, fix)
	}

	return fixes
}

func (da *DebuggingAgent) generateSyntaxErrorFixes(issue *DetectedIssue, request *DebuggingRequest) []*Fix {
	var fixes []*Fix
	fix := &Fix{
		ID:              da.generateFixID(),
		Title:           "Fix Syntax Error",
		Description:     "Correct the syntax error in the code",
		Type:            FixTypeQuick,
		Approach:        "syntax_correction",
		Confidence:      0.9,
		EstimatedEffort: "1-2 minutes",
		RiskAssessment: &RiskAssessment{
			RiskLevel:       RiskLow,
			TestingNeeded:   false,
			PotentialIssues: []string{},
			Mitigations:     []string{"IDE validation"},
		},
	}

	// Generate specific syntax fix based on the error pattern
	codeChange := da.generateSyntaxFixCodeChange(issue, request.Language)
	if codeChange != nil {
		fix.CodeChanges = []*CodeChange{codeChange}
	}

	fixes = append(fixes, fix)
	return fixes
}

func (da *DebuggingAgent) generateRuntimeErrorFixes(issue *DetectedIssue, request *DebuggingRequest) []*Fix {
	var fixes []*Fix
	// Exception handling fix
	fix := &Fix{
		ID:              da.generateFixID(),
		Title:           "Add Exception Handling",
		Description:     "Wrap potentially failing code in try-catch blocks",
		Type:            FixTypeQuick,
		Approach:        "exception_handling",
		Confidence:      0.8,
		EstimatedEffort: "5-10 minutes",
		RiskAssessment: &RiskAssessment{
			RiskLevel:       RiskLow,
			TestingNeeded:   true,
			PotentialIssues: []string{"May hide underlying issues"},
			Mitigations:     []string{"Proper logging", "Meaningful error messages"},
		},
	}

	codeChange := da.generateExceptionHandlingCodeChange(request.Language, issue.Location)
	if codeChange != nil {
		fix.CodeChanges = []*CodeChange{codeChange}
	}

	fixes = append(fixes, fix)
	return fixes
}

func (da *DebuggingAgent) generateMemoryErrorFixes(issue *DetectedIssue, request *DebuggingRequest) []*Fix {
	var fixes []*Fix
	if request.Language == "c" || request.Language == "cpp" {
		// Memory leak fix
		fix := &Fix{
			ID:              da.generateFixID(),
			Title:           "Fix Memory Leak",
			Description:     "Ensure proper memory deallocation",
			Type:            FixTypeQuick,
			Approach:        "memory_management",
			Confidence:      0.7,
			EstimatedEffort: "5-15 minutes",
			RiskAssessment: &RiskAssessment{
				RiskLevel:       RiskMedium,
				TestingNeeded:   true,
				PotentialIssues: []string{"Double-free errors", "Use-after-free"},
				Mitigations:     []string{"Memory debugging tools", "Static analysis"},
			},
		}
		fixes = append(fixes, fix)
	}

	return fixes
}

func (da *DebuggingAgent) generateSecurityFixes(issue *DetectedIssue, request *DebuggingRequest) []*Fix {
	var fixes []*Fix
	if strings.Contains(strings.ToLower(issue.Description), "injection") {
		fix := &Fix{
			ID:              da.generateFixID(),
			Title:           "Prevent Injection Attack",
			Description:     "Use parameterized queries or proper input validation",
			Type:            FixTypeQuick,
			Approach:        "input_validation",
			Confidence:      0.9,
			EstimatedEffort: "10-20 minutes",
			RiskAssessment: &RiskAssessment{
				RiskLevel:       RiskHigh,
				TestingNeeded:   true,
				PotentialIssues: []string{"May break existing functionality"},
				Mitigations:     []string{"Security testing", "Code review"},
			},
		}
		fixes = append(fixes, fix)
	}

	if strings.Contains(strings.ToLower(issue.Description), "xss") {
		fix := &Fix{
			ID:              da.generateFixID(),
			Title:           "Prevent XSS Attack",
			Description:     "Properly escape output and validate input",
			Type:            FixTypeQuick,
			Approach:        "output_escaping",
			Confidence:      0.85,
			EstimatedEffort: "5-15 minutes",
			RiskAssessment: &RiskAssessment{
				RiskLevel:       RiskHigh,
				TestingNeeded:   true,
				PotentialIssues: []string{"May affect display formatting"},
				Mitigations:     []string{"Security testing", "Manual verification"},
			},
		}
		fixes = append(fixes, fix)
	}

	return fixes
}

func (da *DebuggingAgent) generatePerformanceFixes(issue *DetectedIssue, request *DebuggingRequest) []*Fix {
	var fixes []*Fix
	if strings.Contains(strings.ToLower(issue.Description), "loop") {
		fix := &Fix{
			ID:              da.generateFixID(),
			Title:           "Optimize Loop Performance",
			Description:     "Improve loop efficiency or replace with more efficient algorithm",
			Type:            FixTypeRefactor,
			Approach:        "algorithm_optimization",
			Confidence:      0.7,
			EstimatedEffort: "15-45 minutes",
			RiskAssessment: &RiskAssessment{
				RiskLevel:       RiskMedium,
				TestingNeeded:   true,
				PotentialIssues: []string{"May change program behavior", "Complexity increase"},
				Mitigations:     []string{"Performance testing", "Benchmarking"},
			},
		}
		fixes = append(fixes, fix)
	}

	return fixes
}

func (da *DebuggingAgent) generateLLMFixes(ctx context.Context, request *DebuggingRequest, issues []*DetectedIssue) ([]*Fix, error) {
	if len(issues) == 0 {
		return nil, nil
	}
	// Build fix generation prompt
	prompt := da.buildFixGenerationPrompt(request, issues)

	// Call LLM
	llmRequest := &llm.CompletionRequest{
		Prompt:      prompt,
		Model:       da.config.LLMModel,
		MaxTokens:   da.config.MaxTokens,
		Temperature: da.config.Temperature,
	}

	llmResponse, err := da.llmProvider.Complete(ctx, llmRequest)
	if err != nil {
		return nil, fmt.Errorf("LLM completion failed: %v", err)
	}

	// Parse LLM response to extract fixes
	fixes := da.parseLLMFixResponse(llmResponse.Text, request)

	return fixes, nil
}

func (da *DebuggingAgent) buildErrorExplanationPrompt(request *DebuggingRequest) string {
	var prompt strings.Builder
	prompt.WriteString("Explain the following error in detail:\n\n")
	prompt.WriteString(fmt.Sprintf("Error: %s\n\n", request.ErrorMessage))

	if request.Code != "" {
		prompt.WriteString("Code context:\n```")
		prompt.WriteString(request.Language)
		prompt.WriteString("\n")
		prompt.WriteString(request.Code)
		prompt.WriteString("\n```\n\n")
	}

	if request.StackTrace != "" {
		prompt.WriteString("Stack trace:\n")
		prompt.WriteString(request.StackTrace)
		prompt.WriteString("\n\n")
	}

	prompt.WriteString("Please provide:\n")
	prompt.WriteString("1. What this error means\n")
	prompt.WriteString("2. Why it occurs\n")
	prompt.WriteString("3. Common causes\n")
	prompt.WriteString("4. How to fix it\n")
	prompt.WriteString("5. How to prevent it in the future\n")

	return prompt.String()
}

func (da *DebuggingAgent) buildFixGenerationPrompt(request *DebuggingRequest, issues []*DetectedIssue) string {
	var prompt strings.Builder
	prompt.WriteString("Generate fix suggestions for the following code issues:\n\n")

	prompt.WriteString("Code:\n```")
	prompt.WriteString(request.Language)
	prompt.WriteString("\n")
	prompt.WriteString(request.Code)
	prompt.WriteString("\n```\n\n")

	prompt.WriteString("Detected Issues:\n")
	for i, issue := range issues {
		prompt.WriteString(fmt.Sprintf("%d. %s (%s): %s\n", i+1, issue.Title, issue.Type, issue.Description))
	}
	prompt.WriteString("\n")

	if request.ErrorMessage != "" {
		prompt.WriteString(fmt.Sprintf("Error Message: %s\n\n", request.ErrorMessage))
	}

	prompt.WriteString("For each issue, provide:\n")
	prompt.WriteString("1. A specific fix with code changes\n")
	prompt.WriteString("2. Explanation of why this fix works\n")
	prompt.WriteString("3. Risk assessment and testing recommendations\n")
	prompt.WriteString("4. Alternative approaches if applicable\n\n")

	prompt.WriteString("Format fixes clearly with code examples.")

	return prompt.String()
}

func (da *DebuggingAgent) generateExplanation(request *DebuggingRequest, issues []*DetectedIssue, analysis *BugAnalysis) string {
	if len(issues) == 0 {
		return "No issues detected in the provided code."
	}
	var explanation strings.Builder

	explanation.WriteString("## Bug Analysis Summary\n\n")

	if analysis != nil && analysis.Summary != "" {
		explanation.WriteString(analysis.Summary)
		explanation.WriteString("\n\n")

		if analysis.RootCause != "" {
			explanation.WriteString("**Root Cause**: ")
			explanation.WriteString(analysis.RootCause)
			explanation.WriteString("\n\n")
		}
	}

	explanation.WriteString("## Detected Issues\n\n")

	for i, issue := range issues {
		explanation.WriteString(fmt.Sprintf("### %d. %s (%s)\n", i+1, issue.Title, issue.Severity))
		explanation.WriteString(issue.Description)
		explanation.WriteString("\n\n")

		if issue.Location != nil {
			explanation.WriteString(fmt.Sprintf("**Location**: %s:%d\n", issue.Location.FilePath, issue.Location.LineStart))
		}

		explanation.WriteString(fmt.Sprintf("**Confidence**: %.1f%%\n\n", issue.Confidence*100))
	}

	return explanation.String()
}

// Helper methods for code generation
func (da *DebuggingAgent) generateNullCheckCodeChange(language string, location *IssueLocation) *CodeChange {
	if location == nil {
		return nil
	}
	var newCode string
	switch language {
	case "go":
		newCode = "if variable != nil {\n    // existing code\n}"
	case "javascript", "typescript":
		newCode = "if (variable != null) {\n    // existing code\n}"
	case "python":
		newCode = "if variable is not None:\n    # existing code"
	case "java":
		newCode = "if (variable != null) {\n    // existing code\n}"
	default:
		newCode = "// Add null check here"
	}

	return &CodeChange{
		Type:       ChangeTypeAdd,
		LineNumber: location.LineStart,
		NewCode:    newCode,
		Reason:     "Add null check to prevent null pointer exception",
		Impact:     ImpactLow,
	}
}

func (da *DebuggingAgent) generateSyntaxFixCodeChange(issue *DetectedIssue, language string) *CodeChange {
	// This would analyze the specific syntax error and generate appropriate fix
	// Simplified implementation
	return &CodeChange{
		Type:       ChangeTypeModify,
		LineNumber: issue.Location.LineStart,
		NewCode:    "// Fixed syntax error",
		Reason:     "Correct syntax error",
		Impact:     ImpactLow,
	}
}

func (da *DebuggingAgent) generateExceptionHandlingCodeChange(language string, location *IssueLocation) *CodeChange {
	if location == nil {
		return nil
	}
	var newCode string
	switch language {
	case "go":
		newCode = "if err != nil {\n    return err\n}"
	case "javascript", "typescript":
		newCode = "try {\n    // risky code\n} catch (error) {\n    console.error(error);\n}"
	case "python":
		newCode = "try:\n    # risky code\nexcept Exception as e:\n    print(f\"Error: {e}\")"
	case "java":
		newCode = "try {\n    // risky code\n} catch (Exception e) {\n    logger.error(\"Error\", e);\n}"
	default:
		newCode = "// Add exception handling here"
	}

	return &CodeChange{
		Type:       ChangeTypeAdd,
		LineNumber: location.LineStart,
		NewCode:    newCode,
		Reason:     "Add exception handling to prevent runtime errors",
		Impact:     ImpactMedium,
	}
}

// Prevention tips generation
func (da *DebuggingAgent) generatePreventionTipsForIssue(issue *DetectedIssue, language string) []*PreventionTip {
	var tips []*PreventionTip
	switch issue.Type {
	case IssueTypeLogic:
		tips = append(tips, &PreventionTip{
			Category:       "Logic Errors",
			Title:          "Use Unit Testing",
			Description:    "Write comprehensive unit tests to catch logic errors early",
			Implementation: "Implement test-driven development (TDD) practices",
			Tools:          []string{"Jest", "PyTest", "JUnit", "Go Test"},
		})

	case IssueTypeSecurity:
		tips = append(tips, &PreventionTip{
			Category:       "Security",
			Title:          "Input Validation",
			Description:    "Always validate and sanitize user inputs",
			Implementation: "Use established validation libraries and frameworks",
			Tools:          []string{"OWASP validation", "Joi", "Cerberus"},
		})

	case IssueTypePerformance:
		tips = append(tips, &PreventionTip{
			Category:       "Performance",
			Title:          "Profile Your Code",
			Description:    "Regularly profile your application to identify performance bottlenecks",
			Implementation: "Use profiling tools and performance monitoring",
			Tools:          []string{"pprof", "Chrome DevTools", "py-spy"},
		})
	}

	return tips
}

func (da *DebuggingAgent) generateGeneralPreventionTips(language string) []*PreventionTip {
	var tips []*PreventionTip
	tips = append(tips, &PreventionTip{
		Category:       "Code Quality",
		Title:          "Use Static Analysis Tools",
		Description:    "Employ static analysis tools to catch issues before runtime",
		Implementation: "Integrate linters and static analyzers into your CI/CD pipeline",
		Tools:          da.getStaticAnalysisTools(language),
	})

	tips = append(tips, &PreventionTip{
		Category:       "Development Process",
		Title:          "Code Reviews",
		Description:    "Implement thorough code review processes",
		Implementation: "Require peer reviews for all code changes",
		Tools:          []string{"GitHub PR", "GitLab MR", "Gerrit"},
	})

	return tips
}

func (da *DebuggingAgent) generateSecurityPreventionTips(language string) []*PreventionTip {
	return []*PreventionTip{
		{
			Category:       "Security",
			Title:          "Secure Coding Practices",
			Description:    "Follow OWASP guidelines and secure coding standards",
			Implementation: "Regular security training and code audits",
			Tools:          []string{"SonarQube", "CodeQL", "Bandit"},
		},
		{
			Category:       "Security",
			Title:          "Dependency Management",
			Description:    "Keep dependencies updated and scan for vulnerabilities",
			Implementation: "Use dependency scanning tools in CI/CD",
			Tools:          []string{"npm audit", "pip-audit", "OWASP Dependency Check"},
		},
	}
}

func (da *DebuggingAgent) generatePerformancePreventionTips(language string) []*PreventionTip {
	return []*PreventionTip{
		{
			Category:       "Performance",
			Title:          "Performance Testing",
			Description:    "Implement regular performance testing and monitoring",
			Implementation: "Set up performance benchmarks and regression testing",
			Tools:          []string{"Apache Bench", "JMeter", "Locust"},
		},
		{
			Category:       "Performance",
			Title:          "Profiling Integration",
			Description:    "Integrate profiling tools into development workflow",
			Implementation: "Profile during development and before releases",
			Tools:          da.getProfilingTools(language),
		},
	}
}

// Utility methods
func (da *DebuggingAgent) getStaticAnalysisTools(language string) []string {
	tools := map[string][]string{
		"go":         {"golangci-lint", "staticcheck", "gosec"},
		"python":     {"pylint", "flake8", "mypy", "bandit"},
		"javascript": {"ESLint", "JSHint", "SonarJS"},
		"typescript": {"TSLint", "ESLint", "SonarTS"},
		"java":       {"SpotBugs", "PMD", "Checkstyle", "SonarJava"},
		"c":          {"Clang Static Analyzer", "PC-lint", "Cppcheck"},
		"cpp":        {"Clang Static Analyzer", "PC-lint", "Cppcheck"},
	}
	if langTools, exists := tools[language]; exists {
		return langTools
	}

	return []string{"SonarQube", "CodeClimate"}
}

func (da *DebuggingAgent) getProfilingTools(language string) []string {
	tools := map[string][]string{
		"go":         {"pprof", "go tool trace"},
		"python":     {"cProfile", "py-spy", "memory_profiler"},
		"javascript": {"Chrome DevTools", "Node.js profiler"},
		"java":       {"JProfiler", "VisualVM", "Java Flight Recorder"},
		"c":          {"gprof", "Valgrind", "perf"},
		"cpp":        {"gprof", "Valgrind", "perf"},
	}
	if langTools, exists := tools[language]; exists {
		return langTools
	}

	return []string{"Generic profiling tools"}
}

func (da *DebuggingAgent) generateFixID() string {
	return fmt.Sprintf("fix_%d", time.Now().UnixNano())
}

func (da *DebuggingAgent) calculateConfidence(request *DebuggingRequest, response *DebuggingResponse) float64 {
	confidence := 0.7 // Base confidence
	// Adjust based on analysis completeness
	if response.Analysis != nil {
		confidence += 0.1
	}

	// Adjust based on number of detected issues
	if len(response.DetectedIssues) > 0 {
		avgIssueConfidence := da.calculateIssuesConfidence(response.DetectedIssues)
		confidence += float64(avgIssueConfidence) * 0.2
	}

	// Adjust based on fix quality
	if len(response.Fixes) > 0 {
		avgFixConfidence := da.calculateFixesConfidence(response.Fixes)
		confidence += float64(avgFixConfidence) * 0.1
	}

	// Cap at 1.0
	if confidence > 1.0 {
		confidence = 1.0
	}

	return confidence
}

func (da *DebuggingAgent) calculateIssuesConfidence(issues []*DetectedIssue) float32 {
	if len(issues) == 0 {
		return 0.0
	}
	var total float32
	for _, issue := range issues {
		total += issue.Confidence
	}

	return total / float32(len(issues))
}

func (da *DebuggingAgent) calculateFixesConfidence(fixes []*Fix) float32 {
	if len(fixes) == 0 {
		return 0.0
	}
	var total float32
	for _, fix := range fixes {
		total += fix.Confidence
	}

	return total / float32(len(fixes))
}

// Analysis helper methods
func (da *DebuggingAgent) generateAnalysisSummary(issues []*DetectedIssue) string {
	if len(issues) == 0 {
		return "No issues detected"
	}
	issueCount := len(issues)
	severityCount := make(map[IssueSeverity]int)

	for _, issue := range issues {
		severityCount[issue.Severity]++
	}

	summary := fmt.Sprintf("Detected %d issue(s): ", issueCount)
	var parts []string

	for severity, count := range severityCount {
		if count > 0 {
			parts = append(parts, fmt.Sprintf("%d %s", count, severity))
		}
	}

	summary += strings.Join(parts, ", ")
	return summary
}

func (da *DebuggingAgent) inferRootCause(issue *DetectedIssue, request *DebuggingRequest) string {
	if issue == nil {
		return "Unable to determine root cause"
	}
	switch issue.Type {
	case IssueTypeLogic:
		return "Logic error in algorithm or business logic implementation"
	case IssueTypeSyntax:
		return "Syntax error preventing compilation or interpretation"
	case IssueTypeRuntime:
		return "Runtime condition causing unexpected behavior or crashes"
	case IssueTypeMemory:
		return "Memory management issue causing leaks or corruption"
	case IssueTypeSecurity:
		return "Security vulnerability allowing potential exploitation"
	case IssueTypePerformance:
		return "Inefficient algorithm or resource usage"
	default:
		return issue.Description
	}
}

func (da *DebuggingAgent) calculateComplexityScore(issues []*DetectedIssue) float32 {
	if len(issues) == 0 {
		return 0.0
	}
	var totalComplexity float32
	for _, issue := range issues {
		switch issue.Severity {
		case SeverityCritical:
			totalComplexity += 1.0
		case SeverityError:
			totalComplexity += 0.8
		case SeverityWarning:
			totalComplexity += 0.5
		case SeverityInfo:
			totalComplexity += 0.2
		}
	}

	return totalComplexity / float32(len(issues))
}

func (da *DebuggingAgent) collectEvidence(issues []*DetectedIssue) []*Evidence {
	var evidence []*Evidence
	for _, issue := range issues {
		if issue.Evidence != nil {
			evidence = append(evidence, issue.Evidence...)
		}
	}

	return evidence
}

func (da *DebuggingAgent) assessScope(issues []*DetectedIssue, request *DebuggingRequest) string {
	if len(issues) == 1 {
		return "localized"
	} else if len(issues) <= 3 {
		return "moderate"
	}
	return "widespread"
}

func (da *DebuggingAgent) identifyAffectedComponents(issues []*DetectedIssue) []string {
	components := make(map[string]bool)
	for _, issue := range issues {
		if issue.Location != nil {
			if issue.Location.Function != "" {
				components[issue.Location.Function] = true
			}
			if issue.Location.Class != "" {
				components[issue.Location.Class] = true
			}
			if issue.Location.FilePath != "" {
				components[issue.Location.FilePath] = true
			}
		}
	}

	var result []string
	for component := range components {
		result = append(result, component)
	}
	sort.Strings(result)
	return result
}

func (da *DebuggingAgent) assessUserImpact(issue *DetectedIssue) string {
	switch issue.Severity {
	case SeverityCritical:
		return "High - Application crashes or data corruption"
	case SeverityError:
		return "Medium - Feature malfunction or incorrect results"
	case SeverityWarning:
		return "Low - Minor inconvenience or degraded performance"
	case SeverityInfo:
		return "Minimal - No direct user impact"
	default:
		return "Unknown impact"
	}
}

func (da *DebuggingAgent) assessBusinessImpact(issue *DetectedIssue) string {
	switch issue.Severity {
	case SeverityCritical:
		return "High - Revenue loss, security breach, or compliance violation"
	case SeverityError:
		return "Medium - Customer dissatisfaction or operational inefficiency"
	case SeverityWarning:
		return "Low - Minor operational impact"
	case SeverityInfo:
		return "Minimal - No significant business impact"
	default:
		return "Unknown business impact"
	}
}

func (da *DebuggingAgent) calculateTechnicalDebt(issues []*DetectedIssue) float32 {
	var debt float32
	for _, issue := range issues {
		switch issue.Type {
		case IssueTypeSecurity:
			debt += 0.9
		case IssueTypeReliability:
			debt += 0.7
		case IssueTypePerformance:
			debt += 0.5
		case IssueTypeLogic:
			debt += 0.6
		default:
			debt += 0.3
		}
	}

	return debt / float32(len(issues))
}

func (da *DebuggingAgent) parseLLMFixResponse(response string, request *DebuggingRequest) []*Fix {
	// This would implement sophisticated parsing of LLM response
	// For now, a simplified implementation
	var fixes []*Fix

	// Look for fix patterns in the response
	lines := strings.Split(response, "\n")
	var currentFix *Fix

	for _, line := range lines {
		line = strings.TrimSpace(line)

		// Look for fix headers
		if strings.Contains(strings.ToLower(line), "fix") && strings.Contains(line, ":") {
			if currentFix != nil {
				fixes = append(fixes, currentFix)
			}

			currentFix = &Fix{
				ID:              da.generateFixID(),
				Title:           line,
				Type:            FixTypeQuick,
				Confidence:      0.7,
				EstimatedEffort: "5-15 minutes",
				RiskAssessment: &RiskAssessment{
					RiskLevel:     RiskLow,
					TestingNeeded: true,
				},
			}
		} else if currentFix != nil && line != "" {
			// Accumulate description
			if currentFix.Description == "" {
				currentFix.Description = line
			} else {
				currentFix.Description += " " + line
			}
		}
	}

	if currentFix != nil {
		fixes = append(fixes, currentFix)
	}

	return fixes
}

// Component initialization
func (da *DebuggingAgent) initializeCapabilities() {
	da.capabilities = &AgentCapabilities{
		AgentType: AgentTypeDebugging,
		SupportedIntents: []IntentType{
			IntentBugIdentification,
			IntentBugFix,
			IntentCodeReview,
			IntentPerformanceAnalysis,
		},
		SupportedLanguages: []string{
			"go", "python", "javascript", "typescript", "java",
			"cpp", "c", "csharp", "ruby", "php", "rust",
		},
		SupportedFileTypes: []string{
			"go", "py", "js", "ts", "java", "cpp", "c", "cs", "rb", "php", "rs",
		},
		MaxContextSize:    4096,
		SupportsStreaming: false,
		SupportsAsync:     true,
		RequiresContext:   false,
		Capabilities: map[string]interface{}{
			"bug_detection":        da.config.EnableBugDetection,
			"static_analysis":      da.config.EnableStaticAnalysis,
			"dynamic_analysis":     da.config.EnableDynamicAnalysis,
			"pattern_matching":     da.config.EnablePatternMatching,
			"error_analysis":       da.config.EnableErrorAnalysis,
			"stack_trace_analysis": da.config.EnableStackTraceAnalysis,
			"fix_generation":       da.config.EnableFixGeneration,
			"test_generation":      da.config.EnableTestGeneration,
			"severity_assessment":  da.config.EnableSeverityAssessment,
			"known_issues_check":   da.config.EnableKnownIssuesCheck,
		},
	}
}

func (da *DebuggingAgent) initializeComponents() {
	// Initialize bug detector
	da.bugDetector = NewBugDetector(da.config)
	// Initialize error analyzer
	da.errorAnalyzer = NewErrorAnalyzer(da.config)

	// Initialize stack trace analyzer
	if da.config.EnableStackTraceAnalysis {
		da.stackTraceAnalyzer = NewStackTraceAnalyzer(da.config.MaxStackTraceDepth)
	}

	// Initialize static analyzer
	if da.config.EnableStaticAnalysis {
		da.staticAnalyzer = NewStaticAnalyzer(da.config.LanguageAnalyzers)
	}

	// Initialize pattern matcher
	if da.config.EnablePatternMatching {
		da.patternMatcher = NewAntiPatternMatcher(da.config.CriticalPatterns, da.config.WarningPatterns)
	}

	// Initialize fix generator
	if da.config.EnableFixGeneration {
		da.fixGenerator = NewFixGenerator(da.llmProvider)
	}

	// Initialize solution ranker
	da.solutionRanker = NewSolutionRanker()

	// Initialize test generator
	if da.config.EnableTestGeneration {
		da.testGenerator = NewTestGenerator(da.llmProvider)
	}

	// Initialize knowledge base
	if da.config.EnableKnownIssuesCheck {
		da.knownIssuesDB = NewKnownIssuesDatabase(da.config.KnownIssuesDatabase)
	}

	// Initialize solution patterns
	da.solutionPatterns = NewSolutionPatternLibrary()
}

// Metrics methods
func (da *DebuggingAgent) updateMetrics(requestType DebuggingRequestType, success bool, duration time.Duration) {
	da.metrics.mu.Lock()
	defer da.metrics.mu.Unlock()

	da.metrics.TotalRequests++
	da.metrics.RequestsByType[requestType]++

	// Update success rate
	if da.metrics.TotalRequests == 1 {
		if success {
			da.metrics.SuccessRate = 1.0
		} else {
			da.metrics.SuccessRate = 0.0
		}
	} else {
		oldSuccessCount := int64(da.metrics.SuccessRate * float64(da.metrics.TotalRequests-1))
		if success {
			oldSuccessCount++
		}
		da.metrics.SuccessRate = float64(oldSuccessCount) / float64(da.metrics.TotalRequests)
	}

	// Update average response time
	if da.metrics.AverageResponseTime == 0 {
		da.metrics.AverageResponseTime = duration
	} else {
		da.metrics.AverageResponseTime = (da.metrics.AverageResponseTime + duration) / 2
	}

	da.metrics.LastRequest = time.Now()
}

// Required Agent interface methods
func (da *DebuggingAgent) GetCapabilities() *AgentCapabilities {
	return da.capabilities
}

func (da *DebuggingAgent) GetType() AgentType {
	return AgentTypeDebugging
}

func (da *DebuggingAgent) GetVersion() string {
	return "1.0.0"
}

func (da *DebuggingAgent) GetStatus() AgentStatus {
	da.mu.RLock()
	defer da.mu.RUnlock()
	return da.status
}

func (da *DebuggingAgent) Initialize(config interface{}) error {
	if cfg, ok := config.(*DebuggingAgentConfig); ok {
		da.config = cfg
		da.initializeComponents()
		return nil
	}
	return fmt.Errorf("invalid config type")
}

func (da *DebuggingAgent) Start() error {
	da.mu.Lock()
	defer da.mu.Unlock()
	da.status = StatusIdle
	da.logger.Info("Debugging agent started")
	return nil
}

func (da *DebuggingAgent) Stop() error {
	da.mu.Lock()
	defer da.mu.Unlock()
	da.status = StatusStopped
	da.logger.Info("Debugging agent stopped")
	return nil
}

func (da *DebuggingAgent) HealthCheck() error {
	if da.llmProvider == nil {
		return fmt.Errorf("LLM provider not configured")
	}
	// Test LLM connectivity
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*10)
	defer cancel()

	testRequest := &llm.CompletionRequest{
		Prompt:    "Analyze this code for bugs: function test() { return x; }",
		Model:     da.config.LLMModel,
		MaxTokens: 50,
	}

	_, err := da.llmProvider.Complete(ctx, testRequest)
	return err
}

func (da *DebuggingAgent) GetMetrics() *AgentMetrics {
	da.metrics.mu.RLock()
	defer da.metrics.mu.RUnlock()
	return &AgentMetrics{
		RequestsProcessed:   da.metrics.TotalRequests,
		AverageResponseTime: da.metrics.AverageResponseTime,
		SuccessRate:         da.metrics.SuccessRate,
		LastRequestAt:       da.metrics.LastRequest,
	}
}

func (da *DebuggingAgent) ResetMetrics() {
	da.metrics.mu.Lock()
	defer da.metrics.mu.Unlock()
	da.metrics = &DebuggingAgentMetrics{
		RequestsByType: make(map[DebuggingRequestType]int64),
		IssuesByType:   make(map[IssueType]int64),
	}
}

// Placeholder implementations for referenced components
type BugDetector struct {
	config *DebuggingAgentConfig
}

func NewBugDetector(config *DebuggingAgentConfig) *BugDetector {
	return &BugDetector{config: config}
}

type ErrorAnalyzer struct {
	config *DebuggingAgentConfig
}

func NewErrorAnalyzer(config *DebuggingAgentConfig) *ErrorAnalyzer {
	return &ErrorAnalyzer{config: config}
}

func (ea *ErrorAnalyzer) AnalyzeError(errorMsg, code, language string) ([]*DetectedIssue, error) {
	// Placeholder implementation
	return []*DetectedIssue{}, nil
}

type StackTraceAnalyzer struct {
	maxDepth int
}

func NewStackTraceAnalyzer(maxDepth int) *StackTraceAnalyzer {
	return &StackTraceAnalyzer{maxDepth: maxDepth}
}

func (sta *StackTraceAnalyzer) AnalyzeStackTrace(stackTrace, code string) (*StackTraceAnalysisResult, error) {
	// Placeholder implementation
	return &StackTraceAnalysisResult{}, nil
}

type StackTraceAnalysisResult struct {
	Summary   string           `json:"summary"`
	RootCause string           `json:"root_cause"`
	Issues    []*DetectedIssue `json:"issues"`
	Evidence  []*Evidence      `json:"evidence"`
}

type StaticAnalyzer struct {
	languageConfigs map[string]*LanguageAnalyzerConfig
}

func NewStaticAnalyzer(configs map[string]*LanguageAnalyzerConfig) *StaticAnalyzer {
	return &StaticAnalyzer{languageConfigs: configs}
}

func (sa *StaticAnalyzer) AnalyzeCode(code, language string) ([]*DetectedIssue, error) {
	// Placeholder implementation
	return []*DetectedIssue{}, nil
}

type DynamicAnalyzer struct{}

type AntiPatternMatcher struct {
	criticalPatterns []string
	warningPatterns  []string
}

func NewAntiPatternMatcher(critical, warning []string) *AntiPatternMatcher {
	return &AntiPatternMatcher{
		criticalPatterns: critical,
		warningPatterns:  warning,
	}
}

func (apm *AntiPatternMatcher) FindAntiPatterns(code, language string) []*DetectedIssue {
	// Placeholder implementation
	return []*DetectedIssue{}
}

func (apm *AntiPatternMatcher) FindSecurityVulnerabilities(code, language string) []*DetectedIssue {
	// Placeholder implementation
	return []*DetectedIssue{}
}

func (apm *AntiPatternMatcher) FindPerformanceIssues(code, language string) []*DetectedIssue {
	// Placeholder implementation
	return []*DetectedIssue{}
}

type FixGenerator struct {
	llmProvider llm.Provider
}

func NewFixGenerator(llmProvider llm.Provider) *FixGenerator {
	return &FixGenerator{llmProvider: llmProvider}
}

type SolutionRanker struct{}

func NewSolutionRanker() *SolutionRanker {
	return &SolutionRanker{}
}

func (sr *SolutionRanker) RankFixes(fixes []*Fix, request *DebuggingRequest) []*Fix {
	// Sort by confidence descending
	sort.Slice(fixes, func(i, j int) bool {
		return fixes[i].Confidence > fixes[j].Confidence
	})
	return fixes
}

type TestGenerator struct {
	llmProvider llm.Provider
}

func NewTestGenerator(llmProvider llm.Provider) *TestGenerator {
	return &TestGenerator{llmProvider: llmProvider}
}

func (tg *TestGenerator) GenerateTestsForFix(fix *Fix, request *DebuggingRequest) []*GeneratedTestCase {
	// Placeholder implementation
	return []*GeneratedTestCase{}
}

type KnownIssuesDatabase struct {
	databasePath string
}

func NewKnownIssuesDatabase(path string) *KnownIssuesDatabase {
	return &KnownIssuesDatabase{databasePath: path}
}

func (kid *KnownIssuesDatabase) CheckKnownIssues(code, errorMsg, language string) []*KnownIssue {
	// Placeholder implementation
	return []*KnownIssue{}
}

type KnownIssue struct {
	ID          string  `json:"id"`
	Type        string  `json:"type"`
	Severity    string  `json:"severity"`
	Title       string  `json:"title"`
	Description string  `json:"description"`
	Confidence  float32 `json:"confidence"`
}

type SolutionPatternLibrary struct{}

func NewSolutionPatternLibrary() *SolutionPatternLibrary {
	return &SolutionPatternLibrary{}
}
