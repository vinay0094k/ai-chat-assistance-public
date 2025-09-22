package agents

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/app"
	"github.com/yourusername/ai-code-assistant/internal/indexer"
	"github.com/yourusername/ai-code-assistant/internal/llm"
	"github.com/yourusername/ai-code-assistant/internal/logger"
)

// CodeReviewAgent performs comprehensive code reviews
type CodeReviewAgent struct {
	// Core components
	llmProvider    llm.Provider
	indexer        *indexer.UltraFastIndexer
	contextManager *app.ContextManager

	// Agent configuration
	config *CodeReviewAgentConfig
	logger logger.Logger

	// Capabilities
	capabilities *AgentCapabilities

	// Review components
	codeAnalyzer            *CodeAnalyzer
	styleChecker            *StyleChecker
	securityAnalyzer        *SecurityAnalyzer
	performanceAnalyzer     *PerformanceAnalyzer
	maintainabilityAnalyzer *MaintainabilityAnalyzer

	// Quality assessment
	qualityAssessor    *CodeQualityAssessor
	complexityAnalyzer *ComplexityAnalyzer
	codeSmellDetector  *CodeSmellDetector

	// Best practices
	bestPracticesChecker  *BestPracticesChecker
	designPatternAnalyzer *DesignPatternAnalyzer
	architectureAnalyzer  *ArchitectureAnalyzer

	// Statistics and monitoring
	metrics *CodeReviewAgentMetrics

	// State management
	mu     sync.RWMutex
	status AgentStatus
}

// CodeReviewAgentConfig contains code review agent configuration
type CodeReviewAgentConfig struct {
	// Review scope
	EnableCodeAnalysis          bool `json:"enable_code_analysis"`
	EnableStyleReview           bool `json:"enable_style_review"`
	EnableSecurityReview        bool `json:"enable_security_review"`
	EnablePerformanceReview     bool `json:"enable_performance_review"`
	EnableMaintainabilityReview bool `json:"enable_maintainability_review"`

	// Quality assessment
	EnableQualityAssessment  bool `json:"enable_quality_assessment"`
	EnableComplexityAnalysis bool `json:"enable_complexity_analysis"`
	EnableCodeSmellDetection bool `json:"enable_code_smell_detection"`

	// Best practices
	EnableBestPracticesCheck    bool `json:"enable_best_practices_check"`
	EnableDesignPatternAnalysis bool `json:"enable_design_pattern_analysis"`
	EnableArchitectureReview    bool `json:"enable_architecture_review"`

	// Review depth and focus
	ReviewDepth ReviewDepth       `json:"review_depth"`
	FocusAreas  []ReviewFocusArea `json:"focus_areas"`
	ReviewStyle ReviewStyle       `json:"review_style"`

	// Severity and filtering
	MinSeverityLevel         IssueSeverity `json:"min_severity_level"`
	MaxIssuesPerCategory     int           `json:"max_issues_per_category"`
	EnableAutoFixSuggestions bool          `json:"enable_auto_fix_suggestions"`

	// Customization
	CustomRules    []*CustomReviewRule    `json:"custom_rules"`
	IgnorePatterns []string               `json:"ignore_patterns"`
	StyleGuides    map[string]*StyleGuide `json:"style_guides"`

	// Language-specific settings
	LanguageSettings map[string]*LanguageReviewConfig `json:"language_settings"`

	// LLM settings
	LLMModel    string  `json:"llm_model"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float32 `json:"temperature"`

	// Performance settings
	MaxReviewTime          time.Duration `json:"max_review_time"`
	EnableParallelAnalysis bool          `json:"enable_parallel_analysis"`
}

type ReviewDepth string

const (
	ReviewDepthSurface       ReviewDepth = "surface"
	ReviewDepthStandard      ReviewDepth = "standard"
	ReviewDepthDeep          ReviewDepth = "deep"
	ReviewDepthComprehensive ReviewDepth = "comprehensive"
)

type ReviewFocusArea string

const (
	FocusAreaSecurity        ReviewFocusArea = "security"
	FocusAreaPerformance     ReviewFocusArea = "performance"
	FocusAreaMaintainability ReviewFocusArea = "maintainability"
	FocusAreaStyle           ReviewFocusArea = "style"
	FocusAreaComplexity      ReviewFocusArea = "complexity"
	FocusAreaBestPractices   ReviewFocusArea = "best_practices"
	FocusAreaArchitecture    ReviewFocusArea = "architecture"
)

type ReviewStyle string

const (
	ReviewStyleCritical     ReviewStyle = "critical"
	ReviewStyleBalanced     ReviewStyle = "balanced"
	ReviewStyleConstructive ReviewStyle = "constructive"
	ReviewStyleMentoring    ReviewStyle = "mentoring"
)

type CustomReviewRule struct {
	ID          string        `json:"id"`
	Name        string        `json:"name"`
	Description string        `json:"description"`
	Pattern     string        `json:"pattern"`
	Severity    IssueSeverity `json:"severity"`
	Category    string        `json:"category"`
	Message     string        `json:"message"`
	Suggestion  string        `json:"suggestion"`
	Languages   []string      `json:"languages,omitempty"`
}

type StyleGuide struct {
	Name              string            `json:"name"`
	IndentationType   string            `json:"indentation_type"`
	IndentationSize   int               `json:"indentation_size"`
	MaxLineLength     int               `json:"max_line_length"`
	NamingConventions map[string]string `json:"naming_conventions"`
	RequiredPatterns  []string          `json:"required_patterns"`
	ForbiddenPatterns []string          `json:"forbidden_patterns"`
}

type LanguageReviewConfig struct {
	StyleGuide       string   `json:"style_guide"`
	Linters          []string `json:"linters"`
	BestPractices    []string `json:"best_practices"`
	CommonIssues     []string `json:"common_issues"`
	SecurityRules    []string `json:"security_rules"`
	PerformanceRules []string `json:"performance_rules"`
}

// Request and response structures

type CodeReviewRequest struct {
	Code            string             `json:"code"`
	Language        string             `json:"language"`
	Context         *CodeReviewContext `json:"context,omitempty"`
	Options         *CodeReviewOptions `json:"options,omitempty"`
	ExistingIssues  []*ReviewIssue     `json:"existing_issues,omitempty"`
	PullRequestInfo *PullRequestInfo   `json:"pull_request_info,omitempty"`
}

type CodeReviewContext struct {
	FilePath       string         `json:"file_path,omitempty"`
	ProjectType    string         `json:"project_type,omitempty"`
	TargetAudience TargetAudience `json:"target_audience"`
	ReviewPurpose  ReviewPurpose  `json:"review_purpose"`
	TeamStandards  *TeamStandards `json:"team_standards,omitempty"`
	RelatedFiles   []string       `json:"related_files,omitempty"`
	Dependencies   []string       `json:"dependencies,omitempty"`
}

type TargetAudience string

const (
	AudienceJuniorDeveloper TargetAudience = "junior_developer"
	AudienceSeniorDeveloper TargetAudience = "senior_developer"
	AudienceTeamLead        TargetAudience = "team_lead"
	AudienceArchitect       TargetAudience = "architect"
	AudienceGeneral         TargetAudience = "general"
)

type ReviewPurpose string

const (
	PurposePullRequest      ReviewPurpose = "pull_request"
	PurposeCodeImprovement  ReviewPurpose = "code_improvement"
	PurposeSecurityAudit    ReviewPurpose = "security_audit"
	PurposePerformanceAudit ReviewPurpose = "performance_audit"
	PurposeMentoring        ReviewPurpose = "mentoring"
	PurposeCompliance       ReviewPurpose = "compliance"
)

type TeamStandards struct {
	CodingStandards    []string           `json:"coding_standards"`
	StyleGuide         string             `json:"style_guide"`
	SecurityPolicies   []string           `json:"security_policies"`
	PerformanceTargets map[string]float64 `json:"performance_targets"`
	QualityThresholds  map[string]float64 `json:"quality_thresholds"`
}

type CodeReviewOptions struct {
	FocusAreas             []ReviewFocusArea `json:"focus_areas,omitempty"`
	ReviewDepth            ReviewDepth       `json:"review_depth"`
	ReviewStyle            ReviewStyle       `json:"review_style"`
	IncludePositives       bool              `json:"include_positives"`
	IncludeSuggestions     bool              `json:"include_suggestions"`
	IncludeExamples        bool              `json:"include_examples"`
	GenerateFixSuggestions bool              `json:"generate_fix_suggestions"`
	PrioritizeIssues       bool              `json:"prioritize_issues"`
}

type PullRequestInfo struct {
	Title        string        `json:"title"`
	Description  string        `json:"description"`
	Author       string        `json:"author"`
	ChangedFiles []string      `json:"changed_files"`
	LinesAdded   int           `json:"lines_added"`
	LinesRemoved int           `json:"lines_removed"`
	Commits      []*CommitInfo `json:"commits,omitempty"`
}

type CommitInfo struct {
	Hash    string    `json:"hash"`
	Message string    `json:"message"`
	Author  string    `json:"author"`
	Date    time.Time `json:"date"`
}

// Response structures

type CodeReviewResponse struct {
	Summary           *ReviewSummary      `json:"summary"`
	Issues            []*ReviewIssue      `json:"issues,omitempty"`
	Positives         []*ReviewPositive   `json:"positives,omitempty"`
	Suggestions       []*ReviewSuggestion `json:"suggestions,omitempty"`
	QualityAssessment *QualityAssessment  `json:"quality_assessment,omitempty"`
	Metrics           *CodeMetrics        `json:"metrics,omitempty"`
	FixSuggestions    []*FixSuggestion    `json:"fix_suggestions,omitempty"`
	OverallRating     *OverallRating      `json:"overall_rating,omitempty"`
	Recommendations   []*Recommendation   `json:"recommendations,omitempty"`
}

type ReviewSummary struct {
	OverallAssessment  string        `json:"overall_assessment"`
	KeyFindings        []string      `json:"key_findings"`
	MajorIssues        int           `json:"major_issues"`
	MinorIssues        int           `json:"minor_issues"`
	PositiveAspects    []string      `json:"positive_aspects"`
	RecommendedActions []string      `json:"recommended_actions"`
	ReviewTime         time.Duration `json:"review_time"`
	Confidence         float32       `json:"confidence"`
}

type ReviewIssue struct {
	ID              string         `json:"id"`
	Type            IssueType      `json:"type"`
	Severity        IssueSeverity  `json:"severity"`
	Category        IssueCategory  `json:"category"`
	Title           string         `json:"title"`
	Description     string         `json:"description"`
	Location        *IssueLocation `json:"location,omitempty"`
	Code            string         `json:"code,omitempty"`
	Explanation     string         `json:"explanation"`
	Impact          string         `json:"impact"`
	Suggestion      string         `json:"suggestion"`
	Examples        []*CodeExample `json:"examples,omitempty"`
	References      []string       `json:"references,omitempty"`
	Priority        Priority       `json:"priority"`
	EstimatedEffort string         `json:"estimated_effort"`
	Tags            []string       `json:"tags,omitempty"`
}

type IssueCategory string

const (
	CategoryStyle           IssueCategory = "style"
	CategorySecurity        IssueCategory = "security"
	CategoryPerformance     IssueCategory = "performance"
	CategoryMaintainability IssueCategory = "maintainability"
	CategoryComplexity      IssueCategory = "complexity"
	CategoryBestPractices   IssueCategory = "best_practices"
	CategoryBugs            IssueCategory = "bugs"
	CategoryArchitecture    IssueCategory = "architecture"
)

type ReviewPositive struct {
	Type        PositiveType   `json:"type"`
	Title       string         `json:"title"`
	Description string         `json:"description"`
	Location    *IssueLocation `json:"location,omitempty"`
	Code        string         `json:"code,omitempty"`
	Explanation string         `json:"explanation"`
	Value       string         `json:"value"`
}

type PositiveType string

const (
	PositiveCleanCode      PositiveType = "clean_code"
	PositiveGoodPractices  PositiveType = "good_practices"
	PositiveEfficient      PositiveType = "efficient"
	PositiveWellStructured PositiveType = "well_structured"
	PositiveGoodNaming     PositiveType = "good_naming"
	PositiveWellDocumented PositiveType = "well_documented"
)

type ReviewSuggestion struct {
	Type            SuggestionType `json:"type"`
	Title           string         `json:"title"`
	Description     string         `json:"description"`
	Implementation  string         `json:"implementation"`
	Benefits        []string       `json:"benefits"`
	Drawbacks       []string       `json:"drawbacks,omitempty"`
	Priority        Priority       `json:"priority"`
	EstimatedEffort string         `json:"estimated_effort"`
	Examples        []*CodeExample `json:"examples,omitempty"`
}

type QualityAssessment struct {
	OverallScore     float32            `json:"overall_score"`
	CategoryScores   map[string]float32 `json:"category_scores"`
	Strengths        []string           `json:"strengths"`
	Weaknesses       []string           `json:"weaknesses"`
	ImprovementAreas []string           `json:"improvement_areas"`
	QualityTrends    *QualityTrends     `json:"quality_trends,omitempty"`
}

type QualityTrends struct {
	Improving []string `json:"improving"`
	Declining []string `json:"declining"`
	Stable    []string `json:"stable"`
}

type CodeMetrics struct {
	LinesOfCode          int           `json:"lines_of_code"`
	CyclomaticComplexity int           `json:"cyclomatic_complexity"`
	CognitiveComplexity  int           `json:"cognitive_complexity"`
	Maintainability      float32       `json:"maintainability"`
	TestCoverage         float32       `json:"test_coverage,omitempty"`
	DuplicationRatio     float32       `json:"duplication_ratio"`
	TechnicalDebt        time.Duration `json:"technical_debt"`
	CodeSmells           int           `json:"code_smells"`
	SecurityHotspots     int           `json:"security_hotspots"`
}

type FixSuggestion struct {
	IssueID       string    `json:"issue_id"`
	Type          FixType   `json:"type"`
	Title         string    `json:"title"`
	Description   string    `json:"description"`
	OriginalCode  string    `json:"original_code"`
	SuggestedCode string    `json:"suggested_code"`
	Explanation   string    `json:"explanation"`
	Confidence    float32   `json:"confidence"`
	RiskLevel     RiskLevel `json:"risk_level"`
	TestingAdvice string    `json:"testing_advice,omitempty"`
}

type OverallRating struct {
	Grade         Grade   `json:"grade"`
	Score         float32 `json:"score"`
	Justification string  `json:"justification"`
	ComparedTo    string  `json:"compared_to,omitempty"`
	Improvement   string  `json:"improvement,omitempty"`
}

type Grade string

const (
	GradeA Grade = "A" // Excellent
	GradeB Grade = "B" // Good
	GradeC Grade = "C" // Acceptable
	GradeD Grade = "D" // Needs improvement
	GradeF Grade = "F" // Poor
)

type Recommendation struct {
	Type        RecommendationType `json:"type"`
	Title       string             `json:"title"`
	Description string             `json:"description"`
	ActionItems []string           `json:"action_items"`
	Priority    Priority           `json:"priority"`
	Timeline    string             `json:"timeline"`
	Resources   []string           `json:"resources,omitempty"`
}

// CodeReviewAgentMetrics tracks code review performance
type CodeReviewAgentMetrics struct {
	TotalReviews            int64                   `json:"total_reviews"`
	ReviewsByLanguage       map[string]int64        `json:"reviews_by_language"`
	ReviewsByDepth          map[ReviewDepth]int64   `json:"reviews_by_depth"`
	AverageReviewTime       time.Duration           `json:"average_review_time"`
	AverageIssuesFound      float32                 `json:"average_issues_found"`
	AverageQualityScore     float32                 `json:"average_quality_score"`
	IssuesByCategory        map[IssueCategory]int64 `json:"issues_by_category"`
	IssuesBySeverity        map[IssueSeverity]int64 `json:"issues_by_severity"`
	FixSuggestionsGenerated int64                   `json:"fix_suggestions_generated"`
	PositiveFeedbackCount   int64                   `json:"positive_feedback_count"`
	LastReview              time.Time               `json:"last_review"`
	mu                      sync.RWMutex
}

// NewCodeReviewAgent creates a new code review agent
func NewCodeReviewAgent(llmProvider llm.Provider, indexer *indexer.UltraFastIndexer, config *CodeReviewAgentConfig, logger logger.Logger) *CodeReviewAgent {
	if config == nil {
		config = &CodeReviewAgentConfig{
			EnableCodeAnalysis:          true,
			EnableStyleReview:           true,
			EnableSecurityReview:        true,
			EnablePerformanceReview:     true,
			EnableMaintainabilityReview: true,
			EnableQualityAssessment:     true,
			EnableComplexityAnalysis:    true,
			EnableCodeSmellDetection:    true,
			EnableBestPracticesCheck:    true,
			EnableDesignPatternAnalysis: true,
			EnableArchitectureReview:    false, // Requires broader context
			ReviewDepth:                 ReviewDepthStandard,
			ReviewStyle:                 ReviewStyleBalanced,
			MinSeverityLevel:            SeverityInfo,
			MaxIssuesPerCategory:        20,
			EnableAutoFixSuggestions:    true,
			MaxReviewTime:               time.Minute * 5,
			EnableParallelAnalysis:      true,
			LLMModel:                    "gpt-4",
			MaxTokens:                   3072,
			Temperature:                 0.2, // Low temperature for consistent analysis
			FocusAreas: []ReviewFocusArea{
				FocusAreaSecurity, FocusAreaMaintainability,
				FocusAreaPerformance, FocusAreaStyle,
			},
			IgnorePatterns: []string{
				"node_modules/", ".git/", "vendor/", "target/",
			},
			StyleGuides:      make(map[string]*StyleGuide),
			LanguageSettings: make(map[string]*LanguageReviewConfig),
		}

		// Initialize default configurations
		config.StyleGuides = cra.getDefaultStyleGuides()
		config.LanguageSettings = cra.getDefaultLanguageSettings()
	}

	agent := &CodeReviewAgent{
		llmProvider: llmProvider,
		indexer:     indexer,
		config:      config,
		logger:      logger,
		status:      StatusIdle,
		metrics: &CodeReviewAgentMetrics{
			ReviewsByLanguage: make(map[string]int64),
			ReviewsByDepth:    make(map[ReviewDepth]int64),
			IssuesByCategory:  make(map[IssueCategory]int64),
			IssuesBySeverity:  make(map[IssueSeverity]int64),
		},
	}

	// Initialize capabilities
	agent.initializeCapabilities()

	// Initialize components
	agent.initializeComponents()

	return agent
}

// ProcessRequest processes a code review request
func (cra *CodeReviewAgent) ProcessRequest(ctx context.Context, request *AgentRequest) (*AgentResponse, error) {
	start := time.Now()
	cra.status = StatusBusy
	defer func() { cra.status = StatusIdle }()

	// Parse code review request
	reviewRequest, err := cra.parseCodeReviewRequest(request)
	if err != nil {
		return nil, fmt.Errorf("failed to parse code review request: %v", err)
	}

	// Apply timeout
	reviewCtx := ctx
	if cra.config.MaxReviewTime > 0 {
		var cancel context.CancelFunc
		reviewCtx, cancel = context.WithTimeout(ctx, cra.config.MaxReviewTime)
		defer cancel()
	}

	// Perform comprehensive code review
	reviewResponse, err := cra.performCodeReview(reviewCtx, reviewRequest)
	if err != nil {
		cra.updateMetrics(reviewRequest.Language, ReviewDepthStandard, false, time.Since(start), 0)
		return nil, fmt.Errorf("code review failed: %v", err)
	}

	// Create agent response
	response := &AgentResponse{
		RequestID:      request.ID,
		AgentType:      cra.GetType(),
		AgentVersion:   cra.GetVersion(),
		Result:         reviewResponse,
		Confidence:     cra.calculateConfidence(reviewRequest, reviewResponse),
		ProcessingTime: time.Since(start),
		CreatedAt:      time.Now(),
	}

	// Update metrics
	issuesFound := len(reviewResponse.Issues)
	cra.updateMetrics(reviewRequest.Language, reviewRequest.Options.ReviewDepth, true, time.Since(start), issuesFound)

	return response, nil
}

// performCodeReview conducts a comprehensive code review
func (cra *CodeReviewAgent) performCodeReview(ctx context.Context, request *CodeReviewRequest) (*CodeReviewResponse, error) {
	reviewStart := time.Now()

	// Initialize response
	response := &CodeReviewResponse{
		Issues:         []*ReviewIssue{},
		Positives:      []*ReviewPositive{},
		Suggestions:    []*ReviewSuggestion{},
		FixSuggestions: []*FixSuggestion{},
	}

	// Analyze code structure and quality
	codeAnalysis, err := cra.codeAnalyzer.Analyze(request.Code, request.Language)
	if err != nil {
		return nil, fmt.Errorf("code analysis failed: %v", err)
	}

	// Perform different types of reviews based on configuration and focus areas
	var reviewTasks []func() error

	if cra.shouldPerformReview(request.Options, FocusAreaStyle) && cra.config.EnableStyleReview {
		reviewTasks = append(reviewTasks, func() error {
			issues, positives := cra.performStyleReview(request, codeAnalysis)
			response.Issues = append(response.Issues, issues...)
			response.Positives = append(response.Positives, positives...)
			return nil
		})
	}

	if cra.shouldPerformReview(request.Options, FocusAreaSecurity) && cra.config.EnableSecurityReview {
		reviewTasks = append(reviewTasks, func() error {
			issues := cra.performSecurityReview(request, codeAnalysis)
			response.Issues = append(response.Issues, issues...)
			return nil
		})
	}

	if cra.shouldPerformReview(request.Options, FocusAreaPerformance) && cra.config.EnablePerformanceReview {
		reviewTasks = append(reviewTasks, func() error {
			issues, suggestions := cra.performPerformanceReview(request, codeAnalysis)
			response.Issues = append(response.Issues, issues...)
			response.Suggestions = append(response.Suggestions, suggestions...)
			return nil
		})
	}

	if cra.shouldPerformReview(request.Options, FocusAreaMaintainability) && cra.config.EnableMaintainabilityReview {
		reviewTasks = append(reviewTasks, func() error {
			issues, suggestions := cra.performMaintainabilityReview(request, codeAnalysis)
			response.Issues = append(response.Issues, issues...)
			response.Suggestions = append(response.Suggestions, suggestions...)
			return nil
		})
	}

	if cra.shouldPerformReview(request.Options, FocusAreaComplexity) && cra.config.EnableComplexityAnalysis {
		reviewTasks = append(reviewTasks, func() error {
			issues := cra.performComplexityAnalysis(request, codeAnalysis)
			response.Issues = append(response.Issues, issues...)
			return nil
		})
	}

	if cra.shouldPerformReview(request.Options, FocusAreaBestPractices) && cra.config.EnableBestPracticesCheck {
		reviewTasks = append(reviewTasks, func() error {
			issues, positives := cra.performBestPracticesReview(request, codeAnalysis)
			response.Issues = append(response.Issues, issues...)
			response.Positives = append(response.Positives, positives...)
			return nil
		})
	}

	// Execute reviews (parallel if enabled)
	if cra.config.EnableParallelAnalysis && len(reviewTasks) > 1 {
		err = cra.executeParallelReviews(ctx, reviewTasks)
	} else {
		err = cra.executeSequentialReviews(ctx, reviewTasks)
	}

	if err != nil {
		cra.logger.Warn("Some review tasks failed", "error", err)
	}

	// Apply custom rules
	if len(cra.config.CustomRules) > 0 {
		customIssues := cra.applyCustomRules(request.Code, request.Language)
		response.Issues = append(response.Issues, customIssues...)
	}

	// Filter and prioritize issues
	response.Issues = cra.filterAndPrioritizeIssues(response.Issues, request)

	// Generate quality assessment
	if cra.config.EnableQualityAssessment {
		response.QualityAssessment = cra.assessCodeQuality(request, response.Issues, codeAnalysis)
	}

	// Generate code metrics
	response.Metrics = cra.generateCodeMetrics(codeAnalysis, response.Issues)

	// Generate fix suggestions
	if cra.config.EnableAutoFixSuggestions && (request.Options == nil || request.Options.GenerateFixSuggestions) {
		response.FixSuggestions = cra.generateFixSuggestions(ctx, request, response.Issues)
	}

	// Generate overall rating
	response.OverallRating = cra.generateOverallRating(response.QualityAssessment, response.Issues)

	// Generate recommendations
	response.Recommendations = cra.generateRecommendations(request, response)

	// Create summary
	response.Summary = &ReviewSummary{
		OverallAssessment:  cra.generateOverallAssessment(response),
		KeyFindings:        cra.extractKeyFindings(response.Issues),
		MajorIssues:        cra.countIssuesBySeverity(response.Issues, []IssueSeverity{SeverityCritical, SeverityError}),
		MinorIssues:        cra.countIssuesBySeverity(response.Issues, []IssueSeverity{SeverityWarning, SeverityInfo}),
		PositiveAspects:    cra.extractPositiveAspects(response.Positives),
		RecommendedActions: cra.extractRecommendedActions(response.Recommendations),
		ReviewTime:         time.Since(reviewStart),
		Confidence:         cra.calculateReviewConfidence(response),
	}

	return response, nil
}

// Style review implementation
func (cra *CodeReviewAgent) performStyleReview(request *CodeReviewRequest, analysis *CodeAnalysisResult) ([]*ReviewIssue, []*ReviewPositive) {
	var issues []*ReviewIssue
	var positives []*ReviewPositive

	styleGuide := cra.getStyleGuideForLanguage(request.Language)

	// Check indentation
	if indentationIssues := cra.styleChecker.CheckIndentation(request.Code, styleGuide); len(indentationIssues) > 0 {
		for _, issue := range indentationIssues {
			issues = append(issues, &ReviewIssue{
				ID:              cra.generateIssueID(),
				Type:            IssueTypeStyle,
				Severity:        SeverityWarning,
				Category:        CategoryStyle,
				Title:           "Inconsistent Indentation",
				Description:     issue.Description,
				Location:        issue.Location,
				Code:            issue.Code,
				Explanation:     "Consistent indentation improves code readability and maintainability",
				Impact:          "Low - affects readability",
				Suggestion:      issue.Suggestion,
				Priority:        PriorityLow,
				EstimatedEffort: "1-5 minutes",
				Tags:            []string{"style", "indentation"},
			})
		}
	}

	// Check line length
	if lineLengthIssues := cra.styleChecker.CheckLineLength(request.Code, styleGuide.MaxLineLength); len(lineLengthIssues) > 0 {
		for _, issue := range lineLengthIssues {
			issues = append(issues, &ReviewIssue{
				ID:              cra.generateIssueID(),
				Type:            IssueTypeStyle,
				Severity:        SeverityInfo,
				Category:        CategoryStyle,
				Title:           "Line Length Exceeds Limit",
				Description:     fmt.Sprintf("Line exceeds maximum length of %d characters", styleGuide.MaxLineLength),
				Location:        issue.Location,
				Code:            issue.Code,
				Explanation:     "Long lines reduce code readability and can cause issues in code reviews",
				Impact:          "Low - affects readability",
				Suggestion:      "Consider breaking long lines or extracting complex expressions",
				Priority:        PriorityLow,
				EstimatedEffort: "2-10 minutes",
				Tags:            []string{"style", "line-length"},
			})
		}
	}
	// Check naming conventions
	if namingIssues := cra.styleChecker.CheckNamingConventions(request.Code, request.Language, styleGuide); len(namingIssues) > 0 {
		for _, issue := range namingIssues {
			issues = append(issues, &ReviewIssue{
				ID:              cra.generateIssueID(),
				Type:            IssueTypeStyle,
				Severity:        SeverityWarning,
				Category:        CategoryStyle,
				Title:           "Naming Convention Violation",
				Description:     issue.Description,
				Location:        issue.Location,
				Code:            issue.Code,
				Explanation:     "Consistent naming conventions improve code readability and team collaboration",
				Impact:          "Medium - affects code understanding",
				Suggestion:      issue.Suggestion,
				Priority:        PriorityMedium,
				EstimatedEffort: "5-15 minutes",
				Tags:            []string{"style", "naming"},
			})
		}
	}
	// Look for positive style aspects
	if cra.styleChecker.HasConsistentStyle(request.Code, styleGuide) {
		positives = append(positives, &ReviewPositive{
			Type:        PositiveCleanCode,
			Title:       "Consistent Code Style",
			Description: "Code follows consistent styling conventions",
			Explanation: "Consistent styling makes the code easier to read and maintain",
			Value:       "Improves team productivity and code maintainability",
		})
	}

	if cra.styleChecker.HasGoodNaming(request.Code, request.Language) {
		positives = append(positives, &ReviewPositive{
			Type:        PositiveGoodNaming,
			Title:       "Clear Naming",
			Description: "Variables and functions have descriptive names",
			Explanation: "Clear naming reduces the need for comments and improves code self-documentation",
			Value:       "Enhances code readability and reduces cognitive load",
		})
	}

	return issues, positives
}


// Check naming conventions
if namingIssues := cra.styleChecker.CheckNamingConventions(request.Code, request.Language, styleGuide); len(namingIssues) > 0 {
	for _, issue := range namingIssues {
		issues = append(issues, &ReviewIssue{
			ID:              cra.generateIssueID(),
			Type:            IssueTypeStyle,
			Severity:        SeverityWarning,
			Category:        CategoryStyle,
			Title:           "Naming Convention Violation",
			Description:     issue.Description,
			Location:        issue.Location,
			Code:            issue.Code,
			Explanation:     "Consistent naming conventions improve code readability and team collaboration",
			Impact:          "Medium - affects code understanding",
			Suggestion:      issue.Suggestion,
			Priority:        PriorityMedium,
			EstimatedEffort: "5-15 minutes",
			Tags:            []string{"style", "naming"},
		})
	}
}

// Look for positive style aspects
if cra.styleChecker.HasConsistentStyle(request.Code, styleGuide) {
	positives = append(positives, &ReviewPositive{
		Type:        PositiveCleanCode,
		Title:       "Consistent Code Style",
		Description: "Code follows consistent styling conventions",
		Explanation: "Consistent styling makes the code easier to read and maintain",
		Value:       "Improves team productivity and code maintainability",
	})
}

if cra.styleChecker.HasGoodNaming(request.Code, request.Language) {
	positives = append(positives, &ReviewPositive{
		Type:        PositiveGoodNaming,
		Title:       "Clear Naming",
		Description: "Variables and functions have descriptive names",
		Explanation: "Clear naming reduces the need for comments and improves code self-documentation",
		Value:       "Enhances code readability and reduces cognitive load",
	})
}

return issues, positives
}

// Security review implementation
func (cra *CodeReviewAgent) performSecurityReview(request *CodeReviewRequest, analysis *CodeAnalysisResult) []*ReviewIssue {
	var issues []*ReviewIssue
	// Check for common security vulnerabilities
	securityIssues := cra.securityAnalyzer.AnalyzeSecurity(request.Code, request.Language)

	for _, secIssue := range securityIssues {
		severity := cra.mapSecuritySeverity(secIssue.SecurityLevel)

		issue := &ReviewIssue{
			ID:              cra.generateIssueID(),
			Type:            IssueTypeSecurity,
			Severity:        severity,
			Category:        CategorySecurity,
			Title:           secIssue.Title,
			Description:     secIssue.Description,
			Location:        secIssue.Location,
			Code:            secIssue.Code,
			Explanation:     secIssue.Explanation,
			Impact:          secIssue.Impact,
			Suggestion:      secIssue.Mitigation,
			Priority:        cra.mapSeverityToPriority(severity),
			EstimatedEffort: cra.estimateSecurityFixEffort(secIssue.VulnerabilityType),
			References:      secIssue.References,
			Tags:            []string{"security", secIssue.VulnerabilityType},
		}

		// Add examples for common vulnerabilities
		if examples := cra.getSecurityExamples(secIssue.VulnerabilityType); len(examples) > 0 {
			issue.Examples = examples
		}

		issues = append(issues, issue)
	}

	return issues
}

// Performance review implementation
func (cra *CodeReviewAgent) performPerformanceReview(request *CodeReviewRequest, analysis *CodeAnalysisResult) ([]*ReviewIssue, []*ReviewSuggestion) {
	var issues []*ReviewIssue
	var suggestions []*ReviewSuggestion
	// Analyze performance issues
	perfIssues := cra.performanceAnalyzer.AnalyzePerformance(request.Code, request.Language)

	for _, perfIssue := range perfIssues {
		issue := &ReviewIssue{
			ID:              cra.generateIssueID(),
			Type:            IssueTypePerformance,
			Severity:        cra.mapPerformanceSeverity(perfIssue.ImpactLevel),
			Category:        CategoryPerformance,
			Title:           perfIssue.Title,
			Description:     perfIssue.Description,
			Location:        perfIssue.Location,
			Code:            perfIssue.Code,
			Explanation:     perfIssue.Explanation,
			Impact:          perfIssue.Impact,
			Suggestion:      perfIssue.Optimization,
			Priority:        cra.mapImpactToPriority(perfIssue.ImpactLevel),
			EstimatedEffort: perfIssue.EstimatedEffort,
			Tags:            []string{"performance", perfIssue.Category},
		}

		issues = append(issues, issue)
	}

	// Generate performance optimization suggestions
	optimizations := cra.performanceAnalyzer.SuggestOptimizations(request.Code, request.Language)

	for _, opt := range optimizations {
		suggestion := &ReviewSuggestion{
			Type:            SuggestionTypeOptimization,
			Title:           opt.Title,
			Description:     opt.Description,
			Implementation:  opt.Implementation,
			Benefits:        opt.Benefits,
			Drawbacks:       opt.Drawbacks,
			Priority:        opt.Priority,
			EstimatedEffort: opt.EstimatedEffort,
		}

		if len(opt.Examples) > 0 {
			suggestion.Examples = opt.Examples
		}

		suggestions = append(suggestions, suggestion)
	}

	return issues, suggestions
}

// Maintainability review implementation
func (cra *CodeReviewAgent) performMaintainabilityReview(request *CodeReviewRequest, analysis *CodeAnalysisResult) ([]*ReviewIssue, []*ReviewSuggestion) {
	var issues []*ReviewIssue
	var suggestions []*ReviewSuggestion
	// Check maintainability issues
	maintIssues := cra.maintainabilityAnalyzer.AnalyzeMaintainability(request.Code, request.Language, analysis)

	for _, maintIssue := range maintIssues {
		issue := &ReviewIssue{
			ID:              cra.generateIssueID(),
			Type:            IssueTypeMaintainability,
			Severity:        maintIssue.Severity,
			Category:        CategoryMaintainability,
			Title:           maintIssue.Title,
			Description:     maintIssue.Description,
			Location:        maintIssue.Location,
			Code:            maintIssue.Code,
			Explanation:     maintIssue.Explanation,
			Impact:          maintIssue.Impact,
			Suggestion:      maintIssue.Suggestion,
			Priority:        cra.mapSeverityToPriority(maintIssue.Severity),
			EstimatedEffort: maintIssue.EstimatedEffort,
			Tags:            []string{"maintainability", maintIssue.Category},
		}

		issues = append(issues, issue)
	}

	// Generate maintainability improvement suggestions
	improvements := cra.maintainabilityAnalyzer.SuggestImprovements(analysis)

	for _, improvement := range improvements {
		suggestion := &ReviewSuggestion{
			Type:            SuggestionTypeRefactoring,
			Title:           improvement.Title,
			Description:     improvement.Description,
			Implementation:  improvement.Implementation,
			Benefits:        improvement.Benefits,
			Priority:        improvement.Priority,
			EstimatedEffort: improvement.EstimatedEffort,
		}

		suggestions = append(suggestions, suggestion)
	}

	return issues, suggestions
}

// Complexity analysis implementation
func (cra *CodeReviewAgent) performComplexityAnalysis(request *CodeReviewRequest, analysis *CodeAnalysisResult) []*ReviewIssue {
	var issues []*ReviewIssue
	// Analyze cyclomatic complexity
	if analysis.CyclomaticComplexity > 10 {
		issue := &ReviewIssue{
			ID:              cra.generateIssueID(),
			Type:            IssueTypeComplexity,
			Severity:        cra.getComplexitySeverity(analysis.CyclomaticComplexity),
			Category:        CategoryComplexity,
			Title:           "High Cyclomatic Complexity",
			Description:     fmt.Sprintf("Cyclomatic complexity is %d, which exceeds the recommended threshold of 10", analysis.CyclomaticComplexity),
			Explanation:     "High cyclomatic complexity makes code harder to understand, test, and maintain",
			Impact:          "Medium to High - affects testability and maintainability",
			Suggestion:      "Consider breaking down complex functions into smaller, more focused functions",
			Priority:        cra.getComplexityPriority(analysis.CyclomaticComplexity),
			EstimatedEffort: cra.getComplexityFixEffort(analysis.CyclomaticComplexity),
			Tags:            []string{"complexity", "cyclomatic"},
		}

		issues = append(issues, issue)
	}

	// Check for deeply nested code
	if nestingLevel := cra.complexityAnalyzer.GetMaxNestingLevel(request.Code); nestingLevel > 4 {
		issue := &ReviewIssue{
			ID:              cra.generateIssueID(),
			Type:            IssueTypeComplexity,
			Severity:        SeverityWarning,
			Category:        CategoryComplexity,
			Title:           "Deep Nesting",
			Description:     fmt.Sprintf("Code has nesting level of %d, which exceeds the recommended maximum of 4", nestingLevel),
			Explanation:     "Deep nesting makes code harder to read and understand",
			Impact:          "Medium - affects code readability",
			Suggestion:      "Consider using early returns, extracting methods, or guard clauses to reduce nesting",
			Priority:        PriorityMedium,
			EstimatedEffort: "15-45 minutes",
			Tags:            []string{"complexity", "nesting"},
		}

		issues = append(issues, issue)
	}

	// Check for long functions
	longFunctions := cra.complexityAnalyzer.FindLongFunctions(request.Code, request.Language)
	for _, longFunc := range longFunctions {
		issue := &ReviewIssue{
			ID:              cra.generateIssueID(),
			Type:            IssueTypeComplexity,
			Severity:        SeverityInfo,
			Category:        CategoryComplexity,
			Title:           "Long Function",
			Description:     fmt.Sprintf("Function '%s' has %d lines, consider breaking it down", longFunc.Name, longFunc.LineCount),
			Location:        longFunc.Location,
			Explanation:     "Long functions are harder to understand, test, and maintain",
			Impact:          "Medium - affects maintainability",
			Suggestion:      "Extract logical sections into separate functions with descriptive names",
			Priority:        PriorityLow,
			EstimatedEffort: "30-60 minutes",
			Tags:            []string{"complexity", "long-function"},
		}

		issues = append(issues, issue)
	}

	return issues
}

// Best practices review implementation
func (cra *CodeReviewAgent) performBestPracticesReview(request *CodeReviewRequest, analysis *CodeAnalysisResult) ([]*ReviewIssue, []*ReviewPositive) {
	var issues []*ReviewIssue
	var positives []*ReviewPositive
	// Check language-specific best practices
	bestPracticesIssues := cra.bestPracticesChecker.CheckBestPractices(request.Code, request.Language)

	for _, bpIssue := range bestPracticesIssues {
		issue := &ReviewIssue{
			ID:              cra.generateIssueID(),
			Type:            IssueTypeBestPractice,
			Severity:        bpIssue.Severity,
			Category:        CategoryBestPractices,
			Title:           bpIssue.Title,
			Description:     bpIssue.Description,
			Location:        bpIssue.Location,
			Code:            bpIssue.Code,
			Explanation:     bpIssue.Explanation,
			Impact:          bpIssue.Impact,
			Suggestion:      bpIssue.Suggestion,
			References:      bpIssue.References,
			Priority:        cra.mapSeverityToPriority(bpIssue.Severity),
			EstimatedEffort: bpIssue.EstimatedEffort,
			Tags:            []string{"best-practices", bpIssue.Category},
		}

		issues = append(issues, issue)
	}

	// Look for positive best practices
	goodPractices := cra.bestPracticesChecker.FindGoodPractices(request.Code, request.Language)

	for _, practice := range goodPractices {
		positive := &ReviewPositive{
			Type:        PositiveGoodPractices,
			Title:       practice.Title,
			Description: practice.Description,
			Location:    practice.Location,
			Code:        practice.Code,
			Explanation: practice.Explanation,
			Value:       practice.Value,
		}

		positives = append(positives, positive)
	}

	return issues, positives
}

// Utility methods
func (cra *CodeReviewAgent) shouldPerformReview(options *CodeReviewOptions, focusArea ReviewFocusArea) bool {
	if options == nil || len(options.FocusAreas) == 0 {
		// If no specific focus areas, check if it's in default config
		for _, area := range cra.config.FocusAreas {
			if area == focusArea {
				return true
			}
		}
		return true // Default to true if not specified
	}
	for _, area := range options.FocusAreas {
		if area == focusArea {
			return true
		}
	}

	return false
}

func (cra *CodeReviewAgent) executeParallelReviews(ctx context.Context, tasks []func() error) error {
	var wg sync.WaitGroup
	errorChan := make(chan error, len(tasks))
	for _, task := range tasks {
		wg.Add(1)
		go func(t func() error) {
			defer wg.Done()
			if err := t(); err != nil {
				errorChan <- err
			}
		}(task)
	}

	wg.Wait()
	close(errorChan)

	// Collect any errors
	var errors []error
	for err := range errorChan {
		errors = append(errors, err)
	}

	if len(errors) > 0 {
		return errors[0] // Return first error
	}

	return nil
}

func (cra *CodeReviewAgent) executeSequentialReviews(ctx context.Context, tasks []func() error) error {
	for _, task := range tasks {
		if err := task(); err != nil {
			return err
		}
	}
	return nil
}

func (cra *CodeReviewAgent) applyCustomRules(code, language string) []*ReviewIssue {
	var issues []*ReviewIssue
	for _, rule := range cra.config.CustomRules {
		// Skip if rule doesn't apply to this language
		if len(rule.Languages) > 0 && !cra.contains(rule.Languages, language) {
			continue
		}

		// Apply pattern matching
		if rule.Pattern != "" {
			if matched, err := regexp.MatchString(rule.Pattern, code); err == nil && matched {
				issue := &ReviewIssue{
					ID:              cra.generateIssueID(),
					Type:            IssueTypeCustom,
					Severity:        rule.Severity,
					Category:        IssueCategory(rule.Category),
					Title:           rule.Name,
					Description:     rule.Description,
					Explanation:     rule.Message,
					Suggestion:      rule.Suggestion,
					Priority:        cra.mapSeverityToPriority(rule.Severity),
					EstimatedEffort: "5-30 minutes",
					Tags:            []string{"custom-rule", rule.Category},
				}

				issues = append(issues, issue)
			}
		}
	}

	return issues
}

func (cra *CodeReviewAgent) filterAndPrioritizeIssues(issues []*ReviewIssue, request *CodeReviewRequest) []*ReviewIssue {
	var filtered []*ReviewIssue
	// Filter by minimum severity
	for _, issue := range issues {
		if issue.Severity >= cra.config.MinSeverityLevel {
			filtered = append(filtered, issue)
		}
	}

	// Group by category and limit
	categoryCount := make(map[IssueCategory]int)
	var final []*ReviewIssue

	// Sort by priority and severity first
	sort.Slice(filtered, func(i, j int) bool {
		if filtered[i].Priority != filtered[j].Priority {
			return filtered[i].Priority > filtered[j].Priority
		}
		return filtered[i].Severity > filtered[j].Severity
	})

	// Apply category limits
	for _, issue := range filtered {
		if categoryCount[issue.Category] < cra.config.MaxIssuesPerCategory {
			final = append(final, issue)
			categoryCount[issue.Category]++
		}
	}

	return final
}

// Quality assessment methods
func (cra *CodeReviewAgent) assessCodeQuality(request *CodeReviewRequest, issues []*ReviewIssue, analysis *CodeAnalysisResult) *QualityAssessment {
	// Calculate category scores
	categoryScores := cra.calculateCategoryScores(issues)
	// Calculate overall score
	var totalScore float32
	var count int
	for _, score := range categoryScores {
		totalScore += score
		count++
	}

	overallScore := float32(0.8) // Base score
	if count > 0 {
		// Adjust based on issues found
		issueImpact := float32(len(issues)) * 0.05
		overallScore = overallScore - issueImpact

		if overallScore < 0 {
			overallScore = 0
		}
	}

	// Identify strengths and weaknesses
	strengths := cra.identifyStrengths(categoryScores, analysis)
	weaknesses := cra.identifyWeaknesses(issues)
	improvementAreas := cra.identifyImprovementAreas(issues, analysis)

	return &QualityAssessment{
		OverallScore:     overallScore,
		CategoryScores:   cra.convertCategoryScores(categoryScores),
		Strengths:        strengths,
		Weaknesses:       weaknesses,
		ImprovementAreas: improvementAreas,
	}
}

func (cra *CodeReviewAgent) calculateCategoryScores(issues []*ReviewIssue) map[IssueCategory]float32 {
	scores := map[IssueCategory]float32{
		CategoryStyle:         0.8,
		CategorySecurity:      0.8,
		CategoryPerformance:   0.8,
		CategoryMaintainability: 0.8,
		CategoryComplexity:    0.8,
		CategoryBestPractices: 0.8,
	}
	// Reduce scores based on issues found
	for _, issue := range issues {
		impact := float32(0.1)
		switch issue.Severity {
		case SeverityCritical:
			impact = 0.3
		case SeverityError:
			impact = 0.2
		case SeverityWarning:
			impact = 0.1
		case SeverityInfo:
			impact = 0.05
		}

		scores[issue.Category] -= impact
		if scores[issue.Category] < 0 {
			scores[issue.Category] = 0
		}
	}

	return scores
}

func (cra *CodeReviewAgent) convertCategoryScores(scores map[IssueCategory]float32) map[string]float32 {
	converted := make(map[string]float32)
	for category, score := range scores {
		converted[string(category)] = score
	}
	return converted
}

func (cra *CodeReviewAgent) identifyStrengths(categoryScores map[IssueCategory]float32, analysis *CodeAnalysisResult) []string {
	var strengths []string
	for category, score := range categoryScores {
		if score > 0.8 {
			switch category {
			case CategoryStyle:
				strengths = append(strengths, "Consistent code style and formatting")
			case CategorySecurity:
				strengths = append(strengths, "Good security practices")
			case CategoryPerformance:
				strengths = append(strengths, "Efficient code implementation")
			case CategoryMaintainability:
				strengths = append(strengths, "Well-structured and maintainable code")
			case CategoryComplexity:
				strengths = append(strengths, "Appropriate complexity levels")
			case CategoryBestPractices:
				strengths = append(strengths, "Follows language best practices")
			}
		}
	}

	// Add analysis-based strengths
	if analysis.CyclomaticComplexity <= 5 {
		strengths = append(strengths, "Low complexity, easy to understand")
	}

	if len(analysis.Functions) > 0 && len(analysis.Functions) <= 3 {
		strengths = append(strengths, "Good function organization")
	}

	return strengths
}

func (cra *CodeReviewAgent) identifyWeaknesses(issues []*ReviewIssue) []string {
	weaknessMap := make(map[string]bool)
	for _, issue := range issues {
		if issue.Severity >= SeverityWarning {
			switch issue.Category {
			case CategorySecurity:
				weaknessMap["Security vulnerabilities present"] = true
			case CategoryPerformance:
				weaknessMap["Performance optimization opportunities"] = true
			case CategoryMaintainability:
				weaknessMap["Maintainability concerns"] = true
			case CategoryComplexity:
				weaknessMap["High complexity in some areas"] = true
			case CategoryStyle:
				weaknessMap["Inconsistent code style"] = true
			case CategoryBestPractices:
				weaknessMap["Not following some best practices"] = true
			}
		}
	}

	var weaknesses []string
	for weakness := range weaknessMap {
		weaknesses = append(weaknesses, weakness)
	}

	return weaknesses
}

func (cra *CodeReviewAgent) identifyImprovementAreas(issues []*ReviewIssue, analysis *CodeAnalysisResult) []string {
	areaMap := make(map[string]bool)
	// Based on issues
	for _, issue := range issues {
		switch issue.Category {
		case CategorySecurity:
			areaMap["Enhance security measures"] = true
		case CategoryPerformance:
			areaMap["Optimize performance bottlenecks"] = true
		case CategoryMaintainability:
			areaMap["Improve code organization"] = true
		case CategoryComplexity:
			areaMap["Reduce code complexity"] = true
		case CategoryStyle:
			areaMap["Standardize code formatting"] = true
		case CategoryBestPractices:
			areaMap["Adopt better coding practices"] = true
		}
	}

	// Based on analysis
	if analysis.CyclomaticComplexity > 10 {
		areaMap["Break down complex functions"] = true
	}

	if len(analysis.Functions) > 10 {
		areaMap["Consider code organization improvements"] = true
	}

	var areas []string
	for area := range areaMap {
		areas = append(areas, area)
	}

	return areas
}

// Code metrics generation
func (cra *CodeReviewAgent) generateCodeMetrics(analysis *CodeAnalysisResult, issues []*ReviewIssue) *CodeMetrics {
	// Count issues by type
	securityHotspots := 0
	codeSmells := 0
	for _, issue := range issues {
		if issue.Category == CategorySecurity {
			securityHotspots++
		}
		if issue.Category == CategoryMaintainability || issue.Category == CategoryComplexity {
			codeSmells++
		}
	}

	// Calculate technical debt (simplified)
	technicalDebt := time.Duration(len(issues)) * time.Minute * 15 // 15 min per issue average

	// Calculate maintainability index (simplified)
	maintainability := float32(0.8) - float32(len(issues))*0.05
	if maintainability < 0 {
		maintainability = 0
	}

	return &CodeMetrics{
		LinesOfCode:          analysis.LinesOfCode,
		CyclomaticComplexity: analysis.CyclomaticComplexity,
		CognitiveComplexity:  analysis.CyclomaticComplexity, // Simplified
		Maintainability:      maintainability,
		DuplicationRatio:     0.0, // Would need more sophisticated analysis
		TechnicalDebt:        technicalDebt,
		CodeSmells:           codeSmells,
		SecurityHotspots:     securityHotspots,
	}
}

// Fix suggestions generation
func (cra *CodeReviewAgent) generateFixSuggestions(ctx context.Context, request *CodeReviewRequest, issues []*ReviewIssue) []*FixSuggestion {
	var fixSuggestions []*FixSuggestion
	// Generate fixes for high-priority issues
	for _, issue := range issues {
		if issue.Priority >= PriorityMedium && cra.canGenerateAutomaticFix(issue.Type) {
			fix := cra.generateFixForIssue(ctx, request, issue)
			if fix != nil {
				fixSuggestions = append(fixSuggestions, fix)
			}
		}
	}

	return fixSuggestions
}

func (cra *CodeReviewAgent) canGenerateAutomaticFix(issueType IssueType) bool {
	autoFixableTypes := []IssueType{
		IssueTypeStyle,
		IssueTypeSyntax,
		IssueTypeSimpleBug,
	}
	for _, fixableType := range autoFixableTypes {
		if issueType == fixableType {
			return true
		}
	}

	return false
}

func (cra *CodeReviewAgent) generateFixForIssue(ctx context.Context, request *CodeReviewRequest, issue *ReviewIssue) *FixSuggestion {
	// For now, return a simple fix suggestion
	// In practice, this would use more sophisticated analysis
	return &FixSuggestion{
		IssueID:       issue.ID,
		Type:          FixTypeQuick,
		Title:         fmt.Sprintf("Fix: %s", issue.Title),
		Description:   issue.Suggestion,
		OriginalCode:  issue.Code,
		SuggestedCode: cra.generateSuggestedCode(issue),
		Explanation:   fmt.Sprintf("This fix addresses: %s", issue.Description),
		Confidence:    0.7,
		RiskLevel:     RiskLow,
		TestingAdvice: "Please test this change thoroughly before deploying",
	}
}

func (cra *CodeReviewAgent) generateSuggestedCode(issue *ReviewIssue) string {
	// Simplified fix generation - would be more sophisticated in practice
	switch issue.Type {
	case IssueTypeStyle:
		if strings.Contains(issue.Description, "indentation") {
			return "// Fixed indentation\n" + issue.Code
		}
		return "// Style fixed\n" + issue.Code
	default:
		return issue.Code + " // TODO: Apply suggested fix"
	}
}

// Overall rating generation
func (cra *CodeReviewAgent) generateOverallRating(quality *QualityAssessment, issues []*ReviewIssue) *OverallRating {
	if quality == nil {
		return &OverallRating{
			Grade:         GradeC,
			Score:         0.5,
			Justification: "Unable to assess quality",
		}
	}
	score := quality.OverallScore
	var grade Grade

	switch {
	case score >= 0.9:
		grade = GradeA
	case score >= 0.8:
		grade = GradeB
	case score >= 0.7:
		grade = GradeC
	case score >= 0.6:
		grade = GradeD
	default:
		grade = GradeF
	}

	justification := cra.generateRatingJustification(score, issues)

	return &OverallRating{
		Grade:         grade,
		Score:         score,
		Justification: justification,
		ComparedTo:    "Industry standards",
	}
}

func (cra *CodeReviewAgent) generateRatingJustification(score float32, issues []*ReviewIssue) string {
	criticalCount := cra.countIssuesBySeverity(issues, []IssueSeverity{SeverityCritical})
	errorCount := cra.countIssuesBySeverity(issues, []IssueSeverity{SeverityError})
	if score >= 0.9 {
		return "Excellent code quality with minimal issues"
	} else if score >= 0.8 {
		return "Good code quality with minor improvements needed"
	} else if score >= 0.7 {
		return "Acceptable code quality, some issues to address"
	} else if score >= 0.6 {
		return fmt.Sprintf("Below average quality, %d major issues found", criticalCount+errorCount)
	} else {
		return fmt.Sprintf("Poor code quality, immediate attention required (%d critical issues)", criticalCount)
	}
}

// Recommendations generation
func (cra *CodeReviewAgent) generateRecommendations(request *CodeReviewRequest, response *CodeReviewResponse) []*Recommendation {
	var recommendations []*Recommendation

	// Security recommendations
	securityIssues := cra.filterIssuesByCategory(response.Issues, CategorySecurity)
	if len(securityIssues) > 0 {
		rec := &Recommendation{
			Type:        RecommendationSecurity,
			Title:       "Address Security Vulnerabilities",
			Description: fmt.Sprintf("Found %d security issues that need immediate attention", len(securityIssues)),
			ActionItems: cra.generateSecurityActionItems(securityIssues),
			Priority:    PriorityCritical,
			Timeline:    "Immediate - within 24 hours",
			Resources:   []string{"Security team review", "OWASP guidelines", "Security testing tools"},
		}
		recommendations = append(recommendations, rec)
	}

	// Performance recommendations
	performanceIssues := cra.filterIssuesByCategory(response.Issues, CategoryPerformance)
	if len(performanceIssues) > 0 {
		rec := &Recommendation{
			Type:        RecommendationPerformance,
			Title:       "Performance Optimization",
			Description: fmt.Sprintf("Identified %d performance issues that could impact user experience", len(performanceIssues)),
			ActionItems: cra.generatePerformanceActionItems(performanceIssues),
			Priority:    PriorityHigh,
			Timeline:    "Within 1 week",
			Resources:   []string{"Performance profiling tools", "Load testing", "Performance team consultation"},
		}
		recommendations = append(recommendations, rec)
	}

	// Code quality recommendations
	if response.QualityAssessment != nil && response.QualityAssessment.OverallScore < 0.7 {
		rec := &Recommendation{
			Type:        RecommendationQuality,
			Title:       "Code Quality Improvement",
			Description: "Overall code quality is below acceptable standards",
			ActionItems: []string{
				"Refactor complex functions",
				"Improve code documentation",
				"Add unit tests",
				"Follow coding standards consistently",
			},
			Priority:  PriorityMedium,
			Timeline:  "Within 2 weeks",
			Resources: []string{"Code review guidelines", "Refactoring tools", "Static analysis tools"},
		}
		recommendations = append(recommendations, rec)
	}

	// Architecture recommendations (if enabled)
	if cra.config.EnableArchitectureReview {
		complexityIssues := cra.filterIssuesByCategory(response.Issues, CategoryComplexity)
		if len(complexityIssues) > 3 {
			rec := &Recommendation{
				Type:        RecommendationArchitecture,
				Title:       "Architecture Review",
				Description: "Multiple complexity issues suggest potential architectural problems",
				ActionItems: []string{
					"Review overall system architecture",
					"Consider breaking down large components",
					"Evaluate design patterns usage",
					"Plan refactoring strategy",
				},
				Priority:  PriorityMedium,
				Timeline:  "Within 1 month",
				Resources: []string{"Architecture team", "Design pattern resources", "Refactoring guides"},
			}
			recommendations = append(recommendations, rec)
		}
	}

	return recommendations
}

// Helper methods for response generation
func (cra *CodeReviewAgent) generateOverallAssessment(response *CodeReviewResponse) string {
	criticalCount := cra.countIssuesBySeverity(response.Issues, []IssueSeverity{SeverityCritical})
	errorCount := cra.countIssuesBySeverity(response.Issues, []IssueSeverity{SeverityError})
	warningCount := cra.countIssuesBySeverity(response.Issues, []IssueSeverity{SeverityWarning})

	if criticalCount > 0 {
		return fmt.Sprintf("Code requires immediate attention due to %d critical issues", criticalCount)
	} else if errorCount > 0 {
		return fmt.Sprintf("Code has %d significant issues that should be addressed", errorCount)
	} else if warningCount > 0 {
		return fmt.Sprintf("Code is generally good with %d minor issues for improvement", warningCount)
	}
	return "Code looks good with no significant issues found"
}

func (cra *CodeReviewAgent) extractKeyFindings(issues []*ReviewIssue) []string {
	var findings []string
	// Group by category and pick top issues
	categoryIssues := make(map[IssueCategory][]*ReviewIssue)
	for _, issue := range issues {
		categoryIssues[issue.Category] = append(categoryIssues[issue.Category], issue)
	}

	for category, categoryIssueList := range categoryIssues {
		if len(categoryIssueList) > 0 {
			// Sort by severity
			sort.Slice(categoryIssueList, func(i, j int) bool {
				return categoryIssueList[i].Severity > categoryIssueList[j].Severity
			})

			topIssue := categoryIssueList[0]
			finding := fmt.Sprintf("%s: %s", category, topIssue.Title)
			findings = append(findings, finding)
		}
	}

	return findings
}

func (cra *CodeReviewAgent) extractPositiveAspects(positives []*ReviewPositive) []string {
	var aspects []string
	for _, positive := range positives {
		aspects = append(aspects, positive.Title)
	}
	return aspects
}

func (cra *CodeReviewAgent) extractRecommendedActions(recommendations []*Recommendation) []string {
	var actions []string
	for _, rec := range recommendations {
		if rec.Priority >= PriorityHigh {
			actions = append(actions, rec.Title)
		}
	}

	// If no high-priority actions, add medium priority
	if len(actions) == 0 {
		for _, rec := range recommendations {
			if rec.Priority >= PriorityMedium {
				actions = append(actions, rec.Title)
			}
		}
	}

	return actions
}

func (cra *CodeReviewAgent) calculateReviewConfidence(response *CodeReviewResponse) float32 {
	confidence := float32(0.8) // Base confidence
	// Adjust based on analysis completeness
	if response.QualityAssessment != nil {
		confidence += 0.1
	}

	if response.Metrics != nil {
		confidence += 0.05
	}

	if len(response.FixSuggestions) > 0 {
		confidence += 0.05
	}

	return confidence
}

// Utility methods
func (cra *CodeReviewAgent) countIssuesBySeverity(issues []*ReviewIssue, severities []IssueSeverity) int {
	count := 0
	for _, issue := range issues {
		for _, severity := range severities {
			if issue.Severity == severity {
				count++
				break
			}
		}
	}
	return count
}

func (cra *CodeReviewAgent) filterIssuesByCategory(issues []*ReviewIssue, category IssueCategory) []*ReviewIssue {
	var filtered []*ReviewIssue
	for _, issue := range issues {
		if issue.Category == category {
			filtered = append(filtered, issue)
		}
	}
	return filtered
}

func (cra *CodeReviewAgent) generateSecurityActionItems(issues []*ReviewIssue) []string {
	var actions []string
	for _, issue := range issues {
		action := fmt.Sprintf("Fix %s: %s", strings.ToLower(issue.Title), issue.Suggestion)
		actions = append(actions, action)
	}
	return actions
}

func (cra *CodeReviewAgent) generatePerformanceActionItems(issues []*ReviewIssue) []string {
	var actions []string
	for _, issue := range issues {
		action := fmt.Sprintf("Optimize %s: %s", strings.ToLower(issue.Title), issue.Suggestion)
		actions = append(actions, action)
	}
	return actions
}

// Mapping methods
func (cra *CodeReviewAgent) mapSecuritySeverity(securityLevel string) IssueSeverity {
	switch strings.ToLower(securityLevel) {
	case "critical":
		return SeverityCritical
	case "high":
		return SeverityError
	case "medium":
		return SeverityWarning
	case "low":
		return SeverityInfo
	default:
		return SeverityWarning
	}
}

func (cra *CodeReviewAgent) mapPerformanceSeverity(impactLevel string) IssueSeverity {
	switch strings.ToLower(impactLevel) {
	case "high":
		return SeverityError
	case "medium":
		return SeverityWarning
	case "low":
		return SeverityInfo
	default:
		return SeverityInfo
	}
}

func (cra *CodeReviewAgent) mapSeverityToPriority(severity IssueSeverity) Priority {
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

func (cra *CodeReviewAgent) mapImpactToPriority(impactLevel string) Priority {
	switch strings.ToLower(impactLevel) {
	case "high":
		return PriorityHigh
	case "medium":
		return PriorityMedium
	case "low":
		return PriorityLow
	default:
		return PriorityMedium
	}
}

// Configuration methods
func (cra *CodeReviewAgent) getStyleGuideForLanguage(language string) *StyleGuide {
	if guide, exists := cra.config.StyleGuides[language]; exists {
		return guide
	}
	// Return default style guide
	return &StyleGuide{
		Name:              "Default",
		IndentationType:   "spaces",
		IndentationSize:   4,
		MaxLineLength:     120,
		NamingConventions: make(map[string]string),
		RequiredPatterns:  []string{},
		ForbiddenPatterns: []string{},
	}
}

func (cra *CodeReviewAgent) getDefaultStyleGuides() map[string]*StyleGuide {
	return map[string]*StyleGuide{
		"go": {
			Name:            "Go Style Guide",
			IndentationType: "tabs",
			IndentationSize: 1,
			MaxLineLength:   120,
			NamingConventions: map[string]string{
				"function": "camelCase",
				"variable": "camelCase",
				"constant": "UPPER_CASE",
			},
		},
		"python": {
			Name:            "PEP 8",
			IndentationType: "spaces",
			IndentationSize: 4,
			MaxLineLength:   79,
			NamingConventions: map[string]string{
				"function": "snake_case",
				"variable": "snake_case",
				"constant": "UPPER_CASE",
				"class":    "PascalCase",
			},
		},
		"javascript": {
			Name:            "JavaScript Standard",
			IndentationType: "spaces",
			IndentationSize: 2,
			MaxLineLength:   120,
			NamingConventions: map[string]string{
				"function": "camelCase",
				"variable": "camelCase",
				"constant": "UPPER_CASE",
				"class":    "PascalCase",
			},
		},
	}
}

func (cra *CodeReviewAgent) getDefaultLanguageSettings() map[string]*LanguageReviewConfig {
	return map[string]*LanguageReviewConfig{
		"go": {
			StyleGuide:    "go",
			Linters:       []string{"golangci-lint", "staticcheck", "gosec"},
			BestPractices: []string{"Use gofmt for consistent formatting", "Handle errors explicitly", "Use meaningful variable names", "Keep functions short and focused"},
			SecurityRules: []string{"Validate input parameters", "Use context for cancellation", "Avoid hardcoded secrets"},
		},
		"python": {
			StyleGuide:    "python",
			Linters:       []string{"pylint", "flake8", "bandit"},
			BestPractices: []string{"Follow PEP 8 style guide", "Use list comprehensions appropriately", "Handle exceptions properly", "Use type hints"},
			SecurityRules: []string{"Sanitize user input", "Use parameterized queries", "Validate file paths"},
		},
		"javascript": {
			StyleGuide:    "javascript",
			Linters:       []string{"eslint", "jshint"},
			BestPractices: []string{"Use strict mode", "Prefer const over var", "Use async/await for promises", "Handle promise rejections"},
			SecurityRules: []string{"Validate and sanitize input", "Use HTTPS for API calls", "Avoid eval() and similar functions"},
		},
	}
}

// Request parsing and validation
func (cra *CodeReviewAgent) parseCodeReviewRequest(request *AgentRequest) (*CodeReviewRequest, error) {
	// Try to parse from parameters first
	if params, ok := request.Parameters["code_review_request"].(*CodeReviewRequest); ok {
		return params, nil
	}
	// Parse from query and context
	reviewRequest := &CodeReviewRequest{
		Language: cra.inferLanguage(request.Context),
		Options: &CodeReviewOptions{
			ReviewDepth:            cra.config.ReviewDepth,
			ReviewStyle:            cra.config.ReviewStyle,
			IncludePositives:       true,
			IncludeSuggestions:     true,
			IncludeExamples:        true,
			GenerateFixSuggestions: cra.config.EnableAutoFixSuggestions,
			PrioritizeIssues:       true,
		},
	}

	// Extract code from context
	if request.Context != nil && request.Context.SelectedText != "" {
		reviewRequest.Code = request.Context.SelectedText
	}

	// Create review context
	if request.Context != nil {
		reviewRequest.Context = &CodeReviewContext{
			FilePath:       request.Context.CurrentFile,
			TargetAudience: AudienceGeneral,
			ReviewPurpose:  PurposeCodeImprovement,
		}
	}

	return reviewRequest, nil
}

// Complexity and effort estimation methods
func (cra *CodeReviewAgent) getComplexitySeverity(complexity int) IssueSeverity {
	if complexity > 20 {
		return SeverityError
	} else if complexity > 15 {
		return SeverityWarning
	}
	return SeverityInfo
}

func (cra *CodeReviewAgent) getComplexityPriority(complexity int) Priority {
	if complexity > 20 {
		return PriorityHigh
	} else if complexity > 15 {
		return PriorityMedium
	}
	return PriorityLow
}

func (cra *CodeReviewAgent) getComplexityFixEffort(complexity int) string {
	if complexity > 20 {
		return "2-4 hours"
	} else if complexity > 15 {
		return "1-2 hours"
	}
	return "30-60 minutes"
}

func (cra *CodeReviewAgent) estimateSecurityFixEffort(vulnerabilityType string) string {
	effortMap := map[string]string{
		"sql_injection":    "30-60 minutes",
		"xss":              "15-30 minutes",
		"csrf":             "30-90 minutes",
		"hardcoded_secret": "10-20 minutes",
		"weak_crypto":      "60-120 minutes",
	}
	if effort, exists := effortMap[vulnerabilityType]; exists {
		return effort
	}
	return "30-60 minutes"
}

func (cra *CodeReviewAgent) getSecurityExamples(vulnerabilityType string) []*CodeExample {
	examplesMap := map[string][]*CodeExample{
		"sql_injection": {
			{
				Title:       "Vulnerable Code",
				Description: "Direct string concatenation in SQL query",
				Code:        `query = "SELECT * FROM users WHERE id = " + userId`,
				Language:    "javascript",
			},
			{
				Title:       "Secure Code",
				Description: "Using parameterized queries",
				Code:        `query = "SELECT * FROM users WHERE id = ?"; db.query(query, [userId])`,
				Language:    "javascript",
			},
		},
		"xss": {
			{
				Title:       "Vulnerable Code",
				Description: "Direct insertion of user input",
				Code:        `element.innerHTML = userInput`,
				Language:    "javascript",
			},
			{
				Title:       "Secure Code",
				Description: "Proper escaping of user input",
				Code:        `element.textContent = userInput`,
				Language:    "javascript",
			},
		},
	}
	return examplesMap[vulnerabilityType]
}

// Utility helper methods
func (cra *CodeReviewAgent) contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func (cra *CodeReviewAgent) generateIssueID() string {
	return fmt.Sprintf("issue_%d", time.Now().UnixNano())
}

func (cra *CodeReviewAgent) inferLanguage(context *RequestContext) string {
	if context == nil {
		return "unknown"
	}
	if context.ProjectLanguage != "" {
		return context.ProjectLanguage
	}

	if context.CurrentFile != "" {
		ext := strings.ToLower(strings.TrimPrefix(filepath.Ext(context.CurrentFile), "."))
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
	}

	return "unknown"
}

func (cra *CodeReviewAgent) calculateConfidence(request *CodeReviewRequest, response *CodeReviewResponse) float64 {
	confidence := 0.8 // Base confidence
	// Adjust based on analysis completeness
	if response.QualityAssessment != nil {
		confidence += 0.1
	}

	if response.Metrics != nil {
		confidence += 0.05
	}

	// Adjust based on language support
	if request.Language != "unknown" {
		confidence += 0.05
	}

	// Adjust based on number of issues found (more thorough analysis)
	if len(response.Issues) > 0 {
		confidence += 0.05
	}

	// Cap at 1.0
	if confidence > 1.0 {
		confidence = 1.0
	}

	return confidence
}

// Component initialization and interface implementations
func (cra *CodeReviewAgent) initializeCapabilities() {
	cra.capabilities = &AgentCapabilities{
		AgentType: AgentTypeCodeReview,
		SupportedIntents: []IntentType{
			IntentCodeReview,
			IntentCodeImprovement,
			IntentSecurityAudit,
			IntentPerformanceAnalysis,
		},
		SupportedLanguages: []string{
			"go", "python", "javascript", "typescript", "java",
			"cpp", "c", "csharp", "ruby", "php", "rust",
		},
		SupportedFileTypes: []string{
			"go", "py", "js", "ts", "java", "cpp", "c", "cs", "rb", "php", "rs",
		},
		MaxContextSize:    8192,
		SupportsStreaming: false,
		SupportsAsync:     true,
		RequiresContext:   false,
		Capabilities: map[string]interface{}{
			"code_analysis":          cra.config.EnableCodeAnalysis,
			"style_review":           cra.config.EnableStyleReview,
			"security_review":        cra.config.EnableSecurityReview,
			"performance_review":     cra.config.EnablePerformanceReview,
			"maintainability_review": cra.config.EnableMaintainabilityReview,
			"quality_assessment":     cra.config.EnableQualityAssessment,
			"complexity_analysis":    cra.config.EnableComplexityAnalysis,
			"best_practices_check":   cra.config.EnableBestPracticesCheck,
			"auto_fix_suggestions":   cra.config.EnableAutoFixSuggestions,
			"custom_rules":           len(cra.config.CustomRules) > 0,
		},
	}
}

func (cra *CodeReviewAgent) initializeComponents() {
	// Initialize code analyzer
	cra.codeAnalyzer = NewCodeAnalyzer()
	// Initialize style checker
	if cra.config.EnableStyleReview {
		cra.styleChecker = NewStyleChecker(cra.config.StyleGuides)
	}

	// Initialize security analyzer
	if cra.config.EnableSecurityReview {
		cra.securityAnalyzer = NewSecurityAnalyzer(cra.config.LanguageSettings)
	}

	// Initialize performance analyzer
	if cra.config.EnablePerformanceReview {
		cra.performanceAnalyzer = NewPerformanceAnalyzer()
	}

	// Initialize maintainability analyzer
	if cra.config.EnableMaintainabilityReview {
		cra.maintainabilityAnalyzer = NewMaintainabilityAnalyzer()
	}

	// Initialize quality assessor
	if cra.config.EnableQualityAssessment {
		cra.qualityAssessor = NewCodeQualityAssessor()
	}

	// Initialize complexity analyzer
	if cra.config.EnableComplexityAnalysis {
		cra.complexityAnalyzer = NewComplexityAnalyzer()
	}

	// Initialize code smell detector
	if cra.config.EnableCodeSmellDetection {
		cra.codeSmellDetector = NewCodeSmellDetector()
	}

	// Initialize best practices checker
	if cra.config.EnableBestPracticesCheck {
		cra.bestPracticesChecker = NewBestPracticesChecker(cra.config.LanguageSettings)
	}

	// Initialize design pattern analyzer
	if cra.config.EnableDesignPatternAnalysis {
		cra.designPatternAnalyzer = NewDesignPatternAnalyzer()
	}

	// Initialize architecture analyzer
	if cra.config.EnableArchitectureReview {
		cra.architectureAnalyzer = NewArchitectureAnalyzer()
	}
}

// Metrics methods
func (cra *CodeReviewAgent) updateMetrics(language string, depth ReviewDepth, success bool, duration time.Duration, issuesFound int) {
	cra.metrics.mu.Lock()
	defer cra.metrics.mu.Unlock()
	cra.metrics.TotalReviews++
	cra.metrics.ReviewsByLanguage[language]++
	cra.metrics.ReviewsByDepth[depth]++

	// Update average review time
	if cra.metrics.AverageReviewTime == 0 {
		cra.metrics.AverageReviewTime = duration
	} else {
		cra.metrics.AverageReviewTime = (cra.metrics.AverageReviewTime + duration) / 2
	}

	// Update average issues found
	if cra.metrics.TotalReviews == 1 {
		cra.metrics.AverageIssuesFound = float32(issuesFound)
	} else {
		cra.metrics.AverageIssuesFound = (cra.metrics.AverageIssuesFound + float32(issuesFound)) / 2.0
	}

	cra.metrics.LastReview = time.Now()
}

// Required Agent interface methods
func (cra *CodeReviewAgent) GetCapabilities() *AgentCapabilities {
	return cra.capabilities
}

func (cra *CodeReviewAgent) GetType() AgentType {
	return AgentTypeCodeReview
}

func (cra *CodeReviewAgent) GetVersion() string {
	return "1.0.0"
}

func (cra *CodeReviewAgent) GetStatus() AgentStatus {
	cra.mu.RLock()
	defer cra.mu.RUnlock()
	return cra.status
}

func (cra *CodeReviewAgent) Initialize(config interface{}) error {
	if cfg, ok := config.(*CodeReviewAgentConfig); ok {
		cra.config = cfg
		cra.initializeComponents()
		return nil
	}
	return fmt.Errorf("invalid config type")
}

func (cra *CodeReviewAgent) Start() error {
	cra.mu.Lock()
	defer cra.mu.Unlock()
	cra.status = StatusIdle
	cra.logger.Info("Code review agent started")
	return nil
}

func (cra *CodeReviewAgent) Stop() error {
	cra.mu.Lock()
	defer cra.mu.Unlock()
	cra.status = StatusStopped
	cra.logger.Info("Code review agent stopped")
	return nil
}

func (cra *CodeReviewAgent) HealthCheck() error {
	if cra.llmProvider == nil {
		return fmt.Errorf("LLM provider not configured")
	}
	// Test basic functionality
	testCode := "function test() { return true; }"
	analysis, err := cra.codeAnalyzer.Analyze(testCode, "javascript")
	if err != nil {
		return fmt.Errorf("code analyzer not working: %v", err)
	}

	if analysis == nil {
		return fmt.Errorf("code analyzer returned nil")
	}

	return nil
}

func (cra *CodeReviewAgent) GetMetrics() *AgentMetrics {
	cra.metrics.mu.RLock()
	defer cra.metrics.mu.RUnlock()
	return &AgentMetrics{
		RequestsProcessed:   cra.metrics.TotalReviews,
		AverageResponseTime: cra.metrics.AverageReviewTime,
		SuccessRate:         1.0, // Would track failures in real implementation
		LastRequestAt:       cra.metrics.LastReview,
	}
}

func (cra *CodeReviewAgent) ResetMetrics() {
	cra.metrics.mu.Lock()
	defer cra.metrics.mu.Unlock()
	cra.metrics = &CodeReviewAgentMetrics{
		ReviewsByLanguage: make(map[string]int64),
		ReviewsByDepth:    make(map[ReviewDepth]int64),
		IssuesByCategory:  make(map[IssueCategory]int64),
		IssuesBySeverity:  make(map[IssueSeverity]int64),
	}
}

// Placeholder implementations for referenced components would follow here...
// Due to length constraints, I'm showing the structure but not implementing
// every single component in full detail.

// Placeholder component implementations
type StyleChecker struct {
	styleGuides map[string]*StyleGuide
}

func NewStyleChecker(guides map[string]*StyleGuide) *StyleChecker {
	return &StyleChecker{styleGuides: guides}
}

func (sc *StyleChecker) CheckIndentation(code string, guide *StyleGuide) []*StyleIssue {
	// Placeholder implementation
	return []*StyleIssue{}
}

func (sc *StyleChecker) CheckLineLength(code string, maxLength int) []*StyleIssue {
	// Placeholder implementation
	return []*StyleIssue{}
}

func (sc *StyleChecker) CheckNamingConventions(code, language string, guide *StyleGuide) []*StyleIssue {
	// Placeholder implementation
	return []*StyleIssue{}
}

func (sc *StyleChecker) HasConsistentStyle(code string, guide *StyleGuide) bool {
	// Placeholder implementation
	return true
}

func (sc *StyleChecker) HasGoodNaming(code, language string) bool {
	// Placeholder implementation
	return true
}

type StyleIssue struct {
	Description string         `json:"description"`
	Location    *IssueLocation `json:"location"`
	Code        string         `json:"code"`
	Suggestion  string         `json:"suggestion"`
}

// Continue with other placeholder components...
