package agents

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/llm"
	"github.com/yourusername/ai-code-assistant/internal/app"
	"github.com/yourusername/ai-code-assistant/internal/indexer"
	"github.com/yourusername/ai-code-assistant/internal/logger"
	"golang.org/x/tools/go/analysis/passes/defers"
)

// DocumentationAgent generates or updates code documentation
type DocumentationAgent struct {
	// Core components
	llmProvider    llm.Provider
	indexer        *indexer.UltraFastIndexer
	contextManager *app.ContextManager

	// Agent configuration
	config *DocumentationAgentConfig
	logger logger.Logger

	// Capabilities
	capabilities *AgentCapabilities

	// Documentation analysis
	docAnalyzer   *DocumentationAnalyzer
	codeAnalyzer  *CodeAnalyzer
	styleAnalyzer *DocumentationStyleAnalyzer

	// Documentation generation
	docGenerator   *DocumentationGenerator
	templateEngine *DocumentationTemplateEngine
	formatters     map[string]DocumentationFormatter

	// Quality assurance
	qualityChecker     *DocumentationQualityChecker
	consistencyChecker *ConsistencyChecker
	coverageAnalyzer   *CoverageAnalyzer

	// Advanced features
	exampleGenerator *CodeExampleGenerator
	diagramGenerator *DiagramGenerator
	linkResolver     *LinkResolver

	// Statistics and monitoring
	metrics *DocumentationAgentMetrics

	// State management
	mu     sync.RWMutex
	status AgentStatus
}

// DocumentationAgentConfig contains documentation agent configuration
type DocumentationAgentConfig struct {
	// Generation settings
	EnableDocGeneration    bool `json:"enable_doc_generation"`
	EnableDocUpdate        bool `json:"enable_doc_update"`
	EnableAPIDocGeneration bool `json:"enable_api_doc_generation"`
	EnableReadmeGeneration bool `json:"enable_readme_generation"`

	// Documentation formats
	SupportedFormats  []DocFormat `json:"supported_formats"`
	DefaultFormat     DocFormat   `json:"default_format"`
	EnableMultiFormat bool        `json:"enable_multi_format"`

	// Content settings
	EnableExampleGeneration bool `json:"enable_example_generation"`
	EnableDiagramGeneration bool `json:"enable_diagram_generation"`
	EnableLinkGeneration    bool `json:"enable_link_generation"`
	IncludeTypeInfo         bool `json:"include_type_info"`
	IncludeParameterInfo    bool `json:"include_parameter_info"`
	IncludeReturnInfo       bool `json:"include_return_info"`
	IncludeExceptionInfo    bool `json:"include_exception_info"`

	// Quality settings
	EnableQualityCheck     bool    `json:"enable_quality_check"`
	EnableConsistencyCheck bool    `json:"enable_consistency_check"`
	EnableCoverageAnalysis bool    `json:"enable_coverage_analysis"`
	MinQualityScore        float32 `json:"min_quality_score"`

	// Style settings
	DocumentationStyle     DocStyle             `json:"documentation_style"`
	LanguageStyles         map[string]*DocStyle `json:"language_styles"`
	EnableStyleConsistency bool                 `json:"enable_style_consistency"`

	// Advanced features
	EnableSemanticAnalysis  bool `json:"enable_semantic_analysis"`
	EnableContextAwareness  bool `json:"enable_context_awareness"`
	EnableIntelligentUpdate bool `json:"enable_intelligent_update"`

	// LLM settings
	LLMModel        string  `json:"llm_model"`
	MaxTokens       int     `json:"max_tokens"`
	Temperature     float32 `json:"temperature"`
	EnableStreaming bool    `json:"enable_streaming"`

	// Template settings
	TemplateDirectory string            `json:"template_directory"`
	CustomTemplates   map[string]string `json:"custom_templates"`
}

// Documentation types and formats

type DocFormat string

const (
	FormatMarkdown         DocFormat = "markdown"
	FormatJSDoc            DocFormat = "jsdoc"
	FormatJavaDoc          DocFormat = "javadoc"
	FormatPyDoc            DocFormat = "pydoc"
	FormatGoDoc            DocFormat = "godoc"
	FormatRustDoc          DocFormat = "rustdoc"
	FormatXMLDoc           DocFormat = "xmldoc"
	FormatReStructuredText DocFormat = "rst"
)

type DocStyle struct {
	CommentStyle     CommentStyle `json:"comment_style"`
	IncludeAuthor    bool         `json:"include_author"`
	IncludeDate      bool         `json:"include_date"`
	IncludeVersion   bool         `json:"include_version"`
	IncludeSince     bool         `json:"include_since"`
	IncludeSeeAlso   bool         `json:"include_see_also"`
	MaxLineLength    int          `json:"max_line_length"`
	IndentationStyle string       `json:"indentation_style"`
	Sections         []DocSection `json:"sections"`
}

type CommentStyle string

const (
	StyleSingleLine  CommentStyle = "single_line"  // //
	StyleMultiLine   CommentStyle = "multi_line"   // /* */
	StyleDocBlock    CommentStyle = "doc_block"    // /** */
	StyleTripleQuote CommentStyle = "triple_quote" // """ """
	StyleHash        CommentStyle = "hash"         // #
)

type DocSection struct {
	Name     string `json:"name"`
	Required bool   `json:"required"`
	Order    int    `json:"order"`
	Format   string `json:"format"`
}

// Request and response structures

type DocumentationRequest struct {
	Type         DocumentationType     `json:"type"`
	Target       *DocumentationTarget  `json:"target"`
	Code         string                `json:"code,omitempty"`
	ExistingDoc  string                `json:"existing_doc,omitempty"`
	Format       DocFormat             `json:"format,omitempty"`
	Style        *DocStyle             `json:"style,omitempty"`
	Requirements []string              `json:"requirements,omitempty"`
	Options      *DocumentationOptions `json:"options,omitempty"`
}

type DocumentationType string

const (
	DocTypeGenerate  DocumentationType = "generate"
	DocTypeUpdate    DocumentationType = "update"
	DocTypeImprove   DocumentationType = "improve"
	DocTypeTranslate DocumentationType = "translate"
	DocTypeAPIDoc    DocumentationType = "api_doc"
	DocTypeReadme    DocumentationType = "readme"
	DocTypeInline    DocumentationType = "inline"
)

type DocumentationTarget struct {
	Type       TargetType            `json:"type"`
	Identifier string                `json:"identifier"`
	FilePath   string                `json:"file_path,omitempty"`
	LineStart  int                   `json:"line_start,omitempty"`
	LineEnd    int                   `json:"line_end,omitempty"`
	Language   string                `json:"language"`
	Context    *DocumentationContext `json:"context,omitempty"`
}

type TargetType string

const (
	TargetFunction  TargetType = "function"
	TargetClass     TargetType = "class"
	TargetMethod    TargetType = "method"
	TargetVariable  TargetType = "variable"
	TargetConstant  TargetType = "constant"
	TargetInterface TargetType = "interface"
	TargetModule    TargetType = "module"
	TargetFile      TargetType = "file"
	TargetProject   TargetType = "project"
)

type DocumentationContext struct {
	ProjectInfo   *ProjectInfo `json:"project_info,omitempty"`
	ModuleInfo    *ModuleInfo  `json:"module_info,omitempty"`
	RelatedCode   []string     `json:"related_code,omitempty"`
	Dependencies  []string     `json:"dependencies,omitempty"`
	UsageExamples []string     `json:"usage_examples,omitempty"`
	RelatedDocs   []string     `json:"related_docs,omitempty"`
}

type ProjectInfo struct {
	Name          string   `json:"name"`
	Description   string   `json:"description"`
	Version       string   `json:"version"`
	Authors       []string `json:"authors"`
	License       string   `json:"license"`
	Repository    string   `json:"repository"`
	Documentation string   `json:"documentation"`
}

type ModuleInfo struct {
	Name         string   `json:"name"`
	Description  string   `json:"description"`
	Exports      []string `json:"exports"`
	Imports      []string `json:"imports"`
	Dependencies []string `json:"dependencies"`
}

type DocumentationOptions struct {
	IncludeExamples bool        `json:"include_examples"`
	IncludeDiagrams bool        `json:"include_diagrams"`
	IncludeLinks    bool        `json:"include_links"`
	GenerateTests   bool        `json:"generate_tests"`
	DetailLevel     DetailLevel `json:"detail_level"`
	Audience        Audience    `json:"audience"`
	Language        string      `json:"language,omitempty"` // Human language, not programming
}

type DetailLevel string

const (
	DetailBasic         DetailLevel = "basic"
	DetailStandard      DetailLevel = "standard"
	DetailDetailed      DetailLevel = "detailed"
	DetailComprehensive DetailLevel = "comprehensive"
)

type Audience string

const (
	AudienceDeveloper   Audience = "developer"
	AudienceEndUser     Audience = "end_user"
	AudienceMaintainer  Audience = "maintainer"
	AudienceContributor Audience = "contributor"
)

// Documentation response

type DocumentationResponse struct {
	GeneratedDoc string                     `json:"generated_doc"`
	Format       DocFormat                  `json:"format"`
	Quality      *DocumentationQuality      `json:"quality,omitempty"`
	Coverage     *CoverageReport            `json:"coverage,omitempty"`
	Suggestions  []*DocumentationSuggestion `json:"suggestions,omitempty"`
	Examples     []*CodeExample             `json:"examples,omitempty"`
	Diagrams     []*Diagram                 `json:"diagrams,omitempty"`
	Links        []*DocumentationLink       `json:"links,omitempty"`
	Metadata     map[string]interface{}     `json:"metadata,omitempty"`
}

type DocumentationQuality struct {
	OverallScore    float32         `json:"overall_score"`
	Completeness    float32         `json:"completeness"`
	Clarity         float32         `json:"clarity"`
	Accuracy        float32         `json:"accuracy"`
	Consistency     float32         `json:"consistency"`
	Usefulness      float32         `json:"usefulness"`
	Issues          []*QualityIssue `json:"issues,omitempty"`
	Recommendations []string        `json:"recommendations"`
}

type CoverageReport struct {
	TotalElements      int                      `json:"total_elements"`
	DocumentedElements int                      `json:"documented_elements"`
	CoveragePercent    float32                  `json:"coverage_percent"`
	MissingDocs        []*MissingDocumentation  `json:"missing_docs"`
	CoverageByType     map[string]*TypeCoverage `json:"coverage_by_type"`
}

type MissingDocumentation struct {
	Type       TargetType `json:"type"`
	Identifier string     `json:"identifier"`
	FilePath   string     `json:"file_path"`
	LineNumber int        `json:"line_number"`
	Priority   Priority   `json:"priority"`
	Reason     string     `json:"reason"`
}

type TypeCoverage struct {
	Total      int     `json:"total"`
	Documented int     `json:"documented"`
	Percentage float32 `json:"percentage"`
}

type DocumentationSuggestion struct {
	Type            SuggestionType       `json:"type"`
	Title           string               `json:"title"`
	Description     string               `json:"description"`
	Target          *DocumentationTarget `json:"target,omitempty"`
	Priority        Priority             `json:"priority"`
	EstimatedEffort string               `json:"estimated_effort"`
	Example         string               `json:"example,omitempty"`
}

type CodeExample struct {
	Title       string `json:"title"`
	Description string `json:"description"`
	Code        string `json:"code"`
	Language    string `json:"language"`
	Output      string `json:"output,omitempty"`
	Explanation string `json:"explanation,omitempty"`
}

type Diagram struct {
	Type        DiagramType `json:"type"`
	Title       string      `json:"title"`
	Description string      `json:"description"`
	Content     string      `json:"content"`
	Format      string      `json:"format"` // mermaid, plantuml, etc.
}

type DiagramType string

const (
	DiagramFlow         DiagramType = "flow"
	DiagramSequence     DiagramType = "sequence"
	DiagramClass        DiagramType = "class"
	DiagramER           DiagramType = "er"
	DiagramArchitecture DiagramType = "architecture"
)

type DocumentationLink struct {
	Type        LinkType `json:"type"`
	Target      string   `json:"target"`
	Description string   `json:"description"`
	URL         string   `json:"url,omitempty"`
}

type LinkType string

const (
	LinkInternal  LinkType = "internal"
	LinkExternal  LinkType = "external"
	LinkAPI       LinkType = "api"
	LinkReference LinkType = "reference"
)

// DocumentationAgentMetrics tracks agent performance
type DocumentationAgentMetrics struct {
	TotalRequests       int64                       `json:"total_requests"`
	RequestsByType      map[DocumentationType]int64 `json:"requests_by_type"`
	RequestsByFormat    map[DocFormat]int64         `json:"requests_by_format"`
	AverageResponseTime time.Duration               `json:"average_response_time"`
	AverageQualityScore float32                     `json:"average_quality_score"`
	AverageCoverage     float32                     `json:"average_coverage"`
	SuccessRate         float64                     `json:"success_rate"`
	ExamplesGenerated   int64                       `json:"examples_generated"`
	DiagramsGenerated   int64                       `json:"diagrams_generated"`
	LinksGenerated      int64                       `json:"links_generated"`
	LastRequest         time.Time                   `json:"last_request"`
	mu                  sync.RWMutex
}

// NewDocumentationAgent creates a new documentation agent
func NewDocumentationAgent(llmProvider llm.Provider, indexer *indexer.UltraFastIndexer, config *DocumentationAgentConfig, logger logger.Logger) *DocumentationAgent {
	if config == nil {
		config = &DocumentationAgentConfig{
			EnableDocGeneration:    true,
			EnableDocUpdate:        true,
			EnableAPIDocGeneration: true,
			EnableReadmeGeneration: true,
			SupportedFormats: []DocFormat{
				FormatMarkdown, FormatJSDoc, FormatJavaDoc,
				FormatPyDoc, FormatGoDoc, FormatRustDoc,
			},
			DefaultFormat:           FormatMarkdown,
			EnableMultiFormat:       true,
			EnableExampleGeneration: true,
			EnableDiagramGeneration: false, // Expensive operation
			EnableLinkGeneration:    true,
			IncludeTypeInfo:         true,
			IncludeParameterInfo:    true,
			IncludeReturnInfo:       true,
			IncludeExceptionInfo:    true,
			EnableQualityCheck:      true,
			EnableConsistencyCheck:  true,
			EnableCoverageAnalysis:  true,
			MinQualityScore:         0.7,
			EnableSemanticAnalysis:  true,
			EnableContextAwareness:  true,
			EnableIntelligentUpdate: true,
			LLMModel:                "gpt-4",
			MaxTokens:               2048,
			Temperature:             0.3,
			EnableStreaming:         false,
			DocumentationStyle: DocStyle{
				CommentStyle:     StyleDocBlock,
				IncludeAuthor:    false,
				IncludeDate:      false,
				IncludeVersion:   false,
				MaxLineLength:    80,
				IndentationStyle: "spaces",
			},
		}
	}

	agent := &DocumentationAgent{
		llmProvider: llmProvider,
		indexer:     indexer,
		config:      config,
		logger:      logger,
		status:      StatusIdle,
		metrics: &DocumentationAgentMetrics{
			RequestsByType:   make(map[DocumentationType]int64),
			RequestsByFormat: make(map[DocFormat]int64),
		},
		formatters: make(map[string]DocumentationFormatter),
	}

	// Initialize capabilities
	agent.initializeCapabilities()

	// Initialize components
	agent.initializeComponents()

	return agent
}

// ProcessRequest processes a documentation request
func (da *DocumentationAgent) ProcessRequest(ctx context.Context, request *AgentRequest) (*AgentResponse, error) {
	start := time.Now()
	da.status = StatusBusy
	defer func() { da.status = StatusIdle }()

	// Parse documentation request
	docRequest, err := da.parseDocumentationRequest(request)
	if err != nil {
		return nil, fmt.Errorf("failed to parse documentation request: %v", err)
	}

	// Validate request
	if err := da.validateDocumentationRequest(docRequest); err != nil {
		return nil, fmt.Errorf("invalid documentation request: %v", err)
	}

	// Process based on request type
	var docResponse *DocumentationResponse
	switch docRequest.Type {
	case DocTypeGenerate:
		docResponse, err = da.generateDocumentation(ctx, docRequest)
	case DocTypeUpdate:
		docResponse, err = da.updateDocumentation(ctx, docRequest)
	case DocTypeImprove:
		docResponse, err = da.improveDocumentation(ctx, docRequest)
	case DocTypeTranslate:
		docResponse, err = da.translateDocumentation(ctx, docRequest)
	case DocTypeAPIDoc:
		docResponse, err = da.generateAPIDocumentation(ctx, docRequest)
	case DocTypeReadme:
		docResponse, err = da.generateReadme(ctx, docRequest)
	case DocTypeInline:
		docResponse, err = da.generateInlineDocumentation(ctx, docRequest)
	default:
		return nil, fmt.Errorf("unsupported documentation type: %s", docRequest.Type)
	}

	if err != nil {
		da.updateMetrics(docRequest.Type, docRequest.Format, false, time.Since(start))
		return nil, fmt.Errorf("documentation generation failed: %v", err)
	}

	// Perform quality checks if enabled
	if da.config.EnableQualityCheck {
		quality, err := da.performQualityCheck(docResponse.GeneratedDoc, docRequest)
		if err != nil {
			da.logger.Warn("Quality check failed", "error", err)
		} else {
			docResponse.Quality = quality
		}
	}

	// Perform coverage analysis if enabled
	if da.config.EnableCoverageAnalysis && docRequest.Target != nil {
		coverage, err := da.performCoverageAnalysis(docRequest.Target, docResponse.GeneratedDoc)
		if err != nil {
			da.logger.Warn("Coverage analysis failed", "error", err)
		} else {
			docResponse.Coverage = coverage
		}
	}

	// Create agent response
	response := &AgentResponse{
		RequestID:      request.ID,
		AgentType:      da.GetType(),
		AgentVersion:   da.GetVersion(),
		Result:         docResponse,
		Confidence:     da.calculateConfidence(docRequest, docResponse),
		ProcessingTime: time.Since(start),
		CreatedAt:      time.Now(),
	}

	// Update metrics
	da.updateMetrics(docRequest.Type, docRequest.Format, true, time.Since(start))

	return response, nil
}

// generateDocumentation generates new documentation
func (da *DocumentationAgent) generateDocumentation(ctx context.Context, request *DocumentationRequest) (*DocumentationResponse, error) {
	if !da.config.EnableDocGeneration {
		return nil, fmt.Errorf("documentation generation is disabled")
	}

	// Analyze the code to understand its structure and purpose
	analysis, err := da.analyzeCode(request.Code, request.Target)
	if err != nil {
		return nil, fmt.Errorf("code analysis failed: %v", err)
	}

	// Build the documentation prompt
	prompt := da.buildDocumentationPrompt(request, analysis)

	// Get documentation format
	format := request.Format
	if format == "" {
		format = da.config.DefaultFormat
	}

	// Call LLM to generate documentation
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

	// Extract and format the generated documentation
	generatedDoc := da.extractDocumentation(llmResponse.Text, format)

	// Format the documentation according to style guidelines
	if formatter, exists := da.formatters[string(format)]; exists {
		generatedDoc, err = formatter.Format(generatedDoc, request.Style)
		if err != nil {
			da.logger.Warn("Documentation formatting failed", "error", err)
		}
	}

	// Generate examples if requested
	var examples []*CodeExample
	if da.config.EnableExampleGeneration && (request.Options == nil || request.Options.IncludeExamples) {
		examples = da.generateExamples(request, analysis)
	}

	// Generate diagrams if requested and enabled
	var diagrams []*Diagram
	if da.config.EnableDiagramGeneration && request.Options != nil && request.Options.IncludeDiagrams {
		diagrams = da.generateDiagrams(request, analysis)
	}

	// Generate links if requested
	var links []*DocumentationLink
	if da.config.EnableLinkGeneration && (request.Options == nil || request.Options.IncludeLinks) {
		links = da.generateLinks(request, analysis)
	}

	// Generate suggestions
	suggestions := da.generateSuggestions(request, analysis, generatedDoc)

	return &DocumentationResponse{
		GeneratedDoc: generatedDoc,
		Format:       format,
		Examples:     examples,
		Diagrams:     diagrams,
		Links:        links,
		Suggestions:  suggestions,
	}, nil
}

// updateDocumentation updates existing documentation
func (da *DocumentationAgent) updateDocumentation(ctx context.Context, request *DocumentationRequest) (*DocumentationResponse, error) {
	if !da.config.EnableDocUpdate {
		return nil, fmt.Errorf("documentation update is disabled")
	}

	if request.ExistingDoc == "" {
		return nil, fmt.Errorf("existing documentation is required for update")
	}

	// Analyze existing documentation
	existingAnalysis := da.analyzeExistingDocumentation(request.ExistingDoc)

	// Analyze current code
	codeAnalysis, err := da.analyzeCode(request.Code, request.Target)
	if err != nil {
		return nil, fmt.Errorf("code analysis failed: %v", err)
	}

	// Determine what needs to be updated
	updateNeeds := da.identifyUpdateNeeds(existingAnalysis, codeAnalysis)

	// Build update prompt
	prompt := da.buildUpdatePrompt(request, existingAnalysis, codeAnalysis, updateNeeds)

	// Call LLM to update documentation
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

	// Extract updated documentation
	updatedDoc := da.extractDocumentation(llmResponse.Text, request.Format)

	// Generate update suggestions
	suggestions := da.generateUpdateSuggestions(request, updateNeeds, updatedDoc)

	return &DocumentationResponse{
		GeneratedDoc: updatedDoc,
		Format:       request.Format,
		Suggestions:  suggestions,
	}, nil
}

// improveDocumentation improves existing documentation quality
func (da *DocumentationAgent) improveDocumentation(ctx context.Context, request *DocumentationRequest) (*DocumentationResponse, error) {
	if request.ExistingDoc == "" {
		return nil, fmt.Errorf("existing documentation is required for improvement")
	}

	// Analyze documentation quality
	qualityAnalysis := da.analyzeDocumentationQuality(request.ExistingDoc)

	// Identify improvement opportunities
	improvements := da.identifyImprovements(qualityAnalysis)

	// Build improvement prompt
	prompt := da.buildImprovementPrompt(request, qualityAnalysis, improvements)

	// Call LLM to improve documentation
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

	// Extract improved documentation
	improvedDoc := da.extractDocumentation(llmResponse.Text, request.Format)

	// Generate improvement suggestions
	suggestions := da.generateImprovementSuggestions(improvements, improvedDoc)

	return &DocumentationResponse{
		GeneratedDoc: improvedDoc,
		Format:       request.Format,
		Suggestions:  suggestions,
	}, nil
}

// generateAPIDocumentation generates API documentation
func (da *DocumentationAgent) generateAPIDocumentation(ctx context.Context, request *DocumentationRequest) (*DocumentationResponse, error) {
	if !da.config.EnableAPIDocGeneration {
		return nil, fmt.Errorf("API documentation generation is disabled")
	}

	// Analyze API structure
	apiAnalysis, err := da.analyzeAPI(request.Code, request.Target)
	if err != nil {
		return nil, fmt.Errorf("API analysis failed: %v", err)
	}

	// Build API documentation prompt
	prompt := da.buildAPIDocumentationPrompt(request, apiAnalysis)

	// Call LLM to generate API documentation
	llmRequest := &llm.CompletionRequest{
		Prompt:      prompt,
		Model:       da.config.LLMModel,
		MaxTokens:   da.config.MaxTokens,
		Temperature: 0.2, // Lower temperature for more consistent API docs
	}

	llmResponse, err := da.llmProvider.Complete(ctx, llmRequest)
	if err != nil {
		return nil, fmt.Errorf("LLM completion failed: %v", err)
	}

	// Extract API documentation
	apiDoc := da.extractDocumentation(llmResponse.Text, request.Format)

	// Generate API examples
	var examples []*CodeExample
	if da.config.EnableExampleGeneration {
		examples = da.generateAPIExamples(request, apiAnalysis)
	}

	return &DocumentationResponse{
		GeneratedDoc: apiDoc,
		Format:       request.Format,
		Examples:     examples,
	}, nil
}

// generateReadme generates README documentation
func (da *DocumentationAgent) generateReadme(ctx context.Context, request *DocumentationRequest) (*DocumentationResponse, error) {
	if !da.config.EnableReadmeGeneration {
		return nil, fmt.Errorf("README generation is disabled")
	}

	// Analyze project structure
	projectAnalysis, err := da.analyzeProject(request.Target)
	if err != nil {
		return nil, fmt.Errorf("project analysis failed: %v", err)
	}

	// Build README prompt
	prompt := da.buildReadmePrompt(request, projectAnalysis)

	// Call LLM to generate README
	llmRequest := &llm.CompletionRequest{
		Prompt:      prompt,
		Model:       da.config.LLMModel,
		MaxTokens:   da.config.MaxTokens,
		Temperature: 0.4,
	}

	llmResponse, err := da.llmProvider.Complete(ctx, llmRequest)
	if err != nil {
		return nil, fmt.Errorf("LLM completion failed: %v", err)
	}

	// Extract README
	readme := da.extractDocumentation(llmResponse.Text, FormatMarkdown)

	return &DocumentationResponse{
		GeneratedDoc: readme,
		Format:       FormatMarkdown,
	}, nil
}

// generateInlineDocumentation generates inline code documentation
func (da *DocumentationAgent) generateInlineDocumentation(ctx context.Context, request *DocumentationRequest) (*DocumentationResponse, error) {
	// Analyze code for inline documentation opportunities
	inlineAnalysis := da.analyzeInlineDocumentationNeeds(request.Code, request.Target)

	// Build inline documentation prompt
	prompt := da.buildInlineDocumentationPrompt(request, inlineAnalysis)

	// Call LLM to generate inline documentation
	llmRequest := &llm.CompletionRequest{
		Prompt:      prompt,
		Model:       da.config.LLMModel,
		MaxTokens:   da.config.MaxTokens,
		Temperature: 0.2,
	}

	llmResponse, err := da.llmProvider.Complete(ctx, llmRequest)
	if err != nil {
		return nil, fmt.Errorf("LLM completion failed: %v", err)
	}

	// Extract inline documentation
	inlineDoc := da.extractDocumentation(llmResponse.Text, request.Format)

	return &DocumentationResponse{
		GeneratedDoc: inlineDoc,
		Format:       request.Format,
	}, nil
}

// Helper methods for prompt building

func (da *DocumentationAgent) buildDocumentationPrompt(request *DocumentationRequest, analysis *CodeAnalysisResult) string {
	var prompt strings.Builder

	prompt.WriteString("Generate comprehensive documentation for the following code:\n\n")

	// Add code
	prompt.WriteString("Code:\n```")
	if request.Target != nil {
		prompt.WriteString(request.Target.Language)
	}
	prompt.WriteString("\n")
	prompt.WriteString(request.Code)
	prompt.WriteString("\n```\n\n")

	// Add target information
	if request.Target != nil {
		prompt.WriteString(fmt.Sprintf("Target: %s (%s)\n", request.Target.Type, request.Target.Identifier))
		if request.Target.FilePath != "" {
			prompt.WriteString(fmt.Sprintf("File: %s\n", request.Target.FilePath))
		}
		prompt.WriteString("\n")
	}

	// Add analysis insights
	if analysis != nil {
		prompt.WriteString("Code Analysis:\n")
		prompt.WriteString(fmt.Sprintf("- Purpose: %s\n", analysis.Purpose))
		prompt.WriteString(fmt.Sprintf("- Complexity: %s\n", analysis.ComplexityLevel))
		if len(analysis.Parameters) > 0 {
			prompt.WriteString("- Parameters: ")
			for i, param := range analysis.Parameters {
				if i > 0 {
					prompt.WriteString(", ")
				}
				prompt.WriteString(param.Name)
				if param.Type != "" {
					prompt.WriteString(fmt.Sprintf(" (%s)", param.Type))
				}
			}
			prompt.WriteString("\n")
		}
		if analysis.ReturnType != "" {
			prompt.WriteString(fmt.Sprintf("- Returns: %s\n", analysis.ReturnType))
		}
		prompt.WriteString("\n")
	}

	// Add requirements
	if len(request.Requirements) > 0 {
		prompt.WriteString("Documentation requirements:\n")
		for _, req := range request.Requirements {
			prompt.WriteString(fmt.Sprintf("- %s\n", req))
		}
		prompt.WriteString("\n")
	}

	// Add format specification
	format := request.Format
	if format == "" {
		format = da.config.DefaultFormat
	}
	prompt.WriteString(fmt.Sprintf("Format: %s\n", format))

	// Add style guidelines
	if request.Style != nil || da.config.LanguageStyles != nil {
		prompt.WriteString("Style guidelines:\n")
		style := request.Style
		if style == nil && request.Target != nil {
			if langStyle, exists := da.config.LanguageStyles[request.Target.Language]; exists {
				style = langStyle
			} else {
				style = &da.config.DocumentationStyle
			}
		}

		if style != nil {
			prompt.WriteString(fmt.Sprintf("- Comment style: %s\n", style.CommentStyle))
			prompt.WriteString(fmt.Sprintf("- Max line length: %d\n", style.MaxLineLength))
			if len(style.Sections) > 0 {
				prompt.WriteString("- Required sections: ")
				for i, section := range style.Sections {
					if i > 0 {
						prompt.WriteString(", ")
					}
					prompt.WriteString(section.Name)
				}
				prompt.WriteString("\n")
			}
		}
		prompt.WriteString("\n")
	}

	// Add context information
	if request.Target != nil && request.Target.Context != nil {
		da.addContextToPrompt(&prompt, request.Target.Context)
	}

	// Add options
	if request.Options != nil {
		prompt.WriteString("Options:\n")
		prompt.WriteString(fmt.Sprintf("- Detail level: %s\n", request.Options.DetailLevel))
		prompt.WriteString(fmt.Sprintf("- Target audience: %s\n", request.Options.Audience))
		if request.Options.IncludeExamples {
			prompt.WriteString("- Include usage examples\n")
		}
		if request.Options.IncludeDiagrams {
			prompt.WriteString("- Include diagrams where helpful\n")
		}
		prompt.WriteString("\n")
	}

	prompt.WriteString("Generate clear, comprehensive, and well-structured documentation.")

	return prompt.String()
}

func (da *DocumentationAgent) buildUpdatePrompt(request *DocumentationRequest, existingAnalysis *DocumentationAnalysis, codeAnalysis *CodeAnalysisResult, updateNeeds *UpdateNeeds) string {
	var prompt strings.Builder

	prompt.WriteString("Update the following documentation based on code changes:\n\n")

	prompt.WriteString("Current Documentation:\n")
	prompt.WriteString(request.ExistingDoc)
	prompt.WriteString("\n\n")

	prompt.WriteString("Updated Code:\n```")
	if request.Target != nil {

		prompt.WriteString(request.Target.Language)
	}
	prompt.WriteString("\n")
	prompt.WriteString(request.Code)
	prompt.WriteString("\n```\n\n")

	if updateNeeds != nil {
		prompt.WriteString("Required updates:\n")
		for _, update := range updateNeeds.Updates {
			prompt.WriteString(fmt.Sprintf("- %s: %s\n", update.Type, update.Description))
		}
		prompt.WriteString("\n")
	}

	prompt.WriteString(" Maintain the original style and tone of the documentation. Ensure accuracy and completeness.\n")

	return prompt.String()
}
func (da *DocumentationAgent) buildImprovementPrompt(request *DocumentationRequest, qualityAnalysis *DocumentationQualityAnalysis, improvements *ImprovementOpportunities) string {
	var prompt strings.Builder
	prompt.WriteString("Improve the following documentation based on quality analysis:\n\n")
	prompt.WriteString("Current Documentation:\n")
	prompt.WriteString(request.ExistingDoc)
	prompt.WriteString("\n\n")

	if qualityAnalysis != nil {
		prompt.WriteString("Quality Issues Identified:\n")
		for _, issue := range qualityAnalysis.Issues {
			prompt.WriteString(fmt.Sprintf("- %s: %s\n", issue.Type, issue.Description))
		}
		prompt.WriteString("\n")
	}

	if improvements != nil {
		prompt.WriteString("Improvement Opportunities:\n")
		for _, improvement := range improvements.Opportunities {
			prompt.WriteString(fmt.Sprintf("- %s: %s\n", improvement.Type, improvement.Description))
		}
		prompt.WriteString("\n")
	}
	prompt.WriteString(" Enhance clarity, completeness, and usefulness while maintaining the original style and tone.\n")

	return prompt.String()

}

func (da *DocumentationAgent) buildAPIDocumentationPrompt(request *DocumentationRequest, apiAnalysis *APIAnalysisResult) string {
	var prompt strings.Builder

	prompt.WriteString("Generate comprehensive API documentation for the following code:\n\n")
	prompt.WriteString("API Code:\n```")
	if request.Target != nil {
		prompt.WriteString(request.Target.Language)
	}
	prompt.WriteString("\n")
	prompt.WriteString(request.Code)
	prompt.WriteString("\n```\n\n")

	if apiAnalysis != nil {
		prompt.WriteString("API Structure Analysis:\n")
		for _, endpoint := range apiAnalysis.Endpoints {
			prompt.WriteString(fmt.Sprintf("- Endpoint: %s %s\n", endpoint.Method, endpoint.Path))
			if endpoint.Description != "" {
				prompt.WriteString(fmt.Sprintf("  Description: %s\n", endpoint.Description))
			}
			if len(endpoint.Parameters) > 0 {
				prompt.WriteString("  Parameters:\n")
				for _, param := range endpoint.Parameters {
					prompt.WriteString(fmt.Sprintf("    - %s (%s): %s\n", param.Name, param.Type, param.Description))
				}
			}
			if endpoint.ReturnType != "" {
				prompt.WriteString(fmt.Sprintf("  Returns: %s\n", endpoint.ReturnType))
			}
			prompt.WriteString("\n")

		}
		prompt.WriteString(fmt.Sprintf("- Methods: %v\n", apiAnalysis.Methods))
		prompt.WriteString(fmt.Sprintf("- Authentication: %s\n", apiAnalysis.Authtype))
		prompt.WriteString("\n")

	}
	prompt.WriteString("Include: endpoints, descriptions, parameters, responses, examples, error codes, return types, authentication details and usage examples.\n")
	return prompt.String()
}

func (da *DocumentationAgent) buildReadmePrompt(request *DocumentationRequest, projectAnalysis *ProjectAnalysisResult) string {
	var prompt strings.Builder
	prompt.WriteString("Generate a comprehensive README.md for the following project:\n\n")
	prompt.WriteString("Project Information:\n")
	if projectAnalysis != nil {
		prompt.WriteString(fmt.Sprintf("- Name: %s\n", projectAnalysis.Name))
		prompt.WriteString(fmt.Sprintf("- Description: %s\n", projectAnalysis.Description))
		prompt.WriteString(fmt.Sprintf("- Version: %s\n", projectAnalysis.Version))
		prompt.WriteString(fmt.Sprintf("- Languages: %v\n", projectAnalysis.Languages))
		prompt.WriteString(fmt.Sprintf("- Dependencies: %v\n", projectAnalysis.Dependencies))
		prompt.WriteString(fmt.Sprintf("- Entry points: %v\n", projectAnalysis.EntryPoints))
		prompt.WriteString("\n")
	}

	if request.Target != nil && request.Target.Context != nil && request.Target.Context.ProjectInfo != nil {
		info := request.Target.Context.ProjectInfo
		prompt.WriteString("Project Info:\n")
		prompt.WriteString(fmt.Sprintf("- Name: %s\n", info.Name))
		prompt.WriteString(fmt.Sprintf("- Description: %s\n", info.Description))
		prompt.WriteString(fmt.Sprintf("- Version: %s\n", info.Version))
		prompt.WriteString(fmt.Sprintf("- License: %s\n", info.License))
		prompt.WriteString("\n")

	}
	prompt.WriteString("Include: description, installation, usage, API reference, contributing guidelines, and license information.")
	return prompt.String()

}

func (da *DocumentationAgent) buildInlineDocumentationPrompt(request *DocumentationRequest, inlineAnalysis *InlineDocumentationAnalysis) string {
	var prompt strings.Builder
	prompt.WriteString("Generate inline documentation for the following code:\n\n")
	prompt.WriteString("Code:\n```")
	if request.Target != nil {
		prompt.WriteString(request.Target.Language)
	}
	prompt.WriteString("\n")
	prompt.WriteString(request.Code)
	prompt.WriteString("\n```\n\n")

	if inlineAnalysis != nil && len(inlineAnalysis.InlinePoints) > 0 {
		prompt.WriteString("Suggested Inline Documentation Points:\n")
		for _, point := range inlineAnalysis.InlinePoints {
			prompt.WriteString(fmt.Sprintf("- Line %d: %s\n", point.LineNumber, point.Description))
		}
		prompt.WriteString("\n")
	}

	prompt.WriteString("Ensure the inline documentation is clear, concise, and follows best practices for the specified programming language.\n")

	return prompt.String()
}

func (da *DocumentationAgent) addContextToPrompt(prompt *strings.Builder, context *DocumentationContext) {
	if context.ProjectInfo != nil {
		prompt.WriteString("Project Information:\n")
		prompt.WriteString(fmt.Sprintf("- Project: %s\n", context.ProjectInfo.Name))

		prompt.WriteString("\n")
	}
	if context.ModuleInfo != nil {
		prompt.WriteString("Module Information:\n")
		prompt.WriteString(fmt.Sprintf("- Module: %s\n", context.ModuleInfo.Name))
		prompt.WriteString("\n")
	}
	if len(context.RelatedCode) > 0 {
		prompt.WriteString("Related Code Snippets:\n")
		for _, code := range context.RelatedCode {
			prompt.WriteString(fmt.Sprintf("```%s\n%s\n```\n", request.Target.Language, code))
		}
		prompt.WriteString("\n")
	}
	if len(context.Dependencies) > 0 {
		prompt.WriteString("Dependencies:\n")
		for _, dep := range context.Dependencies {
			prompt.WriteString(fmt.Sprintf("- %s\n", dep))
		}
		prompt.WriteString("\n")
	}
	if len(context.UsageExamples) > 0 {
		prompt.WriteString("Usage Examples:\n")
		for _, example := range context.UsageExamples {
			prompt.WriteString(fmt.Sprintf("```%s\n%s\n```\n", request.Target.Language, example))
		}
		prompt.WriteString("\n")
	}
}

//Analysis Methods
func (da *DocumentationAgent) analyzeCode(code string, target *DocumentationTarget) (*CodeAnalysisResult, error) {
if da.codeAnalyzer == nil {
return nil, fmt.Errorf("code analyzer not initialized")
}
language := "unknown"
if target != nil {
	language = target.Language
}

analysis := da.codeAnalyzer.Analyze(code, language)

// Convert to documentation-specific analysis
result := &CodeAnalysisResult{
	Purpose:        da.inferPurpose(code, target),
	ComplexityLevel: da.categorizeComplexity(analysis.CyclomaticComplexity),
	Parameters:     da.extractParameters(code, language),
	ReturnType:     da.extractReturnType(code, language),
	Exceptions:     da.extractExceptions(code, language),
	Dependencies:   da.extractDependencies(code, language),
}

return result, nil
}

func (da *DocumentationAgent) categorizeComplexity(cyclomaticComplexity int) string {
	if cyclomaticComplexity <= 5 {
		return "Low"
	} else if cyclomaticComplexity <= 10 {
		return "Medium"
	} else if cyclomaticComplexity <= 20 {
		return "High"
	}
	return "Very High"
}

func (da *DocumentationAgent) extractParameters(code string, language string) []*Parameter {
	var parameters []*Parameter

	// Simple parameter extraction based on language
	switch language {
	case "go":
		// Go function signature pattern
		funcRegex := regexp.MustCompile(`func\s+\w+\s*\(([^)]*)\)`)
		matches := funcRegex.FindStringSubmatch(code)
		if len(matches) > 1 {
			paramStr := matches[1]
			if paramStr != "" {
				paramParts := strings.Split(paramStr, ",")
				for _, part := range paramParts {
					part = strings.TrimSpace(part)
					if part != "" {
						// Split name and type
						words := strings.Fields(part)
						if len(words) >= 2 {
							parameters = append(parameters, &Parameter{
								Name: words[0],
								Type: strings.Join(words[1:], " "),
							})
						}
					}
				}
			}
		}
	case "python":
		// Python function signature pattern
		funcRegex := regexp.MustCompile(`def\s+\w+\s*\(([^)]*)\)`)
		matches := funcRegex.FindStringSubmatch(code)
		if len(matches) > 1 {
			paramStr := matches[1]
			if paramStr != "" {
				paramParts := strings.Split(paramStr, ",")
				for _, part := range paramParts {
					part = strings.TrimSpace(part)
					if part != "" && part != "self" {
						// Handle type hints
						if strings.Contains(part, ":") {
							nameParts := strings.Split(part, ":")
							parameters = append(parameters, &Parameter{
								Name: strings.TrimSpace(nameParts[0]),
								Type: strings.TrimSpace(nameParts[1]),
							})
						} else {
							parameters = append(parameters, &Parameter{
								Name: part,
								Type: "any",
							})
						}
					}
				}
			}
		}
	case "javascript", "typescript":
		// JavaScript/TypeScript function pattern
		funcRegex := regexp.MustCompile(`function\s+\w+\s*\(([^)]*)\)|(?:\w+\s*=\s*)?\([^)]*\)\s*=>|\w+\s*\([^)]*\)`)
		// This is simplified - would need more sophisticated parsing
		parameters = append(parameters, &Parameter{Name: "params", Type: "object"})
	}

	return parameters
}


func (da *DocumentationAgent) extractReturnType(code string, language string) string {
	switch language {
	case "go":
		// Go return type pattern
		funcRegex := regexp.MustCompile(func\s+\w+\s*\([^)]*\)\s*([^{]*){)
		matches := funcRegex.FindStringSubmatch(code)
		if len(matches) > 1 {
			returnType := strings.TrimSpace(matches[1])
			if returnType != "" {
				return returnType
			}
		}
	case "python":
		// Python return type annotation
		funcRegex := regexp.MustCompile(def\s+\w+\s*\([^)]*\)\s*->\s*([^:]+):)
		matches := funcRegex.FindStringSubmatch(code)
		if len(matches) > 1 {
			return strings.TrimSpace(matches[1])
		}
	case "typescript":
		// TypeScript return type annotation
		funcRegex := regexp.MustCompile(function\s+\w+\s*\([^)]*\)\s*:\s*([^{]+){)
		matches := funcRegex.FindStringSubmatch(code)
		if len(matches) > 1 {
			return strings.TrimSpace(matches[1])
		}
	}

	// Infer return type from return statements
	if strings.Contains(code, "return true") || strings.Contains(code, "return false") {
		return "boolean"
	}

	if regexp.MustCompile(`return\s+\d+`).MatchString(code) {
		return "number"
	}

	if regexp.MustCompile(`return\s+".*"`).MatchString(code) {
		return "string"
	}

	return "void"
}


func (da *DocumentationAgent) extractExceptions(code string, language string) []string {
	var exceptions []string
	// Look for exception patterns
	exceptionPatterns := []string{
		"throw ", "raise ", "panic(", "error(", "Error(", "Exception(",
	}

	for _, pattern := range exceptionPatterns {
		if strings.Contains(code, pattern) {
			exceptions = append(exceptions, "May throw exceptions")
			break
		}
	}

	return exceptions
}

func (da *DocumentationAgent) extractDependencies(code string, language string) []string {
	var dependencies []string
	// Extract import statements
	switch language {
	case "go":
		importRegex := regexp.MustCompile(`import\s+"([^"]+)"`)
		matches := importRegex.FindAllStringSubmatch(code, -1)
		for _, match := range matches {
			if len(match) > 1 {
				dependencies = append(dependencies, match[1])
			}
		}
	case "python":
		importRegex := regexp.MustCompile(`(?:from\s+(\S+)\s+import|import\s+(\S+))`)
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







func (da *DocumentationAgent) inferPurpose(code string, target *DocumentationTarget) string {
	// Simple purpose inference based on code patterns
	codeLower := strings.ToLower(code)
	if strings.Contains(codeLower, "test") || strings.Contains(codeLower, "spec") {
		return "Test function"
	}

	if strings.Contains(codeLower, "validate") || strings.Contains(codeLower, "check") {
		return "Validation function"
	}

	if strings.Contains(codeLower, "parse") || strings.Contains(codeLower, "decode") {
		return "Parser function"
	}

	if strings.Contains(codeLower, "format") || strings.Contains(codeLower, "render") {
		return "Formatting function"
	}

	if strings.Contains(codeLower, "calculate") || strings.Contains(codeLower, "compute") {
		return "Calculation function"
	}

	if target != nil {
		switch target.Type {
		case TargetClass:
			return "Class definition"
		case TargetInterface:
			return "Interface definition"
		case TargetConstant:
			return "Constant definition"
		case TargetVariable:
			return "Variable declaration"
		}
	}

	return "Function implementation"
}










func (da *DocumentationAgent) analyzeExistingDocumentation(existingDoc string) *DocumentationAnalysis {
	analysis := &DocumentationAnalysis{
	HasSummary:     da.hasSummary(existingDoc),
	HasParameters:  da.hasParameterDocs(existingDoc),
	HasReturns:     da.hasReturnDocs(existingDoc),
	HasExamples:    da.hasExamples(existingDoc),
	QualityScore:   da.assessDocQuality(existingDoc),
	Sections:       da.identifySections(existingDoc),
	Format:         da.detectFormat(existingDoc),
	}
	return analysis
}


// Documentation analysis helper methods
func (da DocumentationAgent) hasSummary(doc string) bool {
	// Check if documentation has a summary/description
	lines := strings.Split(doc, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if len(line) > 20 && !strings.HasPrefix(line, "@") && !strings.HasPrefix(line, "") {
			return true
		}
	}
	return false
}

func (da *DocumentationAgent) hasParameterDocs(doc string) bool {
	// Check for parameter documentation
	paramPatterns := []string{"@param", "@parameter", "Args:", "Parameters:", "param "}
	docLower := strings.ToLower(doc)
	for _, pattern := range paramPatterns {
		if strings.Contains(docLower, strings.ToLower(pattern)) {
			return true
		}
	}
	return false
}

func (da *DocumentationAgent) hasReturnDocs(doc string) bool {
	// Check for return documentation
	returnPatterns := []string{"@return", "@returns", "Returns:", "Return:", "returns "}
	docLower := strings.ToLower(doc)
	for _, pattern := range returnPatterns {
		if strings.Contains(docLower, strings.ToLower(pattern)) {
			return true
		}
	}
	return false
}

func (da *DocumentationAgent) hasExamples(doc string) bool {
	// Check for code examples
	examplePatterns := []string{"```", "Example:", "@example", "Usage:", "Sample:"}
	docLower := strings.ToLower(doc)
	for _, pattern := range examplePatterns {
		if strings.Contains(docLower, strings.ToLower(pattern)) {
			return true
		}
	}
	return false
}

func (da *DocumentationAgent) assessDocQuality(doc string) float32 {
	quality := float32(0.5) // Base quality
	if da.hasSummary(doc) {
		quality += 0.2
	}

	if da.hasParameterDocs(doc) {
		quality += 0.15
	}

	if da.hasReturnDocs(doc) {
		quality += 0.1
	}

	if da.hasExamples(doc) {
		quality += 0.15
	}

	// Length bonus/penalty
	wordCount := len(strings.Fields(doc))
	if wordCount > 10 && wordCount < 200 {
		quality += 0.1
	} else if wordCount > 300 {
		quality -= 0.1
	}

	if quality > 1.0 {
		quality = 1.0
	}

	return quality
}

func (da *DocumentationAgent) identifySections(doc string) []string {
	var sections []string
	lines := strings.Split(doc, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)

		// Look for section headers
		if strings.HasSuffix(line, ":") && len(line) > 3 && len(line) < 30 {
			sections = append(sections, strings.TrimSuffix(line, ":"))
		}

		// Look for @ tags
		if strings.HasPrefix(line, "@") {
			parts := strings.Fields(line)
			if len(parts) > 0 {
				sections = append(sections, parts[0])
			}
		}
	}

	return sections
}

func (da *DocumentationAgent) detectFormat(doc string) DocFormat {
	if strings.Contains(doc, "/**") || strings.Contains(doc, "@param") {
		return FormatJSDoc
	}
	if strings.Contains(doc, "\"\"\"") {
		return FormatPyDoc
	}

	if strings.Contains(doc, "//") {
		return FormatGoDoc
	}

	if strings.Contains(doc, "```") || strings.Contains(doc, "#") {
		return FormatMarkdown
	}

	return FormatMarkdown // Default
}

// ################################################################################











func (da *DocumentationAgent) analyzeAPI(code string, target *DocumentationTarget) (*APIAnalysisResult, error) {
	// Analyze API structure, endpoints, parameters, responses
	result := &APIAnalysisResult{
	Endpoints:    da.extractEndpoints(code, target),
	DataModels:   da.extractDataModels(code, target),
	AuthMethods:  da.extractAuthMethods(code, target),
	ErrorCodes:   da.extractErrorCodes(code, target),
	}
	return result, nil
}


func (da *DocumentationAgent) analyzeAPI(code string, target *DocumentationTarget) (*APIAnalysisResult, error) {
	// Analyze API structure, endpoints, parameters, responses
	result := &APIAnalysisResult{
	Endpoints:    da.extractEndpoints(code, target),
	DataModels:   da.extractDataModels(code, target),
	AuthMethods:  da.extractAuthMethods(code, target),
	ErrorCodes:   da.extractErrorCodes(code, target),
	}
	return result, nil
}

	//Basic project analysis logic
func (da *DocumentationAgent) analyzeInlineDocumentationNeeds(code string, target *DocumentationTarget) *InlineAnalysisResult {
	return &InlineAnalysisResult{
		InlinePoints: []InlinePoint{},

	}
}

func (da *DocumentationAgent) analyzeDocumentationQuality(doc string) *QualityAnalysis {
	return &QualityAnalysis{
		Score: 0.7,
		Issues: []QualityIssue{},
		Completeness: 0.8,
		Clarity: 0.75,
		Accuracy: 0.9,
		Consistency: 0.85,
		Usefulness: 0.8,
		Maintainability: 0.6,
	}
}

func (da *DocumentationAgent) identifyUpdateNeeds(existing *DocumentationAnalysis, code *CodeAnalysisResult) *UpdateNeeds {
	return &UpdateNeeds{
		Updates: []*UpdateNeed{},
	}
}

func (da *DocumentationAgent) identifyImprovementOpportunities(quality *QualityAnalysis) *ImprovementOpportunities {
	return &ImprovementOpportunities{
		Opportunities: []*ImprovementOpportunity{},
	}
}

//Generation helper methods

func (da *DocumentationAgent) generateExamples(request *DocumentationRequest, analysis *CodeAnalysisResult) []*CodeExample {
	if da.exampleGenerator == nil {
		return nil
	}
	return da.exampleGenerator.GenerateExamples(request, analysis)
}


func (da *DocumentationAgent) generateDiagrams(request *DocumentationRequest, analysis *CodeAnalysisResult) []*Diagram {
	if da.diagramGenerator == nil {
		return nil
	}
	return da.diagramGenerator.GenerateDiagrams(request, analysis)
}


func (da *DocumentationAgent) generateLinks(request *DocumentationRequest, analysis *CodeAnalysisResult) []*DocumentationLink {
	if da.linkResolver == nil {
		return nil
	}
	return da.linkResolver.GenerateLinks(request, analysis)
}



func (da *DocumentationAgent) generateAPIExamples(request *DocumentationRequest, analysis *APIAnalysisResult) []*CodeExample {
	example := &CodeExample{
		Title:       "API Usage Example",
		Description: "An example of how to call the API endpoint.",
		Code:        "curl -X GET https://api.example.com/endpoint",
		Language:    "bash",
	},
	return []*CodeExample{example}
}

func (da *DocumentationAgent) generateSuggestions(request *DocumentationRequest, analysis *CodeAnalysisResult, doc string) []*DocumentationSuggestion {
	suggestion := &DocumentationSuggestion{
		Title:       "Add Error Handling Section",
		Description: "Consider adding a section on error handling for this function.",
		Priority:    PriorityMedium,
	}
	return []*DocumentationSuggestion{suggestion}
}

func (da *DocumentationAgent) generateSuggestions(request *DocumentationRequest, analysis *CodeAnalysisResult, generatedDoc string) []*DocumentationSuggestion {
	var suggestions []*DocumentationSuggestion
	// Suggest examples if not included
	if !da.hasExamples(generatedDoc) {
		suggestions = append(suggestions, &DocumentationSuggestion{
			Type:        SuggestionTypeBestPractice,
			Title:       "Add Usage Examples",
			Description: "Consider adding code examples to illustrate usage",
			Priority:    PriorityMedium,
			EstimatedEffort: "5-10 minutes",
		})
	}
	// Suggest parameter documentation if missing
	if analysis != nil && len(analysis.Parameters) > 0 && !da.hasParameterDocs(generatedDoc) {
		suggestions = append(suggestions, &DocumentationSuggestion{
			Type:        SuggestionTypeBestPractice,
			Title:       "Document Parameters",
			Description: "Add detailed parameter descriptions",
			Priority:    PriorityHigh,
			EstimatedEffort: "2-5 minutes",
		})
	}

	// Suggest return type documentation if missing
	if analysis != nil && analysis.ReturnType != "" && !da.hasReturnDocs(generatedDoc) {
		suggestions = append(suggestions, &DocumentationSuggestion{
			Type:        SuggestionTypeBestPractice,
			Title:       "Document Return Value",
			Description: "Add description of the return value",
			Priority:    PriorityMedium,
			EstimatedEffort: "1-2 minutes",
		})
	}

	return suggestions
}



func (da *DocumentationAgent) generateImprovementSuggestions(improvements *ImprovementOpportunities, doc string) []*DocumentationSuggestion {
	suggestion := &DocumentationSuggestion{
		Title:       "Enhance Clarity",
		Description: "Rephrase sections to improve clarity and readability.",
		Priority:    PriorityMedium,
	}
	return []*DocumentationSuggestion{suggestion}
}

// Quality and coverage analysis
func (da *DocumentationAgent) performQualityCheck(doc string, request *DocumentationRequest) (*DocumentationQuality, error) {
	if da.qualityChecker == nil {
		return nil, fmt.Errorf("quality checker not initialized")
	}
	return da.qualityChecker.CheckQuality(doc, request)
}

func (da *DocumentationAgent) performCoverageAnalysis(target *DocumentationTarget, doc string) (*CoverageReport, error) {
	if da.coverageAnalyzer == nil {
		return nil, fmt.Errorf("coverage analyzer not initialized")
	}
	return da.coverageAnalyzer.AnalyzeCoverage(target, doc)
}

// Utility methods
func (da *DocumentationAgent) extractDocumentation(text string, format DocFormat) string {
	// Extract documentation from LLM response
	// Look for documentation blocks first
	var docBlockRegex *regexp.Regexp

	switch format {
		case FormatMarkdown:
			docBlockRegex = regexp.MustCompile("```(?:markdown)?\n?(.*?)\n?```")
		case FormatJSDoc:
			docBlockRegex = regexp.MustCompile("/\\*\\*(.*?)\\*/")
		case FormatJavaDoc:
			docBlockRegex = regexp.MustCompile("/\\*\\*(.*?)\\*/")
		case FormatPyDoc:
			docBlockRegex = regexp.MustCompile("\"\"\"(.*?)\"\"\"")
		default:
			docBlockRegex = regexp.MustCompile("```(?:" + string(format) + ")?\n?(.*?)\n?```")
		}

		matches := docBlockRegex.FindStringSubmatch(text)
		if len(matches) > 1 {
			return strings.TrimSpace(matches[1])
		}

	// If no blocks found, clean and return the text
	return strings.TrimSpace(text)
}



func (da *DocumentationAgent) calculateConfidence(request *DocumentationRequest, response *DocumentationResponse) float64 {
	confidence := 0.8 // Base confidence

	// Adjust based on request type
	switch request.Type {
		case DocTypeGenerate:
			confidence = 0.75
		case DocTypeUpdate:
			confidence = 0.85
		case DocTypeImprove:
			confidence = 0.80
		case DocTypeAPIDoc:
			confidence = 0.85
	}

	// Boost confidence based on quality score
	if response.Quality != nil {
		qualityBonus := float64(response.Quality.OverallScore) * 0.2
		confidence += qualityBonus
	}

	// Boost confidence if examples are included
	if len(response.Examples) > 0 {
		confidence += 0.1
	}

	// Cap at 1.0
	if confidence > 1.0 {
		confidence = 1.0
	}

	return confidence
}




func (da *DocumentationAgent) parseDocumentationRequest(request *AgentRequest) (*DocumentationRequest, error) {
	// Try to parse from parameters first
	if params, ok := request.Parameters["documentation_request"].(*DocumentationRequest); ok {
		return params, nil
	}
	// Parse from query and context
	docRequest := &DocumentationRequest{
		Type:   da.inferDocumentationType(request.Intent.Type),
		Format: da.config.DefaultFormat,
	}

	// Extract code from context
	if request.Context != nil && request.Context.SelectedText != "" {
		docRequest.Code = request.Context.SelectedText
		
		// Create target from context
		docRequest.Target = &DocumentationTarget{
			Type:     da.inferTargetType(request.Context.SelectedText),
			FilePath: request.Context.CurrentFile,
			Language: request.Context.ProjectLanguage,
		}
	}

	return docRequest, nil
}

func (da *DocumentationAgent) inferDocumentationType(intentType IntentType) DocumentationType {
	switch intentType {
		case IntentDocGeneration:
			return DocTypeGenerate
		case IntentDocUpdate:
			return DocTypeUpdate
		case IntentDocReview:
			return DocTypeImprove
		default:
			return DocTypeGenerate
		}
}


func (da *DocumentationAgent) inferTargetType(code string) TargetType {
codeLower := strings.ToLower(code)
if strings.Contains(codeLower, "func ") || strings.Contains(codeLower, "function ") || strings.Contains(codeLower, "def ") {
	return TargetFunction
}

if strings.Contains(codeLower, "class ") {
	return TargetClass
}

if strings.Contains(codeLower, "interface ") {
	return TargetInterface
}

if strings.Contains(codeLower, "const ") || strings.Contains(codeLower, "final ") {
	return TargetConstant
}

if strings.Contains(codeLower, "var ") || strings.Contains(codeLower, "let ") {
	return TargetVariable
}
return TargetFunction
}


func (da *DocumentationAgent) validateDocumentationRequest(request *DocumentationRequest) error {
	if request.Type == "" {
		request.Type = DocTypeGenerate
	}
	if request.Code == "" && request.Type != DocTypeTranslate {
		return fmt.Errorf("code is required for documentation generation")
	}

	if request.Type == DocTypeUpdate && request.ExistingDoc == "" {
		return fmt.Errorf("existing documentation is required for update")
	}

	if request.Format == "" {
		request.Format = da.config.DefaultFormat
	}

	return nil
}


// ##################################################################################################
/ Component initialization and management
func (da *DocumentationAgent) initializeCapabilities() {
	da.capabilities = &AgentCapabilities{
		AgentType: AgentTypeDocumentation,
		SupportedIntents: []IntentType{
			IntentDocGeneration,
			IntentDocUpdate,
			IntentDocReview,
		},
		SupportedLanguages: []string{
			"go", "python", "javascript", "typescript", "java",
			"cpp", "c", "csharp", "ruby", "php", "rust",
		},
		SupportedFileTypes: []string{
			"go", "py", "js", "ts", "java", "cpp", "c", "cs", "rb", "php", "rs",
			"md", "rst", "txt",
		},
		MaxContextSize:    4096,
		SupportsStreaming: da.config.EnableStreaming,
		SupportsAsync:     true,
		RequiresContext:   false,
		Capabilities: map[string]interface{}{
			"doc_generation":      da.config.EnableDocGeneration,
			"doc_update":          da.config.EnableDocUpdate,
			"api_doc_generation":  da.config.EnableAPIDocGeneration,
			"readme_generation":   da.config.EnableReadmeGeneration,
			"example_generation":  da.config.EnableExampleGeneration,
			"diagram_generation":  da.config.EnableDiagramGeneration,
			"multi_format":        da.config.EnableMultiFormat,
			"quality_check":       da.config.EnableQualityCheck,
			"coverage_analysis":   da.config.EnableCoverageAnalysis,
		},
	}
}

func (da *DocumentationAgent) initializeComponents() {
	// Initialize documentation analyzer
	da.docAnalyzer = NewDocumentationAnalyzer()

	// Initialize code analyzer
	da.codeAnalyzer = NewCodeAnalyzer()

	// Initialize style analyzer
	da.styleAnalyzer = NewDocumentationStyleAnalyzer()

	// Initialize documentation generator
	da.docGenerator = NewDocumentationGenerator(da.config)

	// Initialize template engine
	if da.config.TemplateDirectory != "" {
		da.templateEngine = NewDocumentationTemplateEngine(da.config.TemplateDirectory)
	}

	// Initialize formatters
	da.initializeFormatters()

	// Initialize quality checker
	if da.config.EnableQualityCheck {
		da.qualityChecker = NewDocumentationQualityChecker()
	}

	// Initialize consistency checker
	if da.config.EnableConsistencyCheck {
		da.consistencyChecker = NewConsistencyChecker()
	}

	// Initialize coverage analyzer
	if da.config.EnableCoverageAnalysis {
		da.coverageAnalyzer = NewCoverageAnalyzer()
	}

	// Initialize example generator
	if da.config.EnableExampleGeneration {
		da.exampleGenerator = NewCodeExampleGenerator(da.llmProvider)
	}

	// Initialize diagram generator
	if da.config.EnableDiagramGeneration {
		da.diagramGenerator = NewDiagramGenerator()
	}

	// Initialize link resolver
	if da.config.EnableLinkGeneration {
		da.linkResolver = NewLinkResolver(da.indexer)
	}
}

func (da *DocumentationAgent) initializeFormatters() {
	// Initialize formatters for different documentation formats
	da.formatters[string(FormatMarkdown)] = NewMarkdownFormatter()
	da.formatters[string(FormatJSDoc)] = NewJSDocFormatter()
	da.formatters[string(FormatJavaDoc)] = NewJavaDocFormatter()
	da.formatters[string(FormatPyDoc)] = NewPyDocFormatter()
	da.formatters[string(FormatGoDoc)] = NewGoDocFormatter()
	da.formatters[string(FormatRustDoc)] = NewRustDocFormatter()
}


// Metrics methods
func (da *DocumentationAgent) updateMetrics(docType DocumentationType, format DocFormat, success bool, duration time.Duration) {
	da.metrics.mu.Lock()
	defer da.metrics.mu.Unlock()

	da.metrics.TotalRequests++
	da.metrics.RequestsByType[docType]++
	da.metrics.RequestsByFormat[format]++

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


//Agent interface methods
func (da *DocumentationAgent) GetCapabilities() *AgentCapabilities {
	return da.capabilities
}

func (da *DocumentationAgent) GetType() AgentType {
	return AgentTypeDocumentation
}

func (da *DocumentationAgent) GetVersion() string {
	return da.capabilities.Version
}

func (da *DocumentationAgent) GetStatus() AgentStatus {
	da.mu.RLock()
	defer da.mu.RUnlock()
	return da.status
}

func (da *DocumentationAgent) Initialize(config interface{}) error {
	if cfg, ok := config.(*DocumentationAgentConfig); ok {
		da.config = cfg
		da.initializeComponents()
		return nil
	}
	return fmt.Errorf("invalid config type")
}

// SetConfig updates the agent's configuration dynamically
func (da *DocumentationAgent) SetConfig(config interface{}) error {
  appConfig, ok := config.(*app.DocumentationConfig)
  if !ok {
          return fmt.Errorf("invalid config type for DocumentationAgent, expected *app.DocumentationConfig")
  }

  da.mu.Lock()
  defer da.mu.Unlock()

  // Convert app.DocumentationConfig to internal DocumentationAgentConfig
  newConfig := &DocumentationAgentConfig{
          EnableDocGeneration:    appConfig.EnableDocGeneration,
          EnableDocUpdate:        appConfig.EnableDocUpdate,
          EnableAPIDocGeneration: appConfig.EnableAPIDocGeneration,
          EnableReadmeGeneration: appConfig.EnableReadmeGeneration,
          EnableExampleGeneration: appConfig.EnableExampleGeneration,
          DefaultFormat:          appConfig.DefaultFormat,
          LLMModel:              appConfig.LLMModel,
          MaxTokens:             appConfig.MaxTokens,
          Temperature:           appConfig.Temperature,
          // Preserve existing internal settings if any
  }

  // Update configuration
  da.config = newConfig

  // Re-initialize components with new config
  da.initializeComponents()

  da.logger.Info("DocumentationAgent configuration updated",
          "default_format", newConfig.DefaultFormat,
          "llm_model", newConfig.LLMModel,
          "max_tokens", newConfig.MaxTokens)

  return nil
}



func (da *DocumentationAgent) Start() error {
	da.mu.Lock()
	defer da.mu.Unlock()
	da.status = StatusIdle
	da.logger.Info("Documentation agent started")
	return nil
}

func (da *DocumentationAgent) Stop() error {
	da.mu.Lock()
	da.logger.Info("Documentation agent stopped")
	return nil
}

func (da *DocumentationAgent) HealthCheck() *agents.HealthStatus {
	startTime := time.Now()
	status := &agents.HealthStatus{
		LastCheckTime:      startTime,
		DependenciesStatus: make(map[string]*agents.HealthStatus),
		Details:            make(map[string]interface{}),
	}

	// Check LLM provider
	if da.llmProvider == nil {
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
		Model:     da.config.LLMModel,
		MaxTokens: 25,
	}

	_, err := da.llmProvider.Complete(ctx, testRequest)
	status.Latency = time.Since(startTime)

	// Get metrics
	metrics := da.GetMetrics()
	status.ErrorCount = metrics.ErrorCount

	// Evaluate health
	healthConfig := da.getHealthCheckConfig()
	if err != nil {
			status.Status = agents.HealthStatusUnhealthy
			status.Message = fmt.Sprintf("LLM provider error: %v", err)
	} else if status.Latency > time.Duration(healthConfig.MaxLatencyMs)*time.Millisecond {
			status.Status = agents.HealthStatusDegraded
			status.Message = fmt.Sprintf("High latency: %v", status.Latency)
	} else {
			status.Status = agents.HealthStatusHealthy
			status.Message = "Documentation agent operational"
	}

	status.Details["default_format"] = da.config.DefaultFormat
	status.Details["llm_model"] = da.config.LLMModel
	return status
}

func (da *DocumentationAgent) getHealthCheckConfig() *agents.HealthCheckConfig {
  	return &agents.HealthCheckConfig{
		MaxLatencyMs:   3000,
		MaxErrorRate:   0.1,
		HealthCheckTimeout: time.Second * 10,
	}
}

func (da *DocumentationAgent) GetMetrics() *AgentMetrics {
	da.metrics.mu.RLock()
	defer da.metrics.mu.RUnlock()
	return &AgentMetrics{
		RequestsProcessed:   da.metrics.TotalRequests,
		AverageResponseTime: da.metrics.AverageResponseTime,
		LastRequest:         da.metrics.LastRequest,
		SuccessRate:         da.metrics.SuccessRate,
		TotalRequestsByType: make(map[DocumentationType]int64),
		TotalRequestsByFormat: make(map[DocFormat]int64),
	}
}



func (da *DocumentationAgent) HandleRequest(ctx context.Context, request *AgentRequest) (*AgentResponse, error) {
	startTime := time.Now()
	success := false
	var response *DocumentationResponse
	var err error

	// Parse and validate the request
	docRequest, err := da.parseDocumentationRequest(request)
	if err != nil {
		da.logger.Error("Failed to parse documentation request", "error", err)
		return nil, err
	}

	err = da.validateDocumentationRequest(docRequest)
	if err != nil {
		da.logger.Error("Invalid documentation request", "error", err)
		return nil, err
	}

}


// Supporting types for analysis results
type CodeAnalysisResult struct {
	Purpose         string
	ComplexityLevel string
	Parameters      []ParameterInfo
	ReturnType      string
}

type ParameterInfo struct {
	Name string
	Type string
}

type DocumentationAnalysis struct {
	Quality      float64
	Issues       []*QualityIssue
	Completeness float64
	Clarity      float64
	Accuracy     float64
	Consistency  float64
	Usefulness   float64
}

type QualityIssue struct {
	Type        string
	Description string
	LineNumber  int
}

type APIAnalysisResult struct {	
	Endpoints []APIEndpoint
	Methods   []string
	Authtype  string	
}

type APIEndpoint struct {
	Method      string
	Path        string
	Description string
	Parameters  []ParameterInfo
	ReturnType  string
}

type ProjectAnalysisResult struct {
	Name        string
	Type        string
	Description string
	Version     string
	Languages   []string
	Dependencies []string
	EntryPoints []string
}	

type InlineAnalysisResult struct {
	InlinePoints []InlinePoint
}	

type InlinePoint struct {
	LineNumber int
	Description string
}

type DocumentationQualityAnalysis struct {
	Score        float64
	Issues       []QualityIssue
	Completeness float64
	Clarity      float64
	Accuracy     float64
	Consistency  float64
	Usefulness   float64
	Maintainability float64
}

type UpdateNeeds struct {
	Updates []*UpdateNeed
}

type UpdateNeed struct {
	Type        string
	Description string
}

type ImprovementOpportunities struct {
	Opportunities []*ImprovementOpportunity
}

type ImprovementOpportunity struct {
	Type        string
	Description string
}

//Formatters interface and implementations

type Formatter interface {
	Format(doc string) (string, error)
}

type MarkdownFormatter struct{}
func (f *MarkdownFormatter) Format(content string, style *DocStyle) (string, error) {
	return content, nil
}

type HTMLFormatter struct{}
func (f *HTMLFormatter) Format(content string, style *DocStyle) (string, error) {
	return content, nil
}

type PDFFormatter struct{}
func (f *PDFFormatter) Format(content string, style *DocStyle) (string, error) {
	return content, nil
}

type PlainTextFormatter struct{}
func (f *PlainTextFormatter) Format(content string, style *DocStyle) (string, error) {
	return content, nil
}

type JavadocFormatter struct{}
func (f *JavadocFormatter) Format(content string, style *DocStyle) (string, error) {
	return content, nil
}	



//Additional enums and constants
type Priority string
const (
	PriorityLow    Priority = "low"
	PriorityMedium Priority = "medium"
	PriorityHigh   Priority = "high"
)	

type SuggestionType string
const (
	SuggestionTypeContent   SuggestionType = "content"
	SuggestionTypeStructure SuggestionType = "diagram"
	SuggestionTypeStyle     SuggestionType = "style"
	SuggestionTypeExamples  SuggestionType = "examples"
)