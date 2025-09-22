package app

import (
	"context"
	"fmt"
	"html/template"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/logger"
)

type LanguageDetector struct{}
type IndentationManager struct{}
type LineNumberGenerator struct{}
type MarkdownParser struct{}
type EmphasisProcessor struct{}
type HeadingProcessor struct{}
type LinkProcessor struct{}
type TableParser struct{}
type ColumnAligner struct{}
type TableStyleApplicator struct{}
type SortingHandler struct{}
type ListDetector struct{}
type NestedListHandler struct{}
type NumberingManager struct{}
type ListStyleManager struct{}
type ExplanationStructureAnalyzer struct{}
type SectionFormatter struct{}
type ExampleFormatter struct{}
type CrossReferencer struct{}
type DocStructureParser struct{}
type APIDocFormatter struct{}
type TOCGenerator struct{}
type CodeExampleManager struct{}
type ErrorCategorizer struct{}
type StackTraceFormatter struct{}
type ErrorSuggestionProvider struct{}
type ErrorColorizer struct{}
type DiffParser struct{}
type ChangeHighlighter struct{}
type DiffContextProvider struct{}
type SideBySideRenderer struct{}
type MarkdownHTMLRenderer struct{}
type HTMLSanitizer struct{}
type HTMLMinifier struct{}
type AccessibilityEnhancer struct{}
type SEOOptimizer struct{}
type EnrichmentContextProvider struct{}
type EnrichmentQualityChecker struct{}
type LinkContextAnalyzer struct{}
type ReferenceContextProvider struct{}
type CodeIndexer struct{}
type StyleCompiler struct{}
type CustomStyleManager struct{}
type ConsistencyChecker struct{}
type AccessibilityChecker struct{}
type PerformanceAnalyzer struct{}
type TextReadabilityAnalyzer struct{}
type CodeReadabilityAnalyzer struct{}
type StructureReadabilityAnalyzer struct{}
type CommonValidator struct{}
type FormattingStructure struct{}

const (
	FormattingStructureType = "structure"
)

// ResponseFormatter formats AI-generated responses into user-friendly outputs
type ResponseFormatter struct {
	// Core components
	logger logger.Logger

	// Formatting engines
	codeFormatter  *CodeFormatter
	textFormatter  *TextFormatter
	tableFormatter *TableFormatter
	listFormatter  *ListFormatter

	// Specialized formatters
	explanationFormatter   *ExplanationFormatter
	documentationFormatter *DocumentationFormatter
	errorFormatter         *ErrorFormatter
	diffFormatter          *DiffFormatter

	// Template engines
	templateEngine   *TemplateEngine
	markdownRenderer *MarkdownRenderer
	htmlRenderer     *HTMLRenderer

	// Language-specific processors
	languageProcessors map[string]LanguageProcessor
	syntaxHighlighters map[string]SyntaxHighlighter

	// Configuration and settings
	config *ResponseFormatterConfig

	// Output customization
	outputCustomizers map[OutputFormat]OutputCustomizer
	themeManager      *ThemeManager
	styleManager      *StyleManager

	// Enhancement modules
	enrichmentEngine  *ContentEnrichmentEngine
	linkDetector      *LinkDetector
	referenceResolver *ReferenceResolver

	// Quality and validation
	qualityChecker *FormattingQualityChecker
	validator      *OutputValidator

	// Performance optimization
	cache       map[string]*FormattedResponse
	cacheExpiry time.Duration
	cacheMu     sync.RWMutex

	// Metrics and monitoring
	metrics *FormattingMetrics

	// State management
	mu            sync.RWMutex
	isInitialized bool
}

// ResponseFormatterConfig contains configuration for response formatting
type ResponseFormatterConfig struct {
	// Core formatting settings
	EnableCodeFormatting  bool `json:"enable_code_formatting"`
	EnableTextFormatting  bool `json:"enable_text_formatting"`
	EnableTableFormatting bool `json:"enable_table_formatting"`
	EnableListFormatting  bool `json:"enable_list_formatting"`

	// Specialized formatting
	EnableExplanationFormatting   bool `json:"enable_explanation_formatting"`
	EnableDocumentationFormatting bool `json:"enable_documentation_formatting"`
	EnableErrorFormatting         bool `json:"enable_error_formatting"`
	EnableDiffFormatting          bool `json:"enable_diff_formatting"`

	// Template and rendering
	EnableTemplateEngine    bool `json:"enable_template_engine"`
	EnableMarkdownRendering bool `json:"enable_markdown_rendering"`
	EnableHTMLRendering     bool `json:"enable_html_rendering"`

	// Language support
	SupportedLanguages       []string `json:"supported_languages"`
	EnableSyntaxHighlighting bool     `json:"enable_syntax_highlighting"`
	EnableLanguageDetection  bool     `json:"enable_language_detection"`

	// Output customization
	DefaultOutputFormat      OutputFormat   `json:"default_output_format"`
	SupportedOutputFormats   []OutputFormat `json:"supported_output_formats"`
	EnableThemeSupport       bool           `json:"enable_theme_support"`
	EnableStyleCustomization bool           `json:"enable_style_customization"`

	// Content enhancement
	EnableContentEnrichment   bool `json:"enable_content_enrichment"`
	EnableLinkDetection       bool `json:"enable_link_detection"`
	EnableReferenceResolution bool `json:"enable_reference_resolution"`
	EnableAutoCorrection      bool `json:"enable_auto_correction"`

	// Quality and validation
	EnableQualityChecking  bool    `json:"enable_quality_checking"`
	EnableOutputValidation bool    `json:"enable_output_validation"`
	MinQualityScore        float64 `json:"min_quality_score"`

	// Performance settings
	EnableCaching         bool          `json:"enable_caching"`
	CacheExpiry           time.Duration `json:"cache_expiry"`
	MaxCacheSize          int           `json:"max_cache_size"`
	EnableAsyncProcessing bool          `json:"enable_async_processing"`

	// Appearance settings
	DefaultTheme   string            `json:"default_theme"`
	Themes         map[string]*Theme `json:"themes"`
	CodeBlockStyle string            `json:"code_block_style"`
	TableStyle     string            `json:"table_style"`

	// Content settings
	MaxContentLength  int  `json:"max_content_length"`
	MaxCodeBlockLines int  `json:"max_code_block_lines"`
	MaxTableRows      int  `json:"max_table_rows"`
	WrapLongLines     bool `json:"wrap_long_lines"`

	// Accessibility settings
	EnableAccessibilityFeatures bool `json:"enable_accessibility_features"`
	IncludeAltText              bool `json:"include_alt_text"`
	UseSemanticMarkup           bool `json:"use_semantic_markup"`

	// Debug and monitoring
	EnableDebugOutput  bool `json:"enable_debug_output"`
	LogFormattingSteps bool `json:"log_formatting_steps"`
	CollectMetrics     bool `json:"collect_metrics"`
}

type OutputFormat string

const (
	OutputFormatPlainText OutputFormat = "plain_text"
	OutputFormatMarkdown  OutputFormat = "markdown"
	OutputFormatHTML      OutputFormat = "html"
	OutputFormatJSON      OutputFormat = "json"
	OutputFormatXML       OutputFormat = "xml"
	OutputFormatTerminal  OutputFormat = "terminal"
	OutputFormatRich      OutputFormat = "rich"
)

type Theme struct {
	Name        string              `json:"name"`
	Description string              `json:"description"`
	Colors      *ColorScheme        `json:"colors"`
	Fonts       *FontSettings       `json:"fonts"`
	Spacing     *SpacingSettings    `json:"spacing"`
	CodeStyle   *CodeStyleSettings  `json:"code_style"`
	TableStyle  *TableStyleSettings `json:"table_style"`
	CustomCSS   string              `json:"custom_css,omitempty"`
}

type ColorScheme struct {
	Background     string `json:"background"`
	Foreground     string `json:"foreground"`
	Primary        string `json:"primary"`
	Secondary      string `json:"secondary"`
	Accent         string `json:"accent"`
	Success        string `json:"success"`
	Warning        string `json:"warning"`
	Error          string `json:"error"`
	CodeBackground string `json:"code_background"`
	CodeBorder     string `json:"code_border"`
}

type FontSettings struct {
	PrimaryFont  string `json:"primary_font"`
	CodeFont     string `json:"code_font"`
	HeadingFont  string `json:"heading_font"`
	BaseFontSize string `json:"base_font_size"`
	CodeFontSize string `json:"code_font_size"`
	LineHeight   string `json:"line_height"`
}

type SpacingSettings struct {
	ParagraphSpacing string `json:"paragraph_spacing"`
	SectionSpacing   string `json:"section_spacing"`
	CodeBlockPadding string `json:"code_block_padding"`
	TableCellPadding string `json:"table_cell_padding"`
	ListItemSpacing  string `json:"list_item_spacing"`
}

type CodeStyleSettings struct {
	BorderRadius     string `json:"border_radius"`
	BorderWidth      string `json:"border_width"`
	ShadowStyle      string `json:"shadow_style"`
	LineNumbersStyle string `json:"line_numbers_style"`
	HighlightStyle   string `json:"highlight_style"`
}

type TableStyleSettings struct {
	BorderStyle        string `json:"border_style"`
	HeaderStyle        string `json:"header_style"`
	AlternateRowColors bool   `json:"alternate_row_colors"`
	HoverEffects       bool   `json:"hover_effects"`
}

// Request and response structures

type FormattingRequest struct {
	// Raw content to format
	Content     string      `json:"content"`
	ContentType ContentType `json:"content_type"`
	Language    string      `json:"language,omitempty"`

	// Formatting options
	OutputFormat OutputFormat       `json:"output_format"`
	Options      *FormattingOptions `json:"options,omitempty"`

	// Context information
	Context *FormattingContext `json:"context,omitempty"`

	// Customization
	Theme        string            `json:"theme,omitempty"`
	CustomStyles map[string]string `json:"custom_styles,omitempty"`

	// Quality requirements
	QualityRequirements *FormattingQualityRequirements `json:"quality_requirements,omitempty"`

	// Metadata
	RequestID string    `json:"request_id,omitempty"`
	UserID    string    `json:"user_id,omitempty"`
	Timestamp time.Time `json:"timestamp"`
}

type ContentType string

const (
	ContentTypeCode          ContentType = "code"
	ContentTypeExplanation   ContentType = "explanation"
	ContentTypeDocumentation ContentType = "documentation"
	ContentTypeError         ContentType = "error"
	ContentTypeDiff          ContentType = "diff"
	ContentTypeTable         ContentType = "table"
	ContentTypeList          ContentType = "list"
	ContentTypeMixed         ContentType = "mixed"
	ContentTypePlainText     ContentType = "plain_text"
)

type FormattingOptions struct {
	// General formatting
	EnableSyntaxHighlighting bool `json:"enable_syntax_highlighting"`
	EnableLineNumbers        bool `json:"enable_line_numbers"`
	EnableWordWrap           bool `json:"enable_word_wrap"`
	EnableAutoDetection      bool `json:"enable_auto_detection"`

	// Code formatting
	IndentationStyle IndentationStyle `json:"indentation_style"`
	IndentSize       int              `json:"indent_size"`
	MaxLineLength    int              `json:"max_line_length"`
	ShowWhitespace   bool             `json:"show_whitespace"`

	// Text formatting
	EnableMarkdownParsing bool `json:"enable_markdown_parsing"`
	EnableEmphasis        bool `json:"enable_emphasis"`
	EnableHeadings        bool `json:"enable_headings"`
	EnableLinkFormatting  bool `json:"enable_link_formatting"`

	// Table formatting
	TableAlignment     TableAlignment `json:"table_alignment"`
	SortableColumns    bool           `json:"sortable_columns"`
	FilterableColumns  bool           `json:"filterable_columns"`
	AlternateRowColors bool           `json:"alternate_row_colors"`

	// List formatting
	ListStyle         ListStyle `json:"list_style"`
	NestedListSupport bool      `json:"nested_list_support"`
	AutoNumbering     bool      `json:"auto_numbering"`

	// Enhancement options
	AddMetadata          bool `json:"add_metadata"`
	IncludeSourceInfo    bool `json:"include_source_info"`
	EnableReferenceLinks bool `json:"enable_reference_links"`

	// Quality options
	ValidateOutput      bool `json:"validate_output"`
	CorrectCommonErrors bool `json:"correct_common_errors"`
	EnhanceReadability  bool `json:"enhance_readability"`

	// Performance options
	UseCache          bool          `json:"use_cache"`
	AsyncProcessing   bool          `json:"async_processing"`
	MaxProcessingTime time.Duration `json:"max_processing_time"`
}

type IndentationStyle string

const (
	IndentationSpaces IndentationStyle = "spaces"
	IndentationTabs   IndentationStyle = "tabs"
	IndentationMixed  IndentationStyle = "mixed"
)

type TableAlignment string

const (
	TableAlignmentLeft   TableAlignment = "left"
	TableAlignmentCenter TableAlignment = "center"
	TableAlignmentRight  TableAlignment = "right"
	TableAlignmentAuto   TableAlignment = "auto"
)

type ListStyle string

const (
	ListStyleBullet   ListStyle = "bullet"
	ListStyleNumbered ListStyle = "numbered"
	ListStyleLettered ListStyle = "lettered"
	ListStyleCustom   ListStyle = "custom"
)

type FormattingContext struct {
	// User context
	UserPreferences *UserFormattingPreferences `json:"user_preferences,omitempty"`
	DisplaySettings *DisplaySettings           `json:"display_settings,omitempty"`

	// Environment context
	Platform     string        `json:"platform,omitempty"`
	Application  string        `json:"application,omitempty"`
	ViewportSize *ViewportSize `json:"viewport_size,omitempty"`

	// Content context
	ParentContent  string   `json:"parent_content,omitempty"`
	RelatedContent []string `json:"related_content,omitempty"`
	ContentPurpose string   `json:"content_purpose,omitempty"`
	TargetAudience string   `json:"target_audience,omitempty"`

	// Technical context
	SupportedFeatures     []string `json:"supported_features,omitempty"`
	Constraints           []string `json:"constraints,omitempty"`
	RequiredCompatibility []string `json:"required_compatibility,omitempty"`
}

type UserFormattingPreferences struct {
	PreferredTheme        string                 `json:"preferred_theme"`
	CodeFontFamily        string                 `json:"code_font_family"`
	FontSize              int                    `json:"font_size"`
	ColorSchemePreference string                 `json:"color_scheme_preference"`
	LayoutPreference      string                 `json:"layout_preference"`
	AccessibilitySettings *AccessibilitySettings `json:"accessibility_settings,omitempty"`
}

type AccessibilitySettings struct {
	HighContrast        bool `json:"high_contrast"`
	LargeText           bool `json:"large_text"`
	ScreenReaderSupport bool `json:"screen_reader_support"`
	ReducedMotion       bool `json:"reduced_motion"`
	FocusIndicators     bool `json:"focus_indicators"`
}

type DisplaySettings struct {
	DarkMode        bool `json:"dark_mode"`
	CompactMode     bool `json:"compact_mode"`
	ShowLineNumbers bool `json:"show_line_numbers"`
	ShowMinimap     bool `json:"show_minimap"`
	WrapLongLines   bool `json:"wrap_long_lines"`
}

type ViewportSize struct {
	Width  int     `json:"width"`
	Height int     `json:"height"`
	DPI    float64 `json:"dpi,omitempty"`
}

type FormattingQualityRequirements struct {
	MinReadabilityScore   float64       `json:"min_readability_score"`
	MaxRenderingTime      time.Duration `json:"max_rendering_time"`
	RequiredAccessibility []string      `json:"required_accessibility"`
	MustValidate          bool          `json:"must_validate"`
	PreserveFormatting    bool          `json:"preserve_formatting"`
}

// Response structures

type FormattingResult struct {
	// Formatted content
	FormattedContent string       `json:"formatted_content"`
	ContentType      ContentType  `json:"content_type"`
	OutputFormat     OutputFormat `json:"output_format"`

	// Formatting metadata
	DetectedLanguage  string               `json:"detected_language,omitempty"`
	AppliedFormatting []*AppliedFormatting `json:"applied_formatting"`
	UsedTheme         string               `json:"used_theme,omitempty"`

	// Quality metrics
	QualityMetrics *FormattingQualityMetrics `json:"quality_metrics"`

	// Additional assets
	StyleSheets []string          `json:"style_sheets,omitempty"`
	Scripts     []string          `json:"scripts,omitempty"`
	Assets      map[string]string `json:"assets,omitempty"`

	// Enhancement results
	DetectedLinks         []*DetectedLink      `json:"detected_links,omitempty"`
	ResolvedReferences    []*ResolvedReference `json:"resolved_references,omitempty"`
	EnhancementsSuggested []*Enhancement       `json:"enhancements_suggested,omitempty"`

	// Validation results
	ValidationResults *FormattingValidationResults `json:"validation_results,omitempty"`

	// Performance data
	ProcessingTime      time.Duration `json:"processing_time"`
	RenderingComplexity float64       `json:"rendering_complexity"`
	CacheHit            bool          `json:"cache_hit"`

	// Metadata
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	GeneratedAt time.Time              `json:"generated_at"`
}

type AppliedFormatting struct {
	Type          FormattingType         `json:"type"`
	Name          string                 `json:"name"`
	Description   string                 `json:"description"`
	Applied       bool                   `json:"applied"`
	StartPosition int                    `json:"start_position,omitempty"`
	EndPosition   int                    `json:"end_position,omitempty"`
	Properties    map[string]interface{} `json:"properties,omitempty"`
}

type FormattingType string

const (
	FormattingSyntaxHighlight FormattingType = "syntax_highlight"
	FormattingIndentation     FormattingType = "indentation"
	FormattingLineNumbers     FormattingType = "line_numbers"
	FormattingCodeBlock       FormattingType = "code_block"
	FormattingTable           FormattingType = "table"
	FormattingList            FormattingType = "list"
	FormattingEmphasis        FormattingType = "emphasis"
	FormattingHeading         FormattingType = "heading"
	FormattingLink            FormattingType = "link"
	FormattingDiff            FormattingType = "diff"
)

type FormattingQualityMetrics struct {
	ReadabilityScore   float64  `json:"readability_score"`
	ConsistencyScore   float64  `json:"consistency_score"`
	AccessibilityScore float64  `json:"accessibility_score"`
	PerformanceScore   float64  `json:"performance_score"`
	OverallQuality     float64  `json:"overall_quality"`
	Issues             []string `json:"issues,omitempty"`
	Strengths          []string `json:"strengths,omitempty"`
}

type DetectedLink struct {
	Text            string        `json:"text"`
	URL             string        `json:"url"`
	Type            LinkType      `json:"type"`
	Position        *TextPosition `json:"position"`
	Confidence      float64       `json:"confidence"`
	IsValid         bool          `json:"is_valid"`
	SuggestedAction string        `json:"suggested_action,omitempty"`
}

type LinkType string

const (
	LinkTypeHTTP      LinkType = "http"
	LinkTypeFile      LinkType = "file"
	LinkTypeEmail     LinkType = "email"
	LinkTypeReference LinkType = "reference"
	LinkTypeInternal  LinkType = "internal"
)

type TextPosition struct {
	Line   int `json:"line"`
	Column int `json:"column"`
	Index  int `json:"index"`
	Length int `json:"length"`
}

type ResolvedReference struct {
	OriginalText string        `json:"original_text"`
	ResolvedText string        `json:"resolved_text"`
	Type         ReferenceType `json:"type"`
	Context      string        `json:"context,omitempty"`
	Confidence   float64       `json:"confidence"`
}

type ReferenceType string

const (
	ReferenceTypeFunction ReferenceType = "function"
	ReferenceTypeClass    ReferenceType = "class"
	ReferenceTypeVariable ReferenceType = "variable"
	ReferenceTypeFile     ReferenceType = "file"
	ReferenceTypeAPI      ReferenceType = "api"
	ReferenceTypeConcept  ReferenceType = "concept"
)

type Enhancement struct {
	Type            EnhancementType `json:"type"`
	Title           string          `json:"title"`
	Description     string          `json:"description"`
	Priority        Priority        `json:"priority"`
	Implementation  string          `json:"implementation"`
	ExpectedBenefit string          `json:"expected_benefit"`
}

type EnhancementType string

const (
	EnhancementTypeClarity       EnhancementType = "clarity"
	EnhancementTypeReadability   EnhancementType = "readability"
	EnhancementTypeAccessibility EnhancementType = "accessibility"
	EnhancementTypePerformance   EnhancementType = "performance"
	EnhancementTypeConsistency   EnhancementType = "consistency"
)

type FormattingValidationResults struct {
	IsValid         bool     `json:"is_valid"`
	ValidationScore float64  `json:"validation_score"`
	PassedChecks    []string `json:"passed_checks"`
	FailedChecks    []string `json:"failed_checks"`
	Warnings        []string `json:"warnings"`
	Errors          []string `json:"errors"`
	Recommendations []string `json:"recommendations"`
}

// Component interfaces and implementations

type LanguageProcessor interface {
	Process(content string, options *FormattingOptions) (string, error)
	SupportsLanguage(language string) bool
	GetFeatures() []string
}

type SyntaxHighlighter interface {
	Highlight(code string, language string) (string, error)
	SupportedLanguages() []string
	SetTheme(theme string) error
}

type OutputCustomizer interface {
	Customize(content string, format OutputFormat, options *FormattingOptions) (string, error)
	SupportsFormat(format OutputFormat) bool
	GetRequiredAssets() []string
}

// Formatter implementations

type CodeFormatter struct {
	languageDetector    *LanguageDetector
	syntaxHighlighter   SyntaxHighlighter
	indentationManager  *IndentationManager
	lineNumberGenerator *LineNumberGenerator
	logger              logger.Logger
}

type TextFormatter struct {
	markdownParser    *MarkdownParser
	emphasisProcessor *EmphasisProcessor
	headingProcessor  *HeadingProcessor
	linkProcessor     *LinkProcessor
	logger            logger.Logger
}

type TableFormatter struct {
	tableParser     *TableParser
	columnAligner   *ColumnAligner
	styleApplicator *TableStyleApplicator
	sortingHandler  *SortingHandler
	logger          logger.Logger
}

type ListFormatter struct {
	listDetector      *ListDetector
	nestedListHandler *NestedListHandler
	numberingManager  *NumberingManager
	styleManager      *ListStyleManager
	logger            logger.Logger
}

type ExplanationFormatter struct {
	structureAnalyzer *ExplanationStructureAnalyzer
	sectionFormatter  *SectionFormatter
	exampleFormatter  *ExampleFormatter
	crossReferencer   *CrossReferencer
	logger            logger.Logger
}

type DocumentationFormatter struct {
	docStructureParser *DocStructureParser
	apiDocFormatter    *APIDocFormatter
	tocGenerator       *TOCGenerator
	codeExampleManager *CodeExampleManager
	logger             logger.Logger
}

type ErrorFormatter struct {
	errorCategorizer    *ErrorCategorizer
	stackTraceFormatter *StackTraceFormatter
	suggestionProvider  *ErrorSuggestionProvider
	colorizer           *ErrorColorizer
	logger              logger.Logger
}

type DiffFormatter struct {
	diffParser          *DiffParser
	changeHighlighter   *ChangeHighlighter
	contextProvider     *DiffContextProvider
	sideByeSideRenderer *SideBySideRenderer
	logger              logger.Logger
}

// Template and rendering engines

type TemplateEngine struct {
	templates        map[string]*template.Template
	templateCache    map[string]*CachedTemplate
	functionRegistry *TemplateFunctionRegistry
	logger           logger.Logger
}

type CachedTemplate struct {
	Template     *template.Template
	CachedAt     time.Time
	AccessCount  int
	LastAccessed time.Time
}

type TemplateFunctionRegistry struct {
	functions map[string]interface{}
	mu        sync.RWMutex
}

type MarkdownRenderer struct {
	parser      *MarkdownParser
	renderer    *MarkdownHTMLRenderer
	extensions  []MarkdownExtension
	customRules []*MarkdownRule
	logger      logger.Logger
}

type MarkdownExtension interface {
	Name() string
	Process(content string) (string, error)
	Priority() int
}

type MarkdownRule struct {
	Name        string
	Pattern     *regexp.Regexp
	Replacement string
	Priority    int
}

type HTMLRenderer struct {
	sanitizer             *HTMLSanitizer
	minifier              *HTMLMinifier
	accessibilityEnhancer *AccessibilityEnhancer
	seoOptimizer          *SEOOptimizer
	logger                logger.Logger
}

// Enhancement and quality components

type ContentEnrichmentEngine struct {
	enrichers       []ContentEnricher
	contextProvider *EnrichmentContextProvider
	qualityChecker  *EnrichmentQualityChecker
	logger          logger.Logger
}

type ContentEnricher interface {
	Enrich(content string, context *FormattingContext) (*EnrichmentResult, error)
	SupportsContentType(contentType ContentType) bool
	GetPriority() int
}

type EnrichmentResult struct {
	EnrichedContent string
	Additions       []*ContentAddition
	Modifications   []*ContentModification
	Suggestions     []*EnrichmentSuggestion
}

type ContentAddition struct {
	Position int
	Content  string
	Type     string
	Reason   string
}

type ContentModification struct {
	StartPosition   int
	EndPosition     int
	OriginalContent string
	ModifiedContent string
	Type            string
	Reason          string
}

type EnrichmentSuggestion struct {
	Type           string
	Description    string
	Implementation string
	Priority       Priority
}

type LinkDetector struct {
	patterns        []*LinkPattern
	validators      map[LinkType]LinkValidator
	contextAnalyzer *LinkContextAnalyzer
	logger          logger.Logger
}

type LinkPattern struct {
	Type               LinkType
	Pattern            *regexp.Regexp
	Confidence         float64
	RequiresValidation bool
}

type LinkValidator interface {
	Validate(url string) (bool, error)
	GetMetadata(url string) (*LinkMetadata, error)
}

type LinkMetadata struct {
	Title       string
	Description string
	Type        string
	IsSecure    bool
	IsReachable bool
}

type ReferenceResolver struct {
	resolvers       map[ReferenceType]ReferenceResolverImpl
	contextProvider *ReferenceContextProvider
	codeIndexer     *CodeIndexer
	logger          logger.Logger
}

type ReferenceResolverImpl interface {
	Resolve(reference string, context *FormattingContext) (*ResolvedReference, error)
	SupportsType(refType ReferenceType) bool
	GetConfidence(reference string) float64
}

// Quality and validation components

type FormattingQualityChecker struct {
	readabilityAnalyzer  *ReadabilityAnalyzer
	consistencyChecker   *ConsistencyChecker
	accessibilityChecker *AccessibilityChecker
	performanceAnalyzer  *PerformanceAnalyzer
	logger               logger.Logger
}

type ReadabilityAnalyzer struct {
	textAnalyzer      *TextReadabilityAnalyzer
	codeAnalyzer      *CodeReadabilityAnalyzer
	structureAnalyzer *StructureReadabilityAnalyzer
}

type OutputValidator struct {
	validators       map[OutputFormat]FormatValidator
	commonValidators []CommonValidator
	logger           logger.Logger
}

type FormatValidator interface {
	Validate(content string, format OutputFormat) (*ValidationResult, error)
	GetSupportedFormat() OutputFormat
	GetValidationRules() []ValidationRule
}

type ValidationResult struct {
	IsValid     bool
	Score       float64
	Issues      []*ValidationIssue
	Suggestions []string
}

type ValidationIssue struct {
	Type         IssueType
	Severity     IssueSeverity
	Message      string
	Position     *TextPosition
	SuggestedFix string
}

type ValidationRule struct {
	Name           string
	Type           string
	Required       bool
	Description    string
	Implementation func(content string) bool
}

// Style and theme management

type ThemeManager struct {
	themes           map[string]*Theme
	defaultTheme     string
	customThemes     map[string]*Theme
	themeInheritance map[string]string
	logger           logger.Logger
}

type StyleManager struct {
	styleSheets        map[string]*StyleSheet
	styleCompiler      *StyleCompiler
	customStyleManager *CustomStyleManager
	logger             logger.Logger
}

type StyleSheet struct {
	Name         string
	Content      string
	Type         StyleType
	Dependencies []string
	Priority     int
	IsMinified   bool
}

type StyleType string

const (
	StyleTypeCSS    StyleType = "css"
	StyleTypeSCSS   StyleType = "scss"
	StyleTypeLESS   StyleType = "less"
	StyleTypeInline StyleType = "inline"
)

// Cache and metrics

type FormattedResponse struct {
	Content      string
	Metadata     map[string]interface{}
	QualityScore float64
	CachedAt     time.Time
	AccessCount  int
	LastAccessed time.Time
}

type FormattingMetrics struct {
	TotalFormatRequests   int64                  `json:"total_format_requests"`
	SuccessfulFormats     int64                  `json:"successful_formats"`
	FormatsByType         map[ContentType]int64  `json:"formats_by_type"`
	FormatsByOutput       map[OutputFormat]int64 `json:"formats_by_output"`
	AverageProcessingTime time.Duration          `json:"average_processing_time"`
	AverageQualityScore   float64                `json:"average_quality_score"`
	CacheHitRate          float64                `json:"cache_hit_rate"`
	ErrorRate             float64                `json:"error_rate"`
	LastReset             time.Time              `json:"last_reset"`
	mu                    sync.RWMutex
}

// NewResponseFormatter creates a new response formatter
func NewResponseFormatter(config *ResponseFormatterConfig, logger logger.Logger) *ResponseFormatter {
	if config == nil {
		config = &ResponseFormatterConfig{
			EnableCodeFormatting:          true,
			EnableTextFormatting:          true,
			EnableTableFormatting:         true,
			EnableListFormatting:          true,
			EnableExplanationFormatting:   true,
			EnableDocumentationFormatting: true,
			EnableErrorFormatting:         true,
			EnableDiffFormatting:          true,
			EnableTemplateEngine:          true,
			EnableMarkdownRendering:       true,
			EnableHTMLRendering:           true,
			SupportedLanguages: []string{
				"go", "python", "javascript", "typescript", "java", "cpp", "csharp",
				"rust", "ruby", "php", "bash", "sql", "json", "yaml", "xml", "html", "css",
			},
			EnableSyntaxHighlighting: true,
			EnableLanguageDetection:  true,
			DefaultOutputFormat:      OutputFormatMarkdown,
			SupportedOutputFormats: []OutputFormat{
				OutputFormatPlainText, OutputFormatMarkdown, OutputFormatHTML,
				OutputFormatTerminal, OutputFormatRich,
			},
			EnableThemeSupport:          true,
			EnableStyleCustomization:    true,
			EnableContentEnrichment:     true,
			EnableLinkDetection:         true,
			EnableReferenceResolution:   true,
			EnableAutoCorrection:        true,
			EnableQualityChecking:       true,
			EnableOutputValidation:      true,
			MinQualityScore:             0.7,
			EnableCaching:               true,
			CacheExpiry:                 time.Hour,
			MaxCacheSize:                1000,
			EnableAsyncProcessing:       false,
			DefaultTheme:                "default",
			CodeBlockStyle:              "github",
			TableStyle:                  "striped",
			MaxContentLength:            1000000, // 1MB
			MaxCodeBlockLines:           1000,
			MaxTableRows:                500,
			WrapLongLines:               true,
			EnableAccessibilityFeatures: true,
			IncludeAltText:              true,
			UseSemanticMarkup:           true,
			EnableDebugOutput:           false,
			LogFormattingSteps:          false,
			CollectMetrics:              true,
		}

		// Initialize default themes
		config.Themes = map[string]*Theme{
			"default": {
				Name:        "Default",
				Description: "Default light theme",
				Colors: &ColorScheme{
					Background:     "#ffffff",
					Foreground:     "#333333",
					Primary:        "#0066cc",
					Secondary:      "#6c757d",
					Accent:         "#28a745",
					Success:        "#28a745",
					Warning:        "#ffc107",
					Error:          "#dc3545",
					CodeBackground: "#f8f9fa",
					CodeBorder:     "#e9ecef",
				},
				Fonts: &FontSettings{
					PrimaryFont:  "Inter, sans-serif",
					CodeFont:     "JetBrains Mono, Consolas, monospace",
					HeadingFont:  "Inter, sans-serif",
					BaseFontSize: "14px",
					CodeFontSize: "13px",
					LineHeight:   "1.5",
				},
				Spacing: &SpacingSettings{
					ParagraphSpacing: "1rem",
					SectionSpacing:   "1.5rem",
					CodeBlockPadding: "1rem",
					TableCellPadding: "0.5rem",
					ListItemSpacing:  "0.25rem",
				},
			},
			"dark": {
				Name:        "Dark",
				Description: "Dark theme for low-light environments",
				Colors: &ColorScheme{
					Background:     "#1a1a1a",
					Foreground:     "#e1e1e1",
					Primary:        "#4dabf7",
					Secondary:      "#868e96",
					Accent:         "#51cf66",
					Success:        "#51cf66",
					Warning:        "#ffd43b",
					Error:          "#ff6b6b",
					CodeBackground: "#2c2c2c",
					CodeBorder:     "#404040",
				},
				// Same fonts and spacing as default
				Fonts:   config.Themes["default"].Fonts,
				Spacing: config.Themes["default"].Spacing,
			},
		}
	}

	rf := &ResponseFormatter{
		logger:             logger,
		config:             config,
		languageProcessors: make(map[string]LanguageProcessor),
		syntaxHighlighters: make(map[string]SyntaxHighlighter),
		outputCustomizers:  make(map[OutputFormat]OutputCustomizer),
		cache:              make(map[string]*FormattedResponse),
		cacheExpiry:        config.CacheExpiry,
		metrics: &FormattingMetrics{
			FormatsByType:   make(map[ContentType]int64),
			FormatsByOutput: make(map[OutputFormat]int64),
			LastReset:       time.Now(),
		},
	}

	// Initialize components
	rf.initializeComponents()

	// Load default language processors
	rf.loadDefaultLanguageProcessors()

	// Initialize themes
	rf.initializeThemes()

	rf.isInitialized = true
	return rf
}

// Main formatting method
func (rf *ResponseFormatter) FormatResponse(ctx context.Context, request *FormattingRequest) (*FormattingResult, error) {
	start := time.Now()

	// Validate request
	if err := rf.validateRequest(request); err != nil {
		return nil, fmt.Errorf("invalid request: %v", err)
	}

	// Check cache first
	if rf.config.EnableCaching && (request.Options == nil || request.Options.UseCache) {
		if cached := rf.getFromCache(request); cached != nil {
			rf.updateCacheMetrics(true)
			return rf.buildResultFromCache(cached, start), nil
		}
		rf.updateCacheMetrics(false)
	}

	// Perform formatting
	result, err := rf.performFormatting(ctx, request)
	if err != nil {
		rf.updateErrorMetrics()
		return nil, fmt.Errorf("formatting failed: %v", err)
	}

	// Set processing time
	result.ProcessingTime = time.Since(start)
	result.GeneratedAt = time.Now()

	// Cache result if quality is good enough
	if rf.config.EnableCaching && result.QualityMetrics.OverallQuality >= rf.config.MinQualityScore {
		rf.cacheResult(request, result)
	}

	// Update metrics
	rf.updateSuccessMetrics(result)

	// Log debug information
	if rf.config.EnableDebugOutput {
		rf.logFormattingResult(request, result)
	}

	return result, nil
}

// Core formatting logic
func (rf *ResponseFormatter) performFormatting(ctx context.Context, request *FormattingRequest) (*FormattingResult, error) {
	result := &FormattingResult{
		FormattedContent:      request.Content,
		ContentType:           request.ContentType,
		OutputFormat:          request.OutputFormat,
		AppliedFormatting:     make([]*AppliedFormatting, 0),
		DetectedLinks:         make([]*DetectedLink, 0),
		ResolvedReferences:    make([]*ResolvedReference, 0),
		EnhancementsSuggested: make([]*Enhancement, 0),
		Metadata:              make(map[string]interface{}),
	}

	// Step 1: Content preprocessing
	preprocessedContent, err := rf.preprocessContent(request.Content, request.ContentType)
	if err != nil {
		return nil, fmt.Errorf("content preprocessing failed: %v", err)
	}
	result.FormattedContent = preprocessedContent

	// Step 2: Language detection if needed
	if request.Language == "" && rf.config.EnableLanguageDetection {
		detectedLang := rf.detectLanguage(request.Content, request.ContentType)
		result.DetectedLanguage = detectedLang
		request.Language = detectedLang
	}

	// Step 3: Apply content-specific formatting
	err = rf.applyContentSpecificFormatting(result, request)
	if err != nil {
		return nil, fmt.Errorf("content-specific formatting failed: %v", err)
	}

	// Step 4: Apply syntax highlighting if applicable
	if request.ContentType == ContentTypeCode && rf.config.EnableSyntaxHighlighting {
		if request.Options == nil || request.Options.EnableSyntaxHighlighting {
			err = rf.applySyntaxHighlighting(result, request.Language, request.Options)
			if err != nil {
				// rf.logger.Warn("Syntax highlighting failed", "error", err)
				rf.logger.Warn("Syntax highlighting failed", map[string]interface{}{"error": err})
			}
		}
	}

	// Step 5: Apply output format specific transformations
	err = rf.applyOutputFormatting(result, request)
	if err != nil {
		return nil, fmt.Errorf("output formatting failed: %v", err)
	}

	// Step 6: Content enrichment
	if rf.config.EnableContentEnrichment {
		err = rf.enrichContent(result, request)
		if err != nil {
			rf.logger.Warn("Content enrichment failed", map[string]interface{}{"error": err})
		}
	}

	// Step 7: Link detection and reference resolution
	if rf.config.EnableLinkDetection {
		links := rf.detectLinks(result.FormattedContent)
		result.DetectedLinks = links
	}

	if rf.config.EnableReferenceResolution {
		references := rf.resolveReferences(result.FormattedContent, request.Context)
		result.ResolvedReferences = references
	}

	// Step 8: Apply theme and styling
	if rf.config.EnableThemeSupport {
		err = rf.applyTheme(result, request)
		if err != nil {
			rf.logger.Warn("Theme application failed", map[string]interface{}{"error": err})
		}
	}

	// Step 9: Quality checking
	if rf.config.EnableQualityChecking {
		qualityMetrics := rf.checkFormattingQuality(result, request)
		result.QualityMetrics = qualityMetrics
	}

	// Step 10: Validation
	if rf.config.EnableOutputValidation {
		validation := rf.validateOutput(result, request)
		result.ValidationResults = validation
	}

	// Step 11: Generate enhancement suggestions
	enhancements := rf.generateEnhancementSuggestions(result, request)
	result.EnhancementsSuggested = enhancements

	// Step 12: Calculate rendering complexity
	result.RenderingComplexity = rf.calculateRenderingComplexity(result)

	return result, nil
}

// Content-specific formatting methods

func (rf *ResponseFormatter) applyContentSpecificFormatting(result *FormattingResult, request *FormattingRequest) error {
	switch request.ContentType {
	case ContentTypeCode:
		return rf.formatCode(result, request)
	case ContentTypeTable:
		return rf.formatTable(result, request)
	case ContentTypeList:
		return rf.formatList(result, request)
	case ContentTypeExplanation:
		return rf.formatExplanation(result, request)
	case ContentTypeDocumentation:
		return rf.formatDocumentation(result, request)
	case ContentTypeError:
		return rf.formatError(result, request)
	case ContentTypeDiff:
		return rf.formatDiff(result, request)
	case ContentTypeMixed:
		return rf.formatMixedContent(result, request)
	default:
		return rf.formatPlainText(result, request)
	}
}

func (rf *ResponseFormatter) formatCode(result *FormattingResult, request *FormattingRequest) error {
	if rf.codeFormatter == nil {
		return nil
	}

	options := &CodeFormattingOptions{}
	if request.Options != nil {
		options.IndentationStyle = request.Options.IndentationStyle
		options.IndentSize = request.Options.IndentSize
		options.MaxLineLength = request.Options.MaxLineLength
		options.EnableLineNumbers = request.Options.EnableLineNumbers
		options.ShowWhitespace = request.Options.ShowWhitespace
	}

	formattedCode, appliedFormatting, err := rf.codeFormatter.Format(
		result.FormattedContent, request.Language, options)
	if err != nil {
		return err
	}

	result.FormattedContent = formattedCode
	result.AppliedFormatting = append(result.AppliedFormatting, appliedFormatting...)

	return nil
}

func (rf *ResponseFormatter) formatTable(result *FormattingResult, request *FormattingRequest) error {
	if rf.tableFormatter == nil {
		return nil
	}

	options := &TableFormattingOptions{}
	if request.Options != nil {
		options.Alignment = request.Options.TableAlignment
		options.SortableColumns = request.Options.SortableColumns
		options.FilterableColumns = request.Options.FilterableColumns
		options.AlternateRowColors = request.Options.AlternateRowColors
	}

	formattedTable, appliedFormatting, err := rf.tableFormatter.Format(
		result.FormattedContent, request.OutputFormat, options)
	if err != nil {
		return err
	}

	result.FormattedContent = formattedTable
	result.AppliedFormatting = append(result.AppliedFormatting, appliedFormatting...)

	return nil
}

func (rf *ResponseFormatter) formatList(result *FormattingResult, request *FormattingRequest) error {
	if rf.listFormatter == nil {
		return nil
	}

	options := &ListFormattingOptions{}
	if request.Options != nil {
		options.Style = request.Options.ListStyle
		options.NestedListSupport = request.Options.NestedListSupport
		options.AutoNumbering = request.Options.AutoNumbering
	}

	formattedList, appliedFormatting, err := rf.listFormatter.Format(
		result.FormattedContent, request.OutputFormat, options)
	if err != nil {
		return err
	}

	result.FormattedContent = formattedList
	result.AppliedFormatting = append(result.AppliedFormatting, appliedFormatting...)

	return nil
}

func (rf *ResponseFormatter) formatExplanation(result *FormattingResult, request *FormattingRequest) error {
	if rf.explanationFormatter == nil {
		return nil
	}

	formattedExplanation, appliedFormatting, err := rf.explanationFormatter.Format(
		result.FormattedContent, request.OutputFormat, request.Context)
	if err != nil {
		return err
	}

	result.FormattedContent = formattedExplanation
	result.AppliedFormatting = append(result.AppliedFormatting, appliedFormatting...)

	return nil
}

func (rf *ResponseFormatter) formatDocumentation(result *FormattingResult, request *FormattingRequest) error {
	if rf.documentationFormatter == nil {
		return nil
	}

	formattedDoc, appliedFormatting, err := rf.documentationFormatter.Format(
		result.FormattedContent, request.OutputFormat, request.Context)
	if err != nil {
		return err
	}

	result.FormattedContent = formattedDoc
	result.AppliedFormatting = append(result.AppliedFormatting, appliedFormatting...)

	return nil
}

func (rf *ResponseFormatter) formatError(result *FormattingResult, request *FormattingRequest) error {
	if rf.errorFormatter == nil {
		return nil
	}

	formattedError, appliedFormatting, err := rf.errorFormatter.Format(
		result.FormattedContent, request.OutputFormat, request.Context)
	if err != nil {
		return err
	}

	result.FormattedContent = formattedError
	result.AppliedFormatting = append(result.AppliedFormatting, appliedFormatting...)

	return nil
}

func (rf *ResponseFormatter) formatDiff(result *FormattingResult, request *FormattingRequest) error {
	if rf.diffFormatter == nil {
		return nil
	}

	formattedDiff, appliedFormatting, err := rf.diffFormatter.Format(
		result.FormattedContent, request.OutputFormat, request.Context)
	if err != nil {
		return err
	}

	result.FormattedContent = formattedDiff
	result.AppliedFormatting = append(result.AppliedFormatting, appliedFormatting...)

	return nil
}

func (rf *ResponseFormatter) formatMixedContent(result *FormattingResult, request *FormattingRequest) error {
	// For mixed content, we need to identify different sections and format them appropriately
	sections := rf.identifyContentSections(result.FormattedContent)

	var formattedSections []string
	for _, section := range sections {
		sectionRequest := &FormattingRequest{
			Content:      section.Content,
			ContentType:  section.Type,
			OutputFormat: request.OutputFormat,
			Language:     request.Language,
			Options:      request.Options,
			Context:      request.Context,
		}

		sectionResult := &FormattingResult{
			FormattedContent: section.Content,
			ContentType:      section.Type,
			OutputFormat:     request.OutputFormat,
		}

		err := rf.applyContentSpecificFormatting(sectionResult, sectionRequest)
		if err != nil {
			rf.logger.Warn("Failed to format section", map[string]interface{}{
				"type":  section.Type,
				"error": err,
			})
			formattedSections = append(formattedSections, section.Content)
		} else {
			formattedSections = append(formattedSections, sectionResult.FormattedContent)
			result.AppliedFormatting = append(result.AppliedFormatting, sectionResult.AppliedFormatting...)
		}
	}

	result.FormattedContent = strings.Join(formattedSections, "\n\n")
	return nil
}

func (rf *ResponseFormatter) formatPlainText(result *FormattingResult, request *FormattingRequest) error {
	if rf.textFormatter == nil {
		return nil
	}

	options := &TextFormattingOptions{}
	if request.Options != nil {
		options.EnableMarkdownParsing = request.Options.EnableMarkdownParsing
		options.EnableEmphasis = request.Options.EnableEmphasis
		options.EnableHeadings = request.Options.EnableHeadings
		options.EnableLinkFormatting = request.Options.EnableLinkFormatting
		options.WrapLongLines = request.Options.EnableWordWrap
	}

	formattedText, appliedFormatting, err := rf.textFormatter.Format(
		result.FormattedContent, request.OutputFormat, options)
	if err != nil {
		return err
	}

	result.FormattedContent = formattedText
	result.AppliedFormatting = append(result.AppliedFormatting, appliedFormatting...)

	return nil
}

// Helper methods

func (rf *ResponseFormatter) validateRequest(request *FormattingRequest) error {
	if request == nil {
		return fmt.Errorf("request cannot be nil")
	}

	if strings.TrimSpace(request.Content) == "" {
		return fmt.Errorf("content cannot be empty")
	}

	if len(request.Content) > rf.config.MaxContentLength {
		return fmt.Errorf("content too long: maximum %d characters", rf.config.MaxContentLength)
	}

	// Validate output format
	supportedFormats := rf.config.SupportedOutputFormats
	formatSupported := false
	for _, format := range supportedFormats {
		if format == request.OutputFormat {
			formatSupported = true
			break
		}
	}

	if !formatSupported {
		return fmt.Errorf("unsupported output format: %s", request.OutputFormat)
	}

	return nil
}

func (rf *ResponseFormatter) preprocessContent(content string, contentType ContentType) (string, error) {
	// Basic preprocessing
	content = strings.TrimSpace(content)

	// Remove excessive whitespace
	content = regexp.MustCompile(`\n{3,}`).ReplaceAllString(content, "\n\n")
	content = regexp.MustCompile(`[ \t]{2,}`).ReplaceAllString(content, " ")

	// Content-specific preprocessing
	switch contentType {
	case ContentTypeCode:
		// Normalize line endings
		content = strings.ReplaceAll(content, "\r\n", "\n")
		content = strings.ReplaceAll(content, "\r", "\n")
	}

	return content, nil
}

func (rf *ResponseFormatter) detectLanguage(content string, contentType ContentType) string {
	if contentType != ContentTypeCode {
		return ""
	}

	// Simple language detection based on patterns
	languagePatterns := map[string]*regexp.Regexp{
		"go":         regexp.MustCompile(`(?m)^package\s+\w+|func\s+\w+|import\s+["']`),
		"python":     regexp.MustCompile(`(?m)^def\s+\w+|^class\s+\w+|import\s+\w+`),
		"javascript": regexp.MustCompile(`(?m)function\s+\w+|const\s+\w+|let\s+\w+|var\s+\w+`),
		"java":       regexp.MustCompile(`(?m)public\s+class|private\s+\w+|import\s+java`),
		"cpp":        regexp.MustCompile(`(?m)#include\s*<|using\s+namespace|int\s+main`),
		"csharp":     regexp.MustCompile(`(?m)using\s+System|public\s+class|namespace\s+\w+`),
	}

	scores := make(map[string]int)
	for lang, pattern := range languagePatterns {
		matches := pattern.FindAllString(content, -1)
		scores[lang] = len(matches)
	}

	// Find language with highest score
	var bestLang string
	var bestScore int
	for lang, score := range scores {
		if score > bestScore {
			bestLang = lang
			bestScore = score
		}
	}

	return bestLang
}

// Cache management
func (rf *ResponseFormatter) getFromCache(request *FormattingRequest) *FormattedResponse {
	rf.cacheMu.RLock()
	defer rf.cacheMu.RUnlock()

	cacheKey := rf.generateCacheKey(request)
	if cached, exists := rf.cache[cacheKey]; exists {
		if time.Since(cached.CachedAt) < rf.cacheExpiry {
			cached.AccessCount++
			cached.LastAccessed = time.Now()
			return cached
		}
		// Remove expired entry
		delete(rf.cache, cacheKey)
	}

	return nil
}

func (rf *ResponseFormatter) cacheResult(request *FormattingRequest, result *FormattingResult) {
	rf.cacheMu.Lock()
	defer rf.cacheMu.Unlock()

	cacheKey := rf.generateCacheKey(request)
	rf.cache[cacheKey] = &FormattedResponse{
		Content:      result.FormattedContent,
		Metadata:     result.Metadata,
		QualityScore: result.QualityMetrics.OverallQuality,
		CachedAt:     time.Now(),
		AccessCount:  0,
		LastAccessed: time.Now(),
	}

	// Cleanup old entries if cache is too large
	if len(rf.cache) > rf.config.MaxCacheSize {
		rf.cleanupCache()
	}
}

func (rf *ResponseFormatter) generateCacheKey(request *FormattingRequest) string {
	// Simple cache key generation
	return fmt.Sprintf("%s_%s_%s_%s",
		request.Content[:min(100, len(request.Content))],
		request.ContentType,
		request.OutputFormat,
		request.Language)
}

func (rf *ResponseFormatter) cleanupCache() {
	// Remove least recently used entries
	type cacheEntry struct {
		key          string
		lastAccessed time.Time
	}

	var entries []cacheEntry
	for key, cached := range rf.cache {
		entries = append(entries, cacheEntry{key, cached.LastAccessed})
	}

	sort.Slice(entries, func(i, j int) bool {
		return entries[i].lastAccessed.Before(entries[j].lastAccessed)
	})

	// Remove oldest 25% of entries
	removeCount := len(entries) / 4
	for i := 0; i < removeCount; i++ {
		delete(rf.cache, entries[i].key)
	}
}

func (rf *ResponseFormatter) buildResultFromCache(cached *FormattedResponse, start time.Time) *FormattingResult {
	return &FormattingResult{
		FormattedContent: cached.Content,
		QualityMetrics: &FormattingQualityMetrics{
			OverallQuality: cached.QualityScore,
		},
		ProcessingTime: time.Since(start),
		CacheHit:       true,
		Metadata:       cached.Metadata,
		GeneratedAt:    time.Now(),
	}
}

// Component initialization
// Component initialization
func (rf *ResponseFormatter) initializeComponents() {
	// Initialize formatters
	if rf.config.EnableCodeFormatting {
		cf, err := NewCodeFormatter(rf.logger)
		if err != nil {
			rf.logger.Warn("Failed to initialize CodeFormatter", map[string]interface{}{"error": err.Error()})
		} else {
			rf.codeFormatter = cf
		}
	}

	if rf.config.EnableTextFormatting {
		tf := NewTextFormatter(rf.logger)
		if tf == nil {
			rf.logger.Warn("Failed to initialize TextFormatter", map[string]interface{}{"component": "TextFormatter"})
		} else {
			rf.textFormatter = tf
		}
	}

	if rf.config.EnableTableFormatting {
		tbl := NewTableFormatter(rf.logger)
		if tbl == nil {
			rf.logger.Warn("Failed to initialize TableFormatter", map[string]interface{}{"component": "TableFormatter"})
		} else {
			rf.tableFormatter = tbl
		}
	}

	if rf.config.EnableListFormatting {
		lf := NewListFormatter(rf.logger)
		if lf == nil {
			rf.logger.Warn("Failed to initialize ListFormatter", map[string]interface{}{"component": "ListFormatter"})
		} else {
			rf.listFormatter = lf
		}
	}

	// Specialized formatters
	if rf.config.EnableExplanationFormatting {
		rf.explanationFormatter = NewExplanationFormatter(rf.logger)
	}
	if rf.config.EnableDocumentationFormatting {
		rf.documentationFormatter = NewDocumentationFormatter(rf.logger)
	}
	if rf.config.EnableErrorFormatting {
		rf.errorFormatter = NewErrorFormatter(rf.logger)
	}
	if rf.config.EnableDiffFormatting {
		rf.diffFormatter = NewDiffFormatter(rf.logger)
	}

	// Template/rendering engines
	if rf.config.EnableTemplateEngine {
		rf.templateEngine = NewTemplateEngine(rf.logger)
	}
	if rf.config.EnableMarkdownRendering {
		rf.markdownRenderer = NewMarkdownRenderer(rf.logger)
	}
	if rf.config.EnableHTMLRendering {
		rf.htmlRenderer = NewHTMLRenderer(rf.logger)
	}

	// Enhancement engines
	if rf.config.EnableContentEnrichment {
		rf.enrichmentEngine = NewContentEnrichmentEngine(rf.logger)
	}
	if rf.config.EnableLinkDetection {
		rf.linkDetector = NewLinkDetector(rf.logger)
	}
	if rf.config.EnableReferenceResolution {
		rf.referenceResolver = NewReferenceResolver(rf.logger)
	}

	// Quality & validation
	if rf.config.EnableQualityChecking {
		rf.qualityChecker = NewFormattingQualityChecker(rf.logger)
	}
	if rf.config.EnableOutputValidation {
		rf.validator = NewOutputValidator(rf.logger)
	}

	// Style management
	if rf.config.EnableThemeSupport {
		rf.themeManager = NewThemeManager(rf.config.Themes, rf.config.DefaultTheme, rf.logger)
	}
	if rf.config.EnableStyleCustomization {
		rf.styleManager = NewStyleManager(rf.logger)
	}
}

func (rf *ResponseFormatter) loadDefaultLanguageProcessors() {
	// Load language processors for supported languages
	for _, lang := range rf.config.SupportedLanguages {
		processor := NewLanguageProcessor(lang, rf.logger)
		rf.languageProcessors[lang] = processor

		if rf.config.EnableSyntaxHighlighting {
			highlighter := NewSyntaxHighlighter(lang, rf.logger)
			rf.syntaxHighlighters[lang] = highlighter
		}
	}
}

func (rf *ResponseFormatter) initializeThemes() {
	// Initialize output customizers for each supported format
	for _, format := range rf.config.SupportedOutputFormats {
		customizer := NewOutputCustomizer(format, rf.logger)
		rf.outputCustomizers[format] = customizer
	}
}

// Metrics methods
func (rf *ResponseFormatter) updateSuccessMetrics(result *FormattingResult) {
	rf.metrics.mu.Lock()
	defer rf.metrics.mu.Unlock()

	rf.metrics.TotalFormatRequests++
	rf.metrics.SuccessfulFormats++
	rf.metrics.FormatsByType[result.ContentType]++
	rf.metrics.FormatsByOutput[result.OutputFormat]++

	// Update averages
	if rf.metrics.TotalFormatRequests == 1 {
		rf.metrics.AverageProcessingTime = result.ProcessingTime
		if result.QualityMetrics != nil {
			rf.metrics.AverageQualityScore = result.QualityMetrics.OverallQuality
		}
	} else {
		count := float64(rf.metrics.TotalFormatRequests)
		rf.metrics.AverageProcessingTime = time.Duration(
			(int64(rf.metrics.AverageProcessingTime)*(int64(count)-1) + int64(result.ProcessingTime)) / int64(count))

		if result.QualityMetrics != nil {
			rf.metrics.AverageQualityScore = (rf.metrics.AverageQualityScore*(count-1) + result.QualityMetrics.OverallQuality) / count
		}
	}
}

func (rf *ResponseFormatter) updateErrorMetrics() {
	rf.metrics.mu.Lock()
	defer rf.metrics.mu.Unlock()

	rf.metrics.TotalFormatRequests++

	if rf.metrics.TotalFormatRequests > 0 {
		rf.metrics.ErrorRate = float64(rf.metrics.TotalFormatRequests-rf.metrics.SuccessfulFormats) / float64(rf.metrics.TotalFormatRequests)
	}
}

func (rf *ResponseFormatter) updateCacheMetrics(hit bool) {
	rf.metrics.mu.Lock()
	defer rf.metrics.mu.Unlock()

	if hit {
		rf.metrics.CacheHitRate = (rf.metrics.CacheHitRate + 1.0) / 2.0
	} else {
		rf.metrics.CacheHitRate = rf.metrics.CacheHitRate / 2.0
	}
}

// Public API methods
func (rf *ResponseFormatter) GetMetrics() *FormattingMetrics {
	rf.metrics.mu.RLock()
	defer rf.metrics.mu.RUnlock()

	metrics := *rf.metrics
	return &metrics
}

func (rf *ResponseFormatter) ResetMetrics() {
	rf.metrics.mu.Lock()
	defer rf.metrics.mu.Unlock()

	rf.metrics = &FormattingMetrics{
		FormatsByType:   make(map[ContentType]int64),
		FormatsByOutput: make(map[OutputFormat]int64),
		LastReset:       time.Now(),
	}
}

func (rf *ResponseFormatter) ClearCache() {
	rf.cacheMu.Lock()
	defer rf.cacheMu.Unlock()

	rf.cache = make(map[string]*FormattedResponse)
}

// Placeholder implementations for formatting components and helper methods

// Additional helper structures that would need implementation
type ContentSection struct {
	Content string
	Type    ContentType
	Start   int
	End     int
}

type CodeFormattingOptions struct {
	IndentationStyle  IndentationStyle
	IndentSize        int
	MaxLineLength     int
	EnableLineNumbers bool
	ShowWhitespace    bool
}

type TableFormattingOptions struct {
	Alignment          TableAlignment
	SortableColumns    bool
	FilterableColumns  bool
	AlternateRowColors bool
}

type ListFormattingOptions struct {
	Style             ListStyle
	NestedListSupport bool
	AutoNumbering     bool
}

type TextFormattingOptions struct {
	EnableMarkdownParsing bool
	EnableEmphasis        bool
	EnableHeadings        bool
	EnableLinkFormatting  bool
	WrapLongLines         bool
}

// Placeholder methods that would need full implementation
func (rf *ResponseFormatter) applySyntaxHighlighting(result *FormattingResult, language string, options *FormattingOptions) error {
	// Implementation would apply syntax highlighting
	return nil
}

func (rf *ResponseFormatter) applyOutputFormatting(result *FormattingResult, request *FormattingRequest) error {
	// Implementation would apply output format specific transformations
	if customizer, exists := rf.outputCustomizers[request.OutputFormat]; exists {
		options := &FormattingOptions{}
		if request.Options != nil {
			// Map request options to formatting options
		}

		customized, err := customizer.Customize(result.FormattedContent, request.OutputFormat, options)
		if err != nil {
			return err
		}

		result.FormattedContent = customized
	}

	return nil
}

func (rf *ResponseFormatter) enrichContent(result *FormattingResult, request *FormattingRequest) error {
	// Implementation would enrich content
	return nil
}

func (rf *ResponseFormatter) detectLinks(content string) []*DetectedLink {
	// Implementation would detect links
	return []*DetectedLink{}
}

func (rf *ResponseFormatter) resolveReferences(content string, context *FormattingContext) []*ResolvedReference {
	// Implementation would resolve references
	return []*ResolvedReference{}
}

func (rf *ResponseFormatter) applyTheme(result *FormattingResult, request *FormattingRequest) error {
	// Implementation would apply theme
	return nil
}

func (rf *ResponseFormatter) checkFormattingQuality(result *FormattingResult, request *FormattingRequest) *FormattingQualityMetrics {
	// Implementation would check quality
	return &FormattingQualityMetrics{
		ReadabilityScore:   0.8,
		ConsistencyScore:   0.8,
		AccessibilityScore: 0.8,
		PerformanceScore:   0.8,
		OverallQuality:     0.8,
	}
}

func (rf *ResponseFormatter) validateOutput(result *FormattingResult, request *FormattingRequest) *FormattingValidationResults {
	// Implementation would validate output
	return &FormattingValidationResults{
		IsValid:         true,
		ValidationScore: 0.9,
	}
}

func (rf *ResponseFormatter) generateEnhancementSuggestions(result *FormattingResult, request *FormattingRequest) []*Enhancement {
	// Implementation would generate suggestions
	return []*Enhancement{}
}

func (rf *ResponseFormatter) calculateRenderingComplexity(result *FormattingResult) float64 {
	// Implementation would calculate complexity
	return 0.5
}

func (rf *ResponseFormatter) identifyContentSections(content string) []*ContentSection {
	// Implementation would identify sections
	return []*ContentSection{{Content: content, Type: ContentTypePlainText}}
}

func (rf *ResponseFormatter) logFormattingResult(request *FormattingRequest, result *FormattingResult) {
	rf.logger.Debug("Formatting completed",
		map[string]interface{}{
			"content_type":    request.ContentType,
			"output_format":   request.OutputFormat,
			"processing_time": result.ProcessingTime,
			"quality_score":   result.QualityMetrics.OverallQuality,
			"cache_hit":       result.CacheHit,
		})
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Constructor functions for components (simplified implementations)
func NewCodeFormatter(logger logger.Logger) (*CodeFormatter, error) {
	codeFormatter := &CodeFormatter{logger: logger}
	return codeFormatter, nil
}

func NewTextFormatter(logger logger.Logger) *TextFormatter {
	return &TextFormatter{logger: logger}
}

func NewTableFormatter(logger logger.Logger) *TableFormatter {
	return &TableFormatter{logger: logger}
}

func NewListFormatter(logger logger.Logger) *ListFormatter {
	return &ListFormatter{logger: logger}
}

func NewExplanationFormatter(logger logger.Logger) *ExplanationFormatter {
	return &ExplanationFormatter{logger: logger}
}

func NewDocumentationFormatter(logger logger.Logger) *DocumentationFormatter {
	return &DocumentationFormatter{logger: logger}
}

func NewErrorFormatter(logger logger.Logger) *ErrorFormatter {
	return &ErrorFormatter{logger: logger}
}

func NewDiffFormatter(logger logger.Logger) *DiffFormatter {
	return &DiffFormatter{logger: logger}
}

func NewTemplateEngine(logger logger.Logger) *TemplateEngine {
	return &TemplateEngine{logger: logger}
}

func NewMarkdownRenderer(logger logger.Logger) *MarkdownRenderer {
	return &MarkdownRenderer{logger: logger}
}

func NewHTMLRenderer(logger logger.Logger) *HTMLRenderer {
	return &HTMLRenderer{logger: logger}
}

func NewContentEnrichmentEngine(logger logger.Logger) *ContentEnrichmentEngine {
	return &ContentEnrichmentEngine{logger: logger}
}

func NewLinkDetector(logger logger.Logger) *LinkDetector {
	return &LinkDetector{logger: logger}
}

func NewReferenceResolver(logger logger.Logger) *ReferenceResolver {
	return &ReferenceResolver{logger: logger}
}

func NewFormattingQualityChecker(logger logger.Logger) *FormattingQualityChecker {
	return &FormattingQualityChecker{logger: logger}
}

func NewOutputValidator(logger logger.Logger) *OutputValidator {
	return &OutputValidator{logger: logger}
}

func NewThemeManager(themes map[string]*Theme, defaultTheme string, logger logger.Logger) *ThemeManager {
	return &ThemeManager{
		themes:       themes,
		defaultTheme: defaultTheme,
		logger:       logger,
	}
}

func NewStyleManager(logger logger.Logger) *StyleManager {
	return &StyleManager{logger: logger}
}

func NewLanguageProcessor(language string, logger logger.Logger) LanguageProcessor {
	// Would return a concrete implementation
	return &DefaultLanguageProcessor{
		language: language,
		logger:   logger,
	}
}

func NewSyntaxHighlighter(language string, logger logger.Logger) SyntaxHighlighter {
	return &DefaultSyntaxHighlighter{
		language: language,
		logger:   logger,
	}
}

func NewOutputCustomizer(format OutputFormat, logger logger.Logger) OutputCustomizer {
	return &DefaultOutputCustomizer{
		format: format,
		logger: logger,
	}
}

// Default implementations
type DefaultLanguageProcessor struct {
	language string
	logger   logger.Logger
}

func (dlp *DefaultLanguageProcessor) Process(content string, options *FormattingOptions) (string, error) {
	// Basic processing - would be expanded for each language
	return content, nil
}

func (dlp *DefaultLanguageProcessor) SupportsLanguage(language string) bool {
	return dlp.language == language
}

func (dlp *DefaultLanguageProcessor) GetFeatures() []string {
	return []string{"basic_formatting"}
}

type DefaultSyntaxHighlighter struct {
	language string
	logger   logger.Logger
}

func (dsh *DefaultSyntaxHighlighter) Highlight(code string, language string) (string, error) {
	// Basic syntax highlighting - would be expanded with actual highlighting logic
	return code, nil
}

func (dsh *DefaultSyntaxHighlighter) SupportedLanguages() []string {
	return []string{dsh.language}
}

func (dsh *DefaultSyntaxHighlighter) SetTheme(theme string) error {
	// Theme setting implementation
	return nil
}

type DefaultOutputCustomizer struct {
	format OutputFormat
	logger logger.Logger
}

func (doc *DefaultOutputCustomizer) Customize(content string, format OutputFormat, options *FormattingOptions) (string, error) {
	// Basic customization based on output format
	switch format {
	case OutputFormatHTML:
		return doc.toHTML(content), nil
	case OutputFormatMarkdown:
		return doc.toMarkdown(content), nil
	case OutputFormatTerminal:
		return doc.toTerminal(content), nil
	default:
		return content, nil
	}
}

func (doc *DefaultOutputCustomizer) SupportsFormat(format OutputFormat) bool {
	return doc.format == format
}

func (doc *DefaultOutputCustomizer) GetRequiredAssets() []string {
	return []string{}
}

func (doc *DefaultOutputCustomizer) toHTML(content string) string {
	// Basic HTML conversion
	content = strings.ReplaceAll(content, "&", "&amp;")
	content = strings.ReplaceAll(content, "<", "&lt;")
	content = strings.ReplaceAll(content, ">", "&gt;")
	content = strings.ReplaceAll(content, "\n", "<br>")
	return fmt.Sprintf("<div class=\"formatted-content\">%s</div>", content)
}

func (doc *DefaultOutputCustomizer) toMarkdown(content string) string {
	// Content is likely already in markdown format
	return content
}

func (doc *DefaultOutputCustomizer) toTerminal(content string) string {
	// Basic terminal formatting with ANSI codes
	return content
}

// Placeholder methods for formatter components
func (cf *CodeFormatter) Format(code string, language string, options *CodeFormattingOptions) (string, []*AppliedFormatting, error) {
	// Implementation would format code according to options
	applied := []*AppliedFormatting{
		{
			Type:        FormattingCodeBlock,
			Name:        "Code Block",
			Description: "Applied code block formatting",
			Applied:     true,
		},
	}
	return code, applied, nil
}

func (tf *TextFormatter) Format(text string, format OutputFormat, options *TextFormattingOptions) (string, []*AppliedFormatting, error) {
	// Implementation would format text according to options
	applied := []*AppliedFormatting{}

	if options.EnableEmphasis {
		applied = append(applied, &AppliedFormatting{
			Type:        FormattingEmphasis,
			Name:        "Text Emphasis",
			Description: "Applied text emphasis formatting",
			Applied:     true,
		})
	}

	return text, applied, nil
}

func (tf *TableFormatter) Format(table string, format OutputFormat, options *TableFormattingOptions) (string, []*AppliedFormatting, error) {
	// Implementation would format table according to options
	applied := []*AppliedFormatting{
		{
			Type:        FormattingTable,
			Name:        "Table Formatting",
			Description: "Applied table formatting",
			Applied:     true,
		},
	}
	return table, applied, nil
}

func (lf *ListFormatter) Format(list string, format OutputFormat, options *ListFormattingOptions) (string, []*AppliedFormatting, error) {
	// Implementation would format list according to options
	applied := []*AppliedFormatting{
		{
			Type:        FormattingList,
			Name:        "List Formatting",
			Description: "Applied list formatting",
			Applied:     true,
		},
	}
	return list, applied, nil
}

func (ef *ExplanationFormatter) Format(explanation string, format OutputFormat, context *FormattingContext) (string, []*AppliedFormatting, error) {
	// Implementation would format explanation with structure
	applied := []*AppliedFormatting{
		{
			Type:        FormattingStructureType,
			Name:        "Explanation Structure",
			Description: "Applied explanation structure formatting",
			Applied:     true,
		},
	}
	return explanation, applied, nil
}

func (df *DocumentationFormatter) Format(doc string, format OutputFormat, context *FormattingContext) (string, []*AppliedFormatting, error) {
	// Implementation would format documentation
	applied := []*AppliedFormatting{
		{
			Type:        FormattingStructureType,
			Name:        "Documentation Structure",
			Description: "Applied documentation formatting",
			Applied:     true,
		},
	}
	return doc, applied, nil
}

func (ef *ErrorFormatter) Format(error string, format OutputFormat, context *FormattingContext) (string, []*AppliedFormatting, error) {
	// Implementation would format error messages
	applied := []*AppliedFormatting{
		{
			Type:        FormattingEmphasis,
			Name:        "Error Formatting",
			Description: "Applied error message formatting",
			Applied:     true,
		},
	}
	return error, applied, nil
}

func (df *DiffFormatter) Format(diff string, format OutputFormat, context *FormattingContext) (string, []*AppliedFormatting, error) {
	// Implementation would format diff output
	applied := []*AppliedFormatting{
		{
			Type:        FormattingDiff,
			Name:        "Diff Formatting",
			Description: "Applied diff formatting",
			Applied:     true,
		},
	}
	return diff, applied, nil
}
