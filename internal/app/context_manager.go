package app

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/logger"
)

// ContextManager maintains and provides access to the current operational context
type ContextManager struct {
	// Core context data
	activeProject    *ProjectContext
	selectedFiles    map[string]*FileContext
	userPreferences  *UserPreferences
	workspaceContext *WorkspaceContext
	sessionContext   *SessionContext

	// Context history and state
	contextHistory []*ContextSnapshot
	contextStack   []*ContextFrame
	maxHistorySize int
	maxStackDepth  int

	// Observers and listeners
	observers       []ContextObserver
	changeListeners []ContextChangeListener

	// Configuration and settings
	config *ContextManagerConfig
	logger logger.Logger

	// State management
	mu         sync.RWMutex
	lastUpdate time.Time
	isDirty    bool

	// Persistence
	persistenceProvider ContextPersistenceProvider
	autoSave            bool
	saveInterval        time.Duration

	// Context enrichment
	enrichmentProviders []ContextEnrichmentProvider

	// Cache and optimization
	contextCache map[string]interface{}
	cacheExpiry  time.Duration
	cacheMu      sync.RWMutex
}

// ProjectContext represents the currently active project
type DependencyInfo struct {
	Dependencies map[string]string `json:"dependencies"`
}

type ProjectContext struct {
	// Basic project information
	ID        string      `json:"id"`
	Name      string      `json:"name"`
	Path      string      `json:"path"`
	Type      ProjectType `json:"type"`
	Language  []string    `json:"languages"`
	Framework []string    `json:"frameworks"`

	// Project structure and metadata
	Structure     *ProjectStructure     `json:"structure"`
	Configuration *ProjectConfiguration `json:"configuration"`
	Dependencies  *DependencyInfo       `json:"dependencies"`

	// Version control information
	VersionControl *VersionControlInfo `json:"version_control"`

	// Build and deployment information
	BuildInfo      *BuildInfo      `json:"build_info"`
	DeploymentInfo *DeploymentInfo `json:"deployment_info"`

	// Documentation and standards
	Documentation   *DocumentationInfo `json:"documentation"`
	CodingStandards *CodingStandards   `json:"coding_standards"`

	// Analytics and insights
	ProjectMetrics *ProjectMetrics  `json:"project_metrics"`
	RecentActivity []*ActivityEntry `json:"recent_activity"`

	// Timestamps and metadata
	CreatedAt    time.Time `json:"created_at"`
	LastAccessed time.Time `json:"last_accessed"`
	LastModified time.Time `json:"last_modified"`

	// Custom metadata
	CustomData map[string]interface{} `json:"custom_data"`
}

type ProjectType string

const (
	ProjectTypeWebApp       ProjectType = "webapp"
	ProjectTypeAPI          ProjectType = "api"
	ProjectTypeCLI          ProjectType = "cli"
	ProjectTypeLibrary      ProjectType = "library"
	ProjectTypeMicroservice ProjectType = "microservice"
	ProjectTypeMobile       ProjectType = "mobile"
	ProjectTypeDesktop      ProjectType = "desktop"
	ProjectTypeUnknown      ProjectType = "unknown"
)

type ProjectStructure struct {
	RootDirectory      string    `json:"root_directory"`
	SourceDirectories  []string  `json:"source_directories"`
	TestDirectories    []string  `json:"test_directories"`
	ConfigFiles        []string  `json:"config_files"`
	DocumentationFiles []string  `json:"documentation_files"`
	BuildFiles         []string  `json:"build_files"`
	GitIgnorePatterns  []string  `json:"gitignore_patterns"`
	FileTree           *FileNode `json:"file_tree"`
}

type FileNode struct {
	Name         string       `json:"name"`
	Path         string       `json:"path"`
	Type         FileNodeType `json:"type"`
	Size         int64        `json:"size"`
	LastModified time.Time    `json:"last_modified"`
	Children     []*FileNode  `json:"children,omitempty"`
	IsIgnored    bool         `json:"is_ignored"`
	Language     string       `json:"language,omitempty"`
}

type FileNodeType string

const (
	FileNodeDirectory FileNodeType = "directory"
	FileNodeFile      FileNodeType = "file"
	FileNodeSymlink   FileNodeType = "symlink"
)

type ProjectConfiguration struct {
	ConfigFiles     map[string]*ConfigFile `json:"config_files"`
	EnvironmentVars map[string]string      `json:"environment_vars"`
	BuildSettings   *BuildSettings         `json:"build_settings"`
	TestSettings    *TestSettings          `json:"test_settings"`
	LintingRules    *LintingRules          `json:"linting_rules"`
	FormattingRules *FormattingRules       `json:"formatting_rules"`
}

type ConfigFile struct {
	Path         string                 `json:"path"`
	Type         string                 `json:"type"`
	Content      map[string]interface{} `json:"content"`
	LastModified time.Time              `json:"last_modified"`
	IsValid      bool                   `json:"is_valid"`
	Errors       []string               `json:"errors,omitempty"`
}

type BuildSettings struct {
	BuildTool       string   `json:"build_tool"`
	BuildCommands   []string `json:"build_commands"`
	OutputDirectory string   `json:"output_directory"`
	SourceMaps      bool     `json:"source_maps"`
	Optimization    string   `json:"optimization"`
	Environment     string   `json:"environment"`
}

type TestSettings struct {
	TestFramework    string            `json:"test_framework"`
	TestDirectories  []string          `json:"test_directories"`
	TestPatterns     []string          `json:"test_patterns"`
	CoverageSettings *CoverageSettings `json:"coverage_settings"`
	TestEnvironment  map[string]string `json:"test_environment"`
}

type CoverageSettings struct {
	Enabled         bool     `json:"enabled"`
	Threshold       float64  `json:"threshold"`
	ReportFormat    []string `json:"report_format"`
	ExcludePatterns []string `json:"exclude_patterns"`
}

type LintingRules struct {
	Enabled        bool                   `json:"enabled"`
	Linters        []string               `json:"linters"`
	Rules          map[string]interface{} `json:"rules"`
	IgnorePatterns []string               `json:"ignore_patterns"`
}

type FormattingRules struct {
	Enabled    bool                   `json:"enabled"`
	Formatter  string                 `json:"formatter"`
	Rules      map[string]interface{} `json:"rules"`
	AutoFormat bool                   `json:"auto_format"`
}

type VersionControlInfo struct {
	Type                  string      `json:"type"`
	RemoteURL             string      `json:"remote_url"`
	CurrentBranch         string      `json:"current_branch"`
	LatestCommit          *CommitInfo `json:"latest_commit"`
	HasUncommittedChanges bool        `json:"has_uncommitted_changes"`
	TrackedFiles          int         `json:"tracked_files"`
	UntrackedFiles        int         `json:"untracked_files"`
	BranchInfo            *BranchInfo `json:"branch_info"`
}

type CommitInfo struct {
	Hash         string    `json:"hash"`
	Author       string    `json:"author"`
	Message      string    `json:"message"`
	Timestamp    time.Time `json:"timestamp"`
	FilesChanged int       `json:"files_changed"`
}

type BranchInfo struct {
	Current        string   `json:"current"`
	All            []string `json:"all"`
	AheadBy        int      `json:"ahead_by"`
	BehindBy       int      `json:"behind_by"`
	UpstreamBranch string   `json:"upstream_branch"`
}

type BuildInfo struct {
	LastBuild          *BuildResult        `json:"last_build"`
	BuildHistory       []*BuildResult      `json:"build_history"`
	BuildConfiguration *BuildConfiguration `json:"build_configuration"`
	Artifacts          []*BuildArtifact    `json:"artifacts"`
}

type BuildResult struct {
	ID        string        `json:"id"`
	Status    BuildStatus   `json:"status"`
	StartTime time.Time     `json:"start_time"`
	EndTime   time.Time     `json:"end_time"`
	Duration  time.Duration `json:"duration"`
	Logs      []string      `json:"logs"`
	Errors    []string      `json:"errors"`
	Warnings  []string      `json:"warnings"`
	Artifacts []string      `json:"artifacts"`
}

type BuildStatus string

const (
	BuildStatusSuccess  BuildStatus = "success"
	BuildStatusFailed   BuildStatus = "failed"
	BuildStatusRunning  BuildStatus = "running"
	BuildStatusCanceled BuildStatus = "canceled"
)

type BuildConfiguration struct {
	Environment  string            `json:"environment"`
	Variables    map[string]string `json:"variables"`
	Commands     []string          `json:"commands"`
	Dependencies []string          `json:"dependencies"`
}

type BuildArtifact struct {
	Name      string    `json:"name"`
	Path      string    `json:"path"`
	Size      int64     `json:"size"`
	Type      string    `json:"type"`
	CreatedAt time.Time `json:"created_at"`
	Checksum  string    `json:"checksum"`
}

type DeploymentInfo struct {
	Environments      []*DeploymentEnvironment `json:"environments"`
	LastDeployment    *DeploymentRecord        `json:"last_deployment"`
	DeploymentHistory []*DeploymentRecord      `json:"deployment_history"`
	DeploymentConfig  *DeploymentConfig        `json:"deployment_config"`
}

type DeploymentEnvironment struct {
	Name         string    `json:"name"`
	Type         string    `json:"type"`
	URL          string    `json:"url"`
	Status       string    `json:"status"`
	LastDeployed time.Time `json:"last_deployed"`
	Version      string    `json:"version"`
}

type DeploymentRecord struct {
	ID          string        `json:"id"`
	Environment string        `json:"environment"`
	Version     string        `json:"version"`
	Status      string        `json:"status"`
	DeployedBy  string        `json:"deployed_by"`
	DeployedAt  time.Time     `json:"deployed_at"`
	Duration    time.Duration `json:"duration"`
	Logs        []string      `json:"logs"`
}

type DeploymentConfig struct {
	Strategy      string   `json:"strategy"`
	AutoDeploy    bool     `json:"auto_deploy"`
	Triggers      []string `json:"triggers"`
	Notifications []string `json:"notifications"`
}

type DocumentationInfo struct {
	ReadmeFile            string                `json:"readme_file"`
	DocumentationDirs     []string              `json:"documentation_dirs"`
	APIDocumentation      *APIDocumentationInfo `json:"api_documentation"`
	CodeCoverage          float64               `json:"code_coverage"`
	DocumentationCoverage float64               `json:"documentation_coverage"`
}

type APIDocumentationInfo struct {
	Type          string    `json:"type"`
	Files         []string  `json:"files"`
	GeneratedDocs string    `json:"generated_docs"`
	LastGenerated time.Time `json:"last_generated"`
}

type CodingStandards struct {
	StyleGuide        string                 `json:"style_guide"`
	LintingRules      map[string]interface{} `json:"linting_rules"`
	FormattingRules   map[string]interface{} `json:"formatting_rules"`
	NamingConventions map[string]string      `json:"naming_conventions"`
	BestPractices     []string               `json:"best_practices"`
}

type ProjectMetrics struct {
	LinesOfCode   int                 `json:"lines_of_code"`
	FileCount     int                 `json:"file_count"`
	TestCoverage  float64             `json:"test_coverage"`
	TechnicalDebt float64             `json:"technical_debt"`
	CodeQuality   float64             `json:"code_quality"`
	Complexity    *ComplexityMetrics  `json:"complexity"`
	Dependencies  *DependencyMetrics  `json:"dependencies"`
	Performance   *PerformanceMetrics `json:"performance"`
	Security      *SecurityMetrics    `json:"security"`
}

type ComplexityMetrics struct {
	CyclomaticComplexity float64       `json:"cyclomatic_complexity"`
	CognitiveComplexity  float64       `json:"cognitive_complexity"`
	Maintainability      float64       `json:"maintainability"`
	TechnicalDebt        time.Duration `json:"technical_debt"`
}

type DependencyMetrics struct {
	TotalDependencies      int `json:"total_dependencies"`
	DirectDependencies     int `json:"direct_dependencies"`
	OutdatedDependencies   int `json:"outdated_dependencies"`
	VulnerableDependencies int `json:"vulnerable_dependencies"`
	LicenseIssues          int `json:"license_issues"`
}

type PerformanceMetrics struct {
	BundleSize        int64         `json:"bundle_size"`
	LoadTime          time.Duration `json:"load_time"`
	BuildTime         time.Duration `json:"build_time"`
	TestExecutionTime time.Duration `json:"test_execution_time"`
}

type SecurityMetrics struct {
	VulnerabilityCount int       `json:"vulnerability_count"`
	SecurityScore      float64   `json:"security_score"`
	LastSecurityScan   time.Time `json:"last_security_scan"`
	CriticalIssues     int       `json:"critical_issues"`
}

type ActivityEntry struct {
	ID          string                 `json:"id"`
	Type        ActivityType           `json:"type"`
	Description string                 `json:"description"`
	Timestamp   time.Time              `json:"timestamp"`
	User        string                 `json:"user"`
	Files       []string               `json:"files"`
	Metadata    map[string]interface{} `json:"metadata"`
}

type ActivityType string

const (
	ActivityFileModified ActivityType = "file_modified"
	ActivityFileCreated  ActivityType = "file_created"
	ActivityFileDeleted  ActivityType = "file_deleted"
	ActivityCommit       ActivityType = "commit"
	ActivityBuild        ActivityType = "build"
	ActivityDeploy       ActivityType = "deploy"
	ActivityTest         ActivityType = "test"
)

// FileContext represents information about selected files
type FileContext struct {
	// Basic file information
	Path         string `json:"path"`
	AbsolutePath string `json:"absolute_path"`
	RelativePath string `json:"relative_path"`
	Name         string `json:"name"`
	Extension    string `json:"extension"`
	Size         int64  `json:"size"`

	// Content and metadata
	Content   string `json:"content,omitempty"`
	Language  string `json:"language"`
	Encoding  string `json:"encoding"`
	LineCount int    `json:"line_count"`

	// Selection and cursor information
	Selection      *TextSelection  `json:"selection,omitempty"`
	CursorPosition *CursorPosition `json:"cursor_position,omitempty"`

	// File analysis
	Syntax    *SyntaxInfo     `json:"syntax,omitempty"`
	Imports   []string        `json:"imports,omitempty"`
	Functions []*FunctionInfo `json:"functions,omitempty"`
	Classes   []*ClassInfo    `json:"classes,omitempty"`
	Variables []*VariableInfo `json:"variables,omitempty"`

	// Version control
	GitStatus  string      `json:"git_status,omitempty"`
	LastCommit *CommitInfo `json:"last_commit,omitempty"`

	// Timestamps
	LastModified time.Time `json:"last_modified"`
	LastAccessed time.Time `json:"last_accessed"`

	// Relationships
	Dependencies []string `json:"dependencies,omitempty"`
	Dependents   []string `json:"dependents,omitempty"`
	RelatedFiles []string `json:"related_files,omitempty"`

	// Analysis results
	Issues  []*CodeIssue `json:"issues,omitempty"`
	Metrics *FileMetrics `json:"metrics,omitempty"`

	// Custom metadata
	Tags       []string               `json:"tags,omitempty"`
	Notes      string                 `json:"notes,omitempty"`
	CustomData map[string]interface{} `json:"custom_data,omitempty"`
}

type TextSelection struct {
	StartLine   int    `json:"start_line"`
	StartColumn int    `json:"start_column"`
	EndLine     int    `json:"end_line"`
	EndColumn   int    `json:"end_column"`
	Text        string `json:"text"`
}

type CursorPosition struct {
	Line   int `json:"line"`
	Column int `json:"column"`
	Offset int `json:"offset"`
}

type SyntaxInfo struct {
	IsValid  bool        `json:"is_valid"`
	Errors   []string    `json:"errors,omitempty"`
	Warnings []string    `json:"warnings,omitempty"`
	AST      interface{} `json:"ast,omitempty"`
}

type FunctionInfo struct {
	Name          string   `json:"name"`
	StartLine     int      `json:"start_line"`
	EndLine       int      `json:"end_line"`
	Parameters    []string `json:"parameters"`
	ReturnType    string   `json:"return_type,omitempty"`
	Visibility    string   `json:"visibility,omitempty"`
	Documentation string   `json:"documentation,omitempty"`
	Complexity    int      `json:"complexity,omitempty"`
}

type ClassInfo struct {
	Name          string          `json:"name"`
	StartLine     int             `json:"start_line"`
	EndLine       int             `json:"end_line"`
	Methods       []*FunctionInfo `json:"methods"`
	Properties    []*VariableInfo `json:"properties"`
	Inheritance   []string        `json:"inheritance,omitempty"`
	Interfaces    []string        `json:"interfaces,omitempty"`
	Documentation string          `json:"documentation,omitempty"`
}

type VariableInfo struct {
	Name          string `json:"name"`
	Type          string `json:"type,omitempty"`
	Line          int    `json:"line"`
	Scope         string `json:"scope"`
	IsConstant    bool   `json:"is_constant"`
	Documentation string `json:"documentation,omitempty"`
}

type CodeIssue struct {
	Type     IssueType     `json:"type"`
	Severity IssueSeverity `json:"severity"`
	Message  string        `json:"message"`
	Line     int           `json:"line"`
	Column   int           `json:"column"`
	Rule     string        `json:"rule,omitempty"`
	Source   string        `json:"source"`
}

type IssueType string

const (
	IssueTypeSyntax      IssueType = "syntax"
	IssueTypeStyle       IssueType = "style"
	IssueTypeSecurity    IssueType = "security"
	IssueTypePerformance IssueType = "performance"
	IssueTypeBug         IssueType = "bug"
	IssueTypeWarning     IssueType = "warning"
)

type IssueSeverity string

const (
	SeverityError   IssueSeverity = "error"
	SeverityWarning IssueSeverity = "warning"
	SeverityInfo    IssueSeverity = "info"
	SeverityHint    IssueSeverity = "hint"
)

type FileMetrics struct {
	LinesOfCode     int     `json:"lines_of_code"`
	LinesOfComments int     `json:"lines_of_comments"`
	BlankLines      int     `json:"blank_lines"`
	Complexity      int     `json:"complexity"`
	Maintainability float64 `json:"maintainability"`
	TestCoverage    float64 `json:"test_coverage,omitempty"`
}

// UserPreferences represents user-specific preferences and settings
type UserPreferences struct {
	// General preferences
	Theme    string `json:"theme"`
	Language string `json:"language"`
	TimeZone string `json:"timezone"`

	// Editor preferences
	EditorSettings *EditorSettings `json:"editor_settings"`

	// AI preferences
	AISettings *AISettings `json:"ai_settings"`

	// Code formatting preferences
	FormattingPrefs *FormattingPreferences `json:"formatting_preferences"`

	// Notification preferences
	Notifications *NotificationSettings `json:"notifications"`

	// Project preferences
	ProjectSettings *ProjectSettings `json:"project_settings"`

	// Privacy and security
	PrivacySettings *PrivacySettings `json:"privacy_settings"`

	// Custom shortcuts and commands
	CustomShortcuts map[string]string `json:"custom_shortcuts"`
	CustomCommands  []*CustomCommand  `json:"custom_commands"`

	// Recent items and history
	RecentProjects []string `json:"recent_projects"`
	RecentFiles    []string `json:"recent_files"`
	SearchHistory  []string `json:"search_history"`

	// Workspace preferences
	WorkspaceLayout *WorkspaceLayout `json:"workspace_layout"`

	// Integration preferences
	Integrations map[string]interface{} `json:"integrations"`

	// Custom data
	CustomData map[string]interface{} `json:"custom_data"`

	// Metadata
	LastUpdated time.Time `json:"last_updated"`
	Version     string    `json:"version"`
}

type EditorSettings struct {
	FontFamily           string `json:"font_family"`
	FontSize             int    `json:"font_size"`
	TabSize              int    `json:"tab_size"`
	InsertSpaces         bool   `json:"insert_spaces"`
	WordWrap             bool   `json:"word_wrap"`
	LineNumbers          bool   `json:"line_numbers"`
	SyntaxHighlight      bool   `json:"syntax_highlight"`
	AutoComplete         bool   `json:"auto_complete"`
	AutoSave             bool   `json:"auto_save"`
	AutoFormat           bool   `json:"auto_format"`
	ShowWhitespace       bool   `json:"show_whitespace"`
	HighlightCurrentLine bool   `json:"highlight_current_line"`
}

type AISettings struct {
	PreferredModel  string  `json:"preferred_model"`
	Temperature     float64 `json:"temperature"`
	MaxTokens       int     `json:"max_tokens"`
	ContextWindow   int     `json:"context_window"`
	AutoSuggestions bool    `json:"auto_suggestions"`
	ExplainCode     bool    `json:"explain_code"`
	CodeCompletion  bool    `json:"code_completion"`
	ReviewCode      bool    `json:"review_code"`
	GenerateTests   bool    `json:"generate_tests"`
	OptimizeCode    bool    `json:"optimize_code"`
	ResponseFormat  string  `json:"response_format"`
}

type FormattingPreferences struct {
	IndentStyle    string `json:"indent_style"`
	IndentSize     int    `json:"indent_size"`
	MaxLineLength  int    `json:"max_line_length"`
	TrailingCommas bool   `json:"trailing_commas"`
	Semicolons     bool   `json:"semicolons"`
	SingleQuotes   bool   `json:"single_quotes"`
	BracketSpacing bool   `json:"bracket_spacing"`
	ArrowParens    string `json:"arrow_parens"`
	PrintWidth     int    `json:"print_width"`
}

type NotificationSettings struct {
	EmailNotifications bool        `json:"email_notifications"`
	PushNotifications  bool        `json:"push_notifications"`
	SoundEnabled       bool        `json:"sound_enabled"`
	NotifyOnBuild      bool        `json:"notify_on_build"`
	NotifyOnDeploy     bool        `json:"notify_on_deploy"`
	NotifyOnError      bool        `json:"notify_on_error"`
	QuietHours         *QuietHours `json:"quiet_hours"`
}

type QuietHours struct {
	Enabled   bool   `json:"enabled"`
	StartTime string `json:"start_time"`
	EndTime   string `json:"end_time"`
	TimeZone  string `json:"timezone"`
}

type ProjectSettings struct {
	AutoDetectLanguage  bool   `json:"auto_detect_language"`
	AutoDetectFramework bool   `json:"auto_detect_framework"`
	DefaultBuildCommand string `json:"default_build_command"`
	DefaultTestCommand  string `json:"default_test_command"`
	PreferredTerminal   string `json:"preferred_terminal"`
	GitIntegration      bool   `json:"git_integration"`
	AutoSyncSettings    bool   `json:"auto_sync_settings"`
}

type PrivacySettings struct {
	ShareUsageData      bool          `json:"share_usage_data"`
	ShareCrashReports   bool          `json:"share_crash_reports"`
	AllowTelemetry      bool          `json:"allow_telemetry"`
	DataRetentionPeriod time.Duration `json:"data_retention_period"`
	AnonymizeData       bool          `json:"anonymize_data"`
}

type CustomCommand struct {
	Name             string            `json:"name"`
	Command          string            `json:"command"`
	Description      string            `json:"description"`
	Shortcut         string            `json:"shortcut"`
	WorkingDirectory string            `json:"working_directory"`
	Environment      map[string]string `json:"environment"`
}

type WorkspaceLayout struct {
	Panels           []*PanelLayout `json:"panels"`
	SplitOrientation string         `json:"split_orientation"`
	SidebarVisible   bool           `json:"sidebar_visible"`
	TerminalVisible  bool           `json:"terminal_visible"`
	MiniMapVisible   bool           `json:"minimap_visible"`
}

type PanelLayout struct {
	Type     string `json:"type"`
	Position string `json:"position"`
	Size     int    `json:"size"`
	Visible  bool   `json:"visible"`
}

// WorkspaceContext represents the current workspace state
type WorkspaceContext struct {
	// Workspace identification
	ID   string        `json:"id"`
	Name string        `json:"name"`
	Path string        `json:"path"`
	Type WorkspaceType `json:"type"`

	// Workspace structure
	Projects   []*ProjectContext `json:"projects"`
	OpenFiles  []string          `json:"open_files"`
	ActiveFile string            `json:"active_file"`

	// Workspace settings
	Settings      *WorkspaceSettings     `json:"settings"`
	Configuration map[string]interface{} `json:"configuration"`

	// Workspace state
	Layout    *WorkspaceLayout `json:"layout"`
	ViewState *ViewState       `json:"view_state"`

	// Recent activity
	RecentActivity []*WorkspaceActivity `json:"recent_activity"`

	// Metadata
	CreatedAt    time.Time `json:"created_at"`
	LastAccessed time.Time `json:"last_accessed"`
	Version      string    `json:"version"`
}

type WorkspaceType string

const (
	WorkspaceTypeSingle WorkspaceType = "single"
	WorkspaceTypeMulti  WorkspaceType = "multi"
	WorkspaceTypeRemote WorkspaceType = "remote"
)

type WorkspaceSettings struct {
	AutoSave       bool          `json:"auto_save"`
	AutoReload     bool          `json:"auto_reload"`
	WatchFiles     bool          `json:"watch_files"`
	IndexFiles     bool          `json:"index_files"`
	BackupEnabled  bool          `json:"backup_enabled"`
	BackupInterval time.Duration `json:"backup_interval"`
	SyncSettings   bool          `json:"sync_settings"`
}

type ViewState struct {
	ActivePanel     string                 `json:"active_panel"`
	PanelStates     map[string]interface{} `json:"panel_states"`
	ScrollPositions map[string]int         `json:"scroll_positions"`
	FoldedSections  map[string][]int       `json:"folded_sections"`
	Bookmarks       []*Bookmark            `json:"bookmarks"`
}

type Bookmark struct {
	ID        string    `json:"id"`
	Name      string    `json:"name"`
	File      string    `json:"file"`
	Line      int       `json:"line"`
	Column    int       `json:"column"`
	CreatedAt time.Time `json:"created_at"`
}

type WorkspaceActivity struct {
	Type        string                 `json:"type"`
	Description string                 `json:"description"`
	Timestamp   time.Time              `json:"timestamp"`
	Files       []string               `json:"files"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// SessionContext represents the current session state
type SessionContext struct {
	// Session identification
	ID           string    `json:"id"`
	UserID       string    `json:"user_id"`
	StartTime    time.Time `json:"start_time"`
	LastActivity time.Time `json:"last_activity"`

	// Session state
	ConversationHistory []*ConversationEntry `json:"conversation_history"`
	CommandHistory      []*CommandEntry      `json:"command_history"`
	UndoStack           []*UndoEntry         `json:"undo_stack"`
	RedoStack           []*RedoEntry         `json:"redo_stack"`

	// Current operation context
	CurrentOperation *OperationContext  `json:"current_operation"`
	OperationQueue   []*QueuedOperation `json:"operation_queue"`

	// Session metrics
	Metrics *SessionMetrics `json:"metrics"`

	// Temporary data
	TemporaryData map[string]interface{} `json:"temporary_data"`
	Cache         map[string]interface{} `json:"cache"`

	// Session settings
	Settings *SessionSettings `json:"settings"`
}

type ConversationEntry struct {
	ID        string                 `json:"id"`
	Type      ConversationType       `json:"type"`
	Content   string                 `json:"content"`
	Response  string                 `json:"response"`
	Timestamp time.Time              `json:"timestamp"`
	Context   map[string]interface{} `json:"context"`
	Metadata  map[string]interface{} `json:"metadata"`
}

type ConversationType string

const (
	ConversationQuery       ConversationType = "query"
	ConversationCommand     ConversationType = "command"
	ConversationExplanation ConversationType = "explanation"
	ConversationReview      ConversationType = "review"
	ConversationGeneration  ConversationType = "generation"
)

type CommandEntry struct {
	ID         string        `json:"id"`
	Command    string        `json:"command"`
	Arguments  []string      `json:"arguments"`
	Result     interface{}   `json:"result"`
	Status     CommandStatus `json:"status"`
	ExecutedAt time.Time     `json:"executed_at"`
	Duration   time.Duration `json:"duration"`
	Error      string        `json:"error,omitempty"`
}

type CommandStatus string

const (
	CommandStatusSuccess  CommandStatus = "success"
	CommandStatusFailed   CommandStatus = "failed"
	CommandStatusRunning  CommandStatus = "running"
	CommandStatusCanceled CommandStatus = "canceled"
)

type UndoEntry struct {
	ID          string      `json:"id"`
	Action      string      `json:"action"`
	Description string      `json:"description"`
	Data        interface{} `json:"data"`
	Timestamp   time.Time   `json:"timestamp"`
}

type RedoEntry struct {
	ID          string      `json:"id"`
	Action      string      `json:"action"`
	Description string      `json:"description"`
	Data        interface{} `json:"data"`
	Timestamp   time.Time   `json:"timestamp"`
}

type OperationContext struct {
	ID               string                 `json:"id"`
	Type             string                 `json:"type"`
	Status           string                 `json:"status"`
	Progress         float64                `json:"progress"`
	StartTime        time.Time              `json:"start_time"`
	EstimatedEndTime time.Time              `json:"estimated_end_time"`
	Description      string                 `json:"description"`
	Metadata         map[string]interface{} `json:"metadata"`
}

type QueuedOperation struct {
	ID                string                 `json:"id"`
	Type              string                 `json:"type"`
	Priority          int                    `json:"priority"`
	Parameters        map[string]interface{} `json:"parameters"`
	QueuedAt          time.Time              `json:"queued_at"`
	EstimatedDuration time.Duration          `json:"estimated_duration"`
}

type SessionMetrics struct {
	CommandsExecuted    int           `json:"commands_executed"`
	QueriesProcessed    int           `json:"queries_processed"`
	FilesModified       int           `json:"files_modified"`
	LinesWritten        int           `json:"lines_written"`
	ErrorsEncountered   int           `json:"errors_encountered"`
	AverageResponseTime time.Duration `json:"average_response_time"`
	SessionDuration     time.Duration `json:"session_duration"`
}

type SessionSettings struct {
	AutoSave        bool          `json:"auto_save"`
	SaveInterval    time.Duration `json:"save_interval"`
	MaxHistorySize  int           `json:"max_history_size"`
	MaxUndoStack    int           `json:"max_undo_stack"`
	TimeoutDuration time.Duration `json:"timeout_duration"`
	VerboseLogging  bool          `json:"verbose_logging"`
}

// Context management structures

type ContextSnapshot struct {
	ID               string                  `json:"id"`
	Timestamp        time.Time               `json:"timestamp"`
	ProjectContext   *ProjectContext         `json:"project_context"`
	SelectedFiles    map[string]*FileContext `json:"selected_files"`
	UserPreferences  *UserPreferences        `json:"user_preferences"`
	WorkspaceContext *WorkspaceContext       `json:"workspace_context"`
	SessionContext   *SessionContext         `json:"session_context"`
	Metadata         map[string]interface{}  `json:"metadata"`
}

type ContextFrame struct {
	ID          string           `json:"id"`
	Name        string           `json:"name"`
	Description string           `json:"description"`
	Context     *ContextSnapshot `json:"context"`
	CreatedAt   time.Time        `json:"created_at"`
	Tags        []string         `json:"tags"`
}

// Observer and listener interfaces

type ContextObserver interface {
	OnContextChanged(ctx *ContextSnapshot)
	OnProjectChanged(project *ProjectContext)
	OnFileSelectionChanged(files map[string]*FileContext)
	OnPreferencesChanged(preferences *UserPreferences)
}

type ContextChangeListener interface {
	OnBeforeContextChange(oldCtx, newCtx *ContextSnapshot) error
	OnAfterContextChange(oldCtx, newCtx *ContextSnapshot)
	OnContextError(err error)
}

// Persistence and enrichment interfaces

type ContextPersistenceProvider interface {
	SaveContext(ctx *ContextSnapshot) error
	LoadContext(id string) (*ContextSnapshot, error)
	ListContexts() ([]*ContextSnapshot, error)
	DeleteContext(id string) error
}

type ContextEnrichmentProvider interface {
	EnrichProjectContext(ctx *ProjectContext) error
	EnrichFileContext(ctx *FileContext) error
	EnrichWorkspaceContext(ctx *WorkspaceContext) error
}

// Configuration

type ContextManagerConfig struct {
	MaxHistorySize    int           `json:"max_history_size"`
	MaxStackDepth     int           `json:"max_stack_depth"`
	AutoSave          bool          `json:"auto_save"`
	SaveInterval      time.Duration `json:"save_interval"`
	CacheExpiry       time.Duration `json:"cache_expiry"`
	EnablePersistence bool          `json:"enable_persistence"`
	EnableEnrichment  bool          `json:"enable_enrichment"`
	WatchFileChanges  bool          `json:"watch_file_changes"`
	IndexFiles        bool          `json:"index_files"`
	LoadConfigFiles   bool          `json:"load_config_files"`
	AnalyzeCode       bool          `json:"analyze_code"`
	DetectLanguages   bool          `json:"detect_languages"`
	DetectFrameworks  bool          `json:"detect_frameworks"`
	LoadGitInfo       bool          `json:"load_git_info"`
	LoadDependencies  bool          `json:"load_dependencies"`
	LoadMetrics       bool          `json:"load_metrics"`
}

// NewContextManager creates a new context manager
func NewContextManager(config *ContextManagerConfig, logger logger.Logger) *ContextManager {
	if config == nil {
		config = &ContextManagerConfig{
			MaxHistorySize:    100,
			MaxStackDepth:     50,
			AutoSave:          true,
			SaveInterval:      time.Minute * 5,
			CacheExpiry:       time.Hour,
			EnablePersistence: true,
			EnableEnrichment:  true,
			WatchFileChanges:  true,
			IndexFiles:        true,
			LoadConfigFiles:   true,
			AnalyzeCode:       true,
			DetectLanguages:   true,
			DetectFrameworks:  true,
			LoadGitInfo:       true,
			LoadDependencies:  true,
			LoadMetrics:       true,
		}
	}

	cm := &ContextManager{
		selectedFiles:       make(map[string]*FileContext),
		contextHistory:      make([]*ContextSnapshot, 0, config.MaxHistorySize),
		contextStack:        make([]*ContextFrame, 0, config.MaxStackDepth),
		maxHistorySize:      config.MaxHistorySize,
		maxStackDepth:       config.MaxStackDepth,
		observers:           make([]ContextObserver, 0),
		changeListeners:     make([]ContextChangeListener, 0),
		config:              config,
		logger:              logger,
		autoSave:            config.AutoSave,
		saveInterval:        config.SaveInterval,
		contextCache:        make(map[string]interface{}),
		cacheExpiry:         config.CacheExpiry,
		enrichmentProviders: make([]ContextEnrichmentProvider, 0),
	}

	// Initialize session context
	cm.sessionContext = &SessionContext{
		ID:                  generateID(),
		StartTime:           time.Now(),
		LastActivity:        time.Now(),
		ConversationHistory: make([]*ConversationEntry, 0),
		CommandHistory:      make([]*CommandEntry, 0),
		UndoStack:           make([]*UndoEntry, 0),
		RedoStack:           make([]*RedoEntry, 0),
		TemporaryData:       make(map[string]interface{}),
		Cache:               make(map[string]interface{}),
		Metrics:             &SessionMetrics{},
		Settings: &SessionSettings{
			AutoSave:        true,
			SaveInterval:    time.Minute * 5,
			MaxHistorySize:  1000,
			MaxUndoStack:    100,
			TimeoutDuration: time.Minute * 30,
			VerboseLogging:  false,
		},
	}

	// Start auto-save routine if enabled
	if config.AutoSave {
		go cm.startAutoSave()
	}

	return cm
}

// Core context management methods

func (cm *ContextManager) SetActiveProject(projectPath string) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	// Load project context
	project, err := cm.loadProjectContext(projectPath)
	if err != nil {
		return fmt.Errorf("failed to load project context: %v", err)
	}

	oldProject := cm.activeProject
	cm.activeProject = project
	cm.lastUpdate = time.Now()
	cm.isDirty = true

	// Notify observers
	for _, observer := range cm.observers {
		observer.OnProjectChanged(project)
	}

	// Create snapshot
	snapshot := cm.createSnapshot()
	cm.addToHistory(snapshot)

	// Notify change listeners
	if oldProject != nil {
		oldSnapshot := &ContextSnapshot{ProjectContext: oldProject}
		for _, listener := range cm.changeListeners {
			listener.OnAfterContextChange(oldSnapshot, snapshot)
		}
	}

	cm.logger.Info("Active project changed", map[string]interface{}{
		"project": project.Name,
		"path":    projectPath,
	})
	return nil
}

func (cm *ContextManager) AddSelectedFile(filePath string) error {
	if !filepath.IsAbs(filePath) {
		return fmt.Errorf("invalid file path: %s", filePath)
	}

	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return fmt.Errorf("file does not exist: %s", filePath)
	}

	cm.mu.Lock()
	defer cm.mu.Unlock()

	// Load file context
	fileCtx, err := cm.loadFileContext(filePath)
	if err != nil {
		return fmt.Errorf("failed to load file context: %v", err)
	}

	cm.selectedFiles[filePath] = fileCtx
	cm.lastUpdate = time.Now()
	cm.isDirty = true

	// Notify observers
	for _, observer := range cm.observers {
		observer.OnFileSelectionChanged(cm.selectedFiles)
	}

	cm.logger.Debug("File added to selection", map[string]interface{}{"file": filePath})
	return nil
}

func (cm *ContextManager) RemoveSelectedFile(filePath string) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	if _, exists := cm.selectedFiles[filePath]; !exists {
		return fmt.Errorf("file not in selection: %s", filePath)
	}

	delete(cm.selectedFiles, filePath)
	cm.lastUpdate = time.Now()
	cm.isDirty = true

	// Notify observers
	for _, observer := range cm.observers {
		observer.OnFileSelectionChanged(cm.selectedFiles)
	}

	cm.logger.Debug("File removed from selection", map[string]interface{}{"file": filePath})
	return nil
}

func (cm *ContextManager) SetUserPreferences(preferences *UserPreferences) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	oldPrefs := cm.userPreferences
	cm.userPreferences = preferences
	cm.userPreferences.LastUpdated = time.Now()
	cm.lastUpdate = time.Now()
	cm.isDirty = true

	// Notify observers
	for _, observer := range cm.observers {
		observer.OnPreferencesChanged(preferences)
	}

	// Create snapshot
	snapshot := cm.createSnapshot()
	cm.addToHistory(snapshot)

	// Notify change listeners
	if oldPrefs != nil {
		oldSnapshot := &ContextSnapshot{UserPreferences: oldPrefs}
		for _, listener := range cm.changeListeners {
			listener.OnAfterContextChange(oldSnapshot, snapshot)
		}
	}

	cm.logger.Info("User preferences updated")
	return nil
}

// Context retrieval methods

func (cm *ContextManager) GetActiveProject() *ProjectContext {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.activeProject
}

func (cm *ContextManager) GetSelectedFiles() map[string]*FileContext {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	// Return a copy to prevent external modification
	result := make(map[string]*FileContext)
	for k, v := range cm.selectedFiles {
		result[k] = v
	}
	return result
}

func (cm *ContextManager) GetUserPreferences() *UserPreferences {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.userPreferences
}

func (cm *ContextManager) GetWorkspaceContext() *WorkspaceContext {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.workspaceContext
}

func (cm *ContextManager) GetSessionContext() *SessionContext {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.sessionContext
}

func (cm *ContextManager) GetCurrentContext() *ContextSnapshot {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.createSnapshot()
}

// Context history and stack management

func (cm *ContextManager) PushContext(name, description string) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	if len(cm.contextStack) >= cm.maxStackDepth {
		return fmt.Errorf("context stack overflow: maximum depth %d reached", cm.maxStackDepth)
	}

	frame := &ContextFrame{
		ID:          generateID(),
		Name:        name,
		Description: description,
		Context:     cm.createSnapshot(),
		CreatedAt:   time.Now(),
	}

	cm.contextStack = append(cm.contextStack, frame)
	cm.logger.Debug("Context pushed to stack", map[string]interface{}{
		"name":       name,
		"stack_size": len(cm.contextStack),
	})
	return nil
}

func (cm *ContextManager) PopContext() (*ContextFrame, error) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	if len(cm.contextStack) == 0 {
		return nil, fmt.Errorf("context stack is empty")
	}

	frame := cm.contextStack[len(cm.contextStack)-1]
	cm.contextStack = cm.contextStack[:len(cm.contextStack)-1]

	// Restore context
	cm.restoreFromSnapshot(frame.Context)

	cm.logger.Debug("Context popped from stack", map[string]interface{}{
		"name":       frame.Name,
		"stack_size": len(cm.contextStack),
	})
	return frame, nil
}

func (cm *ContextManager) GetContextHistory() []*ContextSnapshot {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	// Return a copy
	result := make([]*ContextSnapshot, len(cm.contextHistory))
	copy(result, cm.contextHistory)
	return result
}

// Observer management

func (cm *ContextManager) AddObserver(observer ContextObserver) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.observers = append(cm.observers, observer)
}

func (cm *ContextManager) RemoveObserver(observer ContextObserver) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	for i, obs := range cm.observers {
		if obs == observer {
			cm.observers = append(cm.observers[:i], cm.observers[i+1:]...)
			break
		}
	}
}

func (cm *ContextManager) AddChangeListener(listener ContextChangeListener) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.changeListeners = append(cm.changeListeners, listener)
}

// Enrichment and persistence

func (cm *ContextManager) AddEnrichmentProvider(provider ContextEnrichmentProvider) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.enrichmentProviders = append(cm.enrichmentProviders, provider)
}

func (cm *ContextManager) SetPersistenceProvider(provider ContextPersistenceProvider) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.persistenceProvider = provider
}

func (cm *ContextManager) SaveContext() error {
	cm.mu.RLock()
	snapshot := cm.createSnapshot()
	provider := cm.persistenceProvider
	cm.mu.RUnlock()

	if provider == nil {
		return fmt.Errorf("no persistence provider configured")
	}

	err := provider.SaveContext(snapshot)
	if err != nil {
		cm.logger.Error("Failed to save context", err)
		return err
	}

	cm.mu.Lock()
	cm.isDirty = false
	cm.mu.Unlock()

	cm.logger.Debug("Context saved successfully", map[string]interface{}{"id": snapshot.ID})
	return nil
}

func (cm *ContextManager) LoadContext(id string) error {
	if cm.persistenceProvider == nil {
		return fmt.Errorf("no persistence provider configured")
	}

	snapshot, err := cm.persistenceProvider.LoadContext(id)
	if err != nil {
		return fmt.Errorf("failed to load context: %v", err)
	}

	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.restoreFromSnapshot(snapshot)
	cm.logger.Info("Context loaded successfully", map[string]interface{}{"id": id})
	return nil
}

// Session management

func (cm *ContextManager) AddConversationEntry(entry *ConversationEntry) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.sessionContext.ConversationHistory = append(cm.sessionContext.ConversationHistory, entry)

	// Trim history if too large
	maxSize := cm.sessionContext.Settings.MaxHistorySize
	if len(cm.sessionContext.ConversationHistory) > maxSize {
		cm.sessionContext.ConversationHistory = cm.sessionContext.ConversationHistory[1:]
	}

	cm.sessionContext.LastActivity = time.Now()
	cm.isDirty = true
}

func (cm *ContextManager) AddCommandEntry(entry *CommandEntry) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.sessionContext.CommandHistory = append(cm.sessionContext.CommandHistory, entry)
	cm.sessionContext.Metrics.CommandsExecuted++
	cm.sessionContext.LastActivity = time.Now()
	cm.isDirty = true
}

func (cm *ContextManager) AddUndoEntry(entry *UndoEntry) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.sessionContext.UndoStack = append(cm.sessionContext.UndoStack, entry)

	// Clear redo stack when new undo entry is added
	cm.sessionContext.RedoStack = cm.sessionContext.RedoStack[:0]

	// Trim undo stack if too large
	maxSize := cm.sessionContext.Settings.MaxUndoStack
	if len(cm.sessionContext.UndoStack) > maxSize {
		cm.sessionContext.UndoStack = cm.sessionContext.UndoStack[1:]
	}

	cm.isDirty = true
}

// Cache management

func (cm *ContextManager) GetFromCache(key string) (interface{}, bool) {
	cm.cacheMu.RLock()
	defer cm.cacheMu.RUnlock()

	value, exists := cm.contextCache[key]
	return value, exists
}

func (cm *ContextManager) SetInCache(key string, value interface{}) {
	cm.cacheMu.Lock()
	defer cm.cacheMu.Unlock()

	cm.contextCache[key] = value
}

func (cm *ContextManager) ClearCache() {
	cm.cacheMu.Lock()
	defer cm.cacheMu.Unlock()

	cm.contextCache = make(map[string]interface{})
}

// Private helper methods

func (cm *ContextManager) loadProjectContext(projectPath string) (*ProjectContext, error) {
	// Check cache first
	if cached, exists := cm.GetFromCache("project:" + projectPath); exists {
		if project, ok := cached.(*ProjectContext); ok {
			return project, nil
		}
	}

	project := &ProjectContext{
		ID:           generateID(),
		Path:         projectPath,
		Name:         filepath.Base(projectPath),
		CreatedAt:    time.Now(),
		LastAccessed: time.Now(),
		CustomData:   make(map[string]interface{}),
	}

	// Detect project type and languages
	if cm.config.DetectLanguages {
		err := cm.detectProjectLanguages(project)
		if err != nil {
			cm.logger.Warn("Failed to detect project languages", map[string]interface{}{"error": err})

		}
	}

	// Detect frameworks
	if cm.config.DetectFrameworks {
		err := cm.detectProjectFrameworks(project)
		if err != nil {
			cm.logger.Warn("Failed to detect project frameworks", map[string]interface{}{"error": err})
		}
	}

	// Load project structure
	if cm.config.IndexFiles {
		err := cm.loadProjectStructure(project)
		if err != nil {
			cm.logger.Warn("Failed to load project structure", map[string]interface{}{"error": err})
		}
	}

	// Load Git information
	if cm.config.LoadGitInfo {
		err := cm.loadGitInfo(project)
		if err != nil {
			cm.logger.Debug("No Git information found", map[string]interface{}{"project": projectPath})
		}
	}

	// Load configuration files
	if cm.config.LoadConfigFiles {
		err := cm.loadProjectConfiguration(project)
		if err != nil {
			cm.logger.Warn("Failed to load project configuration", map[string]interface{}{"error": err})
		}
	}

	// Load dependencies
	if cm.config.LoadDependencies {
		err := cm.loadProjectDependencies(project)
		if err != nil {
			cm.logger.Warn("Failed to load project dependencies", map[string]interface{}{"error": err})
		}
	}

	// Enrich project context
	if cm.config.EnableEnrichment {
		for _, provider := range cm.enrichmentProviders {
			if err := provider.EnrichProjectContext(project); err != nil {
				cm.logger.Warn("Failed to enrich project context", map[string]interface{}{"error": err})
			}
		}
	}

	// Cache the project context
	cm.SetInCache("project:"+projectPath, project)

	return project, nil
}

func (cm *ContextManager) loadFileContext(filePath string) (*FileContext, error) {
	// Resolve absolute path early and use it for cache keys
	absPath, err := filepath.Abs(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to get absolute path: %v", err)
	}

	// Check cache first (use absolute path to avoid duplicates)
	if cached, exists := cm.GetFromCache("file:" + absPath); exists {
		if fileCtx, ok := cached.(*FileContext); ok {
			return fileCtx, nil
		}
	}

	// Get file info
	fileInfo, err := os.Stat(absPath)
	if err != nil {
		return nil, fmt.Errorf("failed to stat file: %v", err)
	}

	fileCtx := &FileContext{
		Path:         filePath,
		AbsolutePath: absPath,
		Name:         filepath.Base(absPath),
		Extension:    filepath.Ext(absPath),
		Size:         fileInfo.Size(),
		LastModified: fileInfo.ModTime(),
		LastAccessed: time.Now(),
		CustomData:   make(map[string]interface{}),
	}

	// Determine relative path if we have an active project
	if cm.activeProject != nil && cm.activeProject.Path != "" {
		if relPath, err := filepath.Rel(cm.activeProject.Path, absPath); err == nil {
			// Only set relative path when within project
			if !strings.HasPrefix(relPath, "..") {
				fileCtx.RelativePath = relPath
			}
		}
	}

	// Detect language
	fileCtx.Language = cm.detectFileLanguage(absPath)

	// Load file content if it's a text file and not too large
	if cm.config.AnalyzeCode && fileCtx.Size > 0 && fileCtx.Size < 1024*1024 { // 1MB limit
		contentBytes, err := os.ReadFile(absPath)
		if err == nil {
			fileCtx.Content = string(contentBytes)
			// If file ends with newline, Count("\n") gives number of lines-1, so add 1 when non-empty
			if fileCtx.Content == "" {
				fileCtx.LineCount = 0
			} else {
				fileCtx.LineCount = strings.Count(fileCtx.Content, "\n") + 1
			}
			fileCtx.Encoding = "utf-8" // Simplified; replace with real detection if needed
		} else {
			// Log read error but continue (we can still return a FileContext without content)
			cm.logger.Debug("Failed to read file content", map[string]interface{}{
				"file":  absPath,
				"error": err.Error(),
			})
		}
	}

	// Analyze code structure if content is available
	if fileCtx.Content != "" && cm.config.AnalyzeCode {
		if err := cm.analyzeFileStructure(fileCtx); err != nil {
			cm.logger.Debug("Failed to analyze file structure", map[string]interface{}{
				"file":  absPath,
				"error": err.Error(),
			})
		}
	}

	// Enrich file context
	if cm.config.EnableEnrichment {
		for _, provider := range cm.enrichmentProviders {
			if err := provider.EnrichFileContext(fileCtx); err != nil {
				cm.logger.Debug("Failed to enrich file context", map[string]interface{}{
					"file":     absPath,
					"provider": fmt.Sprintf("%T", provider),
					"error":    err.Error(),
				})
			}
		}
	}

	// Cache the file context (use absolute path)
	cm.SetInCache("file:"+absPath, fileCtx)

	return fileCtx, nil
}

func (cm *ContextManager) createSnapshot() *ContextSnapshot {
	return &ContextSnapshot{
		ID:               generateID(),
		Timestamp:        time.Now(),
		ProjectContext:   cm.activeProject,
		SelectedFiles:    cm.copySelectedFiles(),
		UserPreferences:  cm.userPreferences,
		WorkspaceContext: cm.workspaceContext,
		SessionContext:   cm.sessionContext,
		Metadata:         make(map[string]interface{}),
	}
}

func (cm *ContextManager) copySelectedFiles() map[string]*FileContext {
	result := make(map[string]*FileContext)
	for k, v := range cm.selectedFiles {
		result[k] = v
	}
	return result
}

func (cm *ContextManager) addToHistory(snapshot *ContextSnapshot) {
	cm.contextHistory = append(cm.contextHistory, snapshot)

	// Trim history if too large
	if len(cm.contextHistory) > cm.maxHistorySize {
		cm.contextHistory = cm.contextHistory[1:]
	}
}

func (cm *ContextManager) restoreFromSnapshot(snapshot *ContextSnapshot) {
	cm.activeProject = snapshot.ProjectContext
	cm.selectedFiles = snapshot.SelectedFiles
	cm.userPreferences = snapshot.UserPreferences
	cm.workspaceContext = snapshot.WorkspaceContext
	cm.sessionContext = snapshot.SessionContext
	cm.lastUpdate = time.Now()
	cm.isDirty = true
}

func (cm *ContextManager) startAutoSave() {
	ticker := time.NewTicker(cm.saveInterval)
	defer ticker.Stop()

	for range ticker.C {
		cm.mu.RLock()
		isDirty := cm.isDirty
		cm.mu.RUnlock()

		if isDirty && cm.persistenceProvider != nil {
			if err := cm.SaveContext(); err != nil {
				cm.logger.Error("Auto-save failed", err, map[string]interface{}{
					"operation": "autosave",
				})
			}
		}
	}
}

// Placeholder implementations for language/framework detection
func (cm *ContextManager) detectProjectLanguages(project *ProjectContext) error {
	// Implementation would analyze files to detect languages
	project.Language = []string{"go"} // Simplified
	return nil
}

func (cm *ContextManager) detectProjectFrameworks(project *ProjectContext) error {
	// Implementation would analyze config files to detect frameworks
	project.Framework = []string{} // Simplified
	return nil
}

func (cm *ContextManager) loadProjectStructure(project *ProjectContext) error {
	// Implementation would build file tree
	project.Structure = &ProjectStructure{
		RootDirectory: project.Path,
	}
	return nil
}

func (cm *ContextManager) loadGitInfo(project *ProjectContext) error {
	// Implementation would read Git information
	project.VersionControl = &VersionControlInfo{
		Type: "git",
	}
	return nil
}

func (cm *ContextManager) loadProjectConfiguration(project *ProjectContext) error {
	// Implementation would load and parse config files
	project.Configuration = &ProjectConfiguration{
		ConfigFiles: make(map[string]*ConfigFile),
	}
	return nil
}

func (cm *ContextManager) loadProjectDependencies(project *ProjectContext) error {
	// Implementation would analyze dependency files
	return nil
}

func (cm *ContextManager) detectFileLanguage(filePath string) string {
	ext := strings.ToLower(filepath.Ext(filePath))
	langMap := map[string]string{
		".go":   "go",
		".py":   "python",
		".js":   "javascript",
		".ts":   "typescript",
		".java": "java",
		".cpp":  "cpp",
		".c":    "c",
		".cs":   "csharp",
		".rb":   "ruby",
		".php":  "php",
		".rs":   "rust",
	}

	if lang, exists := langMap[ext]; exists {
		return lang
	}
	return "unknown"
}

func (cm *ContextManager) analyzeFileStructure(fileCtx *FileContext) error {
	// Implementation would parse code to extract functions, classes, etc.
	fileCtx.Functions = []*FunctionInfo{}
	fileCtx.Classes = []*ClassInfo{}
	fileCtx.Variables = []*VariableInfo{}
	return nil
}

// Utility function to generate unique IDs
func generateID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}

func (cm *ContextManager) startCacheCleanup() {
	ticker := time.NewTicker(cm.cacheExpiry / 2)
	go func() {
		for range ticker.C {
			cm.cleanExpiredCacheEntries()
		}
	}()
}

func (cm *ContextManager) cleanExpiredCacheEntries() {
	// Implementation for cleaning expired cache entries
}
