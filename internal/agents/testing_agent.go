package agents

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/agents"
	"github.com/yourusername/ai-code-assistant/internal/app"
	"github.com/yourusername/ai-code-assistant/internal/indexer"
	"github.com/yourusername/ai-code-assistant/internal/logger"
)

// TestingAgent generates comprehensive test suites
type TestingAgent struct {
	// Core components
	llmProvider    llm.Provider
	indexer        *indexer.UltraFastIndexer
	contextManager *app.ContextManager

	// Agent configuration
	config *TestingAgentConfig
	logger logger.Logger

	// Capabilities
	capabilities *AgentCapabilities

	// Test generation engines
	unitTestGenerator        *UnitTestGenerator
	integrationTestGenerator *IntegrationTestGenerator
	e2eTestGenerator         *E2ETestGenerator
	performanceTestGenerator *PerformanceTestGenerator

	// Test analysis engines
	codeAnalyzer       *TestCodeAnalyzer
	coverageAnalyzer   *CoverageAnalyzer
	complexityAnalyzer *TestComplexityAnalyzer
	dependencyAnalyzer *TestDependencyAnalyzer

	// Test strategy engines
	testStrategyEngine *TestStrategyEngine
	scenarioGenerator  *TestScenarioGenerator
	dataGenerator      *TestDataGenerator
	mockGenerator      *MockGenerator

	// Framework support
	frameworkAdapters   map[string]TestFrameworkAdapter
	assertionGenerators map[string]AssertionGenerator
	mockFrameworks      map[string]MockFramework

	// Quality assurance
	testQualityAnalyzer *TestQualityAnalyzer
	testSmellDetector   *TestSmellDetector
	testOptimizer       *TestOptimizer

	// Statistics and monitoring
	metrics *TestingAgentMetrics

	// State management
	mu     sync.RWMutex
	status AgentStatus
}

// TestingAgentConfig contains testing agent configuration
type TestingAgentConfig struct {
	// Test generation capabilities
	EnableUnitTests        bool `json:"enable_unit_tests"`
	EnableIntegrationTests bool `json:"enable_integration_tests"`
	EnableE2ETests         bool `json:"enable_e2e_tests"`
	EnablePerformanceTests bool `json:"enable_performance_tests"`
	EnableSecurityTests    bool `json:"enable_security_tests"`

	// Test quality settings
	EnableTestOptimization   bool `json:"enable_test_optimization"`
	EnableTestAnalysis       bool `json:"enable_test_analysis"`
	EnableTestSmellDetection bool `json:"enable_test_smell_detection"`
	EnableCoverageAnalysis   bool `json:"enable_coverage_analysis"`

	// Generation strategies
	TestGenerationStrategy TestGenerationStrategy `json:"test_generation_strategy"`
	CoverageTarget         float64                `json:"coverage_target"`
	MaxTestsPerFunction    int                    `json:"max_tests_per_function"`
	IncludeEdgeCases       bool                   `json:"include_edge_cases"`
	IncludeErrorCases      bool                   `json:"include_error_cases"`

	// Framework configurations
	UnitTestFrameworks    map[string]*TestFrameworkConfig    `json:"unit_test_frameworks"`
	IntegrationFrameworks map[string]*TestFrameworkConfig    `json:"integration_frameworks"`
	E2EFrameworks         map[string]*TestFrameworkConfig    `json:"e2e_frameworks"`
	MockingFrameworks     map[string]*MockingFrameworkConfig `json:"mocking_frameworks"`

	// Test data generation
	EnableTestDataGeneration bool     `json:"enable_test_data_generation"`
	TestDataStrategies       []string `json:"test_data_strategies"`
	IncludeRandomData        bool     `json:"include_random_data"`
	IncludeRealisticData     bool     `json:"include_realistic_data"`

	// Code analysis settings
	AnalyzeExistingTests bool    `json:"analyze_existing_tests"`
	MinimumCodeCoverage  float64 `json:"minimum_code_coverage"`
	ComplexityThreshold  int     `json:"complexity_threshold"`

	// Output settings
	GenerateTestSuites    bool `json:"generate_test_suites"`
	GenerateDocumentation bool `json:"generate_documentation"`
	IncludeSetupTeardown  bool `json:"include_setup_teardown"`
	IncludeHelperMethods  bool `json:"include_helper_methods"`

	// Language-specific settings
	LanguageConfigurations map[string]*LanguageTestConfig `json:"language_configurations"`

	// Performance settings
	MaxGenerationTime        time.Duration `json:"max_generation_time"`
	EnableParallelGeneration bool          `json:"enable_parallel_generation"`
	CacheGeneratedTests      bool          `json:"cache_generated_tests"`

	// LLM settings
	LLMModel    string  `json:"llm_model"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float32 `json:"temperature"`
}

type TestGenerationStrategy string

const (
	StrategyComprehensive  TestGenerationStrategy = "comprehensive"
	StrategyTargeted       TestGenerationStrategy = "targeted"
	StrategyMinimal        TestGenerationStrategy = "minimal"
	StrategyBehaviorDriven TestGenerationStrategy = "behavior_driven"
	StrategyPropertyBased  TestGenerationStrategy = "property_based"
)

type TestFrameworkConfig struct {
	Name              string            `json:"name"`
	Version           string            `json:"version,omitempty"`
	ImportStatements  []string          `json:"import_statements"`
	TestAnnotations   map[string]string `json:"test_annotations"`
	AssertionMethods  map[string]string `json:"assertion_methods"`
	SetupMethods      []string          `json:"setup_methods"`
	TeardownMethods   []string          `json:"teardown_methods"`
	TestNamingPattern string            `json:"test_naming_pattern"`
	FileNamingPattern string            `json:"file_naming_pattern"`
	TemplateStructure string            `json:"template_structure"`
}

type MockingFrameworkConfig struct {
	Name               string            `json:"name"`
	ImportStatements   []string          `json:"import_statements"`
	MockCreationSyntax string            `json:"mock_creation_syntax"`
	StubSyntax         string            `json:"stub_syntax"`
	VerificationSyntax string            `json:"verification_syntax"`
	SpySyntax          string            `json:"spy_syntax,omitempty"`
	MockAnnotations    map[string]string `json:"mock_annotations,omitempty"`
}

type LanguageTestConfig struct {
	DefaultUnitFramework        string   `json:"default_unit_framework"`
	DefaultIntegrationFramework string   `json:"default_integration_framework"`
	DefaultMockingFramework     string   `json:"default_mocking_framework"`
	TestFileExtension           string   `json:"test_file_extension"`
	TestDirectoryPattern        string   `json:"test_directory_pattern"`
	CommonPatterns              []string `json:"common_patterns"`
	TestingIdioms               []string `json:"testing_idioms"`
	BestPractices               []string `json:"best_practices"`
}

// Request and response structures

type TestGenerationRequest struct {
	Code          string                 `json:"code"`
	Language      string                 `json:"language"`
	TestType      TestType               `json:"test_type"`
	Context       *TestGenerationContext `json:"context,omitempty"`
	Options       *TestGenerationOptions `json:"options,omitempty"`
	ExistingTests []*ExistingTest        `json:"existing_tests,omitempty"`
	Dependencies  []*TestDependency      `json:"dependencies,omitempty"`
}

type TestType string

const (
	TestTypeUnit        TestType = "unit"
	TestTypeIntegration TestType = "integration"
	TestTypeE2E         TestType = "e2e"
	TestTypePerformance TestType = "performance"
	TestTypeSecurity    TestType = "security"
	TestTypeFunctional  TestType = "functional"
	TestTypeAcceptance  TestType = "acceptance"
	TestTypeAll         TestType = "all"
)

type TestGenerationContext struct {
	FilePath             string                `json:"file_path,omitempty"`
	FunctionName         string                `json:"function_name,omitempty"`
	ClassName            string                `json:"class_name,omitempty"`
	ModuleName           string                `json:"module_name,omitempty"`
	ProjectStructure     *ProjectTestStructure `json:"project_structure,omitempty"`
	ExternalDependencies []string              `json:"external_dependencies,omitempty"`
	DatabaseSchemas      []*DatabaseSchema     `json:"database_schemas,omitempty"`
	APISpecifications    []*APISpecification   `json:"api_specifications,omitempty"`
	BusinessRequirements []string              `json:"business_requirements,omitempty"`
}

type ProjectTestStructure struct {
	TestDirectory      string   `json:"test_directory"`
	TestFilePattern    string   `json:"test_file_pattern"`
	Framework          string   `json:"framework"`
	ConfigurationFiles []string `json:"configuration_files"`
	SetupScripts       []string `json:"setup_scripts"`
}

type DatabaseSchema struct {
	Name          string               `json:"name"`
	Tables        []*TableSchema       `json:"tables"`
	Relationships []*TableRelationship `json:"relationships"`
}

type TableSchema struct {
	Name       string          `json:"name"`
	Columns    []*ColumnSchema `json:"columns"`
	PrimaryKey []string        `json:"primary_key"`
	Indexes    []string        `json:"indexes"`
}

type ColumnSchema struct {
	Name         string `json:"name"`
	Type         string `json:"type"`
	Nullable     bool   `json:"nullable"`
	DefaultValue string `json:"default_value,omitempty"`
}

type TableRelationship struct {
	FromTable  string `json:"from_table"`
	ToTable    string `json:"to_table"`
	Type       string `json:"type"`
	ForeignKey string `json:"foreign_key"`
}

type APISpecification struct {
	Endpoint               string          `json:"endpoint"`
	Method                 string          `json:"method"`
	Parameters             []*APIParameter `json:"parameters"`
	ResponseSchema         string          `json:"response_schema"`
	AuthenticationRequired bool            `json:"authentication_required"`
}

type APIParameter struct {
	Name        string `json:"name"`
	Type        string `json:"type"`
	Required    bool   `json:"required"`
	Description string `json:"description,omitempty"`
}

type TestGenerationOptions struct {
	Framework            string              `json:"framework,omitempty"`
	MockingFramework     string              `json:"mocking_framework,omitempty"`
	TestStyle            TestStyle           `json:"test_style"`
	CoverageTarget       float64             `json:"coverage_target,omitempty"`
	IncludeSetup         bool                `json:"include_setup"`
	IncludeTeardown      bool                `json:"include_teardown"`
	GenerateMocks        bool                `json:"generate_mocks"`
	GenerateTestData     bool                `json:"generate_test_data"`
	IncludeDocumentation bool                `json:"include_documentation"`
	TestNamingStyle      TestNamingStyle     `json:"test_naming_style"`
	ValidationLevel      TestValidationLevel `json:"validation_level"`
	SpecificScenarios    []string            `json:"specific_scenarios,omitempty"`
}

type TestStyle string

const (
	TestStyleArrange     TestStyle = "arrange_act_assert"
	TestStyleGiven       TestStyle = "given_when_then"
	TestStyleBDD         TestStyle = "behavior_driven"
	TestStyleTraditional TestStyle = "traditional"
)

type TestNamingStyle string

const (
	NamingStyleDescriptive TestNamingStyle = "descriptive"
	NamingStyleMethodName  TestNamingStyle = "method_name"
	NamingStyleBDD         TestNamingStyle = "bdd"
	NamingStyleShould      TestNamingStyle = "should"
)

type TestValidationLevel string

const (
	ValidationMinimal       TestValidationLevel = "minimal"
	ValidationStandard      TestValidationLevel = "standard"
	ValidationStrict        TestValidationLevel = "strict"
	ValidationComprehensive TestValidationLevel = "comprehensive"
)

type ExistingTest struct {
	FilePath         string              `json:"file_path"`
	TestName         string              `json:"test_name"`
	TestType         TestType            `json:"test_type"`
	CoveredFunctions []string            `json:"covered_functions"`
	TestQuality      *TestQualityMetrics `json:"test_quality,omitempty"`
}

type TestQualityMetrics struct {
	CoveragePercentage   float64  `json:"coverage_percentage"`
	AssertionCount       int      `json:"assertion_count"`
	ComplexityScore      int      `json:"complexity_score"`
	MaintainabilityScore float64  `json:"maintainability_score"`
	Smells               []string `json:"smells,omitempty"`
}

type TestDependency struct {
	Name            string                 `json:"name"`
	Type            DependencyType         `json:"type"`
	MockingRequired bool                   `json:"mocking_required"`
	SetupRequired   bool                   `json:"setup_required"`
	Configuration   map[string]interface{} `json:"configuration,omitempty"`
}

// Response structures

type TestGenerationResponse struct {
	GeneratedTests    []*GeneratedTest        `json:"generated_tests"`
	TestSuite         *TestSuite              `json:"test_suite,omitempty"`
	Coverage          *CoverageReport         `json:"coverage,omitempty"`
	TestStrategy      *TestStrategy           `json:"test_strategy,omitempty"`
	Recommendations   []*TestRecommendation   `json:"recommendations,omitempty"`
	QualityAnalysis   *TestQualityAnalysis    `json:"quality_analysis,omitempty"`
	Documentation     *TestDocumentation      `json:"documentation,omitempty"`
	SetupInstructions *SetupInstructions      `json:"setup_instructions,omitempty"`
	Metadata          *TestGenerationMetadata `json:"metadata"`
}

type GeneratedTest struct {
	Name              string            `json:"name"`
	Type              TestType          `json:"test_type"`
	FilePath          string            `json:"file_path"`
	Code              string            `json:"code"`
	Description       string            `json:"description"`
	TestCases         []*TestCase       `json:"test_cases"`
	Dependencies      []*TestDependency `json:"dependencies"`
	SetupCode         string            `json:"setup_code,omitempty"`
	TeardownCode      string            `json:"teardown_code,omitempty"`
	MockDefinitions   []*MockDefinition `json:"mock_definitions,omitempty"`
	TestData          []*TestDataSet    `json:"test_data,omitempty"`
	AssertionCount    int               `json:"assertion_count"`
	EstimatedCoverage float64           `json:"estimated_coverage"`
}

type TestCase struct {
	Name           string           `json:"name"`
	Description    string           `json:"description"`
	Scenario       string           `json:"scenario"`
	TestType       TestCaseType     `json:"test_type"`
	InputData      interface{}      `json:"input_data"`
	ExpectedOutput interface{}      `json:"expected_output"`
	Preconditions  []string         `json:"preconditions,omitempty"`
	Steps          []*TestStep      `json:"steps"`
	Assertions     []*TestAssertion `json:"assertions"`
	Priority       Priority         `json:"priority"`
}

type TestCaseType string

const (
	TestCaseHappyPath   TestCaseType = "happy_path"
	TestCaseEdgeCase    TestCaseType = "edge_case"
	TestCaseErrorCase   TestCaseType = "error_case"
	TestCaseBoundary    TestCaseType = "boundary"
	TestCaseNegative    TestCaseType = "negative"
	TestCasePerformance TestCaseType = "performance"
)

type TestStep struct {
	StepNumber     int    `json:"step_number"`
	Action         string `json:"action"`
	Description    string `json:"description"`
	Code           string `json:"code"`
	ExpectedResult string `json:"expected_result,omitempty"`
}

type TestAssertion struct {
	Type         AssertionType `json:"type"`
	Description  string        `json:"description"`
	Code         string        `json:"code"`
	Expected     interface{}   `json:"expected"`
	Actual       string        `json:"actual"`
	ErrorMessage string        `json:"error_message,omitempty"`
}

type AssertionType string

const (
	AssertionEquals      AssertionType = "equals"
	AssertionNotEquals   AssertionType = "not_equals"
	AssertionNull        AssertionType = "null"
	AssertionNotNull     AssertionType = "not_null"
	AssertionTrue        AssertionType = "true"
	AssertionFalse       AssertionType = "false"
	AssertionContains    AssertionType = "contains"
	AssertionGreaterThan AssertionType = "greater_than"
	AssertionLessThan    AssertionType = "less_than"
	AssertionThrows      AssertionType = "throws"
	AssertionNotThrows   AssertionType = "not_throws"
)

type MockDefinition struct {
	Name             string        `json:"name"`
	Type             string        `json:"type"`
	Interface        string        `json:"interface"`
	Methods          []*MockMethod `json:"methods"`
	SetupCode        string        `json:"setup_code"`
	BehaviorCode     string        `json:"behavior_code"`
	VerificationCode string        `json:"verification_code,omitempty"`
}

type MockMethod struct {
	Name            string           `json:"name"`
	Parameters      []string         `json:"parameters"`
	ReturnType      string           `json:"return_type"`
	BehaviorType    MockBehaviorType `json:"behavior_type"`
	ReturnValue     interface{}      `json:"return_value,omitempty"`
	ThrowsException string           `json:"throws_exception,omitempty"`
	CallCount       int              `json:"call_count,omitempty"`
}

type MockBehaviorType string

const (
	MockBehaviorReturn   MockBehaviorType = "return"
	MockBehaviorThrow    MockBehaviorType = "throw"
	MockBehaviorCallback MockBehaviorType = "callback"
	MockBehaviorDefault  MockBehaviorType = "default"
)

type TestDataSet struct {
	Name               string       `json:"name"`
	Type               TestDataType `json:"type"`
	Description        string       `json:"description"`
	Data               interface{}  `json:"data"`
	GenerationStrategy string       `json:"generation_strategy"`
	Size               int          `json:"size,omitempty"`
}

type TestDataType string

const (
	TestDataStatic    TestDataType = "static"
	TestDataGenerated TestDataType = "generated"
	TestDataFixture   TestDataType = "fixture"
	TestDataBuilder   TestDataType = "builder"
	TestDataFactory   TestDataType = "factory"
)

type TestSuite struct {
	Name              string                 `json:"name"`
	Description       string                 `json:"description"`
	Tests             []*GeneratedTest       `json:"tests"`
	SetupSuite        string                 `json:"setup_suite,omitempty"`
	TeardownSuite     string                 `json:"teardown_suite,omitempty"`
	Configuration     map[string]interface{} `json:"configuration,omitempty"`
	ExecutionOrder    []string               `json:"execution_order,omitempty"`
	ParallelExecution bool                   `json:"parallel_execution"`
	EstimatedRuntime  time.Duration          `json:"estimated_runtime"`
}

type CoverageReport struct {
	OverallCoverage    float64            `json:"overall_coverage"`
	LineCoverage       float64            `json:"line_coverage"`
	BranchCoverage     float64            `json:"branch_coverage"`
	FunctionCoverage   float64            `json:"function_coverage"`
	CoverageByFile     map[string]float64 `json:"coverage_by_file"`
	CoverageByFunction map[string]float64 `json:"coverage_by_function"`
	UncoveredLines     []*UncoveredArea   `json:"uncovered_lines"`
	CoverageGaps       []*CoverageGap     `json:"coverage_gaps"`
	CoverageTarget     float64            `json:"coverage_target"`
	MeetsTarget        bool               `json:"meets_target"`
}

type UncoveredArea struct {
	FilePath      string `json:"file_path"`
	FunctionName  string `json:"function_name,omitempty"`
	LineNumbers   []int  `json:"line_numbers"`
	Reason        string `json:"reason"`
	Severity      string `json:"severity"`
	SuggestedTest string `json:"suggested_test,omitempty"`
}

type CoverageGap struct {
	Type           string `json:"type"`
	Description    string `json:"description"`
	Location       string `json:"location"`
	Impact         string `json:"impact"`
	Recommendation string `json:"recommendation"`
}

type TestStrategy struct {
	Overview          string                 `json:"overview"`
	Approach          TestGenerationStrategy `json:"approach"`
	TestTypes         []TestType             `json:"test_types"`
	Priorities        []*TestPriority        `json:"priorities"`
	TestingPhases     []*TestingPhase        `json:"testing_phases"`
	RiskAreas         []*RiskArea            `json:"risk_areas"`
	TestingGuidelines []string               `json:"testing_guidelines"`
}

type TestPriority struct {
	Component      string     `json:"component"`
	Priority       Priority   `json:"priority"`
	Rationale      string     `json:"rationale"`
	TestTypes      []TestType `json:"test_types"`
	CoverageTarget float64    `json:"coverage_target"`
}

type TestingPhase struct {
	Phase        string        `json:"phase"`
	Description  string        `json:"description"`
	TestTypes    []TestType    `json:"test_types"`
	Duration     time.Duration `json:"duration"`
	Dependencies []string      `json:"dependencies"`
	Deliverables []string      `json:"deliverables"`
}

type RiskArea struct {
	Area               string `json:"area"`
	RiskLevel          string `json:"risk_level"`
	Description        string `json:"description"`
	MitigationStrategy string `json:"mitigation_strategy"`
	TestingApproach    string `json:"testing_approach"`
}

type TestRecommendation struct {
	Type           RecommendationType            `json:"type"`
	Priority       Priority                      `json:"priority"`
	Category       string                        `json:"category"`
	Title          string                        `json:"title"`
	Description    string                        `json:"description"`
	Benefits       []string                      `json:"benefits"`
	Implementation *RecommendationImplementation `json:"implementation"`
	Examples       []*RecommendationExample      `json:"examples,omitempty"`
}

type RecommendationExample struct {
	Title       string `json:"title"`
	Description string `json:"description"`
	Code        string `json:"code"`
	Context     string `json:"context,omitempty"`
}

type TestQualityAnalysis struct {
	OverallQuality   float64                  `json:"overall_quality"`
	QualityByTest    map[string]float64       `json:"quality_by_test"`
	Strengths        []string                 `json:"strengths"`
	Weaknesses       []string                 `json:"weaknesses"`
	TestSmells       []*TestSmell             `json:"test_smells"`
	BestPractices    []*BestPracticeAdherence `json:"best_practices"`
	ImprovementAreas []*ImprovementArea       `json:"improvement_areas"`
}

type TestSmell struct {
	Name           string        `json:"name"`
	Description    string        `json:"description"`
	Location       *CodeLocation `json:"location"`
	Severity       string        `json:"severity"`
	Impact         string        `json:"impact"`
	Recommendation string        `json:"recommendation"`
	Examples       []string      `json:"examples,omitempty"`
}

type BestPracticeAdherence struct {
	Practice   string   `json:"practice"`
	Adherence  float64  `json:"adherence"`
	Examples   []string `json:"examples"`
	Violations []string `json:"violations,omitempty"`
}

type ImprovementArea struct {
	Area            string   `json:"area"`
	CurrentScore    float64  `json:"current_score"`
	TargetScore     float64  `json:"target_score"`
	Actions         []string `json:"actions"`
	Priority        Priority `json:"priority"`
	EstimatedEffort string   `json:"estimated_effort"`
}

type TestDocumentation struct {
	TestingGuide         string `json:"testing_guide"`
	SetupInstructions    string `json:"setup_instructions"`
	RunningTests         string `json:"running_tests"`
	TestingBestPractices string `json:"testing_best_practices"`
	TroubleshootingGuide string `json:"troubleshooting_guide"`
	MaintenanceGuide     string `json:"maintenance_guide"`
}

type SetupInstructions struct {
	Prerequisites     []string            `json:"prerequisites"`
	InstallationSteps []*InstallationStep `json:"installation_steps"`
	Configuration     *TestConfiguration  `json:"configuration"`
	VerificationSteps []string            `json:"verification_steps"`
	CommonIssues      []*CommonIssue      `json:"common_issues,omitempty"`
}

type InstallationStep struct {
	StepNumber     int      `json:"step_number"`
	Description    string   `json:"description"`
	Command        string   `json:"command,omitempty"`
	ExpectedOutput string   `json:"expected_output,omitempty"`
	Notes          []string `json:"notes,omitempty"`
}

type TestConfiguration struct {
	ConfigFiles          []*ConfigFile            `json:"config_files"`
	EnvironmentVariables map[string]string        `json:"environment_variables,omitempty"`
	DatabaseSetup        *DatabaseSetupConfig     `json:"database_setup,omitempty"`
	ExternalServices     []*ExternalServiceConfig `json:"external_services,omitempty"`
}

type ConfigFile struct {
	FilePath    string `json:"file_path"`
	Content     string `json:"content"`
	Description string `json:"description"`
	Required    bool   `json:"required"`
}

type DatabaseSetupConfig struct {
	DatabaseType     string   `json:"database_type"`
	SetupScripts     []string `json:"setup_scripts"`
	MigrationScripts []string `json:"migration_scripts"`
	TestData         []string `json:"test_data"`
	CleanupScripts   []string `json:"cleanup_scripts"`
}

type ExternalServiceConfig struct {
	ServiceName       string   `json:"service_name"`
	MockingRequired   bool     `json:"mocking_required"`
	SetupInstructions []string `json:"setup_instructions"`
	MockingFramework  string   `json:"mocking_framework,omitempty"`
	TestEndpoints     []string `json:"test_endpoints,omitempty"`
}

type CommonIssue struct {
	Issue       string   `json:"issue"`
	Description string   `json:"description"`
	Symptoms    []string `json:"symptoms"`
	Solutions   []string `json:"solutions"`
	Prevention  []string `json:"prevention,omitempty"`
}

type TestGenerationMetadata struct {
	GenerationTime         time.Duration          `json:"generation_time"`
	GenerationStrategy     TestGenerationStrategy `json:"generation_strategy"`
	TestsGenerated         int                    `json:"tests_generated"`
	LinesGenerated         int                    `json:"lines_generated"`
	Framework              string                 `json:"framework"`
	Language               string                 `json:"language"`
	CoverageAchieved       float64                `json:"coverage_achieved"`
	QualityScore           float64                `json:"quality_score"`
	Confidence             float64                `json:"confidence"`
	LimitationsEncountered []string               `json:"limitations_encountered,omitempty"`
}

// TestingAgentMetrics tracks testing agent performance
type TestingAgentMetrics struct {
	TotalGenerations        int64              `json:"total_generations"`
	GenerationsByType       map[TestType]int64 `json:"generations_by_type"`
	GenerationsByLanguage   map[string]int64   `json:"generations_by_language"`
	AverageGenerationTime   time.Duration      `json:"average_generation_time"`
	TestsGenerated          int64              `json:"tests_generated"`
	AverageCoverageAchieved float64            `json:"average_coverage_achieved"`
	AverageQualityScore     float64            `json:"average_quality_score"`
	FrameworkUsage          map[string]int64   `json:"framework_usage"`
	LastGeneration          time.Time          `json:"last_generation"`
	mu                      sync.RWMutex
}

// NewTestingAgent creates a new testing agent
func NewTestingAgent(llmProvider llm.Provider, indexer *indexer.UltraFastIndexer, config *TestingAgentConfig, logger logger.Logger) *TestingAgent {
	if config == nil {
		config = &TestingAgentConfig{
			EnableUnitTests:          true,
			EnableIntegrationTests:   true,
			EnableE2ETests:           false,
			EnablePerformanceTests:   false,
			EnableSecurityTests:      false,
			EnableTestOptimization:   true,
			EnableTestAnalysis:       true,
			EnableTestSmellDetection: true,
			EnableCoverageAnalysis:   true,
			TestGenerationStrategy:   StrategyComprehensive,
			CoverageTarget:           0.8,
			MaxTestsPerFunction:      10,
			IncludeEdgeCases:         true,
			IncludeErrorCases:        true,
			EnableTestDataGeneration: true,
			TestDataStrategies:       []string{"realistic", "boundary", "random"},
			IncludeRandomData:        true,
			IncludeRealisticData:     true,
			AnalyzeExistingTests:     true,
			MinimumCodeCoverage:      0.7,
			ComplexityThreshold:      10,
			GenerateTestSuites:       true,
			GenerateDocumentation:    true,
			IncludeSetupTeardown:     true,
			IncludeHelperMethods:     true,
			MaxGenerationTime:        time.Minute * 5,
			EnableParallelGeneration: true,
			CacheGeneratedTests:      true,
			LLMModel:                 "gpt-4",
			MaxTokens:                3072,
			Temperature:              0.3,
			UnitTestFrameworks:       make(map[string]*TestFrameworkConfig),
			IntegrationFrameworks:    make(map[string]*TestFrameworkConfig),
			E2EFrameworks:            make(map[string]*TestFrameworkConfig),
			MockingFrameworks:        make(map[string]*MockingFrameworkConfig),
			LanguageConfigurations:   make(map[string]*LanguageTestConfig),
		}

		// Initialize default configurations
		config.UnitTestFrameworks = ta.getDefaultUnitFrameworks()
		config.IntegrationFrameworks = ta.getDefaultIntegrationFrameworks()
		config.MockingFrameworks = ta.getDefaultMockingFrameworks()
		config.LanguageConfigurations = ta.getDefaultLanguageTestConfigs()
	}

	agent := &TestingAgent{
		llmProvider: llmProvider,
		indexer:     indexer,
		config:      config,
		logger:      logger,
		status:      StatusIdle,
		metrics: &TestingAgentMetrics{
			GenerationsByType:     make(map[TestType]int64),
			GenerationsByLanguage: make(map[string]int64),
			FrameworkUsage:        make(map[string]int64),
		},
		frameworkAdapters:   make(map[string]TestFrameworkAdapter),
		assertionGenerators: make(map[string]AssertionGenerator),
		mockFrameworks:      make(map[string]MockFramework),
	}

	// Initialize capabilities
	agent.initializeCapabilities()

	// Initialize components
	agent.initializeComponents()

	return agent
}

// ProcessRequest processes a test generation request
func (ta *TestingAgent) ProcessRequest(ctx context.Context, request *AgentRequest) (*AgentResponse, error) {
	start := time.Now()
	ta.status = StatusBusy
	defer func() { ta.status = StatusIdle }()

	// Parse test generation request
	testRequest, err := ta.parseTestRequest(request)
	if err != nil {
		return nil, fmt.Errorf("failed to parse test request: %v", err)
	}

	// Apply timeout
	testCtx := ctx
	if ta.config.MaxGenerationTime > 0 {
		var cancel context.CancelFunc
		testCtx, cancel = context.WithTimeout(ctx, ta.config.MaxGenerationTime)
		defer cancel()
	}

	// Perform test generation
	testResponse, err := ta.performTestGeneration(testCtx, testRequest)
	if err != nil {
		ta.updateMetrics(testRequest.TestType, testRequest.Language, false, time.Since(start), 0)
		return nil, fmt.Errorf("test generation failed: %v", err)
	}

	// Create agent response
	response := &AgentResponse{
		RequestID:      request.ID,
		AgentType:      ta.GetType(),
		AgentVersion:   ta.GetVersion(),
		Result:         testResponse,
		Confidence:     ta.calculateConfidence(testRequest, testResponse),
		ProcessingTime: time.Since(start),
		CreatedAt:      time.Now(),
	}

	// Update metrics
	testsGenerated := len(testResponse.GeneratedTests)
	ta.updateMetrics(testRequest.TestType, testRequest.Language, true, time.Since(start), testsGenerated)

	return response, nil
}

// performTestGeneration performs comprehensive test generation
func (ta *TestingAgent) performTestGeneration(ctx context.Context, request *TestGenerationRequest) (*TestGenerationResponse, error) {
	response := &TestGenerationResponse{
		GeneratedTests:  []*GeneratedTest{},
		Recommendations: []*TestRecommendation{},
	}

	// Analyze existing code and tests
	codeAnalysis := ta.codeAnalyzer.Analyze(request.Code, request.Language, request.Context)

	// Analyze existing tests if available
	var existingTestAnalysis *ExistingTestAnalysis
	if len(request.ExistingTests) > 0 && ta.config.AnalyzeExistingTests {
		existingTestAnalysis = ta.analyzeExistingTests(request.ExistingTests)
	}

	// Generate test strategy
	testStrategy := ta.testStrategyEngine.GenerateStrategy(
		codeAnalysis,
		request.TestType,
		request.Options,
		existingTestAnalysis,
	)
	response.TestStrategy = testStrategy

	// Generate tests based on type
	var generationTasks []func() error

	if ta.shouldGenerateTestType(request, TestTypeUnit) && ta.config.EnableUnitTests {
		generationTasks = append(generationTasks, func() error {
			unitTests := ta.generateUnitTests(ctx, request, codeAnalysis)
			response.GeneratedTests = append(response.GeneratedTests, unitTests...)
			return nil
		})
	}

	if ta.shouldGenerateTestType(request, TestTypeIntegration) && ta.config.EnableIntegrationTests {
		generationTasks = append(generationTasks, func() error {
			integrationTests := ta.generateIntegrationTests(ctx, request, codeAnalysis)
			response.GeneratedTests = append(response.GeneratedTests, integrationTests...)
			return nil
		})
	}

	if ta.shouldGenerateTestType(request, TestTypeE2E) && ta.config.EnableE2ETests {
		generationTasks = append(generationTasks, func() error {
			e2eTests := ta.generateE2ETests(ctx, request, codeAnalysis)
			response.GeneratedTests = append(response.GeneratedTests, e2eTests...)
			return nil
		})
	}

	if ta.shouldGenerateTestType(request, TestTypePerformance) && ta.config.EnablePerformanceTests {
		generationTasks = append(generationTasks, func() error {
			performanceTests := ta.generatePerformanceTests(ctx, request, codeAnalysis)
			response.GeneratedTests = append(response.GeneratedTests, performanceTests...)
			return nil
		})
	}

	// Execute test generation tasks
	if ta.config.EnableParallelGeneration && len(generationTasks) > 1 {
		err := ta.executeParallelGeneration(ctx, generationTasks)
		if err != nil {
			ta.logger.Warn("Some test generation tasks failed", "error", err)
		}
	} else {
		err := ta.executeSequentialGeneration(ctx, generationTasks)
		if err != nil {
			ta.logger.Warn("Sequential test generation failed", "error", err)
		}
	}

	// Generate test suite if requested
	if ta.config.GenerateTestSuites {
		response.TestSuite = ta.generateTestSuite(response.GeneratedTests, request)
	}

	// Analyze coverage
	if ta.config.EnableCoverageAnalysis {
		response.Coverage = ta.analyzeCoverage(response.GeneratedTests, codeAnalysis)
	}

	// Generate quality analysis
	if ta.config.EnableTestAnalysis {
		response.QualityAnalysis = ta.analyzeTestQuality(response.GeneratedTests)
	}

	// Generate recommendations
	response.Recommendations = ta.generateTestRecommendations(ctx, request, response, codeAnalysis)

	// Generate documentation
	if ta.config.GenerateDocumentation {
		response.Documentation = ta.generateTestDocumentation(response.GeneratedTests, request)
	}

	// Generate setup instructions
	response.SetupInstructions = ta.generateSetupInstructions(response.GeneratedTests, request)

	// Create metadata
	response.Metadata = &TestGenerationMetadata{
		GenerationTime:     time.Since(time.Now().Add(-time.Minute)), // Simplified
		GenerationStrategy: ta.config.TestGenerationStrategy,
		TestsGenerated:     len(response.GeneratedTests),
		LinesGenerated:     ta.calculateTotalLines(response.GeneratedTests),
		Framework:          ta.determineFramework(request),
		Language:           request.Language,
		CoverageAchieved:   ta.calculateAchievedCoverage(response.Coverage),
		QualityScore:       ta.calculateOverallQuality(response.QualityAnalysis),
		Confidence:         ta.calculateGenerationConfidence(response),
	}

	return response, nil
}

// Test generation methods
func (ta *TestingAgent) generateUnitTests(ctx context.Context, request *TestGenerationRequest, analysis *CodeAnalysisResult) []*GeneratedTest {
	var tests []*GeneratedTest
	// Use LLM for intelligent test generation
	llmTests := ta.generateLLMUnitTests(ctx, request, analysis)
	tests = append(tests, llmTests...)

	// Use rule-based generation for comprehensive coverage
	ruleBasedTests := ta.unitTestGenerator.GenerateTests(
		request.Code,
		request.Language,
		request.Options,
		analysis,
	)
	tests = append(tests, ruleBasedTests...)

	// Remove duplicates and optimize
	tests = ta.optimizeTests(tests)

	return tests
}

func (ta *TestingAgent) generateLLMUnitTests(ctx context.Context, request *TestGenerationRequest, analysis *CodeAnalysisResult) []*GeneratedTest {
	prompt := ta.buildUnitTestPrompt(request, analysis)
	llmRequest := &llm.CompletionRequest{
		Prompt:      prompt,
		Model:       ta.config.LLMModel,
		MaxTokens:   ta.config.MaxTokens,
		Temperature: ta.config.Temperature,
	}

	llmResponse, err := ta.llmProvider.Complete(ctx, llmRequest)
	if err != nil {
		ta.logger.Warn("LLM unit test generation failed", "error", err)
		return []*GeneratedTest{}
	}

	// Parse LLM response into test structures
	tests := ta.parseLLMTestResponse(llmResponse.Text, TestTypeUnit)

	// Enhance with additional metadata
	for _, test := range tests {
		test = ta.enhanceGeneratedTest(test, request, analysis)
	}

	return tests
}

func (ta *TestingAgent) generateIntegrationTests(ctx context.Context, request *TestGenerationRequest, analysis *CodeAnalysisResult) []*GeneratedTest {
	var tests []*GeneratedTest
	// Generate integration tests based on dependencies
	if len(request.Dependencies) > 0 {
		dependencyTests := ta.integrationTestGenerator.GenerateForDependencies(
			request.Code,
			request.Language,
			request.Dependencies,
			request.Options,
		)
		tests = append(tests, dependencyTests...)
	}

	// Generate database integration tests if schemas are provided
	if request.Context != nil && len(request.Context.DatabaseSchemas) > 0 {
		dbTests := ta.integrationTestGenerator.GenerateForDatabase(
			request.Code,
			request.Language,
			request.Context.DatabaseSchemas,
			request.Options,
		)
		tests = append(tests, dbTests...)
	}

	// Generate API integration tests if specifications are provided
	if request.Context != nil && len(request.Context.APISpecifications) > 0 {
		apiTests := ta.integrationTestGenerator.GenerateForAPI(
			request.Code,
			request.Language,
			request.Context.APISpecifications,
			request.Options,
		)
		tests = append(tests, apiTests...)
	}

	return tests
}

func (ta *TestingAgent) generateE2ETests(ctx context.Context, request *TestGenerationRequest, analysis *CodeAnalysisResult) []*GeneratedTest {
	var tests []*GeneratedTest
	// Generate E2E tests based on business requirements
	if request.Context != nil && len(request.Context.BusinessRequirements) > 0 {
		businessTests := ta.e2eTestGenerator.GenerateForBusinessRequirements(
			request.Context.BusinessRequirements,
			request.Language,
			request.Options,
		)
		tests = append(tests, businessTests...)
	}

	// Generate user journey tests
	userJourneyTests := ta.e2eTestGenerator.GenerateUserJourneyTests(
		request.Code,
		request.Language,
		analysis,
		request.Options,
	)
	tests = append(tests, userJourneyTests...)

	return tests
}

func (ta *TestingAgent) generatePerformanceTests(ctx context.Context, request *TestGenerationRequest, analysis *CodeAnalysisResult) []*GeneratedTest {
	var tests []*GeneratedTest
	// Generate load tests
	loadTests := ta.performanceTestGenerator.GenerateLoadTests(
		request.Code,
		request.Language,
		analysis,
		request.Options,
	)
	tests = append(tests, loadTests...)

	// Generate stress tests
	stressTests := ta.performanceTestGenerator.GenerateStressTests(
		request.Code,
		request.Language,
		analysis,
		request.Options,
	)
	tests = append(tests, stressTests...)

	// Generate benchmark tests
	benchmarkTests := ta.performanceTestGenerator.GenerateBenchmarkTests(
		request.Code,
		request.Language,
		analysis,
		request.Options,
	)
	tests = append(tests, benchmarkTests...)

	return tests
}

// Test analysis and optimization methods
func (ta *TestingAgent) optimizeTests(tests []*GeneratedTest) []*GeneratedTest {
	if !ta.config.EnableTestOptimization {
		return tests
	}
	// Remove duplicates
	uniqueTests := ta.testOptimizer.RemoveDuplicates(tests)

	// Optimize test order
	optimizedTests := ta.testOptimizer.OptimizeExecutionOrder(uniqueTests)

	// Merge similar tests where appropriate
	mergedTests := ta.testOptimizer.MergeSimilarTests(optimizedTests)

	return mergedTests
}

func (ta *TestingAgent) analyzeTestQuality(tests []*GeneratedTest) *TestQualityAnalysis {
	if !ta.config.EnableTestAnalysis {
		return nil
	}
	qualityByTest := make(map[string]float64)
	var overallQuality float64
	var testSmells []*TestSmell
	var bestPractices []*BestPracticeAdherence

	for _, test := range tests {
		// Analyze individual test quality
		testQuality := ta.testQualityAnalyzer.AnalyzeTest(test)
		qualityByTest[test.Name] = testQuality.OverallScore
		overallQuality += testQuality.OverallScore

		// Detect test smells
		if ta.config.EnableTestSmellDetection {
			smells := ta.testSmellDetector.DetectSmells(test)
			testSmells = append(testSmells, smells...)
		}

		// Check best practices adherence
		practices := ta.testQualityAnalyzer.CheckBestPractices(test)
		bestPractices = append(bestPractices, practices...)
	}

	if len(tests) > 0 {
		overallQuality = overallQuality / float64(len(tests))
	}

	// Identify strengths and weaknesses
	strengths := ta.identifyQualityStrengths(tests, bestPractices)
	weaknesses := ta.identifyQualityWeaknesses(testSmells)

	// Generate improvement areas
	improvementAreas := ta.generateImprovementAreas(testSmells, bestPractices)

	return &TestQualityAnalysis{
		OverallQuality:   overallQuality,
		QualityByTest:    qualityByTest,
		Strengths:        strengths,
		Weaknesses:       weaknesses,
		TestSmells:       testSmells,
		BestPractices:    bestPractices,
		ImprovementAreas: improvementAreas,
	}
}

func (ta *TestingAgent) analyzeCoverage(tests []*GeneratedTest, codeAnalysis *CodeAnalysisResult) *CoverageReport {
	if !ta.config.EnableCoverageAnalysis {
		return nil
	}
	// Calculate coverage metrics
	coverage := ta.coverageAnalyzer.CalculateCoverage(tests, codeAnalysis)

	// Identify uncovered areas
	uncoveredAreas := ta.coverageAnalyzer.FindUncoveredAreas(tests, codeAnalysis)

	// Identify coverage gaps
	coverageGaps := ta.coverageAnalyzer.FindCoverageGaps(tests, codeAnalysis)

	return &CoverageReport{
		OverallCoverage:    coverage.Overall,
		LineCoverage:       coverage.Line,
		BranchCoverage:     coverage.Branch,
		FunctionCoverage:   coverage.Function,
		CoverageByFile:     coverage.ByFile,
		CoverageByFunction: coverage.ByFunction,
		UncoveredLines:     uncoveredAreas,
		CoverageGaps:       coverageGaps,
		CoverageTarget:     ta.config.CoverageTarget,
		MeetsTarget:        coverage.Overall >= ta.config.CoverageTarget,
	}
}

// Helper and utility methods
func (ta *TestingAgent) shouldGenerateTestType(request *TestGenerationRequest, testType TestType) bool {
	if request.TestType == TestTypeAll {
		return true
	}
	return request.TestType == testType
}

func (ta *TestingAgent) executeParallelGeneration(ctx context.Context, tasks []func() error) error {
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

	for err := range errorChan {
		return err
	}

	return nil
}

func (ta *TestingAgent) executeSequentialGeneration(ctx context.Context, tasks []func() error) error {
	for _, task := range tasks {
		if err := task(); err != nil {
			return err
		}
	}
	return nil
}

// Prompt generation methods
func (ta *TestingAgent) buildUnitTestPrompt(request *TestGenerationRequest, analysis *CodeAnalysisResult) string {
	var prompt strings.Builder
	prompt.WriteString("Generate comprehensive unit tests for the following code:\n\n")

	prompt.WriteString("Code:\n```")
	prompt.WriteString(request.Language)
	prompt.WriteString("\n")
	prompt.WriteString(request.Code)
	prompt.WriteString("\n```\n\n")

	if request.Context != nil {
		if request.Context.FunctionName != "" {
			prompt.WriteString(fmt.Sprintf("Function: %s\n", request.Context.FunctionName))
		}
		if request.Context.ClassName != "" {
			prompt.WriteString(fmt.Sprintf("Class: %s\n", request.Context.ClassName))
		}
	}

	prompt.WriteString("Requirements:\n")
	prompt.WriteString("1. Generate unit tests that achieve high code coverage\n")
	prompt.WriteString("2. Include tests for happy path, edge cases, and error conditions\n")
	prompt.WriteString("3. Use appropriate assertions and mocking where needed\n")
	prompt.WriteString("4. Follow best practices for test naming and structure\n")

	if request.Options != nil {
		framework := request.Options.Framework
		if framework == "" {
			framework = ta.getDefaultFramework(request.Language, TestTypeUnit)
		}
		prompt.WriteString(fmt.Sprintf("5. Use %s testing framework\n", framework))

		if request.Options.TestStyle != "" {
			prompt.WriteString(fmt.Sprintf("6. Follow %s test style\n", request.Options.TestStyle))
		}

		if request.Options.CoverageTarget > 0 {
			prompt.WriteString(fmt.Sprintf("7. Target %.0f%% code coverage\n", request.Options.CoverageTarget*100))
		}
	}

	prompt.WriteString("\nGenerate well-structured, maintainable unit tests with clear documentation.")

	return prompt.String()
}

// Framework and configuration methods
func (ta *TestingAgent) getDefaultFramework(language string, testType TestType) string {
	if langConfig, exists := ta.config.LanguageConfigurations[language]; exists {
		switch testType {
		case TestTypeUnit:
			return langConfig.DefaultUnitFramework
		case TestTypeIntegration:
			return langConfig.DefaultIntegrationFramework
		default:
			return langConfig.DefaultUnitFramework
		}
	}
	// Default frameworks by language
	defaults := map[string]string{
		"go":         "testing",
		"python":     "pytest",
		"javascript": "jest",
		"java":       "junit5",
		"csharp":     "nunit",
	}

	if framework, exists := defaults[language]; exists {
		return framework
	}

	return "unknown"
}

func (ta *TestingAgent) getDefaultUnitFrameworks() map[string]*TestFrameworkConfig {
	return map[string]*TestFrameworkConfig{
		"jest": {
			Name:              "Jest",
			ImportStatements:  []string{"const { test, expect } = require('@jest/globals');"},
			TestAnnotations:   map[string]string{"test": "test", "skip": "test.skip"},
			AssertionMethods:  map[string]string{"equals": "expect({actual}).toBe({expected})"},
			SetupMethods:      []string{"beforeEach", "beforeAll"},
			TeardownMethods:   []string{"afterEach", "afterAll"},
			TestNamingPattern: "should {expected behavior} when {condition}",
			FileNamingPattern: "{filename}.test.js",
		},
		"pytest": {
			Name:              "PyTest",
			ImportStatements:  []string{"import pytest"},
			TestAnnotations:   map[string]string{"test": "def test_", "skip": "@pytest.mark.skip"},
			AssertionMethods:  map[string]string{"equals": "assert {actual} == {expected}"},
			SetupMethods:      []string{"setup_method", "setup_class"},
			TeardownMethods:   []string{"teardown_method", "teardown_class"},
			TestNamingPattern: "test_{function_name}{scenario}",
			FileNamingPattern: "test{filename}.py",
		},
		"testing": {
			Name:              "Go Testing",
			ImportStatements:  []string{"import \"testing\""},
			TestAnnotations:   map[string]string{"test": "func Test", "skip": "t.Skip()"},
			AssertionMethods:  map[string]string{"equals": "if {actual} != {expected} { t.Errorf(...) }"},
			TestNamingPattern: "Test{FunctionName}{Scenario}",
			FileNamingPattern: "{filename}_test.go",
		},
	}
}

func (ta *TestingAgent) getDefaultIntegrationFrameworks() map[string]*TestFrameworkConfig {
	return map[string]*TestFrameworkConfig{
		"testcontainers": {
			Name:             "Testcontainers",
			ImportStatements: []string{"import { GenericContainer } from 'testcontainers';"},
		},
	}
}

func (ta *TestingAgent) getDefaultMockingFrameworks() map[string]*MockingFrameworkConfig {
	return map[string]*MockingFrameworkConfig{
		"jest": {
			Name:               "Jest Mocks",
			ImportStatements:   []string{"const { jest } = require('@jest/globals');"},
			MockCreationSyntax: "jest.fn()",
			StubSyntax:         "mockFunction.mockReturnValue({value})",
			VerificationSyntax: "expect(mockFunction).toHaveBeenCalledWith({args})",
		},
		"unittest.mock": {
			Name:               "Python Mock",
			ImportStatements:   []string{"from unittest.mock import Mock, patch"},
			MockCreationSyntax: "Mock()",
			StubSyntax:         "mock_obj.return_value = {value}",
			VerificationSyntax: "mock_obj.assert_called_with({args})",
		},
	}
}

func (ta *TestingAgent) getDefaultLanguageTestConfigs() map[string]*LanguageTestConfig {
	return map[string]*LanguageTestConfig{
		"go": {
			DefaultUnitFramework:        "testing",
			DefaultIntegrationFramework: "testing",
			DefaultMockingFramework:     "testify",
			TestFileExtension:           "_test.go",
			TestDirectoryPattern:        "same_directory",
			CommonPatterns:              []string{"table_driven", "setup_teardown", "benchmarks"},
			TestingIdioms:               []string{"t.Helper()", "t.Parallel()", "subtests"},
			BestPractices:               []string{"Use t.Helper()", "Table-driven tests", "Clear test names"},
		},
		"python": {
			DefaultUnitFramework:        "pytest",
			DefaultIntegrationFramework: "pytest",
			DefaultMockingFramework:     "unittest.mock",
			TestFileExtension:           ".py",
			TestDirectoryPattern:        "tests/",
			CommonPatterns:              []string{"fixtures", "parametrize", "markers"},
			TestingIdioms:               []string{"@pytest.fixture", "@pytest.mark.parametrize"},
			BestPractices:               []string{"Use fixtures", "Parametrized tests", "Clear assertions"},
		},
		"javascript": {
			DefaultUnitFramework:        "jest",
			DefaultIntegrationFramework: "jest",
			DefaultMockingFramework:     "jest",
			TestFileExtension:           ".test.js",
			TestDirectoryPattern:        "tests/",
			CommonPatterns:              []string{"describe_it", "before_after", "snapshots"},
			TestingIdioms:               []string{"describe()", "it()", "beforeEach()"},
			BestPractices:               []string{"Group related tests", "Use descriptive names", "Test one thing"},
		},
	}
}

// Required Agent interface methods
func (ta *TestingAgent) GetCapabilities() *AgentCapabilities {
	return ta.capabilities
}

func (ta *TestingAgent) GetType() AgentType {
	return AgentTypeTesting
}

func (ta *TestingAgent) GetVersion() string {
	return "1.0.0"
}

func (ta *TestingAgent) GetStatus() AgentStatus {
	ta.mu.RLock()
	defer ta.mu.RUnlock()
	return ta.status
}

func (ta *TestingAgent) Initialize(config interface{}) error {
	if cfg, ok := config.(*TestingAgentConfig); ok {
		ta.config = cfg
		ta.initializeComponents()
		return nil
	}
	return fmt.Errorf("invalid config type")
}

// SetConfig updates the agent's configuration dynamically
func (ta *TestingAgent) SetConfig(config interface{}) error {
	appConfig, ok := config.(*app.TestingConfig)
	if !ok {
		return fmt.Errorf("invalid config type for TestingAgent, expected *app.TestingConfig")
	}

	ta.mu.Lock()
	defer ta.mu.Unlock()

	// Convert app.TestingConfig to internal TestingAgentConfig
	newConfig := &TestingAgentConfig{
		EnableUnitTestGeneration:        appConfig.EnableUnitTestGeneration,
		EnableIntegrationTestGeneration: appConfig.EnableIntegrationTestGeneration,
		EnableMockGeneration:            appConfig.EnableMockGeneration,
		CoverageTarget:                  appConfig.CoverageTarget,
		LLMModel:                        appConfig.LLMModel,
		MaxTokens:                       appConfig.MaxTokens,
		Temperature:                     appConfig.Temperature,
		// Preserve existing internal settings if any
	}

	// Update configuration
	ta.config = newConfig

	// Re-initialize components with new config
	ta.initializeComponents()

	ta.logger.Info("TestingAgent configuration updated",
		"coverage_target", newConfig.CoverageTarget,
		"llm_model", newConfig.LLMModel,
		"max_tokens", newConfig.MaxTokens)

	return nil
}

func (ta *TestingAgent) Start() error {
	ta.mu.Lock()
	defer ta.mu.Unlock()
	ta.status = StatusIdle
	ta.logger.Info("Testing agent started")
	return nil
}

func (ta *TestingAgent) Stop() error {
	ta.mu.Lock()
	defer ta.mu.Unlock()
	ta.status = StatusStopped
	ta.logger.Info("Testing agent stopped")
	return nil
}

func (ta *TestingAgent) HealthCheck() *agents.HealthStatus {
	startTime := time.Now()
	status := &agents.HealthStatus{
		LastCheckTime:      startTime,
		DependenciesStatus: make(map[string]*agents.HealthStatus),
		Details:            make(map[string]interface{}),
	}

	// Check LLM provider
	if ta.llmProvider == nil {
		status.Status = agents.HealthStatusUnhealthy
		status.Message = "LLM provider not configured"
		status.Latency = time.Since(startTime)
		return status
	}

	// Check unit test generator
	if ta.unitTestGenerator == nil {
		status.Status = agents.HealthStatusUnhealthy
		status.Message = "Unit test generator not initialized"
		status.Latency = time.Since(startTime)
		return status
	}

	// Test basic functionality
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*3)
	defer cancel()

	testRequest := &llm.CompletionRequest{
		Prompt:    "test",
		Model:     ta.config.LLMModel,
		MaxTokens: 5,
	}

	_, err := ta.llmProvider.Complete(ctx, testRequest)
	status.Latency = time.Since(startTime)

	// Get metrics
	metrics := ta.GetMetrics()
	status.ErrorCount = metrics.ErrorCount

	// Evaluate health
	if err != nil {
		status.Status = agents.HealthStatusUnhealthy
		status.Message = fmt.Sprintf("LLM provider error: %v", err)
	} else if status.Latency > time.Second*2 {
		status.Status = agents.HealthStatusDegraded
		status.Message = "High response latency"
	} else {
		status.Status = agents.HealthStatusHealthy
		status.Message = "Testing agent operational"
	}

	status.Details["coverage_target"] = ta.config.CoverageTarget
	status.Details["llm_model"] = ta.config.LLMModel
	status.Details["mock_generation_enabled"] = ta.config.EnableMockGeneration
	return status
}

func (ta *TestingAgent) GetMetrics() *AgentMetrics {
	ta.metrics.mu.RLock()
	defer ta.metrics.mu.RUnlock()
	return &AgentMetrics{
		RequestsProcessed:   ta.metrics.TotalGenerations,
		AverageResponseTime: ta.metrics.AverageGenerationTime,
		SuccessRate:         0.88,
		LastRequestAt:       ta.metrics.LastGeneration,
	}
}

func (ta *TestingAgent) ResetMetrics() {
	ta.metrics.mu.Lock()
	defer ta.metrics.mu.Unlock()
	ta.metrics = &TestingAgentMetrics{
		GenerationsByType:     make(map[TestType]int64),
		GenerationsByLanguage: make(map[string]int64),
		FrameworkUsage:        make(map[string]int64),
	}
}

// Initialization methods
func (ta *TestingAgent) initializeCapabilities() {
	ta.capabilities = &AgentCapabilities{
		AgentType: AgentTypeTesting,
		SupportedIntents: []IntentType{
			IntentTestGeneration,
			IntentTestAnalysis,
		},
		SupportedLanguages: []string{
			"go", "python", "javascript", "typescript", "java", "csharp",
		},
		MaxContextSize:    5120,
		SupportsStreaming: false,
		SupportsAsync:     true,
		RequiresContext:   false,
		Capabilities: map[string]interface{}{
			"unit_tests":        ta.config.EnableUnitTests,
			"integration_tests": ta.config.EnableIntegrationTests,
			"e2e_tests":         ta.config.EnableE2ETests,
			"performance_tests": ta.config.EnablePerformanceTests,
			"test_optimization": ta.config.EnableTestOptimization,
			"coverage_analysis": ta.config.EnableCoverageAnalysis,
		},
	}
}

func (ta *TestingAgent) initializeComponents() {
	// Initialize test generators
	if ta.config.EnableUnitTests {
		ta.unitTestGenerator = NewUnitTestGenerator()
	}
	if ta.config.EnableIntegrationTests {
		ta.integrationTestGenerator = NewIntegrationTestGenerator()
	}

	if ta.config.EnableE2ETests {
		ta.e2eTestGenerator = NewE2ETestGenerator()
	}

	if ta.config.EnablePerformanceTests {
		ta.performanceTestGenerator = NewPerformanceTestGenerator()
	}

	// Initialize analysis components
	ta.codeAnalyzer = NewTestCodeAnalyzer()

	if ta.config.EnableCoverageAnalysis {
		ta.coverageAnalyzer = NewCoverageAnalyzer()
	}

	// Initialize other components
	ta.testStrategyEngine = NewTestStrategyEngine()
	ta.scenarioGenerator = NewTestScenarioGenerator()
	ta.dataGenerator = NewTestDataGenerator()
	ta.mockGenerator = NewMockGenerator()

	if ta.config.EnableTestAnalysis {
		ta.testQualityAnalyzer = NewTestQualityAnalyzer()
	}

	if ta.config.EnableTestSmellDetection {
		ta.testSmellDetector = NewTestSmellDetector()
	}

	if ta.config.EnableTestOptimization {
		ta.testOptimizer = NewTestOptimizer()
	}
}

// Utility methods (placeholders for actual implementations)
func (ta *TestingAgent) parseTestRequest(request *AgentRequest) (*TestGenerationRequest, error) {
	// Implementation would parse the request appropriately
	return &TestGenerationRequest{
		TestType: TestTypeUnit,
		Language: "go", // Would be determined from context
	}, nil
}

func (ta *TestingAgent) updateMetrics(testType TestType, language string, success bool, duration time.Duration, testsGenerated int) {
	ta.metrics.mu.Lock()
	defer ta.metrics.mu.Unlock()
	ta.metrics.TotalGenerations++
	ta.metrics.GenerationsByType[testType]++
	ta.metrics.GenerationsByLanguage[language]++
	ta.metrics.TestsGenerated += int64(testsGenerated)
	ta.metrics.LastGeneration = time.Now()

	if ta.metrics.AverageGenerationTime == 0 {
		ta.metrics.AverageGenerationTime = duration
	} else {
		ta.metrics.AverageGenerationTime = (ta.metrics.AverageGenerationTime + duration) / 2
	}
}

func (ta *TestingAgent) calculateConfidence(request *TestGenerationRequest, response *TestGenerationResponse) float64 {
	confidence := 0.8 // Base confidence
	if len(response.GeneratedTests) > 0 {
		confidence += 0.1
	}

	if response.Coverage != nil && response.Coverage.OverallCoverage >= ta.config.CoverageTarget {
		confidence += 0.1
	}

	return confidence
}
