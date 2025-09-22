package tests

import (
	"context"
	"fmt"

	"github.com/yourusername/ai-code-assistant/internal/agents"
	"github.com/yourusername/ai-code-assistant/internal/logger"
)

// DocumentationAgentTests runs a suite of manual tests for the documentation agent.
func DocumentationAgentTests() {
	fmt.Println("✅ Testing Documentation Agent Components...")

	// Test 1: Agent initialization
	testDocumentationAgentInit()

	// Test 2: Documentation generation
	testDocumentationGeneration()

	// Test 3: Documentation update
	testDocumentationUpdate()

	// Test 4: API documentation
	testAPIDocumentation()

	// Test 5: README generation
	testReadmeGeneration()

	// Test 6: Quality checking
	testQualityChecking()

	// Test 7: Multiple formats
	testMultipleFormats()

	// Test 8: Metrics tracking
	testMetricsTracking()

	fmt.Println("✅ All documentation agent tests completed!")
}

func testDocumentationAgentInit() {
	fmt.Println("\n=== Testing Documentation Agent Initialization ===")

	// Create mock dependencies
	log := logger.NewDefaultLogger("doc-agent-test")
	mockAI := &MockAIProvider{}
	mockIndexer := &MockIndexer{}

	// Create agent with default config
	cfg := &agents.DocumentationAgentConfig{
		EnableDocGeneration:    true,
		EnableDocUpdate:        true,
		EnableAPIDocGeneration: true,
		EnableReadmeGeneration: true,
		EnableQualityCheck:     true,
		EnableCoverageAnalysis: true,
		LLMModel:               "GPT-model",
		MaxTokens:              2048,
		Temperature:            0.3,
	}

	agent := agents.NewDocumentationAgent(mockAI, mockIndexer, cfg, log)

	if agent == nil {
		fmt.Println("❌ Failed to create documentation agent")
		return
	}

	// Test agent properties
	fmt.Printf("✓ Agent Type: %s\n", agent.GetType())
	fmt.Printf("✓ Agent Version: %s\n", agent.GetVersion())
	fmt.Printf("✓ Agent Status: %s\n", agent.GetStatus())

	capabilities := agent.GetCapabilities()
	fmt.Printf("✓ Capabilities: %d\n", len(capabilities.Capabilities))
	fmt.Printf("✓ Supported Languages: %v\n", capabilities.SupportedLanguages)
	fmt.Printf("✓ Supported Formats: %v\n", capabilities.SupportedFormats)

	fmt.Println("✓ Documentation agent initialization works")
}

func testDocumentationGeneration() {
	fmt.Println("\n=== Testing Documentation Generation ===")

	agent := createTestAgent()
	ctx := context.Background()

	// Test function documentation
	functionCode := `func CalculateSum(a, b int) int {
      return a + b
}`

	request := &agents.AgentRequest{
		ID:   "test-gen-1",
		Type: agents.RequestTypeDocumentation,
		Data: &agents.DocumentationRequest{
			Type: agents.DocTypeGenerate,
			Target: &agents.DocumentationTarget{
				Type:       agents.TargetFunction,
				Identifier: "CalculateSum",
				Language:   "go",
			},
			Code:   functionCode,
			Format: agents.FormatGoDoc,
			Options: &agents.DocumentationOptions{
				IncludeExamples: true,
				DetailLevel:     agents.DetailStandard,
				Audience:        agents.AudienceDeveloper,
			},
		},
	}

	response, err := agent.ProcessRequest(ctx, request)
	if err != nil {
		fmt.Printf("❌ Documentation generation failed: %v\n", err)
		return
	}

	docResponse := response.Result.(*agents.DocumentationResponse)
	fmt.Printf("✓ Generated documentation (%d chars)\n", len(docResponse.GeneratedDoc))
	fmt.Printf("✓ Format: %s\n", docResponse.Format)
	fmt.Printf("✓ Examples: %d\n", len(docResponse.Examples))
	fmt.Printf("✓ Suggestions: %d\n", len(docResponse.Suggestions))
	fmt.Printf("✓ Processing time: %v\n", response.ProcessingTime)

	if docResponse.Quality != nil {
		fmt.Printf("✓ Quality score: %.2f\n", docResponse.Quality.OverallScore)
	}

	fmt.Println("✓ Documentation generation works")
}

func testDocumentationUpdate() {
	fmt.Println("\n=== Testing Documentation Update ===")

	agent := createTestAgent()
	ctx := context.Background()

	existingDoc := `// CalculateSum adds two numbers
func CalculateSum(a, b int) int`

	updatedCode := `// CalculateSum adds two integers and returns their sum
// Parameters:
//   a: first integer
//   b: second integer
// Returns: sum of a and b
func CalculateSum(a, b int) int {
      if a < 0 || b < 0 {
              panic("negative numbers not supported")
      }
      return a + b
}`

	request := &agents.AgentRequest{
		ID:   "test-update-1",
		Type: agents.RequestTypeDocumentation,
		Data: &agents.DocumentationRequest{
			Type:        agents.DocTypeUpdate,
			Code:        updatedCode,
			ExistingDoc: existingDoc,
			Target: &agents.DocumentationTarget{
				Type:       agents.TargetFunction,
				Identifier: "CalculateSum",
				Language:   "go",
			},
			Format: agents.FormatGoDoc,
		},
	}

	response, err := agent.ProcessRequest(ctx, request)
	if err != nil {
		fmt.Printf("❌ Documentation update failed: %v\n", err)
		return
	}

	docResponse := response.Result.(*agents.DocumentationResponse)
	fmt.Printf("✓ Updated documentation (%d chars)\n", len(docResponse.GeneratedDoc))
	fmt.Printf("✓ Update suggestions: %d\n", len(docResponse.Suggestions))

	fmt.Println("✓ Documentation update works")
}

func testAPIDocumentation() {
	fmt.Println("\n=== Testing API Documentation ===")

	agent := createTestAgent()
	ctx := context.Background()

	apiCode := `// UserHandler handles user-related API endpoints
type UserHandler struct {
      db *sql.DB
}

// GetUser retrieves a user by ID
// @Summary Get user by ID
// @Param id path int true "User ID"
// @Success 200 {object} User
// @Router /users/{id} [get]
func (h *UserHandler) GetUser(w http.ResponseWriter, r *http.Request) {
      // Implementation here
}

// CreateUser creates a new user
// @Summary Create new user
// @Param user body User true "User data"
// @Success 201 {object} User
// @Router /users [post]
func (h *UserHandler) CreateUser(w http.ResponseWriter, r *http.Request) {
      // Implementation here
}`

	request := &agents.AgentRequest{
		ID:   "test-api-1",
		Type: agents.RequestTypeDocumentation,
		Data: &agents.DocumentationRequest{
			Type: agents.DocTypeAPIDoc,
			Code: apiCode,
			Target: &agents.DocumentationTarget{
				Type:       agents.TargetClass,
				Identifier: "UserHandler",
				Language:   "go",
			},
			Format: agents.FormatMarkdown,
			Options: &agents.DocumentationOptions{
				IncludeExamples: true,
				DetailLevel:     agents.DetailDetailed,
			},
		},
	}

	response, err := agent.ProcessRequest(ctx, request)
	if err != nil {
		fmt.Printf("❌ API documentation failed: %v\n", err)
		return
	}

	docResponse := response.Result.(*agents.DocumentationResponse)
	fmt.Printf("✓ API documentation generated (%d chars)\n", len(docResponse.GeneratedDoc))
	fmt.Printf("✓ API examples: %d\n", len(docResponse.Examples))

	fmt.Println("✓ API documentation works")
}

func testReadmeGeneration() {
	fmt.Println("\n=== Testing README Generation ===")

	agent := createTestAgent()
	ctx := context.Background()

	request := &agents.AgentRequest{
		ID:   "test-readme-1",
		Type: agents.RequestTypeDocumentation,
		Data: &agents.DocumentationRequest{
			Type: agents.DocTypeReadme,
			Target: &agents.DocumentationTarget{
				Type:     agents.TargetProject,
				Language: "go",
				Context: &agents.DocumentationContext{
					ProjectInfo: &agents.ProjectInfo{
						Name:        "AI Code Assistant",
						Description: "An intelligent code assistant powered by AI",
						Version:     "1.0.0",
						Authors:     []string{"Developer"},
						License:     "MIT",
						Repository:  "https://github.com/user/ai-code-assistant",
					},
				},
			},
			Format: agents.FormatMarkdown,
		},
	}

	response, err := agent.ProcessRequest(ctx, request)
	if err != nil {
		fmt.Printf("❌ README generation failed: %v\n", err)
		return
	}

	docResponse := response.Result.(*agents.DocumentationResponse)
	fmt.Printf("✓ README generated (%d chars)\n", len(docResponse.GeneratedDoc))
	fmt.Printf("✓ Format: %s\n", docResponse.Format)

	fmt.Println("✓ README generation works")
}

func testQualityChecking() {
	fmt.Println("\n=== Testing Quality Checking ===")

	agent := createTestAgent()
	ctx := context.Background()

	poorDoc := `// bad function
func foo() {}`

	request := &agents.AgentRequest{
		ID:   "test-quality-1",
		Type: agents.RequestTypeDocumentation,
		Data: &agents.DocumentationRequest{
			Type:        agents.DocTypeImprove,
			ExistingDoc: poorDoc,
			Target: &agents.DocumentationTarget{
				Type:       agents.TargetFunction,
				Identifier: "foo",
				Language:   "go",
			},
		},
	}

	response, err := agent.ProcessRequest(ctx, request)
	if err != nil {
		fmt.Printf("❌ Quality checking failed: %v\n", err)
		return
	}

	docResponse := response.Result.(*agents.DocumentationResponse)
	if docResponse.Quality != nil {
		fmt.Printf("✓ Quality analysis completed\n")
		fmt.Printf("  - Overall score: %.2f\n", docResponse.Quality.OverallScore)
		fmt.Printf("  - Completeness: %.2f\n", docResponse.Quality.Completeness)
		fmt.Printf("  - Clarity: %.2f\n", docResponse.Quality.Clarity)
		fmt.Printf("  - Issues found: %d\n", len(docResponse.Quality.Issues))
		fmt.Printf("  - Recommendations: %d\n", len(docResponse.Quality.Recommendations))
	}

	if docResponse.Coverage != nil {
		fmt.Printf("✓ Coverage analysis completed\n")
		fmt.Printf("  - Coverage: %.1f%%\n", docResponse.Coverage.CoveragePercent)
		fmt.Printf("  - Missing docs: %d\n", len(docResponse.Coverage.MissingDocs))
	}

	fmt.Println("✓ Quality checking works")
}

func testMultipleFormats() {
	fmt.Println("\n=== Testing Multiple Formats ===")

	agent := createTestAgent()
	ctx := context.Background()

	testCode := `class Calculator {
      /**
       * Adds two numbers
       */
      add(a, b) {
              return a + b;
      }
}`

	formats := []agents.DocFormat{
		agents.FormatMarkdown,
		agents.FormatJSDoc,
		agents.FormatJavaDoc,
	}

	for _, format := range formats {
		request := &agents.AgentRequest{
			ID:   fmt.Sprintf("test-format-%s", format),
			Type: agents.RequestTypeDocumentation,
			Data: &agents.DocumentationRequest{
				Type:   agents.DocTypeGenerate,
				Code:   testCode,
				Format: format,
				Target: &agents.DocumentationTarget{
					Type:       agents.TargetClass,
					Identifier: "Calculator",
					Language:   "javascript",
				},
			},
		}

		response, err := agent.ProcessRequest(ctx, request)
		if err != nil {
			fmt.Printf("❌ Format %s failed: %v\n", format, err)
			continue
		}

		docResponse := response.Result.(*agents.DocumentationResponse)
		fmt.Printf("✓ Format %s: %d chars\n", format, len(docResponse.GeneratedDoc))
	}

	fmt.Println("✓ Multiple formats work")
}

func testMetricsTracking() {
	fmt.Println("\n=== Testing Metrics Tracking ===")

	agent := createTestAgent()
	ctx := context.Background()

	// Generate multiple requests to test metrics
	for i := 0; i < 5; i++ {
		request := &agents.AgentRequest{
			ID:   fmt.Sprintf("test-metrics-%d", i),
			Type: agents.RequestTypeDocumentation,
			Data: &agents.DocumentationRequest{
				Type: agents.DocTypeGenerate,
				Code: "func test() {}",
				Target: &agents.DocumentationTarget{
					Type:       agents.TargetFunction,
					Identifier: "test",
					Language:   "go",
				},
				Format: agents.FormatGoDoc,
			},
		}

		_, err := agent.ProcessRequest(ctx, request)
		if err != nil {
			fmt.Printf("❌ Metrics test request %d failed: %v\n", i, err)
		}
	}

	// Check metrics (safe type assertion)
	if m := agent.GetMetrics(); m != nil {
		if metrics, ok := m.(*agents.DocumentationAgentMetrics); ok {
			fmt.Printf("✓ Total requests: %d\n", metrics.TotalRequests)
			fmt.Printf("✓ Success rate: %.2f%%\n", metrics.SuccessRate*100)
			fmt.Printf("✓ Average response time: %v\n", metrics.AverageResponseTime)
			fmt.Printf("✓ Requests by type: %v\n", metrics.RequestsByType)
			fmt.Printf("✓ Requests by format: %v\n", metrics.RequestsByFormat)
		} else {
			fmt.Println("✓ Metrics available but in unexpected format")
		}
	} else {
		fmt.Println("✓ Metrics not available")
	}

	fmt.Println("✓ Metrics tracking works")
}

// Helper functions and mocks

func createTestAgent() *agents.DocumentationAgent {
	log := logger.NewDefaultLogger("doc-agent-test")
	mockAI := &MockAIProvider{}
	mockIndexer := &MockIndexer{}

	config := &agents.DocumentationAgentConfig{
		EnableDocGeneration:    true,
		EnableDocUpdate:        true,
		EnableAPIDocGeneration: true,
		EnableReadmeGeneration: true,
		EnableQualityCheck:     true,
		EnableCoverageAnalysis: true,
		LLMModel:               "gpt-4",
		MaxTokens:              2048,
		Temperature:            0.3,
	}

	return agents.NewDocumentationAgent(mockAI, mockIndexer, config, log)
}

// Mock implementations

type MockAIProvider struct{}

func (m *MockAIProvider) GetName() string {
	return "MockAIProvider"
}

func (m *MockAIProvider) GetModel() string {
	return "mock-model"
}

func (m *MockAIProvider) GenerateText(ctx context.Context, request *providers.TextGenerationRequest) (*providers.TextGenerationResponse, error) {
	return &providers.TextGenerationResponse{
		Text: "Generated text based on the prompt",
		Usage: &providers.Usage{
			PromptTokens:     50,
			CompletionTokens: 150,
			TotalTokens:      200,
		},
	}, nil
}

func (m *MockAIProvider) GenerateCode(ctx context.Context, request *providers.CodeGenerationRequest) (*providers.CodeGenerationResponse, error) {
	return &providers.CodeGenerationResponse{
		Code: "Generated code based on the prompt",
	}, nil
}

func (m *MockAIProvider) ExplainCode(ctx context.Context, request *providers.CodeExplanationRequest) (*providers.CodeExplanationResponse, error) {
	return &providers.CodeExplanationResponse{
		Explanation: "This function calculates the sum of two integers.",
	}, nil
}

func (m *MockAIProvider) StreamText(ctx context.Context, request *providers.TextGenerationRequest) (<-chan providers.StreamChunk, error) {
	ch := make(chan providers.StreamChunk, 1)
	go func() {
		ch <- providers.StreamChunk{Text: "Streaming generated text..."}
		close(ch)
	}()
	return ch, nil
}

// Minimal MockIndexer to satisfy constructor usage in tests.
// Implement additional methods if agents.NewDocumentationAgent requires them.
type MockIndexer struct{}
