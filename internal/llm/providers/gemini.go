package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type GeminiProvider struct {
	config     *ProviderConfig
	httpClient *http.Client
}

type geminiRequest struct {
	Contents         []geminiContent  `json:"contents"`
	GenerationConfig *geminiGenConfig `json:"generationConfig,omitempty"`
	SafetySettings   []geminySafety   `json:"safetySettings,omitempty"`
}

type geminiContent struct {
	Parts []geminiPart `json:"parts"`
	Role  string       `json:"role,omitempty"`
}

type geminiPart struct {
	Text string `json:"text"`
}

type geminiGenConfig struct {
	Temperature     float64  `json:"temperature,omitempty"`
	TopK            int      `json:"topK,omitempty"`
	TopP            float64  `json:"topP,omitempty"`
	MaxOutputTokens int      `json:"maxOutputTokens,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
}

type geminySafety struct {
	Category  string `json:"category"`
	Threshold string `json:"threshold"`
}

type geminiResponse struct {
	Candidates    []geminiCandidate `json:"candidates"`
	UsageMetadata geminiUsage       `json:"usageMetadata"`
	Error         *geminiError      `json:"error,omitempty"`
}

type geminiCandidate struct {
	Content       geminiContent        `json:"content"`
	FinishReason  string               `json:"finishReason"`
	Index         int                  `json:"index"`
	SafetyRatings []geminiSafetyRating `json:"safetyRatings"`
}

type geminiSafetyRating struct {
	Category    string `json:"category"`
	Probability string `json:"probability"`
}

type geminiUsage struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

type geminiError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
}

func NewGeminiProvider(config *ProviderConfig) *GeminiProvider {
	return &GeminiProvider{
		config: config,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
	}
}

func (p *GeminiProvider) GetName() string {
	return "gemini"
}

func (p *GeminiProvider) GetModel() string {
	return p.config.Model
}

func (p *GeminiProvider) IsEnabled() bool {
	return p.config.Enabled && p.config.APIKey != ""
}

func (p *GeminiProvider) GenerateText(ctx context.Context, req *TextGenerationRequest) (*TextGenerationResponse, error) {
	start := time.Now()

	geminiReq := geminiRequest{
		Contents: p.buildContents(req),
		GenerationConfig: &geminiGenConfig{
			Temperature:     req.Temperature,
			TopP:            req.TopP,
			MaxOutputTokens: req.MaxTokens,
		},
		SafetySettings: p.getDefaultSafetySettings(),
	}

	resp, err := p.makeRequest(ctx, geminiReq)
	if err != nil {
		return nil, err
	}

	if len(resp.Candidates) == 0 {
		return nil, &ProviderError{
			Provider: p.GetName(),
			Code:     "NO_CANDIDATES",
			Message:  "No response candidates returned",
		}
	}

	candidate := resp.Candidates[0]
	if len(candidate.Content.Parts) == 0 {
		return nil, &ProviderError{
			Provider: p.GetName(),
			Code:     "NO_CONTENT",
			Message:  "No content in response",
		}
	}

	content := candidate.Content.Parts[0].Text
	cost := p.CalculateCost(resp.UsageMetadata.PromptTokenCount, resp.UsageMetadata.CandidatesTokenCount)

	return &TextGenerationResponse{
		Content:      content,
		FinishReason: candidate.FinishReason,
		TokensUsed: TokenUsage{
			InputTokens:  resp.UsageMetadata.PromptTokenCount,
			OutputTokens: resp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:  resp.UsageMetadata.TotalTokenCount,
			Cost:         cost,
		},
		Model:    p.config.Model,
		Provider: p.GetName(),
		Duration: time.Since(start),
	}, nil
}

func (p *GeminiProvider) GenerateCode(ctx context.Context, req *CodeGenerationRequest) (*CodeGenerationResponse, error) {
	prompt := p.buildCodeGenerationPrompt(req)

	textReq := &TextGenerationRequest{
		Prompt:      prompt,
		System:      "You are an expert programmer who writes clean, efficient code with comprehensive documentation and tests.",
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
	}

	resp, err := p.GenerateText(ctx, textReq)
	if err != nil {
		return nil, err
	}

	code, tests, docs, explanation := p.parseCodeResponse(resp.Content, req)

	return &CodeGenerationResponse{
		Code:          code,
		Language:      req.Language,
		Explanation:   explanation,
		Tests:         tests,
		Documentation: docs,
		TokensUsed:    resp.TokensUsed,
		Model:         resp.Model,
		Provider:      resp.Provider,
		Duration:      resp.Duration,
	}, nil
}

func (p *GeminiProvider) ExplainCode(ctx context.Context, req *CodeExplanationRequest) (*CodeExplanationResponse, error) {
	prompt := p.buildCodeExplanationPrompt(req)

	textReq := &TextGenerationRequest{
		Prompt:      prompt,
		System:      "You are an expert at explaining code concepts clearly and thoroughly to help others understand programming.",
		MaxTokens:   req.MaxTokens,
		Temperature: 0.3,
	}

	resp, err := p.GenerateText(ctx, textReq)
	if err != nil {
		return nil, err
	}

	overview, steps, concepts, complexity, suggestions := p.parseExplanationResponse(resp.Content)

	return &CodeExplanationResponse{
		Overview:    overview,
		StepByStep:  steps,
		KeyConcepts: concepts,
		Complexity:  complexity,
		Suggestions: suggestions,
		TokensUsed:  resp.TokensUsed,
		Model:       resp.Model,
		Provider:    resp.Provider,
		Duration:    resp.Duration,
	}, nil
}

func (p *GeminiProvider) StreamText(ctx context.Context, req *TextGenerationRequest) (<-chan StreamChunk, error) {
	// Gemini streaming implementation would go here
	ch := make(chan StreamChunk, 1)
	go func() {
		defer close(ch)
		ch <- StreamChunk{
			Content: "Streaming not implemented for Gemini yet",
			Error:   fmt.Errorf("streaming not yet supported"),
		}
	}()
	return ch, nil
}

func (p *GeminiProvider) GetTokenCount(text string) (int, error) {
	// Gemini token counting
	return len(text) / 4, nil
}

func (p *GeminiProvider) CalculateCost(inputTokens, outputTokens int) float64 {
	inputCost := float64(inputTokens) / 1000.0 * p.config.CostPer1KInput
	outputCost := float64(outputTokens) / 1000.0 * p.config.CostPer1KOutput
	return inputCost + outputCost
}

func (p *GeminiProvider) HealthCheck(ctx context.Context) error {
	testReq := &TextGenerationRequest{
		Prompt:    "Hello",
		MaxTokens: 5,
	}

	_, err := p.GenerateText(ctx, testReq)
	return err
}

func (p *GeminiProvider) GetCapabilities() ProviderCapabilities {
	return ProviderCapabilities{
		TextGeneration:     true,
		CodeGeneration:     true,
		CodeExplanation:    true,
		Streaming:          false, // Not implemented yet
		MaxTokens:          p.config.MaxTokens,
		SupportedLanguages: []string{"go", "python", "javascript", "typescript", "java", "c", "cpp"},
		Features:           []string{"chat", "multimodal"},
	}
}

func (p *GeminiProvider) buildContents(req *TextGenerationRequest) []geminiContent {
	var contents []geminiContent

	// Add system message as user context if present
	if req.System != "" {
		contents = append(contents, geminiContent{
			Parts: []geminiPart{{Text: "System: " + req.System}},
			Role:  "user",
		})
	}

	// Add conversation context
	for _, msg := range req.Context {
		role := msg.Role
		if role == "assistant" {
			role = "model"
		}

		contents = append(contents, geminiContent{
			Parts: []geminiPart{{Text: msg.Content}},
			Role:  role,
		})
	}

	// Add current prompt
	contents = append(contents, geminiContent{
		Parts: []geminiPart{{Text: req.Prompt}},
		Role:  "user",
	})

	return contents
}

func (p *GeminiProvider) getDefaultSafetySettings() []geminySafety {
	return []geminySafety{
		{Category: "HARM_CATEGORY_HARASSMENT", Threshold: "BLOCK_MEDIUM_AND_ABOVE"},
		{Category: "HARM_CATEGORY_HATE_SPEECH", Threshold: "BLOCK_MEDIUM_AND_ABOVE"},
		{Category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", Threshold: "BLOCK_MEDIUM_AND_ABOVE"},
		{Category: "HARM_CATEGORY_DANGEROUS_CONTENT", Threshold: "BLOCK_MEDIUM_AND_ABOVE"},
	}
}

func (p *GeminiProvider) buildCodeGenerationPrompt(req *CodeGenerationRequest) string {
	var prompt strings.Builder

	prompt.WriteString(fmt.Sprintf("Generate %s code for: %s\n\n", req.Language, req.Description))

	if req.Context != "" {
		prompt.WriteString(fmt.Sprintf("Context: %s\n\n", req.Context))
	}

	prompt.WriteString("Requirements:\n")
	prompt.WriteString("- Write production-ready, well-structured code\n")
	prompt.WriteString("- Include comprehensive error handling\n")
	prompt.WriteString("- Add meaningful comments and documentation\n")
	prompt.WriteString("- Follow language-specific best practices\n")

	if req.Tests {
		prompt.WriteString("- Provide complete unit tests with good coverage\n")
	}

	if req.Docs {
		prompt.WriteString("- Include detailed documentation with examples\n")
	}

	return prompt.String()
}

func (p *GeminiProvider) buildCodeExplanationPrompt(req *CodeExplanationRequest) string {
	var prompt strings.Builder

	prompt.WriteString("Please provide a comprehensive explanation of this code:\n\n")
	prompt.WriteString("```" + req.Language + "\n")
	prompt.WriteString(req.Code)
	prompt.WriteString("\n```\n\n")

	prompt.WriteString("Include:\n")
	prompt.WriteString("1. What the code does (overview)\n")
	prompt.WriteString("2. How it works (step-by-step)\n")
	prompt.WriteString("3. Important concepts and patterns\n")
	prompt.WriteString("4. Complexity analysis\n")
	prompt.WriteString("5. Potential improvements\n")

	if req.Focus != "" {
		prompt.WriteString(fmt.Sprintf("6. Special attention to: %s\n", req.Focus))
	}

	return prompt.String()
}

func (p *GeminiProvider) makeRequest(ctx context.Context, req geminiRequest) (*geminiResponse, error) {
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, &ProviderError{
			Provider: p.GetName(),
			Code:     "MARSHAL_ERROR",
			Message:  "Failed to marshal request",
			Err:      err,
		}
	}

	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s", p.config.Model, p.config.APIKey)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, &ProviderError{
			Provider: p.GetName(),
			Code:     "REQUEST_ERROR",
			Message:  "Failed to create request",
			Err:      err,
		}
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, &ProviderError{
			Provider: p.GetName(),
			Code:     "HTTP_ERROR",
			Message:  "HTTP request failed",
			Err:      err,
		}
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, &ProviderError{
			Provider: p.GetName(),
			Code:     "READ_ERROR",
			Message:  "Failed to read response",
			Err:      err,
		}
	}

	var geminiResp geminiResponse
	if err := json.Unmarshal(body, &geminiResp); err != nil {
		return nil, &ProviderError{
			Provider: p.GetName(),
			Code:     "UNMARSHAL_ERROR",
			Message:  "Failed to unmarshal response",
			Err:      err,
		}
	}

	if geminiResp.Error != nil {
		return nil, &ProviderError{
			Provider: p.GetName(),
			Code:     fmt.Sprintf("GEMINI_%d", geminiResp.Error.Code),
			Message:  geminiResp.Error.Message,
		}
	}

	return &geminiResp, nil
}

// Parsing methods (similar structure to other providers)
func (p *GeminiProvider) parseCodeResponse(content string, req *CodeGenerationRequest) (code, tests, docs, explanation string) {
	// Similar implementation to other providers
	return parseGenericCodeResponse(content, req.Language, req.Tests, req.Docs)
}

func (p *GeminiProvider) parseExplanationResponse(content string) (overview string, steps []string, concepts []KeyConcept, complexity string, suggestions []string) {
	return parseGenericExplanationResponse(content)
}
