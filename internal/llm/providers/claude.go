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

type ClaudeProvider struct {
	config     *ProviderConfig
	httpClient *http.Client
}

type claudeRequest struct {
	Model       string          `json:"model"`
	Messages    []claudeMessage `json:"messages"`
	MaxTokens   int             `json:"max_tokens"`
	Temperature float64         `json:"temperature,omitempty"`
	TopP        float64         `json:"top_p,omitempty"`
	System      string          `json:"system,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
}

type claudeMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type claudeResponse struct {
	ID           string          `json:"id"`
	Type         string          `json:"type"`
	Role         string          `json:"role"`
	Content      []claudeContent `json:"content"`
	Model        string          `json:"model"`
	StopReason   string          `json:"stop_reason"`
	StopSequence string          `json:"stop_sequence"`
	Usage        claudeUsage     `json:"usage"`
	Error        *claudeError    `json:"error,omitempty"`
}

type claudeContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type claudeUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type claudeError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

func NewClaudeProvider(config *ProviderConfig) *ClaudeProvider {
	return &ClaudeProvider{
		config: config,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
	}
}

func (p *ClaudeProvider) GetName() string {
	return "claude"
}

func (p *ClaudeProvider) GetModel() string {
	return p.config.Model
}

func (p *ClaudeProvider) IsEnabled() bool {
	return p.config.Enabled && p.config.APIKey != ""
}

func (p *ClaudeProvider) GenerateText(ctx context.Context, req *TextGenerationRequest) (*TextGenerationResponse, error) {
	start := time.Now()

	messages := p.buildMessages(req)

	claudeReq := claudeRequest{
		Model:       p.config.Model,
		Messages:    messages,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		System:      req.System,
		Stream:      false,
	}

	resp, err := p.makeRequest(ctx, claudeReq)
	if err != nil {
		return nil, err
	}

	if len(resp.Content) == 0 {
		return nil, &ProviderError{
			Provider: p.GetName(),
			Code:     "NO_CONTENT",
			Message:  "No content returned",
		}
	}

	content := resp.Content[0].Text
	cost := p.CalculateCost(resp.Usage.InputTokens, resp.Usage.OutputTokens)

	return &TextGenerationResponse{
		Content:      content,
		FinishReason: resp.StopReason,
		TokensUsed: TokenUsage{
			InputTokens:  resp.Usage.InputTokens,
			OutputTokens: resp.Usage.OutputTokens,
			TotalTokens:  resp.Usage.InputTokens + resp.Usage.OutputTokens,
			Cost:         cost,
		},
		Model:    resp.Model,
		Provider: p.GetName(),
		Duration: time.Since(start),
	}, nil
}

func (p *ClaudeProvider) GenerateCode(ctx context.Context, req *CodeGenerationRequest) (*CodeGenerationResponse, error) {
	prompt := p.buildCodeGenerationPrompt(req)

	textReq := &TextGenerationRequest{
		Prompt:      prompt,
		System:      "You are Claude, an AI assistant created by Anthropic. You're an expert programmer who writes clean, efficient, well-documented code following best practices.",
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

func (p *ClaudeProvider) ExplainCode(ctx context.Context, req *CodeExplanationRequest) (*CodeExplanationResponse, error) {
	prompt := p.buildCodeExplanationPrompt(req)

	textReq := &TextGenerationRequest{
		Prompt:      prompt,
		System:      "You are Claude, an AI assistant that excels at explaining code clearly and comprehensively. Break down complex code into understandable parts.",
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

func (p *ClaudeProvider) StreamText(ctx context.Context, req *TextGenerationRequest) (<-chan StreamChunk, error) {
	messages := p.buildMessages(req)

	claudeReq := claudeRequest{
		Model:       p.config.Model,
		Messages:    messages,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		System:      req.System,
		Stream:      true,
	}

	return p.makeStreamRequest(ctx, claudeReq)
}

func (p *ClaudeProvider) GetTokenCount(text string) (int, error) {
	// Claude uses a similar token counting approach as OpenAI
	return len(text) / 4, nil
}

func (p *ClaudeProvider) CalculateCost(inputTokens, outputTokens int) float64 {
	inputCost := float64(inputTokens) / 1000.0 * p.config.CostPer1KInput
	outputCost := float64(outputTokens) / 1000.0 * p.config.CostPer1KOutput
	return inputCost + outputCost
}

func (p *ClaudeProvider) HealthCheck(ctx context.Context) error {
	testReq := &TextGenerationRequest{
		Prompt:    "Hello",
		MaxTokens: 5,
	}

	_, err := p.GenerateText(ctx, testReq)
	return err
}

func (p *ClaudeProvider) GetCapabilities() ProviderCapabilities {
	return ProviderCapabilities{
		TextGeneration:     true,
		CodeGeneration:     true,
		CodeExplanation:    true,
		Streaming:          true,
		MaxTokens:          p.config.MaxTokens,
		SupportedLanguages: []string{"go", "python", "javascript", "typescript", "java", "rust", "c", "cpp", "swift", "kotlin"},
		Features:           []string{"chat", "reasoning", "analysis"},
	}
}

func (p *ClaudeProvider) buildMessages(req *TextGenerationRequest) []claudeMessage {
	var messages []claudeMessage

	for _, msg := range req.Context {
		if msg.Role != "system" { // System messages handled separately
			messages = append(messages, claudeMessage{
				Role:    msg.Role,
				Content: msg.Content,
			})
		}
	}

	messages = append(messages, claudeMessage{
		Role:    "user",
		Content: req.Prompt,
	})

	return messages
}

func (p *ClaudeProvider) buildCodeGenerationPrompt(req *CodeGenerationRequest) string {
	var prompt strings.Builder

	prompt.WriteString(fmt.Sprintf("Please generate %s code for: %s\n\n", req.Language, req.Description))

	if req.Context != "" {
		prompt.WriteString(fmt.Sprintf("Context: %s\n\n", req.Context))
	}

	prompt.WriteString("Requirements:\n")
	prompt.WriteString("- Write clean, readable, and maintainable code\n")
	prompt.WriteString("- Follow language best practices and conventions\n")
	prompt.WriteString("- Include appropriate error handling\n")
	prompt.WriteString("- Add clear comments explaining complex logic\n")

	if req.Style != "" {
		prompt.WriteString(fmt.Sprintf("- Follow this coding style: %s\n", req.Style))
	}

	if req.Tests {
		prompt.WriteString("- Include comprehensive unit tests\n")
	}

	if req.Docs {
		prompt.WriteString("- Include documentation with usage examples\n")
	}

	prompt.WriteString("\nPlease structure your response with:\n")
	prompt.WriteString("1. Main implementation\n")
	if req.Tests {
		prompt.WriteString("2. Unit tests\n")
	}
	if req.Docs {
		prompt.WriteString("3. Documentation\n")
	}
	prompt.WriteString("4. Brief explanation of the approach\n")

	return prompt.String()
}

func (p *ClaudeProvider) buildCodeExplanationPrompt(req *CodeExplanationRequest) string {
	var prompt strings.Builder

	prompt.WriteString("Please analyze and explain this code:\n\n")
	prompt.WriteString("```" + req.Language + "\n")
	prompt.WriteString(req.Code)
	prompt.WriteString("\n```\n\n")

	if req.Context != "" {
		prompt.WriteString(fmt.Sprintf("Context: %s\n\n", req.Context))
	}

	detail := req.Detail
	if detail == "" {
		detail = "normal"
	}

	prompt.WriteString("Please provide:\n")
	prompt.WriteString("1. High-level overview of what this code does\n")
	prompt.WriteString("2. Step-by-step breakdown of the logic\n")
	prompt.WriteString("3. Key concepts and patterns used\n")
	prompt.WriteString("4. Complexity assessment\n")

	if req.Focus != "" {
		prompt.WriteString(fmt.Sprintf("5. Special focus on: %s\n", req.Focus))
	}

	switch detail {
	case "brief":
		prompt.WriteString("\nKeep the explanation concise but comprehensive.\n")
	case "detailed":
		prompt.WriteString("\nProvide a thorough, detailed explanation suitable for learning.\n")
	default:
		prompt.WriteString("\nProvide a balanced explanation that's informative but not overwhelming.\n")
	}

	return prompt.String()
}

func (p *ClaudeProvider) makeRequest(ctx context.Context, req claudeRequest) (*claudeResponse, error) {
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, &ProviderError{
			Provider: p.GetName(),
			Code:     "MARSHAL_ERROR",
			Message:  "Failed to marshal request",
			Err:      err,
		}
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", "https://api.anthropic.com/v1/messages", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, &ProviderError{
			Provider: p.GetName(),
			Code:     "REQUEST_ERROR",
			Message:  "Failed to create request",
			Err:      err,
		}
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", p.config.APIKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

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

	var claudeResp claudeResponse
	if err := json.Unmarshal(body, &claudeResp); err != nil {
		return nil, &ProviderError{
			Provider: p.GetName(),
			Code:     "UNMARSHAL_ERROR",
			Message:  "Failed to unmarshal response",
			Err:      err,
		}
	}

	if claudeResp.Error != nil {
		return nil, &ProviderError{
			Provider: p.GetName(),
			Code:     claudeResp.Error.Type,
			Message:  claudeResp.Error.Message,
		}
	}

	return &claudeResp, nil
}

func (p *ClaudeProvider) makeStreamRequest(ctx context.Context, req claudeRequest) (<-chan StreamChunk, error) {
	ch := make(chan StreamChunk, 100)

	go func() {
		defer close(ch)

		// Implementation similar to OpenAI but with Claude's streaming format
		// This would handle Claude's specific SSE format
		ch <- StreamChunk{
			Content: "Streaming not fully implemented for Claude",
			Error:   fmt.Errorf("streaming not yet implemented"),
		}
	}()

	return ch, nil
}

// Parsing methods similar to OpenAI but adapted for Claude's response format
func (p *ClaudeProvider) parseCodeResponse(content string, req *CodeGenerationRequest) (code, tests, docs, explanation string) {
	// Similar implementation to OpenAI, but adapted for Claude's typical response structure
	return p.parseCodeContent(content, req.Language, req.Tests, req.Docs)
}

func (p *ClaudeProvider) parseExplanationResponse(content string) (overview string, steps []string, concepts []KeyConcept, complexity string, suggestions []string) {
	// Similar implementation to OpenAI, adapted for Claude's explanation style
	return p.parseExplanationContent(content)
}

func (p *ClaudeProvider) parseCodeContent(content, language string, includeTests, includeDocs bool) (code, tests, docs, explanation string) {
	sections := strings.Split(content, "\n")
	var currentSection string
	var codeBuilder, testBuilder, docsBuilder, explanationBuilder strings.Builder

	inCodeBlock := false

	for _, line := range sections {
		// Detect code blocks
		if strings.HasPrefix(line, "```") {
			inCodeBlock = !inCodeBlock
			if strings.Contains(line, language) {
				currentSection = "code"
			}
			continue
		}

		// Detect section headers
		lowerLine := strings.ToLower(line)
		switch {
		case strings.Contains(lowerLine, "test") && strings.Contains(lowerLine, "#"):
			currentSection = "tests"
			continue
		case strings.Contains(lowerLine, "documentation") && strings.Contains(lowerLine, "#"):
			currentSection = "docs"
			continue
		case strings.Contains(lowerLine, "explanation") && strings.Contains(lowerLine, "#"):
			currentSection = "explanation"
			continue
		}

		// Assign content to appropriate section
		switch currentSection {
		case "code":
			if inCodeBlock {
				codeBuilder.WriteString(line + "\n")
			}
		case "tests":
			testBuilder.WriteString(line + "\n")
		case "docs":
			docsBuilder.WriteString(line + "\n")
		case "explanation":
			explanationBuilder.WriteString(line + "\n")
		default:
			if !inCodeBlock {
				explanationBuilder.WriteString(line + "\n")
			}
		}
	}

	return strings.TrimSpace(codeBuilder.String()),
		strings.TrimSpace(testBuilder.String()),
		strings.TrimSpace(docsBuilder.String()),
		strings.TrimSpace(explanationBuilder.String())
}

func (p *ClaudeProvider) parseExplanationContent(content string) (overview string, steps []string, concepts []KeyConcept, complexity string, suggestions []string) {
	sections := strings.Split(content, "\n")
	var currentSection string
	var overviewBuilder, complexityBuilder strings.Builder

	for _, line := range sections {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		lowerLine := strings.ToLower(line)
		switch {
		case strings.Contains(lowerLine, "overview") && strings.Contains(line, "#"):
			currentSection = "overview"
			continue
		case strings.Contains(lowerLine, "step") && strings.Contains(line, "#"):
			currentSection = "steps"
			continue
		case strings.Contains(lowerLine, "concept") && strings.Contains(line, "#"):
			currentSection = "concepts"
			continue
		case strings.Contains(lowerLine, "complexity") && strings.Contains(line, "#"):
			currentSection = "complexity"
			continue
		case strings.Contains(lowerLine, "suggestion") && strings.Contains(line, "#"):
			currentSection = "suggestions"
			continue
		}

		switch currentSection {
		case "overview":
			if !strings.Contains(line, "#") {
				overviewBuilder.WriteString(line + " ")
			}
		case "steps":
			if strings.HasPrefix(line, "-") || strings.HasPrefix(line, "*") || strings.Contains(line, ".") {
				steps = append(steps, strings.TrimPrefix(strings.TrimPrefix(line, "-"), "*"))
			}
		case "concepts":
			if strings.Contains(line, ":") && !strings.Contains(line, "#") {
				parts := strings.SplitN(line, ":", 2)
				if len(parts) == 2 {
					concepts = append(concepts, KeyConcept{
						Name:        strings.TrimSpace(parts[0]),
						Description: strings.TrimSpace(parts[1]),
						Importance:  "medium",
					})
				}
			}
		case "complexity":
			if !strings.Contains(line, "#") {
				complexityBuilder.WriteString(line + " ")
			}
		case "suggestions":
			if strings.HasPrefix(line, "-") || strings.HasPrefix(line, "*") {
				suggestions = append(suggestions, strings.TrimPrefix(strings.TrimPrefix(line, "-"), "*"))
			}
		default:
			if overviewBuilder.Len() == 0 && !strings.Contains(line, "#") {
				overviewBuilder.WriteString(line + " ")
			}
		}
	}

	return strings.TrimSpace(overviewBuilder.String()),
		steps,
		concepts,
		strings.TrimSpace(complexityBuilder.String()),
		suggestions
}
