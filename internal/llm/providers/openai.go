package providers

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type OpenAIProvider struct {
	config     *ProviderConfig
	httpClient *http.Client
}

type openAIRequest struct {
	Model       string          `json:"model"`
	Messages    []openAIMessage `json:"messages"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Temperature float64         `json:"temperature,omitempty"`
	TopP        float64         `json:"top_p,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
	Stop        []string        `json:"stop,omitempty"`
}

type openAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []openAIChoice `json:"choices"`
	Usage   openAIUsage    `json:"usage"`
	Error   *openAIError   `json:"error,omitempty"`
}

type openAIChoice struct {
	Index        int            `json:"index"`
	Message      *openAIMessage `json:"message,omitempty"`
	Delta        *openAIMessage `json:"delta,omitempty"`
	FinishReason string         `json:"finish_reason"`
}

type openAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type openAIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

func NewOpenAIProvider(config *ProviderConfig) *OpenAIProvider {
	return &OpenAIProvider{
		config: config,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
	}
}

func (p *OpenAIProvider) GetName() string {
	return "openai"
}

func (p *OpenAIProvider) GetModel() string {
	return p.config.Model
}

func (p *OpenAIProvider) IsEnabled() bool {
	return p.config.Enabled && p.config.APIKey != ""
}

func (p *OpenAIProvider) GenerateText(ctx context.Context, req *TextGenerationRequest) (*TextGenerationResponse, error) {
	start := time.Now()

	messages := p.buildMessages(req)

	openAIReq := openAIRequest{
		Model:       p.config.Model,
		Messages:    messages,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      false,
	}

	resp, err := p.makeRequest(ctx, openAIReq)
	if err != nil {
		return nil, err
	}

	if len(resp.Choices) == 0 {
		return nil, &ProviderError{
			Provider: p.GetName(),
			Code:     "NO_CHOICES",
			Message:  "No response choices returned",
		}
	}

	choice := resp.Choices[0]
	cost := p.CalculateCost(resp.Usage.PromptTokens, resp.Usage.CompletionTokens)

	return &TextGenerationResponse{
		Content:      choice.Message.Content,
		FinishReason: choice.FinishReason,
		TokensUsed: TokenUsage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
			TotalTokens:  resp.Usage.TotalTokens,
			Cost:         cost,
		},
		Model:    resp.Model,
		Provider: p.GetName(),
		Duration: time.Since(start),
	}, nil
}

func (p *OpenAIProvider) GenerateCode(ctx context.Context, req *CodeGenerationRequest) (*CodeGenerationResponse, error) {
	prompt := p.buildCodeGenerationPrompt(req)

	textReq := &TextGenerationRequest{
		Prompt:      prompt,
		System:      "You are an expert programmer. Generate clean, well-documented code.",
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
	}

	resp, err := p.GenerateText(ctx, textReq)
	if err != nil {
		return nil, err
	}

	// Parse the response to extract code, tests, docs
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

func (p *OpenAIProvider) ExplainCode(ctx context.Context, req *CodeExplanationRequest) (*CodeExplanationResponse, error) {
	prompt := p.buildCodeExplanationPrompt(req)

	textReq := &TextGenerationRequest{
		Prompt:      prompt,
		System:      "You are an expert programmer who explains code clearly and comprehensively.",
		MaxTokens:   req.MaxTokens,
		Temperature: 0.3, // Lower temperature for explanations
	}

	resp, err := p.GenerateText(ctx, textReq)
	if err != nil {
		return nil, err
	}

	// Parse explanation response
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

func (p *OpenAIProvider) StreamText(ctx context.Context, req *TextGenerationRequest) (<-chan StreamChunk, error) {
	messages := p.buildMessages(req)

	openAIReq := openAIRequest{
		Model:       p.config.Model,
		Messages:    messages,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      true,
	}

	return p.makeStreamRequest(ctx, openAIReq)
}

func (p *OpenAIProvider) GetTokenCount(text string) (int, error) {
	// Simple approximation for OpenAI models
	// In production, use tiktoken or similar library
	return len(text) / 4, nil
}

func (p *OpenAIProvider) CalculateCost(inputTokens, outputTokens int) float64 {
	inputCost := float64(inputTokens) / 1000.0 * p.config.CostPer1KInput
	outputCost := float64(outputTokens) / 1000.0 * p.config.CostPer1KOutput
	return inputCost + outputCost
}

func (p *OpenAIProvider) HealthCheck(ctx context.Context) error {
	testReq := &TextGenerationRequest{
		Prompt:    "Hello",
		MaxTokens: 5,
	}

	_, err := p.GenerateText(ctx, testReq)
	return err
}

func (p *OpenAIProvider) GetCapabilities() ProviderCapabilities {
	return ProviderCapabilities{
		TextGeneration:     true,
		CodeGeneration:     true,
		CodeExplanation:    true,
		Streaming:          true,
		MaxTokens:          p.config.MaxTokens,
		SupportedLanguages: []string{"go", "python", "javascript", "typescript", "java", "rust", "c", "cpp"},
		Features:           []string{"chat", "completion", "function_calling"},
	}
}

func (p *OpenAIProvider) buildMessages(req *TextGenerationRequest) []openAIMessage {
	var messages []openAIMessage

	if req.System != "" {
		messages = append(messages, openAIMessage{
			Role:    "system",
			Content: req.System,
		})
	}

	// Add context messages
	for _, msg := range req.Context {
		messages = append(messages, openAIMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	// Add current prompt
	messages = append(messages, openAIMessage{
		Role:    "user",
		Content: req.Prompt,
	})

	return messages
}

func (p *OpenAIProvider) buildCodeGenerationPrompt(req *CodeGenerationRequest) string {
	var prompt strings.Builder

	prompt.WriteString(fmt.Sprintf("Generate %s code for: %s\n\n", req.Language, req.Description))

	if req.Context != "" {
		prompt.WriteString(fmt.Sprintf("Context: %s\n\n", req.Context))
	}

	if req.Style != "" {
		prompt.WriteString(fmt.Sprintf("Style requirements: %s\n\n", req.Style))
	}

	prompt.WriteString("Requirements:\n")
	prompt.WriteString("- Write clean, readable code with proper comments\n")
	prompt.WriteString("- Follow best practices for the language\n")
	prompt.WriteString("- Include error handling where appropriate\n")

	if req.Tests {
		prompt.WriteString("- Include unit tests\n")
	}

	if req.Docs {
		prompt.WriteString("- Include documentation\n")
	}

	prompt.WriteString("\nPlease provide:\n")
	prompt.WriteString("1. The main code\n")
	if req.Tests {
		prompt.WriteString("2. Unit tests\n")
	}
	if req.Docs {
		prompt.WriteString("3. Documentation\n")
	}
	prompt.WriteString("4. Brief explanation of the implementation\n")

	return prompt.String()
}

func (p *OpenAIProvider) buildCodeExplanationPrompt(req *CodeExplanationRequest) string {
	var prompt strings.Builder

	prompt.WriteString(fmt.Sprintf("Please explain this %s code:\n\n", req.Language))
	prompt.WriteString("```" + req.Language + "\n")
	prompt.WriteString(req.Code)
	prompt.WriteString("\n```\n\n")

	if req.Context != "" {
		prompt.WriteString(fmt.Sprintf("Additional context: %s\n\n", req.Context))
	}

	detail := req.Detail
	if detail == "" {
		detail = "normal"
	}

	switch detail {
	case "brief":
		prompt.WriteString("Please provide a brief explanation focusing on the main purpose and functionality.\n")
	case "detailed":
		prompt.WriteString("Please provide a detailed explanation including:\n")
		prompt.WriteString("- Step-by-step breakdown\n")
		prompt.WriteString("- Key concepts and patterns used\n")
		prompt.WriteString("- Performance considerations\n")
		prompt.WriteString("- Potential improvements\n")
	default:
		prompt.WriteString("Please provide a clear explanation including:\n")
		prompt.WriteString("- Overview of what the code does\n")
		prompt.WriteString("- Key concepts involved\n")
		prompt.WriteString("- How it works step by step\n")
	}

	if req.Focus != "" {
		prompt.WriteString(fmt.Sprintf("- Focus particularly on: %s\n", req.Focus))
	}

	return prompt.String()
}

func (p *OpenAIProvider) makeRequest(ctx context.Context, req openAIRequest) (*openAIResponse, error) {
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, &ProviderError{
			Provider: p.GetName(),
			Code:     "MARSHAL_ERROR",
			Message:  "Failed to marshal request",
			Err:      err,
		}
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, &ProviderError{
			Provider: p.GetName(),
			Code:     "REQUEST_ERROR",
			Message:  "Failed to create request",
			Err:      err,
		}
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.config.APIKey)

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

	var openAIResp openAIResponse
	if err := json.Unmarshal(body, &openAIResp); err != nil {
		return nil, &ProviderError{
			Provider: p.GetName(),
			Code:     "UNMARSHAL_ERROR",
			Message:  "Failed to unmarshal response",
			Err:      err,
		}
	}

	if openAIResp.Error != nil {
		return nil, &ProviderError{
			Provider: p.GetName(),
			Code:     openAIResp.Error.Code,
			Message:  openAIResp.Error.Message,
		}
	}

	return &openAIResp, nil
}

func (p *OpenAIProvider) makeStreamRequest(ctx context.Context, req openAIRequest) (<-chan StreamChunk, error) {
	ch := make(chan StreamChunk, 100)

	go func() {
		defer close(ch)

		jsonData, err := json.Marshal(req)
		if err != nil {
			ch <- StreamChunk{Error: err}
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(jsonData))
		if err != nil {
			ch <- StreamChunk{Error: err}
			return
		}

		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Authorization", "Bearer "+p.config.APIKey)
		httpReq.Header.Set("Accept", "text/event-stream")

		resp, err := p.httpClient.Do(httpReq)
		if err != nil {
			ch <- StreamChunk{Error: err}
			return
		}
		defer resp.Body.Close()

		// Parse Server-Sent Events
		scanner := bufio.NewScanner(resp.Body)
		var content strings.Builder

		for scanner.Scan() {
			line := scanner.Text()

			if strings.HasPrefix(line, "data: ") {
				data := strings.TrimPrefix(line, "data: ")

				if data == "[DONE]" {
					ch <- StreamChunk{
						Content:      content.String(),
						FinishReason: "stop",
					}
					return
				}

				var streamResp openAIResponse
				if err := json.Unmarshal([]byte(data), &streamResp); err != nil {
					continue
				}

				if len(streamResp.Choices) > 0 {
					choice := streamResp.Choices[0]
					if choice.Delta != nil && choice.Delta.Content != "" {
						content.WriteString(choice.Delta.Content)
						ch <- StreamChunk{
							Content: content.String(),
							Delta:   choice.Delta.Content,
						}
					}
				}
			}
		}

		if err := scanner.Err(); err != nil {
			ch <- StreamChunk{Error: err}
		}
	}()

	return ch, nil
}

func (p *OpenAIProvider) parseCodeResponse(content string, req *CodeGenerationRequest) (code, tests, docs, explanation string) {
	// Simple parsing logic - in production, use more sophisticated parsing
	lines := strings.Split(content, "\n")

	var currentSection string
	var codeBuilder, testBuilder, docsBuilder, explanationBuilder strings.Builder

	for _, line := range lines {
		switch {
		case strings.Contains(line, "```"):
			if strings.Contains(line, req.Language) {
				currentSection = "code"
				continue
			} else if currentSection != "" {
				currentSection = ""
				continue
			}
		case strings.HasPrefix(line, "## Tests") || strings.HasPrefix(line, "# Tests"):
			currentSection = "tests"
			continue
		case strings.HasPrefix(line, "## Documentation") || strings.HasPrefix(line, "# Documentation"):
			currentSection = "docs"
			continue
		case strings.HasPrefix(line, "## Explanation") || strings.HasPrefix(line, "# Explanation"):
			currentSection = "explanation"
			continue
		}

		switch currentSection {
		case "code":
			codeBuilder.WriteString(line + "\n")
		case "tests":
			testBuilder.WriteString(line + "\n")
		case "docs":
			docsBuilder.WriteString(line + "\n")
		case "explanation":
			explanationBuilder.WriteString(line + "\n")
		default:
			if codeBuilder.Len() == 0 {
				explanationBuilder.WriteString(line + "\n")
			}
		}
	}

	return codeBuilder.String(), testBuilder.String(), docsBuilder.String(), explanationBuilder.String()
}

func (p *OpenAIProvider) parseExplanationResponse(content string) (overview string, steps []string, concepts []KeyConcept, complexity string, suggestions []string) {
	// Simple parsing - in production, use more sophisticated parsing
	lines := strings.Split(content, "\n")

	var currentSection string
	var overviewBuilder strings.Builder

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "## Overview") || strings.HasPrefix(line, "# Overview"):
			currentSection = "overview"
			continue
		case strings.HasPrefix(line, "## Step") || strings.HasPrefix(line, "# Step"):
			currentSection = "steps"
			continue
		case strings.HasPrefix(line, "## Key Concepts") || strings.HasPrefix(line, "# Key Concepts"):
			currentSection = "concepts"
			continue
		case strings.HasPrefix(line, "## Complexity") || strings.HasPrefix(line, "# Complexity"):
			currentSection = "complexity"
			continue
		case strings.HasPrefix(line, "## Suggestions") || strings.HasPrefix(line, "# Suggestions"):
			currentSection = "suggestions"
			continue
		}

		switch currentSection {
		case "overview":
			overviewBuilder.WriteString(line + " ")
		case "steps":
			if strings.HasPrefix(line, "-") || strings.HasPrefix(line, "*") || strings.HasPrefix(line, "1.") {
				steps = append(steps, strings.TrimPrefix(strings.TrimPrefix(line, "-"), "*"))
			}
		case "concepts":
			if strings.Contains(line, ":") {
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
			complexity += line + " "
		case "suggestions":
			if strings.HasPrefix(line, "-") || strings.HasPrefix(line, "*") {
				suggestions = append(suggestions, strings.TrimPrefix(strings.TrimPrefix(line, "-"), "*"))
			}
		default:
			if overviewBuilder.Len() == 0 {
				overviewBuilder.WriteString(line + " ")
			}
		}
	}

	overview = strings.TrimSpace(overviewBuilder.String())
	complexity = strings.TrimSpace(complexity)

	return overview, steps, concepts, complexity, suggestions
}
