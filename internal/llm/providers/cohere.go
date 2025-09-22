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

type CohereProvider struct {
	config     *ProviderConfig
	httpClient *http.Client
}

type cohereRequest struct {
	Message     string              `json:"message"`
	Model       string              `json:"model,omitempty"`
	ChatHistory []cohereChatMessage `json:"chat_history,omitempty"`
	Temperature float64             `json:"temperature,omitempty"`
	MaxTokens   int                 `json:"max_tokens,omitempty"`
	K           int                 `json:"k,omitempty"`
	P           float64             `json:"p,omitempty"`
	Stream      bool                `json:"stream,omitempty"`
	Preamble    string              `json:"preamble,omitempty"`
}

type cohereChatMessage struct {
	Role    string `json:"role"`
	Message string `json:"message"`
}

type cohereResponse struct {
	ResponseID   string              `json:"response_id"`
	Text         string              `json:"text"`
	GenerationID string              `json:"generation_id"`
	ChatHistory  []cohereChatMessage `json:"chat_history"`
	FinishReason string              `json:"finish_reason"`
	Meta         cohereMeta          `json:"meta,omitempty"`
	Error        *cohereError        `json:"error,omitempty"`
}

type cohereMeta struct {
	APIVersion  cohereAPIVersion  `json:"api_version"`
	BilledUnits cohereBilledUnits `json:"billed_units"`
}

type cohereAPIVersion struct {
	Version string `json:"version"`
}

type cohereBilledUnits struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type cohereError struct {
	Message string `json:"message"`
}

func NewCohereProvider(config *ProviderConfig) *CohereProvider {
	return &CohereProvider{
		config: config,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
	}
}

func (p *CohereProvider) GetName() string {
	return "cohere"
}

func (p *CohereProvider) GetModel() string {
	return p.config.Model
}

func (p *CohereProvider) IsEnabled() bool {
	return p.config.Enabled && p.config.APIKey != ""
}

func (p *CohereProvider) GenerateText(ctx context.Context, req *TextGenerationRequest) (*TextGenerationResponse, error) {
	start := time.Now()

	chatHistory := p.buildChatHistory(req)

	cohereReq := cohereRequest{
		Message:     req.Prompt,
		Model:       p.config.Model,
		ChatHistory: chatHistory,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		P:           req.TopP,
		Preamble:    req.System,
		Stream:      false,
	}

	resp, err := p.makeRequest(ctx, cohereReq)
	if err != nil {
		return nil, err
	}

	cost := p.CalculateCost(resp.Meta.BilledUnits.InputTokens, resp.Meta.BilledUnits.OutputTokens)

	return &TextGenerationResponse{
		Content:      resp.Text,
		FinishReason: resp.FinishReason,
		TokensUsed: TokenUsage{
			InputTokens:  resp.Meta.BilledUnits.InputTokens,
			OutputTokens: resp.Meta.BilledUnits.OutputTokens,
			TotalTokens:  resp.Meta.BilledUnits.InputTokens + resp.Meta.BilledUnits.OutputTokens,
			Cost:         cost,
		},
		Model:    p.config.Model,
		Provider: p.GetName(),
		Duration: time.Since(start),
	}, nil
}

func (p *CohereProvider) GenerateCode(ctx context.Context, req *CodeGenerationRequest) (*CodeGenerationResponse, error) {
	prompt := p.buildCodeGenerationPrompt(req)

	textReq := &TextGenerationRequest{
		Prompt:      prompt,
		System:      "You are an expert software engineer who writes clean, efficient, well-documented code following industry best practices.",
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

func (p *CohereProvider) ExplainCode(ctx context.Context, req *CodeExplanationRequest) (*CodeExplanationResponse, error) {
	prompt := p.buildCodeExplanationPrompt(req)

	textReq := &TextGenerationRequest{
		Prompt:      prompt,
		System:      "You are an expert programming instructor who excels at breaking down complex code into understandable explanations.",
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

func (p *CohereProvider) StreamText(ctx context.Context, req *TextGenerationRequest) (<-chan StreamChunk, error) {
	ch := make(chan StreamChunk, 1)
	go func() {
		defer close(ch)
		ch <- StreamChunk{
			Content: "Streaming not implemented for Cohere yet",
			Error:   fmt.Errorf("streaming not yet supported"),
		}
	}()
	return ch, nil
}

func (p *CohereProvider) GetTokenCount(text string) (int, error) {
	return len(text) / 4, nil
}

func (p *CohereProvider) CalculateCost(inputTokens, outputTokens int) float64 {
	inputCost := float64(inputTokens) / 1000.0 * p.config.CostPer1KInput
	outputCost := float64(outputTokens) / 1000.0 * p.config.CostPer1KOutput
	return inputCost + outputCost
}

func (p *CohereProvider) HealthCheck(ctx context.Context) error {
	testReq := &TextGenerationRequest{
		Prompt:    "Hello",
		MaxTokens: 5,
	}

	_, err := p.GenerateText(ctx, testReq)
	return err
}

func (p *CohereProvider) GetCapabilities() ProviderCapabilities {
	return ProviderCapabilities{
		TextGeneration:     true,
		CodeGeneration:     true,
		CodeExplanation:    true,
		Streaming:          false,
		MaxTokens:          p.config.MaxTokens,
		SupportedLanguages: []string{"go", "python", "javascript", "java", "c", "cpp"},
		Features:           []string{"chat", "generation"},
	}
}

func (p *CohereProvider) buildChatHistory(req *TextGenerationRequest) []cohereChatMessage {
	var history []cohereChatMessage

	for _, msg := range req.Context {
		if msg.Role != "system" {
			role := msg.Role
			if role == "assistant" {
				role = "CHATBOT"
			} else {
				role = "USER"
			}

			history = append(history, cohereChatMessage{
				Role:    role,
				Message: msg.Content,
			})
		}
	}

	return history
}

func (p *CohereProvider) buildCodeGenerationPrompt(req *CodeGenerationRequest) string {
	var prompt strings.Builder

	prompt.WriteString(fmt.Sprintf("Write %s code that %s\n\n", req.Language, req.Description))

	if req.Context != "" {
		prompt.WriteString(fmt.Sprintf("Context: %s\n\n", req.Context))
	}

	prompt.WriteString("Requirements:\n")
	prompt.WriteString("- Write clean, maintainable code\n")
	prompt.WriteString("- Include proper error handling\n")
	prompt.WriteString("- Add clear, concise comments\n")
	prompt.WriteString("- Follow language conventions\n")

	if req.Tests {
		prompt.WriteString("- Include unit tests\n")
	}

	if req.Docs {
		prompt.WriteString("- Include documentation\n")
	}

	return prompt.String()
}

func (p *CohereProvider) buildCodeExplanationPrompt(req *CodeExplanationRequest) string {
	var prompt strings.Builder

	prompt.WriteString("Explain this code in detail:\n\n")
	prompt.WriteString("```" + req.Language + "\n")
	prompt.WriteString(req.Code)
	prompt.WriteString("\n```\n\n")

	prompt.WriteString("Please provide:\n")
	prompt.WriteString("1. Overview of functionality\n")
	prompt.WriteString("2. Step-by-step breakdown\n")
	prompt.WriteString("3. Key concepts used\n")
	prompt.WriteString("4. Complexity assessment\n")

	if req.Focus != "" {
		prompt.WriteString(fmt.Sprintf("5. Focus on: %s\n", req.Focus))
	}

	return prompt.String()
}

func (p *CohereProvider) makeRequest(ctx context.Context, req cohereRequest) (*cohereResponse, error) {
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, &ProviderError{
			Provider: p.GetName(),
			Code:     "MARSHAL_ERROR",
			Message:  "Failed to marshal request",
			Err:      err,
		}
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", "https://api.cohere.ai/v1/chat", bytes.NewBuffer(jsonData))
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

	var cohereResp cohereResponse
	if err := json.Unmarshal(body, &cohereResp); err != nil {
		return nil, &ProviderError{
			Provider: p.GetName(),
			Code:     "UNMARSHAL_ERROR",
			Message:  "Failed to unmarshal response",
			Err:      err,
		}
	}

	if cohereResp.Error != nil {
		return nil, &ProviderError{
			Provider: p.GetName(),
			Code:     "COHERE_ERROR",
			Message:  cohereResp.Error.Message,
		}
	}

	return &cohereResp, nil
}

func (p *CohereProvider) parseCodeResponse(content string, req *CodeGenerationRequest) (code, tests, docs, explanation string) {
	return parseGenericCodeResponse(content, req.Language, req.Tests, req.Docs)
}

func (p *CohereProvider) parseExplanationResponse(content string) (overview string, steps []string, concepts []KeyConcept, complexity string, suggestions []string) {
	return parseGenericExplanationResponse(content)
}

// Helper functions shared across providers
func parseGenericCodeResponse(content, language string, includeTests, includeDocs bool) (code, tests, docs, explanation string) {
	sections := strings.Split(content, "\n")
	var currentSection string
	var codeBuilder, testBuilder, docsBuilder, explanationBuilder strings.Builder

	inCodeBlock := false

	for _, line := range sections {
		if strings.HasPrefix(line, "```") {
			inCodeBlock = !inCodeBlock
			if strings.Contains(line, language) {
				currentSection = "code"
			}
			continue
		}

		lowerLine := strings.ToLower(line)
		switch {
		case strings.Contains(lowerLine, "test") && (strings.Contains(lowerLine, "#") || strings.Contains(lowerLine, "##")):
			currentSection = "tests"
			continue
		case strings.Contains(lowerLine, "doc") && (strings.Contains(lowerLine, "#") || strings.Contains(lowerLine, "##")):
			currentSection = "docs"
			continue
		case strings.Contains(lowerLine, "explanation") && (strings.Contains(lowerLine, "#") || strings.Contains(lowerLine, "##")):
			currentSection = "explanation"
			continue
		}

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
			if !inCodeBlock && !strings.Contains(line, "#") {
				explanationBuilder.WriteString(line + "\n")
			}
		}
	}

	return strings.TrimSpace(codeBuilder.String()),
		strings.TrimSpace(testBuilder.String()),
		strings.TrimSpace(docsBuilder.String()),
		strings.TrimSpace(explanationBuilder.String())
}

func parseGenericExplanationResponse(content string) (overview string, steps []string, concepts []KeyConcept, complexity string, suggestions []string) {
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
		case (strings.Contains(lowerLine, "overview") || strings.Contains(lowerLine, "summary")) && strings.Contains(line, "#"):
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
		case (strings.Contains(lowerLine, "suggestion") || strings.Contains(lowerLine, "improvement")) && strings.Contains(line, "#"):
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
				cleanStep := strings.TrimPrefix(strings.TrimPrefix(line, "-"), "*")
				cleanStep = strings.TrimSpace(cleanStep)
				if cleanStep != "" {
					steps = append(steps, cleanStep)
				}
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
				cleanSuggestion := strings.TrimPrefix(strings.TrimPrefix(line, "-"), "*")
				cleanSuggestion = strings.TrimSpace(cleanSuggestion)
				if cleanSuggestion != "" {
					suggestions = append(suggestions, cleanSuggestion)
				}
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
