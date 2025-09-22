package providers

import (
	"context"
	"fmt"
	"time"
)

// Provider defines the interface that all AI providers must implement
type Provider interface {
	// GetName returns the provider name (e.g., "openai", "claude", "gemini")
	GetName() string

	// GetModel returns the current model being used
	GetModel() string

	// IsEnabled returns whether the provider is currently enabled
	IsEnabled() bool

	// GenerateText generates text based on the prompt
	GenerateText(ctx context.Context, req *TextGenerationRequest) (*TextGenerationResponse, error)

	// GenerateCode generates code based on the request
	GenerateCode(ctx context.Context, req *CodeGenerationRequest) (*CodeGenerationResponse, error)

	// ExplainCode explains the given code
	ExplainCode(ctx context.Context, req *CodeExplanationRequest) (*CodeExplanationResponse, error)

	// StreamText generates streaming text response
	StreamText(ctx context.Context, req *TextGenerationRequest) (<-chan StreamChunk, error)

	// GetTokenCount estimates token count for the given text
	GetTokenCount(text string) (int, error)

	// CalculateCost calculates cost for the given token usage
	CalculateCost(inputTokens, outputTokens int) float64

	// HealthCheck verifies the provider is accessible
	HealthCheck(ctx context.Context) error

	// GetCapabilities returns provider capabilities
	GetCapabilities() ProviderCapabilities
}

// TextGenerationRequest represents a text generation request
type TextGenerationRequest struct {
	Prompt      string                 `json:"prompt"`
	System      string                 `json:"system,omitempty"`
	MaxTokens   int                    `json:"max_tokens,omitempty"`
	Temperature float64                `json:"temperature,omitempty"`
	TopP        float64                `json:"top_p,omitempty"`
	Stream      bool                   `json:"stream,omitempty"`
	Context     []Message              `json:"context,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// CodeGenerationRequest represents a code generation request
type CodeGenerationRequest struct {
	Description string                 `json:"description"`
	Language    string                 `json:"language"`
	Context     string                 `json:"context,omitempty"`
	Style       string                 `json:"style,omitempty"`
	Tests       bool                   `json:"include_tests,omitempty"`
	Docs        bool                   `json:"include_docs,omitempty"`
	MaxTokens   int                    `json:"max_tokens,omitempty"`
	Temperature float64                `json:"temperature,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// CodeExplanationRequest represents a code explanation request
type CodeExplanationRequest struct {
	Code      string                 `json:"code"`
	Language  string                 `json:"language"`
	Detail    string                 `json:"detail,omitempty"` // brief, normal, detailed
	Focus     string                 `json:"focus,omitempty"`  // logic, performance, security, etc.
	Context   string                 `json:"context,omitempty"`
	MaxTokens int                    `json:"max_tokens,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// TextGenerationResponse represents a text generation response
type TextGenerationResponse struct {
	Content      string                 `json:"content"`
	FinishReason string                 `json:"finish_reason"`
	TokensUsed   TokenUsage             `json:"tokens_used"`
	Model        string                 `json:"model"`
	Provider     string                 `json:"provider"`
	Duration     time.Duration          `json:"duration"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// CodeGenerationResponse represents a code generation response
type CodeGenerationResponse struct {
	Code          string                 `json:"code"`
	Language      string                 `json:"language"`
	Explanation   string                 `json:"explanation,omitempty"`
	Tests         string                 `json:"tests,omitempty"`
	Documentation string                 `json:"documentation,omitempty"`
	Suggestions   []string               `json:"suggestions,omitempty"`
	TokensUsed    TokenUsage             `json:"tokens_used"`
	Model         string                 `json:"model"`
	Provider      string                 `json:"provider"`
	Duration      time.Duration          `json:"duration"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
}

// CodeExplanationResponse represents a code explanation response
type CodeExplanationResponse struct {
	Overview    string                 `json:"overview"`
	StepByStep  []string               `json:"step_by_step"`
	KeyConcepts []KeyConcept           `json:"key_concepts"`
	Complexity  string                 `json:"complexity"`
	Suggestions []string               `json:"suggestions,omitempty"`
	TokensUsed  TokenUsage             `json:"tokens_used"`
	Model       string                 `json:"model"`
	Provider    string                 `json:"provider"`
	Duration    time.Duration          `json:"duration"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// StreamChunk represents a chunk in a streaming response
type StreamChunk struct {
	Content      string                 `json:"content"`
	Delta        string                 `json:"delta"`
	FinishReason string                 `json:"finish_reason,omitempty"`
	TokensUsed   int                    `json:"tokens_used,omitempty"`
	Error        error                  `json:"error,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// Message represents a conversation message
type Message struct {
	Role    string `json:"role"` // user, assistant, system
	Content string `json:"content"`
}

// TokenUsage represents token usage information
type TokenUsage struct {
	InputTokens  int     `json:"input_tokens"`
	OutputTokens int     `json:"output_tokens"`
	TotalTokens  int     `json:"total_tokens"`
	Cost         float64 `json:"cost"`
}

// KeyConcept represents a key concept in code explanation
type KeyConcept struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Importance  string `json:"importance"`
}

// ProviderCapabilities represents provider capabilities
type ProviderCapabilities struct {
	TextGeneration     bool     `json:"text_generation"`
	CodeGeneration     bool     `json:"code_generation"`
	CodeExplanation    bool     `json:"code_explanation"`
	Streaming          bool     `json:"streaming"`
	MaxTokens          int      `json:"max_tokens"`
	SupportedLanguages []string `json:"supported_languages"`
	Features           []string `json:"features"`
}

// ProviderConfig represents provider configuration
type ProviderConfig struct {
	Name            string        `json:"name"`
	APIKey          string        `json:"api_key"`
	Model           string        `json:"model"`
	BaseURL         string        `json:"base_url,omitempty"`
	MaxTokens       int           `json:"max_tokens,omitempty"`
	Temperature     float64       `json:"temperature,omitempty"`
	TopP            float64       `json:"top_p,omitempty"`
	Timeout         time.Duration `json:"timeout,omitempty"`
	MaxRetries      int           `json:"max_retries,omitempty"`
	CostPer1KInput  float64       `json:"cost_per_1k_input"`
	CostPer1KOutput float64       `json:"cost_per_1k_output"`
	Enabled         bool          `json:"enabled"`
	Weight          float64       `json:"weight"`
}

// ProviderError represents provider-specific errors
type ProviderError struct {
	Provider string
	Code     string
	Message  string
	Err      error
}

func (e *ProviderError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("%s provider error [%s]: %s - %v", e.Provider, e.Code, e.Message, e.Err)
	}
	return fmt.Sprintf("%s provider error [%s]: %s", e.Provider, e.Code, e.Message)
}

func (e *ProviderError) Unwrap() error {
	return e.Err
}
