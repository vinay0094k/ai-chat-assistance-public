package utils

import (
	"math"
	"regexp"
	"strings"
	"unicode"
)

// TokenEstimator provides token counting and estimation utilities
type TokenEstimator struct {
	model string
}

// NewTokenEstimator creates a new token estimator
func NewTokenEstimator(model string) *TokenEstimator {
	return &TokenEstimator{
		model: strings.ToLower(model),
	}
}

// EstimateTokens estimates the number of tokens in text
func (te *TokenEstimator) EstimateTokens(text string) int {
	if text == "" {
		return 0
	}

	switch {
	case strings.Contains(te.model, "gpt"):
		return te.estimateGPTTokens(text)
	case strings.Contains(te.model, "claude"):
		return te.estimateClaudeTokens(text)
	case strings.Contains(te.model, "gemini"):
		return te.estimateGeminiTokens(text)
	case strings.Contains(te.model, "cohere"):
		return te.estimateCohereTokens(text)
	default:
		return te.estimateGenericTokens(text)
	}
}

// estimateGPTTokens estimates tokens for GPT models (OpenAI)
func (te *TokenEstimator) estimateGPTTokens(text string) int {
	// GPT tokenization approximation
	// Roughly 1 token per 4 characters for English text
	// Adjust for different content types

	charCount := len(text)
	if charCount == 0 {
		return 0
	}

	// Base estimation: 1 token per 4 characters
	baseTokens := float64(charCount) / 4.0

	// Adjustments based on content characteristics
	multiplier := 1.0

	// Code content typically has more tokens per character
	if te.isCodeContent(text) {
		multiplier *= 1.2
	}

	// Non-English content may have different tokenization
	if te.hasNonEnglishChars(text) {
		multiplier *= 1.3
	}

	// Special characters and punctuation
	specialCharCount := te.countSpecialChars(text)
	if float64(specialCharCount)/float64(charCount) > 0.1 {
		multiplier *= 1.1
	}

	// Word boundaries (spaces) affect tokenization
	wordCount := len(strings.Fields(text))
	if wordCount > 0 {
		avgWordLength := float64(charCount) / float64(wordCount)
		if avgWordLength > 8 { // Long words might be split
			multiplier *= 1.1
		}
	}

	return int(math.Ceil(baseTokens * multiplier))
}

// estimateClaudeTokens estimates tokens for Claude models (Anthropic)
func (te *TokenEstimator) estimateClaudeTokens(text string) int {
	// Claude tokenization is similar to GPT but slightly different
	charCount := len(text)
	if charCount == 0 {
		return 0
	}

	// Base estimation: slightly more efficient than GPT
	baseTokens := float64(charCount) / 3.8

	multiplier := 1.0

	if te.isCodeContent(text) {
		multiplier *= 1.15
	}

	if te.hasNonEnglishChars(text) {
		multiplier *= 1.25
	}

	return int(math.Ceil(baseTokens * multiplier))
}

// estimateGeminiTokens estimates tokens for Gemini models (Google)
func (te *TokenEstimator) estimateGeminiTokens(text string) int {
	charCount := len(text)
	if charCount == 0 {
		return 0
	}

	// Gemini uses SentencePiece, different characteristics
	baseTokens := float64(charCount) / 3.5

	multiplier := 1.0

	if te.isCodeContent(text) {
		multiplier *= 1.3
	}

	if te.hasNonEnglishChars(text) {
		multiplier *= 1.2
	}

	return int(math.Ceil(baseTokens * multiplier))
}

// estimateCohereTokens estimates tokens for Cohere models
func (te *TokenEstimator) estimateCohereTokens(text string) int {
	charCount := len(text)
	if charCount == 0 {
		return 0
	}

	baseTokens := float64(charCount) / 4.2

	multiplier := 1.0

	if te.isCodeContent(text) {
		multiplier *= 1.25
	}

	return int(math.Ceil(baseTokens * multiplier))
}

// estimateGenericTokens provides a generic token estimation
func (te *TokenEstimator) estimateGenericTokens(text string) int {
	// Generic estimation based on word count and character count
	words := strings.Fields(text)
	wordCount := len(words)
	charCount := len(text)

	if wordCount == 0 {
		return int(math.Ceil(float64(charCount) / 4.0))
	}

	// Estimate based on both word and character counts
	wordBasedTokens := float64(wordCount) * 1.3 // Average 1.3 tokens per word
	charBasedTokens := float64(charCount) / 4.0

	// Use the higher estimate for safety
	return int(math.Ceil(math.Max(wordBasedTokens, charBasedTokens)))
}

// isCodeContent determines if text is likely code
func (te *TokenEstimator) isCodeContent(text string) bool {
	codeIndicators := 0
	total := 0

	// Check for common code patterns
	patterns := []string{
		`\{.*\}`,      // Braces
		`\(.*\)`,      // Parentheses
		`\[.*\]`,      // Brackets
		`=.*`,         // Assignment
		`;$`,          // Semicolon at end of line
		`//.*`,        // Single line comments
		`/\*.*\*/`,    // Block comments
		`#.*`,         // Hash comments
		`func\s+\w+`,  // Function definitions
		`class\s+\w+`, // Class definitions
		`import\s+`,   // Import statements
		`package\s+`,  // Package declarations
	}

	for _, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		if re.MatchString(text) {
			codeIndicators++
		}
		total++
	}

	// Also check character distribution
	specialChars := te.countSpecialChars(text)
	alphaChars := te.countAlphaChars(text)

	if len(text) > 0 {
		specialRatio := float64(specialChars) / float64(len(text))
		if specialRatio > 0.15 { // High ratio of special characters
			codeIndicators++
		}

		if alphaChars > 0 {
			digitRatio := float64(te.countDigits(text)) / float64(alphaChars)
			if digitRatio > 0.2 { // High ratio of digits to letters
				codeIndicators++
			}
		}
	}

	// Consider it code if multiple indicators are present
	return float64(codeIndicators)/float64(total+2) > 0.3
}

// hasNonEnglishChars checks for non-English characters
func (te *TokenEstimator) hasNonEnglishChars(text string) bool {
	nonEnglishCount := 0
	totalChars := 0

	for _, r := range text {
		if unicode.IsLetter(r) {
			totalChars++
			if r > 127 { // Non-ASCII
				nonEnglishCount++
			}
		}
	}

	if totalChars == 0 {
		return false
	}

	return float64(nonEnglishCount)/float64(totalChars) > 0.1
}

// countSpecialChars counts special characters
func (te *TokenEstimator) countSpecialChars(text string) int {
	count := 0
	specialChars := "!@#$%^&*(){}[]|\\:;\"'<>?/.,`~-_+=/"

	for _, r := range text {
		if strings.ContainsRune(specialChars, r) {
			count++
		}
	}

	return count
}

// countAlphaChars counts alphabetic characters
func (te *TokenEstimator) countAlphaChars(text string) int {
	count := 0
	for _, r := range text {
		if unicode.IsLetter(r) {
			count++
		}
	}
	return count
}

// countDigits counts numeric digits
func (te *TokenEstimator) countDigits(text string) int {
	count := 0
	for _, r := range text {
		if unicode.IsDigit(r) {
			count++
		}
	}
	return count
}

// EstimateOutputTokens estimates output tokens based on request type
func (te *TokenEstimator) EstimateOutputTokens(requestType string, inputTokens int) int {
	switch strings.ToLower(requestType) {
	case "code_generation":
		// Code generation typically produces 2-4x input length
		return int(float64(inputTokens) * 3.0)
	case "explanation":
		// Explanations are usually 1.5-2x input length
		return int(float64(inputTokens) * 1.8)
	case "refactoring":
		// Refactoring is usually similar length to input
		return int(float64(inputTokens) * 1.2)
	case "search", "query":
		// Search responses are typically shorter
		return int(float64(inputTokens) * 0.5)
	case "documentation":
		// Documentation can be verbose
		return int(float64(inputTokens) * 2.5)
	default:
		// Default conservative estimate
		return int(float64(inputTokens) * 1.5)
	}
}

// CalculateCost calculates cost based on token usage and pricing
func (te *TokenEstimator) CalculateCost(inputTokens, outputTokens int, inputPrice, outputPrice float64) float64 {
	// Prices are typically per 1K tokens
	inputCost := float64(inputTokens) / 1000.0 * inputPrice
	outputCost := float64(outputTokens) / 1000.0 * outputPrice

	return inputCost + outputCost
}

// EstimateTotalCost estimates total cost for a request
func (te *TokenEstimator) EstimateTotalCost(text string, requestType string, inputPrice, outputPrice float64) (int, int, float64) {
	inputTokens := te.EstimateTokens(text)
	outputTokens := te.EstimateOutputTokens(requestType, inputTokens)
	totalCost := te.CalculateCost(inputTokens, outputTokens, inputPrice, outputPrice)

	return inputTokens, outputTokens, totalCost
}

// TokenBudgetCheck checks if a request fits within budget constraints
type TokenBudgetCheck struct {
	EstimatedInputTokens  int     `json:"estimated_input_tokens"`
	EstimatedOutputTokens int     `json:"estimated_output_tokens"`
	EstimatedTotalTokens  int     `json:"estimated_total_tokens"`
	EstimatedCost         float64 `json:"estimated_cost"`
	WithinBudget          bool    `json:"within_budget"`
	BudgetRemaining       float64 `json:"budget_remaining"`
	WarningLevel          string  `json:"warning_level"` // none, low, medium, high
}

// CheckBudget checks if request fits within budget
func (te *TokenEstimator) CheckBudget(text string, requestType string, inputPrice, outputPrice float64, budgetLimit float64, currentUsage float64) *TokenBudgetCheck {
	inputTokens, outputTokens, estimatedCost := te.EstimateTotalCost(text, requestType, inputPrice, outputPrice)

	totalTokens := inputTokens + outputTokens
	remainingBudget := budgetLimit - currentUsage
	withinBudget := estimatedCost <= remainingBudget

	// Determine warning level
	warningLevel := "none"
	usagePercent := (currentUsage + estimatedCost) / budgetLimit

	switch {
	case usagePercent >= 0.95:
		warningLevel = "high"
	case usagePercent >= 0.8:
		warningLevel = "medium"
	case usagePercent >= 0.6:
		warningLevel = "low"
	}

	return &TokenBudgetCheck{
		EstimatedInputTokens:  inputTokens,
		EstimatedOutputTokens: outputTokens,
		EstimatedTotalTokens:  totalTokens,
		EstimatedCost:         estimatedCost,
		WithinBudget:          withinBudget,
		BudgetRemaining:       remainingBudget,
		WarningLevel:          warningLevel,
	}
}

// SplitTextByTokens splits text into chunks of approximately maxTokens
func (te *TokenEstimator) SplitTextByTokens(text string, maxTokens int, overlap int) []string {
	if te.EstimateTokens(text) <= maxTokens {
		return []string{text}
	}

	var chunks []string
	lines := strings.Split(text, "\n")

	currentChunk := ""
	currentTokens := 0

	for _, line := range lines {
		lineTokens := te.EstimateTokens(line)

		// If adding this line would exceed the limit
		if currentTokens+lineTokens > maxTokens && currentChunk != "" {
			chunks = append(chunks, currentChunk)

			// Start new chunk with overlap
			if overlap > 0 && len(chunks) > 0 {
				overlapText := te.getLastNTokens(currentChunk, overlap)
				currentChunk = overlapText + "\n" + line
				currentTokens = te.EstimateTokens(currentChunk)
			} else {
				currentChunk = line
				currentTokens = lineTokens
			}
		} else {
			if currentChunk != "" {
				currentChunk += "\n"
			}
			currentChunk += line
			currentTokens += lineTokens
		}
	}

	// Add the last chunk
	if currentChunk != "" {
		chunks = append(chunks, currentChunk)
	}

	return chunks
}

// getLastNTokens approximates getting the last N tokens from text
func (te *TokenEstimator) getLastNTokens(text string, n int) string {
	if n <= 0 {
		return ""
	}

	totalTokens := te.EstimateTokens(text)
	if totalTokens <= n {
		return text
	}

	// Estimate characters per token
	charsPerToken := float64(len(text)) / float64(totalTokens)
	targetChars := int(float64(n) * charsPerToken)

	if targetChars >= len(text) {
		return text
	}

	// Start from the end and work backwards
	return text[len(text)-targetChars:]
}

// OptimizePrompt optimizes a prompt to fit within token limits
func (te *TokenEstimator) OptimizePrompt(prompt string, maxTokens int) (string, bool) {
	currentTokens := te.EstimateTokens(prompt)
	if currentTokens <= maxTokens {
		return prompt, false // No optimization needed
	}

	// Try various optimization strategies
	optimized := prompt

	// 1. Remove extra whitespace
	optimized = regexp.MustCompile(`\s+`).ReplaceAllString(optimized, " ")
	optimized = strings.TrimSpace(optimized)

	if te.EstimateTokens(optimized) <= maxTokens {
		return optimized, true
	}

	// 2. Remove comments if it's code
	if te.isCodeContent(optimized) {
		optimized = RemoveComments(optimized, "generic")
		if te.EstimateTokens(optimized) <= maxTokens {
			return optimized, true
		}
	}

	// 3. Truncate from the end while preserving structure
	return te.truncateToTokenLimit(optimized, maxTokens), true
}

// truncateToTokenLimit truncates text to fit within token limit
func (te *TokenEstimator) truncateToTokenLimit(text string, maxTokens int) string {
	currentTokens := te.EstimateTokens(text)
	if currentTokens <= maxTokens {
		return text
	}

	// Estimate the proportion to keep
	targetRatio := float64(maxTokens) / float64(currentTokens)
	targetLength := int(float64(len(text)) * targetRatio * 0.9) // 10% safety margin

	if targetLength >= len(text) {
		return text
	}

	// Try to truncate at word boundaries
	words := strings.Fields(text)
	var result strings.Builder

	for _, word := range words {
		testResult := result.String()
		if testResult != "" {
			testResult += " "
		}
		testResult += word

		if te.EstimateTokens(testResult) > maxTokens {
			break
		}

		if result.Len() > 0 {
			result.WriteString(" ")
		}
		result.WriteString(word)
	}

	return result.String()
}
