package main

import (
	"fmt"
	"strings"

	"github.com/yourusername/ai-code-assistant/utils"
)

func main() {
	// Only test with GPT-4
	models := []string{"gpt-4"}

	// Only English texts
	texts := []string{
		"Hello, world!",
		"This is a longer text that should have more tokens for testing purposes.",
		`func calculateSum(numbers []int) int {
    sum := 0
    for _, num := range numbers {
        sum += num
    }
    return sum
}`,
	}

	fmt.Println("✅ Token estimation testing:")
	fmt.Printf("%-15s %-50s %s\n", "Model", "Text Preview", "Tokens")
	fmt.Println(strings.Repeat("-", 80))

	for _, model := range models {
		estimator := utils.NewTokenEstimator(model)

		for i, text := range texts {
			preview := text
			if len(preview) > 45 {
				preview = preview[:42] + "..."
			}

			tokens := estimator.EstimateTokens(text)

			if i == 0 {
				fmt.Printf("%-15s %-50s %d\n", model, preview, tokens)
			} else {
				fmt.Printf("%-15s %-50s %d\n", "", preview, tokens)
			}
		}
		fmt.Println()
	}

	// Test cost calculation with GPT-4
	estimator := utils.NewTokenEstimator("gpt-4")
	text := "Generate a Go function that calculates the fibonacci sequence up to n terms"

	inputTokens, outputTokens, cost := estimator.EstimateTotalCost(
		text,
		"code_generation",
		0.03, // $0.03 per 1K input tokens
		0.06, // $0.06 per 1K output tokens
	)

	fmt.Printf("✅ Cost estimation example:\n")
	fmt.Printf("  Input tokens: %d\n", inputTokens)
	fmt.Printf("  Output tokens: %d\n", outputTokens)
	fmt.Printf("  Total cost: $%.4f\n\n", cost)

	// Test budget checking
	budget := estimator.CheckBudget(text, "code_generation", 0.03, 0.06, 10.00, 7.50)

	fmt.Printf("✅ Budget check:\n")
	fmt.Printf("  Within budget: %t\n", budget.WithinBudget)
	fmt.Printf("  Warning level: %s\n", budget.WarningLevel)
	fmt.Printf("  Budget remaining: $%.2f\n\n", budget.BudgetRemaining)

	// Test text splitting
	longText := strings.Repeat("This is a test sentence that will be repeated many times. ", 50)
	chunks := estimator.SplitTextByTokens(longText, 500, 50)

	fmt.Printf("✅ Text splitting:\n")
	fmt.Printf("  Original length: %d chars\n", len(longText))
	fmt.Printf("  Split into %d chunks\n", len(chunks))
	for i, chunk := range chunks {
		tokens := estimator.EstimateTokens(chunk)
		fmt.Printf("  Chunk %d: %d chars, ~%d tokens\n", i+1, len(chunk), tokens)
	}
}

// ######################################################################################################
// ✅ Token estimation testing:
// Model           Text Preview                                       Tokens
// --------------------------------------------------------------------------------
// gpt-4           Hello, world!                                      4
//                 This is a longer text that should have mor...      18
//                 func calculateSum(numbers []int) int {
//    ...      35

// ✅ Cost estimation example:
//   Input tokens: 19
//   Output tokens: 57
//   Total cost: $0.0040

// ✅ Budget check:
//   Within budget: true
//   Warning level: low
//   Budget remaining: $2.50

// ✅ Text splitting:
//   Original length: 2900 chars
//   Split into 1 chunks
//   Chunk 1: 2900 chars, ~725 tokens

// ######################################################################################################
