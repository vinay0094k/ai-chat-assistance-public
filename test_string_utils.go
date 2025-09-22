package main

import (
	"fmt"

	"github.com/yourusername/ai-code-assistant/utils"
)

func main() {
	processor := utils.NewStringProcessor()

	// Test text cleaning
	dirtyText := "  This   is    some   messy    text  with  extra   spaces!  "
	cleaned := processor.CleanText(dirtyText)
	fmt.Printf("✅ Text cleaning:\n  Input: %q\n  Output: %q\n\n", dirtyText, cleaned)

	// Test keyword extraction
	text := "This is a sample text about machine learning and artificial intelligence algorithms for code analysis"
	keywords := processor.ExtractKeywords(text, 3)
	fmt.Printf("✅ Keywords extracted: %v\n\n", keywords[:5]) // Show first 5

	// Test camelCase word extraction
	identifiers := []string{"getUserName", "calculateTotalAmount", "XMLHttpRequest"}
	fmt.Println("✅ CamelCase word extraction:")
	for _, id := range identifiers {
		words := utils.ExtractCamelCaseWords(id)
		fmt.Printf("  %s -> %v\n", id, words)
	}
	fmt.Println()

	// Test snake_case word extraction
	snakeIds := []string{"get_user_name", "calculate_total_amount", "http_request_handler"}
	fmt.Println("✅ Snake_case word extraction:")
	for _, id := range snakeIds {
		words := utils.ExtractSnakeCaseWords(id)
		fmt.Printf("  %s -> %v\n", id, words)
	}
	fmt.Println()

	// Test string similarity
	pairs := [][2]string{
		{"hello", "helo"},
		{"algorithm", "algorithmic"},
		{"function", "func"},
		{"completely", "different"},
	}

	fmt.Println("✅ String similarity:")
	for _, pair := range pairs {
		similarity := utils.CalculateSimilarity(pair[0], pair[1])
		fmt.Printf("  %s <-> %s: %.2f\n", pair[0], pair[1], similarity)
	}
	fmt.Println()

	// Test comment removal
	goCode := `package main

import "fmt" // Import statement

func main() {
    /* Multi-line comment
       about this function */
    fmt.Println("Hello") // Print greeting
}`

	cleaned = utils.RemoveComments(goCode, "go")
	fmt.Printf("✅ Comment removal:\nOriginal:\n%s\n\nCleaned:\n%s\n\n", goCode, cleaned)

	// Test function name extraction
	functions := utils.ExtractFunctionNames(goCode, "go")
	fmt.Printf("✅ Functions found: %v\n\n", functions)

	// Test import extraction
	imports := utils.ExtractImports(goCode, "go")
	fmt.Printf("✅ Imports found: %v\n", imports)
}


################################### Output##################################################################
✅ Text cleaning:
  Input: "  This   is    some   messy    text  with  extra   spaces!  "
  Output: "this is some messy text with extra spaces!"

✅ Keywords extracted: [code text machine learning intelligence]

✅ CamelCase word extraction:
  getUserName -> [get user name]
  calculateTotalAmount -> [calculate total amount]
  XMLHttpRequest -> [x m l http request]

✅ Snake_case word extraction:
  get_user_name -> [get user name]
  calculate_total_amount -> [calculate total amount]
  http_request_handler -> [http request handler]

✅ String similarity:
  hello <-> helo: 0.80
  algorithm <-> algorithmic: 0.82
  function <-> func: 0.50
  completely <-> different: 0.20

✅ Comment removal:
Original:
package main

import "fmt" // Import statement

func main() {
    /* Multi-line comment
       about this function */
    fmt.Println("Hello") // Print greeting
}

Cleaned:
package main

import "fmt" 

func main() {
    

    fmt.Println("Hello") 
}

✅ Functions found: [main]

✅ Imports found: [fmt]

###################################################################################################
