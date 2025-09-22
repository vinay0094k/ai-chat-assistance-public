package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/yourusername/ai-code-assistant/internal/indexer/language_parsers"
)

func main() {
	fmt.Println("Testing Simple Indexer Components...")

	// Test Go Parser (this should work)
	testGoParser()

	fmt.Println("Simple indexer test completed!")
}

func testGoParser() {
	fmt.Println("\n=== Testing Go Parser ===")

	// Create a temporary Go file
	testFile := "test_sample.go"
	content := `package main

import "fmt"

func main() {
       fmt.Println("Hello World")
}

func helper(name string) string {
       return "Hello " + name
}

type TestStruct struct {
       Name string
       Age  int
}

type TestInterface interface {
       DoSomething() error
}
`

	err := os.WriteFile(testFile, []byte(content), 0644)
	if err != nil {
		log.Printf("Error creating test file: %v", err)
		return
	}
	defer os.Remove(testFile)

	parser := language_parsers.NewGoParser()

	// Get absolute path
	absPath, err := filepath.Abs(testFile)
	if err != nil {
		log.Printf("Error getting absolute path: %v", err)
		return
	}

	result, err := parser.ParseFile(absPath)
	if err != nil {
		log.Printf("Error parsing file: %v", err)
		return
	}

	fmt.Printf("✓ Successfully parsed file: %s\n", testFile)
	fmt.Printf("  - Language: %s\n", parser.GetLanguage())
	fmt.Printf("  - Functions found: %d\n", len(result.Functions))
	fmt.Printf("  - Types found: %d\n", len(result.Types))
	fmt.Printf("  - Imports found: %d\n", len(result.Imports))

	// Print details
	if len(result.Functions) > 0 {
		fmt.Println("  Functions:")
		for _, fn := range result.Functions {
			fmt.Printf("    - %s (line %d)\n", fn.Name, fn.StartLine)
		}
	}

	if len(result.Types) > 0 {
		fmt.Println("  Types:")
		for _, typ := range result.Types {
			fmt.Printf("    - %s (%s, line %d)\n", typ.Name, typ.Kind, typ.StartLine)
		}
	}

	if len(result.Imports) > 0 {
		fmt.Println("  Imports:")
		for _, imp := range result.Imports {
			fmt.Printf("    - %s\n", imp.Path)
		}
	}
}
