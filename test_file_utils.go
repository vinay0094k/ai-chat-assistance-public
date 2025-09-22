package main

import (
	"fmt"
	"os"

	"github.com/yourusername/ai-code-assistant/utils"
)

func main() {
	// Create test filess
	os.MkdirAll("test_project/src", 0755)

	// Create test Go file
	goCode := `package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}
`
	os.WriteFile("test_project/src/main.go", []byte(goCode), 0644)

	// Create test Python file
	pyCode := `#!/usr/bin/env python3

def greet(name):
    """Greet someone"""
    print(f"Hello, {name}!")

if __name__ == "__main__":
    greet("World")
`
	os.WriteFile("test_project/src/hello.py", []byte(pyCode), 0644)

	// Test file info
	info, err := utils.GetFileInfo("test_project/src/main.go")
	if err != nil {
		fmt.Printf("❌ Failed to get file info: %v\n", err)
		return
	}

	fmt.Printf("✅ File info - Name: %s, Size: %d, Language: %s\n",
		info.Name, info.Size, info.Language)

	// Test walking files
	options := &utils.WalkOptions{
		IncludePatterns: []string{"*.go", "*.py"},
		MaxDepth:        -1,
	}

	files, err := utils.WalkFiles("test_project", options)
	if err != nil {
		fmt.Printf("❌ Failed to walk files: %v\n", err)
		return
	}

	fmt.Printf("✅ Found %d files\n", len(files))
	for _, file := range files {
		if !file.IsDir {
			fmt.Printf("  - %s (%s)\n", file.RelativePath, file.Language)
		}
	}

	// Test language detection
	languages := []string{".go", ".py", ".js", ".rs", ".java"}
	fmt.Println("✅ Language detection:")
	for _, ext := range languages {
		lang := utils.DetectLanguage(ext)
		fmt.Printf("  %s -> %s\n", ext, lang)
	}

	// Test file statistics
	projectFiles := utils.GetProjectFiles(files)
	stats := utils.CalculateFileStats(projectFiles)

	fmt.Printf("✅ Project statistics:\n")
	fmt.Printf("  Total files: %d\n", stats.TotalFiles)
	fmt.Printf("  Total size: %d bytes\n", stats.TotalSize)
	fmt.Printf("  Languages: %v\n", stats.LanguageStats)

	// Cleanup
	os.RemoveAll("test_project")
}

// ########################## OUTPUT ######################################
// ✅ File info - Name: main.go, Size: 77, Language: go
// ✅ Found 2 files
//   - src/hello.py (python)
//   - src/main.go (go)
// ✅ Language detection:
//   .go -> go
//   .py -> python
//   .js -> javascript
//   .rs -> rust
//   .java -> java
// ✅ Project statistics:
//   Total files: 2
//   Total size: 218 bytes
//   Languages: map[go:1 python:1]
// ################################################################
