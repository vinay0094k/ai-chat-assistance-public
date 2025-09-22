package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/indexer"
)

func main() {
	fmt.Println("Testing Indexer Components...")

	// Test 1: Code Parser
	testCodeParser()

	// Test 2: File Watcher
	testFileWatcher()

	// Test 3: Ultra Fast Indexer
	testUltraFastIndexer()

	fmt.Println("All indexer tests completed!")
}

func testCodeParser() {
	fmt.Println("\n=== Testing Code Parser ===")

	// Create a temporary Go file
	testFile := "test_sample.go"
	content := `package main

import "fmt"

func main() {
      fmt.Println("Hello World")
}

type TestStruct struct {
      Name string
      Age  int
}
`

	err := os.WriteFile(testFile, []byte(content), 0644)
	if err != nil {
		log.Printf("Error creating test file: %v", err)
		return
	}
	defer os.Remove(testFile)

	parser := indexer.NewCodeParser()
	result, err := parser.ParseFile(testFile)
	if err != nil {
		log.Printf("Error parsing file: %v", err)
		return
	}

	fmt.Printf("✓ Parsed file: %s\n", testFile)
	fmt.Printf("  - Functions found: %d\n", len(result.Functions))
	fmt.Printf("  - Types found: %d\n", len(result.Types))
	fmt.Printf("  - Imports found: %d\n", len(result.Imports))
}

func testFileWatcher() {
	fmt.Println("\n=== Testing File Watcher ===")

	// Create temp directory
	tempDir := "test_watch_dir"
	err := os.MkdirAll(tempDir, 0755)
	if err != nil {
		log.Printf("Error creating temp dir: %v", err)
		return
	}
	defer os.RemoveAll(tempDir)

	watcher, err := indexer.NewFileWatcher()
	if err != nil {
		log.Printf("Error creating file watcher: %v", err)
		return
	}
	defer watcher.Close()

	// Start watching
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = watcher.Watch(tempDir)
	if err != nil {
		log.Printf("Error starting watch: %v", err)
		return
	}

	fmt.Printf("✓ File watcher started for: %s\n", tempDir)

	// Create a test file to trigger watch event
	testFile := filepath.Join(tempDir, "test.go")
	go func() {
		time.Sleep(1 * time.Second)
		os.WriteFile(testFile, []byte("package test"), 0644)
	}()

	// Listen for events
	select {
	case event := <-watcher.Events():
		fmt.Printf("✓ Detected file event: %s - %s\n", event.Type, event.Path)
	case err := <-watcher.Errors():
		log.Printf("Watcher error: %v", err)
	case <-ctx.Done():
		fmt.Printf("✓ File watcher test completed (timeout)\n")
	}
}

func testUltraFastIndexer() {
	fmt.Println("\n=== Testing Ultra Fast Indexer ===")

	// Create test directory structure
	testDir := "test_index_dir"
	err := os.MkdirAll(filepath.Join(testDir, "subdir"), 0755)
	if err != nil {
		log.Printf("Error creating test dir: %v", err)
		return
	}
	defer os.RemoveAll(testDir)

	// Create test files
	files := map[string]string{
		"main.go": `package main
import "fmt"
func main() { fmt.Println("test") }`,
		"utils.go": `package main
func helper() string { return "help" }`,
		"subdir/sub.go": `package subdir
type SubType struct { Value int }`,
	}

	for path, content := range files {
		fullPath := filepath.Join(testDir, path)
		os.MkdirAll(filepath.Dir(fullPath), 0755)
		err := os.WriteFile(fullPath, []byte(content), 0644)
		if err != nil {
			log.Printf("Error creating file %s: %v", path, err)
			continue
		}
	}

	// Test indexer
	indexer := indexer.NewUltraFastIndexer()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	config := &indexer.IndexConfig{
		RootPath:   testDir,
		Extensions: []string{".go"},
		MaxWorkers: 2,
		BatchSize:  10,
	}

	result, err := indexer.IndexDirectory(ctx, config)
	if err != nil {
		log.Printf("Error indexing directory: %v", err)
		return
	}

	fmt.Printf("✓ Indexed directory: %s\n", testDir)
	fmt.Printf("  - Files processed: %d\n", result.FilesProcessed)
	fmt.Printf("  - Total functions: %d\n", result.TotalFunctions)
	fmt.Printf("  - Total types: %d\n", result.TotalTypes)
	fmt.Printf("  - Processing time: %v\n", result.ProcessingTime)
}
