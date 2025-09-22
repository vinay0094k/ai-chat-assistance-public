package main

import (
	"fmt"
	"log"
	"time"

	"github.com/vinay0094k/ai-code-assistant/models"
	"github.com/vinay0094k/ai-code-assistant/storage"
)

func main() {
	config := &storage.DatabaseConfig{
		Path:           "./test_operations.db",
		MaxConnections: 5,
	}

	db, err := storage.NewSQLiteDB(config)
	if err != nil {
		log.Fatal(err)
	}

	if err := db.Connect(); err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	// Test saving a code file
	file := &models.CodeFile{
		ID:           "test-file-1",
		Path:         "/test/main.go",
		RelativePath: "main.go",
		Language:     "go",
		Size:         1024,
		Hash:         "abc123",
		LineCount:    50,
		ChunkCount:   3,
		LastModified: time.Now(),
		IndexedAt:    time.Now(),
		Metadata:     map[string]interface{}{"test": true},
	}

	if err := db.SaveCodeFile(file); err != nil {
		log.Fatal("Failed to save file:", err)
	}

	fmt.Println("✅ Code file saved successfully")

	// Test retrieving the file
	retrieved, err := db.GetCodeFile("test-file-1")
	if err != nil {
		log.Fatal("Failed to retrieve file:", err)
	}

	fmt.Printf("✅ Retrieved file: %s (Language: %s)\n", retrieved.Path, retrieved.Language)

	// Test project statistics
	stats, err := db.GetProjectStatistics("test-project")
	if err != nil {
		log.Fatal("Failed to get stats:", err)
	}

	fmt.Printf("✅ Project stats - Files: %d, Chunks: %d\n", stats.TotalFiles, stats.TotalChunks)
}
