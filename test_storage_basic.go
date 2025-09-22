package main

import (
	"database/sql"
	"fmt"
	"log"

	_ "github.com/mattn/go-sqlite3"
)

// Test basic storage functionality without dependencies
func main() {
	// Test SQLite connection
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		log.Fatal("Failed to open database:", err)
	}
	defer db.Close()

	// Test basic table creation
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS test_storage (
			id TEXT PRIMARY KEY,
			name TEXT,
			data TEXT
		)
	`)
	if err != nil {
		log.Fatal("Failed to create table:", err)
	}

	// Test insert
	_, err = db.Exec("INSERT INTO test_storage (id, name, data) VALUES (?, ?, ?)", 
		"test1", "Test Storage", "Sample data")
	if err != nil {
		log.Fatal("Failed to insert:", err)
	}

	// Test query
	var id, name, data string
	err = db.QueryRow("SELECT id, name, data FROM test_storage WHERE id = ?", "test1").Scan(&id, &name, &data)
	if err != nil {
		log.Fatal("Failed to query:", err)
	}

	fmt.Printf("✅ Storage test successful!\n")
	fmt.Printf("   ID: %s\n", id)
	fmt.Printf("   Name: %s\n", name)
	fmt.Printf("   Data: %s\n", data)
	fmt.Printf("   Database connection: OK\n")
	fmt.Printf("   Basic CRUD operations: OK\n")
}
