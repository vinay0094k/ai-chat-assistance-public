package indexer

import (
	"crypto/md5"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/yourusername/ai-code-assistant/storage"
)

// ChangeDetector tracks file changes for incremental indexing
type ChangeDetector struct {
	db           *storage.SQLiteDB
	fileHashes   map[string]string    // filePath -> hash
	fileModTimes map[string]time.Time // filePath -> modification time
	mu           sync.RWMutex
	config       *ChangeDetectorConfig
	lastScanTime time.Time
	changeStats  *ChangeStatistics
}

// ChangeDetectorConfig contains configuration for change detection
type ChangeDetectorConfig struct {
	HashAlgorithm    string        `json:"hash_algorithm"`    // md5, sha256
	CheckInterval    time.Duration `json:"check_interval"`    // How often to scan for changes
	IgnorePatterns   []string      `json:"ignore_patterns"`   // Patterns to ignore
	IgnoreExtensions []string      `json:"ignore_extensions"` // File extensions to ignore
	MinFileSize      int64         `json:"min_file_size"`     // Minimum file size to track
	MaxFileSize      int64         `json:"max_file_size"`     // Maximum file size to track
	TrackMetadata    bool          `json:"track_metadata"`    // Track file metadata changes
}

// FileChange represents a detected change in a file
type FileChange struct {
	FilePath   string                 `json:"file_path"`
	ChangeType ChangeType             `json:"change_type"`
	OldHash    string                 `json:"old_hash,omitempty"`
	NewHash    string                 `json:"new_hash"`
	OldModTime time.Time              `json:"old_mod_time,omitempty"`
	NewModTime time.Time              `json:"new_mod_time"`
	FileSize   int64                  `json:"file_size"`
	DetectedAt time.Time              `json:"detected_at"`
	Language   string                 `json:"language"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

// ChangeType represents the type of change detected
type ChangeType string

const (
	ChangeTypeAdded    ChangeType = "added"
	ChangeTypeModified ChangeType = "modified"
	ChangeTypeDeleted  ChangeType = "deleted"
	ChangeTypeRenamed  ChangeType = "renamed"
	ChangeTypeMoved    ChangeType = "moved"
)

// ChangeStatistics tracks statistics about detected changes
type ChangeStatistics struct {
	TotalChanges    int64                `json:"total_changes"`
	ChangesByType   map[ChangeType]int64 `json:"changes_by_type"`
	ChangesByLang   map[string]int64     `json:"changes_by_language"`
	LastUpdateTime  time.Time            `json:"last_update_time"`
	ScanCount       int64                `json:"scan_count"`
	AverageScanTime time.Duration        `json:"average_scan_time"`
	FilesTracked    int64                `json:"files_tracked"`
	mu              sync.RWMutex
}

// ChangeSet represents a batch of changes
type ChangeSet struct {
	ID          string        `json:"id"`
	Changes     []*FileChange `json:"changes"`
	DetectedAt  time.Time     `json:"detected_at"`
	ScanTime    time.Duration `json:"scan_time"`
	TriggerType string        `json:"trigger_type"` // manual, scheduled, realtime
}

// NewChangeDetector creates a new change detector
func NewChangeDetector(db *storage.SQLiteDB, config *ChangeDetectorConfig) *ChangeDetector {
	if config == nil {
		config = &ChangeDetectorConfig{
			HashAlgorithm:    "md5",
			CheckInterval:    time.Minute * 5,
			MinFileSize:      1,
			MaxFileSize:      10 * 1024 * 1024, // 10MB
			TrackMetadata:    true,
			IgnoreExtensions: []string{".tmp", ".log", ".cache"},
		}
	}

	return &ChangeDetector{
		db:           db,
		fileHashes:   make(map[string]string),
		fileModTimes: make(map[string]time.Time),
		config:       config,
		changeStats: &ChangeStatistics{
			ChangesByType: make(map[ChangeType]int64),
			ChangesByLang: make(map[string]int64),
		},
	}
}

// Initialize loads existing file information from the database
func (cd *ChangeDetector) Initialize() error {
	cd.mu.Lock()
	defer cd.mu.Unlock()

	// Load file hashes from database
	query := `
		SELECT relative_path, hash, updated_at 
		FROM files 
		WHERE indexed_at IS NOT NULL
	`

	rows, err := cd.db.Query(query)
	if err != nil {
		return fmt.Errorf("failed to load file hashes: %v", err)
	}
	defer rows.Close()

	for rows.Next() {
		var filePath, hash string
		var updatedAt time.Time

		if err := rows.Scan(&filePath, &hash, &updatedAt); err != nil {
			continue
		}

		cd.fileHashes[filePath] = hash
		cd.fileModTimes[filePath] = updatedAt
	}

	cd.lastScanTime = time.Now()
	return nil
}

// ScanDirectory scans a directory for changes
func (cd *ChangeDetector) ScanDirectory(dirPath string) (*ChangeSet, error) {
	start := time.Now()

	changes := make([]*FileChange, 0)
	currentFiles := make(map[string]bool)

	err := filepath.Walk(dirPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			return nil
		}

		// Get relative path
		relPath, err := filepath.Rel(dirPath, path)
		if err != nil {
			return err
		}

		currentFiles[relPath] = true

		// Check if file should be ignored
		if cd.shouldIgnoreFile(relPath, info) {
			return nil
		}

		// Detect changes for this file
		change, err := cd.detectFileChange(relPath, path, info)
		if err != nil {
			return err
		}

		if change != nil {
			changes = append(changes, change)
		}

		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to scan directory: %v", err)
	}

	// Detect deleted files
	cd.mu.RLock()
	for filePath := range cd.fileHashes {
		if !currentFiles[filePath] {
			change := &FileChange{
				FilePath:   filePath,
				ChangeType: ChangeTypeDeleted,
				OldHash:    cd.fileHashes[filePath],
				DetectedAt: time.Now(),
			}
			changes = append(changes, change)
		}
	}
	cd.mu.RUnlock()

	// Update scan statistics
	cd.updateScanStats(time.Since(start))

	changeSet := &ChangeSet{
		ID:          cd.generateChangeSetID(),
		Changes:     changes,
		DetectedAt:  time.Now(),
		ScanTime:    time.Since(start),
		TriggerType: "manual",
	}

	// Update internal state
	cd.updateInternalState(changes)

	return changeSet, nil
}

// DetectChanges detects changes for specific files
func (cd *ChangeDetector) DetectChanges(filePaths []string) (*ChangeSet, error) {
	start := time.Now()
	changes := make([]*FileChange, 0)

	for _, filePath := range filePaths {
		info, err := os.Stat(filePath)
		if err != nil {
			if os.IsNotExist(err) {
				// File was deleted
				cd.mu.RLock()
				oldHash := cd.fileHashes[filePath]
				cd.mu.RUnlock()

				if oldHash != "" {
					change := &FileChange{
						FilePath:   filePath,
						ChangeType: ChangeTypeDeleted,
						OldHash:    oldHash,
						DetectedAt: time.Now(),
					}
					changes = append(changes, change)
				}
			}
			continue
		}

		if cd.shouldIgnoreFile(filePath, info) {
			continue
		}

		change, err := cd.detectFileChange(filePath, filePath, info)
		if err != nil {
			continue
		}

		if change != nil {
			changes = append(changes, change)
		}
	}

	changeSet := &ChangeSet{
		ID:          cd.generateChangeSetID(),
		Changes:     changes,
		DetectedAt:  time.Now(),
		ScanTime:    time.Since(start),
		TriggerType: "targeted",
	}

	cd.updateInternalState(changes)

	return changeSet, nil
}

// detectFileChange detects changes for a single file
func (cd *ChangeDetector) detectFileChange(relPath, fullPath string, info os.FileInfo) (*FileChange, error) {
	// Calculate current hash
	currentHash, err := cd.calculateFileHash(fullPath)
	if err != nil {
		return nil, err
	}

	cd.mu.RLock()
	oldHash := cd.fileHashes[relPath]
	oldModTime := cd.fileModTimes[relPath]
	cd.mu.RUnlock()

	var changeType ChangeType
	var change *FileChange

	if oldHash == "" {
		// New file
		changeType = ChangeTypeAdded
	} else if oldHash != currentHash {
		// File content changed
		changeType = ChangeTypeModified
	} else if cd.config.TrackMetadata && !oldModTime.Equal(info.ModTime()) {
		// Only metadata changed
		changeType = ChangeTypeModified
	} else {
		// No changes
		return nil, nil
	}

	change = &FileChange{
		FilePath:   relPath,
		ChangeType: changeType,
		OldHash:    oldHash,
		NewHash:    currentHash,
		OldModTime: oldModTime,
		NewModTime: info.ModTime(),
		FileSize:   info.Size(),
		DetectedAt: time.Now(),
		Language:   cd.detectLanguage(relPath),
		Metadata: map[string]interface{}{
			"permissions": info.Mode().String(),
			"size_change": info.Size() - cd.getOldFileSize(relPath),
		},
	}

	return change, nil
}

// calculateFileHash calculates hash for a file
func (cd *ChangeDetector) calculateFileHash(filePath string) (string, error) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return "", err
	}

	switch cd.config.HashAlgorithm {
	case "md5":
		hash := md5.Sum(content)
		return fmt.Sprintf("%x", hash), nil
	default:
		return "", fmt.Errorf("unsupported hash algorithm: %s", cd.config.HashAlgorithm)
	}
}

// shouldIgnoreFile checks if a file should be ignored
func (cd *ChangeDetector) shouldIgnoreFile(filePath string, info os.FileInfo) bool {
	// Check file size limits
	if info.Size() < cd.config.MinFileSize || info.Size() > cd.config.MaxFileSize {
		return true
	}

	// Check ignore patterns
	for _, pattern := range cd.config.IgnorePatterns {
		if strings.Contains(filePath, pattern) {
			return true
		}
	}

	// Check ignore extensions
	ext := strings.ToLower(filepath.Ext(filePath))
	for _, ignoreExt := range cd.config.IgnoreExtensions {
		if ext == ignoreExt {
			return true
		}
	}

	// Check common ignore patterns
	if strings.HasPrefix(filepath.Base(filePath), ".") {
		return true
	}

	commonIgnore := []string{
		"node_modules", ".git", "vendor", "__pycache__",
		".vscode", ".idea", "target", "dist", "build",
	}

	for _, ignore := range commonIgnore {
		if strings.Contains(filePath, ignore) {
			return true
		}
	}

	return false
}

// updateInternalState updates internal tracking state
func (cd *ChangeDetector) updateInternalState(changes []*FileChange) {
	cd.mu.Lock()
	defer cd.mu.Unlock()

	for _, change := range changes {
		switch change.ChangeType {
		case ChangeTypeAdded, ChangeTypeModified:
			cd.fileHashes[change.FilePath] = change.NewHash
			cd.fileModTimes[change.FilePath] = change.NewModTime
		case ChangeTypeDeleted:
			delete(cd.fileHashes, change.FilePath)
			delete(cd.fileModTimes, change.FilePath)
		}

		// Update statistics
		cd.changeStats.mu.Lock()
		cd.changeStats.TotalChanges++
		cd.changeStats.ChangesByType[change.ChangeType]++
		cd.changeStats.ChangesByLang[change.Language]++
		cd.changeStats.LastUpdateTime = time.Now()
		cd.changeStats.mu.Unlock()
	}
}

// updateScanStats updates scanning statistics
func (cd *ChangeDetector) updateScanStats(scanTime time.Duration) {
	cd.changeStats.mu.Lock()
	defer cd.changeStats.mu.Unlock()

	cd.changeStats.ScanCount++

	// Calculate rolling average
	if cd.changeStats.AverageScanTime == 0 {
		cd.changeStats.AverageScanTime = scanTime
	} else {
		cd.changeStats.AverageScanTime = (cd.changeStats.AverageScanTime + scanTime) / 2
	}

	cd.changeStats.FilesTracked = int64(len(cd.fileHashes))
}

// GetStatistics returns change detection statistics
func (cd *ChangeDetector) GetStatistics() *ChangeStatistics {
	cd.changeStats.mu.RLock()
	defer cd.changeStats.mu.RUnlock()

	// Create a copy to avoid race conditions
	stats := &ChangeStatistics{
		TotalChanges:    cd.changeStats.TotalChanges,
		LastUpdateTime:  cd.changeStats.LastUpdateTime,
		ScanCount:       cd.changeStats.ScanCount,
		AverageScanTime: cd.changeStats.AverageScanTime,
		FilesTracked:    cd.changeStats.FilesTracked,
		ChangesByType:   make(map[ChangeType]int64),
		ChangesByLang:   make(map[string]int64),
	}

	for k, v := range cd.changeStats.ChangesByType {
		stats.ChangesByType[k] = v
	}

	for k, v := range cd.changeStats.ChangesByLang {
		stats.ChangesByLang[k] = v
	}

	return stats
}

// GetTrackedFiles returns list of currently tracked files
func (cd *ChangeDetector) GetTrackedFiles() []string {
	cd.mu.RLock()
	defer cd.mu.RUnlock()

	files := make([]string, 0, len(cd.fileHashes))
	for filePath := range cd.fileHashes {
		files = append(files, filePath)
	}

	return files
}

// IsFileTracked checks if a file is being tracked
func (cd *ChangeDetector) IsFileTracked(filePath string) bool {
	cd.mu.RLock()
	defer cd.mu.RUnlock()

	_, exists := cd.fileHashes[filePath]
	return exists
}

// GetFileHash returns the stored hash for a file
func (cd *ChangeDetector) GetFileHash(filePath string) string {
	cd.mu.RLock()
	defer cd.mu.RUnlock()

	return cd.fileHashes[filePath]
}

// ForceUpdate forces an update of file tracking information
func (cd *ChangeDetector) ForceUpdate(filePath, hash string, modTime time.Time) {
	cd.mu.Lock()
	defer cd.mu.Unlock()

	cd.fileHashes[filePath] = hash
	cd.fileModTimes[filePath] = modTime
}

// RemoveFile removes a file from tracking
func (cd *ChangeDetector) RemoveFile(filePath string) {
	cd.mu.Lock()
	defer cd.mu.Unlock()

	delete(cd.fileHashes, filePath)
	delete(cd.fileModTimes, filePath)
}

// Helper methods

func (cd *ChangeDetector) detectLanguage(filePath string) string {
	ext := strings.ToLower(filepath.Ext(filePath))

	langMap := map[string]string{
		".go":   "go",
		".py":   "python",
		".js":   "javascript",
		".ts":   "typescript",
		".java": "java",
		".rs":   "rust",
		".cpp":  "cpp",
		".c":    "c",
		".cs":   "csharp",
		".php":  "php",
		".rb":   "ruby",
	}

	if lang, exists := langMap[ext]; exists {
		return lang
	}

	return "unknown"
}

func (cd *ChangeDetector) getOldFileSize(filePath string) int64 {
	// This would typically be stored in the database
	// For now, return 0 as a placeholder
	return 0
}

func (cd *ChangeDetector) generateChangeSetID() string {
	return fmt.Sprintf("cs_%d", time.Now().UnixNano())
}

// SaveChangesToDB persists changes to the database
func (cd *ChangeDetector) SaveChangesToDB(changeSet *ChangeSet) error {
	if len(changeSet.Changes) == 0 {
		return nil
	}

	tx, err := cd.db.BeginTx()
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %v", err)
	}
	defer tx.Rollback()

	// Insert change set
	_, err = tx.Exec(`
		INSERT INTO change_sets (id, detected_at, scan_time, trigger_type, change_count)
		VALUES (?, ?, ?, ?, ?)
	`, changeSet.ID, changeSet.DetectedAt, changeSet.ScanTime.Milliseconds(),
		changeSet.TriggerType, len(changeSet.Changes))

	if err != nil {
		return fmt.Errorf("failed to insert change set: %v", err)
	}

	// Insert individual changes
	for _, change := range changeSet.Changes {
		metadataJSON, _ := json.Marshal(change.Metadata)

		_, err = tx.Exec(`
			INSERT INTO file_changes 
			(change_set_id, file_path, change_type, old_hash, new_hash, 
			 old_mod_time, new_mod_time, file_size, detected_at, language, metadata)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		`, changeSet.ID, change.FilePath, string(change.ChangeType), change.OldHash,
			change.NewHash, change.OldModTime, change.NewModTime, change.FileSize,
			change.DetectedAt, change.Language, string(metadataJSON))

		if err != nil {
			return fmt.Errorf("failed to insert file change: %v", err)
		}
	}

	return tx.Commit()
}

// LoadChangesFromDB loads recent changes from the database
func (cd *ChangeDetector) LoadChangesFromDB(limit int) ([]*ChangeSet, error) {
	query := `
		SELECT id, detected_at, scan_time, trigger_type, change_count
		FROM change_sets
		ORDER BY detected_at DESC
		LIMIT ?
	`

	rows, err := cd.db.Query(query, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to load change sets: %v", err)
	}
	defer rows.Close()

	var changeSets []*ChangeSet
	for rows.Next() {
		var cs ChangeSet
		var scanTimeMs int64

		err := rows.Scan(&cs.ID, &cs.DetectedAt, &scanTimeMs, &cs.TriggerType, &cs.Changes)
		if err != nil {
			continue
		}

		cs.ScanTime = time.Duration(scanTimeMs) * time.Millisecond

		// Load individual changes for this change set
		changes, err := cd.loadChangesForSet(cs.ID)
		if err != nil {
			continue
		}
		cs.Changes = changes

		changeSets = append(changeSets, &cs)
	}

	return changeSets, nil
}

func (cd *ChangeDetector) loadChangesForSet(changeSetID string) ([]*FileChange, error) {
	query := `
		SELECT file_path, change_type, old_hash, new_hash, old_mod_time, 
			   new_mod_time, file_size, detected_at, language, metadata
		FROM file_changes
		WHERE change_set_id = ?
		ORDER BY detected_at
	`

	rows, err := cd.db.Query(query, changeSetID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var changes []*FileChange
	for rows.Next() {
		var change FileChange
		var metadataJSON string

		err := rows.Scan(
			&change.FilePath, &change.ChangeType, &change.OldHash, &change.NewHash,
			&change.OldModTime, &change.NewModTime, &change.FileSize,
			&change.DetectedAt, &change.Language, &metadataJSON,
		)
		if err != nil {
			continue
		}

		// Parse metadata
		if metadataJSON != "" {
			json.Unmarshal([]byte(metadataJSON), &change.Metadata)
		}

		changes = append(changes, &change)
	}

	return changes, nil
}
