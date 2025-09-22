package utils

import (
	"crypto/md5"
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// FileInfo represents detailed file information
type FileInfo struct {
	Path         string      `json:"path"`
	RelativePath string      `json:"relative_path"`
	Name         string      `json:"name"`
	Extension    string      `json:"extension"`
	Size         int64       `json:"size"`
	Mode         os.FileMode `json:"mode"`
	ModTime      time.Time   `json:"mod_time"`
	IsDir        bool        `json:"is_dir"`
	Hash         string      `json:"hash,omitempty"`
	Language     string      `json:"language,omitempty"`
	Depth        int         `json:"depth,omitempty"`
}

// WalkOptions represents options for walking directories
type WalkOptions struct {
	IncludePatterns []string // File patterns to include
	ExcludePatterns []string // File patterns to exclude
	MaxDepth        int      // Maximum directory depth
	FollowSymlinks  bool     // Whether to follow symbolic links
	IncludeHidden   bool     // Whether to include hidden files
	MaxFileSize     int64    // Maximum file size in bytes
}

// FileStats represents statistics about files
type FileStats struct {
	TotalFiles       int            `json:"total_files"`
	TotalSize        int64          `json:"total_size"`
	TotalDirectories int            `json:"total_directories"`
	LanguageStats    map[string]int `json:"language_stats"`
	ExtensionStats   map[string]int `json:"extension_stats"`
	SizeDistribution map[string]int `json:"size_distribution"`
	LargestFiles     []*FileInfo    `json:"largest_files"`
}

// GetFileInfo returns detailed information about a file
func GetFileInfo(path string) (*FileInfo, error) {
	stat, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("failed to stat file %s: %v", path, err)
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("failed to get absolute path: %v", err)
	}

	info := &FileInfo{
		Path:      absPath,
		Name:      stat.Name(),
		Extension: filepath.Ext(stat.Name()),
		Size:      stat.Size(),
		Mode:      stat.Mode(),
		ModTime:   stat.ModTime(),
		IsDir:     stat.IsDir(),
	}

	// Set relative path if possible
	if wd, err := os.Getwd(); err == nil {
		if rel, err := filepath.Rel(wd, absPath); err == nil {
			info.RelativePath = rel
		}
	}

	// Detect language based on file extension
	if !info.IsDir {
		info.Language = DetectLanguage(info.Extension)
	}

	return info, nil
}

// GetFileHash calculates the hash of a file
func GetFileHash(path string, algorithm string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	var hasher io.Writer
	switch strings.ToLower(algorithm) {
	case "md5":
		h := md5.New()
		hasher = h
		if _, err := io.Copy(hasher, file); err != nil {
			return "", fmt.Errorf("failed to calculate MD5: %v", err)
		}
		return fmt.Sprintf("%x", h.Sum(nil)), nil
	case "sha256":
		h := sha256.New()
		hasher = h
		if _, err := io.Copy(hasher, file); err != nil {
			return "", fmt.Errorf("failed to calculate SHA256: %v", err)
		}
		return fmt.Sprintf("%x", h.Sum(nil)), nil
	default:
		return "", fmt.Errorf("unsupported hash algorithm: %s", algorithm)
	}
}

// WalkFiles walks through a directory tree and returns file information
func WalkFiles(root string, options *WalkOptions) ([]*FileInfo, error) {
	if options == nil {
		options = &WalkOptions{
			MaxDepth:       -1, // No limit
			FollowSymlinks: false,
			IncludeHidden:  false,
			MaxFileSize:    -1, // No limit
		}
	}

	var files []*FileInfo

	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Calculate current depth
		rel, _ := filepath.Rel(root, path)
		depth := strings.Count(rel, string(filepath.Separator))
		if path == root {
			depth = 0
		}

		// Check max depth
		if options.MaxDepth >= 0 && depth > options.MaxDepth {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		// Skip hidden files if not included
		if !options.IncludeHidden && strings.HasPrefix(info.Name(), ".") {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		// Check file size limits
		if !info.IsDir() && options.MaxFileSize > 0 && info.Size() > options.MaxFileSize {
			return nil
		}

		// Check include patterns
		if len(options.IncludePatterns) > 0 {
			matched := false
			for _, pattern := range options.IncludePatterns {
				if ok, _ := filepath.Match(pattern, info.Name()); ok {
					matched = true
					break
				}
			}
			if !matched {
				return nil
			}
		}

		// Check exclude patterns
		for _, pattern := range options.ExcludePatterns {
			if ok, _ := filepath.Match(pattern, info.Name()); ok {
				if info.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}
		}

		// Create FileInfo
		fileInfo := &FileInfo{
			Path:      path,
			Name:      info.Name(),
			Extension: filepath.Ext(info.Name()),
			Size:      info.Size(),
			Mode:      info.Mode(),
			ModTime:   info.ModTime(),
			IsDir:     info.IsDir(),
			Depth:     depth, // <--- assign depth
		}

		// Set relative path
		if rel, err := filepath.Rel(root, path); err == nil {
			fileInfo.RelativePath = rel
		}

		// Detect language for files
		if !fileInfo.IsDir {
			fileInfo.Language = DetectLanguage(fileInfo.Extension)
		}

		files = append(files, fileInfo)
		return nil
	})

	return files, err
}

// DetectLanguage detects programming language based on file extension
func DetectLanguage(extension string) string {
	ext := strings.ToLower(extension)

	languageMap := map[string]string{
		".go":    "go",
		".py":    "python",
		".js":    "javascript",
		".ts":    "typescript",
		".java":  "java",
		".c":     "c",
		".cpp":   "cpp",
		".cc":    "cpp",
		".cxx":   "cpp",
		".h":     "c",
		".hpp":   "cpp",
		".cs":    "csharp",
		".php":   "php",
		".rb":    "ruby",
		".rs":    "rust",
		".swift": "swift",
		".kt":    "kotlin",
		".scala": "scala",
		".clj":   "clojure",
		".hs":    "haskell",
		".ml":    "ocaml",
		".fs":    "fsharp",
		".ex":    "elixir",
		".exs":   "elixir",
		".erl":   "erlang",
		".r":     "r",
		".m":     "matlab",
		".pl":    "perl",
		".sh":    "shell",
		".bash":  "shell",
		".zsh":   "shell",
		".fish":  "shell",
		".ps1":   "powershell",
		".html":  "html",
		".htm":   "html",
		".css":   "css",
		".scss":  "scss",
		".sass":  "sass",
		".less":  "less",
		".xml":   "xml",
		".json":  "json",
		".yaml":  "yaml",
		".yml":   "yaml",
		".toml":  "toml",
		".ini":   "ini",
		".cfg":   "ini",
		".conf":  "config",
		".sql":   "sql",
		".md":    "markdown",
		".tex":   "latex",
		".vim":   "vim",
		".lua":   "lua",
		".dart":  "dart",
		".jl":    "julia",
		".nim":   "nim",
		".zig":   "zig",
		".v":     "vlang",
		".cr":    "crystal",
		".elm":   "elm",
		".purs":  "purescript",
		".re":    "reason",
		".f90":   "fortran",
		".f95":   "fortran",
		".f03":   "fortran",
		".f08":   "fortran",
		".ada":   "ada",
		".adb":   "ada",
		".ads":   "ada",
		".pas":   "pascal",
		".pp":    "pascal",
		".d":     "d",
		".vb":    "vb",
		".vbs":   "vbscript",
		".asm":   "assembly",
		".s":     "assembly",
	}

	if lang, exists := languageMap[ext]; exists {
		return lang
	}

	return "unknown"
}

// IsCodeFile checks if a file is a code file based on extension
func IsCodeFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	language := DetectLanguage(ext)
	return language != "unknown"
}

// IsTextFile checks if a file is likely a text file
func IsTextFile(path string) bool {
	// Check by extension first
	if IsCodeFile(path) {
		return true
	}

	textExtensions := []string{
		".txt", ".md", ".rst", ".asciidoc", ".tex", ".rtf",
		".csv", ".tsv", ".log", ".conf", ".cfg", ".ini",
		".env", ".properties", ".gitignore", ".gitattributes",
		".dockerfile", ".makefile", ".cmake", ".gradle",
		".pom", ".sbt", ".cabal", ".stack", ".cargo",
	}

	ext := strings.ToLower(filepath.Ext(path))
	for _, textExt := range textExtensions {
		if ext == textExt {
			return true
		}
	}

	// Check common text file names without extensions
	baseName := strings.ToLower(filepath.Base(path))
	textFiles := []string{
		"readme", "license", "changelog", "contributing",
		"makefile", "dockerfile", "vagrantfile", "gemfile",
		"podfile", "brewfile", "procfile", "cmakelists",
	}

	for _, textFile := range textFiles {
		if baseName == textFile {
			return true
		}
	}

	return false
}

// GetProjectFiles filters files to only include project-relevant files
func GetProjectFiles(files []*FileInfo) []*FileInfo {
	var projectFiles []*FileInfo

	for _, file := range files {
		if file.IsDir {
			continue
		}

		// Include code files
		if IsCodeFile(file.Path) {
			projectFiles = append(projectFiles, file)
			continue
		}

		// Include important project files
		if IsTextFile(file.Path) {
			projectFiles = append(projectFiles, file)
			continue
		}

		// Include configuration files
		if IsConfigFile(file.Path) {
			projectFiles = append(projectFiles, file)
			continue
		}
	}

	return projectFiles
}

// IsConfigFile checks if a file is a configuration file
func IsConfigFile(path string) bool {
	configExtensions := []string{
		".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
		".properties", ".env", ".config", ".settings", ".prefs",
	}

	ext := strings.ToLower(filepath.Ext(path))
	for _, configExt := range configExtensions {
		if ext == configExt {
			return true
		}
	}

	// Check config file names
	baseName := strings.ToLower(filepath.Base(path))
	configFiles := []string{
		".editorconfig", ".gitignore", ".gitattributes", ".eslintrc",
		".prettierrc", ".babelrc", ".npmrc", ".yarnrc", ".nvmrc",
		"tsconfig.json", "package.json", "composer.json", "pom.xml",
		"build.gradle", "cargo.toml", "go.mod", "requirements.txt",
		"pipfile", "poetry.lock", "gemfile", "rakefile",
	}

	for _, configFile := range configFiles {
		if baseName == configFile || strings.HasSuffix(baseName, configFile) {
			return true
		}
	}

	return false
}

// CalculateFileStats calculates statistics for a collection of files
func CalculateFileStats(files []*FileInfo) *FileStats {
	stats := &FileStats{
		LanguageStats:    make(map[string]int),
		ExtensionStats:   make(map[string]int),
		SizeDistribution: make(map[string]int),
		LargestFiles:     make([]*FileInfo, 0),
	}

	var allFiles []*FileInfo

	for _, file := range files {
		if file.IsDir {
			stats.TotalDirectories++
			continue
		}

		stats.TotalFiles++
		stats.TotalSize += file.Size
		allFiles = append(allFiles, file)

		// Language statistics
		if file.Language != "" && file.Language != "unknown" {
			stats.LanguageStats[file.Language]++
		}

		// Extension statistics
		ext := strings.ToLower(file.Extension)
		if ext != "" {
			stats.ExtensionStats[ext]++
		}

		// Size distribution
		sizeCategory := categorizeFileSize(file.Size)
		stats.SizeDistribution[sizeCategory]++
	}

	// Find largest files
	sort.Slice(allFiles, func(i, j int) bool {
		return allFiles[i].Size > allFiles[j].Size
	})

	maxLargest := 10
	if len(allFiles) < maxLargest {
		maxLargest = len(allFiles)
	}
	stats.LargestFiles = allFiles[:maxLargest]

	return stats
}

// categorizeFileSize categorizes file size into buckets
func categorizeFileSize(size int64) string {
	const (
		KB = 1024
		MB = KB * 1024
		GB = MB * 1024
	)

	switch {
	case size < KB:
		return "< 1KB"
	case size < 10*KB:
		return "1KB - 10KB"
	case size < 100*KB:
		return "10KB - 100KB"
	case size < MB:
		return "100KB - 1MB"
	case size < 10*MB:
		return "1MB - 10MB"
	case size < 100*MB:
		return "10MB - 100MB"
	case size < GB:
		return "100MB - 1GB"
	default:
		return "> 1GB"
	}
}

// ReadFileContent reads the content of a file safely
func ReadFileContent(path string, maxSize int64) ([]byte, error) {
	stat, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("failed to stat file: %v", err)
	}

	if maxSize > 0 && stat.Size() > maxSize {
		return nil, fmt.Errorf("file size %d exceeds maximum %d", stat.Size(), maxSize)
	}

	content, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %v", err)
	}

	return content, nil
}

// WriteFileContent writes content to a file safely
func WriteFileContent(path string, content []byte, createDirs bool) error {
	if createDirs {
		dir := filepath.Dir(path)
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("failed to create directory: %v", err)
		}
	}

	if err := os.WriteFile(path, content, 0644); err != nil {
		return fmt.Errorf("failed to write file: %v", err)
	}

	return nil
}

// CopyFile copies a file from src to dst
func CopyFile(src, dst string) error {
	sourceFile, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("failed to open source file: %v", err)
	}
	defer sourceFile.Close()

	// Create destination directory if needed
	if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
		return fmt.Errorf("failed to create destination directory: %v", err)
	}

	destFile, err := os.Create(dst)
	if err != nil {
		return fmt.Errorf("failed to create destination file: %v", err)
	}
	defer destFile.Close()

	if _, err := io.Copy(destFile, sourceFile); err != nil {
		return fmt.Errorf("failed to copy file content: %v", err)
	}

	// Copy file permissions
	sourceInfo, err := os.Stat(src)
	if err != nil {
		return fmt.Errorf("failed to get source file info: %v", err)
	}

	if err := os.Chmod(dst, sourceInfo.Mode()); err != nil {
		return fmt.Errorf("failed to set file permissions: %v", err)
	}

	return nil
}

// BackupFile creates a backup of a file with timestamp
func BackupFile(path string, backupDir string) (string, error) {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return "", fmt.Errorf("source file does not exist: %s", path)
	}

	// Create backup directory
	if err := os.MkdirAll(backupDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create backup directory: %v", err)
	}

	// Generate backup filename with timestamp
	timestamp := time.Now().Format("20060102_150405")
	baseName := filepath.Base(path)
	ext := filepath.Ext(baseName)
	nameWithoutExt := strings.TrimSuffix(baseName, ext)

	backupName := fmt.Sprintf("%s_%s%s", nameWithoutExt, timestamp, ext)
	backupPath := filepath.Join(backupDir, backupName)

	// Copy file to backup location
	if err := CopyFile(path, backupPath); err != nil {
		return "", fmt.Errorf("failed to create backup: %v", err)
	}

	return backupPath, nil
}

// CleanupOldBackups removes old backup files, keeping only the newest ones
func CleanupOldBackups(backupDir string, maxBackups int) error {
	files, err := os.ReadDir(backupDir)
	if err != nil {
		return fmt.Errorf("failed to read backup directory: %v", err)
	}

	if len(files) <= maxBackups {
		return nil // No cleanup needed
	}

	// Sort files by modification time (newest first)
	var fileInfos []os.FileInfo
	for _, file := range files {
		if file.IsDir() {
			continue
		}
		info, err := file.Info()
		if err != nil {
			continue
		}
		fileInfos = append(fileInfos, info)
	}

	sort.Slice(fileInfos, func(i, j int) bool {
		return fileInfos[i].ModTime().After(fileInfos[j].ModTime())
	})

	// Remove old files
	for i := maxBackups; i < len(fileInfos); i++ {
		oldFile := filepath.Join(backupDir, fileInfos[i].Name())
		if err := os.Remove(oldFile); err != nil {
			return fmt.Errorf("failed to remove old backup %s: %v", oldFile, err)
		}
	}

	return nil
}

// FindFiles finds files matching a pattern
func FindFiles(root string, pattern string, recursive bool) ([]string, error) {
	var matches []string

	if recursive {
		err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}

			if info.IsDir() {
				return nil
			}

			matched, err := filepath.Match(pattern, filepath.Base(path))
			if err != nil {
				return err
			}

			if matched {
				matches = append(matches, path)
			}

			return nil
		})
		return matches, err
	} else {
		entries, err := os.ReadDir(root)
		if err != nil {
			return nil, err
		}

		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}

			matched, err := filepath.Match(pattern, entry.Name())
			if err != nil {
				return nil, err
			}

			if matched {
				matches = append(matches, filepath.Join(root, entry.Name()))
			}
		}
	}

	return matches, nil
}

// EnsureDirectory ensures a directory exists, creating it if necessary
func EnsureDirectory(path string) error {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return os.MkdirAll(path, 0755)
	}
	return nil
}

// GetDirectorySize calculates the total size of a directory
func GetDirectorySize(path string) (int64, error) {
	var size int64

	err := filepath.Walk(path, func(filePath string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			size += info.Size()
		}
		return nil
	})

	return size, err
}

// IsEmptyDirectory checks if a directory is empty
func IsEmptyDirectory(path string) (bool, error) {
	entries, err := os.ReadDir(path)
	if err != nil {
		return false, err
	}
	return len(entries) == 0, nil
}
