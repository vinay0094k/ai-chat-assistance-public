package servers

import (
	"context"
	"crypto/md5"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/mcp"
	"github.com/yourusername/ai-code-assistant/utils"
)

// FilesystemServer provides file system operations for MCP
type FilesystemServer struct {
	allowedPaths []string
	restricted   bool
	maxFileSize  int64
}

// FilesystemConfig contains configuration for filesystem operations
type FilesystemConfig struct {
	AllowedPaths []string `json:"allowed_paths"`
	Restricted   bool     `json:"restricted"`
	MaxFileSize  int64    `json:"max_file_size"`
}

// FileInfo represents file information
type FileInfo struct {
	Path        string      `json:"path"`
	Name        string      `json:"name"`
	Size        int64       `json:"size"`
	Mode        fs.FileMode `json:"mode"`
	ModTime     time.Time   `json:"mod_time"`
	IsDir       bool        `json:"is_dir"`
	Permissions string      `json:"permissions"`
	Owner       string      `json:"owner,omitempty"`
	Group       string      `json:"group,omitempty"`
}

// SearchResult represents a file search result
type SearchResult struct {
	Path    string `json:"path"`
	Line    int    `json:"line,omitempty"`
	Content string `json:"content,omitempty"`
	Match   string `json:"match,omitempty"`
}

// NewFilesystemServer creates a new filesystem server
func NewFilesystemServer(config FilesystemConfig) *FilesystemServer {
	if config.MaxFileSize == 0 {
		config.MaxFileSize = 10 * 1024 * 1024 // 10MB default
	}

	return &FilesystemServer{
		allowedPaths: config.AllowedPaths,
		restricted:   config.Restricted,
		maxFileSize:  config.MaxFileSize,
	}
}

// GetTools returns available filesystem tools
func (fs *FilesystemServer) GetTools() []mcp.MCPTool {
	return []mcp.MCPTool{
		{
			Name:        "read_file",
			Description: "Read the contents of a file",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "Path to the file to read",
					},
					"encoding": map[string]interface{}{
						"type":        "string",
						"description": "File encoding (utf-8, binary)",
						"default":     "utf-8",
					},
				},
				"required": []string{"path"},
			},
		},
		{
			Name:        "write_file",
			Description: "Write content to a file",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "Path to the file to write",
					},
					"content": map[string]interface{}{
						"type":        "string",
						"description": "Content to write to the file",
					},
					"encoding": map[string]interface{}{
						"type":        "string",
						"description": "File encoding (utf-8, binary)",
						"default":     "utf-8",
					},
					"create_dirs": map[string]interface{}{
						"type":        "boolean",
						"description": "Create parent directories if they don't exist",
						"default":     false,
					},
				},
				"required": []string{"path", "content"},
			},
		},
		{
			Name:        "list_directory",
			Description: "List files and directories in a path",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "Path to the directory to list",
					},
					"recursive": map[string]interface{}{
						"type":        "boolean",
						"description": "List files recursively",
						"default":     false,
					},
					"include_hidden": map[string]interface{}{
						"type":        "boolean",
						"description": "Include hidden files",
						"default":     false,
					},
					"pattern": map[string]interface{}{
						"type":        "string",
						"description": "Glob pattern to filter files",
					},
				},
				"required": []string{"path"},
			},
		},
		{
			Name:        "search_files",
			Description: "Search for text within files",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "Path to search in",
					},
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Text to search for",
					},
					"regex": map[string]interface{}{
						"type":        "boolean",
						"description": "Use regular expression search",
						"default":     false,
					},
					"case_sensitive": map[string]interface{}{
						"type":        "boolean",
						"description": "Case sensitive search",
						"default":     false,
					},
					"file_pattern": map[string]interface{}{
						"type":        "string",
						"description": "Pattern to filter files to search",
					},
					"max_results": map[string]interface{}{
						"type":        "integer",
						"description": "Maximum number of results",
						"default":     100,
					},
				},
				"required": []string{"path", "query"},
			},
		},
		{
			Name:        "get_file_info",
			Description: "Get detailed information about a file or directory",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "Path to the file or directory",
					},
				},
				"required": []string{"path"},
			},
		},
		{
			Name:        "create_directory",
			Description: "Create a directory",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "Path of the directory to create",
					},
					"recursive": map[string]interface{}{
						"type":        "boolean",
						"description": "Create parent directories if needed",
						"default":     true,
					},
				},
				"required": []string{"path"},
			},
		},
		{
			Name:        "delete_file",
			Description: "Delete a file or directory",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "Path to delete",
					},
					"recursive": map[string]interface{}{
						"type":        "boolean",
						"description": "Delete directories recursively",
						"default":     false,
					},
				},
				"required": []string{"path"},
			},
		},
		{
			Name:        "copy_file",
			Description: "Copy a file or directory",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"source": map[string]interface{}{
						"type":        "string",
						"description": "Source path",
					},
					"destination": map[string]interface{}{
						"type":        "string",
						"description": "Destination path",
					},
					"overwrite": map[string]interface{}{
						"type":        "boolean",
						"description": "Overwrite destination if it exists",
						"default":     false,
					},
				},
				"required": []string{"source", "destination"},
			},
		},
		{
			Name:        "move_file",
			Description: "Move or rename a file or directory",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"source": map[string]interface{}{
						"type":        "string",
						"description": "Source path",
					},
					"destination": map[string]interface{}{
						"type":        "string",
						"description": "Destination path",
					},
				},
				"required": []string{"source", "destination"},
			},
		},
		{
			Name:        "calculate_checksum",
			Description: "Calculate MD5 checksum of a file",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "Path to the file",
					},
				},
				"required": []string{"path"},
			},
		},
	}
}

// ExecuteTool executes a filesystem tool
func (fs *FilesystemServer) ExecuteTool(ctx context.Context, toolName string, input map[string]interface{}) (interface{}, error) {
	switch toolName {
	case "read_file":
		return fs.readFile(input)
	case "write_file":
		return fs.writeFile(input)
	case "list_directory":
		return fs.listDirectory(input)
	case "search_files":
		return fs.searchFiles(input)
	case "get_file_info":
		return fs.getFileInfo(input)
	case "create_directory":
		return fs.createDirectory(input)
	case "delete_file":
		return fs.deleteFile(input)
	case "copy_file":
		return fs.copyFile(input)
	case "move_file":
		return fs.moveFile(input)
	case "calculate_checksum":
		return fs.calculateChecksum(input)
	default:
		return nil, fmt.Errorf("unknown tool: %s", toolName)
	}
}

// readFile reads a file's contents
func (fs *FilesystemServer) readFile(input map[string]interface{}) (interface{}, error) {
	path, ok := input["path"].(string)
	if !ok {
		return nil, fmt.Errorf("path is required")
	}

	if err := fs.validatePath(path); err != nil {
		return nil, err
	}

	encoding, _ := input["encoding"].(string)
	if encoding == "" {
		encoding = "utf-8"
	}

	// Check file size
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("failed to stat file: %v", err)
	}

	if info.Size() > fs.maxFileSize {
		return nil, fmt.Errorf("file too large: %d bytes (max: %d)", info.Size(), fs.maxFileSize)
	}

	if encoding == "binary" {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("failed to read file: %v", err)
		}
		return map[string]interface{}{
			"content":  data,
			"encoding": "binary",
			"size":     len(data),
		}, nil
	}

	content, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %v", err)
	}

	return map[string]interface{}{
		"content":  string(content),
		"encoding": encoding,
		"size":     len(content),
		"lines":    strings.Count(string(content), "\n") + 1,
	}, nil
}

// writeFile writes content to a file
func (fs *FilesystemServer) writeFile(input map[string]interface{}) (interface{}, error) {
	path, ok := input["path"].(string)
	if !ok {
		return nil, fmt.Errorf("path is required")
	}

	content, ok := input["content"].(string)
	if !ok {
		return nil, fmt.Errorf("content is required")
	}

	if err := fs.validatePath(path); err != nil {
		return nil, err
	}

	createDirs, _ := input["create_dirs"].(bool)
	if createDirs {
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			return nil, fmt.Errorf("failed to create directories: %v", err)
		}
	}

	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return nil, fmt.Errorf("failed to write file: %v", err)
	}

	return map[string]interface{}{
		"success":       true,
		"bytes_written": len(content),
		"path":          path,
	}, nil
}

// listDirectory lists files in a directory
func (fs *FilesystemServer) listDirectory(input map[string]interface{}) (interface{}, error) {
	path, ok := input["path"].(string)
	if !ok {
		return nil, fmt.Errorf("path is required")
	}

	if err := fs.validatePath(path); err != nil {
		return nil, err
	}

	recursive, _ := input["recursive"].(bool)
	includeHidden, _ := input["include_hidden"].(bool)
	pattern, _ := input["pattern"].(string)

	var files []FileInfo
	var err error

	if recursive {
		err = filepath.WalkDir(path, func(filePath string, d fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}

			if !includeHidden && strings.HasPrefix(d.Name(), ".") {
				if d.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}

			if pattern != "" {
				matched, matchErr := filepath.Match(pattern, d.Name())
				if matchErr != nil {
					return matchErr
				}
				if !matched {
					return nil
				}
			}

			info, err := d.Info()
			if err != nil {
				return err
			}

			files = append(files, fs.buildFileInfo(filePath, info))
			return nil
		})
	} else {
		entries, readErr := os.ReadDir(path)
		if readErr != nil {
			return nil, fmt.Errorf("failed to read directory: %v", readErr)
		}

		for _, entry := range entries {
			if !includeHidden && strings.HasPrefix(entry.Name(), ".") {
				continue
			}

			if pattern != "" {
				matched, matchErr := filepath.Match(pattern, entry.Name())
				if matchErr != nil {
					continue
				}
				if !matched {
					continue
				}
			}

			info, infoErr := entry.Info()
			if infoErr != nil {
				continue
			}

			filePath := filepath.Join(path, entry.Name())
			files = append(files, fs.buildFileInfo(filePath, info))
		}
	}

	if err != nil {
		return nil, fmt.Errorf("failed to list directory: %v", err)
	}

	// Sort files by name
	sort.Slice(files, func(i, j int) bool {
		return files[i].Name < files[j].Name
	})

	return map[string]interface{}{
		"files": files,
		"count": len(files),
		"path":  path,
	}, nil
}

// searchFiles searches for text within files
func (fs *FilesystemServer) searchFiles(input map[string]interface{}) (interface{}, error) {
	path, ok := input["path"].(string)
	if !ok {
		return nil, fmt.Errorf("path is required")
	}

	query, ok := input["query"].(string)
	if !ok {
		return nil, fmt.Errorf("query is required")
	}

	if err := fs.validatePath(path); err != nil {
		return nil, err
	}

	useRegex, _ := input["regex"].(bool)
	caseSensitive, _ := input["case_sensitive"].(bool)
	filePattern, _ := input["file_pattern"].(string)
	maxResults := 100
	if max, ok := input["max_results"].(float64); ok {
		maxResults = int(max)
	}

	var searchRegex *regexp.Regexp
	var err error

	if useRegex {
		flags := ""
		if !caseSensitive {
			flags = "(?i)"
		}
		searchRegex, err = regexp.Compile(flags + query)
		if err != nil {
			return nil, fmt.Errorf("invalid regex: %v", err)
		}
	} else if !caseSensitive {
		query = strings.ToLower(query)
	}

	var results []SearchResult
	resultCount := 0

	err = filepath.WalkDir(path, func(filePath string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil || resultCount >= maxResults {
			return walkErr
		}

		if d.IsDir() || strings.HasPrefix(d.Name(), ".") {
			return nil
		}

		if filePattern != "" {
			matched, matchErr := filepath.Match(filePattern, d.Name())
			if matchErr != nil || !matched {
				return nil
			}
		}

		// Only search text files
		if !utils.IsTextFile(filePath) {
			return nil
		}

		info, err := d.Info()
		if err != nil || info.Size() > fs.maxFileSize {
			return nil
		}

		content, err := os.ReadFile(filePath)
		if err != nil {
			return nil
		}

		lines := strings.Split(string(content), "\n")
		for lineNum, line := range lines {
			if resultCount >= maxResults {
				break
			}

			var matched bool
			var match string

			if useRegex {
				if loc := searchRegex.FindStringIndex(line); loc != nil {
					matched = true
					match = line[loc[0]:loc[1]]
				}
			} else {
				searchLine := line
				if !caseSensitive {
					searchLine = strings.ToLower(line)
				}
				if strings.Contains(searchLine, query) {
					matched = true
					match = query
				}
			}

			if matched {
				results = append(results, SearchResult{
					Path:    filePath,
					Line:    lineNum + 1,
					Content: strings.TrimSpace(line),
					Match:   match,
				})
				resultCount++
			}
		}

		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("search failed: %v", err)
	}

	return map[string]interface{}{
		"results": results,
		"count":   len(results),
		"query":   query,
		"path":    path,
	}, nil
}

// getFileInfo gets detailed file information
func (fs *FilesystemServer) getFileInfo(input map[string]interface{}) (interface{}, error) {
	path, ok := input["path"].(string)
	if !ok {
		return nil, fmt.Errorf("path is required")
	}

	if err := fs.validatePath(path); err != nil {
		return nil, err
	}

	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("failed to get file info: %v", err)
	}

	fileInfo := fs.buildFileInfo(path, info)

	// Add additional information
	result := map[string]interface{}{
		"path":        fileInfo.Path,
		"name":        fileInfo.Name,
		"size":        fileInfo.Size,
		"mode":        fileInfo.Mode.String(),
		"mod_time":    fileInfo.ModTime,
		"is_dir":      fileInfo.IsDir,
		"permissions": fileInfo.Permissions,
	}

	if !fileInfo.IsDir {
		// Add file-specific info
		ext := filepath.Ext(path)
		result["extension"] = ext
		result["language"] = utils.GetFileLanguage(path)
		result["is_text"] = utils.IsTextFile(path)

		// Get line count for text files
		if utils.IsTextFile(path) && info.Size() < fs.maxFileSize {
			if lineCount, err := utils.CountLines(path); err == nil {
				result["line_count"] = lineCount
			}
		}
	} else {
		// Add directory-specific info
		if entries, err := os.ReadDir(path); err == nil {
			fileCount := 0
			dirCount := 0
			for _, entry := range entries {
				if entry.IsDir() {
					dirCount++
				} else {
					fileCount++
				}
			}
			result["file_count"] = fileCount
			result["dir_count"] = dirCount
		}
	}

	return result, nil
}

// createDirectory creates a directory
func (fs *FilesystemServer) createDirectory(input map[string]interface{}) (interface{}, error) {
	path, ok := input["path"].(string)
	if !ok {
		return nil, fmt.Errorf("path is required")
	}

	if err := fs.validatePath(path); err != nil {
		return nil, err
	}

	recursive, _ := input["recursive"].(bool)

	var err error
	if recursive {
		err = os.MkdirAll(path, 0755)
	} else {
		err = os.Mkdir(path, 0755)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to create directory: %v", err)
	}

	return map[string]interface{}{
		"success": true,
		"path":    path,
	}, nil
}

// deleteFile deletes a file or directory
func (fs *FilesystemServer) deleteFile(input map[string]interface{}) (interface{}, error) {
	path, ok := input["path"].(string)
	if !ok {
		return nil, fmt.Errorf("path is required")
	}

	if err := fs.validatePath(path); err != nil {
		return nil, err
	}

	recursive, _ := input["recursive"].(bool)

	var err error
	if recursive {
		err = os.RemoveAll(path)
	} else {
		err = os.Remove(path)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to delete: %v", err)
	}

	return map[string]interface{}{
		"success": true,
		"path":    path,
	}, nil
}

// copyFile copies a file or directory
func (fs *FilesystemServer) copyFile(input map[string]interface{}) (interface{}, error) {
	source, ok := input["source"].(string)
	if !ok {
		return nil, fmt.Errorf("source is required")
	}

	destination, ok := input["destination"].(string)
	if !ok {
		return nil, fmt.Errorf("destination is required")
	}

	if err := fs.validatePath(source); err != nil {
		return nil, fmt.Errorf("source path invalid: %v", err)
	}

	if err := fs.validatePath(destination); err != nil {
		return nil, fmt.Errorf("destination path invalid: %v", err)
	}

	overwrite, _ := input["overwrite"].(bool)

	// Check if destination exists
	if _, err := os.Stat(destination); err == nil && !overwrite {
		return nil, fmt.Errorf("destination exists and overwrite is false")
	}

	if err := utils.CopyFile(source, destination); err != nil {
		return nil, fmt.Errorf("failed to copy: %v", err)
	}

	return map[string]interface{}{
		"success":     true,
		"source":      source,
		"destination": destination,
	}, nil
}

// moveFile moves or renames a file
func (fs *FilesystemServer) moveFile(input map[string]interface{}) (interface{}, error) {
	source, ok := input["source"].(string)
	if !ok {
		return nil, fmt.Errorf("source is required")
	}

	destination, ok := input["destination"].(string)
	if !ok {
		return nil, fmt.Errorf("destination is required")
	}

	if err := fs.validatePath(source); err != nil {
		return nil, fmt.Errorf("source path invalid: %v", err)
	}

	if err := fs.validatePath(destination); err != nil {
		return nil, fmt.Errorf("destination path invalid: %v", err)
	}

	if err := os.Rename(source, destination); err != nil {
		return nil, fmt.Errorf("failed to move: %v", err)
	}

	return map[string]interface{}{
		"success":     true,
		"source":      source,
		"destination": destination,
	}, nil
}

// calculateChecksum calculates MD5 checksum
func (fs *FilesystemServer) calculateChecksum(input map[string]interface{}) (interface{}, error) {
	path, ok := input["path"].(string)
	if !ok {
		return nil, fmt.Errorf("path is required")
	}

	if err := fs.validatePath(path); err != nil {
		return nil, err
	}

	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	hash := md5.New()
	if _, err := io.Copy(hash, file); err != nil {
		return nil, fmt.Errorf("failed to calculate checksum: %v", err)
	}

	checksum := fmt.Sprintf("%x", hash.Sum(nil))

	return map[string]interface{}{
		"checksum":  checksum,
		"path":      path,
		"algorithm": "md5",
	}, nil
}

// validatePath validates if a path is allowed
func (fs *FilesystemServer) validatePath(path string) error {
	if !fs.restricted {
		return nil
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return fmt.Errorf("invalid path: %v", err)
	}

	// Clean the path to prevent directory traversal
	cleanPath := filepath.Clean(absPath)

	for _, allowedPath := range fs.allowedPaths {
		allowedAbs, err := filepath.Abs(allowedPath)
		if err != nil {
			continue
		}

		if strings.HasPrefix(cleanPath, filepath.Clean(allowedAbs)) {
			return nil
		}
	}

	return fmt.Errorf("path not allowed: %s", path)
}

// buildFileInfo builds FileInfo from os.FileInfo
func (fs *FilesystemServer) buildFileInfo(path string, info os.FileInfo) FileInfo {
	return FileInfo{
		Path:        path,
		Name:        info.Name(),
		Size:        info.Size(),
		Mode:        info.Mode(),
		ModTime:     info.ModTime(),
		IsDir:       info.IsDir(),
		Permissions: info.Mode().Perm().String(),
	}
}
