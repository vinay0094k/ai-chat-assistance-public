package cli

import (
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/app"
	"github.com/yourusername/ai-code-assistant/internal/display"
	"github.com/yourusername/ai-code-assistant/internal/logger"
	"github.com/yourusername/ai-code-assistant/models"
	"github.com/yourusername/ai-code-assistant/storage"
)

// Session represents a CLI session
type Session struct {
	ID           string                 `json:"id"`
	StartTime    time.Time              `json:"start_time"`
	Config       *app.SystemConfig      `json:"-"`
	CLIConfig    *app.CLISpecificConfig `json:"-"`
	DB           *storage.SQLiteDB      `json:"-"`
	Logger       *logger.Logger         `json:"-"`
	Display      *display.Manager       `json:"-"`
	History      *History               `json:"-"`
	CommandCount int                    `json:"command_count"`
	Context      map[string]interface{} `json:"context"`
	Variables    map[string]string      `json:"variables"`
	WorkingDir   string                 `json:"working_dir"`
	ProjectPath  string                 `json:"project_path,omitempty"`
	mutex        sync.RWMutex           `json:"-"`
	closed       bool                   `json:"-"`
}

// NewSession creates a new CLI session
func NewSession(id string, config *app.SystemConfig, cliConfig *app.CLISpecificConfig, db *storage.SQLiteDB, log *logger.Logger, display *display.Manager) *Session {
	session := &Session{
		ID:        id,
		StartTime: time.Now(),
		Config:    config,
		CLIConfig: cliConfig,
		DB:        db,
		Logger:    log,
		Display:   display,
		Context:   make(map[string]interface{}),
		Variables: make(map[string]string),
	}

	// Initialize command history
	session.History = NewHistory(cliConfig.CLI.History.MaxEntries)

	// Set working directory
	if wd, err := os.Getwd(); err == nil {
		session.WorkingDir = wd
	}

	// Save session to database
	session.saveToDatabase()

	return session
}

// GetContext returns a context value
func (s *Session) GetContext(key string) (interface{}, bool) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	value, exists := s.Context[key]
	return value, exists
}

// SetContext sets a context value
func (s *Session) SetContext(key string, value interface{}) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	s.Context[key] = value
}

// GetVariable returns a session variable
func (s *Session) GetVariable(name string) (string, bool) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	value, exists := s.Variables[name]
	return value, exists
}

// SetVariable sets a session variable
func (s *Session) SetVariable(name, value string) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	s.Variables[name] = value
}

// IncrementCommandCount increments the command counter
func (s *Session) IncrementCommandCount() {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	s.CommandCount++
}

// GetUptime returns session uptime
func (s *Session) GetUptime() time.Duration {
	return time.Since(s.StartTime)
}

// AddToHistory adds a command to the session history
func (s *Session) AddToHistory(command string) {
	if s.History != nil {
		s.History.Add(command)
	}
}

// GetHistory returns the session history
func (s *Session) GetHistory() []string {
	if s.History != nil {
		return s.History.GetAll()
	}
	return nil
}

// Close closes the session and performs cleanup
func (s *Session) Close() error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if s.closed {
		return nil
	}

	s.closed = true

	// Save final session state
	s.updateInDatabase()

	// Log session end
	if s.Logger != nil {
		s.Logger.Info("Session ended", map[string]interface{}{
			"session_id":    s.ID,
			"duration":      s.GetUptime().String(),
			"command_count": s.CommandCount,
		})
	}

	return nil
}

// saveToDatabase saves the session to database
func (s *Session) saveToDatabase() {
	if s.DB == nil {
		return
	}

	query := `
		INSERT OR REPLACE INTO sessions 
		(id, start_time, total_queries, created_at, updated_at)
		VALUES (?, ?, ?, ?, ?)
	`

	now := time.Now()
	s.DB.Execute(query, s.ID, s.StartTime, s.CommandCount, now, now)
}

// updateInDatabase updates the session in database
func (s *Session) updateInDatabase() {
	if s.DB == nil {
		return
	}

	query := `
		UPDATE sessions 
		SET total_queries = ?, end_time = ?, updated_at = ?
		WHERE id = ?
	`

	now := time.Now()
	s.DB.Execute(query, s.CommandCount, now, now, s.ID)
}

// IsProjectDirectory checks if current directory is a project
func (s *Session) IsProjectDirectory() bool {
	if s.WorkingDir == "" {
		return false
	}

	// Check for common project files
	projectFiles := []string{
		"go.mod", "package.json", "requirements.txt",
		"Cargo.toml", "pom.xml", "composer.json",
	}

	for _, file := range projectFiles {
		if _, err := os.Stat(filepath.Join(s.WorkingDir, file)); err == nil {
			return true
		}
	}

	return false
}

// DetectProjectType detects the type of project
func (s *Session) DetectProjectType() string {
	if s.WorkingDir == "" {
		return "unknown"
	}

	projectTypes := map[string]string{
		"go.mod":           "go",
		"package.json":     "javascript",
		"requirements.txt": "python",
		"Cargo.toml":       "rust",
		"pom.xml":          "java",
		"composer.json":    "php",
	}

	for file, projectType := range projectTypes {
		if _, err := os.Stat(filepath.Join(s.WorkingDir, file)); err == nil {
			return projectType
		}
	}

	return "unknown"
}

// GetSessionInfo returns session information
func (s *Session) GetSessionInfo() *models.SessionInfo {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	return &models.SessionInfo{
		ID:           s.ID,
		StartTime:    s.StartTime,
		Uptime:       s.GetUptime(),
		CommandCount: s.CommandCount,
		WorkingDir:   s.WorkingDir,
		ProjectPath:  s.ProjectPath,
		ProjectType:  s.DetectProjectType(),
		Variables:    s.Variables,
	}
}
