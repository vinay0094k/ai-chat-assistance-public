package models

import (
	"time"
)

// DisplayConfig represents display configuration
type DisplayConfig struct {
	Colors      *ColorConfig              `json:"colors"`
	Streaming   *StreamingConfig          `json:"streaming"`
	LineNumbers *LineNumberConfig         `json:"line_numbers"`
	Progress    *ProgressConfig           `json:"progress"`
	Syntax      *SyntaxConfig             `json:"syntax"`
	Layout      *LayoutConfig             `json:"layout"`
	Tokens      *TokenDisplayConfig       `json:"tokens"`
	Performance *PerformanceDisplayConfig `json:"performance"`
}

// ColorConfig represents color configuration
type ColorConfig struct {
	Enabled      bool              `json:"enabled"`
	Theme        string            `json:"theme"`
	Primary      string            `json:"primary"`
	Secondary    string            `json:"secondary"`
	Warning      string            `json:"warning"`
	Error        string            `json:"error"`
	Success      string            `json:"success"`
	Info         string            `json:"info"`
	Debug        string            `json:"debug"`
	CustomColors map[string]string `json:"custom_colors,omitempty"`
}

// StreamingConfig represents streaming display configuration
type StreamingConfig struct {
	Enabled             bool              `json:"enabled"`
	CharacterDelay      int               `json:"character_delay"` // ms
	LineDelay           int               `json:"line_delay"`      // ms
	ImportantEmphasis   bool              `json:"important_emphasis"`
	AdaptiveSpeed       bool              `json:"adaptive_speed"`
	SpeedByContent      map[string]string `json:"speed_by_content"`
	ShowTypingIndicator bool              `json:"show_typing_indicator"`
}

// LineNumberConfig represents line number display configuration
type LineNumberConfig struct {
	Enabled   bool   `json:"enabled"`
	Width     int    `json:"width"`
	Style     string `json:"style"` // aligned, padded, minimal
	Color     string `json:"color"`
	Separator string `json:"separator"`
	Relative  bool   `json:"relative"` // Show relative line numbers
}

// ProgressConfig represents progress display configuration
type ProgressConfig struct {
	Enabled        bool              `json:"enabled"`
	BarStyle       string            `json:"bar_style"` // blocks, dots, arrow, unicode
	BarWidth       int               `json:"bar_width"`
	ShowPercentage bool              `json:"show_percentage"`
	ShowETA        bool              `json:"show_eta"`
	ShowSpeed      bool              `json:"show_speed"`
	Theme          map[string]string `json:"theme"`
	Animation      *AnimationConfig  `json:"animation,omitempty"`
}

// AnimationConfig represents animation configuration
type AnimationConfig struct {
	Enabled    bool     `json:"enabled"`
	Type       string   `json:"type"`       // spinner, dots, bars, pulse
	Speed      int      `json:"speed"`      // ms between frames
	Characters []string `json:"characters"` // Animation frames
}

// SyntaxConfig represents syntax highlighting configuration
type SyntaxConfig struct {
	Enabled   bool                       `json:"enabled"`
	Theme     string                     `json:"theme"`
	Languages map[string]*LanguageConfig `json:"languages"`
}

// LanguageConfig represents language-specific syntax configuration
type LanguageConfig struct {
	Enabled  bool              `json:"enabled"`
	Keywords []string          `json:"keywords"`
	Types    []string          `json:"types"`
	Colors   map[string]string `json:"colors"`   // keyword_type -> color
	Patterns map[string]string `json:"patterns"` // pattern_name -> regex
}

// LayoutConfig represents layout configuration
type LayoutConfig struct {
	MaxWidth   int              `json:"max_width"`
	IndentSize int              `json:"indent_size"`
	Margins    *MarginConfig    `json:"margins"`
	Separators *SeparatorConfig `json:"separators"`
	Spacing    *SpacingConfig   `json:"spacing"`
}

// MarginConfig represents margin configuration
type MarginConfig struct {
	Top    int `json:"top"`
	Bottom int `json:"bottom"`
	Left   int `json:"left"`
	Right  int `json:"right"`
}

// SeparatorConfig represents separator configuration
type SeparatorConfig struct {
	Style     string `json:"style"` // line, dots, equals, custom
	Character string `json:"character"`
	Length    int    `json:"length"`
	Color     string `json:"color,omitempty"`
}

// SpacingConfig represents spacing configuration
type SpacingConfig struct {
	BetweenSections int `json:"between_sections"`
	BetweenLines    int `json:"between_lines"`
	BeforePrompt    int `json:"before_prompt"`
	AfterResponse   int `json:"after_response"`
}

// TokenDisplayConfig represents token display configuration
type TokenDisplayConfig struct {
	Enabled  bool               `json:"enabled"`
	Position string             `json:"position"` // bottom, top, inline
	Format   string             `json:"format"`   // simple, detailed, minimal
	RealTime *RealTimeConfig    `json:"real_time"`
	Cost     *CostDisplayConfig `json:"cost"`
	Warnings *WarningConfig     `json:"warnings"`
}

// RealTimeConfig represents real-time display configuration
type RealTimeConfig struct {
	Enabled          bool `json:"enabled"`
	UpdateFrequency  int  `json:"update_frequency"` // ms
	ShowEstimated    bool `json:"show_estimated"`
	ShowRunningTotal bool `json:"show_running_total"`
}

// CostDisplayConfig represents cost display configuration
type CostDisplayConfig struct {
	Enabled          bool   `json:"enabled"`
	Currency         string `json:"currency"`
	Precision        int    `json:"precision"`
	ShowRunningTotal bool   `json:"show_running_total"`
	ShowBreakdown    bool   `json:"show_breakdown"`
}

// WarningConfig represents warning display configuration
type WarningConfig struct {
	Enabled         bool    `json:"enabled"`
	TokenThreshold  int     `json:"token_threshold"`
	CostThreshold   float64 `json:"cost_threshold"`
	ShowPredictions bool    `json:"show_predictions"`
}

// PerformanceDisplayConfig represents performance display configuration
type PerformanceDisplayConfig struct {
	ShowTiming        bool                 `json:"show_timing"`
	ShowMemoryUsage   bool                 `json:"show_memory_usage"`
	ShowThroughput    bool                 `json:"show_throughput"`
	ShowCacheStatus   bool                 `json:"show_cache_status"`
	WarningThresholds *PerformanceWarnings `json:"warning_thresholds"`
}

// PerformanceWarnings represents performance warning thresholds
type PerformanceWarnings struct {
	SlowResponse  int `json:"slow_response"`  // ms
	HighMemory    int `json:"high_memory"`    // MB
	LowThroughput int `json:"low_throughput"` // tokens/sec
}

// DisplayState represents the current display state
type DisplayState struct {
	CurrentTheme   string        `json:"current_theme"`
	ScreenSize     *ScreenSize   `json:"screen_size"`
	TerminalCaps   *TerminalCaps `json:"terminal_caps"`
	ActiveStreams  []string      `json:"active_streams"`
	LastUpdate     time.Time     `json:"last_update"`
	BufferSize     int           `json:"buffer_size"`
	ScrollPosition int           `json:"scroll_position"`
}

// ScreenSize represents screen dimensions
type ScreenSize struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}

// TerminalCaps represents terminal capabilities
type TerminalCaps struct {
	Colors256       bool `json:"colors_256"`
	TrueColor       bool `json:"true_color"`
	Unicode         bool `json:"unicode"`
	CursorControl   bool `json:"cursor_control"`
	AlternateScreen bool `json:"alternate_screen"`
}
