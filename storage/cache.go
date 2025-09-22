// storage/cache.go - Provides caching mechanisms for frequently accessed data
package storage

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/intelligence"
	"github.com/yourusername/ai-code-assistant/internal/learning"
)

// CacheEntry represents a single cache entry
type CacheEntry struct {
	Key         string        `json:"key"`
	Value       interface{}   `json:"value"`
	Size        int64         `json:"size"` // Size in bytes
	CreatedAt   time.Time     `json:"created_at"`
	LastAccess  time.Time     `json:"last_access"`
	AccessCount int64         `json:"access_count"`
	TTL         time.Duration `json:"ttl"`  // Time to live
	Tags        []string      `json:"tags"` // For cache invalidation
}

// CacheStats represents cache performance statistics
type CacheStats struct {
	HitCount      int64     `json:"hit_count"`
	MissCount     int64     `json:"miss_count"`
	HitRate       float64   `json:"hit_rate"`
	TotalSize     int64     `json:"total_size"`
	EntryCount    int64     `json:"entry_count"`
	EvictionCount int64     `json:"eviction_count"`
	AverageSize   float64   `json:"average_size"`
	OldestEntry   time.Time `json:"oldest_entry"`
	NewestEntry   time.Time `json:"newest_entry"`
}

// EvictionPolicy defines different cache eviction strategies
type EvictionPolicy int

const (
	LRU  EvictionPolicy = iota // Least Recently Used
	LFU                        // Least Frequently Used
	FIFO                       // First In, First Out
	TTL                        // Time To Live based
)

// CacheConfig defines cache configuration
type CacheConfig struct {
	MaxSize         int64          `json:"max_size"`         // Maximum cache size in bytes
	MaxEntries      int            `json:"max_entries"`      // Maximum number of entries
	DefaultTTL      time.Duration  `json:"default_ttl"`      // Default TTL for entries
	EvictionPolicy  EvictionPolicy `json:"eviction_policy"`  // Eviction strategy
	CleanupInterval time.Duration  `json:"cleanup_interval"` // How often to run cleanup
	PersistentMode  bool           `json:"persistent_mode"`  // Whether to persist to disk
	PersistentPath  string         `json:"persistent_path"`  // Path for persistent storage
}

// MultiLevelCache provides high-performance caching with multiple levels
type MultiLevelCache struct {
	config    CacheConfig
	l1Cache   map[string]*CacheEntry // In-memory cache
	l2Storage PersistentStorage      // Persistent storage interface
	stats     CacheStats
	mutex     sync.RWMutex
	stopChan  chan struct{}

	// Atomic counters for thread-safe updates
	hitCount      int64
	missCount     int64
	evictionCount int64
}

// PersistentStorage interface for L2 cache storage
type PersistentStorage interface {
	Get(ctx context.Context, key string) (*CacheEntry, error)
	Set(ctx context.Context, key string, entry *CacheEntry) error
	Delete(ctx context.Context, key string) error
	Keys(ctx context.Context, pattern string) ([]string, error)
	Clear(ctx context.Context) error
	Size(ctx context.Context) (int64, error)
}

// NewMultiLevelCache creates a new multi-level cache
func NewMultiLevelCache(config CacheConfig, l2Storage PersistentStorage) *MultiLevelCache {
	cache := &MultiLevelCache{
		config:    config,
		l1Cache:   make(map[string]*CacheEntry),
		l2Storage: l2Storage,
		stopChan:  make(chan struct{}),
	}

	// Start cleanup routine
	go cache.cleanupRoutine()

	// Load from persistent storage if enabled
	if config.PersistentMode && l2Storage != nil {
		go cache.loadFromPersistent()
	}

	return cache
}

// Get retrieves a value from the cache
func (mlc *MultiLevelCache) Get(ctx context.Context, key string) (interface{}, bool) {
	// Try L1 cache first
	if entry := mlc.getFromL1(key); entry != nil {
		atomic.AddInt64(&mlc.hitCount, 1)
		entry.LastAccess = time.Now()
		atomic.AddInt64(&entry.AccessCount, 1)
		return entry.Value, true
	}

	// Try L2 cache if enabled
	if mlc.l2Storage != nil {
		if entry, err := mlc.l2Storage.Get(ctx, key); err == nil && entry != nil {
			// Promote to L1 cache
			mlc.setInL1(key, entry)
			atomic.AddInt64(&mlc.hitCount, 1)
			entry.LastAccess = time.Now()
			atomic.AddInt64(&entry.AccessCount, 1)
			return entry.Value, true
		}
	}

	atomic.AddInt64(&mlc.missCount, 1)
	return nil, false
}

// Set stores a value in the cache
func (mlc *MultiLevelCache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration, tags ...string) error {
	if ttl == 0 {
		ttl = mlc.config.DefaultTTL
	}

	// Calculate size
	size := mlc.calculateSize(value)

	entry := &CacheEntry{
		Key:         key,
		Value:       value,
		Size:        size,
		CreatedAt:   time.Now(),
		LastAccess:  time.Now(),
		AccessCount: 0,
		TTL:         ttl,
		Tags:        tags,
	}

	// Store in L1 cache
	mlc.setInL1(key, entry)

	// Store in L2 cache if enabled
	if mlc.l2Storage != nil {
		go func() {
			mlc.l2Storage.Set(ctx, key, entry)
		}()
	}

	return nil
}

// Delete removes a value from the cache
func (mlc *MultiLevelCache) Delete(ctx context.Context, key string) error {
	mlc.mutex.Lock()
	delete(mlc.l1Cache, key)
	mlc.mutex.Unlock()

	if mlc.l2Storage != nil {
		return mlc.l2Storage.Delete(ctx, key)
	}

	return nil
}

// InvalidateByTags removes all entries with matching tags
func (mlc *MultiLevelCache) InvalidateByTags(ctx context.Context, tags ...string) error {
	mlc.mutex.Lock()
	defer mlc.mutex.Unlock()

	toDelete := make([]string, 0)

	for key, entry := range mlc.l1Cache {
		for _, tag := range tags {
			for _, entryTag := range entry.Tags {
				if tag == entryTag {
					toDelete = append(toDelete, key)
					break
				}
			}
		}
	}

	for _, key := range toDelete {
		delete(mlc.l1Cache, key)
	}

	return nil
}

// Clear removes all entries from the cache
func (mlc *MultiLevelCache) Clear(ctx context.Context) error {
	mlc.mutex.Lock()
	mlc.l1Cache = make(map[string]*CacheEntry)
	mlc.mutex.Unlock()

	if mlc.l2Storage != nil {
		return mlc.l2Storage.Clear(ctx)
	}

	return nil
}

// GetStats returns current cache statistics
func (mlc *MultiLevelCache) GetStats() CacheStats {
	mlc.mutex.RLock()
	defer mlc.mutex.RUnlock()

	hitCount := atomic.LoadInt64(&mlc.hitCount)
	missCount := atomic.LoadInt64(&mlc.missCount)
	total := hitCount + missCount

	var hitRate float64
	if total > 0 {
		hitRate = float64(hitCount) / float64(total)
	}

	var totalSize int64
	var oldestTime, newestTime time.Time
	entryCount := int64(len(mlc.l1Cache))

	for _, entry := range mlc.l1Cache {
		totalSize += entry.Size
		if oldestTime.IsZero() || entry.CreatedAt.Before(oldestTime) {
			oldestTime = entry.CreatedAt
		}
		if newestTime.IsZero() || entry.CreatedAt.After(newestTime) {
			newestTime = entry.CreatedAt
		}
	}

	var averageSize float64
	if entryCount > 0 {
		averageSize = float64(totalSize) / float64(entryCount)
	}

	return CacheStats{
		HitCount:      hitCount,
		MissCount:     missCount,
		HitRate:       hitRate,
		TotalSize:     totalSize,
		EntryCount:    entryCount,
		EvictionCount: atomic.LoadInt64(&mlc.evictionCount),
		AverageSize:   averageSize,
		OldestEntry:   oldestTime,
		NewestEntry:   newestTime,
	}
}

// getFromL1 retrieves from L1 cache
func (mlc *MultiLevelCache) getFromL1(key string) *CacheEntry {
	mlc.mutex.RLock()
	defer mlc.mutex.RUnlock()

	entry, exists := mlc.l1Cache[key]
	if !exists {
		return nil
	}

	// Check TTL
	if entry.TTL > 0 && time.Since(entry.CreatedAt) > entry.TTL {
		// Entry expired, remove it
		go func() {
			mlc.mutex.Lock()
			delete(mlc.l1Cache, key)
			mlc.mutex.Unlock()
		}()
		return nil
	}

	return entry
}

// setInL1 stores in L1 cache with eviction handling
func (mlc *MultiLevelCache) setInL1(key string, entry *CacheEntry) {
	mlc.mutex.Lock()
	defer mlc.mutex.Unlock()

	// Check if we need to evict entries
	if mlc.needsEviction(entry.Size) {
		mlc.evictEntries(entry.Size)
	}

	mlc.l1Cache[key] = entry
}

// needsEviction checks if eviction is needed
func (mlc *MultiLevelCache) needsEviction(newEntrySize int64) bool {
	if len(mlc.l1Cache) >= mlc.config.MaxEntries {
		return true
	}

	currentSize := mlc.getCurrentSize()
	return currentSize+newEntrySize > mlc.config.MaxSize
}

// getCurrentSize calculates current cache size
func (mlc *MultiLevelCache) getCurrentSize() int64 {
	var size int64
	for _, entry := range mlc.l1Cache {
		size += entry.Size
	}
	return size
}

// evictEntries removes entries based on eviction policy
func (mlc *MultiLevelCache) evictEntries(spaceNeeded int64) {
	switch mlc.config.EvictionPolicy {
	case LRU:
		mlc.evictLRU(spaceNeeded)
	case LFU:
		mlc.evictLFU(spaceNeeded)
	case FIFO:
		mlc.evictFIFO(spaceNeeded)
	case TTL:
		mlc.evictExpired()
	}
}

// evictLRU evicts least recently used entries
func (mlc *MultiLevelCache) evictLRU(spaceNeeded int64) {
	var oldestKey string
	var oldestTime time.Time
	spaceFreed := int64(0)

	for spaceFreed < spaceNeeded && len(mlc.l1Cache) > 0 {
		oldestTime = time.Now()
		oldestKey = ""

		// Find least recently used entry
		for key, entry := range mlc.l1Cache {
			if entry.LastAccess.Before(oldestTime) {
				oldestTime = entry.LastAccess
				oldestKey = key
			}
		}

		if oldestKey != "" {
			entry := mlc.l1Cache[oldestKey]
			spaceFreed += entry.Size
			delete(mlc.l1Cache, oldestKey)
			atomic.AddInt64(&mlc.evictionCount, 1)
		}
	}
}

// evictLFU evicts least frequently used entries
func (mlc *MultiLevelCache) evictLFU(spaceNeeded int64) {
	var leastUsedKey string
	var leastCount int64 = -1
	spaceFreed := int64(0)

	for spaceFreed < spaceNeeded && len(mlc.l1Cache) > 0 {
		leastCount = -1
		leastUsedKey = ""

		// Find least frequently used entry
		for key, entry := range mlc.l1Cache {
			if leastCount == -1 || entry.AccessCount < leastCount {
				leastCount = entry.AccessCount
				leastUsedKey = key
			}
		}

		if leastUsedKey != "" {
			entry := mlc.l1Cache[leastUsedKey]
			spaceFreed += entry.Size
			delete(mlc.l1Cache, leastUsedKey)
			atomic.AddInt64(&mlc.evictionCount, 1)
		}
	}
}

// evictFIFO evicts first in, first out
func (mlc *MultiLevelCache) evictFIFO(spaceNeeded int64) {
	var oldestKey string
	var oldestTime time.Time
	spaceFreed := int64(0)

	for spaceFreed < spaceNeeded && len(mlc.l1Cache) > 0 {
		oldestTime = time.Now()
		oldestKey = ""

		// Find oldest entry by creation time
		for key, entry := range mlc.l1Cache {
			if entry.CreatedAt.Before(oldestTime) {
				oldestTime = entry.CreatedAt
				oldestKey = key
			}
		}

		if oldestKey != "" {
			entry := mlc.l1Cache[oldestKey]
			spaceFreed += entry.Size
			delete(mlc.l1Cache, oldestKey)
			atomic.AddInt64(&mlc.evictionCount, 1)
		}
	}
}

// evictExpired removes all expired entries
func (mlc *MultiLevelCache) evictExpired() {
	now := time.Now()
	toDelete := make([]string, 0)

	for key, entry := range mlc.l1Cache {
		if entry.TTL > 0 && now.Sub(entry.CreatedAt) > entry.TTL {
			toDelete = append(toDelete, key)
		}
	}

	for _, key := range toDelete {
		delete(mlc.l1Cache, key)
		atomic.AddInt64(&mlc.evictionCount, 1)
	}
}

// calculateSize estimates the size of a value in bytes
func (mlc *MultiLevelCache) calculateSize(value interface{}) int64 {
	data, err := json.Marshal(value)
	if err != nil {
		// Fallback estimation
		switch v := value.(type) {
		case string:
			return int64(len(v))
		case []byte:
			return int64(len(v))
		case *intelligence.SemanticGraph:
			return int64(len(v.Nodes)*100 + len(v.Edges)*50) // Rough estimate
		case *learning.CodePattern:
			return int64(len(v.Code)*2 + len(v.Description) + 100) // Rough estimate
		default:
			return 1024 // Default 1KB
		}
	}
	return int64(len(data))
}

// cleanupRoutine runs periodic cleanup
func (mlc *MultiLevelCache) cleanupRoutine() {
	ticker := time.NewTicker(mlc.config.CleanupInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			mlc.mutex.Lock()
			mlc.evictExpired()
			mlc.mutex.Unlock()
		case <-mlc.stopChan:
			return
		}
	}
}

// loadFromPersistent loads cache from persistent storage
func (mlc *MultiLevelCache) loadFromPersistent() {
	if mlc.l2Storage == nil {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// This would depend on the specific L2 storage implementation
	// For now, just log that we're loading
	fmt.Println("Loading cache from persistent storage...")
}

// Stop stops the cache and cleanup routines
func (mlc *MultiLevelCache) Stop() {
	close(mlc.stopChan)
}

// Specialized cache implementations for specific data types

// ContextCache specialized cache for context data
type ContextCache struct {
	cache *MultiLevelCache
}

// NewContextCache creates a context-specific cache
func NewContextCache() *ContextCache {
	config := CacheConfig{
		MaxSize:         100 * 1024 * 1024, // 100MB
		MaxEntries:      1000,
		DefaultTTL:      30 * time.Minute,
		EvictionPolicy:  LRU,
		CleanupInterval: 5 * time.Minute,
	}

	return &ContextCache{
		cache: NewMultiLevelCache(config, nil),
	}
}

// GetContext retrieves cached context data
func (cc *ContextCache) GetContext(contextID string) (*intelligence.ContextBuilder, bool) {
	if value, found := cc.cache.Get(context.Background(), contextID); found {
		if ctx, ok := value.(*intelligence.ContextBuilder); ok {
			return ctx, true
		}
	}
	return nil, false
}

// SetContext caches context data
func (cc *ContextCache) SetContext(contextID string, ctx *intelligence.ContextBuilder) {
	cc.cache.Set(context.Background(), contextID, ctx, 0, "context", "intelligence")
}

// PatternCache specialized cache for pattern data
type PatternCache struct {
	cache *MultiLevelCache
}

// NewPatternCache creates a pattern-specific cache
func NewPatternCache() *PatternCache {
	config := CacheConfig{
		MaxSize:         50 * 1024 * 1024, // 50MB
		MaxEntries:      2000,
		DefaultTTL:      1 * time.Hour,
		EvictionPolicy:  LFU,
		CleanupInterval: 10 * time.Minute,
	}

	return &PatternCache{
		cache: NewMultiLevelCache(config, nil),
	}
}

// GetPattern retrieves cached pattern data
func (pc *PatternCache) GetPattern(patternID string) (*learning.CodePattern, bool) {
	if value, found := pc.cache.Get(context.Background(), patternID); found {
		if pattern, ok := value.(*learning.CodePattern); ok {
			return pattern, true
		}
	}
	return nil, false
}

// SetPattern caches pattern data
func (pc *PatternCache) SetPattern(patternID string, pattern *learning.CodePattern) {
	pc.cache.Set(context.Background(), patternID, pattern, 0, "pattern", "learning")
}

// InvalidatePatternsByLanguage removes patterns for a specific language
func (pc *PatternCache) InvalidatePatternsByLanguage(language string) {
	pc.cache.InvalidateByTags(context.Background(), "language:"+language)
}

// SemanticCache specialized cache for semantic analysis data
type SemanticCache struct {
	cache *MultiLevelCache
}

// NewSemanticCache creates a semantic analysis specific cache
func NewSemanticCache() *SemanticCache {
	config := CacheConfig{
		MaxSize:         200 * 1024 * 1024, // 200MB
		MaxEntries:      500,
		DefaultTTL:      2 * time.Hour,
		EvictionPolicy:  LRU,
		CleanupInterval: 15 * time.Minute,
	}

	return &SemanticCache{
		cache: NewMultiLevelCache(config, nil),
	}
}

// GetSemanticGraph retrieves cached semantic graph
func (sc *SemanticCache) GetSemanticGraph(graphID string) (*intelligence.SemanticGraph, bool) {
	if value, found := sc.cache.Get(context.Background(), graphID); found {
		if graph, ok := value.(*intelligence.SemanticGraph); ok {
			return graph, true
		}
	}
	return nil, false
}

// SetSemanticGraph caches semantic graph data
func (sc *SemanticCache) SetSemanticGraph(graphID string, graph *intelligence.SemanticGraph) {
	sc.cache.Set(context.Background(), graphID, graph, 0, "semantic", "graph")
}
