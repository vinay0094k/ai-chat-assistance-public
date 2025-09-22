package cache

import (
	"sync"
	"time"
)

type Cache struct {
       data map[string]CacheItem
       mu   sync.RWMutex
}

type CacheItem struct {
       Value     interface{}
       ExpiresAt time.Time
}

func New() *Cache {
       return &Cache{
               data: make(map[string]CacheItem),
       }
}

func (c *Cache) Set(key string, value interface{}, ttl time.Duration) {
       c.mu.Lock()
       defer c.mu.Unlock()

       c.data[key] = CacheItem{
               Value:     value,
               ExpiresAt: time.Now().Add(ttl),
       }
}

func (c *Cache) Get(key string) (interface{}, bool) {
       c.mu.RLock()
       defer c.mu.RUnlock()

       item, exists := c.data[key]
       if !exists || time.Now().After(item.ExpiresAt) {
               return nil, false
       }

       return item.Value, true
}

func (c *Cache) Delete(key string) {
       c.mu.Lock()
       defer c.mu.Unlock()
       delete(c.data, key)
}

func (c *Cache) Clear() {
       c.mu.Lock()
       defer c.mu.Unlock()
       c.data = make(map[string]CacheItem)
}