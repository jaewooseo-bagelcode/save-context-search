use std::collections::HashMap;

const CACHE_CAPACITY: usize = 100;

pub struct QueryCache {
    /// query → (access_counter, embedding)
    cache: HashMap<String, (u64, Vec<f32>)>,
    counter: u64,
    capacity: usize,
}

impl QueryCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            counter: 0,
            capacity: CACHE_CAPACITY,
        }
    }
    
    /// Get embedding from cache, updating access counter
    pub fn get(&mut self, query: &str) -> Option<Vec<f32>> {
        if let Some((counter, embedding)) = self.cache.get_mut(query) {
            self.counter = self.counter.wrapping_add(1);
            *counter = self.counter;
            return Some(embedding.clone());
        }
        None
    }
    
    /// Put embedding in cache, evicting LRU if at capacity
    pub fn put(&mut self, query: String, embedding: Vec<f32>) {
        // Evict LRU entry if at capacity
        if self.cache.len() >= self.capacity && !self.cache.contains_key(&query) {
            // Find entry with lowest access counter
            let lru_key = self.cache
                .iter()
                .min_by_key(|(_, (counter, _))| counter)
                .map(|(k, _)| k.clone());
            
            if let Some(key) = lru_key {
                self.cache.remove(&key);
            }
        }
        
        self.counter = self.counter.wrapping_add(1);
        self.cache.insert(query, (self.counter, embedding));
    }
    
    /// Clear all cached embeddings
    pub fn clear(&mut self) {
        self.cache.clear();
        self.counter = 0;
    }
    
    /// Get number of cached embeddings
    pub fn len(&self) -> usize {
        self.cache.len()
    }
    
    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

impl Default for QueryCache {
    fn default() -> Self {
        Self::new()
    }
}
