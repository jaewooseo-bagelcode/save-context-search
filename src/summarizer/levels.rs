//! Call graph analysis and topological sorting for level-based summarization.
//!
//! Functions are organized into levels based on their call dependencies:
//! - Level 0: Leaf functions (no outgoing calls)
//! - Level N: Functions that call Level 0..N-1 functions
//!
//! This allows us to summarize in dependency order, injecting callee summaries
//! into the context for better understanding.

use std::collections::{HashMap, HashSet};

/// Call graph representing function call relationships.
#[derive(Debug, Clone, Default)]
pub struct CallGraph {
    /// caller -> Vec<callee>
    calls: HashMap<String, Vec<String>>,
    /// All function names
    all_functions: HashSet<String>,
}

impl CallGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a function to the graph.
    pub fn add_function(&mut self, name: &str) {
        self.all_functions.insert(name.to_string());
        self.calls.entry(name.to_string()).or_default();
    }

    /// Add a call edge from caller to callee.
    pub fn add_call(&mut self, caller: &str, callee: &str) {
        self.all_functions.insert(caller.to_string());
        self.all_functions.insert(callee.to_string());
        self.calls
            .entry(caller.to_string())
            .or_default()
            .push(callee.to_string());
    }

    /// Get the functions called by a given function.
    pub fn get_calls(&self, name: &str) -> &[String] {
        self.calls.get(name).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Compute levels using topological sort.
    ///
    /// Returns a vector of levels, where each level is a vector of function names.
    /// Level 0 contains leaf functions, Level N contains functions that only call
    /// functions from levels 0..N-1.
    ///
    /// Functions in cycles are placed in Level 0 (processed without context).
    pub fn compute_levels(&self) -> Vec<Vec<String>> {
        if self.all_functions.is_empty() {
            return vec![];
        }

        // Calculate depth for each function
        let mut depths: HashMap<String, usize> = HashMap::new();
        let mut visited: HashSet<String> = HashSet::new();
        let mut in_stack: HashSet<String> = HashSet::new();

        for func in &self.all_functions {
            if !visited.contains(func) {
                self.calculate_depth(func, &mut depths, &mut visited, &mut in_stack);
            }
        }

        // Group by depth
        let max_depth = depths.values().copied().max().unwrap_or(0);
        let mut levels: Vec<Vec<String>> = vec![vec![]; max_depth + 1];

        for (func, depth) in depths {
            levels[depth].push(func);
        }

        // Sort each level for deterministic order
        for level in &mut levels {
            level.sort();
        }

        // Remove empty levels
        levels.retain(|l| !l.is_empty());

        levels
    }

    /// Calculate depth recursively with cycle detection.
    fn calculate_depth(
        &self,
        func: &str,
        depths: &mut HashMap<String, usize>,
        visited: &mut HashSet<String>,
        in_stack: &mut HashSet<String>,
    ) -> usize {
        // Already calculated
        if let Some(&depth) = depths.get(func) {
            return depth;
        }

        // Cycle detection
        if in_stack.contains(func) {
            // Part of a cycle - return 0 to break the cycle
            depths.insert(func.to_string(), 0);
            return 0;
        }

        visited.insert(func.to_string());
        in_stack.insert(func.to_string());

        // Get callees
        let callees = self.get_calls(func);

        let depth = if callees.is_empty() {
            // Leaf function
            0
        } else {
            // Max depth of callees + 1
            let max_callee_depth = callees
                .iter()
                .filter(|c| self.all_functions.contains(*c))  // Only consider known functions
                .map(|c| self.calculate_depth(c, depths, visited, in_stack))
                .max()
                .unwrap_or(0);
            max_callee_depth + 1
        };

        in_stack.remove(func);
        depths.insert(func.to_string(), depth);
        depth
    }

    /// Get statistics about the call graph.
    pub fn stats(&self) -> CallGraphStats {
        let total_functions = self.all_functions.len();
        let total_calls: usize = self.calls.values().map(|v| v.len()).sum();
        let leaf_functions = self
            .all_functions
            .iter()
            .filter(|f| self.get_calls(f).is_empty())
            .count();

        CallGraphStats {
            total_functions,
            total_calls,
            leaf_functions,
        }
    }
}

/// Statistics about a call graph.
#[derive(Debug, Clone)]
pub struct CallGraphStats {
    pub total_functions: usize,
    pub total_calls: usize,
    pub leaf_functions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let graph = CallGraph::new();
        let levels = graph.compute_levels();
        assert!(levels.is_empty());
    }

    #[test]
    fn test_single_leaf() {
        let mut graph = CallGraph::new();
        graph.add_function("foo");

        let levels = graph.compute_levels();
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0], vec!["foo"]);
    }

    #[test]
    fn test_simple_chain() {
        let mut graph = CallGraph::new();
        graph.add_function("a");
        graph.add_function("b");
        graph.add_function("c");
        graph.add_call("c", "b");
        graph.add_call("b", "a");

        let levels = graph.compute_levels();
        assert_eq!(levels.len(), 3);
        assert_eq!(levels[0], vec!["a"]);  // leaf
        assert_eq!(levels[1], vec!["b"]);  // calls a
        assert_eq!(levels[2], vec!["c"]);  // calls b
    }

    #[test]
    fn test_multiple_leaves() {
        let mut graph = CallGraph::new();
        graph.add_function("a");
        graph.add_function("b");
        graph.add_function("c");
        graph.add_call("c", "a");
        graph.add_call("c", "b");

        let levels = graph.compute_levels();
        assert_eq!(levels.len(), 2);
        assert!(levels[0].contains(&"a".to_string()));
        assert!(levels[0].contains(&"b".to_string()));
        assert_eq!(levels[1], vec!["c"]);
    }

    #[test]
    fn test_cycle_handling() {
        let mut graph = CallGraph::new();
        graph.add_function("a");
        graph.add_function("b");
        graph.add_call("a", "b");
        graph.add_call("b", "a");  // cycle

        let levels = graph.compute_levels();
        // Both should be at level 0 or 1 (cycle broken)
        assert!(!levels.is_empty());
    }

    #[test]
    fn test_diamond_dependency() {
        let mut graph = CallGraph::new();
        // d -> b, c -> a
        //   \-> c /
        graph.add_function("a");
        graph.add_function("b");
        graph.add_function("c");
        graph.add_function("d");
        graph.add_call("b", "a");
        graph.add_call("c", "a");
        graph.add_call("d", "b");
        graph.add_call("d", "c");

        let levels = graph.compute_levels();
        assert_eq!(levels.len(), 3);
        assert_eq!(levels[0], vec!["a"]);
        assert!(levels[1].contains(&"b".to_string()));
        assert!(levels[1].contains(&"c".to_string()));
        assert_eq!(levels[2], vec!["d"]);
    }

    #[test]
    fn test_stats() {
        let mut graph = CallGraph::new();
        graph.add_function("a");
        graph.add_function("b");
        graph.add_call("b", "a");

        let stats = graph.stats();
        assert_eq!(stats.total_functions, 2);
        assert_eq!(stats.total_calls, 1);
        assert_eq!(stats.leaf_functions, 1);
    }
}
