extern crate rand;

type NodeIdx = usize;
const ROOT: NodeIdx = 0;
const NO_PARENT: NodeIdx = std::usize::MAX;

type BaseIdx = usize;

const THRESHOLD: f32 = 0.95;

#[derive(Debug, Clone)]
struct Node<V: Clone + Default> {
    base: BaseIdx,
    check: NodeIdx,
    value: Option<V>, // TODO: make it more memory-efficient
}

impl<V> Node<V>
where
    V: Clone + Default,
{
    fn new() -> Node<V> {
        Node {
            base: 0,
            check: NO_PARENT,
            value: Default::default(),
        }
    }

    fn is_available(&self) -> bool {
        self.check == NO_PARENT
    }
}

fn add_offset(base: BaseIdx, c: u8) -> NodeIdx {
    base + c as usize
}

#[derive(Debug)]
pub struct Trie<V: Clone + Default> {
    nodes: Vec<Node<V>>,
    n_keys: usize,
}

type Key = Vec<u8>;

impl<V> Trie<V>
where
    V: Clone + Default,
{
    /// Creates a new static trie with the given key-value pairs.
    pub fn new(entries: &[(Key, V)]) -> Trie<V> {
        let mut trie = Trie {
            nodes: vec![Node { base: 0, check: 0, value: None }],
            n_keys: entries.len(),
        };

        // sort entries by keys
        let mut sorted_entries = entries.to_vec();
        sorted_entries.sort_by(|a, b| a.0.cmp(&b.0));

        // insert all entries in a depth-first order
        let mut stack = vec![(ROOT, sorted_entries.as_slice(), 0)];
        let mut base0 = 0;
        // insert a node which corresponds to the given range of entries
        while let Some((node, entries, depth)) = stack.pop() {
            base0 = trie.insert(node, entries, depth, base0, &mut stack);
        }

        // shrink the array
        if let Some(idx) = trie.nodes.iter().rposition(|node| !node.is_available()) {
            // shrink down to the node with the largest index
            trie.nodes.truncate(idx + 1);
        }
        trie.nodes.shrink_to_fit();

        trie
    }

    fn insert<'a>(&mut self, node: NodeIdx, entries: &'a[(Key, V)], depth: usize, base0: BaseIdx, stack: &mut Vec<(NodeIdx, &'a[(Key, V)], usize)>) -> BaseIdx {
        // set value of the node
        let first_key = &entries[0].0;
        let children;
        if first_key.len() <= depth {
            // the node has a value
            self.nodes[node].value = Some(entries[0].1.clone());
            children = &entries[1..];
        } else {
            self.nodes[node].value = None;
            children = &entries;
        };

        // segment subentries by characters
        let mut char_subentries = vec![];
        let mut last_c = 0;
        let mut start = 0;
        for (i, (key, _)) in children.iter().enumerate() {
            let c = key[depth];
            if start < i && last_c != c {
                char_subentries.push((last_c, &children[start..i]));
                start = i;
            }
            last_c = c;
        }
        if start < children.len() {
            char_subentries.push((last_c, &children[start..]));
        }

        // find available base
        let mut base = base0;
        let mut occupied_num = 0;
        loop {
            let mut available = true;
            for (i, (c, _)) in char_subentries.iter().enumerate() {
                let to = add_offset(base, *c);
                self.grow(to);
                if !self.nodes[to].is_available() {
                    if i == 0 {
                        occupied_num += 1;
                    }
                    available = false;
                    break;
                }
            }
            if available {
                break;
            }

            base += 1;
        }

        // set base
        self.nodes[node].base = base;

        // set check of children
        for (c, _) in char_subentries.iter() {
            let to = add_offset(base, *c);
            self.nodes[to].check = node;
        }

        // increase the base offset if the array around the current base is becoming full
        let mut base0 = base0;
        let occupancy = (occupied_num as f32) / ((base - base0 + 1) as f32);
        if occupancy >= THRESHOLD {
            base0 = base;
        }

        // insert recursively
        for (c, sub_entries) in char_subentries.iter().rev() {
            let to = add_offset(base, *c);
            stack.push((to, sub_entries, depth + 1));
        }

        base0
    }

    /// Returns the value associated with the key if any.
    pub fn get(&self, key: &[u8]) -> Option<V> {
        let mut node = ROOT;
        for &c in key {
            match self.child(node, c) {
                Some(to) => node = to,
                None => return None,
            }
        }
        self.nodes[node].value.clone()
    }

    /// Returns the values whose key matches a prefix of the query along with the prefix length.
    pub fn prefix_search(&self, query: &[u8]) -> Vec<(V, usize)> {
        let mut results = vec![];
        let mut node = ROOT;
        for (idx, &c) in query.iter().enumerate() {
            match self.child(node, c) {
                Some(to) => {
                    node = to;
                    match &self.nodes[node].value {
                        Some(item) => results.push((item.clone(), idx + 1)),
                        None => {}
                    }
                }
                None => return results,
            }
        }
        results
    }

    /// Returns the number of keys in the trie.
    pub fn n_keys(&self) -> usize {
        self.n_keys
    }

    fn grow(&mut self, idx: NodeIdx) {
        while self.nodes.len() <= idx {
            let new_size = self.nodes.len() * 2;
            self.nodes.resize(new_size, Node::new());
        }
    }

    fn child(&self, from: NodeIdx, c: u8) -> Option<NodeIdx> {
        let to = self.transition(from, c);
        if to < self.nodes.len() && self.nodes[to].check == from {
            Some(to)
        } else {
            None
        }
    }

    fn transition(&self, from: NodeIdx, c: u8) -> NodeIdx {
        add_offset(self.nodes[from].base, c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::{Rng, thread_rng};
    use std::collections::HashMap;

    fn make_trie() -> Trie<i32> {
        Trie::new(&vec![
            (vec![2, 3, 4], 42),
            (vec![2, 3, 5], 0),
            (vec![2, 3], 49),
            (vec![3], 8),
            (vec![3, 4, 1], 1005),
            (vec![3, 3, 5], 61),
            (vec![3, 4, 1, 2], -5),
        ])
    }

    #[test]
    fn test_get() {
        let trie = make_trie();
        assert_eq!(trie.get(&vec![]), None);
        assert_eq!(trie.get(&vec![2]), None);
        assert_eq!(trie.get(&vec![2, 3, 4, 5]), None);
        assert_eq!(trie.get(&vec![3, 3]), None);
        assert_eq!(trie.get(&vec![3, 3, 5, 6]), None);
        assert_eq!(trie.get(&vec![3, 4]), None);
        assert_eq!(trie.get(&vec![2, 3, 4]), Some(42));
        assert_eq!(trie.get(&vec![2, 3, 5]), Some(0));
        assert_eq!(trie.get(&vec![2, 3]), Some(49));
        assert_eq!(trie.get(&vec![3]), Some(8));
        assert_eq!(trie.get(&vec![3, 4, 1]), Some(1005));
        assert_eq!(trie.get(&vec![3, 3, 5]), Some(61));
        assert_eq!(trie.get(&vec![3, 4, 1, 2]), Some(-5));
    }

    #[test]
    fn test_search() {
        let trie = make_trie();
        assert_eq!(trie.prefix_search(&vec![]), vec![]);
        assert_eq!(trie.prefix_search(&vec![2]), vec![]);
        assert_eq!(
            trie.prefix_search(&vec![2, 3, 4, 5]),
            vec![(49, 2), (42, 3)]
        );
        assert_eq!(trie.prefix_search(&vec![3, 3]), vec![(8, 1)]);
        assert_eq!(trie.prefix_search(&vec![3, 3, 5, 6]), vec![(8, 1), (61, 3)]);
        assert_eq!(trie.prefix_search(&vec![3, 4]), vec![(8, 1)]);
        assert_eq!(trie.prefix_search(&vec![2, 3, 4]), vec![(49, 2), (42, 3)]);
        assert_eq!(trie.prefix_search(&vec![2, 3, 5]), vec![(49, 2), (0, 3)]);
        assert_eq!(trie.prefix_search(&vec![2, 3]), vec![(49, 2)]);
        assert_eq!(trie.prefix_search(&vec![3]), vec![(8, 1)]);
        assert_eq!(trie.prefix_search(&vec![3, 4, 1]), vec![(8, 1), (1005, 3)]);
        assert_eq!(trie.prefix_search(&vec![3, 3, 5]), vec![(8, 1), (61, 3)]);
        assert_eq!(
            trie.prefix_search(&vec![3, 4, 1, 2]),
            vec![(8, 1), (1005, 3), (-5, 4)]
        );
    }

    #[test]
    fn test_100k() {
        let mut rng = thread_rng();

        let mut entries = vec![];
        let mut key = vec![];
        let mut map = HashMap::new();
        while map.len() < 100000 {
            // grow or shrink the key randomly
            if rng.gen::<bool>() {
                key.pop();
            } else {
                key.push(rng.gen::<u8>());
            }

            // create an entry randomly
            if !map.contains_key(&key) && key.len() > 0 && rng.gen::<bool>() {
                let value = rng.gen::<i32>();
                entries.push((key.clone(), value));
                map.insert(key.clone(), value);
            }
        }

        let trie = Trie::new(&entries);

        for (k, &v) in map.iter() {
            assert_eq!(trie.get(k), Some(v));
        }
    }
}
