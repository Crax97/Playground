use std::{ops::Deref, sync::Arc};

#[derive(Clone, Debug)]
pub struct ImmutableString {
    content: ImmutableStringContent,
    id: u64,
    len: usize,
}

#[derive(Clone, Debug)]
pub enum ImmutableStringContent {
    Static(&'static str),
    Dynamic(Arc<str>),
}

impl ImmutableString {
    pub const EMPTY: ImmutableString = ImmutableString::new("");

    // Const constructor
    pub const fn new(content: &'static str) -> Self {
        let id = hash_fnv1a_64(content);
        Self {
            content: ImmutableStringContent::Static(content),
            id,
            len: content.len(),
        }
    }

    // Dynamic constructor, for strings not known at compile time
    pub fn new_dynamic(content: impl AsRef<str>) -> Self {
        let content = content.as_ref().to_string();
        let id = hash_fnv1a_64(content.as_ref());
        let len = content.len();
        Self {
            content: ImmutableStringContent::Dynamic(content.into()),
            id,
            len,
        }
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub fn append(&self, content: impl AsRef<str>) -> Self {
        let new_content = format!("{}{}", self, content.as_ref());
        Self::new_dynamic(new_content)
    }
}

pub const fn hash_fnv1a_64(content: &str) -> u64 {
    const OFFSET_BASIS: u64 = 14695981039346656037;
    const PRIME: u64 = 1099511628211;

    let mut i = 0;
    let bytes = content.as_bytes();
    let len = bytes.len();

    let mut hash = OFFSET_BASIS;
    while i < len {
        hash ^= bytes[i] as u64;
        hash = hash.overflowing_mul(PRIME).0;
        i += 1;
    }

    hash
}

impl std::hash::Hash for ImmutableString {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialEq for ImmutableString {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Ord for ImmutableString {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for ImmutableString {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.id.cmp(&other.id))
    }
}

impl<T: AsRef<str>> From<T> for ImmutableString {
    fn from(value: T) -> Self {
        Self::new_dynamic(value.as_ref())
    }
}

impl Eq for ImmutableString {}

impl std::fmt::Display for ImmutableString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self)
    }
}

impl Deref for ImmutableString {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        match &self.content {
            ImmutableStringContent::Static(s) => s,
            ImmutableStringContent::Dynamic(s) => s,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ImmutableString;

    #[test]
    pub fn equality() {
        const STR_A: ImmutableString = ImmutableString::new("a");
        const STR_B: ImmutableString = ImmutableString::new("b");

        assert_ne!(STR_A, STR_B);
        assert_eq!(STR_A, STR_A);

        assert_eq!(STR_A, "a".into());
        assert_eq!(STR_B, "b".into());
        assert_ne!(STR_A, "b".into());
    }

    #[test]
    pub fn append() {
        const HELLO: ImmutableString = ImmutableString::new("Hello,");
        let hello_world = HELLO.append(" world!");
        assert_eq!(hello_world, "Hello, world!".into());
        assert_ne!(HELLO, hello_world);
    }

    #[test]
    pub fn generic_tests() {
        const EMPTY: ImmutableString = ImmutableString::new("");
        assert!(EMPTY == ImmutableString::EMPTY);
        assert!(EMPTY.is_empty());
        assert!(ImmutableString::EMPTY.is_empty());
        assert!(EMPTY == "".into());
        assert!(ImmutableString::EMPTY == "".into());
    }
}
