//! `InlineVec<T, N>`: a list of at most `N` `Copy` items stored inline. The game state uses it
//! for hands, tricks and seat lists, so cloning a state (the undo snapshot on every card) is a
//! plain copy instead of a dozen heap allocations. It dereferences to a slice.

use std::fmt;
use std::ops::{Deref, DerefMut};

use serde::{Serialize, Serializer};

/// A value for the unused slots (never observable); implemented next to each item type.
pub trait Filler: Copy {
    const FILL: Self;
}

#[derive(Clone, Copy)]
pub struct InlineVec<T: Filler, const N: usize> {
    len: u8,
    items: [T; N],
}

impl<T: Filler, const N: usize> InlineVec<T, N> {
    pub const fn new() -> Self {
        assert!(N <= u8::MAX as usize, "InlineVec capacity exceeds u8");
        InlineVec {
            len: 0,
            items: [T::FILL; N],
        }
    }

    /// Appends `item`; panics when full (the capacities are fixed by the rules).
    #[inline]
    pub fn push(&mut self, item: T) {
        assert!((self.len as usize) < N, "InlineVec full ({N})");
        self.items[self.len as usize] = item;
        self.len += 1;
    }

    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn extend_from_slice(&mut self, items: &[T]) {
        for &item in items {
            self.push(item);
        }
    }

    /// Keeps the items for which `keep` is true, in order.
    pub fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        let mut n = 0;
        for i in 0..self.len as usize {
            let item = self.items[i];
            if keep(&item) {
                self.items[n] = item;
                n += 1;
            }
        }
        self.len = n as u8;
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.items[..self.len as usize]
    }
}

impl<T: Filler, const N: usize> Default for InlineVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Filler, const N: usize> Deref for InlineVec<T, N> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Filler, const N: usize> DerefMut for InlineVec<T, N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.items[..self.len as usize]
    }
}

impl<'a, T: Filler, const N: usize> IntoIterator for &'a InlineVec<T, N> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<T: Filler, const N: usize> FromIterator<T> for InlineVec<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut v = Self::new();
        for item in iter {
            v.push(item);
        }
        v
    }
}

impl<T: Filler + PartialEq, const N: usize> PartialEq for InlineVec<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Filler + Eq, const N: usize> Eq for InlineVec<T, N> {}

impl<T: Filler + fmt::Debug, const N: usize> fmt::Debug for InlineVec<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl<T: Filler + Serialize, const N: usize> Serialize for InlineVec<T, N> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.as_slice().serialize(s)
    }
}
