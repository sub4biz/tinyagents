//! Types shared by the status conversions.

/// A status has no counterpart in the target vocabulary.
///
/// Returned by the fallible (`TryFrom`) conversions, never for a status that
/// has a lossy-but-sensible mapping.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("status `{from}` has no equivalent in {target}")]
pub struct NoEquivalentStatus {
    from: &'static str,
    target: &'static str,
}

impl NoEquivalentStatus {
    pub(crate) fn new(from: &'static str, target: &'static str) -> Self {
        Self { from, target }
    }

    /// Wire label of the status that could not be converted.
    pub fn from_status(&self) -> &'static str {
        self.from
    }

    /// Name of the vocabulary that has no equivalent.
    pub fn target(&self) -> &'static str {
        self.target
    }
}
