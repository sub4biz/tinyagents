//! Optional faithful PNG derivatives. Originals stay under host ownership.

/// Return a smaller losslessly recompressed PNG, or `None` when optimization
/// is unsupported, malformed, oversized, animated, or offers no saving.
///
/// This only changes IDAT compression/filtering. Pixel format, hidden RGB under
/// transparent pixels, interlacing, and every other chunk (including EXIF and
/// color metadata) remain byte-for-byte unchanged. The input is never modified.
/// Work is bounded to 32 MiB of input and 16 million pixels; hosts may enforce
/// tighter upload limits. This helper neither decodes nor changes other formats.
///
/// Requires the `png-optimize` feature. Without it this always returns `None`
/// ("no saving"), so callers keep the original bytes and any size limit they
/// apply to the original is enforced exactly as when optimization fails.
#[cfg(not(feature = "png-optimize"))]
pub fn optimize_png_lossless(_bytes: &[u8]) -> Option<Vec<u8>> {
    None
}

#[cfg(feature = "png-optimize")]
pub fn optimize_png_lossless(bytes: &[u8]) -> Option<Vec<u8>> {
    if bytes.len() > 32 * 1024 * 1024 || bytes.get(..8)? != b"\x89PNG\r\n\x1a\n" {
        return None;
    }
    if bytes.get(8..16)? != b"\0\0\0\rIHDR" {
        return None;
    }
    let width = u32::from_be_bytes(bytes.get(16..20)?.try_into().ok()?);
    let height = u32::from_be_bytes(bytes.get(20..24)?.try_into().ok()?);
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > 16_000_000 {
        return None;
    }
    let original_chunks = unchanged_chunks(bytes)?;
    let options = oxipng::Options {
        optimize_alpha: false,
        bit_depth_reduction: false,
        color_type_reduction: false,
        palette_reduction: false,
        grayscale_reduction: false,
        scale_16: false,
        interlace: None,
        strip: oxipng::StripChunks::None,
        timeout: Some(std::time::Duration::from_secs(5)),
        ..oxipng::Options::from_preset(2)
    };
    let optimized = oxipng::optimize_from_memory(bytes, &options).ok()?;
    if optimized.len() >= bytes.len() || unchanged_chunks(&optimized)? != original_chunks {
        return None;
    }
    Some(optimized)
}

#[cfg(feature = "png-optimize")]
// Even if an optimizer starts rewriting a metadata chunk in a future release,
// the derivative is admitted only when every non-IDAT chunk stays identical.
fn unchanged_chunks(bytes: &[u8]) -> Option<Vec<&[u8]>> {
    let mut position = 8usize;
    let mut chunks = Vec::new();
    while position < bytes.len() {
        let header = bytes.get(position..position.checked_add(8)?)?;
        let size = u32::from_be_bytes(header[..4].try_into().ok()?) as usize;
        let end = position.checked_add(size)?.checked_add(12)?;
        let chunk = bytes.get(position..end)?;
        let name = &header[4..8];
        if (name == b"IHDR" && position != 8) || matches!(name, b"acTL" | b"fcTL" | b"fdAT") {
            return None;
        }
        if name != b"IDAT" {
            chunks.push(chunk);
        }
        position = end;
        if name == b"IEND" {
            return (position == bytes.len()).then_some(chunks);
        }
    }
    None
}

#[cfg(test)]
#[path = "png_tests.rs"]
mod tests;
