use super::*;

#[cfg(feature = "png-optimize")]
fn fixture() -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut encoder = ::png::Encoder::new(&mut out, 64, 64);
        encoder.set_color(::png::ColorType::Rgba);
        encoder.set_depth(::png::BitDepth::Eight);
        encoder.set_compression(::png::Compression::Fast);
        encoder
            .add_text_chunk("Author".into(), "Original metadata".into())
            .unwrap();
        let mut writer = encoder.write_header().unwrap();
        writer
            .write_chunk(
                ::png::chunk::ChunkType(*b"eXIf"),
                b"opaque orientation metadata",
            )
            .unwrap();
        let pixels = [123, 45, 67, 0].repeat(64 * 64);
        writer.write_image_data(&pixels).unwrap();
    }
    out
}

#[cfg(feature = "png-optimize")]
fn decoded(bytes: &[u8]) -> Vec<u8> {
    let mut reader = ::png::Decoder::new(std::io::Cursor::new(bytes))
        .read_info()
        .unwrap();
    let mut pixels = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut pixels).unwrap();
    pixels.truncate(info.buffer_size());
    pixels
}

#[cfg(feature = "png-optimize")]
#[test]
fn png_derivative_is_smaller_preserves_hidden_rgb_and_metadata_and_original() {
    let original = fixture();
    let snapshot = original.clone();
    let optimized = optimize_png_lossless(&original).unwrap();
    assert!(optimized.len() < original.len());
    assert_eq!(decoded(&optimized), decoded(&original));
    assert!(
        optimized
            .windows(b"Original metadata".len())
            .any(|w| w == b"Original metadata")
    );
    assert!(
        optimized
            .windows(b"opaque orientation metadata".len())
            .any(|w| w == b"opaque orientation metadata")
    );
    assert_eq!(unchanged_chunks(&optimized), unchanged_chunks(&original));
    assert_eq!(original, snapshot);
    assert!(optimize_png_lossless(&optimized).is_none());
}

#[cfg(feature = "png-optimize")]
#[test]
fn animation_malformed_non_png_and_excessive_dimensions_are_skipped() {
    assert!(optimize_png_lossless(b"jpeg").is_none());
    assert!(optimize_png_lossless(b"\x89PNG\r\n\x1a\n").is_none());
    let mut animated = fixture();
    // A structurally complete animation-control chunk need not be decoded:
    // the optional derivative must skip it before optimization.
    animated.splice(
        33..33,
        [
            0, 0, 0, 8, b'a', b'c', b'T', b'L', 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
    );
    assert!(optimize_png_lossless(&animated).is_none());
    let mut duplicate_header = fixture();
    let second_header = duplicate_header[8..33].to_vec();
    duplicate_header.splice(33..33, second_header);
    assert!(optimize_png_lossless(&duplicate_header).is_none());
    let mut giant = fixture();
    giant[16..20].copy_from_slice(&u32::MAX.to_be_bytes());
    assert!(optimize_png_lossless(&giant).is_none());
}

#[test]
fn non_png_input_is_skipped_in_every_configuration() {
    assert!(optimize_png_lossless(b"jpeg").is_none());
    assert!(optimize_png_lossless(b"\x89PNG\r\n\x1a\n").is_none());
}

// Without `png-optimize` the helper is a passthrough signal: `None` even for a
// valid, compressible PNG, so callers keep (and size-check) the original.
#[cfg(not(feature = "png-optimize"))]
#[test]
fn valid_png_is_passed_through_without_the_feature() {
    let mut out = Vec::new();
    {
        let mut encoder = ::png::Encoder::new(&mut out, 64, 64);
        encoder.set_color(::png::ColorType::Rgba);
        encoder.set_depth(::png::BitDepth::Eight);
        encoder.set_compression(::png::Compression::Fast);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&vec![0u8; 64 * 64 * 4]).unwrap();
    }
    assert!(optimize_png_lossless(&out).is_none());
}
