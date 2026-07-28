//! Protocol schema compatibility tests.

use atlas_protocol::v1::{decode_envelope, Envelope, ProtocolError, SCHEMA_MAJOR};

#[test]
fn schema_major_is_version_one() {
    assert_eq!(SCHEMA_MAJOR, 1);
}

#[test]
fn envelope_encoding_is_deterministic_and_round_trips() {
    let envelope = Envelope::new("atlas.v1.ucir.Graph", b"payload".to_vec());

    let first = envelope.encode();
    let second = envelope.encode();

    assert_eq!(first, second);
    assert_eq!(decode_envelope(&first), Ok(envelope));
}

#[test]
fn decoder_rejects_unknown_major_versions() {
    let mut bytes = Envelope::new("atlas.v1.Event", Vec::new()).encode();
    bytes[7] = 2;

    assert_eq!(
        decode_envelope(&bytes),
        Err(ProtocolError::UnsupportedMajor {
            found: 2,
            supported: 1
        })
    );
}

#[test]
fn decoder_rejects_trailing_or_missing_bytes() {
    let envelope = Envelope::new("atlas.v1.Event", [1, 2, 3]);
    let mut trailing = envelope.encode();
    trailing.push(4);

    assert_eq!(
        decode_envelope(&trailing),
        Err(ProtocolError::InvalidLength)
    );
    assert_eq!(
        decode_envelope(&trailing[..5]),
        Err(ProtocolError::Truncated)
    );
}
