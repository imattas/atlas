//! Crypto structure recognizers.

/// Crypto recognizer output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CryptoRecognition {
    /// Recognized structure name.
    pub name: String,
    /// Whether confirmation is required before attack launch.
    pub requires_confirmation: bool,
}

/// Recognizes an RSA small-private-exponent shape from public parameters.
#[must_use]
pub fn recognize_rsa_small_private_exponent(n_bits: u32, e: u64) -> Option<CryptoRecognition> {
    if n_bits >= 256 && e > 1 {
        Some(CryptoRecognition {
            name: "rsa-small-private-exponent".to_owned(),
            requires_confirmation: true,
        })
    } else {
        None
    }
}
