#!/bin/sh
# Create (or export) the self-signed code-signing identity used for
# Parlando.app builds.
#
# Why: macOS ties granted permissions (Microphone, Accessibility) to an
# app's code identity. Ad-hoc signed bundles are identified by their content
# hash, so every rebuild is a "new app" and permissions must be granted
# again. A certificate-based signature gives the requirement
#   identifier "com.parlando.menubar" and certificate leaf = H"..."
# which stays the same across releases as long as the same certificate is
# used, on the developer's Mac and on every user's Mac. It does NOT satisfy
# Gatekeeper (that needs Developer ID + notarization); it only keeps
# permissions stable.
#
# Usage:
#   scripts/make_signing_cert.sh              # create in the login keychain if missing
#   scripts/make_signing_cert.sh --export FILE.p12   # export for CI (asks for a password)
#   KEYCHAIN=/path/ci.keychain-db KEYCHAIN_PASSWORD=... scripts/make_signing_cert.sh --import FILE.p12 P12_PASSWORD
#
# Keep the private key: if it is lost and a new certificate is made, every
# user grants the permissions once more.
set -eu

NAME=${SIGNING_CERT_NAME:-Parlando Signing}
KEYCHAIN=${KEYCHAIN:-$HOME/Library/Keychains/login.keychain-db}

say() { printf '\033[1m==> %s\033[0m\n' "$*"; }

identity_hash() {
    security find-identity -v -p codesigning "$KEYCHAIN" 2>/dev/null \
      | sed -n "s/.*[0-9]) \([0-9A-F]*\) \"$NAME\".*/\1/p" | head -1
}

case "${1:-}" in
  --export)
    OUT=${2:?usage: --export FILE.p12}
    HASH=$(identity_hash); [ -n "$HASH" ] || { echo "error: identity '$NAME' not found in $KEYCHAIN" >&2; exit 1; }
    say "Exporting '$NAME' to $OUT (you will be asked for an export password)"
    security export -k "$KEYCHAIN" -t identities -f pkcs12 -o "$OUT"
    chmod 600 "$OUT"
    say "For CI: base64 -i $OUT | pbcopy  ->  secret SIGNING_P12_BASE64, plus SIGNING_P12_PASSWORD"
    exit 0 ;;
  --import)
    P12=${2:?usage: --import FILE.p12 PASSWORD}; P12PW=${3:?password}
    if [ ! -f "$KEYCHAIN" ]; then
        say "Creating keychain $KEYCHAIN"
        security create-keychain -p "${KEYCHAIN_PASSWORD:?KEYCHAIN_PASSWORD required}" "$KEYCHAIN"
        security set-keychain-settings "$KEYCHAIN"
    fi
    security unlock-keychain -p "${KEYCHAIN_PASSWORD:-}" "$KEYCHAIN" 2>/dev/null || true
    security import "$P12" -k "$KEYCHAIN" -P "$P12PW" -T /usr/bin/codesign >/dev/null
    security set-key-partition-list -S apple-tool:,apple: -s -k "${KEYCHAIN_PASSWORD:-}" "$KEYCHAIN" >/dev/null 2>&1 || true
    ;;
  "")
    if [ -n "$(identity_hash)" ]; then
        say "Identity '$NAME' already present in $KEYCHAIN"
        exit 0
    fi
    TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
    say "Generating self-signed code-signing certificate '$NAME' (10 years)"
    cat > "$TMP/cert.cnf" <<CNF
[req]
distinguished_name = dn
x509_extensions = ext
prompt = no
[dn]
CN = $NAME
O = parlando
[ext]
basicConstraints = critical,CA:FALSE
keyUsage = critical,digitalSignature
extendedKeyUsage = critical,codeSigning
subjectKeyIdentifier = hash
CNF
    openssl req -x509 -newkey rsa:2048 -sha256 -days 3650 -nodes \
        -keyout "$TMP/key.pem" -out "$TMP/cert.pem" -config "$TMP/cert.cnf" 2>/dev/null
    # macOS `security import` needs the legacy PKCS#12 encryption.
    openssl pkcs12 -export -inkey "$TMP/key.pem" -in "$TMP/cert.pem" -out "$TMP/id.p12" \
        -passout pass:parlando -name "$NAME" \
        -keypbe PBE-SHA1-3DES -certpbe PBE-SHA1-3DES -macalg sha1 2>/dev/null
    say "Importing into $KEYCHAIN"
    security import "$TMP/id.p12" -k "$KEYCHAIN" -P parlando -T /usr/bin/codesign >/dev/null
    if [ -n "${KEYCHAIN_PASSWORD:-}" ]; then
        security set-key-partition-list -S apple-tool:,apple: -s -k "$KEYCHAIN_PASSWORD" "$KEYCHAIN" >/dev/null 2>&1 || true
    fi
    ;;
  *) echo "usage: $0 [--export FILE.p12 | --import FILE.p12 PASSWORD]" >&2; exit 2 ;;
esac

# Mark the certificate trusted for code signing (user trust domain; no
# admin rights). Without this codesign refuses the identity.
CERT=$(mktemp); trap 'rm -f "$CERT"' EXIT
security find-certificate -c "$NAME" -p "$KEYCHAIN" > "$CERT"
security add-trusted-cert -r trustRoot -p codeSign -k "$KEYCHAIN" "$CERT"
# codesign only finds identities in keychains on the search list.
if ! security list-keychains -d user | grep -q "$KEYCHAIN"; then
    security list-keychains -d user -s "$KEYCHAIN" $(security list-keychains -d user | tr -d '"')
fi
HASH=$(identity_hash)
[ -n "$HASH" ] || { echo "error: identity is not valid for code signing" >&2; exit 1; }
say "Ready: '$NAME' ($HASH)"
say "First use from a terminal may show a keychain dialog: choose 'Always Allow'."
