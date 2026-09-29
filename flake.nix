{
  description = "CAM — affect signal extraction from live camera via Metal + Vision";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/41da1e3ea8e23e094e5e3eeb1e6b830468a7399e";
    flake-parts.url = "github:hercules-ci/flake-parts";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "aarch64-darwin" ];

      perSystem =
        { pkgs, lib, ... }:
        let
          rust-bin = (pkgs.extend inputs.rust-overlay.overlays.default).rust-bin;
          rustToolchain = rust-bin.stable."1.91.0".default.override {
            extensions = [
              "rust-src"
              "clippy"
              "rustfmt"
            ];
          };
        in
        {
          devShells.default = pkgs.mkShell {
            RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
            nativeBuildInputs = [
              rustToolchain
              pkgs.pkg-config
              pkgs.libiconv
              pkgs.apple-sdk_15
            ];
          };
        };
    };
}
