{
  description = "Equinox";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.poetry2nix = {
    url = "github:nix-community/poetry2nix";
    inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        inherit (poetry2nix.legacyPackages.${system}) mkPoetryApplication mkPoetryEnv overrides;
        pkgs = nixpkgs.legacyPackages.${system};
      in
      rec {
        formatter = pkgs.nixpkgs-fmt;
        devShells.default = (mkPoetryEnv {
          projectDir = ./.;
          preferWheels = true;
          python = pkgs.python311;
          editablePackageSources = {
            "equinox" = ./.;
          };
        }).env.overrideAttrs (final: prev: {
          nativeBuildInputs = (prev.nativeBuildInputs or [ ]) ++ [ poetry2nix.packages.${system}.poetry ];
        });
      }
    );
}
