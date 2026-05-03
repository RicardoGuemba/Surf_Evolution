#!/usr/bin/env bash
# Atalho em português — mesma função que SurfPOC.command
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$DIR/SurfPOC.command"
