# EarthCooler CO2Coin

## What is it?

This is a special PoW based coin that can produce REAL Peptide chain sequence info while mining to capture CO2 in the air!

## Modules in project

### CO2Core

Contains block format define, Transcation vefify logic, PoW verify logic

### ProteinFold

Use Deep Learning to fold Peptide Chain which from PoW Payload to Protein 3D shape.

### MBEPredictor

Use Deep Learning to predict Molecular Binding Energy of CO2 between the Generated Protein.

### CO2Srv

The Daemon that handle clients, P2P requests, also mining to verify the block.

### CO2Client

The client that make transcations.

## Current status

Main framework logic done.

Still need bug tests.

Deep Learning part not trained yet, we still need to obtain Dataset to train and evaluate the network.