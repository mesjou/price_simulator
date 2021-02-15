# Price Simulator: A Simulation Platform for Algorithmic Pricing

![Python application](https://github.com/matthias-hettich/price_simulator/workflows/Python%20application/badge.svg)

## Project structure

The core functionality of the price simulator is in folder `price_simulator/src/algorithm`.
The components of the simulator are programmed as separated objects.
Thus, agents, demand, environment and policies have own files and classes.
This object-oriented approach allows a free combination of different agents, demand models, or policies. 
New agents or more sophisticated demand models integrate straightforwardly.
`main.py` shows an examplary entry point to start the simulation.
The environment is able to handle an arbitrary amount of different agents.
This makes a huge variety of experiments easily possible.

## Development

### First usage

To start, you'll need `docker` installed. When done, run this line in price_simulator dir:

```bash
docker build -t price-simulator:1.0 .
```

It will build docker image, used for running other commands.
You can execute scripts using Docker. Docker must be build to reflect changes.

```bash
docker run price-simulator:1.0
```
