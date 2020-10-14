# Price Simulator: A Simulation Platform for Algorithmic Pricing

![Python application](https://github.com/matthias-hettich/price_simulator/workflows/Python%20application/badge.svg)

## Project structure

The core functionality of the price simulator is in folder `price_simulator/src/algorithm`.
The components of the simulator are programmed as separated objects.
Thus, agents, demand, environment and policies have own files and classes.
`main.py` shows an examplary entry point to start the simulation.
The environment is able to handle an arbitrary amount of different agents.
Demand, policies and agents can be changed independently from one another.
This makes a huge variety of experiments easily possible.

## Development

### First usage

To start, you'll need `docker` installed. When done, run this line in price_simulator dir:

```bash
docker build -t price-simulator:1.0 .
```

It will build docker image, used for running other commands.

### Daily development

To start development, edit files in your editor. You can execute scripts using Docker. Docker must be build to reflect changes.
```bash
docker run price-simulator:1.0
```

### Packages management

If you need an additional package in docker, add it to `requirements.in` and run script `./scripts/run-refresh-requirements`.
It will refresh `requirements.txt` file, based on content of `requirements.in`. When finished, you have to rebuild your
docker image to include new packages using `docker build -t price-simulator:1.0 .` command. You also have to do it if someone else updated
`requirements.txt` and you pulled it from remote repository.
