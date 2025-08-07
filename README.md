# Installation

## Development

I use [uv](https://docs.astral.sh/uv/).

```console
$ git clone https://github.com/st-walker/felarc
$ uv sync --group dev
$ uv run jupyter/pytest/whatever
$ # OR:
$ # OR: just jupyter with the .envrc:
$ direnv allow
$ # OR:
$ source .venv/bin/activate
```

# TODO

## Simple TBA

Source: [High Energy Beam Optics](https://bib-pubdb1.desy.de/record/352895)

  * [ ] Refactor notebook functions to package.
  * [ ] Get reasonable Twiss parameters in the TBA based on similar value from the T20 design.
  * [ ] Track using STN beam with CSR and check result.

## New Family

Source: [A New Family of Isochronous Arcs](https://accelconf.web.cern.ch/p95/ARTICLES/RPC/RPC12.PDF)

  * [ ] Implement the basic version with optionally different inner dipole strength.
  * [ ] Do the full parametristation.
  * [ ] Come up with some reasonable implementation.
  * [ ] Track with CSR.


## Simple DBA

  * [ ] Make a DBA.
  * [ ] Track and show impact of CSR
  * [ ] Make DBA

## Combined CSR Mitigation with Isochronicity

Source: [Quasi-isochronous triple-bend achromat with periodic stable optics and negligible coherent-synchrotron-radiation effects](https://journals.aps.org/prab/pdf/10.1103/PhysRevAccelBeams.24.060701)

  * [ ] Fill this out


## Miscellaneous

   * [ ] Add STN10 beam file for 100pC and 250pC with utilities for loading and artificial matching.
   * [ ] TBA plot showing the depenence of the dipole angles on the required dispersions at the inner dipole.
   * [ ] Derive the isochronicity condition.
