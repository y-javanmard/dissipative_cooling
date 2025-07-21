# dissipative\_cooling

A minimal, reproducible playground for **dissipative (algorithmic) cooling** of a quantum register using an ancilla qubit that is periodically reset.  The repository demonstrates how to extract entropy (energy) from a system of qubits by coupling them to a dissipative environment engineered from hardware–native $T_1$ relaxation.

---

## 1  Concept in a Nutshell

> *Pump heat into a bucket you can empty.*

1. **System qubits** encode the quantum state you want to cool toward its ground state.
2. An **ancilla qubit** acts as the “bucket”.  You

   * **Couple** system ↔ ancilla so that energy flows *into* the ancilla.
   * **Reset** the ancilla (force it to \\|0⟩) every cycle → the extracted energy leaves the computer.

Iterating this interaction–reset cycle drives the system toward its lowest‑energy eigenstate provided the reset rate is fast enough.

---

## 2  Minimal Model

### Hamiltonian

$$
H\_{\mathrm{tot}} \;=\; H\_{\mathrm{sys}} \;\; + \;\; H\_{\mathrm{anc}} \;\; + \;\; H\_{\mathrm{int}},
$$

* **System:** choose any problem Hamiltonian (Ising, Heisenberg, molecular, …).
* **Ancilla:** a single qubit with bare Hamiltonian $ H\_{\mathrm{anc}} = \frac{\omega\_a}{2} \, \sigma\_z^{(a)}$.
* **Interaction:** energy‑exchange (e.g. XX or flip‑flop) coupling

$$
H\_{\mathrm{int}} = g \, \bigl( \sigma\_x^{(a)} \sigma\_x^{(s)} + \sigma\_y^{(a)} \sigma\_y^{(s)} \bigr),
$$

yielding the familiar Jaynes–Cummings–like ladder that swaps excitations.

### Dissipation

A hardware reset is modelled by the Lindblad “lowering” operator applied **only** to the ancilla

$$
\mathcal L\_{\mathrm{reset}}[\rho] = \gamma\_\mathrm{r} \Bigl( \sigma^-\_{(a)} \, \rho \, \sigma^+\_{(a)} \;\; - \;\; \tfrac12 \{ \sigma^+\_{(a)}\sigma^-\_{(a)}, \rho \} \Bigr).
$$

Provided $\gamma\_\mathrm{r} \gg g$ every cycle, the ancilla re‑thermalises close to \\|0⟩, carrying entropy away.

---

## 3  Protocol

1. **Initialise** system in any state (random, thermal, excited, …); ancilla in \\|0⟩.
2. **Unitary stroke** – evolve under $H\_{\mathrm{tot}}$ for time $\tau$.
3. **Reset stroke** – apply a fast, unconditional reset to the ancilla (hardware `reset` or active feedback).
4. **Repeat** steps 2–3 until convergence of an observable (energy, magnetisation).

This is a discrete‑time version of a continuous master equation with engineered dissipation.

---

## 4  Circuit Skeleton (Qiskit‑pseudo)

```python
qc = QuantumCircuit(n_system + 1)         # last qubit is ancilla

for cycle in range(N):
    # 1) Unitary interaction
    for (i, a) in couplings:
        qc.cx(i, a)                       # example: CX–CX† generates XX
        qc.cx(a, i)
    qc.barrier()

    # 2) Ancilla reset (hardware‑native)
    qc.reset(ancilla_idx)
    qc.barrier()
```

Replace the `cx` pair with the gate sequence that realises your chosen $H\_{\mathrm{int}}$.

---

## 5  Repository Layout

```
.
├─ src/
│  ├─ dissipative_cooling.py   # high‑level API
│  ├─ circuits.py              # circuit builders
│  └─ analysis.py              # helper for energy & fidelity
├─ examples/
│  ├─ two_spin_cooling.ipynb   # introductory notebook
│  └─ transverse_field.ipynb   # larger Ising demo
├─ tests/
│  └─ test_energy_flow.py
├─ README.md
└─ requirements.txt
```

---

## 6  Running a Demo

```bash
# set up
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run the two‑spin notebook headless
jupyter nbconvert --execute examples/two_spin_cooling.ipynb
```

If you have access to real hardware (IBM Quantum, IonQ, …) set `backend_name` in `examples/config.yaml`; otherwise the notebook defaults to the local simulator with a *T₁/T₂/delay* noise model.

---

## 7  Troubleshooting & Tips

| Symptom                      | Likely cause & remedy                                |
| ---------------------------- | ---------------------------------------------------- |
| Cooling stalls above target  | Increase reset rate or coupling strength $g$.        |
| System heats up (!?)         | Check gate order; the ancilla must *absorb* energy.  |
| Hardware resets are too slow | Use active feedback: measure ancilla, conditional X. |
| Decoherence kills coherence  | Trotterise $H\_{\mathrm{sys}}$ into shorter gates.   |

---

## 8  References

* P. O. Schmidt *et al.* “**Experimental Preparation of Entangled States through Ion‑Photon‑Drop Cooling**” (2005)
* S. G. Schirmer & X. Wang, *Phys. Rev. A* **81**, 062306 (2010).
* M. Reiserer *et al.* “{“}Cooling by Measurement and Feedback”” *Nature* **508**, 237 (2014).
* A. Baio *et al.* “Algorithmic Cooling of a Superconducting Quantum Processor” (2024) – *the implementation this repo reproduces*.

---

## 9  Licence

This work is licensed under the **MIT License** – see `LICENSE` for details.
